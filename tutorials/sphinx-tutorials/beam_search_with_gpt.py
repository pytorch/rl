"""
Beam Search with TorchRL
========================

Key learning
------------

In this tutorial, you will learn how to use TorchRL to implement beam search for efficient text generation.
You will understand how to define a policy, build an environment, and run the policy using a beam search algorithm.

Introduction
------------
Text generation is a fundamental task in natural language processing (NLP) that has numerous applications in chatbots,
language translation, and content creation. One of the challenges in text generation is efficiently exploring the vast
space of possible sequences to find the most coherent and relevant output. Beam search is a popular heuristic search
algorithm used to address this challenge by maintaining a set of candidate solutions (or "beams") at each step and
selecting the top-scoring candidates to move forward to the next step.


Introduction to Beam Search
---------------------------

Beam search is a heuristic search algorithm used in many natural language processing tasks, including machine
translation, summarization, and text generation. It works by maintaining a set of candidate solutions (or "beams") at
each step, and selecting the top-scoring candidates to move forward to the next step.

"""
import argparse

import torch
import tqdm
from tensordict import NonTensorStack, TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule as Prob,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from torch.distributions import Categorical

from torchrl._utils import _make_ordinal_device
from torchrl.data import MCTSForest

from torchrl.envs import LLMHashingEnv
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, pipeline

try:
    is_sphinx = __sphinx_build__
except NameError:
    is_sphinx = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained",
    type=bool,
    default=not is_sphinx,
    help="Set to True to load pre-trained weights, False for random weights.",
)
parser.add_argument(
    "--model",
    choices=["llama3.1", "gpt2"],
    default="gpt2",
    help="Choose the model to use: 'llama3.1' or 'gpt2'.",
)
parser.add_argument(
    "--beta", type=int, default=3, help="Set the beta parameter for the model."
)
parser.add_argument(
    "--pool", type=int, default=1000, help="Set the pool size for processing."
)
parser.add_argument(
    "--nsteps", type=int, default=10, help="Set the number of steps for the process."
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Specify the device to use (e.g., 'cpu', 'cuda').",
)
parser.add_argument(
    "--device_map",
    type=str,
    default="auto",
    help="Specify the device map for model parallelism (e.g., 'auto').",
)

args = parser.parse_args(
    [
        # When executing this in a notebook, change the parameters here, eg
        # "--device", "cuda:0"
    ]
)

################################################
# Build the model
# ---------------
# In this example, we use a pre-trained GPT-2 model as our language model.
# We define a GPTWrapper class to wrap the GPT-2 model and return the output as a TensorDict.

if args.model == "gpt2":
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    if args.pretrained:
        cfg = GPT2Config.from_pretrained("openai-community/gpt2")
    else:
        cfg = GPT2Config()
    llm = GPT2LMHeadModel(cfg).eval().requires_grad_(False)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

elif args.model == "llama3.1":
    if not args.pretrained:
        raise ValueError("llama3.1 can only be used with --pretrained=True")

    model_id = "meta-llama/Llama-3.1-8B"

    if args.device:
        args.device_map = None
    pipeline = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.device_map,
        device=args.device,
    )

    tokenizer = pipeline.tokenizer
    llm = pipeline.model.eval().requires_grad_(False)
    if args.device:
        device = _make_ordinal_device(args.device)
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif torch.mps.is_available():
        torch.mps.empty_cache()
        device = "mps:0"
    else:
        device = "cpu"

torch.set_default_device(device)

text_to_tensor = Seq(
    Mod(tokenizer, in_keys=["query"], out_keys=["out"]),
    # A renaming layer
    Mod(lambda x: x, in_keys=[("out", "input_ids")], out_keys=["observation"]),
).select_out_keys("observation")
td = TensorDict(
    query=NonTensorStack.from_list(["hello world! Give me a high five"] * 4),
    batch_size=[4],
)
print(text_to_tensor(td))


################################################
# LLM Environment with Hashing
# ----------------------------
#
# This environment represents a dataset of text sequences as a Markov Decision Process (MDP), where each observation is
# reduced to a unique integer using a hashing module.
#
# By hashing observations, we can efficiently store and retrieve them: instead of identifying nodes with their
# associated (observation, action) pair, we use a (hash, action) pair. This approach has multiple advantages:
#
# - Observations have a variable shape, making it hard to preallocate storage or store them contiguously and efficiently
#   in memory.
# - Using observations directly incurs extra memory cost as duplicated data will be stored (since successive steps in a
#   branch share several tokens). Successive nodes only differ by an extra action (i.e., an extra token), and
#   lateral (sibling) nodes differ only by their action.
#   The only information we need to store is the action associated with the node itself. To reconstruct the sequence of
#   tokens up to a certain node, we concatenate the actions along the path to that node - an operation for which
#   torchrl has all the necessary tooling.
#
# The environment has two main methods: `_reset`, which initializes the environment with a given observation, and
# `_step`, which takes an action (i.e., the next token in the sequence) and returns the updated observation.
#
# .. figure:: /_static/img/rollout-llm.png
#    :alt: Data collection loop with our LLM environment.
#

env = LLMHashingEnv(vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)

################################################
# Define the policy
# -----------------
#
# In this section, we define a "policy" that takes an observation as input and outputs an action. Note that in our
# context, the term "policy" is used to fit into control frameworks, but at its core, it's simply a language model
# with some additional pre and post-processing steps.
#
# Policy Architecture
# ~~~~~~~~~~~~~~~~~~~
#
# Our policy consists of a sequence of modules
#
# 1. Select unique observations (or nodes) in the input data.
# 2. LLMWrapper: This module wraps the GPT-2 model and provides a convenient interface for generating output.
# 3. Select last logit: This module selects the last logit from the output of the LLMWrapper.
# 4. Probabilistic sampling: This module samples from a categorical distribution to select the next token.
# 5. Reshape: This module reshapes the output to a 1D tensor.
# 6. Top-k selection: This module selects the top-k tokens with the highest probabilities.
#
# Selecting unique observations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# It might be the case that the policy receives multiple identical states in the batch.
# To make the computation more efficient, we want to select only the unique values of these observations.
# Doing this also reduces the chances that we will be generating redundant trajectories.
# Because we have hashes that uniquely define the trajectory up to a given step, it's easier to use these hash values
# to pinpoint the unique nodes rather than using the observations directly.
#
# Notice that indexing the relevant nodes is made easy thanks to tensordict's API!
#


def select_unique_obs(td):
    # Get the obs (the hash)
    hashes = td["hash"]
    hashes = hashes.squeeze()
    assert hashes.ndim == 1
    # the indices of the unique values are the unique values of the inverse indices returned from `unique`
    _, unique_hashes = torch.unique(hashes, dim=0, return_inverse=True)
    unique_hashes = unique_hashes.unique()
    return td[unique_hashes]


################################################
# LLMWrapper
# ~~~~~~~~~~
#
# The LLMWrapper module is a simple wrapper around the LLM. It takes the observation (i.e., the current text)
# as input and outputs the result of the LLM presented as a TensorDict instance.
#


class LLMWrapper(torch.nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def forward(self, x: torch.Tensor) -> TensorDict:
        result = TensorDict.from_dataclass(self.gpt(x, return_dict=True), device=device)
        return result


llm_module = Mod(LLMWrapper(llm), in_keys=["observation"], out_keys=["data"])

################################################
# Select last logits
# ~~~~~~~~~~~~~~~~~~
#
# To select the best actions, we are only going to look at the last logit of the sequence. Another option could
# be to aggregate the logits together using a :meth:`~torch.Tensor.sum` operator.

select_last = Mod(
    lambda x: x[:, -1:], in_keys=[("data", "logits")], out_keys=["logits"]
)

################################################
# Probabilistic Sampling
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The probabilistic sampling module samples from a categorical distribution to select the next token.
# We use a custom ``CategoricalWithoutReplacement`` class to ensure that the same token is not selected twice.
#
# We then use a :class:`~tensordict.nn.ProbabilisticTensorDictModule` to build the distribution from the logits
# and sample from it on-the-fly. Through the ``log_prob_key`` keyword argument, we indicate that we want to register
# the value of the log-probability in the tensordict (which we will need for the Beam search algorithm).
#


class CategoricalWithoutReplacement(Categorical):
    def sample(self, sample_shape=()) -> torch.Tensor:
        n = sample_shape.numel()
        probs = self.probs
        probs_shape = probs.shape
        if len(probs_shape) > 2:
            probs = probs.flatten(0, -2)
        samples = torch.multinomial(probs, n, replacement=False)
        return samples.view((*sample_shape, *probs_shape[:-1]))


prob_module = Prob(
    in_keys=["logits"],
    out_keys=["action"],
    default_interaction_type="random",
    distribution_class=CategoricalWithoutReplacement,
    return_log_prob=True,
    log_prob_key="logits_select",
    num_samples=args.pool,
)


################################################
# Top-k Selection
# ~~~~~~~~~~~~~~~
#
# The top-k selection module selects the top-k tokens with the highest probabilities.


def select_top_k(td: TensorDict, top_k=args.beta) -> TensorDict:
    logits = td["logits_select"]
    topk = logits.topk(top_k, dim=0)
    topk_indices = topk.indices.squeeze(-1)
    return td[topk_indices].set("topk_indices", topk_indices)


################################################
# Putting modules together using :class:`~tensordict.nn.TensorDictSequential`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Orchestrating :class:`~tensordict.nn.TensorDictModule` instances is easy, as every module receives and returns a
# TensorDict instance. Whether one, two or more (or even none!) tensors will need to be accessed, we can just
# concatenate them in a sequence and let :class:`~tensordict.nn.TensorDictSequential` do the rest.
#

policy = Seq(
    # Only get the unique obs
    select_unique_obs,
    # Call to the LLM
    llm_module,
    # Select last logit
    select_last,
    # Sample
    prob_module,
    # Reshape to -1
    lambda td: td.reshape(-1),
    # Top-k
    select_top_k,
)

################################################
# Check specs
# -----------
#
# Verify that the environment's observation and action specs match the input data.

x = tokenizer(["Check out TorchRL!"])["input_ids"]
td = TensorDict(observation=x, batch_size=[1])
td = env.reset(td)
env.check_env_specs(tensordict=td, return_contiguous=False)

################################################
# Create a forest to store the data
# ---------------------------------
#
# A :class:`~torchrl.data.MCTSForest` is a collection of trees. You can think of it as a dynamic dataset
# where we will register every new entry (node, as defined by an observation) using the previous `(observation, action)`
# pair. Later, we will be able to query the Forest for a tree given a starting node through the
# :meth:`~torchrl.data.MCTSForest.get_tree` method.
#

forest = MCTSForest(observation_keys=["hash"], action_keys=["action", "logits_select"])

################################################
# Run policy
# ----------
#
# Here comes the fun part: we will execute the policy and generate new token sequences from it.
#

with torch.no_grad():
    # Total number of candidates
    pool = args.pool
    # Number of selected beams
    beta = args.beta
    x = tokenizer(["Check out TorchRL!"])["input_ids"]
    reset_td = env.reset(
        TensorDict(observation=x, batch_size=[1]).repeat_interleave(args.beta)
    )
    tds = []
    # beam search
    td = reset_td
    reset_td = reset_td[0].clone()

    pbar = tqdm.tqdm(range(args.nsteps))
    for _ in pbar:
        td = policy(td)
        next_td = env.step(td)

        tds.append(next_td)
        next_td_filtered = next_td.exclude(
            "observation", "text", ("next", "observation"), ("next", "text")
        )
        forest.extend(next_td_filtered)
        pbar.set_description(f"Forest length: {len(forest)}")

        print("action", next_td["action"])
        td = env.step_mdp(next_td)
        print("hash", td["hash"])

    tds = TensorDict.lazy_stack(tds, -1)
    for i in range(tds.shape[0]):
        print(tds[i, -1]["next", "text"])

    tree = forest.get_tree(reset_td)
    valid_paths = list(tree.valid_paths())
    print("valid paths", valid_paths)

    for path in valid_paths:
        rollout = tree.rollout_from_path(path)
        print("Check out TorchRL!", tokenizer.decode(rollout["action"].squeeze(-1)))
        print(rollout["logits_select"].sum())

    def make_labels(local_tree, path):
        if path:
            r = tree.rollout_from_path(path)
            actions = r["action"]
            return "Check out TorchRL! " + tokenizer.decode(actions.squeeze(-1))
        return "Check out TorchRL!"

    tree.plot(make_labels=make_labels)
