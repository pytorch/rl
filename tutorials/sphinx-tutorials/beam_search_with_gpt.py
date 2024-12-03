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
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, pipeline
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from torchrl.envs import LLMHashingEnv

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["llama3.1", "gpt2"], default="gpt2")
parser.add_argument("--beta", type=int, default=3)
parser.add_argument("--pool", type=int, default=1000)
parser.add_argument("--nsteps", type=int, default=10)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--device_map", type=str, default="auto")

args = parser.parse_args()

################################################
# Build the model
# ---------------
# In this example, we use a pre-trained GPT-2 model as our language model.
# We define a GPTWrapper class to wrap the GPT-2 model and return the output as a TensorDict.

if args.model == "gpt2":
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    cfg = GPT2Config.from_pretrained("openai-community/gpt2")
    llm = GPT2LMHeadModel(cfg).eval().requires_grad_(False)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

elif args.model == "llama3.1":
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
# Define the policy
# -----------------
# We define a policy that takes the observation as input and outputs an action (i.e., the next token to generate).
#
# Our policy takes the observation (i.e., the current text) as input and outputs an action (i.e., the next token to
# generate). The policy consists of a sequence of modules: first, we use the GPTWrapper to get the output from the
# GPT-2 model, and then we select the top-scoring token using a categorical distribution.


class LLMWrapper(torch.nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def forward(self, x) -> CausalLMOutputWithCrossAttentions:
        result = TensorDict.from_dataclass(self.gpt(x, return_dict=True), device=device)
        return result


class CategoricalWithoutReplacement(Categorical):
    def sample(self, sample_shape=()):
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


def select_unique_obs(td):
    # Get the obs (the hash)
    hashes = td["hash"]
    hashes = hashes.squeeze()
    assert hashes.ndim == 1
    _, unique_hashes = torch.unique(hashes, dim=0, return_inverse=True)
    unique_hashes = unique_hashes.unique()
    return td[unique_hashes]


def select_top_k(td, top_k=args.beta):
    logits = td["logits_select"]
    topk = logits.topk(top_k, dim=0)
    topk_indices = topk.indices.squeeze(-1)
    return td[topk_indices].set("topk_indices", topk_indices)


policy = Seq(
    # Only get the unique obs
    select_unique_obs,
    # Call to the LLM
    Mod(LLMWrapper(llm), in_keys=["observation"], out_keys=["data"]),
    # Select last logit
    Mod(lambda x: x[:, -1:], in_keys=[("data", "logits")], out_keys=["logits"]),
    # Sample
    prob_module,
    # Reshape to -1
    lambda td: td.reshape(-1),
    # Top-k
    select_top_k,
)

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
# The environment has two main methods: `_reset`, which initializes the environment with a given observation, and `_step`, which takes an action (i.e., the next token in the sequence) and returns the updated observation.
#
# .. figure:: /_static/img/rollout-llm.png
#    :alt: Data collection loop with our LLM environment.
#

env = LLMHashingEnv(vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)

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

forest = MCTSForest(observation_keys=["hash"], action_keys=["action", "logits_select"])

################################################
# Run policy
# ----------
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
