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

import torchrl.data
import tqdm
from tensordict import NonTensorStack, TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule as Prob,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from tensordict.tensorclass import NonTensorData
from torch.distributions import Categorical

from torchrl._utils import _make_ordinal_device
from torchrl.data import MCTSForest, SipHash
from torchrl.envs import EnvBase
from torchrl.envs.common import _StepMDP
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, pipeline
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

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
# Build the hash module
# ---------------------
#
# We are going to build a hash module to mark each step in the dataset. In theory, observations could be used directly
# but the shape of each observation in the rollout will differ because the number of tokens is different at each step
# of the trajectory.
#
# Using a hashing module, we can reduce every observation to an integer. Although we cannot recover the prompt directly
# from the hash, we can easily recover this by concatenating the previous actions with the initial prompt.
#
#
# .. figure:: /_static/img/rollout-llm.png
#    :alt: Data collection loop with our LLM environment.
#
siphash = SipHash()


################################################
# Build the environment
# ---------------------
#
# We define an environment that simulates the text generation process.
# The environment has two main methods: _reset, which initializes the environment with a given observation, and
# _step, which takes an action (i.e., the next token to generate) and returns the next observation and reward.


class LLMEnv(EnvBase):
    def __init__(self):
        super().__init__()
        self._batch_locked = False
        _StepMDP(self)

    def _reset(self, tensordict):
        out = tensordict.copy()
        obs = out["observation"]
        if obs.ndim > 1:
            text = tokenizer.batch_decode(obs)
            text = NonTensorStack.from_list(text)
        else:
            text = tokenizer.decode(obs)
            text = NonTensorData(text)
        out["text"] = text

        if obs.ndim > 1:
            out["hash"] = siphash(out["observation"]).unsqueeze(-1)
        else:
            out["hash"] = siphash(out["observation"].unsqueeze(0)).transpose(0, -1)

        if not self.full_done_spec.is_empty():
            out.update(self.full_done_spec.zero(tensordict.shape))
        else:
            out.set("done", torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool))
            out.set(
                "terminated", torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool)
            )
        return out

    def _step(self, tensordict):
        action = tensordict.get("action")
        obs = torch.cat([tensordict.get("observation"), action], -1)

        catval = torch.cat([tensordict.get("hash"), action], -1)
        if obs.ndim > 1:
            new_hash = siphash(catval).unsqueeze(-1)
        else:
            new_hash = siphash(catval.unsqueeze(0)).transpose(0, -1)

        if obs.ndim > 1:
            text = tokenizer.batch_decode(obs)
            text = NonTensorStack.from_list(text)
        else:
            text = tokenizer.decode(obs)
            text = NonTensorData(text)
        return TensorDict(
            observation=obs,
            hash=new_hash,
            text=text,
            reward=torch.zeros((*tensordict.batch_size, 1)),
            done=torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool),
            terminated=torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool),
            batch_size=tensordict.batch_size,
        )

    def _set_seed(self, *args):
        pass


env = LLMEnv()

################################################
# Define specs
# ------------
#

policy = policy.select_out_keys("action")

x = tokenizer(["Check out TorchRL!"])["input_ids"]
td = TensorDict(observation=x, batch_size=[1]).repeat_interleave(args.beta)
td = env.reset(td)
print("data after reset", td)
print("action", policy(td))
# We must indicate what the observations are
env.auto_specs_(policy, tensordict=td, observation_key=["observation", "text", "hash"])
print(env.specs)
# Reset out keys - we want them all
policy.reset_out_keys()
policy = policy.select_out_keys("action", "logits_select")

td = TensorDict(observation=x, batch_size=[1]).repeat_interleave(args.beta, dim=0)
td = env.reset(td)
env.action_spec = torchrl.data.Categorical(n=tokenizer.vocab_size, shape=(1,))
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
