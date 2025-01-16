# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-head Agent and PPO Loss
=============================
This example demonstrates how to use TorchRL to create a multi-head agent with three separate distributions
(Gamma, Kumaraswamy, and Mixture) and train it using Proximal Policy Optimization (PPO) losses.

Step-by-step Explanation
------------------------

1. **Setting Composite Log-Probabilities**:
   - To use composite (=multi-head0 distributions with PPO (or any other algorithm that relies on probability distributions like SAC
     or A2C), you must call `set_composite_lp_aggregate(False).set()`. Not calling this will result in errors during
     execution of your script.
   - From torchrl and tensordict v0.9, this will be the default behavior. Not doing this will result in
     `CompositeDistribution` aggregating the log-probs, which may lead to incorrect log-probabilities.
   - Note that `set_composite_lp_aggregate(False).set()` will cause the sample log-probabilities to be named
     `<action_key>_log_prob` for any probability distribution, not just composite ones. For regular, single-head policies
     for instance, the log-probability will be named `"action_log_prob"`.
     Previously, log-prob keys defaulted to `sample_log_prob`.
2. **Action Grouping**:
   - Actions can be grouped or not; PPO doesn't require them to be grouped.
   - If actions are grouped, calling the policy will result in a `TensorDict` with fields for each agent's action and
     log-probability, e.g., `agent0`, `agent0_log_prob`, etc.

        ... [...]
        ... action: TensorDict(
        ...     fields={
        ...         agent0: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
        ...         agent0_log_prob: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
        ...         agent1: Tensor(shape=torch.Size([4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        ...         agent1_log_prob: Tensor(shape=torch.Size([4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        ...         agent2: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
        ...         agent2_log_prob: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
        ...     batch_size=torch.Size([4]),
        ...     device=None,
        ...     is_shared=False),

   - If actions are not grouped, each agent will have its own `TensorDict` with `action` and `action_log_prob` fields.

        ... [...]
        ... agent0: TensorDict(
        ...     fields={
        ...         action: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
        ...         action_log_prob: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
        ...     batch_size=torch.Size([4]),
        ...     device=None,
        ...     is_shared=False),
        ... agent1: TensorDict(
        ...     fields={
        ...         action: Tensor(shape=torch.Size([4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        ...         action_log_prob: Tensor(shape=torch.Size([4, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
        ...     batch_size=torch.Size([4]),
        ...     device=None,
        ...     is_shared=False),
        ... agent2: TensorDict(
        ...     fields={
        ...         action: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
        ...         action_log_prob: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
        ...     batch_size=torch.Size([4]),
        ...     device=None,
        ...     is_shared=False),

3. **PPO Loss Calculation**:
   - Under the hood, `ClipPPO` will clip individual weights (not the aggregate) and multiply that by the advantage.

The code below sets up a multi-head agent with three distributions and demonstrates how to train it using PPO losses.

"""

import functools

import torch
from tensordict import TensorDict
from tensordict.nn import (
    CompositeDistribution,
    InteractionType,
    ProbabilisticTensorDictModule as Prob,
    ProbabilisticTensorDictSequential as ProbSeq,
    set_composite_lp_aggregate,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
    WrapModule as Wrap,
)
from torch import distributions as d
from torchrl.objectives import ClipPPOLoss, KLPENPPOLoss, PPOLoss

set_composite_lp_aggregate(False).set()

GROUPED_ACTIONS = False

make_params = Mod(
    lambda: (
        torch.ones(4),
        torch.ones(4),
        torch.ones(4, 2),
        torch.ones(4, 2),
        torch.ones(4, 10) / 10,
        torch.zeros(4, 10),
        torch.ones(4, 10),
    ),
    in_keys=[],
    out_keys=[
        ("params", "gamma", "concentration"),
        ("params", "gamma", "rate"),
        ("params", "Kumaraswamy", "concentration0"),
        ("params", "Kumaraswamy", "concentration1"),
        ("params", "mixture", "logits"),
        ("params", "mixture", "loc"),
        ("params", "mixture", "scale"),
    ],
)


def mixture_constructor(logits, loc, scale):
    return d.MixtureSameFamily(
        d.Categorical(logits=logits), d.Normal(loc=loc, scale=scale)
    )


if GROUPED_ACTIONS:
    name_map = {
        "gamma": ("action", "agent0"),
        "Kumaraswamy": ("action", "agent1"),
        "mixture": ("action", "agent2"),
    }
else:
    name_map = {
        "gamma": ("agent0", "action"),
        "Kumaraswamy": ("agent1", "action"),
        "mixture": ("agent2", "action"),
    }

dist_constructor = functools.partial(
    CompositeDistribution,
    distribution_map={
        "gamma": d.Gamma,
        "Kumaraswamy": d.Kumaraswamy,
        "mixture": mixture_constructor,
    },
    name_map=name_map,
)


policy = ProbSeq(
    make_params,
    Prob(
        in_keys=["params"],
        out_keys=list(name_map.values()),
        distribution_class=dist_constructor,
        return_log_prob=True,
        default_interaction_type=InteractionType.RANDOM,
    ),
)

td = policy(TensorDict(batch_size=[4]))
print("Result of policy call", td)

dist = policy.get_dist(td)
log_prob = dist.log_prob(td)
print("Composite log-prob", log_prob)

# Build a dummy value operator
value_operator = Seq(
    Wrap(
        lambda td: td.set("state_value", torch.ones((*td.shape, 1))),
        out_keys=["state_value"],
    )
)

# Create fake data
data = policy(TensorDict(batch_size=[4]))
data.set(
    "next",
    TensorDict(reward=torch.randn(4, 1), done=torch.zeros(4, 1, dtype=torch.bool)),
)

# Instantiate the loss - test the 3 different PPO losses
for loss_cls in (PPOLoss, ClipPPOLoss, KLPENPPOLoss):
    # PPO sets the keys automatically by looking at the policy
    ppo = loss_cls(policy, value_operator)
    print("tensor keys", ppo.tensor_keys)

    # Get the loss values
    loss_vals = ppo(data)
    print("Loss result:", loss_cls, loss_vals)
