# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-head agent and PPO loss
=============================

This example demonstrates how to use TorchRL to create a multi-head agent with three separate distributions
(Gamma, Kumaraswamy, and Mixture) and train it using Proximal Policy Optimization (PPO) losses.

The code first defines a module `make_params` that extracts the parameters of the distributions from an input tensordict.
It then creates a `dist_constructor` function that takes these parameters as input and outputs a CompositeDistribution
object containing the three distributions.

The policy is defined as a ProbabilisticTensorDictSequential module that reads an observation, casts it to parameters,
creates a distribution from these parameters, and samples from the distribution to output multiple actions.

The example tests the policy with fake data across three different PPO losses: PPOLoss, ClipPPOLoss, and KLPENPPOLoss.

Note that the `log_prob` method of the CompositeDistribution object can return either an aggregated tensor or a
fine-grained tensordict with individual log-probabilities, depending on the value of the `aggregate_probabilities`
argument. The PPO loss modules are designed to handle both cases, and will default to `aggregate_probabilities=False`
if not specified.

In particular, if `aggregate_probabilities=False` and `include_sum=True`, the summed log-probs will also be included in
the output tensordict. However, since we have access to the individual log-probs, this feature is not typically used.

"""

import functools

import torch
from tensordict import TensorDict
from tensordict.nn import (
    CompositeDistribution,
    InteractionType,
    ProbabilisticTensorDictModule as Prob,
    ProbabilisticTensorDictSequential as ProbSeq,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
    WrapModule as Wrap,
)
from torch import distributions as d
from torchrl.objectives import ClipPPOLoss, KLPENPPOLoss, PPOLoss

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


# =============================================================================
# Example 0: aggregate_probabilities=None (default) ===========================

dist_constructor = functools.partial(
    CompositeDistribution,
    distribution_map={
        "gamma": d.Gamma,
        "Kumaraswamy": d.Kumaraswamy,
        "mixture": mixture_constructor,
    },
    name_map={
        "gamma": ("agent0", "action"),
        "Kumaraswamy": ("agent1", "action"),
        "mixture": ("agent2", "action"),
    },
    aggregate_probabilities=None,
)


policy = ProbSeq(
    make_params,
    Prob(
        in_keys=["params"],
        out_keys=[("agent0", "action"), ("agent1", "action"), ("agent2", "action")],
        distribution_class=dist_constructor,
        return_log_prob=True,
        default_interaction_type=InteractionType.RANDOM,
    ),
)

td = policy(TensorDict(batch_size=[4]))
print("0. result of policy call", td)

dist = policy.get_dist(td)
log_prob = dist.log_prob(
    td, aggregate_probabilities=False, inplace=False, include_sum=False
)
print("0. non-aggregated log-prob")

# We can also get the log-prob from the policy directly
log_prob = policy.log_prob(
    td, aggregate_probabilities=False, inplace=False, include_sum=False
)
print("0. non-aggregated log-prob (from policy)")

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

# Instantiate the loss
for loss_cls in (PPOLoss, ClipPPOLoss, KLPENPPOLoss):
    ppo = loss_cls(policy, value_operator)

    # Keys are not the default ones - there is more than one action
    ppo.set_keys(
        action=[("agent0", "action"), ("agent1", "action"), ("agent2", "action")],
        sample_log_prob=[
            ("agent0", "action_log_prob"),
            ("agent1", "action_log_prob"),
            ("agent2", "action_log_prob"),
        ],
    )

    # Get the loss values
    loss_vals = ppo(data)
    print("0. ", loss_cls, loss_vals)


# ===================================================================
# Example 1: aggregate_probabilities=True ===========================

dist_constructor.keywords["aggregate_probabilities"] = True

td = policy(TensorDict(batch_size=[4]))
print("1. result of policy call", td)

# Instantiate the loss
for loss_cls in (PPOLoss, ClipPPOLoss, KLPENPPOLoss):
    ppo = loss_cls(policy, value_operator)

    # Keys are not the default ones - there is more than one action. No need to indicate the sample-log-prob key, since
    # there is only one.
    ppo.set_keys(
        action=[("agent0", "action"), ("agent1", "action"), ("agent2", "action")]
    )

    # Get the loss values
    loss_vals = ppo(data)
    print("1. ", loss_cls, loss_vals)


# ===================================================================
# Example 2: aggregate_probabilities=False ===========================

dist_constructor.keywords["aggregate_probabilities"] = False

td = policy(TensorDict(batch_size=[4]))
print("2. result of policy call", td)

# Instantiate the loss
for loss_cls in (PPOLoss, ClipPPOLoss, KLPENPPOLoss):
    ppo = loss_cls(policy, value_operator)

    # Keys are not the default ones - there is more than one action
    ppo.set_keys(
        action=[("agent0", "action"), ("agent1", "action"), ("agent2", "action")],
        sample_log_prob=[
            ("agent0", "action_log_prob"),
            ("agent1", "action_log_prob"),
            ("agent2", "action_log_prob"),
        ],
    )

    # Get the loss values
    loss_vals = ppo(data)
    print("2. ", loss_cls, loss_vals)
