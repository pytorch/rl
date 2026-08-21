# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmarks for the DreamerV3 RSSM."""
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.modules.models.model_based_v3 import (
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)

# The DMC Walker training shape of the DreamerV3 example.
_BATCH_SIZE = 16
_TIME_STEPS = 64
_NUM_CATEGORICALS = 32
_NUM_CLASSES = 4
_BELIEF_DIM = 512
_HIDDEN_DIM = 64
_EMBEDDING_DIM = 64
_ACTION_DIM = 6


def _make_rollout(device: torch.device) -> RSSMRolloutV3:
    torch.manual_seed(0)
    prior = RSSMPriorV3(
        action_shape=torch.Size([_ACTION_DIM]),
        hidden_dim=_HIDDEN_DIM,
        rnn_hidden_dim=_BELIEF_DIM,
        num_categoricals=_NUM_CATEGORICALS,
        num_classes=_NUM_CLASSES,
        action_dim=_ACTION_DIM,
        recurrent_model="block_gru",
        num_blocks=8,
        num_layers=1,
        prior_num_layers=2,
    )
    posterior = RSSMPosteriorV3(
        hidden_dim=_HIDDEN_DIM,
        num_categoricals=_NUM_CATEGORICALS,
        num_classes=_NUM_CLASSES,
        rnn_hidden_dim=_BELIEF_DIM,
        obs_embed_dim=_EMBEDDING_DIM,
        use_rms_norm=True,
    )
    return RSSMRolloutV3(
        TensorDictModule(
            prior,
            in_keys=["state", "belief", "action"],
            out_keys=[
                ("next", "prior_logits"),
                ("next", "state"),
                ("next", "belief"),
            ],
        ),
        TensorDictModule(
            posterior,
            in_keys=[("next", "belief"), ("next", "encoded_latents")],
            out_keys=[("next", "posterior_logits"), ("next", "state")],
        ),
    ).to(device)


def _make_tensordict(device: torch.device) -> TensorDict:
    state_dim = _NUM_CATEGORICALS * _NUM_CLASSES
    return TensorDict(
        {
            "state": torch.zeros(_BATCH_SIZE, _TIME_STEPS, state_dim, device=device),
            "belief": torch.zeros(_BATCH_SIZE, _TIME_STEPS, _BELIEF_DIM, device=device),
            "action": torch.randn(_BATCH_SIZE, _TIME_STEPS, _ACTION_DIM, device=device),
            "is_init": torch.zeros(
                _BATCH_SIZE, _TIME_STEPS, 1, dtype=torch.bool, device=device
            ),
            "next": {
                "encoded_latents": torch.randn(
                    _BATCH_SIZE, _TIME_STEPS, _EMBEDDING_DIM, device=device
                )
            },
        },
        [_BATCH_SIZE, _TIME_STEPS],
    )


def _call(rollout: RSSMRolloutV3, tensordict: TensorDict, device: torch.device) -> None:
    out = rollout(tensordict)
    out.get(("next", "posterior_logits")).square().mean().backward()
    rollout.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()


# The compiled scopes need minutes to compile, over the 240 s suite timeout.
def test_dreamer_v3_rssm_rollout(benchmark) -> None:
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    rollout = _make_rollout(device)
    tensordict = _make_tensordict(device)

    _call(rollout, tensordict, device)

    benchmark.extra_info.update(
        {
            "batch_size": _BATCH_SIZE,
            "sequence_length": _TIME_STEPS,
            "belief_dim": _BELIEF_DIM,
            "transitions_per_call": _BATCH_SIZE * _TIME_STEPS,
            "device": str(device),
        }
    )
    benchmark(_call, rollout, tensordict, device)


if __name__ == "__main__":
    _, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
