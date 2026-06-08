# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.modules import CEMPlanner, ValueOperator
from torchrl.modules.planners.mppi import MPPIPlanner
from torchrl.objectives.value import TDLambdaEstimator

from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import MockBatchedUnLockedEnv


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("batch_size", [3, 5])
class TestPlanner:
    def test_CEM_model_free_env(self, device, batch_size, seed=1):
        env = MockBatchedUnLockedEnv(device=device)
        torch.manual_seed(seed)
        planner = CEMPlanner(
            env,
            planning_horizon=10,
            optim_steps=2,
            num_candidates=100,
            top_k=2,
        )
        td = env.reset(TensorDict(batch_size=batch_size).to(device))
        td_copy = td.clone()
        td = planner(td)
        assert (
            td.get("action").shape[-len(env.action_spec.shape) :]
            == env.action_spec.shape
        )
        assert env.action_spec.is_in(td.get("action"))

        for key in td.keys():
            if key != "action":
                assert torch.allclose(td[key], td_copy[key])

    def test_MPPI(self, device, batch_size, seed=1):
        torch.manual_seed(seed)
        env = MockBatchedUnLockedEnv(device=device)
        value_net = nn.LazyLinear(1, device=device)
        value_net = ValueOperator(value_net, in_keys=["observation"])
        advantage_module = TDLambdaEstimator(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
        )
        value_net(env.reset())
        planner = MPPIPlanner(
            env,
            advantage_module,
            temperature=1.0,
            planning_horizon=10,
            optim_steps=2,
            num_candidates=100,
            top_k=2,
        )
        td = env.reset(TensorDict(batch_size=batch_size).to(device))
        td_copy = td.clone()
        td = planner(td)
        assert (
            td.get("action").shape[-len(env.action_spec.shape) :]
            == env.action_spec.shape
        )
        assert env.action_spec.is_in(td.get("action"))

        for key in td.keys():
            if key != "action":
                assert torch.allclose(td[key], td_copy[key])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
