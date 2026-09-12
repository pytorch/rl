# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.modules import CEMPlanner, ValueOperator
from torchrl.modules.planners.common import _mask_post_done_reward
from torchrl.modules.planners.mppi import MPPIPlanner
from torchrl.objectives.value import TDLambdaEstimator

from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import MockBatchedUnLockedEnv

FIRST_REWARD = 1.25
LATER_REWARD = 50.0
PLANNING_HORIZON = 4


class _DoneAfterOneStepEnv(EnvBase):
    """Batch-unlocked env that is done after one step.

    The first transition yields ``FIRST_REWARD``. After that, a non-breaking
    rollout auto-resets and later transitions yield ``LATER_REWARD``.
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, _batch_locked=False, **kwargs)

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.observation_spec = Composite(
            observation=Unbounded((1,), device=device),
            device=device,
        )
        self.action_spec = Unbounded((1,), device=device)
        self.reward_spec = Unbounded((1,), device=device)
        self._started = False

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is None:
            batch_size = self.batch_size
            device = self.device
        else:
            batch_size = tensordict.batch_size
            device = tensordict.device if tensordict.device is not None else self.device
        if tensordict is None or "_reset" not in tensordict.keys():
            self._started = False
        done = torch.zeros(*batch_size, 1, dtype=torch.bool, device=device)
        return TensorDict(
            {
                "observation": torch.zeros(
                    *batch_size, 1, device=device, dtype=torch.get_default_dtype()
                ),
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=batch_size,
            device=device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        batch_size = tensordict.batch_size
        device = tensordict.device if tensordict.device is not None else self.device
        reward_value = LATER_REWARD if self._started else FIRST_REWARD
        self._started = True
        done = torch.ones(*batch_size, 1, dtype=torch.bool, device=device)
        return TensorDict(
            {
                "observation": tensordict.get("observation") + 1,
                "reward": torch.full(
                    (*batch_size, 1),
                    reward_value,
                    device=device,
                    dtype=torch.get_default_dtype(),
                ),
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=batch_size,
            device=device,
        )

    def _set_seed(self, seed: int | None) -> None:
        pass


class _SurviveIfPositiveEnv(EnvBase):
    """Reward is 1 each step. The episode ends when ``action[..., :1] < 0``."""

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, _batch_locked=False, **kwargs)

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.observation_spec = Composite(
            observation=Unbounded((1,), device=device),
            device=device,
        )
        self.action_spec = Unbounded((1,), device=device)
        self.reward_spec = Unbounded((1,), device=device)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is None:
            batch_size = self.batch_size
            device = self.device
        else:
            batch_size = tensordict.batch_size
            device = tensordict.device if tensordict.device is not None else self.device
        done = torch.zeros(*batch_size, 1, dtype=torch.bool, device=device)
        return TensorDict(
            {
                "observation": torch.zeros(
                    *batch_size, 1, device=device, dtype=torch.get_default_dtype()
                ),
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=batch_size,
            device=device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get("action")
        device = action.device
        done = action[..., :1] < 0
        return TensorDict(
            {
                "observation": tensordict.get("observation") + 1,
                "reward": torch.ones(
                    *tensordict.batch_size,
                    1,
                    device=device,
                    dtype=torch.get_default_dtype(),
                ),
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=tensordict.batch_size,
            device=device,
        )

    def _set_seed(self, seed: int | None) -> None:
        pass


class _SumRewardAdvantage(nn.Module):
    """Advantage equal to the trajectory return, written at every time step."""

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        reward = tensordict.get(("next", "reward"))
        ret = reward.sum(dim=-2, keepdim=True)
        tensordict.set("advantage", ret.expand_as(reward))
        return tensordict


def _make_planner(name, env, *, planning_horizon=PLANNING_HORIZON):
    if name == "cem":
        return CEMPlanner(
            env,
            planning_horizon=planning_horizon,
            optim_steps=1,
            num_candidates=8,
            top_k=2,
            reward_key=("next", "reward"),
        )
    return MPPIPlanner(
        env,
        _SumRewardAdvantage(),
        temperature=1.0,
        planning_horizon=planning_horizon,
        optim_steps=1,
        num_candidates=8,
        top_k=2,
        reward_key=("next", "reward"),
    )


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

    @pytest.mark.parametrize("planner_name", ["cem", "mppi"])
    def test_planner_ignores_post_done_rewards(self, device, batch_size, planner_name):
        env = _DoneAfterOneStepEnv(device=device)
        planner = _make_planner(planner_name, env)
        td = env.reset(TensorDict(batch_size=batch_size).to(device))
        num_candidates = 8
        expanded = (
            td.unsqueeze(-1).expand(*td.batch_size, num_candidates).to_tensordict()
        )

        def policy(tensordict):
            tensordict.set(
                "action",
                torch.zeros(
                    *tensordict.batch_size,
                    *env.action_spec.shape,
                    device=tensordict.device,
                    dtype=env.action_spec.dtype,
                ),
            )
            return tensordict

        rollout = env.rollout(
            max_steps=PLANNING_HORIZON,
            policy=policy,
            auto_reset=False,
            tensordict=expanded.clone(),
            break_when_any_done=False,
        )
        raw = rollout.get(("next", "reward"))
        assert raw.shape[-2] == PLANNING_HORIZON
        torch.testing.assert_close(
            raw[..., 0, :],
            torch.full_like(raw[..., 0, :], FIRST_REWARD),
        )
        torch.testing.assert_close(
            raw[..., 1:, :],
            torch.full_like(raw[..., 1:, :], LATER_REWARD),
        )

        masked = _mask_post_done_reward(
            rollout, reward_key=("next", "reward"), time_dim=-2
        )
        expected = torch.zeros_like(raw)
        expected[..., 0, :] = FIRST_REWARD
        torch.testing.assert_close(masked, expected)
        torch.testing.assert_close(
            masked.sum(dim=-2),
            torch.full_like(masked.sum(dim=-2), FIRST_REWARD),
        )

        td = env.reset(TensorDict(batch_size=batch_size).to(device))
        out = planner(td.clone())
        assert env.action_spec.is_in(out.get("action"))


@pytest.mark.parametrize("device", get_default_devices())
class TestPlannerDoneMask:
    def test_mask_post_done_reward_values(self, device):
        reward = torch.tensor(
            [[[1.0], [2.0], [3.0]]],
            device=device,
            dtype=torch.get_default_dtype(),
        )
        done = torch.tensor([[[True], [False], [True]]], device=device)
        td = TensorDict(
            {"next": {"reward": reward, "done": done}},
            batch_size=[1, 3],
            device=device,
        )
        masked = _mask_post_done_reward(td, reward_key=("next", "reward"))
        expected = torch.tensor(
            [[[1.0], [0.0], [0.0]]],
            device=device,
            dtype=torch.get_default_dtype(),
        )
        torch.testing.assert_close(masked, expected)

        done_later = torch.tensor([[[False], [True], [False]]], device=device)
        td_later = TensorDict(
            {"next": {"reward": reward, "done": done_later}},
            batch_size=[1, 3],
            device=device,
        )
        masked_later = _mask_post_done_reward(
            td_later, reward_key=("next", "reward")
        )
        expected_later = torch.tensor(
            [[[1.0], [2.0], [0.0]]],
            device=device,
            dtype=torch.get_default_dtype(),
        )
        torch.testing.assert_close(masked_later, expected_later)

    def test_mask_post_done_reward_nested_keys(self, device):
        reward = torch.zeros(2, 3, 1, device=device, dtype=torch.get_default_dtype())
        reward[:, 0] = 1.0
        reward[:, 1] = 2.0
        reward[:, 2] = 3.0
        done = torch.zeros(2, 3, 1, dtype=torch.bool, device=device)
        done[:, 0] = True
        td = TensorDict(
            {"next": {"agents": {"reward": reward, "done": done}}},
            batch_size=[2, 3],
            device=device,
        )
        masked = _mask_post_done_reward(
            td,
            reward_key=("next", "agents", "reward"),
            done_key=("next", "agents", "done"),
        )
        expected = torch.zeros_like(reward)
        expected[:, 0] = 1.0
        torch.testing.assert_close(masked, expected)

    @pytest.mark.parametrize("planner_name", ["cem", "mppi"])
    def test_planner_prefers_surviving_actions(self, device, planner_name, seed=0):
        torch.manual_seed(seed)
        env = _SurviveIfPositiveEnv(device=device)
        if planner_name == "cem":
            planner = CEMPlanner(
                env,
                planning_horizon=4,
                optim_steps=3,
                num_candidates=64,
                top_k=8,
                reward_key=("next", "reward"),
            )
        else:
            planner = MPPIPlanner(
                env,
                _SumRewardAdvantage(),
                temperature=1.0,
                planning_horizon=4,
                optim_steps=3,
                num_candidates=64,
                top_k=8,
                reward_key=("next", "reward"),
            )
        td = env.reset(TensorDict(batch_size=()).to(device))
        out = planner(td)
        # Surviving (non-negative) first actions score 4; dying at step 0 scores 1.
        assert out.get("action").item() > 0


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
