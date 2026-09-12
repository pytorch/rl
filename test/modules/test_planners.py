# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
from contextlib import contextmanager
from unittest.mock import patch

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
DIE_REWARD = 1.0
LIVE_REWARD = 5.0
JACKPOT_REWARD = 50.0
TWO_SEQ_HORIZON = 3


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


class _TwoSequenceEnv(EnvBase):
    """Reward equals the action when it is non-negative; a negative action dies.

    Dying pays ``DIE_REWARD`` and sets ``done``. After a non-breaking rollout
    auto-reset, later non-negative actions still pay their face value, so a
    die-then-jackpot sequence can beat a steady live sequence on the unmasked
    sum while losing once rewards after the first ``("next", "done")`` are
    dropped. ``terminated`` follows ``done`` only when ``terminate_on_done``.
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, _batch_locked=False, **kwargs)

    def __init__(self, device="cpu", *, terminate_on_done: bool = True):
        super().__init__(device=device)
        self.observation_spec = Composite(
            observation=Unbounded((1,), device=device),
            device=device,
        )
        self.action_spec = Unbounded((1,), device=device)
        self.reward_spec = Unbounded((1,), device=device)
        self.terminate_on_done = terminate_on_done

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
        die = action[..., :1] < 0
        reward = torch.where(
            die,
            torch.full_like(action[..., :1], DIE_REWARD),
            action[..., :1],
        )
        terminated = die if self.terminate_on_done else torch.zeros_like(die)
        return TensorDict(
            {
                "observation": tensordict.get("observation") + 1,
                "reward": reward,
                "done": die,
                "terminated": terminated,
            },
            batch_size=tensordict.batch_size,
            device=device,
        )

    def _set_seed(self, seed: int | None) -> None:
        pass


def _die_then_jackpot_sequence(device) -> torch.Tensor:
    return torch.tensor(
        [[-1.0], [JACKPOT_REWARD], [JACKPOT_REWARD]],
        device=device,
        dtype=torch.get_default_dtype(),
    )


def _steady_live_sequence(device) -> torch.Tensor:
    return torch.full(
        (TWO_SEQ_HORIZON, 1),
        LIVE_REWARD,
        device=device,
        dtype=torch.get_default_dtype(),
    )


def _rollout_action_sequence(env, actions):
    t = 0

    def policy(tensordict):
        nonlocal t
        tensordict.set(
            "action",
            actions[t].expand(*tensordict.batch_size, *actions.shape[1:]),
        )
        t += 1
        return tensordict

    td = env.reset(TensorDict(batch_size=(), device=actions.device))
    return env.rollout(
        max_steps=actions.shape[0],
        policy=policy,
        auto_reset=False,
        tensordict=td,
        break_when_any_done=False,
    )


def _sum_until_first_done(rollout):
    """Sum rewards through the first ``("next", "done")``; unmasked total."""
    reward = rollout.get(("next", "reward")).reshape(rollout.shape[-1])
    done = rollout.get(("next", "done")).reshape(rollout.shape[-1])
    unmasked = reward.sum()
    hits = done.nonzero(as_tuple=False)
    if hits.numel():
        return reward[: int(hits[0, 0]) + 1].sum(), unmasked
    return unmasked, unmasked


@contextmanager
def _fixed_planner_candidates(planner_name, *sequences):
    """Make CEM/MPPI's first (and only) sample the given action sequences."""
    stacked = torch.stack(sequences, dim=0)

    def fake_randn(*size, device=None, dtype=None, **kwargs):
        if len(size) == 1 and not isinstance(size[0], int):
            shape = tuple(size[0])
        else:
            shape = tuple(size)
        actions = stacked.to(device=device, dtype=dtype)
        if shape[-stacked.ndim :] != tuple(actions.shape):
            raise ValueError(
                f"planner randn shape {shape} does not end with {tuple(actions.shape)}"
            )
        return actions.expand(shape).clone()

    module = {
        "cem": "torchrl.modules.planners.cem",
        "mppi": "torchrl.modules.planners.mppi",
    }[planner_name]
    with patch(f"{module}.torch.randn", fake_randn):
        yield


class _SumRewardAdvantage(nn.Module):
    """Advantage equal to the trajectory return, written at every time step."""

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        reward = tensordict.get(("next", "reward"))
        ret = reward.sum(dim=-2, keepdim=True)
        tensordict.set("advantage", ret.expand_as(reward))
        return tensordict


def _make_planner(
    name,
    env,
    *,
    planning_horizon=PLANNING_HORIZON,
    num_candidates=8,
    top_k=2,
):
    if name == "cem":
        return CEMPlanner(
            env,
            planning_horizon=planning_horizon,
            optim_steps=1,
            num_candidates=num_candidates,
            top_k=top_k,
            reward_key=("next", "reward"),
        )
    return MPPIPlanner(
        env,
        _SumRewardAdvantage(),
        temperature=1.0,
        planning_horizon=planning_horizon,
        optim_steps=1,
        num_candidates=num_candidates,
        top_k=top_k,
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

    @pytest.mark.parametrize("planner_name", ["cem", "mppi"])
    @pytest.mark.parametrize("terminate_on_done", [True, False])
    def test_planner_picks_higher_masked_sequence(
        self, device, planner_name, terminate_on_done
    ):
        env = _TwoSequenceEnv(device=device, terminate_on_done=terminate_on_done)
        die_seq = _die_then_jackpot_sequence(device)
        live_seq = _steady_live_sequence(device)

        die_rollout = _rollout_action_sequence(env, die_seq)
        live_rollout = _rollout_action_sequence(env, live_seq)
        die_done = die_rollout.get(("next", "done"))
        die_terminated = die_rollout.get(("next", "terminated"))
        assert die_done[0].all()
        assert not live_rollout.get(("next", "done")).any()
        if terminate_on_done:
            torch.testing.assert_close(die_terminated, die_done)
        else:
            assert not die_terminated.any()

        die_masked, die_unmasked = _sum_until_first_done(die_rollout)
        live_masked, live_unmasked = _sum_until_first_done(live_rollout)
        torch.testing.assert_close(
            die_masked, die_masked.new_tensor(DIE_REWARD)
        )
        torch.testing.assert_close(
            die_unmasked,
            die_unmasked.new_tensor(DIE_REWARD + 2 * JACKPOT_REWARD),
        )
        torch.testing.assert_close(
            live_masked, live_masked.new_tensor(TWO_SEQ_HORIZON * LIVE_REWARD)
        )
        torch.testing.assert_close(live_unmasked, live_masked)
        assert live_masked > die_masked
        assert die_unmasked > live_unmasked

        planner = _make_planner(
            planner_name,
            env,
            planning_horizon=TWO_SEQ_HORIZON,
            num_candidates=4,
            top_k=2,
        )
        td = env.reset(TensorDict(batch_size=(), device=device))
        with _fixed_planner_candidates(
            planner_name, die_seq, live_seq, die_seq, live_seq
        ):
            out = planner(td)
        torch.testing.assert_close(out.get("action"), live_seq[0])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
