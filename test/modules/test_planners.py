# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import Literal
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase, ExplorationType, set_exploration_type
from torchrl.modules import CEMPlanner, TdMpc2Planner, ValueOperator
from torchrl.modules.planners.mppi import MPPIPlanner
from torchrl.objectives.value import TDLambdaEstimator

from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import MockBatchedUnLockedEnv

DIE_REWARD = 1.0
LIVE_REWARD = 5.0
JACKPOT_REWARD = 50.0
TWO_SEQ_HORIZON = 3


class _TwoSequenceEnv(EnvBase):
    """Reward equals the action when non-negative; a negative action dies.

    Dying pays ``DIE_REWARD``. ``terminated`` follows ``done`` only when
    ``terminate_on_done``. ``done_layout`` is ``"root"``, ``"nested"``
    (flags only under ``agent``), ``"both"``, or ``"nested_parent"``
    (the ``"both"`` layout nested under ``team``). With ``"both"`` or
    ``"nested_parent"``, the parent ``done`` stays false so a finished
    nested agent does not end the env.
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, _batch_locked=False, **kwargs)

    def __init__(
        self,
        device="cpu",
        *,
        terminate_on_done: bool = True,
        done_layout: Literal["root", "nested", "both", "nested_parent"] = "root",
    ):
        super().__init__(device=device)
        self.observation_spec = Composite(
            observation=Unbounded((1,), device=device),
            device=device,
        )
        self.action_spec = Unbounded((1,), device=device)
        self.reward_spec = Unbounded((1,), device=device)
        self.terminate_on_done = terminate_on_done
        self.done_layout = done_layout
        if done_layout != "root":
            flag = Categorical(2, dtype=torch.bool, shape=(1,), device=device)
            agent = Composite(
                {"done": flag.clone(), "terminated": flag.clone()},
                shape=(),
                device=device,
            )
            if done_layout == "nested_parent":
                spec = {
                    "team": Composite(
                        {
                            "done": flag.clone(),
                            "terminated": flag.clone(),
                            "agent": agent,
                        },
                        shape=(),
                        device=device,
                    )
                }
            else:
                spec = {"agent": agent}
                if done_layout == "both":
                    spec["done"] = flag.clone()
                    spec["terminated"] = flag.clone()
            self.done_spec = Composite(spec, shape=(), device=device)

    def _done_payload(self, done, terminated):
        if self.done_layout == "root":
            return {"done": done, "terminated": terminated}
        child = {"done": done, "terminated": terminated}
        if self.done_layout == "nested":
            return {"agent": child}
        false = torch.zeros_like(done)
        parent = {"done": false, "terminated": false, "agent": child}
        if self.done_layout == "nested_parent":
            return {"team": parent}
        return parent

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
                **self._done_payload(done, done.clone()),
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
                **self._done_payload(die, terminated),
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
    planning_horizon=TWO_SEQ_HORIZON,
    num_candidates=4,
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
        temperature=0.1,
        planning_horizon=planning_horizon,
        optim_steps=1,
        num_candidates=num_candidates,
        top_k=top_k,
        reward_key=("next", "reward"),
    )


def _assert_planner_selects(env, planner_name, device, expected_action):
    die_seq = _die_then_jackpot_sequence(device)
    live_seq = _steady_live_sequence(device)
    planner = _make_planner(planner_name, env)
    td = env.reset()
    with _fixed_planner_candidates(planner_name, die_seq, live_seq, die_seq, live_seq):
        out = planner(td)
    torch.testing.assert_close(out.get("action"), expected_action)


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


@pytest.mark.parametrize("device", get_default_devices())
class TestPlannerDoneMask:
    @pytest.mark.parametrize("planner_name", ["cem", "mppi"])
    @pytest.mark.parametrize("terminate_on_done", [True, False])
    @pytest.mark.parametrize("done_layout", ["root", "nested", "both", "nested_parent"])
    def test_planner_picks_higher_masked_sequence(
        self, device, planner_name, terminate_on_done, done_layout
    ):
        env = _TwoSequenceEnv(
            device=device,
            terminate_on_done=terminate_on_done,
            done_layout=done_layout,
        )
        has_root = "done" in env.done_keys
        has_nested = ("agent", "done") in env.done_keys
        has_team = ("team", "done") in env.done_keys
        has_team_agent = ("team", "agent", "done") in env.done_keys
        assert has_root is (done_layout in ("root", "both"))
        assert has_nested is (done_layout in ("nested", "both"))
        assert has_team is (done_layout == "nested_parent")
        assert has_team_agent is (done_layout == "nested_parent")
        # Parent done stays false when a parent and child group coexist, so
        # the jackpot return is the env return and must beat the live sequence.
        expected = (
            _die_then_jackpot_sequence(device)[0]
            if done_layout in ("both", "nested_parent")
            else _steady_live_sequence(device)[0]
        )
        _assert_planner_selects(env, planner_name, device, expected)


TDMPC2_OBSERVATION_DIM = 5
TDMPC2_ACTION_DIM = 2
TDMPC2_LATENT_DIM = 8
TDMPC2_HORIZON = 2


def _make_tdmpc2_planner(**kwargs):
    from hydra.utils import instantiate
    from torchrl.trainers.algorithms.configs import (
        TdMpc2PolicyPriorConfig,
        TdMpc2QEnsembleConfig,
        TdMpc2WorldModelConfig,
    )

    world_model = instantiate(
        TdMpc2WorldModelConfig(
            observation_dim=TDMPC2_OBSERVATION_DIM,
            action_dim=TDMPC2_ACTION_DIM,
            latent_dim=TDMPC2_LATENT_DIM,
            encoder_dim=12,
            mlp_dim=16,
            simnorm_dim=4,
            num_bins=5,
        )
    )
    policy_prior = instantiate(
        TdMpc2PolicyPriorConfig(
            latent_dim=TDMPC2_LATENT_DIM,
            action_dim=TDMPC2_ACTION_DIM,
            mlp_dim=16,
        )
    )
    q_ensemble = instantiate(
        TdMpc2QEnsembleConfig(
            latent_dim=TDMPC2_LATENT_DIM,
            action_dim=TDMPC2_ACTION_DIM,
            mlp_dim=16,
            num_q=2,
            num_bins=5,
            dropout=0.0,
        )
    )
    planner_kwargs = {
        "horizon": TDMPC2_HORIZON,
        "discount": 0.97,
        "num_samples": 4,
        "num_elites": 2,
        "num_pi_trajs": 1,
        "iterations": 2,
    }
    planner_kwargs.update(kwargs)
    return TdMpc2Planner(world_model, policy_prior, q_ensemble, **planner_kwargs)


def _make_tdmpc2_td(batch_shape=(2,), *, is_init=True, previous_mean=None):
    observation = torch.randn(*batch_shape, TDMPC2_OBSERVATION_DIM)
    if previous_mean is None:
        previous_mean = torch.zeros(*batch_shape, TDMPC2_HORIZON, TDMPC2_ACTION_DIM)
    return TensorDict(
        {
            "observation": observation,
            "is_init": torch.full((*batch_shape, 1), is_init, dtype=torch.bool),
            "_tdmpc2_prev_mean": previous_mean,
        },
        batch_size=batch_shape,
    )


class TestTdMpc2Planner:
    def test_output(self):
        planner = _make_tdmpc2_planner()
        td = _make_tdmpc2_td()
        output = planner(td)

        assert output is td
        assert output["action"].shape == (2, TDMPC2_ACTION_DIM)
        assert output["next", "_tdmpc2_prev_mean"].shape == (
            2,
            TDMPC2_HORIZON,
            TDMPC2_ACTION_DIM,
        )
        assert planner.out_keys == ["action", ("next", "_tdmpc2_prev_mean")]
        assert torch.isfinite(output["action"]).all()
        assert (output["action"].abs() <= 1).all()

    def test_eval_exploration(self):
        planner = _make_tdmpc2_planner(
            num_samples=1,
            num_elites=1,
            num_pi_trajs=0,
            iterations=1,
            min_std=0.25,
            max_std=0.25,
        )
        td = _make_tdmpc2_td(batch_shape=(1,))

        planner.train()
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            torch.manual_seed(0)
            deterministic = planner(td.clone())
        with set_exploration_type(ExplorationType.RANDOM):
            torch.manual_seed(0)
            exploratory = planner(td.clone())

        fitted_action = deterministic["next", "_tdmpc2_prev_mean"][..., 0, :]
        torch.testing.assert_close(deterministic["action"], fitted_action)
        assert not torch.allclose(exploratory["action"], deterministic["action"])

        planner.eval()
        with set_exploration_type(None):
            torch.manual_seed(0)
            direct_eval = planner(td.clone())
        torch.testing.assert_close(
            direct_eval["action"],
            direct_eval["next", "_tdmpc2_prev_mean"][..., 0, :],
        )
        planner.train()

    def test_batch_dims(self):
        planner = _make_tdmpc2_planner(num_pi_trajs=0)
        td = _make_tdmpc2_td((2, 3))
        planner(td)

        assert td["action"].shape == (2, 3, TDMPC2_ACTION_DIM)
        assert td["next", "_tdmpc2_prev_mean"].shape == (
            2,
            3,
            TDMPC2_HORIZON,
            TDMPC2_ACTION_DIM,
        )

    def test_primer(self):
        planner = _make_tdmpc2_planner()
        primer = planner.make_tensordict_primer()

        assert planner.in_keys[-1] == "_tdmpc2_prev_mean"
        assert planner.out_keys[-1] == ("next", "_tdmpc2_prev_mean")
        assert primer.primers["_tdmpc2_prev_mean"].shape == (
            TDMPC2_HORIZON,
            TDMPC2_ACTION_DIM,
        )

    def test_partial_reset(self):
        planner = _make_tdmpc2_planner(num_pi_trajs=0)
        observations = torch.randn(2, TDMPC2_OBSERVATION_DIM)
        previous_mean = torch.stack(
            [
                torch.zeros(TDMPC2_HORIZON, TDMPC2_ACTION_DIM),
                torch.ones(TDMPC2_HORIZON, TDMPC2_ACTION_DIM),
            ]
        )
        batch = TensorDict(
            {
                "observation": observations,
                "is_init": torch.tensor([[True], [False]]),
                "_tdmpc2_prev_mean": previous_mean,
            },
            batch_size=[2],
        )

        torch.manual_seed(0)
        batched = planner(batch)
        torch.manual_seed(0)
        first = planner(
            TensorDict(
                {
                    "observation": observations[0],
                    "is_init": torch.tensor(True),
                    "_tdmpc2_prev_mean": previous_mean[0],
                },
                batch_size=[],
            )
        )
        second = planner(
            TensorDict(
                {
                    "observation": observations[1],
                    "is_init": torch.tensor(False),
                    "_tdmpc2_prev_mean": previous_mean[1],
                },
                batch_size=[],
            )
        )

        torch.testing.assert_close(batched["action"][0], first["action"])
        torch.testing.assert_close(batched["action"][1], second["action"])

    def test_invalid_iterations(self):
        with pytest.raises(ValueError):
            _make_tdmpc2_planner(iterations=0)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
