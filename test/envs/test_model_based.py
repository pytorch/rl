# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import re

import numpy as np
import pytest
import torch

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data import Composite, TensorDictReplayBuffer, Unbounded
from torchrl.data.tensor_specs import Categorical
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based import WorldModelEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import SafeModule, WorldModel
from torchrl.modules.tensordict_module import WorldModelWrapper
from torchrl.testing import CatLinear, get_default_devices
from torchrl.testing.mocking_classes import ActionObsMergeLinear, DummyModelBasedEnvBase


class TestModelBasedEnvBase:
    @staticmethod
    def world_model():
        return WorldModelWrapper(
            SafeModule(
                ActionObsMergeLinear(5, 4),
                in_keys=["hidden_observation", "action"],
                out_keys=["hidden_observation"],
            ),
            SafeModule(
                nn.Linear(4, 1),
                in_keys=["hidden_observation"],
                out_keys=["reward"],
            ),
        )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_mb_rollout(self, device, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        world_model = self.world_model()
        mb_env = DummyModelBasedEnvBase(
            world_model, device=device, batch_size=torch.Size([10])
        )
        check_env_specs(mb_env)
        rollout = mb_env.rollout(max_steps=100)
        expected_keys = {
            ("next", key)
            for key in (*mb_env.observation_spec.keys(), "reward", "done", "terminated")
        }
        expected_keys = expected_keys.union(
            set(mb_env.input_spec["full_action_spec"].keys())
        )
        expected_keys = expected_keys.union(
            set(mb_env.input_spec["full_state_spec"].keys())
        )
        expected_keys = expected_keys.union({"done", "terminated", "next"})
        assert set(rollout.keys(True)) == expected_keys
        assert rollout[("next", "hidden_observation")].shape == (10, 100, 4)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_mb_env_batch_lock(self, device, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        world_model = WorldModelWrapper(
            SafeModule(
                ActionObsMergeLinear(5, 4),
                in_keys=["hidden_observation", "action"],
                out_keys=["hidden_observation"],
            ),
            SafeModule(
                nn.Linear(4, 1),
                in_keys=["hidden_observation"],
                out_keys=["reward"],
            ),
        )
        mb_env = DummyModelBasedEnvBase(
            world_model, device=device, batch_size=torch.Size([10])
        )
        assert not mb_env.batch_locked

        with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
            mb_env.batch_locked = False
        td = mb_env.reset()
        td["action"] = mb_env.full_action_spec[mb_env.action_key].rand()
        td_expanded = td.unsqueeze(-1).expand(10, 2).reshape(-1).to_tensordict()
        mb_env.step(td)

        with pytest.raises(
            RuntimeError,
            match=re.escape("Expected a tensordict with shape==env.batch_size"),
        ):
            mb_env.step(td_expanded)

        mb_env = DummyModelBasedEnvBase(
            world_model, device=device, batch_size=torch.Size([])
        )
        assert not mb_env.batch_locked

        with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
            mb_env.batch_locked = False
        td = mb_env.reset()
        td["action"] = mb_env.full_action_spec[mb_env.action_key].rand()
        td_expanded = td.expand(2)
        mb_env.step(td)
        # we should be able to do a step with a tensordict that has been expended
        mb_env.step(td_expanded)


# ---------------------------------------------------------------------------
# WorldModel + WorldModelEnv tests
# (moved here from test/test_world_model.py per review consolidation request)
# ---------------------------------------------------------------------------

_WM_OBS_DIM = 8
_WM_LATENT_DIM = 4
_WM_ACTION_DIM = 2
_WM_BATCH = 3


def _make_linear_world_model(
    with_done_head: bool = False, with_decoder: bool = False
) -> WorldModel:
    encoder = TensorDictModule(
        torch.nn.Linear(_WM_OBS_DIM, _WM_LATENT_DIM),
        in_keys=["observation"],
        out_keys=["latent"],
    )
    dynamics = TensorDictModule(
        CatLinear(_WM_LATENT_DIM + _WM_ACTION_DIM, _WM_LATENT_DIM),
        in_keys=["latent", "action"],
        out_keys=[("next", "latent")],
    )
    reward_head = TensorDictModule(
        torch.nn.Linear(_WM_LATENT_DIM, 1),
        in_keys=[("next", "latent")],
        out_keys=[("next", "reward")],
    )
    done_head = None
    if with_done_head:
        done_head = TensorDictModule(
            torch.nn.Linear(_WM_LATENT_DIM, 1),
            in_keys=[("next", "latent")],
            out_keys=[("next", "done")],
        )
    decoder = None
    if with_decoder:
        decoder = TensorDictModule(
            torch.nn.Linear(_WM_LATENT_DIM, _WM_OBS_DIM),
            in_keys=["latent"],
            out_keys=["reconstructed_observation"],
        )
    return WorldModel(
        encoder, dynamics, reward_head, done_head=done_head, decoder=decoder
    )


def _make_start_td(batch: int = _WM_BATCH) -> TensorDict:
    wm = _make_linear_world_model()
    td = TensorDict(
        {"observation": torch.randn(batch, _WM_OBS_DIM)}, batch_size=[batch]
    )
    return wm.encode(td)


def _constant_policy(
    latent_dim: int = _WM_LATENT_DIM, action_dim: int = _WM_ACTION_DIM
):
    return TensorDictModule(
        torch.nn.Linear(latent_dim, action_dim),
        in_keys=["latent"],
        out_keys=["action"],
    )


class _SpecOnlyEnv(EnvBase):
    """Minimal :class:`EnvBase` used to supply specs to :class:`WorldModelEnv`.

    Not steppable on its own — only its specs are read by the env wrapper.
    """

    def __init__(self, batch_size: int = _WM_BATCH, device: str = "cpu") -> None:
        super().__init__(batch_size=[batch_size], device=device)
        self.observation_spec = Composite(
            latent=Unbounded(shape=(batch_size, _WM_LATENT_DIM), device=device),
            shape=[batch_size],
            device=device,
        )
        self.action_spec = Unbounded(shape=(batch_size, _WM_ACTION_DIM), device=device)
        self.reward_spec = Unbounded(shape=(batch_size, 1), device=device)
        self.done_spec = Categorical(
            n=2, shape=(batch_size, 1), dtype=torch.bool, device=device
        )

    def _reset(self, tensordict, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError

    def _step(self, tensordict):  # pragma: no cover - never called
        raise NotImplementedError

    def _set_seed(self, seed):  # pragma: no cover - never called
        return seed


class TestWorldModelForward:
    def test_output_keys_present(self):
        wm = _make_linear_world_model()
        td = TensorDict(
            {
                "observation": torch.randn(_WM_BATCH, _WM_OBS_DIM),
                "action": torch.randn(_WM_BATCH, _WM_ACTION_DIM),
            },
            batch_size=[_WM_BATCH],
        )
        out = wm(td)
        assert ("next", "latent") in out.keys(include_nested=True)
        assert ("next", "reward") in out.keys(include_nested=True)

    def test_encode_shortcut(self):
        wm = _make_linear_world_model()
        td = TensorDict(
            {"observation": torch.randn(_WM_BATCH, _WM_OBS_DIM)},
            batch_size=[_WM_BATCH],
        )
        encoded = wm.encode(td)
        assert "latent" in encoded.keys()

    def test_step_shortcut(self):
        wm = _make_linear_world_model()
        td = _make_start_td()
        td["action"] = torch.randn(_WM_BATCH, _WM_ACTION_DIM)
        stepped = wm.step(td)
        assert ("next", "latent") in stepped.keys(include_nested=True)

    def test_decode_shortcut(self):
        wm = _make_linear_world_model(with_decoder=True)
        td = _make_start_td()
        decoded = wm.decode(td)
        assert "reconstructed_observation" in decoded.keys()

    def test_decode_without_decoder_raises(self):
        wm = _make_linear_world_model(with_decoder=False)
        td = _make_start_td()
        with pytest.raises(RuntimeError, match="decoder"):
            wm.decode(td)

    def test_nested_latent_key(self):
        """A WorldModel composed with nested latent keys still works end-to-end."""
        encoder = TensorDictModule(
            torch.nn.Linear(_WM_OBS_DIM, _WM_LATENT_DIM),
            in_keys=["observation"],
            out_keys=[("encoded", "latent")],
        )
        dynamics = TensorDictModule(
            CatLinear(_WM_LATENT_DIM + _WM_ACTION_DIM, _WM_LATENT_DIM),
            in_keys=[("encoded", "latent"), "action"],
            out_keys=[("next", "encoded", "latent")],
        )
        reward_head = TensorDictModule(
            torch.nn.Linear(_WM_LATENT_DIM, 1),
            in_keys=[("next", "encoded", "latent")],
            out_keys=[("next", "reward")],
        )
        wm = WorldModel(encoder, dynamics, reward_head)
        td = TensorDict(
            {
                "observation": torch.randn(_WM_BATCH, _WM_OBS_DIM),
                "action": torch.randn(_WM_BATCH, _WM_ACTION_DIM),
            },
            batch_size=[_WM_BATCH],
        )
        out = wm(td)
        assert ("next", "encoded", "latent") in out.keys(include_nested=True)


class TestWorldModelEnv:
    """End-to-end tests for :class:`WorldModelEnv`.

    The world model owns prediction; the env owns rollout semantics. Each test
    builds a ``WorldModelEnv`` around a ``WorldModel`` and drives it through
    :meth:`EnvBase.rollout` so reset/step/done handling stays consistent with
    every other TorchRL env.
    """

    @staticmethod
    def _make_env(wm: WorldModel) -> WorldModelEnv:
        return WorldModelEnv(wm, base_env=_SpecOnlyEnv(batch_size=_WM_BATCH))

    def test_rollout_shape(self):
        wm = _make_linear_world_model()
        env = self._make_env(wm)
        start_td = _make_start_td()
        policy = _constant_policy()
        horizon = 5
        out = env.rollout(max_steps=horizon, policy=policy, tensordict=start_td)
        assert out.shape == torch.Size([_WM_BATCH, horizon])

    def test_rollout_no_done_head(self):
        """Without a done head, rollouts run for the requested ``max_steps``."""
        wm = _make_linear_world_model(with_done_head=False)
        env = self._make_env(wm)
        start_td = _make_start_td()
        policy = _constant_policy()
        out = env.rollout(max_steps=4, policy=policy, tensordict=start_td)
        assert out.shape == torch.Size([_WM_BATCH, 4])

    def test_rollout_break_when_done(self):
        """When the done head always predicts done=True, env.rollout stops early."""
        wm = _make_linear_world_model(with_done_head=True)
        # Override done head to always emit True so we can verify early stop.
        wm.done_head = TensorDictModule(
            lambda x: torch.ones(*x.shape[:-1], 1, dtype=torch.bool),
            in_keys=[("next", "latent")],
            out_keys=[("next", "done")],
        )
        from tensordict.nn import TensorDictSequential

        wm._step_seq = TensorDictSequential(wm.dynamics, wm.reward_head, wm.done_head)

        env = self._make_env(wm)
        start_td = _make_start_td()
        policy = _constant_policy()
        out = env.rollout(
            max_steps=10,
            policy=policy,
            tensordict=start_td,
            break_when_any_done=True,
        )
        assert out.shape[1] == 1

    def test_rollout_contains_reward(self):
        wm = _make_linear_world_model()
        env = self._make_env(wm)
        start_td = _make_start_td()
        policy = _constant_policy()
        out = env.rollout(max_steps=3, policy=policy, tensordict=start_td)
        assert ("next", "reward") in out.keys(include_nested=True)

    def test_rollout_replay_buffer_compatible(self):
        """Imagined rollout can be added directly to a TensorDictReplayBuffer."""
        from torchrl.data import LazyTensorStorage

        wm = _make_linear_world_model()
        env = self._make_env(wm)
        start_td = _make_start_td()
        policy = _constant_policy()
        rollout_td = env.rollout(max_steps=5, policy=policy, tensordict=start_td)
        flat = rollout_td.reshape(-1)
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=100))
        rb.extend(flat)
        assert len(rb) == flat.batch_size[0]

    def test_reset_requires_latent(self):
        """WorldModelEnv refuses to reset without an explicit starting latent."""
        wm = _make_linear_world_model()
        env = self._make_env(wm)
        with pytest.raises(RuntimeError, match="initial latent"):
            env.reset()
