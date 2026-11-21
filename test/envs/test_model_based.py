# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import re

import numpy as np
import pytest
import torch

from tensordict import TensorDict, unravel_key
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data import Composite, TensorDictReplayBuffer, Unbounded
from torchrl.data.tensor_specs import Categorical
from torchrl.envs import SerialEnv
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based import WorldModelEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import SafeModule, WorldModel
from torchrl.modules.tensordict_module import WorldModelWrapper
from torchrl.testing import CatLinear, get_default_devices
from torchrl.testing.mocking_classes import (
    ActionObsMergeLinear,
    ContinuousActionVecMockEnv,
    DummyModelBasedEnvBase,
)


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

    @staticmethod
    def _make_observation_world_model(
        latent_key, observation_dim, action_dim, *, encoder=None, decoder=False
    ):
        next_latent_key = unravel_key(("next", latent_key))
        if encoder is None:
            encoder = TensorDictModule(
                nn.Sequential(
                    nn.Linear(observation_dim, _WM_LATENT_DIM),
                    nn.BatchNorm1d(_WM_LATENT_DIM),
                ),
                in_keys=["observation"],
                out_keys=[latent_key],
            )
        dynamics = TensorDictModule(
            CatLinear(_WM_LATENT_DIM + action_dim, _WM_LATENT_DIM),
            in_keys=[latent_key, "action"],
            out_keys=[next_latent_key],
        )
        reward_head = TensorDictModule(
            nn.Linear(_WM_LATENT_DIM, 1),
            in_keys=[next_latent_key],
            out_keys=[("next", "reward")],
        )
        decoder_module = None
        if decoder:
            decoder_module = TensorDictModule(
                nn.Linear(_WM_LATENT_DIM, observation_dim),
                in_keys=[next_latent_key],
                out_keys=[("next", "observation")],
            )
        return WorldModel(encoder, dynamics, reward_head, decoder=decoder_module)

    @pytest.mark.parametrize("latent_key", ["latent", ("encoded", "latent")])
    @pytest.mark.parametrize("override", [None, "composite", "leaf"])
    def test_latent_advances_with_observation_env(self, latent_key, override):
        """The imagined latent follows the dynamics when the reference env exposes raw observations.

        :class:`ContinuousActionVecMockEnv` is unbatched and declares
        ``"observation"`` rather than the latent, as a real env does, so the
        latent spec (dim 4, unlike the 7-dim observation) comes from the
        encoder and is added to an override that omits it (a non-composite
        override is stored under ``"observation"``). Before the fix the latent
        was dropped after each step, so every step restarted from the reset
        latent, and action / reward specs lost their feature dim.
        """
        base_env = ContinuousActionVecMockEnv()
        observation_dim = base_env.observation_spec["observation"].shape[-1]
        action_dim = base_env.action_spec.shape[-1]
        wm = self._make_observation_world_model(latent_key, observation_dim, action_dim)
        observation_spec = None
        if override == "composite":
            observation_spec = Composite(
                pixels=Unbounded(shape=(_WM_BATCH, 3)), shape=[_WM_BATCH]
            )
        elif override == "leaf":
            observation_spec = Unbounded(shape=(_WM_BATCH, observation_dim))
        env = WorldModelEnv(
            wm,
            base_env=base_env,
            batch_size=[_WM_BATCH],
            observation_spec=observation_spec,
            latent_key=latent_key,
        )
        assert env.action_spec.shape == (_WM_BATCH, action_dim)
        assert env.reward_spec.shape == (_WM_BATCH, 1)
        assert env.observation_spec[latent_key].shape == (_WM_BATCH, _WM_LATENT_DIM)
        if override == "leaf":
            assert "observation" in env.observation_spec.keys()

        policy = TensorDictModule(
            nn.Linear(_WM_LATENT_DIM, action_dim),
            in_keys=[latent_key],
            out_keys=["action"],
        )
        wm.eval()
        start_td = wm.encode(
            TensorDict(
                {"observation": torch.randn(_WM_BATCH, observation_dim)},
                batch_size=[_WM_BATCH],
            )
        )
        horizon = 4
        next_latent_key = unravel_key(("next", latent_key))
        with torch.no_grad():
            rollout = env.rollout(horizon, policy=policy, tensordict=start_td)
            latent = start_td.get(latent_key)
            expected = []
            for _ in range(horizon):
                latent = wm.dynamics.module(latent, policy.module(latent))
                expected.append(latent)
        torch.testing.assert_close(
            rollout.get(next_latent_key), torch.stack(expected, dim=1)
        )

    @pytest.mark.parametrize("reference", ["unbatched", "batched"])
    def test_decoded_observations_are_exposed(self, reference):
        """With the default spec, a decoder's ``("next", "observation")`` reaches the rollout.

        The reference env is unbatched, or batched with a batch size (2) that
        differs from the imagined one, which only works once the reference
        env's batch dims are dropped from its specs.
        """
        if reference == "batched":
            base_env = SerialEnv(2, ContinuousActionVecMockEnv)
        else:
            base_env = ContinuousActionVecMockEnv()
        observation_dim = base_env.observation_spec["observation"].shape[-1]
        action_dim = base_env.action_spec.shape[-1]
        wm = self._make_observation_world_model(
            "latent", observation_dim, action_dim, decoder=True
        )
        env = WorldModelEnv(wm, base_env=base_env, batch_size=[_WM_BATCH])
        assert env.action_spec.shape == (_WM_BATCH, action_dim)
        assert env.observation_spec["observation"].shape == (
            _WM_BATCH,
            observation_dim,
        )
        wm.eval()
        start_td = wm.encode(
            TensorDict(
                {"observation": torch.randn(_WM_BATCH, observation_dim)},
                batch_size=[_WM_BATCH],
            )
        )
        with torch.no_grad():
            rollout = env.rollout(3, tensordict=start_td)
            expected = wm.decoder.module(rollout["next", "latent"])
        torch.testing.assert_close(rollout["next", "observation"], expected)

    def test_latent_spec_inference_leaves_encoder_untouched(self):
        """Inferring the latent spec leaves a train-mode encoder and the global RNG untouched.

        The encoder holds a BatchNorm (whose running statistics a train-mode
        pass would update) and samples noise (which would advance the RNG).
        """
        base_env = ContinuousActionVecMockEnv()
        observation_dim = base_env.observation_spec["observation"].shape[-1]
        action_dim = base_env.action_spec.shape[-1]
        batch_norm = nn.BatchNorm1d(_WM_LATENT_DIM)
        encoder = TensorDictSequential(
            TensorDictModule(
                nn.Sequential(nn.Linear(observation_dim, _WM_LATENT_DIM), batch_norm),
                in_keys=["observation"],
                out_keys=["latent"],
            ),
            TensorDictModule(
                lambda latent: latent + torch.randn_like(latent),
                in_keys=["latent"],
                out_keys=["latent"],
            ),
        )
        wm = self._make_observation_world_model(
            "latent", observation_dim, action_dim, encoder=encoder
        )
        wm.train()
        running_mean = batch_norm.running_mean.clone()
        rng_state = torch.get_rng_state()
        WorldModelEnv(wm, base_env=base_env, batch_size=[_WM_BATCH])
        assert batch_norm.training
        torch.testing.assert_close(batch_norm.running_mean, running_mean)
        assert torch.equal(torch.get_rng_state(), rng_state)

    def test_latent_spec_inference_reads_the_action(self):
        """An encoder conditioned on the previous action gets its latent spec inferred."""
        base_env = ContinuousActionVecMockEnv()
        observation_dim = base_env.observation_spec["observation"].shape[-1]
        action_dim = base_env.action_spec.shape[-1]
        encoder = TensorDictModule(
            CatLinear(observation_dim + action_dim, _WM_LATENT_DIM),
            in_keys=["observation", "action"],
            out_keys=["latent"],
        )
        wm = self._make_observation_world_model(
            "latent", observation_dim, action_dim, encoder=encoder
        )
        env = WorldModelEnv(wm, base_env=base_env, batch_size=[_WM_BATCH])
        assert env.observation_spec["latent"].shape == (_WM_BATCH, _WM_LATENT_DIM)

    def test_latent_spec_inference_composite_latent(self):
        """A latent stored as a sub-tensordict gets one spec entry per leaf."""
        base_env = ContinuousActionVecMockEnv()
        observation_dim = base_env.observation_spec["observation"].shape[-1]
        action_dim = base_env.action_spec.shape[-1]
        encoder = TensorDictModule(
            lambda observation: (observation[..., :4], observation[..., :2].long()),
            in_keys=["observation"],
            out_keys=[("latent", "state"), ("latent", "belief")],
        )
        wm = self._make_observation_world_model(
            "latent", observation_dim, action_dim, encoder=encoder
        )
        env = WorldModelEnv(wm, base_env=base_env, batch_size=[_WM_BATCH])
        spec = env.observation_spec["latent"]
        assert isinstance(spec, Composite)
        assert spec["state"].shape == (_WM_BATCH, 4)
        assert spec["belief"].shape == (_WM_BATCH, 2)
        assert spec["belief"].dtype == torch.long

    def test_latent_spec_inference_error(self):
        """An encoder that reads a key the reference env does not declare gets a clear error."""
        base_env = ContinuousActionVecMockEnv()
        observation_dim = base_env.observation_spec["observation"].shape[-1]
        encoder = TensorDictModule(
            CatLinear(2 * observation_dim, _WM_LATENT_DIM),
            in_keys=["observation", "belief"],
            out_keys=["latent"],
        )
        wm = _make_linear_world_model()
        wm = WorldModel(encoder, wm.dynamics, wm.reward_head)
        with pytest.raises(ValueError, match="Pass an observation_spec"):
            WorldModelEnv(wm, base_env=base_env, batch_size=[_WM_BATCH])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
