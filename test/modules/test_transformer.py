# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import copy
import os
import pickle
import warnings

import pytest
import torch
from tensordict import from_module, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.collectors import Collector, VanillaWeightUpdater
from torchrl.envs import InitTracker, SerialEnv, TransformedEnv
from torchrl.modules import CausalTransformer, set_recurrent_mode, TransformerModule
from torchrl.modules.tensordict_module.transformer import (
    positions_from_is_init,
    segment_causal_mask_from_is_init,
)
from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv


class TestMaskHelpers:
    def test_positions_from_is_init(self):
        is_init = torch.tensor(
            [
                [False, False, True, False, False],
                [True, False, False, True, True],
            ]
        )
        positions = positions_from_is_init(is_init)
        expected = torch.tensor([[0, 1, 0, 1, 2], [0, 1, 2, 0, 0]])
        torch.testing.assert_close(positions, expected)

    def test_positions_window_start_is_zero(self):
        is_init = torch.zeros(2, 4, dtype=torch.bool)
        positions = positions_from_is_init(is_init)
        torch.testing.assert_close(positions, torch.arange(4).expand(2, 4))

    def test_positions_rejects_non_bool(self):
        with pytest.raises(ValueError, match="boolean"):
            positions_from_is_init(torch.zeros(2, 4))

    def test_segment_mask_blocks_cross_episode_attention(self):
        is_init = torch.tensor([[False, False, True, False]])
        mask = segment_causal_mask_from_is_init(is_init)
        expected = torch.tensor(
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, False, True, False],
                    [False, False, True, True],
                ]
            ]
        )
        torch.testing.assert_close(mask, expected)


def _window(obs, is_init, batch_shape):
    return TensorDict({"observation": obs, "is_init": is_init}, batch_shape)


def _run_steps(module, obs, is_init, batch_shape):
    """Feed a ``[*batch, T]`` trajectory one step at a time; stack the outputs."""
    outs = []
    for step in range(obs.shape[-2]):
        td = _window(obs[..., step, :], is_init[..., step, :], batch_shape)
        module(td)
        assert "transformer_state" not in td.keys()
        outs.append(td["embed"].clone())
    return torch.stack(outs, dim=-2)


class _MinimalBackbone(nn.Module):
    """Contract-only backbone: a projection standing in for attention."""

    def __init__(self, input_size, hidden_size, max_seq_len):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.num_layers = 1
        self.num_heads = 1
        self.head_dim = hidden_size
        self.max_seq_len = max_seq_len
        self.cache_dtypes = []

    def forward(self, features, positions, mask=None, kv_cache=None):
        out = self.proj(features)
        if kv_cache is not None:
            rows = torch.arange(out.shape[0], device=out.device)
            kv_cache[rows, positions.squeeze(-1)] = out.squeeze(1).to(kv_cache.dtype)
        return out, kv_cache

    def new_kv_cache(self, batch_size, *, device=None, dtype=None):
        self.cache_dtypes.append(dtype)
        dtype = self.proj.weight.dtype if dtype is None else dtype
        return torch.zeros(
            batch_size, self.max_seq_len, self.head_dim, device=device, dtype=dtype
        )

    def reset_kv_cache(self, kv_cache, mask):
        return kv_cache.masked_fill_(mask.view(-1, 1, 1), 0)


class TestTransformerModule:
    @staticmethod
    def _make_module(
        input_size=5,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        max_seq_len=12,
        device=None,
        **kwargs,
    ):
        return TransformerModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            in_key="observation",
            out_key="embed",
            device=device,
            **kwargs,
        )

    @staticmethod
    def _trajectory(shape, t, device=None, dtype=torch.float32):
        torch.manual_seed(0)
        obs = torch.randn(*shape, t, 5, device=device, dtype=dtype)
        is_init = torch.zeros(*shape, t, 1, dtype=torch.bool, device=device)
        is_init[..., 0, :] = True
        if t > 7:
            is_init[..., 0, 4, :] = True
            is_init[..., -1, 7, :] = True
        return obs, is_init

    def test_errs(self):
        with pytest.raises(ValueError, match="divisible"):
            CausalTransformer(3, 10, 1, num_heads=3, max_seq_len=8)
        with pytest.raises(ValueError, match="hidden_size must be passed"):
            TransformerModule(input_size=3, in_key="observation", out_key="embed")
        with pytest.raises(ValueError, match="num_heads and max_seq_len"):
            TransformerModule(
                input_size=3, hidden_size=8, in_key="observation", out_key="embed"
            )
        with pytest.raises(ValueError, match="cannot be passed along"):
            TransformerModule(
                input_size=3,
                transformer=CausalTransformer(3, 8, 1, num_heads=2, max_seq_len=8),
                in_key="observation",
                out_key="embed",
            )
        with pytest.raises(ValueError, match="must expose"):
            TransformerModule(
                transformer=nn.Linear(3, 8), in_key="observation", out_key="embed"
            )

        class NoCache(nn.Module):
            num_layers = num_heads = head_dim = max_seq_len = 1

        with pytest.raises(ValueError, match="must implement 'new_kv_cache'"):
            TransformerModule(
                transformer=NoCache(), in_key="observation", out_key="embed"
            )
        with pytest.raises(ValueError, match="expects 1 input"):
            TransformerModule(
                input_size=3,
                hidden_size=8,
                num_heads=2,
                max_seq_len=8,
                in_keys=["observation", "extra"],
                out_key="embed",
            )
        with pytest.raises(ValueError, match="expects 1 output"):
            TransformerModule(
                input_size=3,
                hidden_size=8,
                num_heads=2,
                max_seq_len=8,
                in_key="observation",
                out_keys=["embed", "extra"],
            )
        with pytest.raises(ValueError, match="not both"):
            TransformerModule(
                input_size=3,
                hidden_size=8,
                num_heads=2,
                max_seq_len=8,
                out_key="embed",
            )

        class _NoDtype(_MinimalBackbone):
            def new_kv_cache(self, batch_size, *, device=None):
                return super().new_kv_cache(batch_size, device=device)

        with pytest.raises(ValueError, match="dtype keyword"):
            TransformerModule(
                transformer=_NoDtype(3, 8, max_seq_len=8),
                in_key="observation",
                out_key="embed",
            )

    @pytest.mark.parametrize("shape", [[3], [2, 3]])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_step_vs_window_parity(self, shape, device):
        t = 10
        module = self._make_module(max_seq_len=t, device=device)
        obs, is_init = self._trajectory(shape, t, device=device)
        td = _window(obs, is_init, [*shape, t])
        with set_recurrent_mode(True):
            module(td)
        assert "transformer_state" not in td.keys()
        step_out = _run_steps(module, obs, is_init, shape)
        torch.testing.assert_close(td["embed"], step_out, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
    def test_non_float32_module(self, dtype):
        t = 10
        module = self._make_module(max_seq_len=t).to(dtype)
        obs, is_init = self._trajectory([3], t, dtype=dtype)
        td = _window(obs, is_init, [3, t])
        with set_recurrent_mode(True):
            module(td)
        step_out = _run_steps(module, obs, is_init, [3])
        assert step_out.dtype == dtype
        tol = 5e-2 if dtype is torch.bfloat16 else 1e-10
        torch.testing.assert_close(td["embed"], step_out, atol=tol, rtol=tol)

    def test_autocast_collection(self):
        t = 10
        module = self._make_module(max_seq_len=t)
        obs, is_init = self._trajectory([3], t)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            td = _window(obs, is_init, [3, t])
            with set_recurrent_mode(True):
                module(td)
            step_out = _run_steps(module, obs, is_init, [3])
        torch.testing.assert_close(
            td["embed"].float(), step_out.float(), atol=5e-2, rtol=5e-2
        )

    def test_window_requires_episode_aligned_rows(self):
        module = self._make_module()
        obs, is_init = self._trajectory([2], 8)
        is_init[1, 0] = False
        with set_recurrent_mode(True), pytest.raises(
            ValueError, match="episode-aligned"
        ):
            module(_window(obs, is_init, [2, 8]))

    @pytest.mark.parametrize("update", ["in_place", "swap"])
    def test_weight_update_restarts_streams(self, update):
        module = self._make_module()
        obs, _ = self._trajectory([2], 3)
        is_init = torch.tensor([[[True], [False]], [[True], [False]]])
        _run_steps(module, obs[:, :2], is_init, [2])
        if update == "in_place":
            with torch.no_grad():
                module.transformer.in_proj.weight.add_(0.5)
        else:
            params = from_module(module).clone()
            with torch.no_grad():
                params.apply_(lambda t: t.add_(0.5) if t.is_floating_point() else t)
            params.to_module(module)
        continued = _window(obs[:, 2], torch.zeros(2, 1, dtype=torch.bool), [2])
        module(continued)
        module.reset_cache()
        fresh = _window(obs[:, 2], torch.ones(2, 1, dtype=torch.bool), [2])
        module(fresh)
        torch.testing.assert_close(continued["embed"], fresh["embed"])

    def test_reset_cache_and_batch_change(self):
        module = self._make_module()
        obs, is_init = self._trajectory([2], 4)
        first = _run_steps(module, obs, is_init, [2])
        module.reset_cache()
        second = _run_steps(module, obs, is_init, [2])
        torch.testing.assert_close(first, second)
        wider_obs = torch.cat([obs, obs[:1]], 0)
        wider_init = torch.cat([is_init, is_init[:1]], 0)
        wider = _run_steps(module, wider_obs, wider_init, [3])
        torch.testing.assert_close(wider[:2], first)

    def test_inference_mode_step_keeps_cache_usable(self):
        module = self._make_module()
        reference = copy.deepcopy(module)
        obs, is_init = self._trajectory([2], 3)
        is_init[0, 2] = True
        outs = []
        with torch.inference_mode():
            td = _window(obs[:, 0], is_init[:, 0], [2])
            module(td)
            outs.append(td["embed"].clone())
        for step in (1, 2):
            td = _window(obs[:, step], is_init[:, step], [2])
            module(td)
            outs.append(td["embed"].clone())
        expected = _run_steps(reference, obs, is_init, [2])
        torch.testing.assert_close(torch.stack(outs, dim=-2), expected)

    @pytest.mark.parametrize("transfer", ["pickle", "deepcopy"])
    def test_copies_start_with_an_empty_cache(self, transfer):
        module = self._make_module()
        obs, is_init = self._trajectory([2], 3)
        empty_size = len(pickle.dumps(module))
        _run_steps(module, obs[:, :2], is_init[:, :2], [2])
        assert len(pickle.dumps(module)) == empty_size
        if transfer == "pickle":
            copied = pickle.loads(pickle.dumps(module))
        else:
            copied = copy.deepcopy(module)
        continued = _window(obs[:, 2], torch.zeros(2, 1, dtype=torch.bool), [2])
        copied(continued)
        fresh = _window(obs[:, 2], torch.ones(2, 1, dtype=torch.bool), [2])
        module.reset_cache()
        module(fresh)
        torch.testing.assert_close(continued["embed"], fresh["embed"])

    def test_max_seq_len_exceeded_raises(self):
        module = self._make_module(max_seq_len=4)
        with set_recurrent_mode(True), pytest.raises(RuntimeError, match="max_seq_len"):
            module(
                _window(
                    torch.randn(1, 6, 5),
                    torch.tensor([True] + [False] * 5).view(1, 6, 1),
                    [1, 6],
                )
            )
        module(_window(torch.randn(2, 5), torch.ones(2, 1, dtype=torch.bool), [2]))
        for _ in range(3):
            module(_window(torch.randn(2, 5), torch.zeros(2, 1, dtype=torch.bool), [2]))
        with pytest.raises(RuntimeError, match="max_seq_len"):
            module(_window(torch.randn(2, 5), torch.zeros(2, 1, dtype=torch.bool), [2]))

    @pytest.mark.skipif(os.name == "nt", reason="inductor is not available on Windows")
    @pytest.mark.parametrize("recurrent", [False, True])
    def test_fullgraph_compile_matches_eager(self, recurrent):
        module = self._make_module(max_seq_len=8, validate_windows=not recurrent)
        obs, is_init = self._trajectory([2], 3)
        if recurrent:
            td = _window(obs, is_init, [2, 3])
            with set_recurrent_mode(True):
                eager = module(td.clone())["embed"]
                compiled = torch.compile(module, fullgraph=True)(td.clone())["embed"]
            torch.testing.assert_close(compiled, eager, atol=1e-5, rtol=1e-5)
            return
        eager = _run_steps(module, obs, is_init, [2])
        module.reset_cache()
        compiled = _run_steps(torch.compile(module, fullgraph=True), obs, is_init, [2])
        torch.testing.assert_close(compiled, eager, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(os.name == "nt", reason="inductor is not available on Windows")
    def test_window_validation_survives_default_compile(self):
        module = torch.compile(self._make_module(max_seq_len=8))
        obs, is_init = self._trajectory([2], 3)
        is_init[1, 0] = False
        with set_recurrent_mode(True), pytest.raises(
            ValueError, match="episode-aligned"
        ):
            module(_window(obs, is_init, [2, 3]))

    def test_custom_backbone(self):
        backbone = _MinimalBackbone(5, 16, max_seq_len=6)
        module = TransformerModule(
            transformer=backbone, in_key="observation", out_key="embed"
        )
        td = _window(torch.randn(2, 5), torch.ones(2, 1, dtype=torch.bool), [2])
        module(td)
        assert td["embed"].shape == (2, 16)
        is_init = torch.zeros(2, 3, 1, dtype=torch.bool)
        is_init[:, 0] = True
        window = _window(torch.randn(2, 3, 5), is_init, [2, 3])
        with set_recurrent_mode(True):
            module(window)
        assert window["embed"].shape == (2, 3, 16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            module(td)
        assert backbone.cache_dtypes == [None, torch.bfloat16]

    def test_nested_in_key(self):
        module = TransformerModule(
            input_size=5,
            hidden_size=16,
            num_heads=4,
            max_seq_len=8,
            in_key=("data", "observation"),
            out_key=("data", "embed"),
        )
        td = TensorDict(
            {
                ("data", "observation"): torch.randn(2, 5),
                "is_init": torch.ones(2, 1, dtype=torch.bool),
            },
            [2],
        )
        module(td)
        assert td["data", "embed"].shape == (2, 16)

    @staticmethod
    def _make_env():
        return TransformedEnv(ContinuousActionVecMockEnv(), InitTracker())

    def _make_policy(self, env, module):
        policy = TensorDictSequential(
            module,
            TensorDictModule(
                nn.LazyLinear(env.action_spec.shape[-1]),
                in_keys=["embed"],
                out_keys=["action"],
            ),
        )
        policy(env.reset())
        module.reset_cache()
        return policy

    def test_serial_env_rollout_matches_window(self):
        def make_env():
            return self._make_env()

        env = SerialEnv(2, make_env)
        obs_dim = env.observation_spec["observation"].shape[-1]
        module = self._make_module(input_size=obs_dim, max_seq_len=64)
        policy = self._make_policy(env, module)
        rollout = env.rollout(5, policy)
        assert rollout["embed"].shape[:2] == (2, 5)
        with set_recurrent_mode(True):
            window = module(rollout.exclude("embed").clone())
        torch.testing.assert_close(
            window["embed"], rollout["embed"], atol=1e-5, rtol=1e-5
        )

    def test_collector_continuing_episode_across_batches(self):
        env = self._make_env()
        obs_dim = env.observation_spec["observation"].shape[-1]
        module = self._make_module(input_size=obs_dim, max_seq_len=64)
        policy = self._make_policy(env, module)
        collector = Collector(env, policy, frames_per_batch=8, total_frames=16)
        try:
            batches = [data.clone() for data in collector]
        finally:
            collector.shutdown()
        assert len(batches) == 2
        second = batches[1]
        assert not second["is_init"][0].any()
        with set_recurrent_mode(True), pytest.raises(
            ValueError, match="episode-aligned"
        ):
            module(second.exclude("embed").clone())
        episode = torch.cat(batches, dim=0)
        with set_recurrent_mode(True):
            window = module(episode.exclude("embed").clone())
        torch.testing.assert_close(
            window["embed"], episode["embed"], atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("update", ["fallback", "legacy", "state_dict"])
    @pytest.mark.parametrize("compile", [False, True], ids=["eager", "compiled"])
    def test_collector_weight_update_restarts_streams(self, update, compile):
        """Every collector weight path must restart the streams, eager and compiled."""
        if compile and os.name == "nt":
            pytest.skip("inductor is not available on Windows")
        env = self._make_env()
        obs_dim = env.observation_spec["observation"].shape[-1]
        module = self._make_module(input_size=obs_dim, max_seq_len=64)
        policy = self._make_policy(env, module)
        if compile:
            policy.module[0] = torch.compile(module)
        kwargs = {}
        if update == "legacy":
            kwargs["weight_updater"] = VanillaWeightUpdater(
                policy_weights=TensorDict.from_module(policy).data.lock_()
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            collector = Collector(
                env, policy, frames_per_batch=4, total_frames=8, **kwargs
            )
        try:
            iterator = iter(collector)
            next(iterator)
            weights = TensorDict.from_module(policy).data.clone()
            weights.apply_(lambda t: t.add_(0.5) if t.is_floating_point() else t)
            if update == "state_dict":
                state = collector.state_dict()
                state["policy_state_dict"] = weights.flatten_keys(".").to_dict()
                collector.load_state_dict(state)
            else:
                collector.update_policy_weights_(weights)
            second = next(iterator).clone()
        finally:
            collector.shutdown()
        assert not second["is_init"][0].any()
        fresh = TransformerModule(
            transformer=copy.deepcopy(module.transformer),
            in_key="observation",
            out_key="embed",
        )
        transformer_weights = weights["module", "0"]
        if "_orig_mod" in transformer_weights.keys():
            transformer_weights = transformer_weights["_orig_mod"]
        TensorDict.from_module(fresh.transformer).data.update_(
            transformer_weights["transformer"]
        )
        first_step = _window(
            second["observation"][:1], torch.ones(1, 1, dtype=torch.bool), [1]
        )
        fresh(first_step)
        torch.testing.assert_close(
            second["embed"][:1], first_step["embed"], atol=1e-4, rtol=1e-4
        )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
