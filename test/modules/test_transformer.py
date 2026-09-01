# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.collectors import Collector
from torchrl.envs import InitTracker, SerialEnv, TransformedEnv
from torchrl.envs.utils import step_mdp
from torchrl.modules import CausalTransformer, set_recurrent_mode, TransformerModule
from torchrl.modules.tensordict_module.transformer import (
    positions_from_is_init,
    segment_causal_mask_from_is_init,
)
from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv

_STATE_KEYS = [
    ("transformer_state", "k"),
    ("transformer_state", "v"),
    ("transformer_state", "pos"),
]


def _step_state_forward(td):
    for key in _STATE_KEYS:
        td[key] = td[("next", *key)]
    return td


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

    def test_segment_mask_is_causal(self):
        is_init = torch.zeros(1, 6, dtype=torch.bool)
        mask = segment_causal_mask_from_is_init(is_init)
        torch.testing.assert_close(mask[0], torch.ones(6, 6, dtype=torch.bool).tril())

    def test_segment_mask_diagonal_always_true(self):
        is_init = torch.rand(3, 8) > 0.5
        mask = segment_causal_mask_from_is_init(is_init)
        assert mask.diagonal(dim1=-2, dim2=-1).all()


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
        with pytest.raises(ValueError, match="4 inputs"):
            TransformerModule(
                input_size=3,
                hidden_size=8,
                num_heads=2,
                max_seq_len=8,
                in_keys=["observation"],
                out_key="embed",
            )
        with pytest.raises(ValueError, match="4 outputs"):
            TransformerModule(
                input_size=3,
                hidden_size=8,
                num_heads=2,
                max_seq_len=8,
                in_key="observation",
                out_keys=["embed"],
            )
        with pytest.raises(ValueError, match="not both"):
            TransformerModule(
                input_size=3,
                hidden_size=8,
                num_heads=2,
                max_seq_len=8,
                out_key="embed",
            )

    def test_single_step(self):
        module = self._make_module()
        td = TensorDict(
            {
                "observation": torch.randn(2, 5),
                "is_init": torch.ones(2, 1, dtype=torch.bool),
            },
            [2],
        )
        module(td)
        assert td["embed"].shape == (2, 16)
        pos = td[("next", "transformer_state", "pos")]
        torch.testing.assert_close(pos, torch.ones_like(pos))
        td_next = step_mdp(td, keep_other=True)
        _step_state_forward(td)
        td_next.update(td.select(*_STATE_KEYS))
        td_next["is_init"] = torch.zeros(2, 1, dtype=torch.bool)
        module(td_next)
        pos = td_next[("next", "transformer_state", "pos")]
        torch.testing.assert_close(pos, 2 * torch.ones_like(pos))
        assert not torch.isclose(td_next["embed"], td["embed"]).all()

    @pytest.mark.parametrize("shape", [[3], [2, 3]])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_step_vs_window_parity(self, shape, device):
        torch.manual_seed(0)
        t = 10
        module = self._make_module(max_seq_len=t, device=device)
        obs = torch.randn(*shape, t, 5, device=device)
        is_init = torch.zeros(*shape, t, 1, dtype=torch.bool, device=device)
        is_init[..., 0, :] = True
        is_init[..., 0, 4, :] = True
        is_init[..., -1, 7, :] = True

        td = TensorDict({"observation": obs, "is_init": is_init}, [*shape, t])
        with set_recurrent_mode(True):
            module(td)
        window_out = td["embed"]

        td_step = TensorDict(
            {"observation": obs[..., 0, :], "is_init": is_init[..., 0, :]}, shape
        )
        step_outs = []
        for step in range(t):
            td_step["observation"] = obs[..., step, :]
            td_step["is_init"] = is_init[..., step, :]
            module(td_step)
            step_outs.append(td_step["embed"].clone())
            _step_state_forward(td_step)
        step_out = torch.stack(step_outs, dim=-2)
        torch.testing.assert_close(window_out, step_out, atol=1e-5, rtol=1e-5)

    def test_reset_forgets_history(self):
        torch.manual_seed(0)
        module = self._make_module()
        obs = torch.randn(1, 8, 5)
        is_init = torch.zeros(1, 8, 1, dtype=torch.bool)
        is_init[:, 0] = True
        is_init[:, 5] = True
        td = TensorDict({"observation": obs, "is_init": is_init}, [1, 8])
        with set_recurrent_mode(True):
            module(td)

        fresh = TensorDict(
            {
                "observation": obs[:, 5:],
                "is_init": torch.tensor([True, False, False]).view(1, 3, 1),
            },
            [1, 3],
        )
        with set_recurrent_mode(True):
            module(fresh)
        torch.testing.assert_close(td["embed"][:, 5:], fresh["embed"])

    def test_no_state_written_in_recurrent_mode(self):
        module = self._make_module()
        td = TensorDict(
            {
                "observation": torch.randn(2, 4, 5),
                "is_init": torch.zeros(2, 4, 1, dtype=torch.bool),
            },
            [2, 4],
        )
        with set_recurrent_mode(True):
            module(td)
        assert ("next", "transformer_state", "k") not in td.keys(True)
        assert "transformer_state" not in td.keys()

    def test_max_seq_len_exceeded_raises(self):
        module = self._make_module(max_seq_len=4)
        td = TensorDict(
            {
                "observation": torch.randn(1, 6, 5),
                "is_init": torch.zeros(1, 6, 1, dtype=torch.bool),
            },
            [1, 6],
        )
        with set_recurrent_mode(True), pytest.raises(RuntimeError, match="max_seq_len"):
            module(td)

        td_step = TensorDict(
            {
                "observation": torch.randn(2, 5),
                "is_init": torch.ones(2, 1, dtype=torch.bool),
            },
            [2],
        )
        for _ in range(4):
            module(td_step)
            _step_state_forward(td_step)
            td_step["is_init"] = torch.zeros(2, 1, dtype=torch.bool)
        with pytest.raises(RuntimeError, match="max_seq_len"):
            module(td_step)

    def test_custom_backbone(self):
        backbone = CausalTransformer(5, 16, 1, num_heads=2, max_seq_len=6)
        module = TransformerModule(
            transformer=backbone, in_key="observation", out_key="embed"
        )
        td = TensorDict(
            {
                "observation": torch.randn(2, 5),
                "is_init": torch.ones(2, 1, dtype=torch.bool),
            },
            [2],
        )
        module(td)
        assert td["embed"].shape == (2, 16)

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

    def _make_env(self, transformer_module):
        env = TransformedEnv(ContinuousActionVecMockEnv(), InitTracker())
        env.append_transform(transformer_module.make_tensordict_primer())
        return env

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
        return policy

    def test_primer_env_rollout(self):
        env0 = TransformedEnv(ContinuousActionVecMockEnv(), InitTracker())
        obs_dim = env0.observation_spec["observation"].shape[-1]
        module = self._make_module(input_size=obs_dim, max_seq_len=64)
        env = self._make_env(module)
        policy = self._make_policy(env, module)
        rollout = env.rollout(6, policy)
        assert ("next", "transformer_state", "pos") in rollout.keys(True)
        pos = rollout[("next", "transformer_state", "pos")].squeeze(-1)
        torch.testing.assert_close(pos, torch.arange(1, 7))

    def test_primer_serial_env(self):
        env0 = TransformedEnv(ContinuousActionVecMockEnv(), InitTracker())
        obs_dim = env0.observation_spec["observation"].shape[-1]
        module = self._make_module(input_size=obs_dim, max_seq_len=64)

        def make_env():
            env = TransformedEnv(ContinuousActionVecMockEnv(), InitTracker())
            env.append_transform(module.make_tensordict_primer())
            return env

        env = SerialEnv(2, make_env)
        policy = self._make_policy(env, module)
        rollout = env.rollout(5, policy)
        assert rollout[("next", "transformer_state", "k")].shape[:2] == (2, 5)

    def test_collector_round_trip(self):
        env0 = TransformedEnv(ContinuousActionVecMockEnv(), InitTracker())
        obs_dim = env0.observation_spec["observation"].shape[-1]
        module = self._make_module(input_size=obs_dim, max_seq_len=64)
        env = self._make_env(module)
        policy = self._make_policy(env, module)
        collector = Collector(env, policy, frames_per_batch=8, total_frames=16)
        try:
            for data in collector:
                assert ("next", "transformer_state", "pos") in data.keys(True)
                window = data.exclude(
                    *_STATE_KEYS, *(("next", *key) for key in _STATE_KEYS)
                )
                with set_recurrent_mode(True):
                    module(window)
                assert window["embed"].shape[: len(data.shape)] == data.shape
                break
        finally:
            collector.shutdown()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
