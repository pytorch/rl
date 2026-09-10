# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Hot-path benchmarks for :class:`~torchrl.modules.TransformerModule`.

Three measurements: the cached single-step path used during collection, the
windowed path used during training, and a rollout through an environment
whose returned storage must not grow with the cache size, since the key/value
cache lives in the module and never in the tensordict.
"""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.envs import InitTracker, TransformedEnv
from torchrl.modules import set_recurrent_mode, TransformerModule
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv

_DEVICE = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
_STEPS_PER_CALL = 16


def _make_module(input_size: int, max_seq_len: int) -> TransformerModule:
    return TransformerModule(
        input_size=input_size,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        max_seq_len=max_seq_len,
        in_key="observation",
        out_key="embed",
        device=_DEVICE,
    )


def _run_cached_steps(module: TransformerModule, obs: torch.Tensor) -> None:
    """Step ``_STEPS_PER_CALL`` times from an episode start on every stream."""
    num_envs = obs.shape[0]
    is_init = torch.ones(num_envs, 1, dtype=torch.bool, device=obs.device)
    for step in range(_STEPS_PER_CALL):
        td = TensorDict({"observation": obs[:, step], "is_init": is_init}, [num_envs])
        module(td)
        is_init = torch.zeros_like(is_init)


def _tensordict_bytes(td: TensorDict) -> int:
    return sum(value.numel() * value.element_size() for value in td.values(True, True))


@pytest.mark.parametrize("num_envs", [8, 32])
@pytest.mark.parametrize("context", [64, 256])
def test_transformer_cached_step(benchmark, num_envs: int, context: int) -> None:
    module = _make_module(16, context)
    obs = torch.randn(num_envs, _STEPS_PER_CALL, 16, device=_DEVICE)
    with torch.no_grad():
        _run_cached_steps(module, obs)
        benchmark(_run_cached_steps, module, obs)


@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("window", [32, 128])
def test_transformer_window(benchmark, batch_size: int, window: int) -> None:
    module = _make_module(16, window)
    is_init = torch.zeros(batch_size, window, 1, dtype=torch.bool, device=_DEVICE)
    is_init[:, 0] = True
    is_init[:, window // 2] = True
    td = TensorDict(
        {
            "observation": torch.randn(batch_size, window, 16, device=_DEVICE),
            "is_init": is_init,
        },
        [batch_size, window],
    )

    def run():
        with set_recurrent_mode(True):
            module(td.clone())

    run()
    benchmark(run)


@pytest.mark.parametrize("context", [64, 512])
def test_transformer_rollout_storage_is_cache_free(benchmark, context: int) -> None:
    """Rollout storage depends on the rollout length, never on the cache size."""
    env = TransformedEnv(ContinuousActionVecMockEnv(device=_DEVICE), InitTracker())
    obs_dim = env.observation_spec["observation"].shape[-1]
    module = _make_module(obs_dim, context)
    policy = TensorDictSequential(
        module,
        TensorDictModule(
            nn.Linear(256, env.action_spec.shape[-1], device=_DEVICE),
            in_keys=["embed"],
            out_keys=["action"],
        ),
    )
    with torch.no_grad():
        short = env.rollout(16, policy)
        module.reset_cache()
        rollout = benchmark(env.rollout, 32, policy)
    assert "transformer_state" not in rollout.keys()
    short_bytes = _tensordict_bytes(short)
    long_bytes = _tensordict_bytes(rollout)
    assert long_bytes == 2 * short_bytes
    one_stream_cache = module.transformer.new_kv_cache(1)
    cache_bytes = sum(c.numel() * c.element_size() for c in one_stream_cache)
    assert long_bytes < cache_bytes


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only"])
