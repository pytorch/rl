# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from torchrl.envs import TransformedEnv
from torchrl.envs.libs.mjlab import MJLabEnv, MJLabWrapper
from torchrl.envs.transforms import InitTracker
from torchrl.envs.utils import check_env_specs

_has_mjlab = importlib.util.find_spec("mjlab") is not None
_CARTPOLE_TASK = "Mjlab-Cartpole-Balance"

requires_mjlab = pytest.mark.skipif(not _has_mjlab, reason="mjlab not installed")
requires_mjlab_cuda = pytest.mark.skipif(
    not _has_mjlab or not torch.cuda.is_available(),
    reason="mjlab integration tests require mjlab and CUDA",
)


def _load_cartpole_cfg(
    num_envs: int,
    *,
    episode_length_steps: int | None = None,
    camera_sensor: bool = False,
):
    pytest.importorskip("mjlab")
    try:
        import mjlab.tasks  # noqa: F401
        from mjlab.sensor import CameraSensorCfg
        from mjlab.tasks.registry import load_env_cfg
    except ImportError as err:
        pytest.skip(f"mjlab task registry dependencies unavailable: {err}")

    cfg = load_env_cfg(_CARTPOLE_TASK)
    cfg.scene.num_envs = num_envs
    if camera_sensor:
        cfg.scene.sensors = (
            *cfg.scene.sensors,
            CameraSensorCfg(
                name="pixels",
                camera_name="cartpole/fixed",
                width=32,
                height=24,
                data_types=("rgb",),
            ),
        )
    if episode_length_steps is not None:
        cfg.episode_length_s = (
            cfg.sim.mujoco.timestep * cfg.decimation * episode_length_steps
        )
    return cfg


def _make_raw_cartpole_env(
    *,
    num_envs: int,
    device: str = "cuda:0",
    render_mode: str | None = None,
    episode_length_steps: int | None = None,
    camera_sensor: bool = False,
):
    pytest.importorskip("mjlab")
    try:
        from mjlab.envs import ManagerBasedRlEnv
    except ImportError as err:
        pytest.skip(f"mjlab environment dependencies unavailable: {err}")

    cfg = _load_cartpole_cfg(
        num_envs=num_envs,
        episode_length_steps=episode_length_steps,
        camera_sensor=camera_sensor,
    )
    return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)


@contextmanager
def _cartpole_env(
    *,
    num_envs: int,
    device: str = "cuda:0",
    episode_length_steps: int | None = None,
    camera_sensor: bool = False,
    **kwargs: Any,
) -> Iterator[MJLabEnv]:
    cfg = _load_cartpole_cfg(
        num_envs=num_envs,
        episode_length_steps=episode_length_steps,
        camera_sensor=camera_sensor,
    )
    env = MJLabEnv(_CARTPOLE_TASK, cfg=cfg, device=device, **kwargs)
    try:
        yield env
    finally:
        env.close()


def _raw_mjlab_env(env: Any):
    while isinstance(env, TransformedEnv):
        env = env.base_env
    return env._env


def _force_single_timeout_next_step(env: Any, env_id: int) -> None:
    raw_env = _raw_mjlab_env(env)
    raw_env.episode_length_buf.zero_()
    raw_env.episode_length_buf[env_id] = raw_env.max_episode_length - 1


@requires_mjlab
def test_batch_size_and_pixels_validation_use_mjlab_task_cfg():
    cfg = _load_cartpole_cfg(num_envs=3)
    with pytest.raises(ValueError, match="one-dimensional"):
        MJLabEnv(_CARTPOLE_TASK, cfg=cfg, batch_size=[3, 1])
    with pytest.raises(ValueError, match="does not match"):
        MJLabEnv(_CARTPOLE_TASK, cfg=cfg, num_envs=3, batch_size=[2])
    with pytest.raises(ValueError, match="CameraSensor"):
        MJLabEnv(_CARTPOLE_TASK, cfg=cfg, num_envs=3, from_pixels=True)


@pytest.mark.gpu
@requires_mjlab_cuda
def test_wrapper_specs_rollout_and_device_with_real_mjlab_cartpole():
    raw_env = _make_raw_cartpole_env(num_envs=2)
    env = None
    try:
        with pytest.raises(ValueError, match="does not match"):
            MJLabWrapper(raw_env, device="cpu")
        with pytest.raises(ValueError, match="does not match"):
            MJLabWrapper(raw_env, batch_size=[1])
        with pytest.raises(ValueError, match="one-dimensional"):
            MJLabWrapper(raw_env, batch_size=[2, 1])
        env = MJLabWrapper(raw_env)
        raw_env = None
        assert env.batch_size == torch.Size([2])
        assert env.device == torch.device("cuda:0")
        assert env.action_spec.device == torch.device("cuda:0")
        assert env.observation_spec.device == torch.device("cuda:0")
        assert env._env.cfg.auto_reset is False
        check_env_specs(env)
        rollout = env.rollout(3)
        assert rollout.batch_size == torch.Size([2, 3])
        assert rollout["actor"].shape == torch.Size([2, 3, 5])
        assert rollout[("next", "reward")].shape == torch.Size([2, 3, 1])
        env.set_seed(123, static_seed=True)
        assert env._env.cfg.seed == 123
    finally:
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()


@pytest.mark.gpu
@requires_mjlab_cuda
def test_torchrl_driven_autoreset_resets_only_done_rows_with_real_mjlab():
    with _cartpole_env(num_envs=3, episode_length_steps=2) as base_env:
        env = TransformedEnv(base_env, InitTracker())
        td = env.reset()
        _force_single_timeout_next_step(env, env_id=1)
        td.set("action", env.action_spec.zero())
        step_td, next_td = env.step_and_maybe_reset(td)
        done = step_td[("next", "done")].squeeze(-1)
        assert done.tolist() == [False, True, False]
        assert not _raw_mjlab_env(env)._manual_reset_pending.any()
        assert _raw_mjlab_env(env).episode_length_buf.tolist() == [1, 0, 1]
        assert next_td["is_init"].squeeze(-1).tolist() == [False, True, False]
        assert not next_td["done"].any()
        next_td.set("action", env.action_spec.zero())
        env.step(next_td)


@pytest.mark.gpu
@requires_mjlab_cuda
def test_native_autoreset_invalidates_terminal_obs_with_real_mjlab():
    with _cartpole_env(
        num_envs=3, episode_length_steps=2, native_autoreset=True
    ) as env:
        assert env._env.cfg.auto_reset is True
        td = env.reset()
        _force_single_timeout_next_step(env, env_id=1)
        td.set("action", env.action_spec.zero())
        step_td, next_td = env.step_and_maybe_reset(td)
        done = step_td[("next", "done")].squeeze(-1)
        assert done.tolist() == [False, True, False]
        assert not env._env._manual_reset_pending.any()
        assert torch.isnan(step_td[("next", "actor")][done]).all()
        assert torch.isnan(step_td[("next", "critic")][done]).all()
        assert torch.isfinite(next_td["actor"][done]).all()
        assert torch.isfinite(next_td["critic"][done]).all()
        assert not next_td["done"].any()


@pytest.mark.gpu
@requires_mjlab_cuda
def test_batched_camera_sensor_pixels_with_real_mjlab_cartpole():
    with _cartpole_env(
        num_envs=3,
        camera_sensor=True,
        from_pixels=True,
        pixels_only=True,
    ) as env:
        assert env._env.render_mode is None
        td = env.reset()
        assert set(td.keys()) == {"pixels", "done", "terminated", "truncated"}
        pixels = td["pixels"]
        assert pixels.shape == torch.Size([3, 24, 32, 3])
        assert pixels.dtype == torch.uint8
        td.set("action", env.action_spec.zero())
        step_td = env.step(td)
        next_pixels = step_td[("next", "pixels")]
        assert next_pixels.shape == pixels.shape
        assert next_pixels.dtype == torch.uint8
        assert torch.isfinite(next_pixels.float()).all()


@pytest.mark.gpu
@requires_mjlab_cuda
def test_single_env_render_pixels_fallback_with_real_mjlab_cartpole():
    with _cartpole_env(
        num_envs=1,
        from_pixels=True,
        pixels_only=True,
    ) as env:
        assert env._env.render_mode == "rgb_array"
        td = env.reset()
        assert set(td.keys()) == {"pixels", "done", "terminated", "truncated"}
        pixels = td["pixels"]
        assert pixels.shape[0] == 1
        assert pixels.ndim == 4
        assert pixels.shape[-1] == 3
        assert pixels.dtype == torch.uint8
        frame = torch.as_tensor(env.render())
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
        assert frame.dtype == torch.uint8


if __name__ == "__main__":
    pytest.main([__file__])
