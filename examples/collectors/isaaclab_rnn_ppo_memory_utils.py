# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for the Isaac Lab recurrent PPO example.

Isaac Lab is only imported lazily inside :func:`make_env`, so the
top-level main script can run without ``isaaclab`` on the Python path
and only the worker subprocesses (where ``make_env`` is called) pay the
import cost.
"""
from __future__ import annotations

import argparse
import os
from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import (
    AddStateIndependentNormalScale,
    TensorDictModule,
    TensorDictSequential,
)
from torchrl._utils import logger as torchrl_logger
from torchrl.envs import (
    ExplorationType,
    RandomTruncationTransform,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.transforms import Compose
from torchrl.modules import (
    LSTMModule,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.record import WandbLogger

RnnBackend = Literal["cudnn", "pad", "scan", "triton"]


_RECURRENT_STATE_KEYS = {
    "recurrent_state_h",
    "recurrent_state_c",
    "('next', 'recurrent_state_h')",
    "('next', 'recurrent_state_c')",
}


def _leaf_shape_summary(tensordict: TensorDictBase) -> dict[str, dict[str, str]]:
    return {
        str(key): {
            "shape": str(tuple(value.shape)),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        for key, value in tensordict.items(include_nested=True, leaves_only=True)
        if hasattr(value, "shape")
    }


def _metric_float(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            value = value.float().mean()
        return float(value.cpu())
    return float(value)


def _tensor_stats(prefix: str, value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float()
    return {
        f"{prefix}/mean": _metric_float(value.mean()),
        f"{prefix}/std": _metric_float(value.std(unbiased=False)),
        f"{prefix}/min": _metric_float(value.min()),
        f"{prefix}/max": _metric_float(value.max()),
    }


def _loss_metrics(loss_acc: TensorDictBase, loss_count: int) -> dict[str, float]:
    metrics = {}
    for key, value in loss_acc.items():
        value = value / loss_count
        key = str(key)
        if key.startswith("loss_"):
            key = f"loss/{key.removeprefix('loss_')}"
        elif key.startswith("grad_norm"):
            key = key.replace("grad_norm", "grad_norm/")
        metrics[f"training/{key}"] = _metric_float(value)
    return metrics


def _inference_metrics(
    data: TensorDictBase,
    *,
    frames: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "inference/frames": frames,
        "inference/batch_numel": data.numel(),
        "inference/batch_ndim": data.ndim,
    }
    reward = data.get(("next", "reward"), default=None)
    if reward is not None:
        metrics.update(_tensor_stats("inference/reward", reward))
    episode_reward = data.get(("next", "episode_reward"), default=None)
    done = data.get(("next", "done"), default=None)
    if episode_reward is not None and done is not None:
        episode_reward = episode_reward.squeeze(-1)
        done = done.squeeze(-1).to(torch.bool)
        end_of_traj_reward = episode_reward[done]
        if end_of_traj_reward.numel():
            metrics.update(
                _tensor_stats(
                    "inference/end_of_traj_episode_reward", end_of_traj_reward
                )
            )
    return metrics


def _rendered_eval_metrics(data: TensorDictBase) -> dict[str, torch.Tensor]:
    mask = data.get(("collector", "mask"))[0]
    if mask.ndim > 1:
        mask = mask.squeeze(-1)
    pixels = data[0].get(("next", "pixels"))[mask.to(torch.bool)]
    pixels = pixels[..., :3].permute(0, 3, 1, 2)
    return {"video": pixels.to(torch.uint8).unsqueeze(0).cpu()}


def _log_eval_result(
    result: dict[str, object],
    *,
    experiment_logger: WandbLogger | None,
) -> None:
    if not result:
        return
    result = dict(result)
    step = result.pop("eval/step", None)
    video = result.pop("eval/video", None)
    if experiment_logger is None:
        torchrl_logger.info({"phase": "eval_done", **result})
        return
    if result:
        experiment_logger.log_metrics(result, step=step)
    if video is not None:
        experiment_logger.log_video("eval/video", video, step=step)


def _cuda_metrics(prefix: str, device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        f"telemetry/{prefix}/allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        f"telemetry/{prefix}/reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        f"telemetry/{prefix}/max_allocated_gb": torch.cuda.max_memory_allocated(device)
        / 1e9,
        f"telemetry/{prefix}/max_reserved_gb": torch.cuda.max_memory_reserved(device)
        / 1e9,
    }


def _assert_rollout_shapes(
    tensordict: TensorDictBase,
    *,
    expected_shape: torch.Size,
    hidden_size: int,
    phase: str,
) -> None:
    if tensordict.shape != expected_shape:
        raise RuntimeError(
            f"{phase}: expected TensorDict shape {expected_shape}, "
            f"got {tensordict.shape}."
        )
    expected_state_shape = (*expected_shape, 1, hidden_size)
    for key, value in tensordict.items(include_nested=True, leaves_only=True):
        if not hasattr(value, "shape"):
            continue
        if value.shape[: len(expected_shape)] != expected_shape:
            raise RuntimeError(
                f"{phase}: key {key} has shape {tuple(value.shape)}, "
                f"which does not start with {tuple(expected_shape)}."
            )
        if str(key) in _RECURRENT_STATE_KEYS and tuple(value.shape) != tuple(
            expected_state_shape
        ):
            raise RuntimeError(
                f"{phase}: key {key} has recurrent-state shape "
                f"{tuple(value.shape)}, expected {tuple(expected_state_shape)}."
            )


def _normalize_rollout_batch(
    tensordict: TensorDictBase, expected_shape: torch.Size
) -> TensorDictBase:
    if tensordict.shape == expected_shape:
        return tensordict
    if tensordict.shape == torch.Size((1, *expected_shape)):
        return tensordict.squeeze(0)
    if tensordict.ndim < 2 or tensordict.shape[-1] != expected_shape[-1]:
        raise RuntimeError(
            f"Expected collected batch ending in time shape {tuple(expected_shape)}, "
            f"got {tuple(tensordict.shape)}."
        )
    if tensordict.shape[:-1].numel() != expected_shape[0]:
        raise RuntimeError(
            f"Expected collected batch with {expected_shape[0]} env elements before "
            f"time, got shape {tuple(tensordict.shape)}."
        )
    return tensordict.reshape(expected_shape)


def _init_isaac_app(
    device: str | None = None,
    *,
    enable_cameras: bool = False,
    rendering_mode: Literal["performance", "balanced", "quality"] | None = None,
    cuda_visible_devices: str | None = None,
    nvidia_lib_dir: str | None = None,
    vulkan_icd: str | None = None,
    xdg_runtime_dir: str | None = None,
) -> None:
    """Start Isaac Lab's AppLauncher in headless mode inside a worker."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if nvidia_lib_dir is not None:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{nvidia_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        )
    if vulkan_icd is not None:
        os.environ["VK_ICD_FILENAMES"] = vulkan_icd
    if xdg_runtime_dir is not None:
        os.environ["XDG_RUNTIME_DIR"] = xdg_runtime_dir
        os.makedirs(xdg_runtime_dir, mode=0o700, exist_ok=True)
        os.chmod(xdg_runtime_dir, 0o700)

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="TorchRL Isaac Lab env launcher.")
    AppLauncher.add_app_launcher_args(parser)
    launch_args = ["--headless"]
    if enable_cameras:
        launch_args.append("--enable_cameras")
        if rendering_mode is not None:
            launch_args.extend(["--rendering_mode", rendering_mode])
    if device is not None:
        launch_args.extend(["--device", device])
    args_cli, _ = parser.parse_known_args(launch_args)
    AppLauncher(args_cli)


def make_env(
    task: str,
    num_envs: int,
    max_episode_steps: int,
    device: str,
    *,
    random_init_steps: int = 0,
    random_init_random: bool = True,
    render: bool = False,
    render_backend: Literal["isaac_rtx", "newton_warp", "ovrtx"] | None = None,
    compile_env: bool | dict | None = False,
):
    """Build an Isaac Lab env. Imports ``isaaclab`` lazily (worker-only).

    The ``compile_env`` argument forwards to the ``compile=...`` constructor
    kwarg on :class:`~torchrl.envs.TransformedEnv`, which compiles the env's
    ``step_and_maybe_reset`` path with :func:`torch.compile`. Pass ``True``
    for default options or a ``dict`` of :func:`torch.compile` kwargs.
    """
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    if task == "Isaac-Ant-v0":
        cfg = AntEnvCfg()
        if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
            cfg.scene.num_envs = num_envs
        if hasattr(cfg, "sim") and hasattr(cfg.sim, "device"):
            cfg.sim.device = device
        if hasattr(cfg, "device"):
            cfg.device = device
        if (
            hasattr(cfg, "episode_length_s")
            and hasattr(cfg, "sim")
            and hasattr(cfg.sim, "dt")
        ):
            cfg.episode_length_s = max_episode_steps * cfg.sim.dt
        if render:
            IsaacLabWrapper.add_tiled_camera_config(
                cfg,
                renderer_backend=render_backend,
                width=320,
                height=240,
                pos=(-7.0, 0.0, 3.0),
                rot=(0.9945, 0.0, 0.1045, 0.0),
                render_interval=cfg.sim.render_interval,
            )
        env = gym.make(task, cfg=cfg)
    else:
        env = gym.make(task)
    transforms = [RewardSum()]
    if random_init_steps:
        transforms.extend(
            [
                StepCounter(max_steps=max_episode_steps),
                RandomTruncationTransform(
                    min_horizon=max(1, max_episode_steps - random_init_steps),
                    max_horizon=max_episode_steps,
                    prob=float(random_init_random),
                    first_episode_prob=float(random_init_random),
                ),
            ]
        )
    transformed_env_kwargs = {}
    if compile_env:
        transformed_env_kwargs["compile"] = compile_env
    return TransformedEnv(
        IsaacLabWrapper(
            env,
            device=torch.device(device),
            native_autoreset=bool(random_init_steps),
            from_tiled_camera=render,
        ),
        Compose(*transforms),
        **transformed_env_kwargs,
    )


def make_models(
    *,
    obs_dim: int,
    action_dim: int,
    hidden_size: int,
    rnn_backend: RnnBackend,
    device: torch.device,
) -> tuple[ProbabilisticActor, ValueOperator, TensorDictSequential]:
    """Build the actor, critic head, and the full value module sharing the backbone.

    The actor is a :class:`~torchrl.modules.ProbabilisticActor` wrapping a
    shared embed + LSTM backbone and a per-action Gaussian head. The value
    module reuses the same backbone (so a single GAE pass amortises the
    recurrent compute across the policy and the value head).
    """
    embed = TensorDictModule(
        nn.Linear(obs_dim, hidden_size, device=device),
        in_keys=["policy"],
        out_keys=["embed"],
    )
    lstm_backend = "pad" if rnn_backend == "cudnn" else rnn_backend
    lstm = LSTMModule(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        in_keys=["embed", "recurrent_state_h", "recurrent_state_c"],
        out_keys=[
            "lstm_out",
            ("next", "recurrent_state_h"),
            ("next", "recurrent_state_c"),
        ],
        recurrent_backend=lstm_backend,
        device=device,
    )
    backbone = TensorDictSequential(embed, lstm)

    actor_head = TensorDictModule(
        nn.Sequential(
            MLP(
                in_features=hidden_size,
                out_features=action_dim,
                num_cells=[],
                activation_class=nn.Tanh,
                device=device,
            ),
            AddStateIndependentNormalScale(action_dim, scale_lb=1e-4).to(device),
        ),
        in_keys=["lstm_out"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=TensorDictSequential(backbone, actor_head),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -torch.ones(action_dim, device=device),
            "high": torch.ones(action_dim, device=device),
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # The value module reuses the backbone (shared params, single LSTM call
    # per GAE/update pass). An identity TDM caches the lstm_out under a
    # distinct key so the critic head and the actor head don't fight over
    # write semantics on a shared key.
    value_feature = TensorDictModule(
        nn.Identity(), in_keys=["lstm_out"], out_keys=["value_lstm_out"]
    )
    critic = ValueOperator(
        nn.Linear(hidden_size, 1, device=device),
        in_keys=["value_lstm_out"],
    ).to(device)
    full_value = TensorDictSequential(backbone, value_feature, critic)
    return actor, critic, full_value
