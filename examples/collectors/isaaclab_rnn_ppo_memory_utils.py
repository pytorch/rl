# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for the Isaac Lab recurrent PPO memory comparison."""
from __future__ import annotations

import argparse
import uuid
from typing import Literal

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
import torch.nn as nn
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    AddStateIndependentNormalScale,
    TensorDictModule,
    TensorDictSequential,
)
from torchrl._utils import logger as torchrl_logger
from torchrl.data import LazyMemmapStorage, LazyTensorStorage
from torchrl.envs import ExplorationType
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
from torchrl.modules import (
    LSTMModule,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.record import WandbLogger


RnnBackend = Literal["cudnn", "scan", "triton"]
StorageKind = Literal["cuda", "cpu", "memmap"]
DATA_FLOW_KEYS = (
    "action",
    "sample_log_prob",
    "policy",
    ("next", "reward"),
    ("next", "done"),
    ("next", "policy_version"),
    "advantage",
    "value_target",
)
INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "full-off":
        args.collector_backend = "single"
        args.storage = "cuda"
        args.compact_obs = False
        args.shifted_gae = False
        args.rnn_backend = "cudnn"
        args.compile_update = False
        args.cudagraph_update = False
    elif args.preset == "full-on":
        args.collector_backend = "single"
        args.storage = "memmap"
        args.compact_obs = True
        args.shifted_gae = True
        args.rnn_backend = "scan"
        args.compile_update = True
        args.cudagraph_update = True
    elif args.preset == "full-on-triton":
        args.collector_backend = "single"
        args.storage = "memmap"
        args.compact_obs = True
        args.shifted_gae = True
        args.rnn_backend = "triton"
        args.compile_update = True
        args.cudagraph_update = True
    return args


def _make_isaac_cfg(num_envs: int, max_episode_steps: int, device: str):
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
    return cfg


def make_env(
    task: str,
    num_envs: int,
    max_episode_steps: int,
    device: str,
) -> IsaacLabWrapper:
    if task == "Isaac-Ant-v0":
        env = gym.make(task, cfg=_make_isaac_cfg(num_envs, max_episode_steps, device))
    else:
        env = gym.make(task)
    return IsaacLabWrapper(env, device=torch.device(device))


def make_actor(
    obs_dim: int,
    action_dim: int,
    hidden_size: int,
    rnn_backend: RnnBackend,
    device: torch.device,
) -> ProbabilisticActor:
    backbone = make_backbone(obs_dim, hidden_size, rnn_backend, device)
    head = make_actor_head(action_dim, hidden_size, device)
    return make_actor_from_modules(backbone, head, action_dim, device)


def make_backbone(
    obs_dim: int,
    hidden_size: int,
    rnn_backend: RnnBackend,
    device: torch.device,
) -> TensorDictSequential:
    recurrent_backend = "pad" if rnn_backend == "cudnn" else rnn_backend
    embed = TensorDictModule(
        nn.Linear(obs_dim, hidden_size, device=device),
        in_keys=["policy"],
        out_keys=["embed"],
    )
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
        recurrent_backend=recurrent_backend,
        device=device,
    )
    return TensorDictSequential(embed, lstm)


def make_actor_head(
    action_dim: int, hidden_size: int, device: torch.device
) -> TensorDictModule:
    head = TensorDictModule(
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
    return head


def make_actor_from_modules(
    backbone: TensorDictSequential,
    head: TensorDictModule,
    action_dim: int,
    device: torch.device,
) -> ProbabilisticActor:
    module = TensorDictSequential(backbone, head)
    return ProbabilisticActor(
        module=module,
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


def make_critic_head(hidden_size: int, device: torch.device) -> ValueOperator:
    return ValueOperator(
        nn.Linear(hidden_size, 1, device=device),
        in_keys=["value_lstm_out"],
    ).to(device)


def make_full_value(
    backbone: TensorDictSequential,
    critic_head: ValueOperator,
) -> TensorDictSequential:
    cache_value_feature = TensorDictModule(
        nn.Identity(),
        in_keys=["lstm_out"],
        out_keys=["value_lstm_out"],
    )
    return TensorDictSequential(backbone, cache_value_feature, critic_head)


def make_critic(obs_dim: int, hidden_size: int, device: torch.device) -> ValueOperator:
    value_mlp = MLP(
        in_features=obs_dim,
        out_features=1,
        num_cells=[hidden_size, hidden_size],
        activation_class=nn.Tanh,
        device=device,
    )
    return ValueOperator(value_mlp, in_keys=["policy"]).to(device)


def _make_storage(
    args: argparse.Namespace,
    device: torch.device,
    max_size: int,
    ndim: int = 1,
    storage_kind: StorageKind | None = None,
    name: str | None = None,
):
    storage_kind = storage_kind or args.storage
    if storage_kind == "memmap":
        prefix = name or args.preset
        run_dir = args.memmap_dir / f"{prefix}_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return LazyMemmapStorage(
            max_size=max_size,
            scratch_dir=run_dir,
            ndim=ndim,
            shared_init=True,
            auto_cleanup=True,
        )
    if storage_kind == "cuda":
        return LazyTensorStorage(max_size=max_size, device=device, ndim=ndim)
    return LazyTensorStorage(max_size=max_size, device="cpu", ndim=ndim)


def _cuda_stats(device: torch.device, prefix: str = "cuda") -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        f"{prefix}/allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        f"{prefix}/reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        f"{prefix}/max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        f"{prefix}/max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def _all_cuda_stats(
    collector_device: torch.device, train_device: torch.device
) -> dict[str, float]:
    stats = _cuda_stats(collector_device, "collector_cuda")
    stats.update(_cuda_stats(train_device, "train_cuda"))
    return stats


def _add_metric_namespaces(payload: dict[str, float | int | str]) -> None:
    extra: dict[str, float | int | str] = {}
    for key, value in payload.items():
        if key.startswith("reward/"):
            extra[f"inference/{key}"] = value
        elif key in ("done/fraction", "terminated/fraction", "truncated/fraction"):
            extra[f"inference/{key}"] = value
        elif key.startswith("batch/"):
            extra[f"inference/{key}"] = value
        elif key.startswith("collector_cuda/"):
            extra[f"inference/cuda/{key.removeprefix('collector_cuda/')}"] = value
        elif key.startswith("train_cuda/"):
            extra[f"training/cuda/{key.removeprefix('train_cuda/')}"] = value
        elif key == "time/collect_or_sample_s":
            extra["inference/time/collect_or_sample_s"] = value
        elif key in ("time/adv_s", "time/update_s"):
            extra[f"training/{key}"] = value
        elif key.startswith("time/"):
            extra[f"training/{key}"] = value
        elif key.startswith("loss/grad_norm_"):
            grad_name = key.removeprefix("loss/grad_norm_")
            extra[f"training/grad_norm/{grad_name}"] = value
        elif key.startswith("loss/loss_"):
            loss_name = key.removeprefix("loss/loss_")
            extra[f"training/loss/{loss_name}"] = value
        elif key.startswith("loss/"):
            extra[f"training/{key.removeprefix('loss/')}"] = value
    payload.update(extra)


def _log(
    experiment_logger: WandbLogger | None,
    payload: dict[str, float | int | str],
    step: int,
) -> None:
    _add_metric_namespaces(payload)
    torchrl_logger.info(payload)
    if experiment_logger is not None:
        experiment_logger.log_metrics(
            payload,
            step=step,
            override_global_step=True,
        )


def _add_batch_stats(
    metrics: dict[str, float | int | str], batch: TensorDictBase
) -> None:
    reward = batch.get(("next", "reward"), None)
    if reward is None:
        reward = batch.get("reward", None)
    if reward is not None:
        reward = reward.float()
        metrics["reward/mean"] = float(reward.mean().detach().cpu())
        metrics["reward/std"] = float(reward.std().detach().cpu())
        metrics["reward/min"] = float(reward.min().detach().cpu())
        metrics["reward/max"] = float(reward.max().detach().cpu())
        if reward.ndim > 1:
            rollout_reward = reward.squeeze(-1) if reward.shape[-1] == 1 else reward
            metrics["reward/rollout_sum_mean"] = float(
                rollout_reward.sum(-1).mean().detach().cpu()
            )

    for name in ("done", "terminated", "truncated"):
        value = batch.get(("next", name), None)
        if value is None:
            value = batch.get(name, None)
        if value is not None:
            metrics[f"{name}/fraction"] = float(value.float().mean().detach().cpu())


def _policy_version_int(version: int | str | None) -> int | None:
    if isinstance(version, int):
        return version
    if isinstance(version, str):
        try:
            return int(version)
        except ValueError:
            return None
    return None


def _add_policy_version_metrics(
    metrics: dict[str, float | int | str],
    batch: TensorDictBase,
    collector_policy_version: int | str | None,
) -> None:
    policy_version = batch.get(("next", "policy_version"), None)
    if policy_version is None:
        policy_version = batch.get("policy_version", None)
    if not torch.is_tensor(policy_version):
        return
    policy_version = policy_version.detach().reshape(-1)
    if policy_version.numel() == 0 or policy_version.dtype not in INTEGER_DTYPES:
        return

    policy_version_min = int(policy_version.min().cpu())
    policy_version_max = int(policy_version.max().cpu())
    policy_version_mean = float(policy_version.to(torch.float64).mean().cpu())
    metrics["batch/policy_version/min"] = policy_version_min
    metrics["batch/policy_version/max"] = policy_version_max
    metrics["batch/policy_version/mean"] = policy_version_mean
    if policy_version_min == policy_version_max:
        metrics["batch/policy_version"] = policy_version_min

    collector_policy_version_int = _policy_version_int(collector_policy_version)
    if collector_policy_version_int is None:
        return
    metrics["batch/policy_staleness/min"] = (
        collector_policy_version_int - policy_version_max
    )
    metrics["batch/policy_staleness/max"] = (
        collector_policy_version_int - policy_version_min
    )
    metrics["batch/policy_staleness/mean"] = (
        collector_policy_version_int - policy_version_mean
    )
    if policy_version_min == policy_version_max:
        metrics["batch/policy_staleness"] = (
            collector_policy_version_int - policy_version_min
        )


def _batch_metrics(
    batch: TensorDictBase,
    collector_policy_version: int | str | None,
) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "batch/numel": batch.numel(),
        "batch/ndim": batch.ndim,
        "batch/device": str(batch.device),
    }
    _add_batch_stats(metrics, batch)
    _add_policy_version_metrics(metrics, batch, collector_policy_version)
    return metrics


def _assert_time_batch_shape(
    data: TensorDictBase,
    expected_batch_size: int,
    expected_seq_len: int,
    label: str,
) -> None:
    expected = (expected_batch_size, expected_seq_len)
    actual = tuple(data.batch_size)
    if actual != expected:
        raise AssertionError(
            f"{label} must have batch shape [B, T] = {expected}, got {actual}."
        )


def _make_fake_training_batch(
    *,
    batch_size: int,
    seq_len: int,
    obs_dim: int,
    action_dim: int,
    hidden_size: int,
    device: torch.device,
) -> TensorDict:
    batch_shape = (batch_size, seq_len)
    feature_shape = (*batch_shape, 1)
    recurrent_shape = (*batch_shape, 1, hidden_size)
    is_init = torch.zeros(feature_shape, dtype=torch.bool, device=device)
    is_init[:, 0] = True
    done = torch.zeros(feature_shape, dtype=torch.bool, device=device)
    return TensorDict(
        {
            "policy": torch.zeros(*batch_shape, obs_dim, device=device),
            "action": torch.zeros(*batch_shape, action_dim, device=device),
            "sample_log_prob": torch.zeros(batch_shape, device=device),
            "advantage": torch.zeros(feature_shape, device=device),
            "value_target": torch.zeros(feature_shape, device=device),
            "recurrent_state_h": torch.zeros(recurrent_shape, device=device),
            "recurrent_state_c": torch.zeros(recurrent_shape, device=device),
            "is_init": is_init,
            "done": done,
            "terminated": done.clone(),
            "truncated": done.clone(),
            "next": {
                "policy": torch.zeros(*batch_shape, obs_dim, device=device),
                "reward": torch.zeros(feature_shape, device=device),
                "done": done.clone(),
                "terminated": done.clone(),
                "truncated": done.clone(),
                "recurrent_state_h": torch.zeros(recurrent_shape, device=device),
                "recurrent_state_c": torch.zeros(recurrent_shape, device=device),
            },
        },
        batch_size=batch_shape,
        device=device,
    )


def _key_name(key) -> str:
    if isinstance(key, tuple):
        return "/".join(key)
    return key


def _add_data_flow_metrics(
    metrics: dict[str, float | int | str],
    prefix: str,
    data: TensorDictBase,
) -> None:
    metrics[f"{prefix}/numel"] = data.numel()
    metrics[f"{prefix}/ndim"] = data.ndim
    metrics[f"{prefix}/device"] = str(data.device)
    for key in DATA_FLOW_KEYS:
        if key not in data.keys(True, True):
            continue
        value = data[key]
        if not torch.is_tensor(value):
            continue
        value = value.detach()
        name = _key_name(key)
        metrics[f"{prefix}/{name}/shape"] = "x".join(str(dim) for dim in value.shape)
        if value.dtype.is_floating_point:
            finite = torch.isfinite(value)
            metrics[f"{prefix}/{name}/finite_fraction"] = float(
                finite.float().mean().cpu()
            )
            metrics[f"{prefix}/{name}/mean"] = float(value.float().mean().cpu())
            metrics[f"{prefix}/{name}/std"] = float(value.float().std().cpu())
        elif value.dtype in INTEGER_DTYPES:
            metrics[f"{prefix}/{name}/min"] = int(value.min().cpu())
            metrics[f"{prefix}/{name}/max"] = int(value.max().cpu())
            metrics[f"{prefix}/{name}/mean"] = float(
                value.to(torch.float64).mean().cpu()
            )
        elif value.dtype == torch.bool:
            metrics[f"{prefix}/{name}/true_fraction"] = float(
                value.float().mean().cpu()
            )


def _add_buffer_compare_metrics(
    metrics: dict[str, float | int | str],
    source: TensorDictBase,
    stored: TensorDictBase,
) -> None:
    for key in DATA_FLOW_KEYS:
        source_has_key = key in source.keys(True, True)
        stored_has_key = key in stored.keys(True, True)
        if not source_has_key or not stored_has_key:
            continue
        source_value = source[key]
        stored_value = stored[key]
        if not torch.is_tensor(source_value) or not torch.is_tensor(stored_value):
            continue
        if source_value.shape != stored_value.shape:
            metrics[f"data_flow/train_buffer/{_key_name(key)}/shape_match"] = 0.0
            continue
        name = _key_name(key)
        metrics[f"data_flow/train_buffer/{name}/shape_match"] = 1.0
        if source_value.dtype == torch.bool:
            neq = source_value.detach().cpu() != stored_value.detach().cpu()
            metrics[f"data_flow/train_buffer/{name}/neq_fraction"] = float(
                neq.float().mean()
            )
        elif source_value.dtype.is_floating_point:
            diff = (source_value.detach().cpu() - stored_value.detach().cpu()).abs()
            metrics[f"data_flow/train_buffer/{name}/max_abs_diff"] = float(diff.max())
            metrics[f"data_flow/train_buffer/{name}/mean_abs_diff"] = float(diff.mean())


def _log_iteration_metrics(
    *,
    experiment_logger: WandbLogger | None,
    step: int,
    iteration: int,
    args: argparse.Namespace,
    batch_metrics: dict[str, float | int | str],
    collector_device: torch.device,
    train_device: torch.device,
    frames_per_batch: int,
    updates_per_epoch: int,
    sample_num_slices: int,
    sample_seq_len: int,
    timing_metrics: dict[str, float],
    loss_acc: TensorDictBase | None,
    loss_count: int,
    collector_generation: int | None = None,
    collector_policy_version: int | str | None = None,
    collector_frames: int | None = None,
    collector_direct_replay_buffer: bool = False,
) -> None:
    metrics: dict[str, float | int | str] = {
        "iteration": iteration,
        "phase": "train_warmup" if iteration < args.warmup_iterations else "train",
        "updates/per_epoch": updates_per_epoch,
        "updates/total": updates_per_epoch * args.ppo_epochs,
        "training/minibatch_frames": args.mini_batch_steps,
        "training/rnn_batch_size": sample_num_slices,
        "training/rnn_seq_len": sample_seq_len,
        "training/reshape_sampled_slices": float(args.reshape_sampled_slices),
        "training/rollout_frames": frames_per_batch,
        "training/target_ppo_epochs": args.ppo_epochs,
        "training/effective_epochs": (
            updates_per_epoch * args.ppo_epochs * args.mini_batch_steps
        )
        / frames_per_batch,
    }
    metrics.update(batch_metrics)
    metrics.update(timing_metrics)
    update_s = timing_metrics.get("time/update_s", 0.0)
    if loss_acc is not None and loss_count != 0:
        loss_mean = loss_acc.apply(
            lambda x: (x / loss_count).float().mean(), batch_size=[]
        )
        for key, value in loss_mean.items():
            metrics[f"loss/{key}"] = float(value.detach().cpu())

    if collector_generation is not None:
        metrics["collector/generation"] = collector_generation
    if collector_policy_version is not None:
        metrics["collector/policy_version"] = collector_policy_version
    if collector_frames is not None:
        metrics["collector/frames"] = collector_frames
    if collector_direct_replay_buffer:
        metrics["collector/direct_replay_buffer"] = 1.0

    metrics.update(_all_cuda_stats(collector_device, train_device))
    iter_s = timing_metrics.get("time/iter_s", 0.0)
    collect_or_sample_s = timing_metrics.get("time/collect_or_sample_s", 0.0)
    batch_frames = metrics.get("batch/numel", frames_per_batch)
    if not isinstance(batch_frames, int):
        batch_frames = frames_per_batch

    metrics["training/optimizer_steps"] = loss_count
    metrics["training/optimizer_steps_per_second"] = (
        loss_count / update_s if update_s > 0 else 0.0
    )
    metrics["training/rollout_frames_per_second_wall"] = (
        frames_per_batch / iter_s if iter_s > 0 else 0.0
    )
    metrics["collector/frames_per_second_wall"] = (
        batch_frames / iter_s if iter_s > 0 else 0.0
    )
    metrics["collector/frames_per_second_blocking"] = (
        batch_frames / collect_or_sample_s if collect_or_sample_s > 0 else 0.0
    )
    metrics["training/sample_frames_per_second_update"] = (
        loss_count * args.mini_batch_steps / update_s if update_s > 0 else 0.0
    )
    if iteration >= args.warmup_iterations:
        metrics["frames"] = (iteration - args.warmup_iterations + 1) * frames_per_batch
    _log(experiment_logger, metrics, step)
