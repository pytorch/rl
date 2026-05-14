# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Throwaway Isaac Lab recurrent PPO memory comparison.

This is intentionally an experiment harness, not a polished example.  It runs
large-vectorized Isaac Lab PPO with an LSTM policy and records the memory gap
between an unoptimized path and the current memory-saving path.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import uuid
from functools import partial
from pathlib import Path
from typing import Literal

os.environ.setdefault("TORCHRL_PROFILING", "1")

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isaac Lab LSTM PPO memory comparison."
    )
    parser.add_argument(
        "--preset",
        choices=["custom", "full-off", "full-on", "full-on-triton"],
        default="custom",
    )
    parser.add_argument("--task", default="Isaac-Ant-v0")
    parser.add_argument("--num-envs", type=int, default=16_384)
    parser.add_argument("--num-collectors", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--mini-batch-steps", type=int, default=8_192)
    parser.add_argument(
        "--reshape-sampled-slices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deprecated no-op kept for old launch scripts. Minibatches are "
            "sampled as [B, T] trajectory batches."
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--obs-dim", type=int, default=60)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--critic-coeff", type=float, default=1.0)
    parser.add_argument(
        "--collector-backend",
        choices=["single", "sync", "async"],
        default="async",
    )
    parser.add_argument("--storage", choices=["cuda", "cpu", "memmap"], default="cuda")
    parser.add_argument(
        "--collector-buffer-storage",
        choices=["cpu", "memmap"],
        default="memmap",
        help="Storage used by the double-buffered collector staging buffers.",
    )
    parser.add_argument("--double-buffer-collector", action="store_true")
    parser.add_argument(
        "--storing-device",
        default=None,
        help=(
            "Collector output device. Use 'cpu' to move rollout data off the "
            "collector GPU before yielding it; leave unset for collector default."
        ),
    )
    parser.add_argument(
        "--compact-obs", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--shifted-gae", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--vectorized-gae", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deactivate-gae-vmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use GAE's Python fallback for value-network calls. Recurrent "
            "value networks contain reset-dependent control flow that is not "
            "compatible with torch.vmap."
        ),
    )
    parser.add_argument(
        "--rnn-backend",
        choices=["cudnn", "scan", "triton"],
        default="cudnn",
    )
    parser.add_argument(
        "--compile-update", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument(
        "--cudagraph-update", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--cudagraph-warmup", type=int, default=8)
    parser.add_argument(
        "--cudagraph-trees", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--empty-cache-after-warmup", action="store_true")
    parser.add_argument("--train-device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--memmap-dir", type=Path, default=Path("/tmp/torchrl_isaac_rnn_train_buffer")
    )
    parser.add_argument("--wandb-project", default="torchrl-isaac-rnn-memory")
    parser.add_argument("--wandb-group", default="full-toggle-comparison")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--debug-data-flow", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


launch_args = sys.argv[1:]
if not any(arg == "--headless" or arg.startswith("--headless=") for arg in launch_args):
    launch_args = [*launch_args, "--headless"]
args_cli, _ = _parser().parse_known_args(launch_args)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _restore_local_editable_paths() -> None:
    for module_name in ("tensordict", "torchrl"):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "__file__", None) is None:
            del sys.modules[module_name]

    for path in ("/root/tensordict", "/root/td", "/root/rl-isaac"):
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)


_restore_local_editable_paths()

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
import torch._dynamo
import torch._inductor.config
import torch.nn as nn
import torch.optim
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    AddStateIndependentNormalScale,
    CudaGraphModule,
    TensorDictModule,
    TensorDictSequential,
)
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.collectors import Collector, MultiCollector
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
from torchrl.modules import (
    LSTMModule,
    MLP,
    ProbabilisticActor,
    set_recurrent_mode,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record import WandbLogger
from torchrl.weight_update import MultiProcessWeightSyncScheme


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


def _batch_metrics(batch: TensorDictBase) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "batch/numel": batch.numel(),
        "batch/ndim": batch.ndim,
        "batch/device": str(batch.device),
    }
    _add_batch_stats(metrics, batch)
    return metrics


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
    if collector_direct_replay_buffer:
        metrics["collector/direct_replay_buffer"] = 1.0

    metrics.update(_all_cuda_stats(collector_device, train_device))
    iter_s = timing_metrics.get("time/iter_s", 0.0)

    metrics["training/optimizer_steps"] = loss_count
    metrics["training/optimizer_steps_per_second"] = (
        loss_count / update_s if update_s > 0 else 0.0
    )
    metrics["training/rollout_frames_per_second_wall"] = (
        frames_per_batch / iter_s if iter_s > 0 else 0.0
    )
    metrics["training/sample_frames_per_second_update"] = (
        loss_count * args.mini_batch_steps / update_s if update_s > 0 else 0.0
    )
    if iteration >= args.warmup_iterations:
        metrics["frames"] = (iteration - args.warmup_iterations + 1) * frames_per_batch
    _log(experiment_logger, metrics, step)


def main() -> None:
    args = _apply_preset(args_cli)
    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.compact_obs and not args.shifted_gae:
        raise ValueError("--compact-obs requires --shifted-gae for this script.")
    if args.num_envs % args.num_collectors:
        raise ValueError("--num-envs must be divisible by --num-collectors.")
    if args.compile_update:
        torch._dynamo.config.capture_scalar_outputs = True
    if args.compile_update and args.cudagraph_update:
        torch._inductor.config.triton.cudagraph_trees = args.cudagraph_trees
    if args.cudagraph_update:
        # The first cudagraph_warmup + 1 calls are pre-capture (uncompiled
        # or compile-only). Make sure the "warmup" window absorbs all of
        # those so the first reported real iteration is post-capture.
        min_warmup = args.cudagraph_warmup + 1
        if args.warmup_iterations < min_warmup:
            torchrl_logger.info(
                {
                    "phase": "warmup_iterations_autobump",
                    "from": args.warmup_iterations,
                    "to": min_warmup,
                }
            )
            args.warmup_iterations = min_warmup

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    collector_device = torch.device(args.device)
    train_device = torch.device(args.train_device)
    storing_device = (
        torch.device(args.storing_device) if args.storing_device is not None else None
    )
    if train_device.type == "cuda":
        torch.cuda.set_device(train_device)
        torch.cuda.manual_seed_all(args.seed)

    per_collector_envs = args.num_envs // args.num_collectors
    frames_per_batch = args.num_envs * args.rollout_steps

    backbone = make_backbone(
        args.obs_dim,
        args.hidden_size,
        args.rnn_backend,
        train_device,
    )
    actor_head = make_actor_head(args.action_dim, args.hidden_size, train_device)
    actor = make_actor_from_modules(
        backbone,
        actor_head,
        args.action_dim,
        train_device,
    )
    critic = make_critic_head(args.hidden_size, train_device)
    full_value = make_full_value(backbone, critic)

    optim = group_optimizers(
        torch.optim.Adam(
            actor.parameters(),
            lr=args.lr,
            eps=1e-5,
            capturable=args.cudagraph_update,
        ),
        torch.optim.Adam(
            critic.parameters(),
            lr=args.lr,
            eps=1e-5,
            capturable=args.cudagraph_update,
        ),
    )
    actor_params = tuple(actor.parameters())
    critic_params = tuple(critic.parameters())
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=full_value,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        critic_coeff=args.critic_coeff,
        normalize_advantage=True,
    )
    adv_module = GAE(
        gamma=args.gamma,
        lmbda=args.gae_lambda,
        value_network=full_value,
        average_gae=False,
        shifted=args.shifted_gae,
        vectorized=args.vectorized_gae,
        deactivate_vmap=args.deactivate_gae_vmap,
        device=train_device,
    )

    def compute_loss(batch: TensorDictBase):
        with set_recurrent_mode(True):
            return loss_module(batch)

    def update(batch: TensorDictBase):
        loss = compute_loss(batch)
        total_loss = loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
        total_loss_value = total_loss.detach()
        total_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_params, max_norm=float("inf")
        )
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_params, max_norm=float("inf")
        )
        grad_norm = (actor_grad_norm.square() + critic_grad_norm.square()).sqrt()
        optim.step()
        optim.zero_grad(set_to_none=True)
        loss_out = loss.detach()
        loss_out.set("loss_total", total_loss_value)
        loss_out.set("grad_norm_actor", actor_grad_norm.detach())
        loss_out.set("grad_norm_critic", critic_grad_norm.detach())
        loss_out.set("grad_norm_total", grad_norm.detach())
        return loss_out

    if args.compile_update:
        update = compile_with_warmup(update, mode=args.compile_mode, warmup=1)

    if args.cudagraph_update:
        update = CudaGraphModule(
            update, in_keys=[], out_keys=[], warmup=args.cudagraph_warmup
        )

    sample_num_slices = max(1, args.mini_batch_steps // args.rollout_steps)
    sample_seq_len = args.rollout_steps
    train_buffer = ReplayBuffer(
        storage=_make_storage(args, train_device, args.num_envs, ndim=1),
        sampler=SamplerWithoutReplacement(
            drop_last=True,
        ),
        batch_size=sample_num_slices,
    )
    updates_per_epoch = math.ceil(args.num_envs / sample_num_slices)
    collector_replay_buffer = None
    if args.double_buffer_collector:
        collector_replay_buffer = ReplayBuffer(
            storage=_make_storage(
                args,
                torch.device("cpu"),
                args.num_envs,
                storage_kind=args.collector_buffer_storage,
                name="collector_replay_buffer",
            ),
            batch_size=args.num_envs,
        )

    make_env_fn = partial(
        make_env,
        task=args.task,
        num_envs=per_collector_envs,
        max_episode_steps=args.max_episode_steps,
        device=str(collector_device),
    )
    collector_rnn_backend = "cudnn"
    collector_policy_factory = partial(
        make_actor,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        rnn_backend=collector_rnn_backend,
        device=collector_device,
    )
    if args.collector_backend == "single":
        collector = Collector(
            make_env_fn,
            policy_factory=collector_policy_factory,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            policy_device=collector_device,
            no_cuda_sync=True,
            trust_policy=True,
            storing_device=storing_device,
            compact_obs=args.compact_obs,
            auto_register_policy_transforms=True,
            track_policy_version=True,
            replay_buffer=collector_replay_buffer,
            extend_buffer=True,
        )
    else:
        collector = MultiCollector(
            [make_env_fn] * args.num_collectors,
            sync=args.collector_backend == "sync",
            policy_factory=collector_policy_factory,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            policy_device=collector_device,
            no_cuda_sync=True,
            trust_policy=True,
            storing_device=storing_device,
            compact_obs=args.compact_obs,
            auto_register_policy_transforms=True,
            track_policy_version=True,
            replay_buffer=collector_replay_buffer,
            extend_buffer=True,
            weight_sync_schemes={"policy": MultiProcessWeightSyncScheme()},
        )

    config = vars(args).copy()
    config.update(
        {
            "frames_per_batch": frames_per_batch,
            "per_collector_envs": per_collector_envs,
            "collector_rnn_backend": collector_rnn_backend,
            "updates_per_epoch": updates_per_epoch,
            "storing_device": str(storing_device)
            if storing_device is not None
            else None,
            "double_buffer_collector": args.double_buffer_collector,
            "collector_replay_buffer": args.double_buffer_collector,
            "collector_policy_sync": "before_next",
            "collector_policy_version_tracking": True,
            "collector_weight_sync_scheme": "MultiProcessWeightSyncScheme"
            if args.collector_backend != "single"
            else "single_process_fallback",
            "debug_data_flow": args.debug_data_flow,
        }
    )
    experiment_logger = None
    if args.wandb_mode != "disabled":
        experiment_logger = WandbLogger(
            exp_name=(
                args.wandb_name or f"{args.preset}-{args.storage}-{args.rnn_backend}"
            ),
            project=args.wandb_project,
            offline=args.wandb_mode in ("offline", "dryrun"),
            group=args.wandb_group,
        )
        experiment_logger.log_hparams(config)

    for stats_device in (collector_device, train_device):
        if stats_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(stats_device)

    global_step = 0
    active_generation = 0
    total_iterations = args.iterations + args.warmup_iterations
    completed_iterations = 0
    current_iteration = -1
    current_phase = "before_loop"
    collector_iter = iter(collector)
    try:
        for iteration in range(total_iterations):
            current_iteration = iteration
            timeit.erase()
            current_phase = "collector_policy_sync"
            with timeit("collector_policy_sync_s"):
                collector.update_policy_weights_(
                    weights=TensorDict.from_module(actor).data
                )
                collector.increment_version()
            active_generation += 1
            collector_policy_version = collector.get_policy_version()
            timeit.mark_start("collect_or_sample_s")
            current_phase = "collector_next"
            collected_batch = next(collector_iter)
            timeit.mark_env("collect_or_sample_s")
            current_phase = "iteration_start"
            torchrl_logger.info(
                {
                    "phase": "iteration_start",
                    "iteration": iteration,
                    "collector_generation": active_generation,
                    "collector_policy_version": collector_policy_version,
                }
            )
            with timeit("iter_s"):
                if collector_replay_buffer is None:
                    if collected_batch is None:
                        raise RuntimeError(
                            "collector returned None but no replay buffer was provided"
                        )
                    data = collected_batch
                else:
                    if collected_batch is not None:
                        raise RuntimeError(
                            "collector yielded data despite being given a replay buffer"
                        )
                    current_phase = "collector_replay_buffer_read"
                    data = collector_replay_buffer[:]
                torchrl_logger.info(
                    {
                        "phase": "collector_batch_ready",
                        "iteration": iteration,
                        "collector_replay_buffer": collector_replay_buffer is not None,
                        "batch_numel": data.numel(),
                        "batch_ndim": data.ndim,
                    }
                )
                current_phase = "batch_metrics"
                batch_metrics = _batch_metrics(data)
                if args.debug_data_flow:
                    _add_data_flow_metrics(batch_metrics, "data_flow/source", data)
                current_phase = "batch_to_train_device"
                data = data.to(train_device)
                torchrl_logger.info({"phase": "gae_start", "iteration": iteration})
                current_phase = "gae"
                with timeit("adv_s"), torch.no_grad(), set_recurrent_mode(True):
                    data = adv_module(data)
                torchrl_logger.info({"phase": "gae_done", "iteration": iteration})
                if args.debug_data_flow:
                    _add_data_flow_metrics(batch_metrics, "data_flow/gae", data)
                current_phase = "train_buffer_extend"
                with timeit("collector_to_train_buffer_s"):
                    train_buffer.empty()
                    train_buffer.extend(data)
                torchrl_logger.info(
                    {"phase": "train_buffer_ready", "iteration": iteration}
                )
                if args.debug_data_flow:
                    stored_data = train_buffer[:]
                    _add_data_flow_metrics(
                        batch_metrics, "data_flow/train_buffer", stored_data
                    )
                    _add_buffer_compare_metrics(batch_metrics, data, stored_data)
                del data

                loss_acc = None
                loss_count = 0
                torchrl_logger.info({"phase": "update_start", "iteration": iteration})
                with timeit("update_s"):
                    for epoch in range(args.ppo_epochs):
                        torchrl_logger.info(
                            {
                                "phase": "epoch_start",
                                "iteration": iteration,
                                "epoch": epoch,
                            }
                        )
                        for mini_batch in train_buffer:
                            current_phase = "update_minibatch"
                            mini_batch = mini_batch.to(train_device)
                            loss = update(mini_batch)
                            loss_acc = loss if loss_acc is None else loss_acc + loss
                            loss_count += 1
                            if loss_count == 1:
                                torchrl_logger.info(
                                    {
                                        "phase": "first_optimizer_step_done",
                                        "iteration": iteration,
                                    }
                                )
                    torchrl_logger.info(
                        {
                            "phase": "update_done",
                            "iteration": iteration,
                            "loss_count": loss_count,
                        }
                    )

            timing_metrics = timeit.todict(percall=False, prefix="time")

            _log_iteration_metrics(
                experiment_logger=experiment_logger,
                step=global_step,
                iteration=iteration,
                args=args,
                batch_metrics=batch_metrics,
                collector_device=collector_device,
                train_device=train_device,
                frames_per_batch=frames_per_batch,
                updates_per_epoch=updates_per_epoch,
                sample_num_slices=sample_num_slices,
                sample_seq_len=sample_seq_len,
                timing_metrics=timing_metrics,
                loss_acc=loss_acc,
                loss_count=loss_count,
                collector_generation=active_generation,
                collector_policy_version=collector_policy_version,
                collector_direct_replay_buffer=args.double_buffer_collector,
            )
            global_step += 1
            completed_iterations += 1
            if iteration < args.warmup_iterations:
                for stats_device in (collector_device, train_device):
                    if stats_device.type == "cuda":
                        if args.empty_cache_after_warmup:
                            torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(stats_device)
        torchrl_logger.info(
            {
                "phase": "loop_exhausted",
                "completed_iterations": completed_iterations,
                "target_iterations": total_iterations,
            }
        )
    except BaseException as err:
        torchrl_logger.exception(
            "run aborted during phase=%s iteration=%s with %s: %s",
            current_phase,
            current_iteration,
            type(err).__name__,
            err,
        )
        raise
    finally:
        collector.shutdown()
        if experiment_logger is not None:
            experiment_logger.experiment.finish()
        simulation_app.close()


if __name__ == "__main__":
    with set_exploration_type(ExplorationType.RANDOM):
        main()
