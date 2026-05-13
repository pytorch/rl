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
import importlib.util
import logging
import math
import os
import sys
import time
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
    parser.add_argument("--updates-per-iteration", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--obs-dim", type=int, default=60)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--critic-coeff", type=float, default=1.0)
    parser.add_argument("--collector-backend", choices=["single", "async"], default="async")
    parser.add_argument("--storage", choices=["cuda", "cpu", "memmap"], default="cuda")
    parser.add_argument("--compact-obs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shifted-gae", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vectorized-gae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rnn-backend",
        choices=["cudnn", "scan", "triton"],
        default="cudnn",
    )
    parser.add_argument("--compile-update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument("--cudagraph-update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cudagraph-warmup", type=int, default=8)
    parser.add_argument("--cudagraph-trees", action=argparse.BooleanOptionalAction, default=True)
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
from torchrl._utils import compile_with_warmup, logger as torchrl_logger
from torchrl.collectors import Collector, MultiAsyncCollector
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
from torchrl.modules import (
    LSTMModule,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
    set_recurrent_mode,
)
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE

_has_wandb = importlib.util.find_spec("wandb") is not None
if _has_wandb:
    import wandb


RnnBackend = Literal["cudnn", "scan", "triton"]


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
    if hasattr(cfg, "episode_length_s") and hasattr(cfg, "sim") and hasattr(cfg.sim, "dt"):
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
    module = TensorDictSequential(embed, lstm, head)
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
):
    if args.storage == "memmap":
        run_dir = args.memmap_dir / f"{args.preset}_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return LazyMemmapStorage(
            max_size=max_size,
            scratch_dir=run_dir,
            ndim=ndim,
            shared_init=True,
            auto_cleanup=True,
        )
    if args.storage == "cuda":
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
        elif key.startswith("loss/grad_norm_"):
            grad_name = key.removeprefix("loss/grad_norm_")
            extra[f"training/grad_norm/{grad_name}"] = value
        elif key.startswith("loss/loss_"):
            loss_name = key.removeprefix("loss/loss_")
            extra[f"training/loss/{loss_name}"] = value
        elif key.startswith("loss/"):
            extra[f"training/{key.removeprefix('loss/')}"] = value
    payload.update(extra)


def _log(wandb_run, payload: dict[str, float | int | str], step: int) -> None:
    _add_metric_namespaces(payload)
    torchrl_logger.info(payload)
    if wandb_run is not None:
        wandb_run.log(payload, step=step)


def _maybe_to_device(data: TensorDictBase, device: torch.device) -> TensorDictBase:
    if data.device == device:
        return data
    return data.to(device, non_blocking=True)


def _leaf_contiguous(data: TensorDictBase) -> TensorDictBase:
    return data.contiguous()


def _get_optional(data: TensorDictBase, key):
    try:
        return data.get(key)
    except KeyError:
        return None


def _add_batch_stats(
    metrics: dict[str, float | int | str], batch: TensorDictBase
) -> None:
    reward = _get_optional(batch, ("next", "reward"))
    if reward is None:
        reward = _get_optional(batch, "reward")
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
        value = _get_optional(batch, ("next", name))
        if value is None:
            value = _get_optional(batch, name)
        if value is not None:
            metrics[f"{name}/fraction"] = float(value.float().mean().detach().cpu())


def _grad_norm(params: tuple[torch.nn.Parameter, ...], device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for param in params:
        grad = param.grad
        if grad is not None:
            total = total + grad.detach().float().square().sum()
    return total.sqrt()


def _sync_collector_policy(collector: Collector | MultiAsyncCollector, actor: nn.Module) -> None:
    collector.update_policy_weights_(weights=TensorDict.from_module(actor).data)


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
    if train_device.type == "cuda":
        torch.cuda.set_device(train_device)
        torch.cuda.manual_seed_all(args.seed)

    per_collector_envs = args.num_envs // args.num_collectors
    frames_per_batch = args.num_envs * args.rollout_steps

    actor = make_actor(
        args.obs_dim,
        args.action_dim,
        args.hidden_size,
        args.rnn_backend,
        train_device,
    )
    critic = make_critic(args.obs_dim, args.hidden_size, train_device)

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
        critic_network=critic,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        critic_coeff=args.critic_coeff,
        normalize_advantage=True,
    )
    adv_module = GAE(
        gamma=args.gamma,
        lmbda=args.gae_lambda,
        value_network=critic,
        average_gae=False,
        shifted=args.shifted_gae,
        vectorized=args.vectorized_gae,
        device=train_device,
    )
    def compute_loss(batch: TensorDictBase):
        with set_recurrent_mode(True):
            return loss_module(batch)

    if args.compile_update:
        compute_loss = compile_with_warmup(
            compute_loss, mode=args.compile_mode, warmup=1
        )

    def update(batch: TensorDictBase):
        loss = compute_loss(batch)
        total_loss = (
            loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
        )
        total_loss_value = total_loss.detach()
        total_loss.backward()
        actor_grad_norm = _grad_norm(actor_params, train_device)
        critic_grad_norm = _grad_norm(critic_params, train_device)
        grad_norm = (actor_grad_norm.square() + critic_grad_norm.square()).sqrt()
        optim.step()
        optim.zero_grad(set_to_none=True)
        loss_out = loss.detach()
        loss_out.set("loss_total", total_loss_value)
        loss_out.set("grad_norm_actor", actor_grad_norm.detach())
        loss_out.set("grad_norm_critic", critic_grad_norm.detach())
        loss_out.set("grad_norm_total", grad_norm.detach())
        return loss_out

    if args.cudagraph_update:
        update = CudaGraphModule(
            update, in_keys=[], out_keys=[], warmup=args.cudagraph_warmup
        )

    sample_num_slices = max(1, args.mini_batch_steps // args.rollout_steps)
    train_buffer = ReplayBuffer(
        storage=_make_storage(args, train_device, frames_per_batch, ndim=2),
        sampler=SliceSampler(num_slices=sample_num_slices),
        batch_size=args.mini_batch_steps,
    )
    updates_per_epoch = max(
        1,
        args.updates_per_iteration,
        math.ceil(frames_per_batch / args.mini_batch_steps),
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
            no_cuda_sync=True,
            trust_policy=True,
            compact_obs=args.compact_obs,
            auto_register_policy_transforms=True,
        )
    else:
        collector = MultiAsyncCollector(
            [make_env_fn] * args.num_collectors,
            policy_factory=collector_policy_factory,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            no_cuda_sync=True,
            trust_policy=True,
            compact_obs=args.compact_obs,
            cat_results="stack",
            auto_register_policy_transforms=True,
        )
    _sync_collector_policy(collector, actor)

    config = vars(args).copy()
    config.update(
        {
            "frames_per_batch": frames_per_batch,
            "per_collector_envs": per_collector_envs,
            "collector_rnn_backend": collector_rnn_backend,
            "updates_per_epoch": updates_per_epoch,
        }
    )
    wandb_run = None
    if _has_wandb and args.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"{args.preset}-{args.storage}-{args.rnn_backend}",
            mode=args.wandb_mode,
            config=config,
        )

    for stats_device in (collector_device, train_device):
        if stats_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(stats_device)

    collector_iter = iter(collector)
    global_step = 0
    try:
        for iteration in range(args.iterations + args.warmup_iterations):
            metrics: dict[str, float | int | str] = {
                "iteration": iteration,
                "phase": "train_warmup"
                if iteration < args.warmup_iterations
                else "train",
            }
            torchrl_logger.info({"phase": "iteration_start", "iteration": iteration})
            t0 = time.perf_counter()

            torchrl_logger.info({"phase": "collect_start", "iteration": iteration})
            batch = next(collector_iter)

            metrics["time/collect_or_sample_s"] = time.perf_counter() - t0
            if batch is None:
                torchrl_logger.info({"phase": "empty_batch", "iteration": iteration})
                continue
            metrics["batch/numel"] = batch.numel()
            metrics["batch/ndim"] = batch.ndim
            _add_batch_stats(metrics, batch)

            loss_acc = None
            loss_count = 0
            adv_time = 0.0
            t_update = time.perf_counter()
            torchrl_logger.info({"phase": "update_start", "iteration": iteration})
            metrics["updates/per_epoch"] = updates_per_epoch
            metrics["updates/total"] = updates_per_epoch * args.ppo_epochs
            for epoch in range(args.ppo_epochs):
                t_adv = time.perf_counter()
                torchrl_logger.info(
                    {
                        "phase": "advantage_start",
                        "iteration": iteration,
                        "epoch": epoch,
                    }
                )
                epoch_batch = _leaf_contiguous(_maybe_to_device(batch, train_device))
                with torch.no_grad(), set_recurrent_mode(True):
                    epoch_batch = adv_module(epoch_batch)
                adv_time += time.perf_counter() - t_adv

                train_buffer.empty()
                train_buffer.extend(epoch_batch)

                for _ in range(updates_per_epoch):
                    mini_batch = train_buffer.sample()
                    mini_batch = _maybe_to_device(mini_batch, train_device)
                    mini_batch = _leaf_contiguous(mini_batch)
                    loss = update(mini_batch)
                    loss_acc = loss if loss_acc is None else loss_acc + loss
                    loss_count += 1
            metrics["time/adv_s"] = adv_time
            metrics["time/update_s"] = time.perf_counter() - t_update
            if loss_acc is not None:
                loss_mean = loss_acc.apply(
                    lambda x: (x / loss_count).float().mean(), batch_size=[]
                )
                for key, value in loss_mean.items():
                    metrics[f"loss/{key}"] = float(value.detach().cpu())

            metrics.update(_all_cuda_stats(collector_device, train_device))
            metrics["time/iter_s"] = time.perf_counter() - t0
            if iteration >= args.warmup_iterations:
                metrics["frames"] = (
                    iteration - args.warmup_iterations + 1
                ) * frames_per_batch
            _log(wandb_run, metrics, global_step)
            global_step += 1
            if iteration < args.warmup_iterations:
                for stats_device in (collector_device, train_device):
                    if stats_device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(stats_device)

            _sync_collector_policy(collector, actor)
    except BaseException:
        torchrl_logger.exception("Isaac RNN PPO memory comparison failed.")
        raise
    finally:
        collector.shutdown()
        if wandb_run is not None:
            wandb_run.finish()
        simulation_app.close()


if __name__ == "__main__":
    with set_exploration_type(ExplorationType.RANDOM):
        main()
