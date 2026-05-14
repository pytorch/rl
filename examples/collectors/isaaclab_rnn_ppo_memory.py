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
from functools import partial
from pathlib import Path

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

import torch
import torch._dynamo
import torch._inductor.config
import torch.optim

from isaaclab_rnn_ppo_memory_utils import (
    _add_buffer_compare_metrics,
    _add_data_flow_metrics,
    _apply_preset,
    _assert_time_batch_shape,
    _batch_metrics,
    _log_iteration_metrics,
    _make_storage,
    make_actor,
    make_actor_from_modules,
    make_actor_head,
    make_backbone,
    make_critic_head,
    make_env,
    make_full_value,
)
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.collectors import Collector, MultiCollector
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import set_recurrent_mode
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record import WandbLogger
from torchrl.weight_update import MultiProcessWeightSyncScheme


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
    collector_frames = 0
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
                    # Clone the data to avoid race conditions with the collector
                    data = collector_replay_buffer[:].clone()
                torchrl_logger.info(
                    {
                        "phase": "collector_batch_ready",
                        "iteration": iteration,
                        "collector_replay_buffer": collector_replay_buffer is not None,
                        "batch_numel": data.numel(),
                        "batch_ndim": data.ndim,
                    }
                )
                collector_frames += data.numel()
                current_phase = "batch_metrics"
                batch_metrics = _batch_metrics(data, collector_policy_version)
                if args.debug_data_flow:
                    _add_data_flow_metrics(batch_metrics, "data_flow/source", data)
                current_phase = "batch_to_train_device"

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
                        torchrl_logger.info(
                            {
                                "phase": "gae_start",
                                "iteration": iteration,
                                "epoch": epoch,
                            }
                        )
                        current_phase = "gae"
                        with (
                            timeit(name="adv_s"),
                            torch.no_grad(),
                            set_recurrent_mode(True),
                        ):
                            _assert_time_batch_shape(
                                data,
                                args.num_envs,
                                args.rollout_steps,
                                "gae_batch",
                            )
                            epoch_data = data.to(train_device)
                            epoch_data = adv_module(epoch_data)
                        torchrl_logger.info(
                            {
                                "phase": "gae_done",
                                "iteration": iteration,
                                "epoch": epoch,
                            }
                        )
                        if args.debug_data_flow and epoch == 0:
                            _add_data_flow_metrics(
                                batch_metrics, "data_flow/gae", epoch_data
                            )
                        current_phase = "train_buffer_extend"
                        with timeit("collector_to_train_buffer_s"):
                            train_buffer.empty()
                            train_buffer.extend(epoch_data)
                        torchrl_logger.info(
                            {
                                "phase": "train_buffer_ready",
                                "iteration": iteration,
                                "epoch": epoch,
                            }
                        )
                        if args.debug_data_flow and epoch == 0:
                            stored_data = train_buffer[:]
                            _add_data_flow_metrics(
                                batch_metrics, "data_flow/train_buffer", stored_data
                            )
                            _add_buffer_compare_metrics(
                                batch_metrics, epoch_data, stored_data
                            )
                        for mini_batch in train_buffer:
                            current_phase = "update_minibatch"
                            mini_batch = mini_batch.to(train_device)
                            _assert_time_batch_shape(
                                mini_batch,
                                sample_num_slices,
                                sample_seq_len,
                                "mini_batch",
                            )
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
                del data

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
                collector_frames=collector_frames,
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
