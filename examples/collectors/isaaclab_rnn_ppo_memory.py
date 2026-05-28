# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Recurrent PPO on Isaac Lab with configurable recurrent rollout storage.

Demonstrates running large-vectorized Isaac Lab environments with an LSTM
policy through a :class:`~torchrl.collectors.MultiCollector`. Isaac Lab is
launched only in the worker subprocesses (the main process never imports
``isaaclab``) so the main process can stay light and own the trainer.

Key TorchRL features exercised:

- :class:`~torchrl.collectors.MultiCollector` with ``policy_factory`` (each
  worker builds its own policy copy and receives weights via
  :class:`~torchrl.weight_update.MultiProcessWeightSyncScheme`).
- Optional ``compact_obs=True`` to drop the redundant ``("next", obs)``.
  Shifted value estimation reconstructs next observations by shifting the
  root observations when the next observations are absent.
- Configurable :class:`~torchrl.objectives.value.GAE` shifted mode. The
  ``shifted=True`` path keeps a budgeted constant-shape single value-network
  call and is friendly to ``torch.compile`` + scan/triton LSTM.
  ``shifted=False`` keeps the full observation representation.
- :class:`~torchrl.modules.LSTMModule` with a configurable
  ``recurrent_backend``: during collection (``set_recurrent_mode=False``)
  the LSTM auto-uses cuDNN regardless of the backend; the configured
  backend kicks in only during training (``set_recurrent_mode=True``).
- Optional ``torch.compile`` + ``CudaGraphModule`` around the update step.
- Optional asynchronous rendered evaluation through
  :class:`~torchrl.collectors.Evaluator` on a dedicated eval GPU.

Run on a SLURM Isaac Lab container:

.. code-block:: bash

    python isaaclab_rnn_ppo_memory.py --num-envs 16384 --num-collectors 2 \
        --rollout-steps 32 --rnn-backend scan --compile-update --cudagraph-update

    python isaaclab_rnn_ppo_memory.py --eval --eval-device cuda:2 \
        --eval-num-envs 16 --random-init-steps 16
"""
from __future__ import annotations

import argparse
import logging
import math
from functools import partial

import torch
import torch.optim

from isaaclab_rnn_ppo_memory_utils import _init_isaac_app, make_env, make_models
from tensordict import TensorDictBase
from tensordict.nn import CudaGraphModule
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.collectors import Evaluator, MultiCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import set_recurrent_mode
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record import WandbLogger
from torchrl.weight_update import MultiProcessWeightSyncScheme


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Env / rollout
    parser.add_argument("--task", default="Isaac-Ant-v0")
    parser.add_argument("--num-envs", type=int, default=16_384)
    parser.add_argument("--num-collectors", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--random-init-steps",
        type=int,
        default=0,
        help=(
            "Maximum number of random reset-time environment steps. A value "
            "greater than 0 jitters vectorized envs within each batch."
        ),
    )
    parser.add_argument(
        "--random-init-random",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample reset-time jitter independently per env. Use "
            "--no-random-init-random to apply --random-init-steps to every env."
        ),
    )
    # Model
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--obs-dim", type=int, default=60)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument(
        "--rnn-backend",
        choices=["cudnn", "pad", "scan", "triton"],
        default="scan",
        help=(
            "LSTM backend used during training (set_recurrent_mode=True). "
            "'cudnn' is an alias for the pad/nn.LSTM path. "
            "Collection (set_recurrent_mode=False) always falls back to cuDNN."
        ),
    )
    # PPO
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--mini-batch-steps", type=int, default=8_192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--critic-coeff", type=float, default=1.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--compact-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop redundant ('next', obs) keys from collector batches. Use "
            "--no-compact-obs for the full baseline representation."
        ),
    )
    parser.add_argument(
        "--gae-shifted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the budgeted constant-shape single-call GAE path. Use "
            "--no-gae-shifted for the full shifted=False reference path."
        ),
    )
    parser.add_argument(
        "--gae-shifted-budget",
        type=int,
        default=1,
        help=(
            "Number of extra value-network slots used by shifted GAE. "
            "A budget of 2 can retain one internal reset plus the rollout "
            "boundary without dropping samples."
        ),
    )
    parser.add_argument(
        "--deactivate-vmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace vmap in GAE with a Python loop. Required for "
            "--gae-shifted=false when the value network is recurrent "
            "(cuDNN ops do not compose with vmap)."
        ),
    )
    # Compile / cudagraph
    parser.add_argument(
        "--compile-update", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile-gae",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap the GAE module with torch.compile.",
    )
    parser.add_argument(
        "--env-compile-warmup",
        type=int,
        default=0,
        help=(
            "Compile worker env step_and_maybe_reset after this many eager "
            "warmup calls. 0 disables env compile."
        ),
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
    # Devices
    parser.add_argument("--collector-device", default="cuda:0")
    parser.add_argument("--train-device", default="cuda:1")
    parser.add_argument("--eval-device", default="cuda:2")
    parser.add_argument(
        "--eval-worker-device",
        default=None,
        help=(
            "Device used inside the evaluator worker. If "
            "--eval-cuda-visible-devices is set, defaults to cuda:0 so the "
            "rendering worker can use a remapped single visible GPU."
        ),
    )
    parser.add_argument(
        "--sync-collector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Synchronous MultiCollector. Async (default) yields per-worker batches.",
    )
    # Evaluation
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable asynchronous rendered evaluation on --eval-device.",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-num-envs", type=int, default=16)
    parser.add_argument("--eval-num-trajectories", type=int, default=16)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument(
        "--eval-random-init-steps",
        type=int,
        default=None,
        help=(
            "Maximum reset-time jitter for the eval env. Defaults to "
            "--random-init-steps."
        ),
    )
    parser.add_argument(
        "--eval-backend",
        choices=["process", "thread"],
        default="process",
        help=(
            "Evaluator backend. The process backend keeps Isaac Lab and the "
            "eval CUDA context isolated from training."
        ),
    )
    parser.add_argument(
        "--eval-render-backend",
        choices=["isaac_rtx", "newton_warp", "ovrtx"],
        default=None,
        help=(
            "Renderer backend for the eval tiled camera. Defaults to Isaac "
            "Lab's TiledCamera default."
        ),
    )
    parser.add_argument("--eval-cuda-visible-devices", default=None)
    parser.add_argument("--eval-nvidia-lib-dir", default=None)
    parser.add_argument("--eval-vulkan-icd", default=None)
    parser.add_argument("--eval-xdg-runtime-dir", default=None)
    parser.add_argument("--eval-shutdown-timeout", type=float, default=30.0)
    # Logging
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="torchrl-isaac-rnn-memory")
    parser.add_argument("--wandb-group", default="default")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.num_envs % args.num_collectors:
        raise ValueError("--num-envs must be divisible by --num-collectors.")
    if (
        args.compile_update
        or args.compile_gae
        or args.env_compile_warmup > 0
        or args.cudagraph_update
    ):
        torch._dynamo.config.capture_scalar_outputs = True
    gae_shifted = args.gae_shifted

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    collector_device = torch.device(args.collector_device)
    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)
    if train_device.type == "cuda":
        torch.cuda.set_device(train_device)
        torch.cuda.manual_seed_all(args.seed)

    per_collector_envs = args.num_envs // args.num_collectors
    frames_per_batch = args.num_envs * args.rollout_steps
    sample_num_slices = max(1, args.mini_batch_steps // args.rollout_steps)
    expected_rollout_shape = torch.Size((args.num_envs, args.rollout_steps))

    # ---- Training model (lives on train_device, recurrent_mode=True path) ----
    actor, critic, full_value = make_models(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        rnn_backend=args.rnn_backend,
        device=train_device,
    )

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
        shifted=gae_shifted,
        shifted_budget=args.gae_shifted_budget,
        deactivate_vmap=args.deactivate_vmap,
        device=train_device,
    )
    if args.compile_gae:
        adv_module = torch.compile(adv_module, mode=args.compile_mode)
    optim = group_optimizers(
        torch.optim.Adam(
            actor.parameters(), lr=args.lr, eps=1e-5, capturable=args.cudagraph_update
        ),
        torch.optim.Adam(
            critic.parameters(), lr=args.lr, eps=1e-5, capturable=args.cudagraph_update
        ),
    )

    def update(batch):
        with set_recurrent_mode(True):
            loss = loss_module(batch)
        total = loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()),
            max_norm=args.clip_grad_norm,
        )
        optim.step()
        optim.zero_grad(set_to_none=True)
        loss_out = loss.detach()
        loss_out.set("loss_total", total.detach())
        loss_out.set("grad_norm", grad_norm.detach())
        return loss_out

    if args.compile_update:
        update = torch.compile(update, mode=args.compile_mode)
    if args.cudagraph_update:
        update = CudaGraphModule(
            update, in_keys=[], out_keys=[], warmup=args.cudagraph_warmup
        )

    # ---- Collector (Isaac lives in workers; main process never imports it) ----
    compile_env: bool | dict[str, int]
    if args.env_compile_warmup > 0:
        compile_env = {"warmup": args.env_compile_warmup}
    else:
        compile_env = False
    make_env_fn = partial(
        make_env,
        task=args.task,
        num_envs=per_collector_envs,
        max_episode_steps=args.max_episode_steps,
        device=str(collector_device),
        random_init_steps=args.random_init_steps,
        random_init_random=args.random_init_random,
        compile_env=compile_env,
    )
    # The worker's actor uses the same backend; during collection the LSTM
    # runs with set_recurrent_mode=False and auto-dispatches to cuDNN.
    collector_policy_factory = partial(
        _make_collector_actor,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        rnn_backend=args.rnn_backend,
        device=collector_device,
    )
    collector = MultiCollector(
        [make_env_fn] * args.num_collectors,
        sync=args.sync_collector,
        policy_factory=collector_policy_factory,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        policy_device=collector_device,
        storing_device="cpu",
        no_cuda_sync=True,
        trust_policy=True,
        compact_obs=args.compact_obs,
        init_fn=partial(_init_isaac_app, device=str(collector_device)),
        auto_register_policy_transforms=True,
        weight_sync_schemes={"policy": MultiProcessWeightSyncScheme()},
    )

    evaluator = None
    if args.eval:
        eval_worker_device = torch.device(
            args.eval_worker_device
            or ("cuda:0" if args.eval_cuda_visible_devices is not None else eval_device)
        )
        eval_max_steps = args.eval_max_steps or args.max_episode_steps
        eval_random_init_steps = (
            args.random_init_steps
            if args.eval_random_init_steps is None
            else args.eval_random_init_steps
        )
        make_eval_env_fn = partial(
            make_env,
            task=args.task,
            num_envs=args.eval_num_envs,
            max_episode_steps=eval_max_steps,
            device=str(eval_worker_device),
            random_init_steps=eval_random_init_steps,
            random_init_random=args.random_init_random,
            render=True,
            render_backend=args.eval_render_backend,
        )
        evaluator = Evaluator(
            make_eval_env_fn,
            policy_factory=partial(
                _make_eval_actor,
                obs_dim=args.obs_dim,
                action_dim=args.action_dim,
                hidden_size=args.hidden_size,
                rnn_backend=args.rnn_backend,
                device=str(eval_worker_device),
            ),
            num_trajectories=args.eval_num_trajectories,
            max_steps=None if eval_random_init_steps else eval_max_steps,
            frames_per_batch=args.eval_num_envs * eval_max_steps,
            backend=args.eval_backend,
            init_fn=partial(
                _init_isaac_app,
                device=str(eval_worker_device),
                enable_cameras=True,
                rendering_mode="performance",
                cuda_visible_devices=args.eval_cuda_visible_devices,
                nvidia_lib_dir=args.eval_nvidia_lib_dir,
                vulkan_icd=args.eval_vulkan_icd,
                xdg_runtime_dir=args.eval_xdg_runtime_dir,
            ),
            device=eval_worker_device,
            metrics_fn=_rendered_eval_metrics,
            dump_video=False,
            busy_policy="skip",
            collector_kwargs={
                "policy_device": eval_worker_device,
                "env_device": eval_worker_device,
                "storing_device": "cpu",
                "no_cuda_sync": True,
                "trust_policy": True,
                "auto_register_policy_transforms": True,
                "compact_obs": False,
            },
            weight_sync_schemes={"policy": MultiProcessWeightSyncScheme()}
            if args.eval_backend == "process"
            else None,
        )

    train_buffer = ReplayBuffer(
        storage=LazyTensorStorage(args.num_envs, device="cpu", ndim=1),
        sampler=SamplerWithoutReplacement(drop_last=True),
        batch_size=sample_num_slices,
    )
    updates_per_epoch = math.ceil(args.num_envs / sample_num_slices)

    experiment_logger: WandbLogger | None = None
    if args.wandb_mode != "disabled":
        experiment_logger = WandbLogger(
            exp_name=args.wandb_name or f"{args.task}-{args.rnn_backend}",
            project=args.wandb_project,
            offline=args.wandb_mode in ("offline", "dryrun"),
            group=args.wandb_group,
        )
        experiment_logger.log_hparams(vars(args))

    # ---- Training loop ----
    collector_iter = iter(collector)
    try:
        for iteration in range(args.iterations):
            timeit.erase()
            if evaluator is not None:
                result = evaluator.poll()
                if result is not None:
                    _log_eval_result(
                        result,
                        experiment_logger=experiment_logger,
                    )
            with timeit("collector_policy_sync"):
                collector.update_policy_weights_(actor)
            with timeit("collector_next"):
                collected_batch = next(collector_iter)
            with timeit("training"):
                data = _normalize_rollout_batch(collected_batch, expected_rollout_shape)
                loss_acc = None
                loss_count = 0
                for epoch in range(args.ppo_epochs):
                    epoch_data = data.to(train_device)
                    _assert_rollout_shapes(
                        epoch_data,
                        expected_shape=expected_rollout_shape,
                        hidden_size=args.hidden_size,
                        phase="before_gae",
                    )
                    with timeit("advantage"), torch.no_grad(), set_recurrent_mode(True):
                        epoch_data = adv_module(epoch_data)
                    _assert_rollout_shapes(
                        epoch_data,
                        expected_shape=expected_rollout_shape,
                        hidden_size=args.hidden_size,
                        phase="after_gae",
                    )
                    if iteration == 0 and epoch == 0:
                        torchrl_logger.info(
                            {
                                "phase": "pre_train_buffer_extend",
                                "epoch_data_shape": tuple(epoch_data.shape),
                                "epoch_data_leaf_shapes": _leaf_shape_summary(
                                    epoch_data
                                ),
                                "epoch_data_0_shape": tuple(epoch_data[0].shape),
                                "epoch_data_0_leaf_shapes": _leaf_shape_summary(
                                    epoch_data[0]
                                ),
                            }
                        )
                    train_buffer.empty()
                    train_buffer.extend(epoch_data)
                    if train_buffer._storage._storage.shape != expected_rollout_shape:
                        raise RuntimeError(
                            "Expected train buffer storage shape "
                            f"{expected_rollout_shape}, got "
                            f"{train_buffer._storage._storage.shape}."
                        )
                    for mini_batch in train_buffer:
                        with timeit("update"):
                            loss = update(mini_batch.to(train_device))
                        loss_acc = loss if loss_acc is None else loss_acc + loss
                        loss_count += 1
            if experiment_logger is not None:
                metrics = timeit.todict(percall=False, prefix="time")
                if loss_acc is not None and loss_count > 0:
                    metrics.update(_loss_metrics(loss_acc, loss_count))
                metrics.update(
                    {
                        "training/iteration": iteration,
                        "training/updates_per_epoch": updates_per_epoch,
                        "training/updates_total": loss_count,
                    }
                )
                metrics.update(
                    _inference_metrics(
                        data,
                        frames=(iteration + 1) * frames_per_batch,
                    )
                )
                metrics.update(_cuda_metrics("collector_cuda", collector_device))
                metrics.update(_cuda_metrics("train_cuda", train_device))
                experiment_logger.log_metrics(metrics, step=iteration)
            if (
                evaluator is not None
                and args.eval_every > 0
                and iteration % args.eval_every == 0
            ):
                accepted = evaluator.trigger_eval(
                    actor,
                    step=(iteration + 1) * frames_per_batch,
                )
                if not accepted:
                    torchrl_logger.info(
                        {"phase": "eval_skipped", "iteration": iteration}
                    )
            torchrl_logger.info({"phase": "iteration_done", "iteration": iteration})
    finally:
        if evaluator is not None:
            result = evaluator.poll(timeout=args.eval_shutdown_timeout)
            if result is not None:
                _log_eval_result(
                    result,
                    experiment_logger=experiment_logger,
                )
            evaluator.shutdown(timeout=args.eval_shutdown_timeout)
        collector.shutdown()
        if experiment_logger is not None:
            experiment_logger.experiment.finish()


def _make_collector_actor(
    obs_dim: int,
    action_dim: int,
    hidden_size: int,
    rnn_backend: str,
    device,
):
    """Worker-side actor factory: one fresh ProbabilisticActor per worker."""
    actor, _, _ = make_models(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        rnn_backend=rnn_backend,
        device=device,
    )
    return actor


def _make_eval_actor(
    *unused_args,
    obs_dim: int,
    action_dim: int,
    hidden_size: int,
    rnn_backend: str,
    device,
):
    del unused_args
    actor, _, _ = make_models(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        rnn_backend=rnn_backend,
        device=torch.device(device),
    )
    return actor


if __name__ == "__main__":
    with set_exploration_type(ExplorationType.RANDOM):
        main()
