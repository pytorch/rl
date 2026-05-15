# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Recurrent PPO on Isaac Lab with compact rollouts and shifted GAE.

Demonstrates running large-vectorized Isaac Lab environments with an LSTM
policy through a :class:`~torchrl.collectors.MultiCollector`. Isaac Lab is
launched only in the worker subprocesses (the main process never imports
``isaaclab``) so the main process can stay light and own the trainer.

Key TorchRL features exercised:

- :class:`~torchrl.collectors.MultiCollector` with ``policy_factory`` (each
  worker builds its own policy copy and receives weights via
  :class:`~torchrl.weight_update.MultiProcessWeightSyncScheme`).
- ``compact_obs=True`` to drop the redundant ``("next", obs)``. Shifted
  value estimation reconstructs next observations by shifting the root
  observations when the next observations are absent.
- :class:`~torchrl.objectives.value.GAE` with ``shifted=True``: uses the
  root observation shift when compact rollouts omit ``("next", obs)``.
- :class:`~torchrl.modules.LSTMModule` with a configurable
  ``recurrent_backend``: during collection (``set_recurrent_mode=False``)
  the LSTM auto-uses cuDNN regardless of the backend; the configured
  backend kicks in only during training (``set_recurrent_mode=True``).
- Optional ``torch.compile`` + ``CudaGraphModule`` around the update step.

Run on a SLURM Isaac Lab container:

.. code-block:: bash

    python isaaclab_rnn_ppo_memory.py --num-envs 16384 --num-collectors 2 \
        --rollout-steps 32 --rnn-backend scan --compile-update --cudagraph-update
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
from torchrl.collectors import MultiCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import set_recurrent_mode
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record import WandbLogger
from torchrl.weight_update import MultiProcessWeightSyncScheme


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Env / rollout
    parser.add_argument("--task", default="Isaac-Ant-v0")
    parser.add_argument("--num-envs", type=int, default=16_384)
    parser.add_argument("--num-collectors", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=20)
    # Model
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--obs-dim", type=int, default=60)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument(
        "--rnn-backend",
        choices=["pad", "scan", "triton"],
        default="scan",
        help=(
            "LSTM backend used during training (set_recurrent_mode=True). "
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
    # Compile / cudagraph
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
    # Devices
    parser.add_argument("--collector-device", default="cuda:0")
    parser.add_argument("--train-device", default="cuda:1")
    parser.add_argument(
        "--sync-collector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Synchronous MultiCollector. Async (default) yields per-worker batches.",
    )
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
    if args.compile_update or args.cudagraph_update:
        torch._dynamo.config.capture_scalar_outputs = True

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    collector_device = torch.device(args.collector_device)
    train_device = torch.device(args.train_device)
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
        shifted=True,
        device=train_device,
    )
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
    make_env_fn = partial(
        make_env,
        task=args.task,
        num_envs=per_collector_envs,
        max_episode_steps=args.max_episode_steps,
        device=str(collector_device),
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
        compact_obs=True,
        init_fn=partial(_init_isaac_app, device=str(collector_device)),
        auto_register_policy_transforms=True,
        track_policy_version=True,
        weight_sync_schemes={"policy": MultiProcessWeightSyncScheme()},
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
    try:
        for iteration, collected_batch in enumerate(collector):
            if iteration >= args.iterations:
                break
            timeit.erase()
            with timeit("collector_policy_sync"):
                collector.update_policy_weights_(actor)
            with timeit("iteration"):
                data = collected_batch
                loss_acc = None
                loss_count = 0
                for epoch in range(args.ppo_epochs):
                    epoch_data = data.to(train_device)
                    with timeit("advantage"), torch.no_grad(), set_recurrent_mode(True):
                        epoch_data = adv_module(epoch_data)
                    if epoch_data.shape != expected_rollout_shape:
                        raise RuntimeError(
                            f"Expected epoch_data shape {expected_rollout_shape}, "
                            f"got {epoch_data.shape}."
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
                    metrics.update(
                        {
                            f"loss/{k}": float(v / loss_count)
                            for k, v in loss_acc.items()
                        }
                    )
                metrics.update(
                    {
                        "iteration": iteration,
                        "frames": (iteration + 1) * frames_per_batch,
                        "updates_per_epoch": updates_per_epoch,
                    }
                )
                experiment_logger.log_metrics(metrics, step=iteration)
            torchrl_logger.info({"phase": "iteration_done", "iteration": iteration})
    finally:
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


if __name__ == "__main__":
    with set_exploration_type(ExplorationType.RANDOM):
        main()
