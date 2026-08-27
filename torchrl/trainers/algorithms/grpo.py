# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GRPO (Group Relative Policy Optimization) Trainer for LLM alignment.

This module provides a :class:`GRPOTrainer` and :class:`GRPOOptimizationStepper`
that integrate the GRPO / LLM alignment training loop into the standard
TorchRL :class:`~torchrl.trainers.Trainer` framework.

The key differences from a standard RL trainer are:

- Mixed-precision training via ``torch.amp.autocast`` and ``torch.amp.GradScaler``.
- Gradient accumulation across multiple micro-batches.
- A *weight-sync sender* (wired through
  :class:`~torchrl.trainers.UpdateWeights`) that pushes updated weights back
  to the inference engine (vLLM / SGLang) after a configurable number of
  optimizer steps.
- LLM-specific logging (KL divergence, ESS, per-token loss, etc.).
- Native support for the ``RayReplayBuffer`` and ``RayLLMCollector``
  used in the SOTA GRPO scripts.
"""

from __future__ import annotations

import gc
import pathlib
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import torch
from tensordict import TensorDictBase
from torch import nn, optim
from torch.amp import GradScaler
from torchrl.checkpoint import Checkpoint, CheckpointRotation
from torchrl.collectors import BaseCollector
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.data.utils import DEVICE_TYPING
from torchrl.objectives.common import LossModule
from torchrl.record.loggers import Logger
from torchrl.trainers.trainers import (
    LogScalar,
    OptimizationStepper,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)


class GRPOOptimizationStepper(OptimizationStepper):
    """Mixed-precision optimization step for GRPO / LLM alignment training.

    This stepper wraps each forward/backward pass in ``torch.amp.autocast``
    and optionally scales gradients with ``torch.amp.GradScaler`` (for fp16).
    It also implements *gradient accumulation*: gradients are accumulated for
    ``gradient_accumulation_steps`` micro-batches before the optimizer is
    stepped and zeroed.

    Args:
        optimizer (optim.Optimizer): The optimizer to use.
        mixed_precision (bool, optional): Whether to enable mixed-precision
            training. Default: ``False``.
        autocast_dtype (torch.dtype, optional): The dtype to use inside
            ``autocast``. Default: ``torch.bfloat16``.
        gradient_accumulation_steps (int, optional): Number of micro-batches
            over which gradients are accumulated before a step. Default: ``1``.
        clip_norm (float, optional): Maximum gradient norm for clipping.
            Default: ``1.0``.

    .. note::
        ``GradScaler`` is only enabled when ``mixed_precision=True`` *and*
        ``autocast_dtype=torch.float16``.  With bfloat16 (the recommended
        dtype for modern GPUs) the scaler is a no-op and is not created.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        *,
        mixed_precision: bool = False,
        autocast_dtype: torch.dtype = torch.bfloat16,
        gradient_accumulation_steps: int = 1,
        clip_norm: float | None = 1.0,
    ) -> None:
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        self.optimizer = optimizer
        self.mixed_precision = mixed_precision
        self.autocast_dtype = autocast_dtype
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.clip_norm = clip_norm

        # GradScaler is only useful for fp16; bf16 doesn't need it.
        self._use_scaler = mixed_precision and (autocast_dtype == torch.float16)
        self.scaler = GradScaler("cuda", enabled=self._use_scaler)

        # Internal micro-batch counter (reset after every optimizer step).
        self._micro_step: int = 0
        self._optimizer_step_count: int = 0

    @property
    def optimizer_step_count(self) -> int:
        """Number of completed optimizer steps.

        Discounts gradient-accumulation micro-steps and steps skipped by the
        GradScaler on overflow. Read by hooks that act on an optimizer-step
        cadence (e.g. :class:`~torchrl.trainers.UpdateWeights` with
        ``interval_unit="optim_steps"``).
        """
        return self._optimizer_step_count

    # ------------------------------------------------------------------
    # Checkpointing (optimizer + scaler state)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        if self._micro_step % self.gradient_accumulation_steps != 0:
            raise RuntimeError(
                f"Cannot save stepper state mid-accumulation. (micro_step={self._micro_step}, "
                f"accumulation_steps={self.gradient_accumulation_steps}). "
                "Adjust your save_interval to align with the gradient accumulation window."
            )

        sd: dict[str, Any] = {
            "optimizer": self.optimizer.state_dict(),
            "micro_step": self._micro_step,
            "optimizer_step_count": self._optimizer_step_count,
        }
        if self._use_scaler:
            sd["scaler"] = self.scaler.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._micro_step = state_dict.get("micro_step", 0)
        self._optimizer_step_count = state_dict.get("optimizer_step_count", 0)
        if self._use_scaler and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, trainer: Trainer, sub_batch: TensorDictBase) -> TensorDictBase:
        """Perform one GRPO forward pass and scaled backward pass.

        The optimizer is only stepped and zeroed every
        ``gradient_accumulation_steps`` calls.

        Args:
            trainer (Trainer): The owning :class:`~torchrl.trainers.Trainer`.
            sub_batch (TensorDictBase): Mini-batch used for this step.

        Returns:
            A :class:`~tensordict.TensorDict` with scalar metrics (losses,
            grad_norm) suitable for logging.
        """
        # ---- forward pass (optionally under autocast) ----
        with torch.amp.autocast(
            "cuda",
            enabled=self.mixed_precision,
            dtype=self.autocast_dtype,
        ):
            losses_td = trainer.compute_loss(sub_batch)
            # Sum all loss_* keys and normalise by accumulation steps.
            loss_items = [v for k, v in losses_td.items() if k.startswith("loss")]
            if not loss_items:
                raise RuntimeError(
                    "GRPOLoss returned no 'loss_*' keys. "
                    "Make sure your loss module prefixes scalar outputs with 'loss'."
                )
            loss = sum(loss_items) / self.gradient_accumulation_steps

        # ---- backward pass ----
        if self._use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._micro_step += 1

        # ---- optimizer step every `gradient_accumulation_steps` micro-batches ----
        if self._micro_step % self.gradient_accumulation_steps == 0:
            if self._use_scaler:
                self.scaler.unscale_(self.optimizer)

            grad_norm = 0.0
            if self.clip_norm is not None:
                params = [
                    p for group in self.optimizer.param_groups for p in group["params"]
                ]
                grad_norm = float(nn.utils.clip_grad_norm_(params, self.clip_norm))

            if self._use_scaler:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # If scale dropped, an overflow occurred and optimizer.step() was skipped.
                if self.scaler.get_scale() >= scale_before:
                    self._optimizer_step_count += 1
            else:
                self.optimizer.step()
                self._optimizer_step_count += 1
            self.optimizer.zero_grad(set_to_none=True)
            losses_td["grad_norm"] = torch.tensor(grad_norm)

            # Free memory after the step.
            gc.collect()

        return losses_td.detach()


class GRPOTrainer(Trainer):
    """A trainer for LLM alignment using GRPO (or compatible) objectives.

    .. warning::
        This is an experimental/prototype feature. The API may change in future
        versions. Please report any issues or feedback to help improve this
        implementation.

    This trainer integrates the full GRPO training loop —
    mixed-precision, gradient accumulation, inference-weight synchronization,
    and LLM-specific logging — into the standard
    :class:`~torchrl.trainers.Trainer` hook system.

    It is designed to work with:

    - :class:`~torchrl.objectives.llm.GRPOLoss` (or any ``LossModule`` whose
      outputs start with ``"loss_"``)
    - :class:`~torchrl.collectors.llm.RayLLMCollector`
    - :class:`~torchrl.data.replay_buffers.RayReplayBuffer` (or any
      :class:`~torchrl.data.ReplayBuffer`)

    The weight-sync sender is intentionally decoupled from the trainer so that
    neither ``vllm`` nor ``sglang`` need to be imported by the core library.

    Examples:
        >>> from torchrl.trainers.algorithms.grpo import GRPOTrainer
        >>> import torch
        >>> from unittest.mock import MagicMock
        >>> collector = MagicMock()
        >>> loss_module = MagicMock()
        >>> optimizer = torch.optim.Adam(torch.nn.Linear(2, 2).parameters())
        >>> trainer = GRPOTrainer(
        ...     collector=collector,
        ...     total_frames=100,
        ...     frame_skip=1,
        ...     optim_steps_per_batch=2,
        ...     loss_module=loss_module,
        ...     optimizer=optimizer,
        ...     log_rewards=False,
        ...     log_kl=False,
        ... )
        >>> # trainer.train()

    Args:
        collector (BaseCollector): The data collector (typically a
            :class:`~torchrl.collectors.llm.RayLLMCollector`).
        total_frames (int): Total number of frames / dialog turns.
        frame_skip (int): Frame skip value (set to 1 for LLM tasks).
        optim_steps_per_batch (int, optional): Number of micro-batches drawn
            from the replay buffer per collected batch and epoch. ``None``
            (default) iterates over the whole replay buffer once per epoch.
        loss_module (LossModule): The GRPO loss module.
        optimizer (optim.Optimizer, optional): Optimizer. Required when
            ``optimization_stepper`` is not provided.
        optimization_stepper (GRPOOptimizationStepper, optional): Custom
            stepper. If omitted, a :class:`GRPOOptimizationStepper` is
            constructed automatically from ``optimizer`` and the
            mixed-precision arguments below.
        weight_sync_sender (optional): Object with an ``update_weights()``
            method used to push training weights to the inference engine.
            Pass ``None`` to disable weight synchronization (useful for
            offline testing).
        weight_update_frequency (int, optional): Optimizer steps between
            weight pushes to the inference engine when ``async_collection=True``
            (registered at the ``post_optim`` stage through
            :class:`~torchrl.trainers.UpdateWeights`). In sync mode weights
            are pushed once per collected batch and this value is unused.
            Default: ``1``.
        empty_replay_buffer_on_weight_update (bool, optional): If ``True``,
            the replay buffer is emptied after each weight push (sync GRPO).
            Default: ``False``.
        replay_buffer (ReplayBuffer, optional): The replay buffer used for
            sampling.
        batch_size (int, optional): Override the replay buffer's batch size.
        device (torch.device, optional): Device on which sampled batches are
            placed before the loss forward pass (typically the training
            device). ``None`` leaves samples on their storage device.
        mixed_precision (bool, optional): Enable autocast + GradScaler.
            Default: ``False``.
        autocast_dtype (torch.dtype, optional): dtype for ``autocast``.
            Default: ``torch.bfloat16``.
        gradient_accumulation_steps (int, optional): Gradient accumulation.
            Default: ``1``.
        logger (Logger, optional): Logger (e.g. ``WandbLogger``).
        clip_grad_norm (bool, optional): Unused — clipping is handled by the
            stepper. Kept for API compatibility.
        clip_norm (float, optional): Gradient clip norm. Default: ``1.0``.
        progress_bar (bool, optional): Show a ``tqdm`` progress bar.
        seed (int, optional): Random seed.
        save_trainer_interval (int, optional): Frame interval between saves.
        log_interval (int, optional): Frame interval between logs.
        save_trainer_file (str | Path, optional): Path for legacy saves.
        checkpoint (Checkpoint, optional): Unified checkpoint object.
        checkpoint_rotation (CheckpointRotation, optional): Rotation policy.
        checkpoint_metadata (Callable, optional): Extra metadata callback.
        num_epochs (int, optional): Epochs per collected batch. Default: ``1``.
        async_collection (bool, optional): Whether data is collected
            asynchronously (``grpo-async`` mode). Default: ``False``.
        log_timings (bool, optional): Log timing of each hook. Default: ``False``.
        auto_log_optim_steps (bool, optional): Log ``optim_steps`` after each
            optimization loop. Default: ``True``.
        log_rewards (bool, optional): Log reward / return statistics.
            Default: ``True``.
        log_kl (bool, optional): Log KL-divergence keys from the loss output.
            Default: ``True``.

    Examples:
        >>> from torchrl.trainers.algorithms.grpo import GRPOTrainer
        >>> # Assuming you have a collector, loss_fn, optimizer, replay_buffer,
        >>> # and weight_sync_sender already constructed (see SOTA scripts):
        >>> trainer = GRPOTrainer(
        ...     collector=collector,
        ...     total_frames=cfg.train.total_dialog_turns,
        ...     frame_skip=1,
        ...     optim_steps_per_batch=cfg.train.epochs,
        ...     loss_module=loss_fn,
        ...     optimizer=optimizer,
        ...     weight_sync_sender=sender,
        ...     weight_update_frequency=1,
        ...     empty_replay_buffer_on_weight_update=cfg.train.empty_replay_buffer,
        ...     replay_buffer=replay_buffer,
        ...     mixed_precision=cfg.train.mixed_precision,
        ...     gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        ...     clip_norm=cfg.optimizer.clip_grad_norm,
        ...     logger=wandb_logger,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        *,
        collector: BaseCollector,
        total_frames: int,
        frame_skip: int = 1,
        optim_steps_per_batch: int | None = None,
        loss_module: LossModule | Callable[[TensorDictBase], TensorDictBase],
        optimizer: optim.Optimizer | None = None,
        optimization_stepper: GRPOOptimizationStepper | None = None,
        # LLM-specific
        weight_sync_sender: Any | None = None,
        weight_update_frequency: int = 1,
        empty_replay_buffer_on_weight_update: bool = False,
        # Replay buffer
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int | None = None,
        device: DEVICE_TYPING | None = None,
        # Mixed precision / gradient accumulation
        mixed_precision: bool = False,
        autocast_dtype: torch.dtype = torch.bfloat16,
        gradient_accumulation_steps: int = 1,
        # Standard trainer args
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = 1.0,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
        checkpoint: Checkpoint | None = None,
        checkpoint_rotation: CheckpointRotation | None = None,
        checkpoint_metadata: Callable[[Trainer], Mapping[str, Any]] | None = None,
        num_epochs: int = 1,
        async_collection: bool = False,
        log_timings: bool = False,
        auto_log_optim_steps: bool = True,
        # Logging toggles
        log_rewards: bool = True,
        log_kl: bool = True,
    ) -> None:
        warnings.warn(
            "GRPOTrainer is an experimental/prototype feature. The API may "
            "change in future versions. Please report any issues or feedback "
            "to help improve this implementation.",
            UserWarning,
            stacklevel=2,
        )

        # --- Build a stepper if one wasn't provided ---
        if optimization_stepper is None:
            if optimizer is None:
                raise ValueError(
                    "GRPOTrainer requires either an `optimizer` or a custom "
                    "`optimization_stepper`."
                )
            optimization_stepper = GRPOOptimizationStepper(
                optimizer=optimizer,
                mixed_precision=mixed_precision,
                autocast_dtype=autocast_dtype,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_norm=clip_norm,
            )

        super().__init__(
            collector=collector,
            total_frames=total_frames,
            frame_skip=frame_skip,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=loss_module,
            optimizer=optimizer,
            optimization_stepper=optimization_stepper,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            logger=logger,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            progress_bar=progress_bar,
            seed=seed,
            save_trainer_interval=save_trainer_interval,
            log_interval=log_interval,
            save_trainer_file=save_trainer_file,
            checkpoint=checkpoint,
            checkpoint_rotation=checkpoint_rotation,
            checkpoint_metadata=checkpoint_metadata,
            num_epochs=num_epochs,
            async_collection=async_collection,
            log_timings=log_timings,
            auto_log_optim_steps=auto_log_optim_steps,
        )

        self.replay_buffer = replay_buffer
        self.async_collection = async_collection

        # --- Wire replay buffer hooks ---
        # LLM collectors created with a replay_buffer write to it directly and
        # yield None; in that case the trainer must not extend the buffer again.
        collector_extends_buffer = getattr(collector, "replay_buffer", None) is not None
        if replay_buffer is not None:
            rb_trainer = ReplayBufferTrainer(
                replay_buffer,
                batch_size=None,
                flatten_tensordicts=False,
                memmap=False,
                device=device,
                # Sync mode iterates over the buffer once per epoch (matching
                # the reference GRPO loop); async mode draws random samples of
                # the buffer's own batch size.
                iterate=not async_collection,
            )
            if not async_collection and not collector_extends_buffer:
                # In sync mode: push collected data into the buffer before each epoch.
                self.register_op("pre_epoch", rb_trainer.extend)
            self.register_op("process_optim_batch", rb_trainer.sample)

        # --- Wire weight-sync sender hook ---
        if weight_sync_sender is not None:
            if async_collection:
                # Push weights every `weight_update_frequency` optimizer steps,
                # in the middle of the optimization loop if needed. This keeps
                # the inference engine close to the training policy, which is
                # essential for meaningful importance sampling in GRPO.
                update_weights = UpdateWeights(
                    update_weights_interval=weight_update_frequency,
                    trainer=self,
                    sender=weight_sync_sender,
                    interval_unit="optim_steps",
                )
            else:
                # Sync mode: push weights once per collected batch, after the
                # optimization epochs have consumed it.
                update_weights = UpdateWeights(
                    trainer=self,
                    sender=weight_sync_sender,
                )
            update_weights.register(self)
        if empty_replay_buffer_on_weight_update and replay_buffer is not None:
            # Sync GRPO: flush the on-policy buffer once its batch has been
            # consumed and the inference weights have been refreshed.
            self.register_op("post_steps", self._empty_replay_buffer)

        # --- Logging hooks ---
        if log_rewards:
            self._setup_reward_logging()
        if log_kl:
            self._setup_kl_logging()

    def _empty_replay_buffer(self) -> None:
        self.replay_buffer.empty(empty_write_count=False)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _setup_reward_logging(self) -> None:
        """Register hooks to log GRPO reward / return statistics."""
        for reduction in ("mean", "max"):
            hook = LogScalar(
                key=("next", "reward"),
                logname=f"reward_{reduction}",
                log_pbar=(reduction == "mean"),
                include_std=(reduction == "mean"),
                reduction=reduction,
            )
            # The collected batch is only available at pre_steps_log when the
            # collector yields real batches. With async collection or a
            # buffer-writing collector the batch is None there, so rewards are
            # logged from the optimization sub-batches instead.
            stage = (
                "post_optim_log"
                if (self.async_collection or self.replay_buffer is not None)
                else "pre_steps_log"
            )
            self.register_op(stage, hook)

    def _setup_kl_logging(self) -> None:
        """Register hooks to log KL divergence keys emitted by GRPOLoss."""
        self.register_op("post_optim_complete_log", self._log_kl_metrics)

    def _log_kl_metrics(
        self, optim_steps: int, average_losses: TensorDictBase | None
    ) -> dict[str, float]:
        """Read KL metrics from the averaged optimization output."""
        del optim_steps
        if average_losses is None:
            return {}
        metrics = {}
        for kl_key in ("kl_to_ref", "kl_to_inference"):
            value = average_losses.get(kl_key, None)
            if value is not None:
                metrics[kl_key] = value.float().mean().item()
        return metrics
