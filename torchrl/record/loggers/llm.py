# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

import torch
from tensordict import TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.record.loggers.common import Logger

if TYPE_CHECKING:
    from torchrl.collectors.llm import LLMCollector
    from torchrl.data.replay_buffers import ReplayBuffer
    from torchrl.objectives.llm.grpo import LLMLossOutput
    from torchrl.objectives.llm.sft import SFTLossOutput


__all__ = ["PostTrainingLogger"]

# Fields inspected via duck-typing. GRPOLossOutput and SFTLossOutput each
# expose a subset; absent or None fields are silently skipped.
_LOSS_FIELDS: tuple[str, ...] = (
    "loss_objective",
    "loss_sft",
    "clip_fraction",
    "kl_approx",
    "ESS",
    "entropy",
    "loss_entropy",
    "loss_kl_to_ref",
    "kl_to_ref",
    "loss_kl_to_inference",
    "kl_to_inference",
)


class PostTrainingLogger:
    """Standardized logger for LLM post-training metrics.

    Loss fields are read via ``getattr`` duck-typing, so this class works
    with :class:`~torchrl.objectives.llm.GRPOLossOutput` and
    :class:`~torchrl.objectives.llm.SFTLossOutput`. Fields that are absent
    or ``None`` are silently skipped.

    Args:
        logger (Logger): Backend logger to emit to.
        start_time (float, optional): ``time.time()`` value captured at the
            start of training for throughput computation.
    """

    def __init__(
        self,
        logger: Logger,
        start_time: float | None = None,
    ) -> None:
        self._logger = logger
        self._start_time = start_time

    def log_training_step(
        self,
        loss: LLMLossOutput | SFTLossOutput,
        step: int,
        *,
        grad_norm: float | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> dict[str, Any]:
        """Log loss components and optimizer state for one gradient step.

        Metric keys follow the ``training/<field>`` convention.

        Args:
            loss (LLMLossOutput | SFTLossOutput): Loss output object.
            step (int): Current global gradient-step counter.

        Keyword Args:
            grad_norm (float, optional): Gradient norm after clipping.
            gradient_accumulation_steps (int): Number of gradient steps per optimizer step.

        Returns:
            dict[str, Any]: The metrics dict that was logged.
        """
        metrics: dict[str, Any] = {}

        for field in _LOSS_FIELDS:
            val = getattr(loss, field, None)
            if val is not None:
                scalar = val.mean() if isinstance(val, torch.Tensor) else val
                metrics[f"training/{field}"] = float(scalar)

        if grad_norm is not None:
            emit_val = (
                float(grad_norm) if step % gradient_accumulation_steps == 0 else 0.0
            )
            metrics["training/grad_norm"] = emit_val

        metrics["training/gradient_steps"] = step
        metrics["training/optim_steps"] = step // gradient_accumulation_steps

        self._logger.log_metrics(metrics, step=step)
        torchrl_logger.debug(f"PostTrainingLogger.log_training_step: {list(metrics)}")
        return metrics

    def log_collection_step(
        self,
        batch: TensorDictBase,
        *,
        replay_buffer: ReplayBuffer | None = None,
        collector: LLMCollector | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Log reward stats, buffer utilization, and policy staleness.

        Args:
            batch (TensorDictBase): The batch sampled from the replay buffer.

        Keyword Args:
            replay_buffer (ReplayBuffer, optional): The active replay buffer.
            collector (LLMCollector, optional): The active collector.
            step (int, optional): Global gradient-step counter.

        Returns:
            dict[str, Any]: The metrics dict that was logged.
        """
        metrics: dict[str, Any] = {}

        with torch.no_grad():
            try:
                reward_list = batch.get(("next", "reward"), default=None, as_list=True)
                if reward_list is not None:
                    reward_tensor = torch.cat(reward_list).float()
                    metrics["buffer/reward_mean"] = float(reward_tensor.mean())
                    metrics["buffer/reward_std"] = float(reward_tensor.std())
                    metrics["buffer/reward_min"] = float(reward_tensor.min())
                    metrics["buffer/reward_max"] = float(reward_tensor.max())
            except Exception as exc:  # noqa: BLE001
                torchrl_logger.debug(
                    f"PostTrainingLogger: could not read reward from batch: {exc}"
                )

            try:
                response_list = batch.get(
                    ("tokens", "response"), default=None, as_list=True
                )
                if response_list is not None:
                    lengths = torch.tensor(
                        [t.numel() for t in response_list], dtype=torch.float
                    )
                    metrics["buffer/seq_length_mean"] = float(lengths.mean())
            except Exception as exc:  # noqa: BLE001
                torchrl_logger.debug(
                    f"PostTrainingLogger: could not read response tokens: {exc}"
                )

            if replay_buffer is not None:
                try:
                    metrics["buffer/write_count"] = int(replay_buffer.write_count)
                    # _storage is private; max_size is a public plain attribute on Storage.
                    storage = replay_buffer._storage  # noqa: SLF001
                    if hasattr(storage, "max_size") and storage.max_size > 0:
                        metrics["buffer/utilization"] = (
                            len(replay_buffer) / storage.max_size
                        )
                except Exception as exc:  # noqa: BLE001
                    torchrl_logger.debug(
                        f"PostTrainingLogger: could not read replay buffer stats: {exc}"
                    )

            if collector is not None and hasattr(collector, "policy_version"):
                try:
                    current_version = int(collector.policy_version)
                    metrics["inference/policy_version"] = current_version

                    version_list = batch.get(
                        ("next", "policy_version"), default=None, as_list=True
                    )
                    if version_list is not None:
                        versions = torch.stack(version_list).float()
                        staleness = current_version - versions
                        metrics["inference/staleness_mean"] = float(staleness.mean())
                        metrics["inference/staleness_max"] = float(staleness.max())
                except Exception as exc:  # noqa: BLE001
                    torchrl_logger.debug(
                        f"PostTrainingLogger: could not compute staleness: {exc}"
                    )

            if self._start_time is not None and step is not None:
                elapsed = time.time() - self._start_time
                if elapsed > 0:
                    metrics["throughput/gradient_steps_per_second"] = float(
                        step / elapsed
                    )

        if metrics:
            self._logger.log_metrics(metrics, step=step)
        torchrl_logger.debug(f"PostTrainingLogger.log_collection_step: {list(metrics)}")
        return metrics

    def log_weight_sync(
        self,
        latency_s: float,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Log weight synchronization latency.

        Args:
            latency_s (float): Time in seconds taken for the weight synchronization.
            step (int, optional): Global gradient-step counter.

        Returns:
            dict[str, Any]: The metrics dict that was logged.
        """
        metrics: dict[str, Any] = {"weight_sync/latency_s": float(latency_s)}
        self._logger.log_metrics(metrics, step=step)
        torchrl_logger.debug(f"PostTrainingLogger.log_weight_sync: {metrics}")
        return metrics
