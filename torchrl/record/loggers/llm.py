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


class PostTrainingLogger:
    """Standardized logger for LLM post-training metrics.

    Loss outputs are :class:`~tensordict.TensorClass` instances
    (:class:`~torchrl.objectives.llm.GRPOLossOutput`,
    :class:`~torchrl.objectives.llm.SFTLossOutput`, ...), so their populated
    tensor fields are discovered by iterating the tensorclass rather than
    through a hardcoded field list: a new loss term added to a loss output is
    logged automatically.

    Args:
        logger (Logger): Backend logger to emit to.
        start_time (float, optional): ``time.time()`` value captured at the
            start of training for throughput computation. (A wall-clock
            timestamp rather than :func:`torchrl.timeit`, which measures code
            blocks, not an anchor shared with the caller's loop.)

    Examples:
        >>> import torch
        >>> from torchrl.objectives.llm.sft import SFTLossOutput
        >>> from torchrl.record.loggers import CSVLogger, PostTrainingLogger
        >>> logger = PostTrainingLogger(CSVLogger("my_exp"))
        >>> metrics = logger.log_training_step(
        ...     SFTLossOutput(loss_sft=torch.tensor(0.5)), step=1, grad_norm=1.2
        ... )
        >>> sorted(metrics)
        ['training/grad_norm', 'training/gradient_steps', 'training/loss_sft', 'training/optim_steps']
    """

    def __init__(
        self,
        logger: Logger,
        start_time: float | None = None,
    ) -> None:
        self._logger = logger
        self._start_time = start_time
        self._warned_sites: set[str] = set()

    def _warn_once(self, site: str, exc: Exception) -> None:
        # An observability component that silently emits nothing is worse than
        # one that raises: surface the first failure per site at WARNING level,
        # then stay quiet so a structural mismatch does not flood the logs.
        if site not in self._warned_sites:
            self._warned_sites.add(site)
            torchrl_logger.warning(
                f"PostTrainingLogger: could not compute {site!r} metrics and "
                f"will not retry the warning ({type(exc).__name__}: {exc}). "
                "The corresponding keys are missing from the logged metrics."
            )

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

        # TensorClass iteration only yields populated fields, so every loss
        # term and diagnostic the loss output carries is logged, including
        # fields added after this logger was written. Plain objects (duck
        # typing) fall back to their attribute dict.
        fields = loss.items() if hasattr(loss, "items") else vars(loss).items()
        for field, val in fields:
            if val is None:
                continue
            scalar = val.mean() if isinstance(val, torch.Tensor) else val
            metrics[f"training/{field}"] = float(scalar)

        # Only emitted on actual optimizer steps: logging a literal 0.0 on
        # accumulation steps would pollute every aggregate over the series.
        if grad_norm is not None and step % gradient_accumulation_steps == 0:
            metrics["training/grad_norm"] = float(grad_norm)

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
                    # as_list=True returns a plain tensor when the batch is
                    # dense (padded) rather than ragged.
                    if isinstance(reward_list, torch.Tensor):
                        reward_tensor = reward_list.reshape(-1).float()
                    else:
                        reward_tensor = torch.cat(reward_list).float()
                    metrics["batch/reward_mean"] = float(reward_tensor.mean())
                    if reward_tensor.numel() > 1:
                        # std of a single element is NaN
                        metrics["batch/reward_std"] = float(reward_tensor.std())
                    metrics["batch/reward_min"] = float(reward_tensor.min())
                    metrics["batch/reward_max"] = float(reward_tensor.max())
            except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
                self._warn_once("batch/reward", exc)

            try:
                response_list = batch.get(
                    ("tokens", "response"), default=None, as_list=True
                )
                if isinstance(response_list, torch.Tensor):
                    # dense (padded) batch: one row per sequence
                    metrics["batch/seq_length_mean"] = float(response_list.shape[-1])
                elif response_list is not None:
                    lengths = torch.tensor(
                        [t.numel() for t in response_list], dtype=torch.float
                    )
                    metrics["batch/seq_length_mean"] = float(lengths.mean())
            except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
                self._warn_once("batch/seq_length", exc)

            if replay_buffer is not None:
                try:
                    metrics["buffer/write_count"] = int(replay_buffer.write_count)
                    # RayReplayBuffer has no _storage; guard with getattr.
                    storage = getattr(replay_buffer, "_storage", None)  # noqa: SLF001
                    if (
                        storage is not None
                        and hasattr(storage, "max_size")
                        and storage.max_size > 0
                    ):
                        metrics["buffer/utilization"] = (
                            len(replay_buffer) / storage.max_size
                        )
                except (AttributeError, RuntimeError, TypeError) as exc:
                    self._warn_once("buffer", exc)

            if collector is not None and hasattr(collector, "policy_version"):
                try:
                    current_version = int(collector.policy_version)
                    metrics["inference/policy_version"] = current_version

                    version_list = batch.get(
                        ("next", "policy_version"), default=None, as_list=True
                    )
                    if isinstance(version_list, torch.Tensor):
                        versions = version_list.float()
                    elif version_list is not None:
                        versions = torch.stack(version_list).float()
                    else:
                        versions = None
                    if versions is not None:
                        staleness = current_version - versions
                        metrics["inference/staleness_mean"] = float(staleness.mean())
                        metrics["inference/staleness_max"] = float(staleness.max())
                except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
                    self._warn_once("inference/staleness", exc)

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
