# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
import multiprocessing
import time
from collections import OrderedDict
from numbers import Real
from typing import Any

import torch
from tensordict import NestedKey, TensorDictBase, unravel_key

from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.utils import INT_CLASSES


class ReplayFlowControl:
    """Coordinates replay-ratio pacing and policy publication.

    ``ReplayFlowControl`` is a learner-side coordinator for continuously
    overwritten replay buffers. It limits the cumulative number of sampled
    transitions to ``samples_per_insert * write_count`` and can prevent a
    caller from publishing a policy whose version is too far ahead of the
    versions in the latest sampled batch. Physical buffer occupancy is not
    used as a pressure signal.

    With ``max_policy_lag`` enabled, publication is disabled until at least one
    controlled sample establishes a version baseline. The gate compares a
    candidate against the oldest version in the latest batch, so every record
    in that batch must satisfy the configured lag bound.

    The coordinator does not change replay-buffer behavior. All controlled
    learner samples must go through :meth:`sample`; direct calls to the replay
    buffer are still possible but count against the same cumulative sample
    budget. Producer admission and stale-entry filtering remain separate
    concerns.

    Args:
        replay_buffer (ReplayBuffer): replay buffer whose cumulative counters
            and readiness condition are used for pacing. Create the coordinator
            after calling ``replay_buffer.share(True)`` when it will be used
            across processes.
        samples_per_insert (float): maximum cumulative number of sampled
            transitions per inserted transition. Must be finite and positive.
        max_policy_lag (int, optional): maximum allowed difference between a
            candidate publication version and the oldest policy version in the
            latest sampled batch. Publication is disabled until a sample has
            established this baseline. ``None`` disables publication gating.
        policy_version_key (NestedKey, optional): TensorDict key containing the
            behavior-policy version. Defaults to
            ``("next", "policy_version")``.
        initial_policy_version (int, optional): initially published policy
            version. Defaults to ``0``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import (
        ...     LazyTensorStorage,
        ...     ReplayFlowControl,
        ...     TensorDictReplayBuffer,
        ... )
        >>> replay = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(8), batch_size=2
        ... )
        >>> _ = replay.extend(TensorDict({
        ...     "value": torch.arange(2),
        ...     ("next", "policy_version"): torch.zeros(2, dtype=torch.long),
        ... }, batch_size=[2]))
        >>> control = ReplayFlowControl(
        ...     replay, samples_per_insert=1.0, max_policy_lag=1
        ... )
        >>> batch = control.sample(timeout=1.0)
        >>> control.can_publish(1)
        True

    .. note::
        ``ReplayFlowControl`` controls a cumulative sample-to-insert ratio, not
        instantaneous throughput. Checkpoint the replay buffer and the
        coordinator together so their cumulative counters stay aligned. The
        coordinator checkpoint does not contain the replay buffer's
        ``write_count`` or ``samples_returned`` values.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        *,
        samples_per_insert: float,
        max_policy_lag: int | None = None,
        policy_version_key: NestedKey = ("next", "policy_version"),
        initial_policy_version: int = 0,
    ):
        if not isinstance(replay_buffer, ReplayBuffer):
            raise TypeError("replay_buffer must be a ReplayBuffer instance.")
        if isinstance(samples_per_insert, bool) or not isinstance(
            samples_per_insert, Real
        ):
            raise TypeError("samples_per_insert must be a finite positive number.")
        samples_per_insert = float(samples_per_insert)
        if not math.isfinite(samples_per_insert) or samples_per_insert <= 0:
            raise ValueError("samples_per_insert must be a finite positive number.")
        self._validate_policy_lag(max_policy_lag)
        self._validate_policy_version(initial_policy_version)

        self.replay_buffer = replay_buffer
        self.samples_per_insert = samples_per_insert
        self.max_policy_lag = max_policy_lag
        self.policy_version_key = unravel_key(policy_version_key)

        shared = replay_buffer.shared
        if shared:
            self._reserved_samples = multiprocessing.Value("q", 0, lock=False)
            self._sample_wait_count = multiprocessing.Value("q", 0, lock=False)
            self._sample_wait_time = multiprocessing.Value("d", 0.0, lock=False)
            self._published_policy_version = multiprocessing.Value(
                "q", initial_policy_version, lock=False
            )
            self._policy_publication_count = multiprocessing.Value("q", 0, lock=False)
            self._latest_sample_min_version = multiprocessing.Value("q", 0, lock=False)
            self._latest_sample_max_version = multiprocessing.Value("q", 0, lock=False)
            self._policy_lag_count = multiprocessing.Value("q", 0, lock=False)
            self._policy_lag_sum = multiprocessing.Value("q", 0, lock=False)
            self._policy_lag_min = multiprocessing.Value("q", 0, lock=False)
            self._policy_lag_max = multiprocessing.Value("q", 0, lock=False)
            self._shutdown = multiprocessing.Value("b", False, lock=False)
        else:
            self._reserved_samples = 0
            self._sample_wait_count = 0
            self._sample_wait_time = 0.0
            self._published_policy_version = initial_policy_version
            self._policy_publication_count = 0
            self._latest_sample_min_version = 0
            self._latest_sample_max_version = 0
            self._policy_lag_count = 0
            self._policy_lag_sum = 0
            self._policy_lag_min = 0
            self._policy_lag_max = 0
            self._shutdown = False

    @staticmethod
    def _validate_policy_lag(max_policy_lag: int | None) -> None:
        if max_policy_lag is None:
            return
        if isinstance(max_policy_lag, bool) or not isinstance(
            max_policy_lag, INT_CLASSES
        ):
            raise TypeError("max_policy_lag must be a non-negative integer or None.")
        if max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative.")

    @staticmethod
    def _validate_policy_version(policy_version: int) -> None:
        if isinstance(policy_version, bool) or not isinstance(
            policy_version, INT_CLASSES
        ):
            raise TypeError("policy_version must be a non-negative integer.")
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative.")

    @staticmethod
    def _value(value: int | float | Any) -> int | float | bool:
        if hasattr(value, "value"):
            return value.value
        return value

    def _get(self, name: str) -> int | float | bool:
        return self._value(getattr(self, name))

    def _set(self, name: str, value: int | float | bool) -> None:
        current = getattr(self, name)
        if hasattr(current, "value"):
            current.value = value
        else:
            setattr(self, name, value)

    def _increment(self, name: str, value: int | float) -> None:
        self._set(name, self._get(name) + value)

    def _resolve_batch_size(self, batch_size: int | None) -> int:
        if batch_size is None:
            batch_size = self.replay_buffer.batch_size
        if batch_size is None:
            raise RuntimeError(
                "batch_size not specified. Configure it on the replay buffer or "
                "pass it to ReplayFlowControl.sample()."
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, INT_CLASSES):
            raise TypeError("batch_size must be a positive integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        return int(batch_size)

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        if isinstance(timeout, bool) or not isinstance(timeout, Real):
            raise TypeError("timeout must be a non-negative number or None.")
        if timeout < 0:
            raise ValueError("timeout must be non-negative.")
        return time.monotonic() + float(timeout)

    def _sample_budget(self, replay_stats: dict[str, Any]) -> int:
        target_samples = math.floor(
            replay_stats["write_count"] * self.samples_per_insert
        )
        return (
            target_samples
            - replay_stats["samples_returned"]
            - int(self._get("_reserved_samples"))
        )

    def _sample_decision(self, batch_size: int) -> str:
        if self._get("_shutdown") or getattr(
            self.replay_buffer, "_service_shutdown", False
        ):
            return "shutdown"
        if not self.replay_buffer.can_sample(batch_size):
            return "wait_for_replay"
        if self._sample_budget(self.replay_buffer.stats()) < batch_size:
            return "wait_for_inserts"
        return "sample"

    def can_sample(self, batch_size: int | None = None) -> bool:
        """Returns whether replay readiness and the ratio budget allow a sample."""
        batch_size = self._resolve_batch_size(batch_size)
        condition = self.replay_buffer._readiness_condition
        with condition:
            return self._sample_decision(batch_size) == "sample"

    def _finish_wait(self, started_at: float | None) -> None:
        if started_at is not None:
            self._increment("_sample_wait_time", time.monotonic() - started_at)

    def _reserve_sample(
        self,
        batch_size: int,
        *,
        deadline: float | None,
        cancel_event: Any | None,
    ) -> None:
        condition = self.replay_buffer._readiness_condition
        started_at = None
        with condition:
            while True:
                decision = self._sample_decision(batch_size)
                if decision == "shutdown":
                    self._finish_wait(started_at)
                    raise RuntimeError(
                        "A shut down replay flow controller cannot sample."
                    )
                if cancel_event is not None and cancel_event.is_set():
                    self._finish_wait(started_at)
                    raise RuntimeError("Replay-flow-controlled sampling was cancelled.")
                if decision == "sample":
                    self._increment("_reserved_samples", batch_size)
                    self._finish_wait(started_at)
                    return
                if started_at is None:
                    started_at = time.monotonic()
                    self._increment("_sample_wait_count", 1)
                wait_time = None
                if deadline is not None:
                    wait_time = deadline - time.monotonic()
                    if wait_time <= 0:
                        self._finish_wait(started_at)
                        raise TimeoutError(
                            "Replay flow control did not permit sampling within "
                            "the requested timeout."
                        )
                if cancel_event is not None:
                    wait_time = 0.1 if wait_time is None else min(wait_time, 0.1)
                condition.wait(wait_time)

    def _record_policy_versions(self, data: Any) -> None:
        if self.max_policy_lag is None:
            return
        if not isinstance(data, TensorDictBase):
            raise TypeError(
                "max_policy_lag requires TensorDict replay samples containing "
                f"{self.policy_version_key!r}."
            )
        versions = data.get(self.policy_version_key, None)
        if versions is None:
            raise KeyError(
                f"Could not find policy-version key {self.policy_version_key!r} "
                "in the sampled TensorDict."
            )
        if not isinstance(versions, torch.Tensor):
            raise TypeError(
                "Policy versions must use an integer tensor; UUID and other "
                "non-tensor policy versions are not supported."
            )
        if versions.numel() == 0:
            raise RuntimeError("The sampled policy-version tensor is empty.")
        if versions.is_floating_point() or versions.is_complex():
            raise TypeError("Policy versions must use an integer dtype.")

        versions = versions.to(dtype=torch.int64)
        batch_min = int(versions.min().item())
        batch_max = int(versions.max().item())
        version_sum = int(versions.sum().item())
        version_count = versions.numel()
        published_version = int(self._get("_published_policy_version"))
        batch_lag_min = published_version - batch_max
        batch_lag_max = published_version - batch_min
        lag_sum = published_version * version_count - version_sum

        if int(self._get("_policy_lag_count")) == 0:
            self._set("_policy_lag_min", batch_lag_min)
            self._set("_policy_lag_max", batch_lag_max)
        else:
            self._set(
                "_policy_lag_min",
                min(int(self._get("_policy_lag_min")), batch_lag_min),
            )
            self._set(
                "_policy_lag_max",
                max(int(self._get("_policy_lag_max")), batch_lag_max),
            )
        self._set("_latest_sample_min_version", batch_min)
        self._set("_latest_sample_max_version", batch_max)
        self._increment("_policy_lag_count", version_count)
        self._increment("_policy_lag_sum", lag_sum)

    def sample(
        self,
        batch_size: int | None = None,
        return_info: bool = False,
        *,
        timeout: float | None = None,
        cancel_event: Any | None = None,
    ) -> Any:
        """Samples after replay readiness and ratio budget permit the request.

        Args:
            batch_size (int, optional): requested number of transitions.
                Defaults to the replay buffer batch size.
            return_info (bool, optional): if ``True``, returns ``(data, info)``.
                Defaults to ``False``.
            timeout (float, optional): maximum seconds to wait for replay data
                and ratio budget. ``None`` waits indefinitely.
            cancel_event (optional): event-like object exposing ``is_set()``.

        Returns:
            The sampled batch, or ``(batch, info)`` when ``return_info=True``.

        Raises:
            TimeoutError: if the timeout expires before sampling is permitted.
            RuntimeError: if cancelled or shut down while waiting.
        """
        batch_size = self._resolve_batch_size(batch_size)
        deadline = self._deadline(timeout)
        self._reserve_sample(batch_size, deadline=deadline, cancel_event=cancel_event)
        condition = self.replay_buffer._readiness_condition
        try:
            remaining = None
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
            data, info = self.replay_buffer.sample(
                batch_size,
                return_info=True,
                wait=True,
                timeout=remaining,
                cancel_event=cancel_event,
            )
            with condition:
                self._record_policy_versions(data)
        finally:
            with condition:
                # Release the full reservation even when a sampler returns a
                # short batch. The replay buffer counts only returned records,
                # leaving the unspent ratio budget available to another call.
                self._increment("_reserved_samples", -batch_size)
                condition.notify_all()
        if return_info:
            return data, info
        return data

    def can_publish(self, policy_version: int) -> bool:
        """Returns whether a candidate satisfies the latest batch's oldest version."""
        self._validate_policy_version(policy_version)
        condition = self.replay_buffer._readiness_condition
        with condition:
            if self._get("_shutdown") or getattr(
                self.replay_buffer, "_service_shutdown", False
            ):
                return False
            if self.max_policy_lag is None:
                return True
            if int(self._get("_policy_lag_count")) == 0:
                return False
            latest_min = int(self._get("_latest_sample_min_version"))
            return policy_version - latest_min <= self.max_policy_lag

    def record_policy_publication(self, policy_version: int) -> None:
        """Records a policy publication after checking monotonicity and lag."""
        self._validate_policy_version(policy_version)
        condition = self.replay_buffer._readiness_condition
        with condition:
            current = int(self._get("_published_policy_version"))
            if policy_version < current:
                raise ValueError(
                    "policy_version must not move backwards: "
                    f"got {policy_version} after {current}."
                )
            if policy_version == current:
                return
            if (
                self.max_policy_lag is not None
                and int(self._get("_policy_lag_count")) == 0
            ):
                raise RuntimeError(
                    "Policy publication with max_policy_lag requires at least one "
                    "controlled replay sample to establish a version baseline."
                )
            if not self.can_publish(policy_version):
                raise RuntimeError(
                    f"Policy version {policy_version} exceeds max_policy_lag="
                    f"{self.max_policy_lag} for the latest sampled batch."
                )
            self._set("_published_policy_version", policy_version)
            self._increment("_policy_publication_count", 1)
            condition.notify_all()

    def stats(self) -> dict[str, int | float | bool | str | None]:
        """Returns a cheap snapshot of replay-ratio and policy-lag state."""
        condition = self.replay_buffer._readiness_condition
        with condition:
            replay_stats = self.replay_buffer.stats()
            write_count = int(replay_stats["write_count"])
            samples_returned = int(replay_stats["samples_returned"])
            sample_ratio = samples_returned / write_count if write_count else 0.0
            sample_budget = self._sample_budget(replay_stats)
            policy_lag_count = int(self._get("_policy_lag_count"))
            if policy_lag_count:
                policy_lag_mean = float(self._get("_policy_lag_sum")) / policy_lag_count
                policy_lag_min = int(self._get("_policy_lag_min"))
                policy_lag_max = int(self._get("_policy_lag_max"))
                latest_sample_min_version = int(self._get("_latest_sample_min_version"))
                latest_sample_max_version = int(self._get("_latest_sample_max_version"))
                latest_policy_lag = (
                    int(self._get("_published_policy_version"))
                    - latest_sample_min_version
                )
            else:
                policy_lag_mean = None
                policy_lag_min = None
                policy_lag_max = None
                latest_sample_min_version = None
                latest_sample_max_version = None
                latest_policy_lag = None
            try:
                batch_size = self._resolve_batch_size(None)
            except RuntimeError:
                sample_decision = "batch_size_required"
            else:
                sample_decision = self._sample_decision(batch_size)
            published_policy_version = int(self._get("_published_policy_version"))
            if self.max_policy_lag is None:
                publication_decision = "publish"
            elif not policy_lag_count:
                publication_decision = "wait_for_sample"
            elif self.can_publish(published_policy_version + 1):
                publication_decision = "publish"
            else:
                publication_decision = "hold_for_policy_lag"
            if sample_decision != "sample":
                control_decision = sample_decision
            else:
                control_decision = publication_decision

            return {
                "control_decision": control_decision,
                "sample_decision": sample_decision,
                "publication_decision": publication_decision,
                "target_samples_per_insert": self.samples_per_insert,
                "write_count": write_count,
                "samples_returned": samples_returned,
                "samples_per_insert": sample_ratio,
                "sample_budget": sample_budget,
                "reserved_samples": int(self._get("_reserved_samples")),
                "sample_wait_count": int(self._get("_sample_wait_count")),
                "sample_wait_time": float(self._get("_sample_wait_time")),
                "max_policy_lag": self.max_policy_lag,
                "published_policy_version": published_policy_version,
                "policy_publication_count": int(self._get("_policy_publication_count")),
                "latest_sample_min_policy_version": latest_sample_min_version,
                "latest_sample_max_policy_version": latest_sample_max_version,
                "latest_policy_lag": latest_policy_lag,
                "policy_lag_count": policy_lag_count,
                "policy_lag_mean": policy_lag_mean,
                "policy_lag_min": policy_lag_min,
                "policy_lag_max": policy_lag_max,
                "shutdown": bool(self._get("_shutdown")),
            }

    def state_dict(self) -> dict[str, Any]:
        """Returns checkpoint state for the controller."""
        condition = self.replay_buffer._readiness_condition
        with condition:
            return OrderedDict(
                samples_per_insert=self.samples_per_insert,
                max_policy_lag=self.max_policy_lag,
                policy_version_key=self.policy_version_key,
                sample_wait_count=int(self._get("_sample_wait_count")),
                sample_wait_time=float(self._get("_sample_wait_time")),
                published_policy_version=int(self._get("_published_policy_version")),
                policy_publication_count=int(self._get("_policy_publication_count")),
                latest_sample_min_version=int(self._get("_latest_sample_min_version")),
                latest_sample_max_version=int(self._get("_latest_sample_max_version")),
                policy_lag_count=int(self._get("_policy_lag_count")),
                policy_lag_sum=int(self._get("_policy_lag_sum")),
                policy_lag_min=int(self._get("_policy_lag_min")),
                policy_lag_max=int(self._get("_policy_lag_max")),
            )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores checkpointed controller configuration and counters."""
        samples_per_insert = state_dict["samples_per_insert"]
        if isinstance(samples_per_insert, bool) or not isinstance(
            samples_per_insert, Real
        ):
            raise TypeError("samples_per_insert must be a finite positive number.")
        samples_per_insert = float(samples_per_insert)
        if not math.isfinite(samples_per_insert) or samples_per_insert <= 0:
            raise ValueError("samples_per_insert must be a finite positive number.")
        max_policy_lag = state_dict["max_policy_lag"]
        self._validate_policy_lag(max_policy_lag)

        condition = self.replay_buffer._readiness_condition
        with condition:
            self.samples_per_insert = samples_per_insert
            self.max_policy_lag = max_policy_lag
            self.policy_version_key = unravel_key(state_dict["policy_version_key"])
            self._set("_reserved_samples", 0)
            self._set("_shutdown", False)
            self._set("_sample_wait_count", state_dict["sample_wait_count"])
            self._set("_sample_wait_time", state_dict["sample_wait_time"])
            self._set(
                "_published_policy_version", state_dict["published_policy_version"]
            )
            self._set(
                "_policy_publication_count",
                state_dict["policy_publication_count"],
            )
            self._set(
                "_latest_sample_min_version",
                state_dict["latest_sample_min_version"],
            )
            self._set(
                "_latest_sample_max_version",
                state_dict["latest_sample_max_version"],
            )
            self._set("_policy_lag_count", state_dict["policy_lag_count"])
            self._set("_policy_lag_sum", state_dict["policy_lag_sum"])
            self._set("_policy_lag_min", state_dict["policy_lag_min"])
            self._set("_policy_lag_max", state_dict["policy_lag_max"])
            condition.notify_all()

    def shutdown(self) -> None:
        """Stops the controller and wakes all blocked sampling calls."""
        condition = self.replay_buffer._readiness_condition
        with condition:
            self._set("_shutdown", True)
            condition.notify_all()
