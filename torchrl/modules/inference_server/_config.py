# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torchrl._utils import _make_ordinal_device


def _as_device(device: torch.device | str | int | None) -> torch.device | None:
    if device is None:
        return None
    return torch.device(device)


@dataclass
class InferenceDeviceConfig:
    """Device placement for asynchronous policy-server collection.

    This config separates the devices used by the environment, the remote
    policy, the actor-side action TensorDict, and the returned collector batch.

    All fields accept :class:`torch.device`, ``str``, or ``None`` and are
    normalized to ``torch.device | None`` at construction time.

    Args:
        policy_device (torch.device or str, optional): device that owns the
            policy and receives batched server inputs.
        output_device (torch.device or str, optional): device for inference
            results returned by the server.
        env_device (torch.device or str, optional): device used by env workers
            when stepping environments. If ``output_device`` is omitted, this is
            the natural device for returned actions.
        storing_device (torch.device or str, optional): device used for
            collected transitions yielded by the collector.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import (
        ...     InferenceDeviceConfig,
        ...     InferenceServer,
        ...     ThreadingTransport,
        ... )
        >>> policy = TensorDictModule(
        ...     nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ... )
        >>> transport = ThreadingTransport()
        >>> device_config = InferenceDeviceConfig(
        ...     policy_device="cpu", output_device="cpu"
        ... )
        >>> with InferenceServer(policy, transport, device_config=device_config):
        ...     client = transport.client()
        ...     result = client(TensorDict({"observation": torch.randn(4)}))
        >>> result["action"].device.type
        'cpu'
    """

    policy_device: torch.device | str | None = None
    output_device: torch.device | str | None = None
    env_device: torch.device | str | None = None
    storing_device: torch.device | str | None = None

    def __post_init__(self) -> None:
        self.policy_device = _as_device(self.policy_device)
        self.output_device = _as_device(self.output_device)
        self.env_device = _as_device(self.env_device)
        self.storing_device = _as_device(self.storing_device)

    def server_output_device(self) -> torch.device | None:
        """Return the actor-side device expected from the policy server."""
        if self.output_device is not None:
            return self.output_device
        return self.env_device


def _resolve_device_config(
    device_config: InferenceDeviceConfig | None = None,
    *,
    device: torch.device | str | int | None = None,
    policy_device: torch.device | str | int | None = None,
    output_device: torch.device | str | int | None = None,
    env_device: torch.device | str | int | None = None,
    storing_device: torch.device | str | int | None = None,
    allow_storing_device: bool = True,
    collector_defaults: bool = False,
) -> InferenceDeviceConfig:
    """Resolve loose device kwargs and/or a device config into a single config.

    This is the single source of truth for device-precedence rules shared by
    :class:`~torchrl.modules.inference_server.InferenceServer`,
    :class:`~torchrl.modules.inference_server.ProcessInferenceServer`,
    :class:`~torchrl.collectors.AsyncBatchedCollector` and the regular
    collectors (see https://github.com/pytorch/rl/issues/3943). The rules are:

    - ``device_config`` is mutually exclusive with every loose device kwarg.
    - ``device`` is an alias/default for ``policy_device``.
    - ``output_device`` falls back to the explicitly-provided ``env_device``
      (the natural device for actions returned to env workers).
    - With ``collector_defaults=True`` (regular-collector semantics),
      ``device`` also fills unset ``env_device`` and ``storing_device``,
      devices are ordinalized (e.g. ``"cuda"`` -> ``"cuda:0"``), and an unset
      ``storing_device`` falls back to the shared env/policy device when the
      two coincide.

    Args:
        device_config (InferenceDeviceConfig, optional): pre-built device
            config. Mutually exclusive with every other device argument.

    Keyword Args:
        device (torch.device, str or int, optional): generic device, used as
            an alias for ``policy_device`` (and, with
            ``collector_defaults=True``, as a default for ``env_device`` and
            ``storing_device``).
        policy_device (torch.device, str or int, optional): device that owns
            the policy.
        output_device (torch.device, str or int, optional): device for
            inference results returned by a policy server.
        env_device (torch.device, str or int, optional): device used when
            stepping environments.
        storing_device (torch.device, str or int, optional): device for
            collected transitions.
        allow_storing_device (bool, optional): when ``False``, a non-``None``
            ``storing_device`` is rejected (policy servers do not consume it).
            Defaults to ``True``.
        collector_defaults (bool, optional): enable the regular-collector
            fallbacks described above. Defaults to ``False``.

    Returns:
        InferenceDeviceConfig: a config with all precedence rules applied and
        every field normalized to ``torch.device | None``.
    """
    if device_config is not None:
        explicit = [
            name
            for name, value in (
                ("device", device),
                ("policy_device", policy_device),
                ("output_device", output_device),
                ("env_device", env_device),
                ("storing_device", storing_device),
            )
            if value is not None
        ]
        if explicit:
            raise ValueError(
                "device_config is mutually exclusive with the explicit device "
                f"keyword arguments (got {', '.join(explicit)})."
            )
    else:
        device_config = InferenceDeviceConfig(
            policy_device=policy_device,
            output_device=output_device,
            env_device=env_device,
            storing_device=storing_device,
        )
    if not allow_storing_device and device_config.storing_device is not None:
        raise ValueError(
            "storing_device is a collector-level setting that the "
            "server does not consume. The server only uses "
            "policy_device and output_device (with env_device as a "
            "fallback for output_device). Pass storing_device to the "
            "collector instead."
        )
    device = _as_device(device)
    policy_device = device_config.policy_device
    output_device = device_config.server_output_device()
    env_device = device_config.env_device
    storing_device = device_config.storing_device
    if policy_device is None:
        policy_device = device
    if collector_defaults:
        if env_device is None:
            env_device = device
        if storing_device is None:
            storing_device = device
        policy_device = _make_ordinal_device(policy_device)
        output_device = _make_ordinal_device(output_device)
        env_device = _make_ordinal_device(env_device)
        storing_device = _make_ordinal_device(storing_device)
        if storing_device is None and env_device == policy_device:
            storing_device = env_device
    return InferenceDeviceConfig(
        policy_device=policy_device,
        output_device=output_device,
        env_device=env_device,
        storing_device=storing_device,
    )


@dataclass
class InferenceServerConfig:
    """Server-side execution, batching, timeout, and instrumentation settings.

    Args:
        service_backend (str, optional): execution backend for the policy server.
            ``"thread"`` runs the serve loop in a background thread of the
            constructing process; ``"process"`` runs a dedicated server
            process (which requires a picklable ``policy_factory`` and a
            multiprocessing-capable transport). Defaults to ``"thread"``.
        max_batch_size (int, optional): maximum number of requests per forward
            pass. Defaults to ``64``.
        min_batch_size (int, optional): minimum number of requests to
            accumulate after the first request arrives. Defaults to ``1``.
        timeout (float, optional): seconds to wait for more requests before
            flushing a partial batch. Defaults to ``0.01``.
        collect_stats (bool, optional): whether to collect lightweight
            throughput and latency stats. Defaults to ``True``.
        stats_window_size (int, optional): number of recent timing samples kept
            for percentile stats. Defaults to ``1024``.
        max_inflight_per_env (int, optional): maximum unresolved remote-policy
            requests each environment coordinator may have inflight (consumed
            by :class:`~torchrl.collectors.AsyncBatchedCollector` when
            building its clients). Defaults to ``None`` (unbounded), so the
            guard never throttles by surprise; set an explicit bound when
            backpressure is wanted.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import (
        ...     InferenceServer,
        ...     InferenceServerConfig,
        ...     ThreadingTransport,
        ... )
        >>> policy = TensorDictModule(
        ...     nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ... )
        >>> transport = ThreadingTransport()
        >>> config = InferenceServerConfig(max_batch_size=8, timeout=0.001)
        >>> with InferenceServer(policy, transport, server_config=config) as server:
        ...     client = transport.client()
        ...     result = client(TensorDict({"observation": torch.randn(4)}))
        >>> result["action"].shape
        torch.Size([2])
        >>> server.max_batch_size
        8
    """

    service_backend: Literal["thread", "process"] = "thread"
    max_batch_size: int = 64
    min_batch_size: int = 1
    timeout: float = 0.01
    collect_stats: bool = True
    stats_window_size: int = 1024
    max_inflight_per_env: int | None = None

    def __post_init__(self) -> None:
        if self.service_backend not in ("thread", "process"):
            raise ValueError(
                f"service_backend={self.service_backend!r} is not supported. "
                "Expected 'thread' or 'process'."
            )
        if self.max_inflight_per_env is not None and self.max_inflight_per_env < 1:
            raise ValueError(
                f"max_inflight_per_env must be at least 1 (got "
                f"{self.max_inflight_per_env}); use None to disable the guard."
            )
