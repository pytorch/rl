from __future__ import annotations

import abc
import contextlib
import functools
import typing
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.base import NO_DEFAULT
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torch import nn as nn
from torch.utils.data import IterableDataset
from torchrl.collectors.utils import _map_weight

from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.weight_update.utils import _resolve_attr
from torchrl.weight_update.weight_sync_schemes import WeightSyncScheme


@dataclass
class ProfileConfig:
    """Configuration for profiling collector workers.

    This class holds all the settings for profiling collector rollouts
    using PyTorch's profiler. It's designed to work across all collector types.

    Attributes:
        workers: List of worker indices to profile. For single-process collectors
            (like Collector), this is ignored. For multi-process collectors
            (like MultiSyncCollector, MultiAsyncCollector), only the specified
            workers will be profiled. Defaults to [0].
        num_rollouts: Total number of rollouts to profile (including warmup).
            After this many rollouts, profiling stops. Defaults to 3.
        warmup_rollouts: Number of rollouts to skip before starting actual
            profiling. This allows JIT/compile warmup. Defaults to 1.
        save_path: Path to save the profiling trace. If None, traces are saved
            to "./collector_profile_{worker_idx}.json". Supports {worker_idx}
            placeholder for worker-specific files.
        activities: List of profiler activities. Defaults to CPU and CUDA.
        record_shapes: Whether to record tensor shapes. Defaults to True.
        profile_memory: Whether to profile memory usage. Defaults to False.
        with_stack: Whether to record stack traces. Defaults to True.
        with_flops: Whether to compute FLOPS. Defaults to False.
        on_trace_ready: Optional callback when trace is ready. If None,
            traces are exported to Chrome trace format at save_path.

    Example:
        >>> from torchrl.collectors import MultiSyncCollector, ProfileConfig
        >>> collector = MultiSyncCollector(...)
        >>> collector.enable_profile(
        ...     workers=[0],
        ...     num_rollouts=5,
        ...     warmup_rollouts=2,
        ...     save_path="./traces/worker_{worker_idx}.json",
        ... )
        >>> for data in collector:
        ...     # First worker will be profiled for rollouts 2-4
        ...     process(data)
    """

    workers: list[int] = field(default_factory=lambda: [0])
    num_rollouts: int = 3
    warmup_rollouts: int = 1
    save_path: str | Path | None = None
    activities: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = True
    with_flops: bool = False
    on_trace_ready: Callable | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_rollouts <= self.warmup_rollouts:
            raise ValueError(
                f"num_rollouts ({self.num_rollouts}) must be greater than "
                f"warmup_rollouts ({self.warmup_rollouts})"
            )
        if self.warmup_rollouts < 0:
            raise ValueError(
                f"warmup_rollouts must be >= 0, got {self.warmup_rollouts}"
            )

    def get_save_path(self, worker_idx: int) -> Path:
        """Get the save path for a specific worker.

        Args:
            worker_idx: The worker index.

        Returns:
            Path object for the trace file.
        """
        if self.save_path is None:
            return Path(f"./collector_profile_{worker_idx}.json")
        path_str = str(self.save_path).format(worker_idx=worker_idx)
        return Path(path_str)

    def get_activities(self) -> list:
        """Get PyTorch profiler activity list.

        Returns:
            List of torch.profiler.ProfilerActivity values.
        """
        import torch.profiler

        activity_map = {
            "cpu": torch.profiler.ProfilerActivity.CPU,
            "cuda": torch.profiler.ProfilerActivity.CUDA,
        }
        result = []
        for activity in self.activities:
            activity_lower = activity.lower()
            if activity_lower in activity_map:
                # Only add CUDA if CUDA is available
                if activity_lower == "cuda" and not torch.cuda.is_available():
                    continue
                result.append(activity_map[activity_lower])
        return result

    def should_profile_worker(self, worker_idx: int) -> bool:
        """Check if a specific worker should be profiled.

        Args:
            worker_idx: The worker index to check.

        Returns:
            True if this worker should be profiled.
        """
        return worker_idx in self.workers


class BaseCollector(IterableDataset, metaclass=abc.ABCMeta):
    """Base class for data collectors."""

    _task = None
    _iterator = None
    _iteration_started = False
    total_frames: int
    requested_frames_per_batch: int
    frames_per_batch: int
    trust_policy: bool
    compiled_policy: bool
    cudagraphed_policy: bool
    _weight_updater: WeightUpdaterBase | None = None
    _weight_sync_schemes: dict[str, WeightSyncScheme] | None = None
    verbose: bool = False
    _profile_config: ProfileConfig | None = None

    def enable_profile(
        self,
        *,
        workers: list[int] | None = None,
        num_rollouts: int = 3,
        warmup_rollouts: int = 1,
        save_path: str | Path | None = None,
        activities: list[str] | None = None,
        record_shapes: bool = True,
        profile_memory: bool = False,
        with_stack: bool = True,
        with_flops: bool = False,
        on_trace_ready: Callable | None = None,
    ) -> None:
        """Enable profiling for collector worker rollouts.

        This method configures the collector to profile rollouts using PyTorch's
        profiler. For multi-process collectors, profiling happens in the worker
        processes. For single-process collectors (Collector), profiling happens
        in the main process.

        Args:
            workers: List of worker indices to profile. Defaults to [0].
                For single-process collectors, this is ignored.
            num_rollouts: Total number of rollouts to run the profiler for
                (including warmup). Profiling stops after this many rollouts.
                Defaults to 3.
            warmup_rollouts: Number of rollouts to skip before starting actual
                profiling. Useful for JIT/compile warmup. The profiler runs
                but discards data during warmup. Defaults to 1.
            save_path: Path to save the profiling trace. Supports {worker_idx}
                placeholder for worker-specific files. If None, traces are
                saved to "./collector_profile_{worker_idx}.json".
            activities: List of profiler activities ("cpu", "cuda").
                Defaults to ["cpu", "cuda"].
            record_shapes: Whether to record tensor shapes. Defaults to True.
            profile_memory: Whether to profile memory usage. Defaults to False.
            with_stack: Whether to record Python stack traces. Defaults to True.
            with_flops: Whether to compute FLOPS. Defaults to False.
            on_trace_ready: Optional callback when trace is ready. If None,
                traces are exported to Chrome trace format at save_path.

        Raises:
            RuntimeError: If called after iteration has started.
            ValueError: If num_rollouts <= warmup_rollouts.

        Example:
            >>> from torchrl.collectors import MultiSyncCollector
            >>> collector = MultiSyncCollector(
            ...     create_env_fn=[make_env] * 4,
            ...     policy=policy,
            ...     frames_per_batch=1000,
            ...     total_frames=100000,
            ... )
            >>> collector.enable_profile(
            ...     workers=[0],
            ...     num_rollouts=5,
            ...     warmup_rollouts=2,
            ...     save_path="./traces/worker_{worker_idx}.json",
            ... )
            >>> # Worker 0 will be profiled for rollouts 2, 3, 4
            >>> for data in collector:
            ...     train(data)
            >>> collector.shutdown()

        Note:
            - Profiling adds overhead, so only profile specific workers
            - The trace file can be viewed in Chrome's trace viewer
              (chrome://tracing) or with PyTorch's TensorBoard plugin
            - For multi-process collectors, this must be called BEFORE
              iteration starts as it needs to configure workers
        """
        if self._iteration_started:
            raise RuntimeError(
                "Cannot enable profiling after iteration has started. "
                "Call enable_profile() before iterating over the collector."
            )

        if workers is None:
            workers = [0]
        if activities is None:
            activities = ["cpu", "cuda"]

        self._profile_config = ProfileConfig(
            workers=workers,
            num_rollouts=num_rollouts,
            warmup_rollouts=warmup_rollouts,
            save_path=save_path,
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            on_trace_ready=on_trace_ready,
        )

    @property
    def profile_config(self) -> ProfileConfig | None:
        """Get the profiling configuration.

        Returns:
            ProfileConfig if profiling is enabled, None otherwise.
        """
        return self._profile_config

    @property
    def weight_updater(self) -> WeightUpdaterBase:
        return self._weight_updater

    @weight_updater.setter
    def weight_updater(self, value: WeightUpdaterBase | None):
        if value is not None:
            if not isinstance(value, WeightUpdaterBase) and callable(
                value
            ):  # Fall back to default constructor
                value = value()
            value.register_collector(self)
            if value.collector is not self:
                raise RuntimeError("Failed to register collector.")
        self._weight_updater = value

    @property
    def worker_idx(self) -> int | None:
        """Get the worker index for this collector.

        Returns:
            The worker index (0-indexed).

        Raises:
            RuntimeError: If worker_idx has not been set.
        """
        if not hasattr(self, "_worker_idx"):
            raise RuntimeError(
                "worker_idx has not been set. This collector may not have been "
                "initialized as a worker in a distributed setup."
            )
        return self._worker_idx

    @worker_idx.setter
    def worker_idx(self, value: int | None) -> None:
        """Set the worker index for this collector.

        Args:
            value: The worker index (0-indexed) or None.
        """
        self._worker_idx = value

    def cascade_execute(self, attr_path: str, *args, **kwargs) -> Any:
        """Execute a method on a nested attribute of this collector.

        This method allows remote callers to invoke methods on nested attributes
        of the collector without needing to know the full structure. It's particularly
        useful for calling methods on weight sync schemes from the sender side.

        Args:
            attr_path: Full path to the callable, e.g.,
                "_receiver_schemes['model_id']._set_dist_connection_info"
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            The return value of the method call.

        Examples:
            >>> collector.cascade_execute(
            ...     "_receiver_schemes['policy']._set_dist_connection_info",
            ...     connection_info_ref,
            ...     worker_idx=0
            ... )
        """
        attr = _resolve_attr(self, attr_path)
        if callable(attr):
            return attr(*args, **kwargs)
        else:
            if args or kwargs:
                raise ValueError(
                    f"Arguments and keyword arguments are not supported for non-callable attributes. Got {args} and {kwargs} for {attr_path}"
                )
            return attr

    def _get_policy_and_device(
        self,
        policy: Callable[[Any], Any] | None = None,
        policy_device: Any = NO_DEFAULT,
        env_maker: Any | None = None,
        env_maker_kwargs: dict[str, Any] | None = None,
    ) -> tuple[TensorDictModule, None | Callable[[], dict]]:
        """Util method to get a policy and its device given the collector __init__ inputs.

        We want to copy the policy and then move the data there, not call policy.to(device).

        Args:
            policy (TensorDictModule, optional): a policy to be used
            policy_device (torch.device, optional): the device where the policy should be placed.
                Defaults to self.policy_device
            env_maker (a callable or a batched env, optional): the env_maker function for this device/policy pair.
            env_maker_kwargs (a dict, optional): the env_maker function kwargs.

        """
        if policy_device is NO_DEFAULT:
            policy_device = self.policy_device

        if not policy_device:
            return policy, None

        if isinstance(policy, nn.Module):
            param_and_buf = TensorDict.from_module(policy, as_module=True)
        else:
            # Because we want to reach the warning
            param_and_buf = TensorDict()

        i = -1
        for p in param_and_buf.values(True, True):
            i += 1
            if p.device != policy_device:
                # Then we need casting
                break
        else:
            if i == -1 and not self.trust_policy:
                # We trust that the policy policy device is adequate
                warnings.warn(
                    "A policy device was provided but no parameter/buffer could be found in "
                    "the policy. Casting to policy_device is therefore impossible. "
                    "The collector will trust that the devices match. To suppress this "
                    "warning, set `trust_policy=True` when building the collector."
                )
            return policy, None

        # Create a stateless policy, then populate this copy with params on device
        def get_original_weights(policy=policy):
            td = TensorDict.from_module(policy)
            return td.data

        # We need to use ".data" otherwise buffers may disappear from the `get_original_weights` function
        with param_and_buf.data.to("meta").to_module(policy):
            policy_new_device = deepcopy(policy)

        param_and_buf_new_device = param_and_buf.apply(
            functools.partial(_map_weight, policy_device=policy_device),
            filter_empty=False,
        )
        param_and_buf_new_device.to_module(policy_new_device)
        # Sanity check
        if set(TensorDict.from_module(policy_new_device).keys(True, True)) != set(
            get_original_weights().keys(True, True)
        ):
            raise RuntimeError("Failed to map weights. The weight sets mismatch.")
        return policy_new_device, get_original_weights

    def start(self):
        """Starts the collector for asynchronous data collection.

        This method initiates the background collection of data, allowing for decoupling of data collection and training.

        The collected data is typically stored in a replay buffer passed during the collector's initialization.

        .. note:: After calling this method, it's essential to shut down the collector using :meth:`~.async_shutdown`
            when you're done with it to free up resources.

        .. warning:: Asynchronous data collection can significantly impact training performance due to its decoupled nature.
            Ensure you understand the implications for your specific algorithm before using this mode.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError(
            f"Collector start() is not implemented for {type(self).__name__}."
        )

    @contextlib.contextmanager
    def pause(self):
        """Context manager that pauses the collector if it is running free."""
        raise NotImplementedError(
            f"Collector pause() is not implemented for {type(self).__name__}."
        )

    def async_shutdown(
        self, timeout: float | None = None, close_env: bool = True
    ) -> None:
        """Shuts down the collector when started asynchronously with the `start` method.

        Args:
            timeout (float, optional): The maximum time to wait for the collector to shutdown.
            close_env (bool, optional): If True, the collector will close the contained environment.
                Defaults to `True`.

        .. seealso:: :meth:`~.start`

        """
        return self.shutdown(timeout=timeout, close_env=close_env)

    def _extract_weights_if_needed(self, weights: Any, model_id: str) -> Any:
        """Extract weights from a model if needed.

        For the new weight sync scheme system, weight preparation is handled
        by the scheme's prepare_weights() method. This method now only handles
        legacy weight updater cases.

        Args:
            weights: Either already-extracted weights or a model to extract from.
            model_id: The model identifier for resolving string paths.

        Returns:
            Extracted weights in the appropriate format.
        """
        # New weight sync schemes handle preparation themselves
        if self._weight_sync_schemes:
            # Just pass through - WeightSender will call scheme.prepare_weights()
            return weights

        # Legacy weight updater path
        return self._legacy_extract_weights(weights, model_id)

    def _legacy_extract_weights(self, weights: Any, model_id: str) -> Any:
        """Legacy weight extraction for old weight updater system.

        Args:
            weights: Either already-extracted weights or a model to extract from.
            model_id: The model identifier.

        Returns:
            Extracted weights.
        """
        if weights is None:
            if model_id == "policy" and hasattr(self, "policy_weights"):
                return self.policy_weights
            elif model_id == "policy" and hasattr(self, "_policy_weights_dict"):
                policy_device = (
                    self.policy_device
                    if not isinstance(self.policy_device, (list, tuple))
                    else self.policy_device[0]
                )
                return self._policy_weights_dict.get(policy_device)
            return None

        return weights

    @property
    def _legacy_weight_updater(self) -> bool:
        return self._weight_updater is not None

    # Overloads for update_policy_weights_ to support multiple calling conventions
    @overload
    def update_policy_weights_(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | nn.Module | dict,
        /,
    ) -> None:
        ...

    @overload
    def update_policy_weights_(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | nn.Module | dict,
        /,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        model_id: str | None = None,
    ) -> None:
        ...

    @overload
    def update_policy_weights_(
        self,
        *,
        weights: TensorDictBase | dict,
        model_id: str | None = None,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
    ) -> None:
        ...

    @overload
    def update_policy_weights_(
        self,
        *,
        policy: TensorDictModuleBase | nn.Module,
        model_id: str | None = None,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
    ) -> None:
        ...

    @overload
    def update_policy_weights_(
        self,
        *,
        weights_dict: dict[
            str, TensorDictBase | TensorDictModuleBase | nn.Module | dict
        ],
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
    ) -> None:
        ...

    def update_policy_weights_(
        self,
        policy_or_weights: TensorDictBase
        | TensorDictModuleBase
        | nn.Module
        | dict
        | None = None,
        *,
        weights: TensorDictBase | dict | None = None,
        policy: TensorDictModuleBase | nn.Module | None = None,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        model_id: str | None = None,
        weights_dict: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Update policy weights for the data collector.

        This method synchronizes the policy weights used by the collector with the latest
        trained weights. It supports both local and remote weight updates, depending on
        the collector configuration.

        The method accepts weights in multiple forms for convenience:

        Examples:
            >>> # Pass policy module as positional argument
            >>> collector.update_policy_weights_(policy_module)
            >>>
            >>> # Pass TensorDict weights as positional argument
            >>> collector.update_policy_weights_(weights_tensordict)
            >>>
            >>> # Use keyword arguments for clarity
            >>> collector.update_policy_weights_(weights=weights_td, model_id="actor")
            >>> collector.update_policy_weights_(policy=actor_module, model_id="actor")
            >>>
            >>> # Update multiple models atomically
            >>> collector.update_policy_weights_(weights_dict={
            ...     "actor": actor_weights,
            ...     "critic": critic_weights,
            ... })

        Args:
            policy_or_weights: The weights to update with. Can be:

                - ``nn.Module``: A policy module whose weights will be extracted
                - ``TensorDictModuleBase``: A TensorDict module whose weights will be extracted
                - ``TensorDictBase``: A TensorDict containing weights
                - ``dict``: A regular dict containing weights
                - ``None``: Will try to get weights from server using ``_get_server_weights()``

        Keyword Args:
            weights: Alternative to positional argument. A TensorDict or dict containing
                weights to update. Cannot be used together with ``policy_or_weights`` or ``policy``.
            policy: Alternative to positional argument. An ``nn.Module`` or ``TensorDictModuleBase``
                whose weights will be extracted. Cannot be used together with ``policy_or_weights``
                or ``weights``.
            worker_ids: Identifiers for the workers to update. Relevant when the collector
                has multiple workers. Can be int, list of ints, device, or list of devices.
            model_id: The model identifier to update (default: ``"policy"``).
                Cannot be used together with ``weights_dict``.
            weights_dict: Dictionary mapping model_id to weights for updating
                multiple models atomically. Keys should match model_ids registered in
                ``weight_sync_schemes``. Cannot be used together with ``model_id``,
                ``policy_or_weights``, ``weights``, or ``policy``.

        Raises:
            TypeError: If ``worker_ids`` is provided but no ``weight_updater`` is configured.
            ValueError: If conflicting parameters are provided.

        .. note:: Users should extend the ``WeightUpdaterBase`` classes to customize
            the weight update logic for specific use cases.

        .. seealso:: :class:`~torchrl.collectors.LocalWeightsUpdaterBase` and
            :meth:`~torchrl.collectors.RemoteWeightsUpdaterBase`.

        """
        # Handle the different keyword argument forms
        if weights is not None:
            if policy_or_weights is not None:
                raise ValueError(
                    "Cannot specify both positional 'policy_or_weights' and keyword 'weights'"
                )
            if policy is not None:
                raise ValueError("Cannot specify both 'weights' and 'policy'")
            policy_or_weights = weights

        if policy is not None:
            if policy_or_weights is not None:
                raise ValueError(
                    "Cannot specify both positional 'policy_or_weights' and keyword 'policy'"
                )
            policy_or_weights = policy
        if self._legacy_weight_updater:
            return self._legacy_weight_update_impl(
                policy_or_weights=policy_or_weights,
                worker_ids=worker_ids,
                model_id=model_id,
                weights_dict=weights_dict,
                **kwargs,
            )
        else:
            return self._weight_update_impl(
                policy_or_weights=policy_or_weights,
                worker_ids=worker_ids,
                model_id=model_id,
                weights_dict=weights_dict,
                **kwargs,
            )

    def _legacy_weight_update_impl(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | dict | None = None,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        model_id: str | None = None,
        weights_dict: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if weights_dict is not None:
            raise ValueError("weights_dict is not supported with legacy weight updater")
        if model_id is not None:
            raise ValueError("model_id is not supported with legacy weight updater")
        # Fall back to old weight updater system
        self.weight_updater(
            policy_or_weights=policy_or_weights, worker_ids=worker_ids, **kwargs
        )

    def _weight_update_impl(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | dict | None = None,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        model_id: str | None = None,
        weights_dict: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if "policy_weights" in kwargs:
            warnings.warn(
                "`policy_weights` is deprecated. Use `policy_or_weights` instead.",
                DeprecationWarning,
            )
            policy_or_weights = kwargs.pop("policy_weights")

        if weights_dict is not None and model_id is not None:
            raise ValueError("Cannot specify both 'weights_dict' and 'model_id'")

        if weights_dict is not None and policy_or_weights is not None:
            raise ValueError(
                "Cannot specify both 'weights_dict' and 'policy_or_weights'"
            )

        if self._weight_sync_schemes:
            if model_id is None:
                model_id = "policy"
            if policy_or_weights is not None and weights_dict is None:
                # Use model_id as the key, not hardcoded "policy"
                weights_dict = {model_id: policy_or_weights}
            elif weights_dict is None:
                weights_dict = {model_id: policy_or_weights}
            for target_model_id, weights in weights_dict.items():
                if target_model_id not in self._weight_sync_schemes:
                    raise KeyError(
                        f"Model '{target_model_id}' not found in registered weight sync schemes. "
                        f"Available models: {list(self._weight_sync_schemes.keys())}"
                    )
                processed_weights = self._extract_weights_if_needed(
                    weights, target_model_id
                )
                # Use new send() API with worker_ids support
                scheme = self._weight_sync_schemes.get(target_model_id)
                if not isinstance(scheme, WeightSyncScheme):
                    raise TypeError(f"Expected WeightSyncScheme, got {target_model_id}")
                self._send_weights_scheme(
                    scheme=scheme,
                    processed_weights=processed_weights,
                    worker_ids=worker_ids,
                    model_id=target_model_id,
                )
        elif self._weight_updater is not None:
            # unreachable
            raise RuntimeError
        else:
            # No weight updater configured, try fallback
            self._maybe_fallback_update(policy_or_weights, model_id=model_id)

    def _maybe_fallback_update(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | dict | None = None,
        *,
        model_id: str | None = None,
    ) -> None:
        """Fallback weight update when no scheme is configured.

        Override in subclasses to provide custom fallback behavior.
        By default, this is a no-op.
        """

    def _send_weights_scheme(self, *, model_id, scheme, processed_weights, worker_ids):
        # method to override if the scheme requires an RPC call to receive the weights
        scheme.send(weights=processed_weights, worker_ids=worker_ids)

    def _receive_weights_scheme(self):
        """Receive weights for all registered receiver schemes.

        scheme.receive() handles both applying weights locally and cascading
        to sub-collectors via context.update_policy_weights_().
        """
        if not hasattr(self, "_receiver_schemes"):
            raise RuntimeError("No receiver schemes registered.")

        for scheme in self._receiver_schemes.values():
            scheme.receive()

    # Overloads for receive_weights to support multiple calling conventions
    @overload
    def receive_weights(self) -> None:
        ...

    @overload
    def receive_weights(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | nn.Module | dict,
        /,
    ) -> None:
        ...

    @overload
    def receive_weights(
        self,
        *,
        weights: TensorDictBase | dict,
    ) -> None:
        ...

    @overload
    def receive_weights(
        self,
        *,
        policy: TensorDictModuleBase | nn.Module,
    ) -> None:
        ...

    def receive_weights(
        self,
        policy_or_weights: TensorDictBase
        | TensorDictModuleBase
        | nn.Module
        | dict
        | None = None,
        *,
        weights: TensorDictBase | dict | None = None,
        policy: TensorDictModuleBase | nn.Module | None = None,
    ) -> None:
        """Receive and apply weights to the collector's policy.

        This method applies weights to the local policy. When receiver schemes are
        registered, it delegates to those schemes. Otherwise, it directly applies
        the provided weights.

        The method accepts weights in multiple forms for convenience:

        Examples:
            >>> # Receive from registered schemes (distributed collectors)
            >>> collector.receive_weights()
            >>>
            >>> # Apply weights from a policy module (positional)
            >>> collector.receive_weights(trained_policy)
            >>>
            >>> # Apply weights from a TensorDict (positional)
            >>> collector.receive_weights(weights_tensordict)
            >>>
            >>> # Use keyword arguments for clarity
            >>> collector.receive_weights(weights=weights_td)
            >>> collector.receive_weights(policy=trained_policy)

        Args:
            policy_or_weights: The weights to apply. Can be:

                - ``nn.Module``: A policy module whose weights will be extracted and applied
                - ``TensorDictModuleBase``: A TensorDict module whose weights will be extracted
                - ``TensorDictBase``: A TensorDict containing weights
                - ``dict``: A regular dict containing weights
                - ``None``: Receive from registered schemes or mirror from original policy

        Keyword Args:
            weights: Alternative to positional argument. A TensorDict or dict containing
                weights to apply. Cannot be used together with ``policy_or_weights`` or ``policy``.
            policy: Alternative to positional argument. An ``nn.Module`` or ``TensorDictModuleBase``
                whose weights will be extracted. Cannot be used together with ``policy_or_weights``
                or ``weights``.

        Raises:
            ValueError: If conflicting parameters are provided or if arguments are passed
                when receiver schemes are registered.

        """
        # Handle the different keyword argument forms
        if weights is not None:
            if policy_or_weights is not None:
                raise ValueError(
                    "Cannot specify both positional 'policy_or_weights' and keyword 'weights'"
                )
            if policy is not None:
                raise ValueError("Cannot specify both 'weights' and 'policy'")
            policy_or_weights = weights

        if policy is not None:
            if policy_or_weights is not None:
                raise ValueError(
                    "Cannot specify both positional 'policy_or_weights' and keyword 'policy'"
                )
            policy_or_weights = policy

        if getattr(self, "_receiver_schemes", None) is not None:
            if policy_or_weights is not None:
                raise ValueError(
                    "Cannot specify 'policy_or_weights' when using 'receiver_schemes'. Schemes should know how to get the weights."
                )
            self._receive_weights_scheme()
            return

        # No weight updater configured
        # For single-process collectors, apply weights locally if explicitly provided
        if policy_or_weights is not None:
            from torchrl.weight_update.weight_sync_schemes import WeightStrategy

            # Use WeightStrategy to apply weights properly
            strategy = WeightStrategy(extract_as="tensordict")

            # Extract weights if needed
            if isinstance(policy_or_weights, nn.Module):
                weights = strategy.extract_weights(policy_or_weights)
            else:
                weights = policy_or_weights

            # Apply to local policy
            if hasattr(self, "policy") and isinstance(self.policy, nn.Module):
                strategy.apply_weights(self.policy, weights)
        # Otherwise, no action needed - policy is local and changes are immediately visible

    def register_scheme_receiver(
        self,
        weight_recv_schemes: dict[str, WeightSyncScheme],
        *,
        synchronize_weights: bool = True,
    ):  # noqa: D417
        """Set up receiver schemes for this collector to receive weights from parent collectors.

        This method initializes receiver schemes and stores them in _receiver_schemes
        for later use by _receive_weights_scheme() and receive_weights().

        Receiver schemes enable cascading weight updates across collector hierarchies:
        - Parent collector sends weights via its weight_sync_schemes (senders)
        - Child collector receives weights via its weight_recv_schemes (receivers)
        - If child is also a parent (intermediate node), it can propagate to its own children

        Args:
            weight_recv_schemes (dict[str, WeightSyncScheme]): Dictionary of {model_id: WeightSyncScheme} to set up as receivers.
                These schemes will receive weights from parent collectors.

        Keyword Args:
            synchronize_weights (bool, optional): If True, synchronize weights immediately after registering the schemes.
                Defaults to `True`.
        """
        # Initialize _receiver_schemes if not already present
        if not hasattr(self, "_receiver_schemes"):
            self._receiver_schemes = {}

        # Initialize each scheme on the receiver side
        for model_id, scheme in weight_recv_schemes.items():
            if not scheme.initialized_on_receiver:
                if scheme.initialized_on_sender:
                    raise RuntimeError(
                        "Weight sync scheme cannot be initialized on both sender and receiver."
                    )
                scheme.init_on_receiver(
                    model_id=model_id,
                    context=self,
                    worker_idx=self.worker_idx,
                )

            # Store the scheme for later use in receive_weights()
            self._receiver_schemes[model_id] = scheme

        # Perform initial synchronization
        if synchronize_weights:
            for scheme in weight_recv_schemes.values():
                if not scheme.synchronized_on_receiver:
                    scheme.connect(worker_idx=self.worker_idx)

    def __iter__(self) -> Iterator[TensorDictBase]:
        # Mark that iteration has started (used by enable_profile check)
        self._iteration_started = True
        try:
            yield from self.iterator()
        except Exception:
            self.shutdown()
            raise

    def next(self):
        try:
            if self._iterator is None:
                self._iterator = iter(self)
            out = next(self._iterator)
            # if any, we don't want the device ref to be passed in distributed settings
            if out is not None and (out.device != "cpu"):
                out = out.copy().clear_device_()
            return out
        except StopIteration:
            return None

    @abc.abstractmethod
    def shutdown(
        self,
        timeout: float | None = None,
        close_env: bool = True,
        raise_on_error: bool = True,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def iterator(self) -> Iterator[TensorDictBase]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def _read_compile_kwargs(self, compile_policy, cudagraph_policy):
        self.compiled_policy = compile_policy not in (False, None)
        self.cudagraphed_policy = cudagraph_policy not in (False, None)
        self.compiled_policy_kwargs = (
            {} if not isinstance(compile_policy, typing.Mapping) else compile_policy
        )
        self.cudagraphed_policy_kwargs = (
            {} if not isinstance(cudagraph_policy, typing.Mapping) else cudagraph_policy
        )

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        return string

    def __class_getitem__(self, index):
        raise NotImplementedError

    def __len__(self) -> int:
        if self.total_frames > 0:
            return -(self.total_frames // -self.requested_frames_per_batch)
        raise RuntimeError("Non-terminating collectors do not have a length")

    def init_updater(self, *args, **kwargs):
        """Initialize the weight updater with custom arguments.

        This method passes the arguments to the weight updater's init method.
        If no weight updater is set, this is a no-op.

        Args:
            *args: Positional arguments for weight updater initialization
            **kwargs: Keyword arguments for weight updater initialization
        """
        if self.weight_updater is not None:
            self.weight_updater.init(*args, **kwargs)


def _make_legacy_metaclass(parent_metaclass):
    """Create a legacy metaclass for deprecated collector names.

    This factory creates a metaclass that inherits from the given parent metaclass
    to avoid metaclass conflicts.
    """

    class _LegacyMeta(parent_metaclass):
        """Metaclass for deprecated collector class names.

        Raises a deprecation warning when the old class name is instantiated,
        and ensures isinstance() checks work for both old and new names.
        """

        def __call__(cls, *args, **kwargs):
            warnings.warn(
                f"{cls.__name__} has been deprecated and will be removed in v0.13. "
                f"Please use {cls.__bases__[0].__name__} instead.",
                category=DeprecationWarning,
            )
            return super().__call__(*args, **kwargs)

        def __instancecheck__(cls, instance):
            if super().__instancecheck__(instance):
                return True
            parent_cls = cls.__bases__[0]
            return isinstance(instance, parent_cls)

    return _LegacyMeta


# Default legacy metaclass for classes with abc.ABCMeta
_LegacyCollectorMeta = _make_legacy_metaclass(abc.ABCMeta)


class DataCollectorBase(BaseCollector, metaclass=_LegacyCollectorMeta):
    """Deprecated version of :class:`~torchrl.collectors.BaseCollector`."""

    ...
