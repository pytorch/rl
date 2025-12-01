from __future__ import annotations

import abc
import contextlib
import functools
import typing
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.base import NO_DEFAULT
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torch import nn as nn
from torch.utils.data import IterableDataset
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors.utils import _map_weight

from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.weight_update.utils import _resolve_attr
from torchrl.weight_update.weight_sync_schemes import WeightSyncScheme


class DataCollectorBase(IterableDataset, metaclass=abc.ABCMeta):
    """Base class for data collectors."""

    _task = None
    _iterator = None
    total_frames: int
    requested_frames_per_batch: int
    frames_per_batch: int
    trust_policy: bool
    compiled_policy: bool
    cudagraphed_policy: bool
    _weight_updater: WeightUpdaterBase | None = None
    _weight_sync_schemes: dict[str, WeightSyncScheme] | None = None
    verbose: bool = False

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
    def worker_idx(self) -> int:
        """Get the worker index for this collector.

        Returns:
            The worker index (0-indexed).

        Raises:
            RuntimeError: If worker_idx has not been set.
        """
        if not hasattr(self, "_worker_idx") or self._worker_idx is None:
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

    def update_policy_weights_(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | dict | None = None,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        model_id: str | None = None,
        weights_dict: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Updates the policy weights for the data collector, accommodating both local and remote execution contexts.

        This method ensures that the policy weights used by the data collector are synchronized with the latest
        trained weights. It supports both local and remote weight updates, depending on the configuration of the
        data collector. The local (download) update is performed before the remote (upload) update, such that weights
        can be transferred to the children workers from a server.

        Args:
            policy_or_weights (TensorDictBase | TensorDictModuleBase | dict | None): The weights to update with. Can be:
                - TensorDictModuleBase: A policy module whose weights will be extracted
                - TensorDictBase: A TensorDict containing weights
                - dict: A regular dict containing weights
                - None: Will try to get weights from server using _get_server_weights()
            worker_ids (int | List[int] | torch.device | List[torch.device] | None, optional): Identifiers for the
                workers that need to be updated. This is relevant when the collector has more than one worker associated
                with it.
            model_id (str | None, optional): The model identifier to update. If provided, only updates this specific
                model. Cannot be used together with weights_dict.
            weights_dict (dict[str, Any] | None, optional): Dictionary mapping model_id to weights for updating
                multiple models atomically. Keys should match the model_ids registered in weight_sync_schemes.
                Cannot be used together with model_id or policy_or_weights.

        Raises:
            TypeError: If `worker_ids` is provided but no `weight_updater` is configured.
            ValueError: If conflicting parameters are provided (e.g., both model_id and weights_dict).

        .. note:: Users should extend the `WeightUpdaterBase` classes to customize
            the weight update logic for specific use cases. This method should not be overwritten.

        .. seealso:: :class:`~torchrl.collectors.LocalWeightsUpdaterBase` and
            :meth:`~torchrl.collectors.RemoteWeightsUpdaterBase`.

        """
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
                torchrl_logger.debug("weight update -- getting scheme")
                scheme = self._weight_sync_schemes.get(target_model_id)
                if not isinstance(scheme, WeightSyncScheme):
                    raise TypeError(f"Expected WeightSyncScheme, got {target_model_id}")
                torchrl_logger.debug(
                    f"calling send() on scheme {type(scheme).__name__}"
                )
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
            return self.receive_weights(policy_or_weights)

    def _send_weights_scheme(self, *, model_id, scheme, processed_weights, worker_ids):
        # method to override if the scheme requires an RPC call to receive the weights
        scheme.send(weights=processed_weights, worker_ids=worker_ids)

    def _receive_weights_scheme(self, cascade_weights: bool = True):
        # Receive weights for all registered schemes
        updates = {}
        if not hasattr(self, "_receiver_schemes"):
            raise RuntimeError("No receiver schemes registered.")

        for model_id, scheme in self._receiver_schemes.items():
            # scheme.receive() pulls weights from the transport and applies them locally
            # For RPC/Ray: weights are already passed as argument, receive() is a no-op
            # For Distributed: receive() pulls from TCPStore
            # For MultiProcess: receive() checks the pipe
            torchrl_logger.debug(
                f"Receiving weights for scheme {type(scheme).__name__} for model '{model_id}' on worker {self._worker_idx}"
            )
            received_weights = scheme.receive()
            if received_weights is not None:
                updates[model_id] = received_weights

        # If we have nested collectors (e.g., MultiSyncDataCollector with inner workers)
        # AND we actually received updates, propagate them down via their senders
        if (
            cascade_weights
            and updates
            and hasattr(self, "_weight_sync_schemes")
            and self._weight_sync_schemes
        ):
            # Build weights_dict for all models that need propagation to nested collectors
            weights_dict = {}
            for model_id in updates:
                if model_id in self._weight_sync_schemes:
                    # This model has a sender scheme - propagate to nested workers
                    weights_dict[model_id] = updates[model_id]
                else:
                    # Clear error message when model_id mismatch
                    raise KeyError(
                        f"Received weights for model '{model_id}' but no sender "
                        f"scheme found to propagate to sub-collectors. "
                        f"Available sender schemes: {list(self._weight_sync_schemes.keys())}. "
                        f"To receive weights without cascading, call with cascade_weights=False."
                    )

            if weights_dict:
                # Propagate to nested collectors via their sender schemes
                torchrl_logger.debug(
                    f"Cascading weights to nested collectors: {weights_dict}"
                )
                self.update_policy_weights_(weights_dict=weights_dict)

    def receive_weights(self, policy_or_weights: TensorDictBase | None = None):
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
        elif (
            hasattr(self, "_original_policy")
            and isinstance(self._original_policy, nn.Module)
            and hasattr(self, "policy")
            and isinstance(self.policy, nn.Module)
        ):
            # If no weights were provided, mirror weights from the original (trainer) policy
            from torchrl.weight_update.weight_sync_schemes import WeightStrategy

            strategy = WeightStrategy(extract_as="tensordict")
            weights = strategy.extract_weights(self._original_policy)
            # Cast weights to the policy device before applying
            if self.policy_device is not None:
                weights = weights.to(self.policy_device)
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
                    worker_idx=getattr(self, "_worker_idx", None),
                )

            # Store the scheme for later use in receive_weights()
            self._receiver_schemes[model_id] = scheme

        # Perform initial synchronization
        if synchronize_weights:
            for model_id, scheme in weight_recv_schemes.items():
                if not scheme.synchronized_on_receiver:
                    torchrl_logger.debug(
                        f"Synchronizing weights for scheme {type(scheme).__name__} for model '{model_id}'"
                    )
                    scheme.connect(worker_idx=getattr(self, "_worker_idx", None))

    def __iter__(self) -> Iterator[TensorDictBase]:
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
