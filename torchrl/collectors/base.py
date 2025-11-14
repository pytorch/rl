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
from torchrl.collectors.utils import _map_weight

from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.weight_update import WeightReceiver, WeightSender, WeightSyncScheme


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
    _weight_senders: dict[str, WeightSender] | None = None
    _weight_receivers: dict[str, WeightReceiver] | None = None
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

        if policy_or_weights is not None:
            weights_dict = {"policy": policy_or_weights}

        # Priority: new weight sync schemes > old weight updater system
        if self._weight_senders:
            if model_id is not None:
                # Compose weight_dict
                weights_dict = {model_id: policy_or_weights}
            if weights_dict is None:
                if "policy" in self._weight_senders:
                    weights_dict = {"policy": policy_or_weights}
                elif len(self._weight_senders) == 1:
                    single_model_id = next(iter(self._weight_senders.keys()))
                    weights_dict = {single_model_id: policy_or_weights}
                else:
                    raise ValueError(
                        "Cannot determine the model to update. Please provide a weights_dict."
                    )
            for target_model_id, weights in weights_dict.items():
                if target_model_id not in self._weight_senders:
                    raise KeyError(
                        f"Model '{target_model_id}' not found in registered weight senders. "
                        f"Available models: {list(self._weight_senders.keys())}"
                    )
                processed_weights = self._extract_weights_if_needed(
                    weights, target_model_id
                )
                # Use new send() API with worker_ids support
                self._weight_senders[target_model_id].send(
                    weights=processed_weights, worker_ids=worker_ids
                )
        elif self._weight_updater is not None:
            # unreachable
            raise RuntimeError
        else:
            return self.receive_weights(policy_or_weights)

    def receive_weights(self, policy_or_weights: TensorDictBase | None = None):
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
