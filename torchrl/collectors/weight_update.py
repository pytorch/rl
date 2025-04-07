# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import weakref
from abc import abstractmethod
from typing import Any, Callable, TypeVar

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl._utils import logger as torchrl_logger

Policy = TypeVar("Policy", bound=TensorDictModuleBase)


class WeightUpdateReceiverBase(metaclass=abc.ABCMeta):
    """A base class for updating local policy weights from a server.

    This class provides an interface for downloading and updating the weights of a policy
    on a local inference worker. The update process is decentralized, meaning the inference
    worker is responsible for fetching the weights from the server.

    To extend this class, implement the following abstract methods:

    - `_get_server_weights`: Define how to retrieve the weights from the server.
    - `_get_local_weights`: Define how to access the current local weights.
    - `_maybe_map_weights`: Optionally transform the server weights to match the local weights.

    Attributes:
        policy (Policy, optional): The policy whose weights are to be updated.
        get_weights_from_policy (Callable, optional): A function to extract weights from the policy.
        get_weights_from_server (Callable, optional): A function to fetch weights from the server.
        weight_map_fn (Callable, optional): A function to map server weights to local weights.
        cache_policy_weights (bool): Whether to cache the policy weights locally.

    Methods:
        update_weights: Updates the local weights with the server weights.


    .. seealso:: :class:`~torchrl.collectors.RemoteWeightsUpdaterBase` and
        :meth:`~torchrl.collectors.DataCollectorBase.update_policy_weights_`.

    """

    _collector_wr: Any = None

    def register_collector(self, collector: DataCollectorBase):  # noqa
        """Register a collector in the updater.

        Once registered, the updater will not accept another collector.

        Args:
            collector (DataCollectorBase): The collector to register.

        """
        if self._collector_wr is not None:
            raise RuntimeError("Cannot register collector twice.")
        self._collector_wr = weakref.ref(collector)

    @property
    def collector(self) -> torchrl.collectors.DataCollectorBase:  # noqa
        return self._collector_wr() if self._collector_wr is not None else None

    @abstractmethod
    def _get_server_weights(self) -> TensorDictBase:
        ...

    @abstractmethod
    def _get_local_weights(self) -> TensorDictBase:
        ...

    @abstractmethod
    def _maybe_map_weights(
        self, server_weights: TensorDictBase, local_weights: TensorDictBase
    ) -> TensorDictBase:
        ...

    def _update_local_weights(
        self, local_weights: TensorDictBase, mapped_weights: TensorDictBase
    ) -> TensorDictBase:
        local_weights.update_(mapped_weights)

    def __call__(
        self,
        weights: TensorDictBase | None = None,
    ):
        return self.update_weights(weights=weights)

    def update_weights(self, weights: TensorDictBase | None = None) -> TensorDictBase:
        if weights is None:
            # get server weights (source)
            server_weights = self._get_server_weights()
        else:
            server_weights = weights
        # Get local weights
        local_weights = self._get_local_weights()

        # Optionally map the weights
        mapped_weights = self._maybe_map_weights(server_weights, local_weights)

        # Update the weights
        self._update_local_weights(local_weights, mapped_weights)


class WeightUpdateSenderBase(metaclass=abc.ABCMeta):
    """A base class for updating remote policy weights on inference workers.

    This class provides an interface for uploading and synchronizing the weights of a policy
    across remote inference workers. The update process is centralized, meaning the server
    is responsible for distributing the weights to the inference nodes.

    To extend this class, implement the following abstract methods:

    - `_sync_weights_with_worker`: Define how to synchronize weights with a specific worker.
    - `_get_server_weights`: Define how to retrieve the weights from the server.
    - `_maybe_map_weights`: Optionally transform the server weights before distribution.
    - `all_worker_ids`: Provide a list of all worker identifiers.

    Attributes:
        policy (Policy, optional): The policy whose weights are to be updated.

    Methods:
        update_weights: Updates the weights on specified or all remote workers.
        register_collector: Registers a collector. This should be called automatically by the collector
            upon registration of the updater.

    .. seealso:: :class:`~torchrl.collectors.LocalWeightsUpdaterBase` and
        :meth:`~torchrl.collectors.DataCollectorBase.update_policy_weights_`.

    """

    _collector_wr: Any = None

    def register_collector(self, collector: DataCollectorBase):  # noqa
        """Register a collector in the updater.

        Once registered, the updater will not accept another collector.

        Args:
            collector (DataCollectorBase): The collector to register.

        """
        if self._collector_wr is not None:
            raise RuntimeError("Cannot register collector twice.")
        self._collector_wr = weakref.ref(collector)

    @property
    def collector(self) -> torch.collector.DataCollectorBase:  # noqa
        return self._collector_wr() if self._collector_wr is not None else None

    @abstractmethod
    def _sync_weights_with_worker(
        self, worker_id: int | torch.device, server_weights: TensorDictBase
    ) -> TensorDictBase:
        ...

    @abstractmethod
    def _get_server_weights(self) -> TensorDictBase:
        ...

    @abstractmethod
    def _maybe_map_weights(self, server_weights: TensorDictBase) -> TensorDictBase:
        ...

    @abstractmethod
    def all_worker_ids(self) -> list[int] | list[torch.device]:
        ...

    def _skip_update(self, worker_id: int | torch.device) -> bool:
        return False

    def __call__(
        self,
        weights: TensorDictBase | None = None,
        worker_ids: torch.device | int | list[int] | list[torch.device] | None = None,
    ):
        return self.update_weights(weights=weights, worker_ids=worker_ids)

    def update_weights(
        self,
        weights: TensorDictBase | None = None,
        worker_ids: torch.device | int | list[int] | list[torch.device] | None = None,
    ):
        if weights is None:
            # Get the weights on server (local)
            server_weights = self._get_server_weights()
        else:
            server_weights = weights

        self._maybe_map_weights(server_weights)

        # Get the remote weights (inference workers)
        if isinstance(worker_ids, (int, torch.device)):
            worker_ids = [worker_ids]
        elif worker_ids is None:
            worker_ids = self.all_worker_ids()
        for worker in worker_ids:
            if self._skip_update(worker):
                continue
            self._sync_weights_with_worker(worker, server_weights)


# Specialized classes
class VanillaWeightUpdater(WeightUpdateReceiverBase):
    """A simple implementation of `WeightUpdateReceiverBase` for updating local policy weights.

    The `VanillaLocalWeightUpdater` class provides a basic mechanism for updating the weights
    of a local policy by directly fetching them from a specified source. It is typically used
    in scenarios where the weight update logic is straightforward and does not require any
    complex mapping or transformation.

    This class is used by default in the `SyncDataCollector` when no custom local weights updater
    is provided.

    Args:
        weight_getter (Callable[[], TensorDictBase]): A callable that returns the latest policy
            weights from the server or another source.
        policy_weights (TensorDictBase): The current weights of the local policy that need to be updated.

    Methods:
        _get_server_weights: Retrieves the latest weights from the specified source.
        _get_local_weights: Accesses the current local policy weights.
        _map_weights: Directly maps server weights to local weights without transformation.
        _maybe_map_weights: Optionally maps server weights to local weights (no-op in this implementation).
        _update_local_weights: Updates the local policy weights with the mapped weights.

    .. note::
        This class assumes that the server weights can be directly applied to the local policy
        without any additional processing. If your use case requires more complex weight mapping,
        consider extending `WeightUpdateReceiverBase` with a custom implementation.

    .. seealso:: :class:`~torchrl.collectors.WeightUpdateReceiverBase` and :class:`~torchrl.collectors.SyncDataCollector`.
    """

    def __init__(
        self,
        weight_getter: Callable[[], TensorDictBase],
        policy_weights: TensorDictBase,
    ):
        self.weight_getter = weight_getter
        self.policy_weights = policy_weights

    def _get_server_weights(self) -> TensorDictBase:
        return self.weight_getter() if self.weight_getter is not None else None

    def _get_local_weights(self) -> TensorDictBase:
        return self.policy_weights

    def _map_weights(self, server_weights: TensorDictBase) -> TensorDictBase:
        return server_weights

    def _maybe_map_weights(
        self, server_weights: TensorDictBase, local_weights: TensorDictBase
    ) -> TensorDictBase:
        return server_weights

    def _update_local_weights(
        self, local_weights: TensorDictBase, mapped_weights: TensorDictBase
    ) -> TensorDictBase:
        if local_weights is None or mapped_weights is None:
            return
        local_weights.update_(mapped_weights)


class MultiProcessedWeightUpdate(WeightUpdateSenderBase):
    """A remote weight updater for synchronizing policy weights across multiple processes or devices.

    The `MultiProcessedRemoteWeightUpdate` class provides a mechanism for updating the weights
    of a policy across multiple inference workers in a multiprocessed environment. It is designed
    to handle the distribution of weights from a central server to various devices or processes
    that are running the policy.
    This class is typically used in multiprocessed data collectors where each process or device
    requires an up-to-date copy of the policy weights.

    Args:
        get_server_weights (Callable[[], TensorDictBase] | None): A callable that retrieves the
            latest policy weights from the server or another centralized source.
        policy_weights (Dict[torch.device, TensorDictBase]): A dictionary mapping each device or
            process to its current policy weights, which will be updated.

    Methods:
        all_worker_ids: Returns a list of all worker identifiers (devices or processes).
        _sync_weights_with_worker: Synchronizes the server weights with a specific worker.
        _get_server_weights: Retrieves the latest weights from the server.
        _maybe_map_weights: Optionally maps server weights before distribution (no-op in this implementation).

    .. note::
        This class assumes that the server weights can be directly applied to the workers without
        any additional processing. If your use case requires more complex weight mapping or synchronization
        logic, consider extending `WeightUpdateSenderBase` with a custom implementation.

    .. seealso:: :class:`~torchrl.collectors.WeightUpdateSenderBase` and
        :class:`~torchrl.collectors.DataCollectorBase`.

    """

    def __init__(
        self,
        get_server_weights: Callable[[], TensorDictBase] | None,
        policy_weights: dict[torch.device, TensorDictBase],
    ):
        self.weights_getter = get_server_weights
        self._policy_weights = policy_weights

    def all_worker_ids(self) -> list[int] | list[torch.device]:
        return list(self._policy_weights)

    def _sync_weights_with_worker(
        self, worker_id: int | torch.device, server_weights: TensorDictBase
    ) -> TensorDictBase:
        if server_weights is None:
            return
        self._policy_weights[worker_id].data.update_(server_weights)

    def _get_server_weights(self) -> TensorDictBase:
        # The weights getter can be none if no mapping is required
        if self.weights_getter is None:
            return
        weights = self.weights_getter()
        if weights is None:
            return
        return weights.data

    def _maybe_map_weights(self, server_weights: TensorDictBase) -> TensorDictBase:
        return server_weights


class RayWeightUpdater(WeightUpdateSenderBase):
    """A remote weight updater for synchronizing policy weights across remote workers using Ray.

    The `RayWeightUpdateSender` class provides a mechanism for updating the weights of a policy
    across remote inference workers managed by Ray. It leverages Ray's distributed computing
    capabilities to efficiently distribute policy weights to remote collectors.
    This class is typically used in distributed data collectors where each remote worker requires
    an up-to-date copy of the policy weights.

    Args:
        policy_weights (TensorDictBase): The current weights of the policy that need to be distributed
            to remote workers.
        remote_collectors (List): A list of remote collectors that will receive the updated policy weights.
        max_interval (int, optional): The maximum number of batches between weight updates for each worker.
            Defaults to 0, meaning weights are updated every batch.

    Methods:
        all_worker_ids: Returns a list of all worker identifiers (indices of remote collectors).
        _get_server_weights: Retrieves the latest weights from the server and stores them in Ray's object store.
        _maybe_map_weights: Optionally maps server weights before distribution (no-op in this implementation).
        _sync_weights_with_worker: Synchronizes the server weights with a specific remote worker using Ray.
        _skip_update: Determines whether to skip the weight update for a specific worker based on the interval.

    .. note::
        This class assumes that the server weights can be directly applied to the remote workers without
        any additional processing. If your use case requires more complex weight mapping or synchronization
        logic, consider extending `WeightUpdateSenderBase` with a custom implementation.

    .. seealso:: :class:`~torchrl.collectors.WeightUpdateSenderBase` and
        :class:`~torchrl.collectors.distributed.RayCollector`.

    """

    def __init__(
        self,
        policy_weights: TensorDictBase,
        remote_collectors: list,
        max_interval: int = 0,
    ):
        self.policy_weights = policy_weights
        self.remote_collectors = remote_collectors
        self.max_interval = max(0, max_interval)
        self._batches_since_weight_update = [0] * len(self.remote_collectors)

    def all_worker_ids(self) -> list[int] | list[torch.device]:
        return list(range(len(self.remote_collectors)))

    def _get_server_weights(self) -> TensorDictBase:
        import ray

        return ray.put(self.policy_weights.data)

    def _maybe_map_weights(self, server_weights: TensorDictBase) -> TensorDictBase:
        return server_weights

    def _sync_weights_with_worker(
        self, worker_id: int, server_weights: TensorDictBase
    ) -> TensorDictBase:
        torchrl_logger.info(f"syncing weights with worker {worker_id}")
        c = self.remote_collectors[worker_id]
        c.update_policy_weights_.remote(policy_weights=server_weights)
        self._batches_since_weight_update[worker_id] = 0

    def _skip_update(self, worker_id: int) -> bool:
        self._batches_since_weight_update[worker_id] += 1
        # Use gt because we just incremented it
        if self._batches_since_weight_update[worker_id] > self.max_interval:
            return False
        return True
