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


class WeightUpdaterBase(metaclass=abc.ABCMeta):
    """A base class for updating remote policy weights on inference workers.

    The weight updater is the central piece of the weight update scheme:

    - In leaf collector nodes, it is responsible for sending the weights to the policy, which can be as simple as
      updating a state-dict, or more complex if an inference server is being used.
    - In server collector nodes, it is responsible for sending the weights to the leaf collectors.

    In a collector, the updater is called within :meth:`~torchrl.collector.DataCollectorBase.update_policy_weights_`.`

    The main method of this class is the :meth:`~.push_weights` method, which updates the policy weights in the worker /
    policy.

    To extend this class, implement the following abstract methods:

    - `_get_server_weights` (optional): Define how to retrieve the weights from the server if they are not passed to
        the updater directly. This method is only called if the weights (hanlde) is not passed directly.
    - `_sync_weights_with_worker`: Define how to synchronize weights with a specific worker.
        This method must be implemented by child classes.
    - `_maybe_map_weights`: Optionally transform the server weights before distribution.
        By default, this method returns the weights unchanged.
    - `all_worker_ids`: Provide a list of all worker identifiers.
        Returns `None` by default (no worker id).

    Attributes:
        collector: The collector (or any container) of the weight receiver. The collector is registered via
            :meth:`~torchrl.collectors.WeightUpdateReceiverBase.register_collector`.

    Methods:
        push_weights: Updates the weights on specified or all remote workers.
            The `__call__` method is a proxy to `push_weights`.
        register_collector: Registers the collector (or any container) in the receiver through a weakref.
            This will be called automatically by the collector upon registration of the updater.

    .. seealso:: :meth:`~torchrl.collectors.DataCollectorBase.update_policy_weights_`.

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
        """The collector or container of the receiver.

        Returns `None` if the container is out-of-scope or not set.
        """
        return self._collector_wr() if self._collector_wr is not None else None

    def _get_server_weights(self) -> Any:
        """An optional method to gather weights from the server.

        This method is called only if the weights (handle) are not passed directly to the update method.
        """
        raise NotImplementedError

    @abstractmethod
    def _sync_weights_with_worker(
        self, *, worker_id: int | torch.device | None = None, server_weights: Any
    ) -> Any:
        """An abstract method that updates the weights on specified workers.

        The worker id can be `None` if there are no workers associated with the sender.
        """
        ...

    def _maybe_map_weights(self, server_weights: Any) -> Any:
        """Optionally transforms the server weights to match the local weights."""
        return server_weights

    def all_worker_ids(self) -> list[int] | list[torch.device] | None:
        """Returns a list of all worker identifiers or `None` if there are no workers associated."""
        return

    def _skip_update(self, worker_id: int | torch.device) -> bool:
        """A method to determine if a worker should be skipped."""
        return False

    def __call__(
        self,
        weights: Any = None,
        worker_ids: torch.device | int | list[int] | list[torch.device] | None = None,
    ):
        """A proxy to :meth:`~.push_weights`."""
        return self.push_weights(weights=weights, worker_ids=worker_ids)

    def push_weights(
        self,
        *,
        weights: Any | None = None,
        worker_ids: torch.device | int | list[int] | list[torch.device] | None = None,
    ):
        """Updates the weights of the policy, or on specified / all remote workers.

        Args:
            weights (Any): The source weights to push to the policy / workers.
            worker_ids (torch.device | int | list[int] | list[torch.device] | None = None): an optional list of
                workers to update.

        Returns: nothing.

        """
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
        if worker_ids is None:
            self._sync_weights_with_worker(server_weights=server_weights)
            return
        for worker in worker_ids:
            if self._skip_update(worker):
                continue
            self._sync_weights_with_worker(
                worker_id=worker, server_weights=server_weights
            )


# Specialized classes
class VanillaWeightUpdater(WeightUpdaterBase):
    """A simple implementation of :class:`~torchrl.collectors.WeightUpdaterBase` for updating local policy weights.

    The `VanillaWeightSender` class provides a basic mechanism for updating the weights
    of a local policy by directly fetching them from a specified source. It is typically used
    in scenarios where the weight update logic is straightforward and does not require any
    complex mapping or transformation.

    This class is used by default in the `SyncDataCollector` when no custom weight sender
    is provided.

    .. seealso:: :class:`~torchrl.collectors.WeightUpdateReceiverBase` and :class:`~torchrl.collectors.SyncDataCollector`.

    Keyword Args:
        weight_getter (Callable[[], TensorDictBase], optional): a callable that returns the weights from the server.
            If not provided, the weights must be passed to :meth:`~.update_weights` directly.
        policy_weights (TensorDictBase): a TensorDictBase containing the policy weights to be updated
            in-place.
    """

    def __init__(
        self,
        *,
        weight_getter: Callable[[], TensorDictBase] | None = None,
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

    def _maybe_map_weights(self, server_weights: TensorDictBase) -> TensorDictBase:
        return server_weights

    def _sync_weights_with_worker(
        self, *, worker_id: None = None, server_weights: TensorDictBase
    ) -> TensorDictBase:
        if server_weights is None:
            return
        self.policy_weights.update_(server_weights)


class MultiProcessedWeightUpdate(WeightUpdaterBase):
    """A remote weight updater for synchronizing policy weights across multiple processes or devices.

    The `MultiProcessedWeightUpdater` class provides a mechanism for updating the weights
    of a policy across multiple inference workers in a multiprocessed environment. It is designed
    to handle the distribution of weights from a central server to various devices or processes
    that are running the policy.
    This class is typically used in multiprocessed data collectors where each process or device
    requires an up-to-date copy of the policy weights.

    Keyword Args:
        get_server_weights (Callable[[], TensorDictBase] | None): A callable that retrieves the
            latest policy weights from the server or another centralized source.
        policy_weights (Dict[torch.device, TensorDictBase]): A dictionary mapping each device or
            process to its current policy weights, which will be updated.

    .. note::
        This class assumes that the server weights can be directly applied to the workers without
        any additional processing. If your use case requires more complex weight mapping or synchronization
        logic, consider extending `WeightUpdaterBase` with a custom implementation.

    .. seealso:: :class:`~torchrl.collectors.WeightUpdaterBase` and
        :class:`~torchrl.collectors.DataCollectorBase`.

    """

    def __init__(
        self,
        *,
        get_server_weights: Callable[[], TensorDictBase] | None,
        policy_weights: dict[torch.device, TensorDictBase],
    ):
        self.weights_getter = get_server_weights
        self._policy_weights = policy_weights

    def all_worker_ids(self) -> list[int] | list[torch.device]:
        return list(self._policy_weights)

    def _sync_weights_with_worker(
        self, worker_id: int | torch.device, server_weights: TensorDictBase | None
    ) -> TensorDictBase | None:
        if server_weights is None:
            return
        self._policy_weights[worker_id].data.update_(server_weights)

    def _get_server_weights(self) -> TensorDictBase | None:
        # The weights getter can be none if no mapping is required
        if self.weights_getter is None:
            return
        weights = self.weights_getter()
        if weights is None:
            return
        return weights.data

    def _maybe_map_weights(self, server_weights: TensorDictBase) -> TensorDictBase:
        return server_weights


class RayWeightUpdater(WeightUpdaterBase):
    """A remote weight updater for synchronizing policy weights across remote workers using Ray.

    The `RayWeightUpdater` class provides a mechanism for updating the weights of a policy
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
        logic, consider extending `WeightUpdaterBase` with a custom implementation.

    .. seealso:: :class:`~torchrl.collectors.WeightUpdaterBase` and
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

    def _get_server_weights(self) -> Any:
        import ray

        return ray.put(self.policy_weights.data)

    def _maybe_map_weights(self, server_weights: Any) -> Any:
        return server_weights

    def _sync_weights_with_worker(self, worker_id: int, server_weights: Any) -> Any:
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
