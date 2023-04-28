import logging
import warnings
from torch.utils.data import IterableDataset
from typing import Callable, Dict, Iterator, List, OrderedDict, Union, Optional
from torch.optim import Optimizer, Adam


import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    DataCollectorBase,
    DEFAULT_EXPLORATION_TYPE,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.objectives import LossModule
from torchrl.collectors.utils import split_trajectories
from torchrl.envs import EnvBase, EnvCreator


class GradientWorker:
    """
    This Python class serves as a solution to abstract data collection and gradient
    computation.

    This class is an iterable that yields model gradients until a target number of collected
    frames is reached.
    """
    def __init__(
        self,
        policy: Callable[[TensorDict], TensorDict],
        collector: DataCollectorBase,
        objective: LossModule,
        replay_buffer: ReplayBuffer,
        optimizer: Optimizer = Adam,
        updates_per_batch: int = 1,
    ):

        self.policy = policy
        self.objective = objective
        self.collector = collector
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.updates_per_batch = updates_per_batch

    def __iter__(self) -> Iterator[TensorDictBase]:
        return self.iterator()

    def next(self):
        try:
            if self._iterator is None:
                self._iterator = iter(self)
            out = next(self._iterator)
            # if any, we don't want the device ref to be passed in distributed settings
            out.clear_device_()
            return out
        except StopIteration:
            return None

    def update_policy_weights_(
        self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    def iterator(self) -> Iterator[TensorDictBase]:
        grads = self._step_iterator()
        return grads

    def _step_iterator(self):

        # Collect batch
        data = self.collector.next()

        # Add to replay buffer
        self.replay_buffer.extend(data)

        # For num updates per batch
        for _ in range(self.updates_per_batch):

            # Sample batch from replay buffer
            mini_batch = self.replay_buffer.sample()

            # Compute loss
            loss = self.objective(mini_batch)
            loss_sum = sum([item for key, item in loss.items() if key.startswith("loss")])

            # Backprop loss
            self.optimizer.zero_grad()
            loss_sum.backprop()

            # Get gradients as a Tensordict
            import ipdb; ipdb.set_trace()  # Get grads only
            policy_grads = TensorDict(dict(self.policy.named_grads()), [])

            yield policy_grads

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        raise NotImplementedError

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError
