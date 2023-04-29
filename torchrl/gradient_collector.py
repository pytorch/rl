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


class GradientCollector:
    """
    This Python class serves as a solution to abstract data collection and gradient
    computation.

    This class is an iterable that yields model gradients until a target number of collected
    frames is reached.

    Args:
        policy (Callable): Instance of TensorDictModule class.
            Must accept TensorDictBase object as input.
    """
    def __init__(
        self,
        policy: Callable[[TensorDict], TensorDict],
        collector: DataCollectorBase,  # TODO: can we get away passing the instantiated class ?
        objective: LossModule,  # TODO: can we get away passing the instantiated class ?
        replay_buffer: ReplayBuffer,  # TODO: can we get away passing the instantiated class ?
        optimizer: Optimizer = Adam,  # TODO: can we get away passing the instantiated class ?
        critic: Callable[[TensorDict], TensorDict] = None,
        updates_per_batch: int = 1,
        device: torch.device = "cpu",
    ):

        self.device = device
        self.updates_per_batch = updates_per_batch

        # Get Actor Critic instance
        # Could be instantiated passing class + params if necessary
        self.policy = policy  # TODO: do I really need the policy if we can get the params from objective ?
        self.policy_params = TensorDict(dict(self.policy.named_parameters()), [])
        self.critic = critic  # TODO: do I really need the critic if we can get the params from objective ?

        # Get loss instance
        # Could be instantiated passing class + params if necessary
        self.objective = objective
        self.objective_params = TensorDict(dict(self.objective.named_parameters()), [])

        # Get collector instance.
        # Could be instantiated passing class + params if necessary.
        self.collector = collector

        # Get storage instance.
        # Could be instantiated passing class + params if necessary
        self.replay_buffer = replay_buffer

        # Get storage instance.
        # Could be instantiated passing class + params if necessary
        self.optimizer = optimizer

        # Check if policy parameters are being tracked by optimizer
        for name, param in policy.named_parameters():
            if optimizer.state.get(name) is None:
                warnings.warn(f"{name} is NOT being tracked by the optimizer")

        # Check if critic parameters are being tracked by optimizer
        if self.critic is not None:
            for name, param in critic.named_parameters():
                if optimizer.state.get(name) is None:
                    warnings.warn(f"{name} is NOT being tracked by the optimizer")

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

        # TODO: is this the right way to do it ?
        # Update policy and critic
        self.objective_params.apply(lambda x: x.data).update_(policy_weights)

        self.collector.update_policy_weights_()

    def shutdown(self):
        self.collector.shutdown()

    def iterator(self) -> Iterator[TensorDictBase]:
        grads = self._step_iterator()
        return grads

    def _step_iterator(self):
        """Computes next gradient in each iteration."""

        for data in self.collector:

            # Add to replay buffer
            self.replay_buffer.extend(data)

            for _ in range(self.updates_per_batch):

                # Sample batch from replay buffer
                mini_batch = self.replay_buffer.sample().to(self.device)

                # Compute loss
                loss = self.objective(mini_batch)
                loss_sum = sum([item for key, item in loss.items() if key.startswith("loss")])

                # Backprop loss
                self.optimizer.zero_grad()
                loss_sum.backward()

                # Get gradients as a Tensordict
                params = TensorDict(dict(self.objective.named_parameters()), [])
                grads = params.apply(lambda x: x.grad)

                del mini_batch
                del params

                yield grads

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        self.collector.set_seed(seed, static_seed)

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError
