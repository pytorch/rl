from typing import Callable, Dict, Iterator, List, OrderedDict, Union, Optional

import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase


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
        configuration: Dict,
        make_actor_critic: Callable,
        make_collector: Callable,
        make_objective: Callable,
        make_buffer: Callable,
        updates_per_batch: int = 1,
        device: torch.device = "cpu",
    ):

        self.device = device
        self.updates_per_batch = updates_per_batch

        # Get Actor Critic instance
        # Could be instantiated passing class + params if necessary
        self.policy, self.critic = make_actor_critic(configuration)
        self.policy.to(self.device)
        self.critic.to(self.device)

        # Get loss instance
        # Could be instantiated passing class + params if necessary
        self.objective = make_objective(configuration.loss, actor_network=self.policy, value_network=self.critic)
        self.params = TensorDict(dict(self.objective.named_parameters()), [])

        # Get collector instance.
        # Could be instantiated passing class + params if necessary.
        self.collector = make_collector(configuration, policy=self.policy)

        # Get storage instance.
        # Could be instantiated passing class + params if necessary
        self.replay_buffer = make_buffer(configuration)

        # Get grads
        self.grads = self.params.apply(lambda x: x.grad)

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
            self,
            weights: Optional[TensorDictBase] = None,
    ) -> None:
        self.params.apply_(lambda x, y: setattr(x, 'data', y), weights)
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

                # Zero gradients
                self.grads.zero_()

                # Backprop loss
                loss_sum.backward()

                yield self.grads

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return self.collector.set_seed(seed, static_seed)

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError
