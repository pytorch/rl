from typing import Callable, Dict, Iterator, List, OrderedDict, Union, Optional

import copy
import torch
import itertools
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase


class GradientCollector:
    """
    This Python class serves as a solution to abstract data collection and gradient
    computation.

    This class is an iterable that yields model gradients until a target number of collected
    frames is reached.

    Args:
        configuration: A dictionary containing the configuration parameters.
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
        self.objective, self.advantage = make_objective(configuration.loss, actor_network=self.policy, value_network=self.critic)
        self.params = TensorDict(dict(self.objective.named_parameters()), [])

        # Get collector instance.
        # Could be instantiated passing class + params if necessary.
        self.collector = make_collector(configuration, policy=self.policy)

        # Get storage instance.
        # Could be instantiated passing class + params if necessary
        self.replay_buffer = make_buffer(configuration)

        # Get grads. TODO: this does not seem to work
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

        # for g, p in zip(weights, self.objective.parameters()):
        #     p.data = torch.from_numpy(g).to(self.device)
        #     p.grad.zero_()

        params = list(self.policy.parameters()) + list(self.critic.parameters())
        for g, p in zip(weights, params):
            p.data = torch.from_numpy(g).to(self.device)
            p.grad.zero_()

        self.collector.update_policy_weights_()

        # params = itertools.chain(self.collector.parameters())
        # for g, p in zip(weights, params):
        #     p.data = torch.from_numpy(g).to(self.device)
        #     p.grad.zero_()

    def shutdown(self):
        self.collector.shutdown()

    def iterator(self) -> Iterator[TensorDictBase]:
        grads = self._step_iterator()
        return grads

    def _step_iterator(self):
        """Computes next gradient in each iteration."""

        for data in self.collector:

            data_view = data.reshape(-1)

            # Compute GAE
            with torch.no_grad():
                data_view = self.advantage(data_view)

            # Add to replay buffer
            self.replay_buffer.extend(data_view)

            for iter in range(self.updates_per_batch):

                print(iter)

                # Sample batch from replay buffer
                mini_batch = self.replay_buffer.sample().to(self.device)

                # Compute loss
                loss = self.objective(mini_batch)
                # loss_sum = sum([item for key, item in loss.items() if key.startswith("loss")])
                loss_sum = (
                        loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )

                # Backprop loss
                loss_sum.backward()

                # Get gradients
                grads = []
                # for p in self.objective.parameters():
                #     if p.grad is not None:
                #         grads.append(p.grad.data.cpu().numpy())
                #     else:
                #         grads.append(None)

                for p in list(self.policy.parameters()) + list(self.critic.parameters()):
                    if p.grad is not None:
                        grads.append(p.grad.data.cpu().numpy())
                    else:
                        grads.append(None)

                yield grads

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return self.collector.set_seed(seed, static_seed)

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError
