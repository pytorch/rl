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
        objective,
        updates_per_batch: int = 1,
        device: torch.device = "cpu",
    ):

        self.device = device
        self.updates_per_batch = updates_per_batch
        self.objective = objective


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

        print("Updating remote policy...")

        # for g, p in zip(weights, self.objective.parameters()):
        #     p.data = torch.from_numpy(g).to(self.device)
        #     p.grad.zero_()

        # params = self.objective.parameters()
        # for g, p in zip(weights, params):
        #     p.data = torch.from_numpy(g).to(self.device)
        #     p.grad.zero_()

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
        pass

    def compute_gradients(self, mini_batch):
        """Computes next gradient in each iteration."""

        mini_batch = mini_batch.to(self.device)

        # Compute loss
        loss = self.objective(mini_batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

        # Backprop loss
        print("Computing remote gradients...")
        loss_sum.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.objective.parameters(), max_norm=0.5)

        params = TensorDict.from_module(self.objective).lock_()
        grads = params.apply(lambda p: p.grad).lock_()

        yield grads

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return self.collector.set_seed(seed, static_seed)

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError
