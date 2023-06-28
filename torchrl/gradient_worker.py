import copy
import itertools
from typing import Callable, Dict, Iterator, List, Optional, OrderedDict, Union

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase


def set_grad(p):
    """Initializes gradients to zero."""
    p.grad = torch.zeros_like(p.data)
    return p


class GradientWorker:
    """Worker that computes gradients for a given objective."""

    def __init__(
        self,
        objective,
        device: torch.device = "cpu",
    ):
        """Initializes the worker.

        objective: The objective to optimize.
        device: The device to use for computation.
        """

        self.device = device
        self.objective = objective

        self.weights = TensorDict(dict(self.objective.named_parameters()), [])
        self.weights.apply(set_grad)  # Initialize gradients to zero
        self.weights.lock_()

        self.weights_data = self.weights.apply(lambda p: p.data)
        self.weights_data.lock_()

        self.grads = self.weights.apply(lambda p: p.grad)
        self.grads.lock_()

    def update_policy_weights_(
        self,
        weights: TensorDictBase,
    ) -> None:
        self.grads.zero_()
        self.weights_data.update_(weights)

    def compute_gradients(
        self,
        mini_batch: TensorDictBase,
    ) -> TensorDictBase:
        """Computes gradients for the given mini-batch."""

        mini_batch = mini_batch.to(self.device)

        # Compute loss
        loss = self.objective(mini_batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

        # Backprop loss
        print("Computing remote gradients...")
        loss_sum.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.objective.parameters(), max_norm=0.5
        )

        return self.grads
