from typing import Callable, Dict, Iterator, List, OrderedDict, Union, Optional

import copy
import torch
import itertools
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
import torch
from torch import nn
from tensordict import TensorDict


def set_grad(p):
    """Initializes gradients to zero."""
    p.grad = torch.zeros_like(p.data)
    return p


def apply_weights(w1, w2):
    """Applies weights to a model and re-sets gradients to zero."""
    w1.data.copy_(w2)
    w1.grad.zero_()
    return w1


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
        self.weights.apply(set_grad)
        self.weights.lock_()

    def update_policy_weights_(
            self,
            weights: TensorDictBase,
    ) -> None:
        self.weights.apply(apply_weights, weights)
        # self.weights.update_(weights)  # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
        # self.weights.update(weights)  #  RuntimeError: Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present.

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
        grad_norm = torch.nn.utils.clip_grad_norm_(self.objective.parameters(), max_norm=0.5)

        # Get gradients
        grads = self.weights.apply(lambda p: p.grad)

        return grads
