from typing import Callable, Dict, Iterator, List, OrderedDict, Union, Optional

import copy
import torch
import itertools
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import nn
import torch
from tensordict import TensorDict


class GradientWorker:
    """Worker that computes gradients for a given objective."""

    def __init__(
        self,
        objective,
        device: torch.device = "cpu",
    ):

        self.device = device
        self.objective = objective

    def update_policy_weights_(
            self,
            weights,
            grads,
    ) -> None:

        for w, p in zip(weights, self.objective.parameters()):
            if w is not None:
                p.grad.zero_()
                p.data = torch.from_numpy(w).to(self.device)

        for g, p in zip(grads, self.objective.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g).to(self.device)

    def compute_gradients(self, mini_batch):
        """Computes next gradient in each iteration."""

        mini_batch = mini_batch.to("cuda")

        # Compute loss
        loss = self.objective(mini_batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

        # Backprop loss
        print("Computing remote gradients...")
        loss_sum.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.objective.parameters(), max_norm=0.5)

        # Get gradients
        grads = []
        for p in self.objective.parameters():
            if p.grad is not None:
                grads.append(p.grad.cpu().numpy())
            else:
                grads.append(None)

        return grads
