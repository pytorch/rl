# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Union

import numpy as np

import torch
from tensordict.nn import dispatch
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey

from torch import distributions as d
from torchrl.modules import ProbabilisticActor

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import distance_loss


class OnlineDTLoss(LossModule):
    r"""TorchRL implementation of the Online Decision Transformer loss.

    Presented in `"Online Decision Transformer" <https://arxiv.org/abs/2202.05607>`

    Args:
        actor_network (ProbabilisticActor): stochastic actor
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is None (no minimum value).
        max_alpha (float, optional): max value of alpha.
            Default is None (no maximum value).
        fixed_alpha (bool, optional): if ``True``, alpha will be fixed to its
            initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
            Default is ``False``.
        target_entropy (float or str, optional): Target entropy for the
            stochastic policy. Default is "auto", where target entropy is
            computed as :obj:`-prod(n_actions)`.
        samples_mc_entropy (int): number of samples to estimate the entropy

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            observation (NestedKey): The input tensordict key where the observation is expected.
                Defaults to ``"observation"``.
            return_to_go (NestedKey): The input tensordict key where the return_to_go is expected.
                Defaults to ``"return_to_go"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
        """

        action: NestedKey = "action"
        observation: NestedKey = "observation"
        return_to_go: NestedKey = "return_to_go"
        done: NestedKey = "done"

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        *,
        alpha_init: float = 1.0,
        min_alpha: float = None,
        max_alpha: float = None,
        fixed_alpha: bool = False,
        target_entropy: Union[str, float] = "auto",
        samples_mc_entropy: int = 1,
    ) -> None:
        self._in_keys = None
        self._out_keys = None
        super().__init__()

        # Actor Network
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=False,
            funs_to_decorate=["forward", "get_dist"],
        )
        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")

        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        if bool(min_alpha) ^ bool(max_alpha):
            min_alpha = min_alpha if min_alpha else 0.0
            if max_alpha == 0:
                raise ValueError("max_alpha must be either None or greater than 0.")
            max_alpha = max_alpha if max_alpha else 1e9
        if min_alpha:
            self.register_buffer(
                "min_log_alpha", torch.tensor(min_alpha, device=device).log()
            )
        else:
            self.min_log_alpha = None
        if max_alpha:
            self.register_buffer(
                "max_log_alpha", torch.tensor(max_alpha, device=device).log()
            )
        else:
            self.max_log_alpha = None
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer(
                "log_alpha", torch.tensor(math.log(alpha_init), device=device)
            )
        else:
            self.register_parameter(
                "log_alpha",
                torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
            )

        if target_entropy == "auto":
            if actor_network.spec is None:
                raise RuntimeError(
                    "Cannot infer the dimensionality of the action. Consider providing "
                    "the target entropy explicitely or provide the spec of the "
                    "action tensor in the actor network."
                )
            target_entropy = -float(np.prod(actor_network.spec["action"].shape))
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )

        self.samples_mc_entropy = samples_mc_entropy
        self._set_in_keys()

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.return_to_go),
            ("next", self.tensor_keys.done),
            *self.tensor_keys.action,
            *[("next", key) for key in self.tensor_keys.action],
            *self.tensor_keys.observation,
        ]

        self._in_keys = list(set(keys))

    @property
    def alpha(self):
        if self.min_log_alpha is not None:
            self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss", "loss_log_likelihood", "loss_alpha", "alpha", "entropy"]
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        x = dist.rsample((self.samples_mc_entropy,))
        log_p = dist.log_prob(x)
        # log_p: (batch_size, context_len,
        return -log_p.mean(axis=0)

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the loss for the Online Decision Transformer."""
        # extract action targets
        target_actions = tensordict.get(self.tensor_keys.action).detach()

        action_dist = self.actor_network.get_dist(
            tensordict, params=self.actor_network_params
        )

        loss_log_likelihood = action_dist.log_prob(target_actions).mean()
        entropy = self.get_entropy_bonus(action_dist).mean()
        loss = -(loss_log_likelihood + self.alpha.detach() * entropy)

        loss_alpha = self.log_alpha.exp() * (entropy - self.target_entropy).detach()

        out = {
            "loss": loss,
            "loss_log_likelihood": -loss_log_likelihood,
            "entropy": entropy.detach(),
            "loss_alpha": loss_alpha,
            "alpha": self.alpha.detach(),
        }
        return TensorDict(out, [])


class DTLoss(LossModule):
    r"""TorchRL implementation of the Online Decision Transformer loss.

    Presented in `"Decision Transformer: Reinforcement Learning via Sequence Modeling" <https://arxiv.org/abs/2106.01345>`

    Args:
        actor_network (ProbabilisticActor): stochastic actor

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            observation (NestedKey): The input tensordict key where the observation is expected.
                Defaults to ``"observation"``.
            return_to_go (NestedKey): The input tensordict key where the return_to_go is expected.
                Defaults to ``"return_to_go"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
        """

        action: NestedKey = "action"
        observation: NestedKey = "observation"
        return_to_go: NestedKey = "return_to_go"
        done: NestedKey = "done"

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        actor_network: ProbabilisticActor,
    ) -> None:
        self._in_keys = None
        self._out_keys = None
        super().__init__()

        # Actor Network
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=False,
            funs_to_decorate=["forward"],
        )

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.return_to_go),
            ("next", self.tensor_keys.done),
            *self.tensor_keys.action,
            *[("next", key) for key in self.tensor_keys.action],
            *self.tensor_keys.observation,
        ]

        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss"]
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the loss for the Online Decision Transformer."""
        # extract action targets
        target_actions = tensordict.get(self.tensor_keys.action).detach()

        pred_actions = self.actor_network(
            tensordict, params=self.actor_network_params
        ).get(self.tensor_keys.action)
        loss = distance_loss(
            pred_actions,
            target_actions,
            loss_function="l2",
        ).mean()
        out = {
            "loss": loss,
        }
        return TensorDict(out, [])
