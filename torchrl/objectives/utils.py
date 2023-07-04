# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import warnings
from enum import Enum
from typing import Iterable, Optional, Union

import torch
from tensordict.nn import TensorDictModule
from tensordict.tensordict import is_tensor_collection, TensorDict, TensorDictBase
from torch import nn, Tensor
from torch.nn import functional as F

from torchrl.envs.utils import step_mdp

_GAMMA_LMBDA_DEPREC_WARNING = (
    "Passing gamma / lambda parameters through the loss constructor "
    "is deprecated and will be removed soon. To customize your value function, "
    "run `loss_module.make_value_estimator(ValueEstimators.<value_fun>, gamma=val)`."
)


class ValueEstimators(Enum):
    """Value function enumerator for custom-built estimators.

    Allows for a flexible usage of various value functions when the loss module
    allows it.

    Examples:
        >>> dqn_loss = DQNLoss(actor)
        >>> dqn_loss.make_value_estimator(ValueEstimators.TD0, gamma=0.9)

    """

    TD0 = "Bootstrapped TD (1-step return)"
    TD1 = "TD(1) (infinity-step return)"
    TDLambda = "TD(lambda)"
    GAE = "Generalized advantage estimate"


def default_value_kwargs(value_type: ValueEstimators):
    """Default value function keyword argument generator.

    Args:
        value_type (Enum.value): the value function type, from the
        :class:`~torchrl.objectives.utils.ValueEstimators` class.

    Examples:
        >>> kwargs = default_value_kwargs(ValueEstimators.TDLambda)
        {"gamma": 0.99, "lmbda": 0.95}

    """
    if value_type == ValueEstimators.TD1:
        return {"gamma": 0.99, "differentiable": True}
    elif value_type == ValueEstimators.TD0:
        return {"gamma": 0.99, "differentiable": True}
    elif value_type == ValueEstimators.GAE:
        return {"gamma": 0.99, "lmbda": 0.95, "differentiable": True}
    elif value_type == ValueEstimators.TDLambda:
        return {"gamma": 0.99, "lmbda": 0.95, "differentiable": True}
    else:
        raise NotImplementedError(f"Unknown value type {value_type}.")


class _context_manager:
    def __init__(self, value=True):
        self.value = value
        self.prev = []

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_context


def distance_loss(
    v1: torch.Tensor,
    v2: torch.Tensor,
    loss_function: str,
    strict_shape: bool = True,
) -> torch.Tensor:
    """Computes a distance loss between two tensors.

    Args:
        v1 (Tensor): a tensor with a shape compatible with v2
        v2 (Tensor): a tensor with a shape compatible with v1
        loss_function (str): One of "l2", "l1" or "smooth_l1" representing which loss function is to be used.
        strict_shape (bool): if False, v1 and v2 are allowed to have a different shape.
            Default is ``True``.

    Returns:
         A tensor of the shape v1.view_as(v2) or v2.view_as(v1) with values equal to the distance loss between the
        two.

    """
    if v1.shape != v2.shape and strict_shape:
        raise RuntimeError(
            f"The input tensors have shapes {v1.shape} and {v2.shape} which are incompatible."
        )

    if loss_function == "l2":
        value_loss = F.mse_loss(
            v1,
            v2,
            reduction="none",
        )

    elif loss_function == "l1":
        value_loss = F.l1_loss(
            v1,
            v2,
            reduction="none",
        )

    elif loss_function == "smooth_l1":
        value_loss = F.smooth_l1_loss(
            v1,
            v2,
            reduction="none",
        )
    else:
        raise NotImplementedError(f"Unknown loss {loss_function}")
    return value_loss


class TargetNetUpdater:
    """An abstract class for target network update in Double DQN/DDPG.

    Args:
        loss_module (DQNLoss or DDPGLoss): loss module where the target network should be updated.

    """

    def __init__(
        self,
        loss_module: "LossModule",  # noqa: F821
    ):
        _has_update_associated = getattr(loss_module, "_has_update_associated", None)
        loss_module._has_update_associated = True
        try:
            _target_names = []
            # for properties
            for name in loss_module.__class__.__dict__:
                if (
                    name.startswith("target_")
                    and (name.endswith("params") or name.endswith("buffers"))
                    and (getattr(loss_module, name) is not None)
                ):
                    _target_names.append(name)

            # for regular lists: raise an exception
            for name in loss_module.__dict__:
                if (
                    name.startswith("target_")
                    and (name.endswith("params") or name.endswith("buffers"))
                    and (getattr(loss_module, name) is not None)
                ):
                    raise RuntimeError(
                        "Your module seems to have a target tensor list contained "
                        "in a non-dynamic structure (such as a list). If the "
                        "module is cast onto a device, the reference to these "
                        "tensors will be lost."
                    )

            if len(_target_names) == 0:
                raise RuntimeError(
                    "Did not find any target parameters or buffers in the loss module."
                )

            _source_names = ["".join(name.split("target_")) for name in _target_names]

            for _source in _source_names:
                try:
                    getattr(loss_module, _source)
                except AttributeError as err:
                    raise RuntimeError(
                        f"Incongruent target and source parameter lists: "
                        f"{_source} is not an attribute of the loss_module"
                    ) from err

            self._target_names = _target_names
            self._source_names = _source_names
            self.loss_module = loss_module
            self.initialized = False
            self.init_()
            _has_update_associated = True
        finally:
            loss_module._has_update_associated = _has_update_associated

    @property
    def _targets(self):
        return TensorDict(
            {name: getattr(self.loss_module, name) for name in self._target_names},
            [],
        )

    @property
    def _sources(self):
        return TensorDict(
            {name: getattr(self.loss_module, name) for name in self._source_names},
            [],
        )

    def init_(self) -> None:
        if self.initialized:
            warnings.warn("Updated already initialized.")
        found_distinct = False
        self._distinct = {}
        for key, source in self._sources.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            key = ("target_" + key[0], *key[1:])
            target = self._targets[key]
            # for p_source, p_target in zip(source, target):
            if target.requires_grad:
                raise RuntimeError("the target parameter is part of a graph.")
            self._distinct[key] = target.data_ptr() != source.data.data_ptr()
            found_distinct = found_distinct or self._distinct[key]
            target.data.copy_(source.data)
        if not found_distinct:
            raise RuntimeError(
                f"The target and source data are identical for all params. "
                "Have you created proper target parameters? "
                "If the loss has a ``delay_value`` kwarg, make sure to set it "
                "to True if it is not done by default. "
                f"If no target parameter is needed, do not use a target updater such as {type(self)}."
            )

        self.initialized = True

    def step(self) -> None:
        if not self.initialized:
            raise Exception(
                f"{self.__class__.__name__} must be "
                f"initialized (`{self.__class__.__name__}.init_()`) before calling step()"
            )
        for key, source in self._sources.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            key = ("target_" + key[0], *key[1:])
            if not self._distinct[key]:
                continue
            target = self._targets[key]
            if target.requires_grad:
                raise RuntimeError("the target parameter is part of a graph.")
            if target.is_leaf:
                self._step(source, target)
            else:
                target.copy_(source)

    def _step(self, p_source: Tensor, p_target: Tensor) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        string = (
            f"{self.__class__.__name__}(sources={self._sources}, targets="
            f"{self._targets})"
        )
        return string


class SoftUpdate(TargetNetUpdater):
    r"""A soft-update class for target network update in Double DQN/DDPG.

    This was proposed in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING", https://arxiv.org/pdf/1509.02971.pdf

    One and only one decay factor (tau or eps) must be specified.

    Args:
        loss_module (DQNLoss or DDPGLoss): loss module where the target network should be updated.
        eps (scalar): epsilon in the update equation:
            .. math::

                \theta_t = \theta_{t-1} * \epsilon + \theta_t * (1-\epsilon)

            Exclusive with ``tau``.
        tau (scalar): Polyak tau. It is equal to ``1-eps``, and exclusive with it.
    """

    def __init__(
        self,
        loss_module: Union[
            "DQNLoss",  # noqa: F821
            "DDPGLoss",  # noqa: F821
            "SACLoss",  # noqa: F821
            "REDQLoss",  # noqa: F821
            "TD3Loss",  # noqa: F821
        ],
        *,
        eps: float = None,
        tau: Optional[float] = None,
    ):
        if eps is None and tau is None:
            warnings.warn(
                "Neither eps nor tau was provided. Taking the default value "
                "eps=0.999. This behaviour will soon be deprecated.",
                category=DeprecationWarning,
            )
            eps = 0.999
        if (eps is None) ^ (tau is None):
            if eps is None:
                eps = 1 - tau
        else:
            raise ValueError("One and only one argument (tau or eps) can be specified.")
        if eps < 0.5:
            warnings.warn(
                "Found an eps value < 0.5, which is unexpected. "
                "You may want to use the `tau` keyword argument instead."
            )
        if not (eps <= 1.0 and eps >= 0.0):
            raise ValueError(
                f"Got eps = {eps} when it was supposed to be between 0 and 1."
            )
        super(SoftUpdate, self).__init__(loss_module)
        self.eps = eps

    def _step(self, p_source: Tensor, p_target: Tensor) -> None:
        p_target.data.copy_(p_target.data * self.eps + p_source.data * (1 - self.eps))


class HardUpdate(TargetNetUpdater):
    """A hard-update class for target network update in Double DQN/DDPG (by contrast with soft updates).

    This was proposed in the original Double DQN paper: "Deep Reinforcement Learning with Double Q-learning",
    https://arxiv.org/abs/1509.06461.

    Args:
        loss_module (DQNLoss or DDPGLoss): loss module where the target network should be updated.

    Keyword Args:
        value_network_update_interval (scalar): how often the target network should be updated.
            default: 1000
    """

    def __init__(
        self,
        loss_module: Union["DQNLoss", "DDPGLoss", "SACLoss", "TD3Loss"],  # noqa: F821
        *,
        value_network_update_interval: float = 1000,
    ):
        super(HardUpdate, self).__init__(loss_module)
        self.value_network_update_interval = value_network_update_interval
        self.counter = 0

    def _step(self, p_source: Tensor, p_target: Tensor) -> None:
        if self.counter == self.value_network_update_interval:
            p_target.data.copy_(p_source.data)

    def step(self) -> None:
        super().step()
        if self.counter == self.value_network_update_interval:
            self.counter = 0
        else:
            self.counter += 1


class hold_out_net(_context_manager):
    """Context manager to hold a network out of a computational graph."""

    def __init__(self, network: nn.Module) -> None:
        self.network = network
        try:
            self.p_example = next(network.parameters())
        except (AttributeError, StopIteration):
            self.p_example = torch.tensor([])
        self._prev_state = []

    def __enter__(self) -> None:
        self._prev_state.append(self.p_example.requires_grad)
        self.network.requires_grad_(False)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.network.requires_grad_(self._prev_state.pop())


class hold_out_params(_context_manager):
    """Context manager to hold a list of parameters out of a computational graph."""

    def __init__(self, params: Iterable[Tensor]) -> None:
        if isinstance(params, TensorDictBase):
            self.params = params.detach()
        else:
            self.params = tuple(p.detach() for p in params)

    def __enter__(self) -> None:
        return self.params

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


@torch.no_grad()
def next_state_value(
    tensordict: TensorDictBase,
    operator: Optional[TensorDictModule] = None,
    next_val_key: str = "state_action_value",
    gamma: float = 0.99,
    pred_next_val: Optional[Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Computes the next state value (without gradient) to compute a target value.

    The target value is ususally used to compute a distance loss (e.g. MSE):
        L = Sum[ (q_value - target_value)^2 ]
    The target value is computed as
        r + gamma ** n_steps_to_next * value_next_state
    If the reward is the immediate reward, n_steps_to_next=1. If N-steps rewards are used, n_steps_to_next is gathered
    from the input tensordict.

    Args:
        tensordict (TensorDictBase): Tensordict containing a reward and done key (and a n_steps_to_next key for n-steps
            rewards).
        operator (ProbabilisticTDModule, optional): the value function operator. Should write a 'next_val_key'
            key-value in the input tensordict when called. It does not need to be provided if pred_next_val is given.
        next_val_key (str, optional): key where the next value will be written.
            Default: 'state_action_value'
        gamma (float, optional): return discount rate.
            default: 0.99
        pred_next_val (Tensor, optional): the next state value can be provided if it is not computed with the operator.

    Returns:
        a Tensor of the size of the input tensordict containing the predicted value state.

    """
    if "steps_to_next_obs" in tensordict.keys():
        steps_to_next_obs = tensordict.get("steps_to_next_obs").squeeze(-1)
    else:
        steps_to_next_obs = 1

    rewards = tensordict.get(("next", "reward")).squeeze(-1)
    done = tensordict.get(("next", "done")).squeeze(-1)
    if done.all() or gamma == 0:
        return rewards

    if pred_next_val is None:
        next_td = step_mdp(tensordict)  # next_observation -> observation
        next_td = next_td.select(*operator.in_keys)
        operator(next_td, **kwargs)
        pred_next_val_detach = next_td.get(next_val_key).squeeze(-1)
    else:
        pred_next_val_detach = pred_next_val.squeeze(-1)
    done = done.to(torch.float)
    target_value = (1 - done) * pred_next_val_detach
    rewards = rewards.to(torch.float)
    target_value = rewards + (gamma**steps_to_next_obs) * target_value
    return target_value


def _cache_values(fun):
    """Caches the tensordict returned by a property."""
    name = fun.__name__

    def new_fun(self, netname=None):
        __dict__ = self.__dict__
        _cache = __dict__["_cache"]
        attr_name = name
        if netname is not None:
            attr_name += "_" + netname
        if attr_name in _cache:
            out = _cache[attr_name]
            return out
        if netname is not None:
            out = fun(self, netname)
        else:
            out = fun(self)
        if is_tensor_collection(out):
            out.lock_()
        _cache[attr_name] = out
        return out

    return new_fun
