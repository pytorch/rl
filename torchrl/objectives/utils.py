# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import re
import warnings
from enum import Enum
from typing import Iterable

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase, unravel_key
from tensordict.nn import TensorDictModule
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import dropout

try:
    from torch import vmap
except ImportError as err:
    try:
        from functorch import vmap
    except ImportError as err_ft:
        raise err_ft from err
from torchrl.envs.utils import step_mdp

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling

_GAMMA_LMBDA_DEPREC_ERROR = (
    "Passing gamma / lambda parameters through the loss constructor "
    "is a deprecated feature. To customize your value function, "
    "run `loss_module.make_value_estimator(ValueEstimators.<value_fun>, gamma=val)`."
)

RANDOM_MODULE_LIST = (dropout._DropoutNd,)


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
    VTrace = "V-trace"


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
    elif value_type == ValueEstimators.VTrace:
        return {"gamma": 0.99, "differentiable": True}
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
        loss_module: LossModule,  # noqa: F821
    ):
        from torchrl.objectives.common import LossModule

        if not isinstance(loss_module, LossModule):
            raise ValueError("The loss_module must be a LossModule instance.")
        _has_update_associated = getattr(loss_module, "_has_update_associated", None)
        for k in loss_module._has_update_associated.keys():
            loss_module._has_update_associated[k] = True
        try:
            _target_names = []
            for name, _ in loss_module.named_children():
                # the TensorDictParams is a nn.Module instance
                if name.startswith("target_") and name.endswith("_params"):
                    _target_names.append(name)

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
            for k in loss_module._has_update_associated.keys():
                loss_module._has_update_associated[k] = _has_update_associated

    @property
    def _targets(self):
        targets = self.__dict__.get("_targets_val", None)
        if targets is None:
            targets = self.__dict__["_targets_val"] = TensorDict(
                {name: getattr(self.loss_module, name) for name in self._target_names},
                [],
            )
        return targets

    @_targets.setter
    def _targets(self, targets):
        self.__dict__["_targets_val"] = targets

    @property
    def _sources(self):
        sources = self.__dict__.get("_sources_val", None)
        if sources is None:
            sources = self.__dict__["_sources_val"] = TensorDict(
                {name: getattr(self.loss_module, name) for name in self._source_names},
                [],
            )
        return sources

    @_sources.setter
    def _sources(self, sources):
        self.__dict__["_sources_val"] = sources

    def init_(self) -> None:
        if self.initialized:
            warnings.warn("Updated already initialized.")
        found_distinct = False
        self._distinct_and_params = {}
        for key, source in self._sources.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            key = ("target_" + key[0], *key[1:])
            target = self._targets[key]
            # for p_source, p_target in zip(source, target):
            if target.requires_grad:
                raise RuntimeError("the target parameter is part of a graph.")
            self._distinct_and_params[key] = (
                target.is_leaf
                and source.requires_grad
                and target.data_ptr() != source.data.data_ptr()
            )
            found_distinct = found_distinct or self._distinct_and_params[key]
            target.data.copy_(source.data)
        if not found_distinct:
            raise RuntimeError(
                f"The target and source data are identical for all params. "
                "Have you created proper target parameters? "
                "If the loss has a ``delay_value`` kwarg, make sure to set it "
                "to True if it is not done by default. "
                f"If no target parameter is needed, do not use a target updater such as {type(self)}."
            )

        # filter the target_ out
        def filter_target(key):
            if isinstance(key, tuple):
                return (filter_target(key[0]), *key[1:])
            return key[7:]

        self._sources = self._sources.select(
            *[
                filter_target(key)
                for (key, val) in self._distinct_and_params.items()
                if val
            ]
        ).lock_()
        self._targets = self._targets.select(
            *(key for (key, val) in self._distinct_and_params.items() if val)
        ).lock_()

        self.initialized = True

    def step(self) -> None:
        if not self.initialized:
            raise Exception(
                f"{self.__class__.__name__} must be "
                f"initialized (`{self.__class__.__name__}.init_()`) before calling step()"
            )
        for key, param in self._sources.items():
            target = self._targets.get(f"target_{key}")
            if target.requires_grad:
                raise RuntimeError("the target parameter is part of a graph.")
            self._step(param, target)

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
        loss_module: (
            DQNLoss  # noqa: F821
            | DDPGLoss  # noqa: F821
            | SACLoss  # noqa: F821
            | REDQLoss  # noqa: F821
            | TD3Loss  # noqa: F821  # noqa: F821
        ),
        *,
        eps: float = None,
        tau: float | None = None,
    ):
        if eps is None and tau is None:
            raise RuntimeError(
                "Neither eps nor tau was provided. This behavior is deprecated.",
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
        super().__init__(loss_module)
        self.eps = eps

    def _step(
        self, p_source: Tensor | TensorDictBase, p_target: Tensor | TensorDictBase
    ) -> None:
        p_target.data.lerp_(p_source.data, 1 - self.eps)


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
        loss_module: DQNLoss | DDPGLoss | SACLoss | TD3Loss,  # noqa: F821
        *,
        value_network_update_interval: float = 1000,
    ):
        super().__init__(loss_module)
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
        for p in network.parameters():
            self.mode = p.requires_grad
            break
        else:
            self.mode = True

    def __enter__(self) -> None:
        if self.mode:
            if is_dynamo_compiling():
                self._params = TensorDict.from_module(self.network)
                self._params.data.to_module(self.network)
            else:
                self.network.requires_grad_(False)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.mode:
            if is_dynamo_compiling():
                self._params.to_module(self.network)
            else:
                self.network.requires_grad_()


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
    operator: TensorDictModule | None = None,
    next_val_key: str = "state_action_value",
    gamma: float = 0.99,
    pred_next_val: Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Computes the next state value (without gradient) to compute a target value.

    The target value is usually used to compute a distance loss (e.g. MSE):
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
        gamma (:obj:`float`, optional): return discount rate.
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


def _cache_values(func):
    """Caches the tensordict returned by a property."""
    name = func.__name__

    @functools.wraps(func)
    def new_func(self, netname=None):
        if is_dynamo_compiling():
            if netname is not None:
                return func(self, netname)
            else:
                return func(self)
        __dict__ = self.__dict__
        _cache = __dict__.setdefault("_cache", {})
        attr_name = name
        if netname is not None:
            attr_name += "_" + netname
        if attr_name in _cache:
            out = _cache[attr_name]
            return out
        if netname is not None:
            out = func(self, netname)
        else:
            out = func(self)
        # TODO: decide what to do with locked tds in functional calls
        # if is_tensor_collection(out):
        #     out.lock_()
        _cache[attr_name] = out
        return out

    return new_func


def _vmap_func(module, *args, func=None, **kwargs):
    try:

        def decorated_module(*module_args_params):
            params = module_args_params[-1]
            module_args = module_args_params[:-1]
            with params.to_module(module):
                if func is None:
                    return module(*module_args)
                else:
                    return getattr(module, func)(*module_args)

        return vmap(decorated_module, *args, **kwargs)  # noqa: TOR101

    except RuntimeError as err:
        if re.match(
            r"vmap: called random operation while in randomness error mode", str(err)
        ):
            raise RuntimeError(
                "Please use <loss_module>.set_vmap_randomness('different') to handle random operations during vmap."
            ) from err


def _reduce(tensor: torch.Tensor, reduction: str) -> float | torch.Tensor:
    """Reduces a tensor given the reduction method."""
    if reduction == "none":
        result = tensor
    elif reduction == "mean":
        result = tensor.mean()
    elif reduction == "sum":
        result = tensor.sum()
    else:
        raise NotImplementedError(f"Unknown reduction method {reduction}")
    return result


def _clip_value_loss(
    old_state_value: torch.Tensor,
    state_value: torch.Tensor,
    clip_value: torch.Tensor,
    target_return: torch.Tensor,
    loss_value: torch.Tensor,
    loss_critic_type: str,
):
    """Value clipping method for loss computation.

    This method computes a clipped state value from the old state value and the state value,
    and returns the most pessimistic value prediction between clipped and non-clipped options.
    It also computes the clip fraction.
    """
    pre_clipped = state_value - old_state_value
    clipped = pre_clipped.clamp(-clip_value, clip_value)
    with torch.no_grad():
        clip_fraction = (pre_clipped != clipped).to(state_value.dtype).mean()
    state_value_clipped = old_state_value + clipped
    loss_value_clipped = distance_loss(
        target_return,
        state_value_clipped,
        loss_function=loss_critic_type,
    )
    # Chose the most pessimistic value prediction between clipped and non-clipped
    loss_value = torch.max(loss_value, loss_value_clipped)
    return loss_value, clip_fraction


def _get_default_device(net):
    for p in net.parameters():
        return p.device
    else:
        return getattr(torch, "get_default_device", lambda: torch.device("cpu"))()


def group_optimizers(*optimizers: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Groups multiple optimizers into a single one.

    All optimizers are expected to have the same type.
    """
    cls = None
    params = []
    for optimizer in optimizers:
        if optimizer is None:
            continue
        if cls is None:
            cls = type(optimizer)
        if cls is not type(optimizer):
            raise ValueError("Cannot group optimizers of different type.")
        params.extend(optimizer.param_groups)
    return cls(params)


def _sum_td_features(data: TensorDictBase) -> torch.Tensor:
    # Sum all features and return a tensor
    return data.sum(dim="feature", reduce=True)


def _maybe_get_or_select(td, key_or_keys, target_shape=None):
    if isinstance(key_or_keys, (str, tuple)):
        return td.get(key_or_keys, as_padded_tensor=True)
    result = td.select(*key_or_keys)
    if target_shape is not None and result.shape != target_shape:
        result.batch_size = target_shape
    return result


def _maybe_add_or_extend_key(
    tensor_keys: list[NestedKey],
    key_or_list_of_keys: NestedKey | list[NestedKey],
    prefix: NestedKey = None,
):
    if prefix is not None:
        if isinstance(key_or_list_of_keys, NestedKey):
            tensor_keys.append(unravel_key((prefix, key_or_list_of_keys)))
        else:
            tensor_keys.extend([unravel_key((prefix, k)) for k in key_or_list_of_keys])
        return
    if isinstance(key_or_list_of_keys, NestedKey):
        tensor_keys.append(key_or_list_of_keys)
    else:
        tensor_keys.extend(key_or_list_of_keys)
