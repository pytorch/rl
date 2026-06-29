# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import importlib
import re
import warnings
from collections.abc import Callable, Iterable
from copy import copy
from enum import Enum
from typing import Any, TypeVar

import torch
from tensordict import (
    is_tensorclass,
    NestedKey,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from tensordict.nn import TensorDictModule
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import dropout
from torch.utils._pytree import tree_map

try:
    from torch import vmap
except ImportError as err:
    try:
        from functorch import vmap
    except ImportError as err_ft:
        raise err_ft from err
from torchrl._utils import implement_for
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
    MAGAE = "Multi-agent generalized advantage estimate"
    VTrace = "V-trace"


_BUILTIN_VALUE_ESTIMATOR_DEFAULTS: dict[ValueEstimators, dict[str, Any]] = {
    ValueEstimators.TD0: {"gamma": 0.99, "differentiable": True},
    ValueEstimators.TD1: {"gamma": 0.99, "differentiable": True},
    ValueEstimators.TDLambda: {
        "gamma": 0.99,
        "lmbda": 0.95,
        "differentiable": True,
    },
    ValueEstimators.GAE: {"gamma": 0.99, "lmbda": 0.95, "differentiable": True},
    ValueEstimators.MAGAE: {"gamma": 0.99, "lmbda": 0.95, "differentiable": True},
    ValueEstimators.VTrace: {"gamma": 0.99, "differentiable": True},
}


# ---------------------------------------------------------------------------
# Value-estimator registry
# ---------------------------------------------------------------------------
#
# Historically, every loss that wanted to pick between TD0 / GAE / V-Trace /
# etc. shipped its own ``make_value_estimator`` body with a hard-coded
# ``if/elif`` chain that knew the class names, the default kwargs, and any
# per-estimator construction quirks (e.g. V-Trace needs the actor). Adding a
# new estimator therefore meant touching ~15 loss files.
#
# The registry below decouples those three things:
#   - which class implements a given ``ValueEstimators`` enum entry
#   - what default hyper-parameters that class expects
#   - how to wire the estimator against a particular ``LossModule``
#
# Estimators self-register via the :func:`register_value_estimator` decorator
# at class-definition time. Loss modules can then build the right estimator
# with a single call to :func:`build_value_estimator`, regardless of how many
# concrete estimator classes exist.
#
# The registry accepts either an enum value or its lowercase string alias
# (e.g. ``"gae"``), which is convenient for config-driven setups.


class _ValueEstimatorRegistryEntry:
    """One row of the value-estimator registry."""

    __slots__ = ("cls", "default_kwargs")

    def __init__(self, cls: type, default_kwargs: dict) -> None:
        self.cls = cls
        self.default_kwargs = dict(default_kwargs)


_VALUE_ESTIMATOR_REGISTRY: dict[Any, _ValueEstimatorRegistryEntry] = {}


def register_value_estimator(value_type: Any, *, default_kwargs: dict | None = None):
    """Decorator: register an estimator class against a :class:`ValueEstimators` entry.

    Args:
        value_type: the enum entry this class implements.
        default_kwargs: hyperparameter defaults applied when a loss calls
            ``make_value_estimator(value_type)`` without overriding them.

    Example:
        >>> @register_value_estimator(
        ...     ValueEstimators.GAE,
        ...     default_kwargs={"gamma": 0.99, "lmbda": 0.95, "differentiable": True},
        ... )
        ... class GAE(ValueEstimatorBase):
        ...     ...
    """

    def _decorator(cls):
        _VALUE_ESTIMATOR_REGISTRY[value_type] = _ValueEstimatorRegistryEntry(
            cls, default_kwargs or {}
        )
        return cls

    return _decorator


def _value_estimator_aliases() -> list[str]:
    aliases = [member.name.lower() for member in ValueEstimators]
    aliases.extend(
        key.name.lower()
        for key in _VALUE_ESTIMATOR_REGISTRY
        if isinstance(key, Enum) and key.name.lower() not in aliases
    )
    return aliases


def _registered_value_type(value_type) -> bool:
    try:
        return value_type in _VALUE_ESTIMATOR_REGISTRY
    except TypeError:
        return False


def _ensure_builtin_value_estimators_registered() -> None:
    if all(value_type in _VALUE_ESTIMATOR_REGISTRY for value_type in ValueEstimators):
        return
    # Importing the module triggers the registration decorators on the built-in
    # estimator classes. This keeps direct uses of torchrl.objectives.utils
    # independent of import order while avoiding a module-top circular import.
    importlib.import_module("torchrl.objectives.value.advantages")


def _coerce_value_type(value_type):
    """Allow string aliases like ``"gae"`` alongside registered keys."""
    if isinstance(value_type, ValueEstimators):
        return value_type
    if isinstance(value_type, str):
        if _registered_value_type(value_type):
            return value_type
        # Accept both the enum *member* name ("GAE") and a lowercase alias
        # ("gae") for ergonomics with hydra / yaml configs.
        key = value_type.lower()
        for member in ValueEstimators:
            if member.name.lower() == key:
                return member
        for registered_key in _VALUE_ESTIMATOR_REGISTRY:
            if isinstance(registered_key, Enum) and registered_key.name.lower() == key:
                return registered_key
        raise KeyError(
            f"Unknown value estimator alias {value_type!r}. "
            f"Known aliases: {_value_estimator_aliases()}."
        )
    if _registered_value_type(value_type) or isinstance(value_type, Enum):
        return value_type
    raise TypeError(
        f"value_type must be a registered enum value or a string alias, "
        f"got {type(value_type).__name__}."
    )


def get_value_estimator_entry(value_type) -> _ValueEstimatorRegistryEntry:
    """Look up the registry entry for ``value_type`` (enum or string alias)."""
    coerced = _coerce_value_type(value_type)
    if isinstance(coerced, ValueEstimators):
        _ensure_builtin_value_estimators_registered()
    try:
        return _VALUE_ESTIMATOR_REGISTRY[coerced]
    except KeyError as exc:
        raise NotImplementedError(
            f"No value estimator registered for {coerced!r}. "
            "Register one with @register_value_estimator(...) at class definition time."
        ) from exc


def build_value_estimator(loss_module, value_type, **hyperparams):
    """Construct a value estimator for ``loss_module`` using the registry.

    Resolves the class via :func:`get_value_estimator_entry`, merges the
    registry defaults with the caller's ``hyperparams``, then delegates the
    final wiring to ``cls.for_loss(loss_module, **merged)``. Estimator
    subclasses with construction quirks (V-Trace needs the actor network)
    override ``for_loss`` rather than every loss owning the quirk.
    """
    entry = get_value_estimator_entry(value_type)
    merged = {**entry.default_kwargs, **hyperparams}
    return entry.cls.for_loss(loss_module, **merged)


def dispatch_value_estimator(
    loss_module,
    value_type,
    *,
    supported: Iterable[Any],
    tensor_keys: dict[str, NestedKey] | None = None,
    **hyperparams,
):
    """Convenience wrapper for ``make_value_estimator`` bodies.

    Most losses share the exact same dispatch boilerplate:

    1. validate ``value_type`` against a small set of supported estimators;
    2. merge ``self.gamma`` (if any) and registry defaults into ``hyperparams``;
    3. build the estimator with :func:`build_value_estimator`;
    4. apply the loss's ``tensor_keys`` to the estimator via ``set_keys``.

    This helper does all four and assigns the estimator to
    ``loss_module._value_estimator`` and ``loss_module.value_type``.

    Args:
        loss_module: the loss whose value estimator to build.
        value_type: the requested :class:`ValueEstimators` member (or a
            string alias).
        supported: the set of :class:`ValueEstimators` members the loss
            knows how to use. ``value_type`` is checked against this set;
            anything outside raises :class:`NotImplementedError` with a
            message naming both the value type and the loss class.
        tensor_keys: optional dict of ``key_name -> NestedKey`` that gets
            forwarded to ``value_estimator.set_keys(**tensor_keys)``. If
            ``None`` (default), no ``set_keys`` call is made and the caller
            is expected to wire the keys explicitly afterwards.
        **hyperparams: forwarded to :func:`build_value_estimator`.
    """
    supported_set = set(supported)
    coerced = _coerce_value_type(value_type)
    if coerced not in supported_set:
        supported_names = sorted(
            getattr(value_type, "name", str(value_type)) for value_type in supported_set
        )
        raise NotImplementedError(
            f"Value type {coerced!r} is not implemented for loss "
            f"{type(loss_module).__name__}. Supported value types: "
            f"{supported_names}."
        )
    loss_module.value_type = coerced
    hp = dict(hyperparams)
    if hasattr(loss_module, "gamma"):
        hp.setdefault("gamma", loss_module.gamma)
    estimator = build_value_estimator(loss_module, coerced, **hp)
    loss_module._value_estimator = estimator
    if tensor_keys is not None:
        estimator.set_keys(**tensor_keys)
    return estimator


def default_value_kwargs(value_type: ValueEstimators):
    """Default value function keyword argument generator.

    Now reads from :data:`_VALUE_ESTIMATOR_REGISTRY` so any
    :func:`register_value_estimator`-decorated class is picked up
    automatically. Retained as a top-level function for back-compat with
    callers that don't want to touch the registry directly.

    Args:
        value_type (Enum.value): the value function type, from the
        :class:`~torchrl.objectives.utils.ValueEstimators` class.

    Examples:
        >>> kwargs = default_value_kwargs(ValueEstimators.TDLambda)
        {"gamma": 0.99, "lmbda": 0.95, "differentiable": True}
    """
    coerced = _coerce_value_type(value_type)
    if isinstance(coerced, ValueEstimators):
        _ensure_builtin_value_estimators_registered()
        if coerced not in _VALUE_ESTIMATOR_REGISTRY:
            return dict(_BUILTIN_VALUE_ESTIMATOR_DEFAULTS[coerced])
    try:
        return dict(_VALUE_ESTIMATOR_REGISTRY[coerced].default_kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"No value estimator registered for {coerced!r}. "
            "Register one with @register_value_estimator(...) at class definition time."
        ) from exc


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


TensorLike = TypeVar("TensorLike", Tensor, TensorDict)


def distance_loss(
    v1: TensorLike,
    v2: TensorLike,
    loss_function: str,
    strict_shape: bool = True,
) -> TensorLike:
    """Computes a distance loss between two tensors.

    Args:
        v1 (Tensor | TensorDict): a tensor or tensordict with a shape compatible with v2.
        v2 (Tensor | TensorDict): a tensor or tensordict with a shape compatible with v1.
        loss_function (str): One of "l2", "l1" or "smooth_l1" representing which loss function is to be used.
        strict_shape (bool): if False, v1 and v2 are allowed to have a different shape.
            Default is ``True``.

    Returns:
         A tensor or tensordict of the shape v1.view_as(v2) or v2.view_as(v1)
        with values equal to the distance loss between the two.

    """
    if v1.shape != v2.shape and strict_shape:
        raise RuntimeError(
            f"The input tensors or tensordicts have shapes {v1.shape} and {v2.shape} which are incompatible."
        )

    if loss_function == "l2":
        return F.mse_loss(v1, v2, reduction="none")

    if loss_function == "l1":
        return F.l1_loss(v1, v2, reduction="none")

    if loss_function == "smooth_l1":
        return F.smooth_l1_loss(v1, v2, reduction="none")

    raise NotImplementedError(f"Unknown loss {loss_function}.")


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
        eps: float | None = None,
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
                self._params.data.to_module(self.network, preserve_module_state=False)
            else:
                self.network.requires_grad_(False)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.mode:
            if is_dynamo_compiling():
                self._params.to_module(self.network, preserve_module_state=False)
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


def _vmap_func(module, *args, func=None, pseudo_vmap: bool = False, **kwargs):
    try:

        def decorated_module(*module_args_params):
            params = module_args_params[-1]
            module_args = module_args_params[:-1]
            with params.to_module(module, preserve_module_state=False):
                if func is None:
                    r = module(*module_args)
                else:
                    r = getattr(module, func)(*module_args)
                return r

        if not pseudo_vmap:
            return vmap(decorated_module, *args, **kwargs)  # noqa: TOR101
        return _pseudo_vmap(decorated_module, *args, **kwargs)

    except RuntimeError as err:
        if re.match(
            r"vmap: called random operation while in randomness error mode", str(err)
        ):
            raise RuntimeError(
                "Please use <loss_module>.set_vmap_randomness('different') to handle random operations during vmap."
            ) from err


@implement_for("torch", "2.7")
def _pseudo_vmap(
    func: Callable,
    in_dims: Any = 0,
    out_dims: Any = 0,
    randomness: str | None = None,
    *,
    chunk_size=None,
):
    if randomness is not None and randomness not in ("different", "error"):
        raise ValueError(
            f"pseudo_vmap only supports 'different' or 'error' randomness modes, but got {randomness=}. If another mode is required, please "
            "submit an issue in TorchRL."
        )
    from tensordict.nn.functional_modules import _exclude_td_from_pytree

    def _unbind(d, x):
        if d is not None and hasattr(x, "unbind"):
            return x.unbind(d)
        # Generator to reprod the value
        return (copy(x) for _ in range(1000))

    def _stack(d, x):
        if d is not None:
            x = list(x)
            return torch.stack(list(x), d)
        return x

    @functools.wraps(func)
    def new_func(*args, in_dims=in_dims, out_dims=out_dims, **kwargs):
        with _exclude_td_from_pytree():
            # Unbind inputs
            if isinstance(in_dims, int):
                in_dims = (in_dims,) * len(args)
            if isinstance(out_dims, int):
                out_dims = (out_dims,)

            vs = zip(*tuple(tree_map(_unbind, in_dims, args)))
            rs = []
            for v in vs:
                r = func(*v, **kwargs)
                if not isinstance(r, tuple):
                    r = (r,)
                rs.append(r)
            rs = tuple(zip(*rs))
            vs = tuple(tree_map(_stack, out_dims, rs))
            if len(vs) == 1:
                return vs[0]
            return vs

    return new_func


@implement_for("torch", None, "2.7")
def _pseudo_vmap(  # noqa: F811
    func: Callable,
    in_dims: Any = 0,
    out_dims: Any = 0,
    randomness: str | None = None,
    *,
    chunk_size=None,
):
    @functools.wraps(func)
    def new_func(*args, in_dims=in_dims, out_dims=out_dims, **kwargs):
        raise NotImplementedError("This implementation is not supported for torch<2.7")

    return new_func


def _reduce(
    tensor: torch.Tensor,
    reduction: str,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> float | torch.Tensor:
    """Reduces a tensor given the reduction method.

    Args:
        tensor (torch.Tensor): The tensor to reduce.
        reduction (str): The reduction method.
        mask (torch.Tensor, optional): A mask to apply to the tensor before reducing.
        weights (torch.Tensor, optional): Importance sampling weights for weighted reduction.
            When provided with reduction="mean", computes: (tensor * weights).sum() / weights.sum()
            When provided with reduction="sum", computes: (tensor * weights).sum()
            This is used for proper bias correction with prioritized replay buffers.

    Returns:
        float | torch.Tensor: The reduced tensor.
    """
    if reduction == "none":
        if weights is None:
            result = tensor
            if mask is not None:
                result = result[mask]
        elif mask is not None:
            masked_weight = weights[mask]
            masked_tensor = tensor[mask]
            result = masked_tensor * masked_weight
        else:
            result = tensor * weights
    elif reduction == "mean":
        if weights is not None:
            # Weighted average: (tensor * weights).sum() / weights.sum()
            if mask is not None:
                if tensor.shape != weights.shape:
                    raise ValueError(
                        f"Tensor and weights shapes must match, but got {tensor.shape} and {weights.shape}"
                    )
                mask = mask.to(dtype=weights.dtype)
                masked_weight = weights * mask
                result = (tensor * masked_weight).sum() / masked_weight.sum()
            else:
                if tensor.shape != weights.shape:
                    raise ValueError(
                        f"Tensor and weights shapes must match, but got {tensor.shape} and {weights.shape}"
                    )
                result = (tensor * weights).sum() / weights.sum()
        elif mask is not None:
            mask = mask.to(dtype=tensor.dtype)
            result = (tensor * mask).sum() / mask.sum()
        else:
            result = tensor.mean()
    elif reduction == "sum":
        if weights is not None:
            # Weighted sum: (tensor * weights).sum()
            if mask is not None:
                if tensor.shape != weights.shape:
                    raise ValueError(
                        f"Tensor and weights shapes must match, but got {tensor.shape} and {weights.shape}"
                    )
                mask = mask.to(dtype=weights.dtype)
                result = (tensor * weights * mask).sum()
            else:
                if tensor.shape != weights.shape:
                    raise ValueError(
                        f"Tensor and weights shapes must match, but got {tensor.shape} and {weights.shape}"
                    )
                result = (tensor * weights).sum()
        elif mask is not None:
            mask = mask.to(dtype=tensor.dtype)
            result = (tensor * mask).sum()
        else:
            result = tensor.sum()
    else:
        raise NotImplementedError(f"Unknown reduction method {reduction}")
    return result


def _clip_value_loss(
    old_state_value: torch.Tensor | TensorDict,
    state_value: torch.Tensor | TensorDict,
    clip_value: torch.Tensor | TensorDict,
    target_return: torch.Tensor | TensorDict,
    loss_value: torch.Tensor | TensorDict,
    loss_critic_type: str,
) -> tuple[torch.Tensor | TensorDict, torch.Tensor]:
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
    loss_value = torch.maximum(loss_value, loss_value_clipped)
    return loss_value, clip_fraction


def _validate_clip_epsilon(
    clip_epsilon: float | tuple[float, float]
) -> tuple[float, float]:
    """Normalize and validate a PPO clip threshold.

    Accepts a float (symmetric clipping) or a ``(eps_low, eps_high)`` pair
    (asymmetric, DAPO Clip-Higher style) and returns the validated
    ``(eps_low, eps_high)`` bounds.
    """
    if isinstance(clip_epsilon, (tuple, list)):
        if len(clip_epsilon) != 2:
            raise ValueError(
                f"clip_epsilon tuple must have length 2, got {clip_epsilon}."
            )
        eps_low, eps_high = (float(clip_epsilon[0]), float(clip_epsilon[1]))
    else:
        eps_low = eps_high = float(clip_epsilon)
    if eps_low < 0 or eps_high < 0:
        raise ValueError(
            f"clip_epsilon values must be non-negative, got ({eps_low}, {eps_high})."
        )
    if eps_low >= 1.0:
        raise ValueError(
            f"clip_epsilon low must be < 1 (to keep 1 - eps_low > 0), got {eps_low}."
        )
    return eps_low, eps_high


def _get_default_device(net):
    for p in net.parameters():
        return p.device
    else:
        return getattr(torch, "get_default_device", lambda: torch.device("cpu"))()


def _make_writable(td: TensorDictBase) -> TensorDictBase:
    """Returns a container that accepts new keys, for use as network scratch.

    Networks write their ``out_keys`` into the tensordict they run on. A
    tensorclass has a fixed schema and rejects keys that were not declared as
    fields, so it is converted to a plain :class:`~tensordict.TensorDict`.
    Dynamic containers (``TensorDict``, lazy stacks) already accept new keys and
    are returned unchanged to avoid a needless clone on the hot path.
    """
    return td.to_tensordict() if is_tensorclass(td) else td


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


def _maybe_get_or_select(
    td,
    key_or_keys,
    target_shape=None,
    padding_side: str = "left",
    padding_value: int = 0,
):
    if isinstance(key_or_keys, (str, tuple)):
        return td.get(
            key_or_keys,
            as_padded_tensor=True,
            padding_side=padding_side,
            padding_value=padding_value,
        )
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
