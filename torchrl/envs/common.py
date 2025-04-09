# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import warnings
from copy import deepcopy
from functools import partial, wraps
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn as nn
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    TensorDictBase,
    unravel_key,
)
from tensordict.base import _is_leaf_nontensor, NO_DEFAULT
from tensordict.utils import is_non_tensor, NestedKey
from torchrl._utils import (
    _ends_with,
    _make_ordinal_device,
    _replace_last,
    implement_for,
    prod,
    seed_generator,
)

from torchrl.data.tensor_specs import (
    Categorical,
    Composite,
    NonTensor,
    TensorSpec,
    Unbounded,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.utils import (
    _make_compatible_policy,
    _repr_by_depth,
    _StepMDP,
    _terminated_or_truncated,
    _update_during_reset,
    check_env_specs as check_env_specs_func,
    get_available_libraries,
)

LIBRARIES = get_available_libraries()


def _tensor_to_np(t):
    return t.detach().cpu().numpy()


dtype_map = {
    torch.float: np.float32,
    torch.double: np.float64,
    torch.bool: bool,
}


def _maybe_unlock(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        is_locked = self.is_spec_locked
        try:
            if is_locked:
                self.set_spec_lock_(False)
            result = func(self, *args, **kwargs)
        finally:
            if is_locked:
                if not hasattr(self, "_cache"):
                    self._cache = {}
                self._cache.clear()
                self.set_spec_lock_(True)
        return result

    return wrapper


def _cache_value(func):
    """Caches the result of the decorated function in env._cache dictionary."""
    func_name = func.__name__

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_spec_locked:
            return func(self, *args, **kwargs)
        result = self.__dict__.setdefault("_cache", {}).get(func_name, NO_DEFAULT)
        if result is NO_DEFAULT:
            result = func(self, *args, **kwargs)
            self.__dict__.setdefault("_cache", {})[func_name] = result
        return result

    return wrapper


def _clear_cache_when_set(func):
    """A decorator for EnvBase methods that should clear the caches when called."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # if there's no cache we'll just recompute the value
        if "_cache" not in self.__dict__:
            self._cache = {}
        else:
            self._cache.clear()
        result = func(self, *args, **kwargs)
        self._cache.clear()
        return result

    return wrapper


class EnvMetaData:
    """A class for environment meta-data storage and passing in multiprocessed settings."""

    def __init__(
        self,
        *,
        tensordict: TensorDictBase,
        specs: Composite,
        batch_size: torch.Size,
        env_str: str,
        device: torch.device,
        batch_locked: bool,
        device_map: dict,
    ):
        self.device = device
        self.tensordict = tensordict
        self.specs = specs
        self.batch_size = batch_size
        self.env_str = env_str
        self.batch_locked = batch_locked
        self.device_map = device_map
        self.has_dynamic_specs = _has_dynamic_specs(specs)

    @property
    def tensordict(self) -> TensorDictBase:
        td = self._tensordict.copy()
        if td.device != self.device:
            if self.device is None:
                return td.clear_device_()
            else:
                return td.to(self.device)
        return td

    @property
    def specs(self):
        return self._specs.to(self.device)

    @tensordict.setter
    def tensordict(self, value: TensorDictBase):
        self._tensordict = value.to("cpu")

    @specs.setter
    def specs(self, value: Composite):
        self._specs = value.to("cpu")

    @staticmethod
    def metadata_from_env(env) -> EnvMetaData:
        tensordict = env.fake_tensordict().clone()

        for done_key in env.done_keys:
            tensordict.set(
                _replace_last(done_key, "_reset"),
                torch.zeros_like(tensordict.get(("next", done_key))),
            )

        specs = env.specs.to("cpu")

        batch_size = env.batch_size
        try:
            env_str = str(env)
        except Exception:
            env_str = f"{env.__class__.__name__}()"
        device = env.device
        specs = specs.to("cpu")
        batch_locked = env.batch_locked
        # we need to save the device map, as the tensordict will be placed on cpu
        device_map = {}

        def fill_device_map(name, val, device_map=device_map):
            device_map[name] = val.device

        tensordict.named_apply(fill_device_map, nested_keys=True, filter_empty=True)
        return EnvMetaData(
            tensordict=tensordict,
            specs=specs,
            batch_size=batch_size,
            env_str=env_str,
            device=device,
            batch_locked=batch_locked,
            device_map=device_map,
        )

    def expand(self, *size: int) -> EnvMetaData:
        tensordict = self.tensordict.expand(*size).clone()
        batch_size = torch.Size(list(size))
        return EnvMetaData(
            tensordict=tensordict,
            specs=self.specs.expand(*size),
            batch_size=batch_size,
            env_str=self.env_str,
            device=self.device,
            batch_locked=self.batch_locked,
            device_map=self.device_map,
        )

    def clone(self):
        return EnvMetaData(
            tensordict=self.tensordict.clone(),
            specs=self.specs.clone(),
            batch_size=torch.Size([*self.batch_size]),
            env_str=deepcopy(self.env_str),
            device=self.device,
            batch_locked=self.batch_locked,
            device_map=self.device_map,
        )

    def to(self, device: DEVICE_TYPING) -> EnvMetaData:
        if device is not None:
            device = _make_ordinal_device(torch.device(device))
            device_map = {key: device for key in self.device_map}
        tensordict = self.tensordict.contiguous().to(device)
        specs = self.specs.to(device)
        return EnvMetaData(
            tensordict=tensordict,
            specs=specs,
            batch_size=self.batch_size,
            env_str=self.env_str,
            device=device,
            batch_locked=self.batch_locked,
            device_map=device_map,
        )

    def __getitem__(self, item):
        from tensordict.utils import _getitem_batch_size

        return EnvMetaData(
            tensordict=self.tensordict[item],
            specs=self.specs[item],
            batch_size=_getitem_batch_size(self.batch_size, item),
            env_str=self.env_str,
            device=self.device,
            batch_locked=self.batch_locked,
            device_map=self.device_map,
        )


class _EnvPostInit(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        spec_locked = kwargs.pop("spec_locked", True)
        auto_reset = kwargs.pop("auto_reset", False)
        auto_reset_replace = kwargs.pop("auto_reset_replace", True)
        instance: EnvBase = super().__call__(*args, **kwargs)
        if "_cache" not in instance.__dict__:
            instance._cache = {}

        if spec_locked:
            instance.input_spec.lock_(recurse=True)
            instance.output_spec.lock_(recurse=True)
        instance._is_spec_locked = spec_locked

        # we create the done spec by adding a done/terminated entry if one is missing
        instance._create_done_specs()
        # we access lazy attributed to make sure they're built properly.
        # This isn't done in `__init__` because we don't know if super().__init__
        # will be called before or after the specs, batch size etc are set.
        _ = instance.done_spec
        _ = instance.reward_keys
        # _ = instance.action_keys
        _ = instance.state_spec
        if auto_reset:
            from torchrl.envs.transforms.transforms import (
                AutoResetEnv,
                AutoResetTransform,
            )

            return AutoResetEnv(
                instance, AutoResetTransform(replace=auto_reset_replace)
            )

        done_keys = set(instance.full_done_spec.keys(True, True))
        obs_keys = set(instance.full_observation_spec.keys(True, True))
        reward_keys = set(instance.full_reward_spec.keys(True, True))
        # state_keys can match obs_keys so we don't test that
        action_keys = set(instance.full_action_spec.keys(True, True))
        state_keys = set(instance.full_state_spec.keys(True, True))
        total_set = set()
        for keyset in (done_keys, obs_keys, reward_keys):
            if total_set.intersection(keyset):
                raise RuntimeError(
                    f"The set of keys of one spec collides (culprit: {total_set.intersection(keyset)}) with another."
                )
            total_set = total_set.union(keyset)
        total_set = set()
        for keyset in (state_keys, action_keys):
            if total_set.intersection(keyset):
                raise RuntimeError(
                    f"The set of keys of one spec collides (culprit: {total_set.intersection(keyset)}) with another."
                )
            total_set = total_set.union(keyset)
        return instance


class EnvBase(nn.Module, metaclass=_EnvPostInit):
    """Abstract environment parent class.

    Keyword Args:
        device (torch.device): The device of the environment. Deviceless environments
            are allowed (device=None). If not ``None``, all specs will be cast
            on that device and it is expected that all inputs and outputs will
            live on that device.
            Defaults to ``None``.
        batch_size (torch.Size or equivalent, optional): batch-size of the environment.
            Corresponds to the leading dimension of all the input and output
            tensordicts the environment reads and writes. Defaults to an empty batch-size.
        run_type_checks (bool, optional): If ``True``, type-checks will occur
            at every reset and every step. Defaults to ``False``.
        allow_done_after_reset (bool, optional): if ``True``, an environment can
            be done after a call to :meth:`reset` is made. Defaults to ``False``.
        spec_locked (bool, optional): if ``True``, the specs are locked and can only be
            modified if :meth:`~torchrl.envs.EnvBase.set_spec_lock_` is called.

            .. note:: The locking is achieved by the `EnvBase` metaclass. It does not appear in the
                `__init__` method and is included in the keyword arguments strictly for type-hinting purpose.

            .. seealso:: :ref:`Locking environment specs <Environment-lock>`.

            Defaults to ``True``.
        auto_reset (bool, optional): if ``True``, the env is assumed to reset automatically
            when done. Defaults to ``False``.

            .. note:: The auto-resetting is achieved by the `EnvBase` metaclass. It does not appear in the
                `__init__` method and is included in the keyword arguments strictly for type-hinting purpose.

            .. seealso:: The :ref:`auto-resetting environments API <autoresetting_envs>` section in the API
                documentation.

    Attributes:
        done_spec (Composite): equivalent to ``full_done_spec`` as all
            ``done_specs`` contain at least a ``"done"`` and a ``"terminated"`` entry
        action_spec (TensorSpec): the spec of the action. Links to the spec of the leaf
            action if only one action tensor is to be expected. Otherwise links to
            ``full_action_spec``.
        observation_spec (Composite): equivalent to ``full_observation_spec``.
        reward_spec (TensorSpec): the spec of the reward. Links to the spec of the leaf
            reward if only one reward tensor is to be expected. Otherwise links to
            ``full_reward_spec``.
        state_spec (Composite): equivalent to ``full_state_spec``.
        full_done_spec (Composite): a composite spec such that ``full_done_spec.zero()``
            returns a tensordict containing only the leaves encoding the done status of the
            environment.
        full_action_spec (Composite): a composite spec such that ``full_action_spec.zero()``
            returns a tensordict containing only the leaves encoding the action of the
            environment.
        full_observation_spec (Composite): a composite spec such that ``full_observation_spec.zero()``
            returns a tensordict containing only the leaves encoding the observation of the
            environment.
        full_reward_spec (Composite): a composite spec such that ``full_reward_spec.zero()``
            returns a tensordict containing only the leaves encoding the reward of the
            environment.
        full_state_spec (Composite): a composite spec such that ``full_state_spec.zero()``
            returns a tensordict containing only the leaves encoding the inputs (actions
            excluded) of the environment.
        batch_size (torch.Size): The batch-size of the environment.
        device (torch.device): the device where the input/outputs of the environment
            are to be expected. Can be ``None``.
        is_spec_locked (bool): returns ``True`` if the specs are locked. See the :attr:`spec_locked`
            argument above.

    Methods:
        step (TensorDictBase -> TensorDictBase): step in the environment
        reset (TensorDictBase, optional -> TensorDictBase): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (TensorDictBase, optional -> TensorDictBase): random step given the action spec
        rollout (Callable, ... -> TensorDictBase): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)

    Examples:
        >>> from torchrl.envs import EnvBase
        >>> class CounterEnv(EnvBase):
        ...     def __init__(self, batch_size=(), device=None, **kwargs):
        ...         self.observation_spec = Composite(
        ...             count=Unbounded(batch_size, device=device, dtype=torch.int64))
        ...         self.action_spec = Unbounded(batch_size, device=device, dtype=torch.int8)
        ...         # done spec and reward spec are set automatically
        ...     def _step(self, tensordict):
        ...
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = GymEnv("Pendulum-v1")
        >>> env.batch_size  # how many envs are run at once
        torch.Size([])
        >>> env.input_spec
        Composite(
            full_state_spec: None,
            full_action_spec: Composite(
                action: BoundedContinuous(
                    shape=torch.Size([1]),
                    space=ContinuousBox(
                        low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                        high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
        >>> env.action_spec
        BoundedContinuous(
            shape=torch.Size([1]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> env.observation_spec
        Composite(
            observation: BoundedContinuous(
                shape=torch.Size([3]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous), device=cpu, shape=torch.Size([]))
        >>> env.reward_spec
        UnboundedContinuous(
            shape=torch.Size([1]),
            space=None,
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> env.done_spec
        Categorical(
            shape=torch.Size([1]),
            space=DiscreteBox(n=2),
            device=cpu,
            dtype=torch.bool,
            domain=discrete)
        >>> # the output_spec contains all the expected outputs
        >>> env.output_spec
        Composite(
            full_reward_spec: Composite(
                reward: UnboundedContinuous(
                    shape=torch.Size([1]),
                    space=None,
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([])),
            full_observation_spec: Composite(
                observation: BoundedContinuous(
                    shape=torch.Size([3]),
                    space=ContinuousBox(
                        low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                        high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([])),
            full_done_spec: Composite(
                done: Categorical(
                    shape=torch.Size([1]),
                    space=DiscreteBox(n=2),
                    device=cpu,
                    dtype=torch.bool,
                    domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

    .. note:: Learn more about dynamic specs and environments :ref:`here <dynamic_envs>`.
    """

    _batch_size: torch.Size | None
    _device: torch.device | None
    _is_spec_locked: bool = False

    def __init__(
        self,
        *,
        device: DEVICE_TYPING = None,
        batch_size: torch.Size | None = None,
        run_type_checks: bool = False,
        allow_done_after_reset: bool = False,
        spec_locked: bool = True,
        auto_reset: bool = False,
    ):
        if "_cache" not in self.__dict__:
            self._cache = {}
        super().__init__()

        self.__dict__.setdefault("_batch_size", None)
        self.__dict__.setdefault("_device", None)

        if batch_size is not None:
            # we want an error to be raised if we pass batch_size but
            # it's already been set
            batch_size = self.batch_size = torch.Size(batch_size)
        else:
            batch_size = torch.Size(())

        if device is not None:
            device = self.__dict__["_device"] = _make_ordinal_device(
                torch.device(device)
            )

        output_spec = self.__dict__.get("_output_spec")
        if output_spec is None:
            output_spec = self.__dict__["_output_spec"] = Composite(
                shape=batch_size, device=device
            )
        elif self._output_spec.device != device and device is not None:
            self.__dict__["_output_spec"] = self.__dict__["_output_spec"].to(
                self.device
            )
        input_spec = self.__dict__.get("_input_spec")
        if input_spec is None:
            input_spec = self.__dict__["_input_spec"] = Composite(
                shape=batch_size, device=device
            )
        elif self._input_spec.device != device and device is not None:
            self.__dict__["_input_spec"] = self.__dict__["_input_spec"].to(self.device)

        output_spec.unlock_(recurse=True)
        input_spec.unlock_(recurse=True)
        if "full_observation_spec" not in output_spec:
            output_spec["full_observation_spec"] = Composite(batch_size=batch_size)
        if "full_done_spec" not in output_spec:
            output_spec["full_done_spec"] = Composite(batch_size=batch_size)
        if "full_reward_spec" not in output_spec:
            output_spec["full_reward_spec"] = Composite(batch_size=batch_size)
        if "full_state_spec" not in input_spec:
            input_spec["full_state_spec"] = Composite(batch_size=batch_size)
        if "full_action_spec" not in input_spec:
            input_spec["full_action_spec"] = Composite(batch_size=batch_size)

        if "is_closed" not in self.__dir__():
            self.is_closed = True
        self._run_type_checks = run_type_checks
        self._allow_done_after_reset = allow_done_after_reset

    def set_spec_lock_(self, mode: bool = True) -> EnvBase:
        """Locks or unlocks the environment's specs.

        Args:
            mode (bool): Whether to lock (`True`) or unlock (`False`) the specs. Defaults to `True`.

        Returns:
            EnvBase: The environment instance itself.

        .. seealso:: :ref:`Locking environment specs <Environment-lock>`.

        """
        output_spec = self.__dict__.get("_output_spec")
        input_spec = self.__dict__.get("_input_spec")
        if mode:
            if output_spec is not None:
                output_spec.lock_(recurse=True)
            if input_spec is not None:
                input_spec.lock_(recurse=True)
        else:
            self._cache.clear()
            if output_spec is not None:
                output_spec.unlock_(recurse=True)
            if input_spec is not None:
                input_spec.unlock_(recurse=True)
        self.__dict__["_is_spec_locked"] = mode
        return self

    @property
    def is_spec_locked(self):
        """Gets whether the environment's specs are locked.

        This property can be modified directly.

        Returns:
            bool: True if the specs are locked, False otherwise.

        .. seealso:: :ref:`Locking environment specs <Environment-lock>`.

        """
        return self.__dict__.get("_is_spec_locked", False)

    @is_spec_locked.setter
    def is_spec_locked(self, value: bool):
        self.set_spec_lock_(value)

    def auto_specs_(
        self,
        policy: Callable[[TensorDictBase], TensorDictBase],
        *,
        tensordict: TensorDictBase | None = None,
        action_key: NestedKey | list[NestedKey] = "action",
        done_key: NestedKey | list[NestedKey] | None = None,
        observation_key: NestedKey | list[NestedKey] = "observation",
        reward_key: NestedKey | list[NestedKey] = "reward",
    ):
        """Automatically sets the specifications (specs) of the environment based on a random rollout using a given policy.

        This method performs a rollout using the provided policy to infer the input and output specifications of the environment.
        It updates the environment's specs for actions, observations, rewards, and done signals based on the data collected
        during the rollout.

        Args:
            policy (Callable[[TensorDictBase], TensorDictBase]):
                A callable policy that takes a `TensorDictBase` as input and returns a `TensorDictBase` as output.
                This policy is used to perform the rollout and determine the specs.

        Keyword Args:
            tensordict (TensorDictBase, optional):
                An optional `TensorDictBase` instance to be used as the initial state for the rollout.
                If not provided, the environment's `reset` method will be called to obtain the initial state.
            action_key (NestedKey or List[NestedKey], optional):
                The key(s) used to identify actions in the `TensorDictBase`. Defaults to "action".
            done_key (NestedKey or List[NestedKey], optional):
                The key(s) used to identify done signals in the `TensorDictBase`. Defaults to ``None``, which will
                attempt to use ["done", "terminated", "truncated"] as potential keys.
            observation_key (NestedKey or List[NestedKey], optional):
                The key(s) used to identify observations in the `TensorDictBase`. Defaults to "observation".
            reward_key (NestedKey or List[NestedKey], optional):
                The key(s) used to identify rewards in the `TensorDictBase`. Defaults to "reward".

        Returns:
            EnvBase: The environment instance with updated specs.

        Raises:
            RuntimeError: If there are keys in the output specs that are not accounted for in the provided keys.
        """
        if self.batch_locked or tensordict is None:
            batch_size = self.batch_size
        else:
            batch_size = tensordict.batch_size
        if tensordict is None:
            tensordict = self.reset()

        # Input specs
        tensordict.update(policy(tensordict))
        step_0 = self.step(tensordict.copy())
        tensordict2 = step_0.get("next").copy()
        step_1 = self.step(policy(tensordict2).copy())
        nexts_0: TensorDictBase = step_0.pop("next")
        nexts_1: TensorDictBase = step_1.pop("next")

        input_spec_stack = {}
        tensordict.apply(
            partial(_tensor_to_spec, stack=input_spec_stack),
            tensordict2,
            named=True,
            nested_keys=True,
            is_leaf=_is_leaf_nontensor,
        )
        input_spec = Composite(input_spec_stack, batch_size=batch_size)
        if not self.batch_locked and batch_size != self.batch_size:
            while input_spec.shape:
                input_spec = input_spec[0]
        if isinstance(action_key, NestedKey):
            action_key = [action_key]
        full_action_spec = input_spec.separates(*action_key, default=None)

        # Output specs

        output_spec_stack = {}
        nexts_0.apply(
            partial(_tensor_to_spec, stack=output_spec_stack),
            nexts_1,
            named=True,
            nested_keys=True,
            is_leaf=_is_leaf_nontensor,
        )

        output_spec = Composite(output_spec_stack, batch_size=batch_size)
        if not self.batch_locked and batch_size != self.batch_size:
            while output_spec.shape:
                output_spec = output_spec[0]

        if done_key is None:
            done_key = ["done", "terminated", "truncated"]
        full_done_spec = output_spec.separates(*done_key, default=None)
        if full_done_spec is not None:
            self.full_done_spec = full_done_spec

        if isinstance(reward_key, NestedKey):
            reward_key = [reward_key]
        full_reward_spec = output_spec.separates(*reward_key, default=None)

        if isinstance(observation_key, NestedKey):
            observation_key = [observation_key]
        full_observation_spec = output_spec.separates(*observation_key, default=None)
        if not output_spec.is_empty(recurse=True):
            raise RuntimeError(
                f"Keys {list(output_spec.keys(True, True))} are unaccounted for. "
                f"Make sure you have passed all the leaf names to the auto_specs_ method."
            )

        if full_action_spec is not None:
            self.full_action_spec = full_action_spec
        if full_done_spec is not None:
            self.full_done_spec = full_done_spec
        if full_observation_spec is not None:
            self.full_observation_spec = full_observation_spec
        if full_reward_spec is not None:
            self.full_reward_spec = full_reward_spec
        full_state_spec = input_spec
        self.full_state_spec = full_state_spec

        return self

    @wraps(check_env_specs_func)
    def check_env_specs(self, *args, **kwargs):
        kwargs.setdefault("return_contiguous", not self._has_dynamic_specs)
        return check_env_specs_func(self, *args, **kwargs)

    check_env_specs.__doc__ = check_env_specs_func.__doc__

    def cardinality(self, tensordict: TensorDictBase | None = None) -> int:
        """The cardinality of the action space.

        By default, this is just a wrapper around :meth:`env.action_space.cardinality <~torchrl.data.TensorSpec.cardinality>`.

        This class is useful when the action spec is variable:

        - The number of actions can be undefined, e.g., ``Categorical(n=-1)``;
        - The action cardinality may depend on the action mask;
        - The shape can be dynamic, as in ``Unbound(shape=(-1))``.

        In these cases, the :meth:`cardinality` should be overwritten,

        Args:
            tensordict (TensorDictBase, optional): a tensordict containing the data required to compute the cardinality.

        """
        return self.full_action_spec.cardinality()

    @classmethod
    def __new__(cls, *args, _inplace_update=False, _batch_locked=True, **kwargs):
        # inplace update will write tensors in-place on the provided tensordict.
        # This is risky, especially if gradients need to be passed (in-place copy
        # for tensors that are part of computational graphs will result in an error).
        # It can also lead to inconsistencies when calling rollout.
        cls._inplace_update = _inplace_update
        cls._batch_locked = _batch_locked
        cls._device = None
        # cached in_keys to be excluded from update when calling step
        cls._cache_in_keys = None

        # We may assign _input_spec to the cls, but it must be assigned to the instance
        # we pull it off, and place it back where it belongs
        _input_spec = None
        if hasattr(cls, "_input_spec"):
            _input_spec = cls._input_spec.clone()
            delattr(cls, "_input_spec")
        _output_spec = None
        if hasattr(cls, "_output_spec"):
            _output_spec = cls._output_spec.clone()
            delattr(cls, "_output_spec")
        env = super().__new__(cls)
        if _input_spec is not None:
            env.__dict__["_input_spec"] = _input_spec
        if _output_spec is not None:
            env.__dict__["_output_spec"] = _output_spec
        return env

        return super().__new__(cls)

    def __setattr__(self, key, value):
        if key in (
            "_input_spec",
            "_observation_spec",
            "_action_spec",
            "_reward_spec",
            "_output_spec",
            "_state_spec",
            "_done_spec",
        ):
            raise AttributeError(
                "To set an environment spec, please use `env.observation_spec = obs_spec` (without the leading"
                " underscore)."
            )
        super().__setattr__(key, value)

    @property
    def batch_locked(self) -> bool:
        """Whether the environment can be used with a batch size different from the one it was initialized with or not.

        If True, the env needs to be used with a tensordict having the same batch size as the env.
        batch_locked is an immutable property.
        """
        return self._batch_locked

    @batch_locked.setter
    def batch_locked(self, value: bool) -> None:
        raise RuntimeError("batch_locked is a read-only property")

    @property
    def run_type_checks(self) -> bool:
        return self._run_type_checks

    @run_type_checks.setter
    def run_type_checks(self, run_type_checks: bool) -> None:
        self._run_type_checks = run_type_checks

    @property
    def batch_size(self) -> torch.Size:
        """Number of envs batched in this environment instance organised in a `torch.Size()` object.

        Environment may be similar or different but it is assumed that they have little if
        not no interactions between them (e.g., multi-task or batched execution
        in parallel).

        """
        _batch_size = self.__dict__.get("_batch_size")
        if _batch_size is None:
            _batch_size = self._batch_size = torch.Size([])
        return _batch_size

    @batch_size.setter
    @_maybe_unlock
    def batch_size(self, value: torch.Size) -> None:
        self._batch_size = torch.Size(value)
        if (
            hasattr(self, "output_spec")
            and self.output_spec.shape[: len(value)] != value
        ):
            self.output_spec.shape = value
        if hasattr(self, "input_spec") and self.input_spec.shape[: len(value)] != value:
            self.input_spec.shape = value

    @property
    def shape(self):
        """Equivalent to :attr:`~.batch_size`."""
        return self.batch_size

    @property
    def device(self) -> torch.device:
        device = self.__dict__.get("_device")
        return device

    @device.setter
    def device(self, value: torch.device) -> None:
        device = self.__dict__.get("_device")
        if device is None:
            self.__dict__["_device"] = value
            return
        raise RuntimeError("device cannot be set. Call env.to(device) instead.")

    def ndimension(self):
        return len(self.batch_size)

    @property
    def ndim(self):
        return self.ndimension()

    def append_transform(
        self,
        transform: Transform | Callable[[TensorDictBase], TensorDictBase],  # noqa: F821
    ) -> EnvBase:
        """Returns a transformed environment where the callable/transform passed is applied.

        Args:
            transform (Transform or Callable[[TensorDictBase], TensorDictBase]): the transform to apply
                to the environment.

        Examples:
            >>> from torchrl.envs import GymEnv
            >>> import torch
            >>> env = GymEnv("CartPole-v1")
            >>> loc = 0.5
            >>> scale = 1.0
            >>> transform = lambda data: data.set("observation", (data.get("observation") - loc)/scale)
            >>> env = env.append_transform(transform=transform)
            >>> print(env)
            TransformedEnv(
                env=GymEnv(env=CartPole-v1, batch_size=torch.Size([]), device=cpu),
                transform=_CallableTransform(keys=[]))

        """
        from torchrl.envs.transforms.transforms import TransformedEnv

        return TransformedEnv(self, transform)

    # Parent specs: input and output spec.
    @property
    def input_spec(self) -> TensorSpec:
        """Input spec.

        The composite spec containing all specs for data input to the environments.

        It contains:

        - "full_action_spec": the spec of the input actions
        - "full_state_spec": the spec of all other environment inputs

        This attribute is locked and should be read-only.
        Instead, to set the specs contained in it, use the respective properties.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.input_spec
            Composite(
                full_state_spec: None,
                full_action_spec: Composite(
                    action: BoundedContinuous(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))


        """
        input_spec = self.__dict__.get("_input_spec")
        if input_spec is None:
            is_locked = self.is_spec_locked
            if is_locked:
                self.set_spec_lock_(False)
            input_spec = Composite(
                full_state_spec=None,
                shape=self.batch_size,
                device=self.device,
            )
            self.__dict__["_input_spec"] = input_spec
            if is_locked:
                self.set_spec_lock_(True)
        return input_spec

    @input_spec.setter
    def input_spec(self, value: TensorSpec) -> None:
        raise RuntimeError("input_spec is protected.")

    @property
    def output_spec(self) -> TensorSpec:
        """Output spec.

        The composite spec containing all specs for data output from the environments.

        It contains:

        - "full_reward_spec": the spec of reward
        - "full_done_spec": the spec of done
        - "full_observation_spec": the spec of all other environment outputs

        This attribute is locked and should be read-only.
        Instead, to set the specs contained in it, use the respective properties.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.output_spec
            Composite(
                full_reward_spec: Composite(
                    reward: UnboundedContinuous(
                        shape=torch.Size([1]),
                        space=None,
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous), device=cpu, shape=torch.Size([])),
                full_observation_spec: Composite(
                    observation: BoundedContinuous(
                        shape=torch.Size([3]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous), device=cpu, shape=torch.Size([])),
                full_done_spec: Composite(
                    done: Categorical(
                        shape=torch.Size([1]),
                        space=DiscreteBox(n=2),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))


        """
        output_spec = self.__dict__.get("_output_spec")
        if output_spec is None:
            is_locked = self.is_spec_locked
            if is_locked:
                self.set_spec_lock_(False)
            output_spec = Composite(
                shape=self.batch_size,
                device=self.device,
            )
            self.__dict__["_output_spec"] = output_spec
            if is_locked:
                self.set_spec_lock_(True)
        return output_spec

    @output_spec.setter
    def output_spec(self, value: TensorSpec) -> None:
        raise RuntimeError("output_spec is protected.")

    @property
    @_cache_value
    def action_keys(self) -> list[NestedKey]:
        """The action keys of an environment.

        By default, there will only be one key named "action".

        Keys are sorted by depth in the data tree.
        """
        keys = self.full_action_spec.keys(True, True)
        keys = sorted(keys, key=_repr_by_depth)
        return keys

    @property
    @_cache_value
    def state_keys(self) -> list[NestedKey]:
        """The state keys of an environment.

        By default, there will only be one key named "state".

        Keys are sorted by depth in the data tree.
        """
        state_keys = self.__dict__.get("_state_keys")
        if state_keys is not None:
            return state_keys
        keys = self.input_spec["full_state_spec"].keys(True, True)
        keys = sorted(keys, key=_repr_by_depth)
        self.__dict__["_state_keys"] = keys
        return keys

    @property
    def action_key(self) -> NestedKey:
        """The action key of an environment.

        By default, this will be "action".

        If there is more than one action key in the environment, this function will raise an exception.
        """
        if len(self.action_keys) > 1:
            raise KeyError(
                "action_key requested but more than one key present in the environment"
            )
        return self.action_keys[0]

    # Action spec: action specs belong to input_spec
    @property
    def action_spec(self) -> TensorSpec:
        """The ``action`` spec.

        The ``action_spec`` is always stored as a composite spec.

        If the action spec is provided as a simple spec, this will be returned.

            >>> env.action_spec = Unbounded(1)
            >>> env.action_spec
            UnboundedContinuous(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)

        If the action spec is provided as a composite spec and contains only one leaf,
        this function will return just the leaf.

            >>> env.action_spec = Composite({"nested": {"action": Unbounded(1)}})
            >>> env.action_spec
            UnboundedContinuous(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)

        If the action spec is provided as a composite spec and has more than one leaf,
        this function will return the whole spec.

            >>> env.action_spec = Composite({"nested": {"action": Unbounded(1), "another_action": Categorical(1)}})
            >>> env.action_spec
            Composite(
                nested: Composite(
                    action: UnboundedContinuous(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    another_action: Categorical(
                        shape=torch.Size([]),
                        space=DiscreteBox(n=1),
                        device=cpu,
                        dtype=torch.int64,
                        domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        To retrieve the full spec passed, use:

            >>> env.input_spec["full_action_spec"]

        This property is mutable.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.action_spec
            BoundedContinuous(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)
        """
        try:
            action_spec = self.input_spec["full_action_spec"]
        except (KeyError, AttributeError):
            raise KeyError("Failed to find the action_spec.")

        if len(self.action_keys) > 1:
            out = action_spec
        else:
            if len(self.action_keys) == 1 and self.action_keys[0] != "action":
                warnings.warn(
                    "You are querying a non-trivial, single action_spec, i.e., there is only "
                    "one action known by the environment but it is not named `'action'`. "
                    "Currently, env.action_spec returns the leaf but for consistency with the "
                    "setter, this will return the full spec instead (from v0.8 and on).",
                    category=DeprecationWarning,
                )
            try:
                out = action_spec[self.action_key]
            except KeyError:
                # the key may have changed
                raise KeyError(
                    "The action_key attribute seems to have changed. "
                    "This occurs when a action_spec is updated without "
                    "calling `env.action_spec = new_spec`. "
                    "Make sure you rely on this  type of command "
                    "to set the action and other specs."
                )

        return out

    @action_spec.setter
    @_maybe_unlock
    def action_spec(self, value: TensorSpec) -> None:
        device = self.input_spec._device
        if not hasattr(value, "shape"):
            raise TypeError(
                f"action_spec of type {type(value)} do not have a shape attribute."
            )
        if value.shape[: len(self.batch_size)] != self.batch_size:
            raise ValueError(
                f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size}). "
                "Please use `env.action_spec_unbatched = value` to set unbatched versions instead."
            )

        if not isinstance(value, Composite):
            value = Composite(
                action=value.to(device), shape=self.batch_size, device=device
            )

        self.input_spec["full_action_spec"] = value.to(device)

    @property
    def full_action_spec(self) -> Composite:
        """The full action spec.

        ``full_action_spec`` is a :class:`~torchrl.data.Composite`` instance
        that contains all the action entries.

        Examples:
            >>> from torchrl.envs import BraxEnv
            >>> for envname in BraxEnv.available_envs:
            ...     break
            >>> env = BraxEnv(envname)
            >>> env.full_action_spec
            Composite(
                action: BoundedContinuous(
                    shape=torch.Size([8]),
                    space=ContinuousBox(
                        low=Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, contiguous=True),
                        high=Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, contiguous=True)),
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([]))

        """
        full_action_spec = self.input_spec.get("full_action_spec", None)
        if full_action_spec is None:
            is_locked = self.is_spec_locked
            if is_locked:
                self.set_spec_lock_(False)
            full_action_spec = Composite(shape=self.batch_size, device=self.device)
            self.input_spec["full_action_spec"] = full_action_spec
            if is_locked:
                self.set_spec_lock_(True)
        return full_action_spec

    @full_action_spec.setter
    def full_action_spec(self, spec: Composite) -> None:
        self.action_spec = spec

    # Reward spec
    @property
    @_cache_value
    def reward_keys(self) -> list[NestedKey]:
        """The reward keys of an environment.

        By default, there will only be one key named "reward".

        Keys are sorted by depth in the data tree.
        """
        reward_keys = sorted(self.full_reward_spec.keys(True, True), key=_repr_by_depth)
        return reward_keys

    @property
    @_cache_value
    def observation_keys(self) -> list[NestedKey]:
        """The observation keys of an environment.

        By default, there will only be one key named "observation".

        Keys are sorted by depth in the data tree.
        """
        observation_keys = sorted(
            self.full_observation_spec.keys(True, True), key=_repr_by_depth
        )
        return observation_keys

    @property
    def reward_key(self):
        """The reward key of an environment.

        By default, this will be "reward".

        If there is more than one reward key in the environment, this function will raise an exception.
        """
        if len(self.reward_keys) > 1:
            raise KeyError(
                "reward_key requested but more than one key present in the environment"
            )
        return self.reward_keys[0]

    # Reward spec: reward specs belong to output_spec
    @property
    def reward_spec(self) -> TensorSpec:
        """The ``reward`` spec.

        The ``reward_spec`` is always stored as a composite spec.

        If the reward spec is provided as a simple spec, this will be returned.

            >>> env.reward_spec = Unbounded(1)
            >>> env.reward_spec
            UnboundedContinuous(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)

        If the reward spec is provided as a composite spec and contains only one leaf,
        this function will return just the leaf.

            >>> env.reward_spec = Composite({"nested": {"reward": Unbounded(1)}})
            >>> env.reward_spec
            UnboundedContinuous(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous)

        If the reward spec is provided as a composite spec and has more than one leaf,
        this function will return the whole spec.

            >>> env.reward_spec = Composite({"nested": {"reward": Unbounded(1), "another_reward": Categorical(1)}})
            >>> env.reward_spec
            Composite(
                nested: Composite(
                    reward: UnboundedContinuous(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    another_reward: Categorical(
                        shape=torch.Size([]),
                        space=DiscreteBox(n=1),
                        device=cpu,
                        dtype=torch.int64,
                        domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        To retrieve the full spec passed, use:

            >>> env.output_spec["full_reward_spec"]

        This property is mutable.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.reward_spec
            UnboundedContinuous(
                shape=torch.Size([1]),
                space=None,
                device=cpu,
                dtype=torch.float32,
                domain=continuous)
        """
        try:
            reward_spec = self.output_spec["full_reward_spec"]
        except (KeyError, AttributeError):
            # populate the "reward" entry
            # this will be raised if there is not full_reward_spec (unlikely) or no reward_key
            # Since output_spec is lazily populated with an empty composite spec for
            # reward_spec, the second case is much more likely to occur.
            self.reward_spec = Unbounded(
                shape=(*self.batch_size, 1),
                device=self.device,
            )
            reward_spec = self.output_spec["full_reward_spec"]

        reward_keys = self.reward_keys
        if len(reward_keys) > 1 or not len(reward_keys):
            return reward_spec
        else:
            if len(self.reward_keys) == 1 and self.reward_keys[0] != "reward":
                warnings.warn(
                    "You are querying a non-trivial, single reward_spec, i.e., there is only "
                    "one reward known by the environment but it is not named `'reward'`. "
                    "Currently, env.reward_spec returns the leaf but for consistency with the "
                    "setter, this will return the full spec instead (from v0.8 and on).",
                    category=DeprecationWarning,
                )
            return reward_spec[self.reward_keys[0]]

    @reward_spec.setter
    @_maybe_unlock
    def reward_spec(self, value: TensorSpec) -> None:
        device = self.output_spec._device
        if not hasattr(value, "shape"):
            raise TypeError(
                f"reward_spec of type {type(value)} do not have a shape " f"attribute."
            )
        if value.shape[: len(self.batch_size)] != self.batch_size:
            raise ValueError(
                f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size}). "
                "Please use `env.reward_spec_unbatched = value` to set unbatched versions instead."
            )
        if not isinstance(value, Composite):
            value = Composite(
                reward=value.to(device), shape=self.batch_size, device=device
            )
        for leaf in value.values(True, True):
            if len(leaf.shape) == 0:
                raise RuntimeError(
                    "the reward_spec's leaves shape cannot be empty (this error"
                    " usually comes from trying to set a reward_spec"
                    " with a null number of dimensions. Try using a multidimensional"
                    " spec instead, for instance with a singleton dimension at the tail)."
                )
        self.output_spec["full_reward_spec"] = value.to(device)

    @property
    def full_reward_spec(self) -> Composite:
        """The full reward spec.

        ``full_reward_spec`` is a :class:`~torchrl.data.Composite`` instance
        that contains all the reward entries.

        Examples:
            >>> import gymnasium
            >>> from torchrl.envs import GymWrapper, TransformedEnv, RenameTransform
            >>> base_env = GymWrapper(gymnasium.make("Pendulum-v1"))
            >>> env = TransformedEnv(base_env, RenameTransform("reward", ("nested", "reward")))
            >>> env.full_reward_spec
            Composite(
                nested: Composite(
                    reward: UnboundedContinuous(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous), device=None, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        """
        try:
            return self.output_spec["full_reward_spec"]
        except KeyError:
            # populate the "reward" entry
            # this will be raised if there is not full_reward_spec (unlikely) or no reward_key
            # Since output_spec is lazily populated with an empty composite spec for
            # reward_spec, the second case is much more likely to occur.
            self.reward_spec = Unbounded(
                shape=(*self.batch_size, 1),
                device=self.device,
            )
            return self.output_spec["full_reward_spec"]

    @full_reward_spec.setter
    @_maybe_unlock
    def full_reward_spec(self, spec: Composite) -> None:
        self.reward_spec = spec.to(self.device) if self.device is not None else spec

    # done spec
    @property
    @_cache_value
    def done_keys(self) -> list[NestedKey]:
        """The done keys of an environment.

        By default, there will only be one key named "done".

        Keys are sorted by depth in the data tree.
        """
        done_keys = sorted(self.full_done_spec.keys(True, True), key=_repr_by_depth)
        return done_keys

    @property
    def done_key(self):
        """The done key of an environment.

        By default, this will be "done".

        If there is more than one done key in the environment, this function will raise an exception.
        """
        done_keys = self.done_keys
        if len(done_keys) > 1:
            raise KeyError(
                "done_key requested but more than one key present in the environment"
            )
        return done_keys[0]

    @property
    def full_done_spec(self) -> Composite:
        """The full done spec.

        ``full_done_spec`` is a :class:`~torchrl.data.Composite`` instance
        that contains all the done entries.
        It can be used to generate fake data with a structure that mimics the
        one obtained at runtime.

        Examples:
            >>> import gymnasium
            >>> from torchrl.envs import GymWrapper
            >>> env = GymWrapper(gymnasium.make("Pendulum-v1"))
            >>> env.full_done_spec
            Composite(
                done: Categorical(
                    shape=torch.Size([1]),
                    space=DiscreteBox(n=2),
                    device=cpu,
                    dtype=torch.bool,
                    domain=discrete),
                truncated: Categorical(
                    shape=torch.Size([1]),
                    space=DiscreteBox(n=2),
                    device=cpu,
                    dtype=torch.bool,
                    domain=discrete), device=cpu, shape=torch.Size([]))

        """
        return self.output_spec["full_done_spec"]

    @full_done_spec.setter
    @_maybe_unlock
    def full_done_spec(self, spec: Composite) -> None:
        self.done_spec = spec.to(self.device) if self.device is not None else spec

    # Done spec: done specs belong to output_spec
    @property
    def done_spec(self) -> TensorSpec:
        """The ``done`` spec.

        The ``done_spec`` is always stored as a composite spec.

        If the done spec is provided as a simple spec, this will be returned.

            >>> env.done_spec = Categorical(2, dtype=torch.bool)
            >>> env.done_spec
            Categorical(
                shape=torch.Size([]),
                space=DiscreteBox(n=2),
                device=cpu,
                dtype=torch.bool,
                domain=discrete)

        If the done spec is provided as a composite spec and contains only one leaf,
        this function will return just the leaf.

            >>> env.done_spec = Composite({"nested": {"done": Categorical(2, dtype=torch.bool)}})
            >>> env.done_spec
            Categorical(
                shape=torch.Size([]),
                space=DiscreteBox(n=2),
                device=cpu,
                dtype=torch.bool,
                domain=discrete)

        If the done spec is provided as a composite spec and has more than one leaf,
        this function will return the whole spec.

            >>> env.done_spec = Composite({"nested": {"done": Categorical(2, dtype=torch.bool), "another_done": Categorical(2, dtype=torch.bool)}})
            >>> env.done_spec
            Composite(
                nested: Composite(
                    done: Categorical(
                        shape=torch.Size([]),
                        space=DiscreteBox(n=2),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete),
                    another_done: Categorical(
                        shape=torch.Size([]),
                        space=DiscreteBox(n=2),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        To always retrieve the full spec passed, use:

            >>> env.output_spec["full_done_spec"]

        This property is mutable.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.done_spec
            Categorical(
                shape=torch.Size([1]),
                space=DiscreteBox(n=2),
                device=cpu,
                dtype=torch.bool,
                domain=discrete)
        """
        done_spec = self.output_spec["full_done_spec"]
        return done_spec

    @_maybe_unlock
    def _create_done_specs(self):
        """Reads through the done specs and makes it so that it's complete.

        If the done_specs contain only a ``"done"`` entry, a similar ``"terminated"`` entry is created.
        Same goes if only ``"terminated"`` key is present.

        If none of ``"done"`` and ``"terminated"`` can be found and the spec is not
        empty, nothing is changed.

        """
        try:
            full_done_spec = self.output_spec["full_done_spec"]
        except KeyError:
            full_done_spec = Composite(
                shape=self.output_spec.shape, device=self.output_spec.device
            )
            full_done_spec["done"] = Categorical(
                n=2,
                shape=(*full_done_spec.shape, 1),
                dtype=torch.bool,
                device=self.device,
            )
            full_done_spec["terminated"] = Categorical(
                n=2,
                shape=(*full_done_spec.shape, 1),
                dtype=torch.bool,
                device=self.device,
            )
            self.output_spec["full_done_spec"] = full_done_spec
            return

        def check_local_done(spec):
            shape = None
            for key, item in list(
                spec.items()
            ):  # list to avoid error due to in-loop changes
                # in the case where the spec is non-empty and there is no done and no terminated, we do nothing
                if key == "done" and "terminated" not in spec.keys():
                    spec["terminated"] = item.clone()
                elif key == "terminated" and "done" not in spec.keys():
                    spec["done"] = item.clone()
                elif isinstance(item, Composite):
                    check_local_done(item)
                else:
                    if shape is None:
                        shape = item.shape
                        continue
                    # checks that all shape match
                    if shape != item.shape:
                        raise ValueError(
                            f"All shapes should match in done_spec {spec} (shape={shape}, key={key})."
                        )

            # if the spec is empty, we need to add a done and terminated manually
            if spec.is_empty():
                spec["done"] = Categorical(
                    n=2, shape=(*spec.shape, 1), dtype=torch.bool, device=self.device
                )
                spec["terminated"] = Categorical(
                    n=2, shape=(*spec.shape, 1), dtype=torch.bool, device=self.device
                )

        if_locked = self.is_spec_locked
        if if_locked:
            self.is_spec_locked = False
        check_local_done(full_done_spec)
        self.output_spec["full_done_spec"] = full_done_spec
        if if_locked:
            self.is_spec_locked = True
        return

    @done_spec.setter
    @_maybe_unlock
    def done_spec(self, value: TensorSpec) -> None:
        device = self.output_spec.device
        if not hasattr(value, "shape"):
            raise TypeError(
                f"done_spec of type {type(value)} do not have a shape " f"attribute."
            )
        if value.shape[: len(self.batch_size)] != self.batch_size:
            raise ValueError(
                f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
            )
        if not isinstance(value, Composite):
            value = Composite(
                done=value.to(device),
                terminated=value.to(device),
                shape=self.batch_size,
                device=device,
            )
        for leaf in value.values(True, True):
            if len(leaf.shape) == 0:
                raise RuntimeError(
                    "the done_spec's leaves shape cannot be empty (this error"
                    " usually comes from trying to set a reward_spec"
                    " with a null number of dimensions. Try using a multidimensional"
                    " spec instead, for instance with a singleton dimension at the tail)."
                )
        self.output_spec["full_done_spec"] = value.to(device)
        self._create_done_specs()

    # observation spec: observation specs belong to output_spec
    @property
    def observation_spec(self) -> Composite:
        """Observation spec.

        Must be a :class:`torchrl.data.Composite` instance.
        The keys listed in the spec are directly accessible after reset and step.

        In TorchRL, even though they are not properly speaking "observations"
        all info, states, results of transforms etc. outputs from the environment are stored in the
        ``observation_spec``.

        Therefore, ``"observation_spec"`` should be thought as
        a generic data container for environment outputs that are not done or reward data.

        Examples:
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> env.observation_spec
            Composite(
                observation: BoundedContinuous(
                    shape=torch.Size([3]),
                    space=ContinuousBox(
                        low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                        high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([]))

        """
        observation_spec = self.output_spec.get("full_observation_spec", default=None)
        if observation_spec is None:
            is_locked = self.is_spec_locked
            if is_locked:
                self.set_spec_lock_(False)
            observation_spec = Composite(shape=self.batch_size, device=self.device)
            self.output_spec["full_observation_spec"] = observation_spec
            if is_locked:
                self.set_spec_lock_(True)

        return observation_spec

    @observation_spec.setter
    @_maybe_unlock
    def observation_spec(self, value: TensorSpec) -> None:
        if not isinstance(value, Composite):
            value = Composite(
                observation=value,
                device=self.device,
                batch_size=self.output_spec.batch_size,
            )
        elif value.shape[: len(self.batch_size)] != self.batch_size:
            raise ValueError(
                f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
            )
        if value.shape[: len(self.batch_size)] != self.batch_size:
            raise ValueError(
                f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
            )
        device = self.output_spec._device
        self.output_spec["full_observation_spec"] = (
            value.to(device) if device is not None else value
        )

    @property
    def full_observation_spec(self) -> Composite:
        return self.observation_spec

    @full_observation_spec.setter
    @_maybe_unlock
    def full_observation_spec(self, spec: Composite):
        self.observation_spec = spec

    # state spec: state specs belong to input_spec
    @property
    def state_spec(self) -> Composite:
        """State spec.

        Must be a :class:`torchrl.data.Composite` instance.
        The keys listed here should be provided as input alongside actions to the environment.

        In TorchRL, even though they are not properly speaking "state"
        all inputs to the environment that are not actions are stored in the
        ``state_spec``.

        Therefore, ``"state_spec"`` should be thought as
        a generic data container for environment inputs that are not action data.

        Examples:
            >>> from torchrl.envs import BraxEnv
            >>> for envname in BraxEnv.available_envs:
            ...     break
            >>> env = BraxEnv(envname)
            >>> env.state_spec
            Composite(
                state: Composite(
                    pipeline_state: Composite(
                        q: UnboundedContinuous(
                            shape=torch.Size([15]),
                            space=None,
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                [...], device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))


        """
        state_spec = self.input_spec["full_state_spec"]
        if state_spec is None:
            is_locked = self.is_spec_locked
            if is_locked:
                self.set_spec_lock_(False)
            state_spec = Composite(shape=self.batch_size, device=self.device)
            self.input_spec["full_state_spec"] = state_spec
            if is_locked:
                self.set_spec_lock_(True)
        return state_spec

    @state_spec.setter
    @_maybe_unlock
    def state_spec(self, value: Composite) -> None:
        if value is None:
            self.input_spec["full_state_spec"] = Composite(
                device=self.device, shape=self.batch_size
            )
        else:
            device = self.input_spec.device
            if not isinstance(value, Composite):
                raise TypeError("The type of an state_spec must be Composite.")
            elif value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                )
            if value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                )
            self.input_spec["full_state_spec"] = (
                value.to(device) if device is not None else value
            )

    @property
    def full_state_spec(self) -> Composite:
        """The full state spec.

        ``full_state_spec`` is a :class:`~torchrl.data.Composite`` instance
        that contains all the state entries (ie, the input data that is not action).

        Examples:
            >>> from torchrl.envs import BraxEnv
            >>> for envname in BraxEnv.available_envs:
            ...     break
            >>> env = BraxEnv(envname)
            >>> env.full_state_spec
            Composite(
                state: Composite(
                    pipeline_state: Composite(
                        q: UnboundedContinuous(
                            shape=torch.Size([15]),
                            space=None,
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                [...], device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

        """
        return self.state_spec

    @full_state_spec.setter
    @_maybe_unlock
    def full_state_spec(self, spec: Composite) -> None:
        self.state_spec = spec

    # Single-env specs can be used to remove the batch size from the spec
    @property
    def batch_dims(self) -> int:
        """Number of batch dimensions of the env."""
        return len(self.batch_size)

    def _make_single_env_spec(self, spec: TensorSpec) -> TensorSpec:
        if not self.batch_dims:
            return spec
        idx = tuple(0 for _ in range(self.batch_dims))
        return spec[idx]

    @property
    def full_action_spec_unbatched(self) -> Composite:
        """Returns the action spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.full_action_spec)

    @full_action_spec_unbatched.setter
    @_maybe_unlock
    def full_action_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.full_action_spec = spec

    @property
    def action_spec_unbatched(self) -> TensorSpec:
        """Returns the action spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.action_spec)

    @action_spec_unbatched.setter
    @_maybe_unlock
    def action_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.action_spec = spec

    @property
    def full_observation_spec_unbatched(self) -> Composite:
        """Returns the observation spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.full_observation_spec)

    @full_observation_spec_unbatched.setter
    @_maybe_unlock
    def full_observation_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.full_observation_spec = spec

    @property
    def observation_spec_unbatched(self) -> Composite:
        """Returns the observation spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.observation_spec)

    @observation_spec_unbatched.setter
    @_maybe_unlock
    def observation_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.observation_spec = spec

    @property
    def full_reward_spec_unbatched(self) -> Composite:
        """Returns the reward spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.full_reward_spec)

    @full_reward_spec_unbatched.setter
    @_maybe_unlock
    def full_reward_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.full_reward_spec = spec

    @property
    def reward_spec_unbatched(self) -> TensorSpec:
        """Returns the reward spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.reward_spec)

    @reward_spec_unbatched.setter
    @_maybe_unlock
    def reward_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.reward_spec = spec

    @property
    def full_done_spec_unbatched(self) -> Composite:
        """Returns the done spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.full_done_spec)

    @full_done_spec_unbatched.setter
    @_maybe_unlock
    def full_done_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.full_done_spec = spec

    @property
    def done_spec_unbatched(self) -> TensorSpec:
        """Returns the done spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.done_spec)

    @done_spec_unbatched.setter
    @_maybe_unlock
    def done_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.done_spec = spec

    @property
    def output_spec_unbatched(self) -> Composite:
        """Returns the output spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.output_spec)

    @output_spec_unbatched.setter
    @_maybe_unlock
    def output_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.output_spec = spec

    @property
    def input_spec_unbatched(self) -> Composite:
        """Returns the input spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.input_spec)

    @input_spec_unbatched.setter
    @_maybe_unlock
    def input_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.input_spec = spec

    @property
    def full_state_spec_unbatched(self) -> Composite:
        """Returns the state spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.full_state_spec)

    @full_state_spec_unbatched.setter
    @_maybe_unlock
    def full_state_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.full_state_spec = spec

    @property
    def state_spec_unbatched(self) -> TensorSpec:
        """Returns the state spec of the env as if it had no batch dimensions."""
        return self._make_single_env_spec(self.state_spec)

    @state_spec_unbatched.setter
    @_maybe_unlock
    def state_spec_unbatched(self, spec: Composite):
        spec = spec.expand(self.batch_size + spec.shape)
        self.state_spec = spec

    def _skip_tensordict(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Creates a "skip" tensordict, ie a placeholder for when a step is skipped
        next_tensordict = self.full_done_spec.zero()
        next_tensordict.update(self.full_observation_spec.zero())
        next_tensordict.update(self.full_reward_spec.zero())

        # Copy the data from tensordict in `next`
        keys = set()

        def select_and_clone(name, x, y):
            keys.add(name)
            if y is not None:
                if y.device == x.device:
                    return x.clone()
                return x.to(y.device)

        result = tensordict._fast_apply(
            select_and_clone,
            next_tensordict,
            device=self.device,
            default=None,
            filter_empty=True,
            is_leaf=_is_leaf_nontensor,
            named=True,
            nested_keys=True,
        )
        result.update(next_tensordict.exclude(*keys).filter_empty_())
        return result

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Makes a step in the environment.

        Step accepts a single argument, tensordict, which usually carries an 'action' key which indicates the action
        to be taken.
        Step will call an out-place private method, _step, which is the method to be re-written by EnvBase subclasses.

        Args:
            tensordict (TensorDictBase): Tensordict containing the action to be taken.
                If the input tensordict contains a ``"next"`` entry, the values contained in it
                will prevail over the newly computed values. This gives a mechanism
                to override the underlying computations.

        Returns:
            the input tensordict, modified in place with the resulting observations, done state and reward
            (+ others if needed).

        """
        # sanity check
        self._assert_tensordict_shape(tensordict)
        partial_steps = tensordict.pop("_step", None)

        next_tensordict = None

        if partial_steps is not None:
            tensordict_batch_size = None
            if not self.batch_locked:
                # Batched envs have their own way of dealing with this - batched envs that are not batched-locked may fail here
                if partial_steps.all():
                    partial_steps = None
                else:
                    tensordict_batch_size = tensordict.batch_size
                    partial_steps = partial_steps.view(tensordict_batch_size)
                    tensordict = tensordict[partial_steps]
            else:
                if not partial_steps.any():
                    next_tensordict = self._skip_tensordic(tensordict)
                else:
                    # trust that the _step can handle this!
                    tensordict.set("_step", partial_steps)
            if tensordict_batch_size is None:
                tensordict_batch_size = self.batch_size

        next_preset = tensordict.get("next", None)

        if next_tensordict is None:
            next_tensordict = self._step(tensordict)
            next_tensordict = self._step_proc_data(next_tensordict)
        if next_preset is not None:
            # tensordict could already have a "next" key
            # this could be done more efficiently by not excluding but just passing
            # the necessary keys
            next_tensordict.update(
                next_preset.exclude(*next_tensordict.keys(True, True))
            )
        tensordict.set("next", next_tensordict)
        if partial_steps is not None and tensordict_batch_size != self.batch_size:
            result = tensordict.new_zeros(tensordict_batch_size)

            if tensordict_batch_size == tensordict.batch_size:

                def select_and_clone(x, y):
                    if y is not None:
                        if x.device == y.device:
                            return x.clone()
                        return x.to(y.device)

                result.update(
                    tensordict._fast_apply(
                        select_and_clone,
                        result,
                        device=result.device,
                        filter_empty=True,
                        default=None,
                        batch_size=result.batch_size,
                        is_leaf=_is_leaf_nontensor,
                    )
                )
            if partial_steps.any():
                result[partial_steps] = tensordict
            return result
        return tensordict

    @classmethod
    def _complete_done(
        cls, done_spec: Composite, data: TensorDictBase
    ) -> TensorDictBase:
        """Completes the data structure at step time to put missing done keys."""
        # by default, if a done key is missing, it is assumed that it is False
        # except in 2 cases: (1) there is a "done" but no "terminated" or (2)
        # there is a "terminated" but no "done".
        if done_spec.ndim:
            leading_dim = data.shape[: -done_spec.ndim]
        else:
            leading_dim = data.shape
        vals = {}
        i = -1
        for i, (key, item) in enumerate(done_spec.items()):  # noqa: B007
            val = data.get(key, None)
            if isinstance(item, Composite):
                if val is not None:
                    cls._complete_done(item, val)
                continue
            shape = (*leading_dim, *item.shape)
            if val is not None:
                if val.shape != shape:
                    val = val.reshape(shape)
                    data.set(key, val)
                vals[key] = val

        if len(vals) < i + 1:
            # complete missing dones: we only want to do that if we don't have enough done values
            data_keys = set(data.keys())
            done_spec_keys = set(done_spec.keys())
            for key, item in done_spec.items(False, True):
                val = vals.get(key, None)
                if (
                    key == "done"
                    and val is not None
                    and "terminated" in done_spec_keys
                    and "terminated" not in data_keys
                ):
                    if "truncated" in data_keys:
                        raise RuntimeError(
                            "Cannot infer the value of terminated when only done and truncated are present."
                        )
                    data.set("terminated", val)
                    data_keys.add("terminated")
                elif (
                    key == "terminated"
                    and val is not None
                    and "done" in done_spec_keys
                    and "done" not in data_keys
                ):
                    if "truncated" in data_keys:
                        val = val | data.get("truncated")
                    data.set("done", val)
                    data_keys.add("done")
                elif val is None and key not in data_keys:
                    # we must keep this here: we only want to fill with 0s if we're sure
                    # done should not be copied to terminated or terminated to done
                    # in this case, just fill with 0s
                    data.set(key, item.zero(leading_dim))
        return data

    def _step_proc_data(self, next_tensordict_out):
        batch_size = self.batch_size
        dims = len(batch_size)
        leading_batch_size = (
            next_tensordict_out.batch_size[:-dims]
            if dims
            else next_tensordict_out.shape
        )
        for reward_key in self.reward_keys:
            expected_reward_shape = torch.Size(
                [
                    *leading_batch_size,
                    *self.output_spec["full_reward_spec"][reward_key].shape,
                ]
            )
            # If the reward has a variable shape, we don't want to perform this check
            if all(s > 0 for s in expected_reward_shape):
                reward = next_tensordict_out.get(reward_key)
                actual_reward_shape = reward.shape
                if actual_reward_shape != expected_reward_shape:
                    reward = reward.view(expected_reward_shape)
                    next_tensordict_out.set(reward_key, reward)

        self._complete_done(self.full_done_spec, next_tensordict_out)

        if self.run_type_checks:
            for key, spec in self.observation_spec.items():
                obs = next_tensordict_out.get(key)
                spec.type_check(obs)

            for reward_key in self.reward_keys:
                if (
                    next_tensordict_out.get(reward_key).dtype
                    is not self.output_spec[
                        unravel_key(("full_reward_spec", reward_key))
                    ].dtype
                ):
                    raise TypeError(
                        f"expected reward.dtype to be {self.output_spec[unravel_key(('full_reward_spec',reward_key))]} "
                        f"but got {next_tensordict_out.get(reward_key).dtype}"
                    )

            for done_key in self.done_keys:
                if (
                    next_tensordict_out.get(done_key).dtype
                    is not self.output_spec["full_done_spec", done_key].dtype
                ):
                    raise TypeError(
                        f"expected done.dtype to be {self.output_spec['full_done_spec', done_key].dtype} but got {next_tensordict_out.get(done_key).dtype}"
                    )
        return next_tensordict_out

    def _get_in_keys_to_exclude(self, tensordict):
        if self._cache_in_keys is None:
            self._cache_in_keys = list(
                set(self.input_spec.keys(True)).intersection(
                    tensordict.keys(True, True)
                )
            )
        return self._cache_in_keys

    @classmethod
    def register_gym(
        cls,
        id: str,
        *,
        entry_point: Callable | None = None,
        transform: Transform | None = None,  # noqa: F821
        info_keys: list[NestedKey] | None = None,
        backend: str = None,
        to_numpy: bool = False,
        reward_threshold: float | None = None,
        nondeterministic: bool = False,
        max_episode_steps: int | None = None,
        order_enforce: bool = True,
        autoreset: bool | None = None,
        disable_env_checker: bool = False,
        apply_api_compatibility: bool = False,
        **kwargs,
    ):
        """Registers an environment in gym(nasium).

        This method is designed with the following scopes in mind:

        - Incorporate a TorchRL-first environment in a framework that uses Gym;
        - Incorporate another environment (eg, DeepMind Control, Brax, Jumanji, ...)
          in a framework that uses Gym.

        Args:
            id (str): the name of the environment. Should follow the
                `gym naming convention <https://www.gymlibrary.dev/content/environment_creation/#registering-envs>`_.

        Keyword Args:
            entry_point (callable, optional): the entry point to build the environment.
                If none is passed, the parent class will be used as entry point.
                Typically, this is used to register an environment that does not
                necessarily inherit from the base being used:

                    >>> from torchrl.envs import DMControlEnv
                    >>> DMControlEnv.register_gym("DMC-cheetah-v0", env_name="cheetah", task="run")
                    >>> # equivalently
                    >>> EnvBase.register_gym("DMC-cheetah-v0", entry_point=DMControlEnv, env_name="cheetah", task="run")

            transform (torchrl.envs.Transform): a transform (or list of transforms
                within a :class:`torchrl.envs.Compose` instance) to be used with the env.
                This arg can be passed during a call to :func:`~gym.make` (see
                example below).
            info_keys (List[NestedKey], optional): if provided, these keys will
                be used to build the info dictionary and will be excluded from
                the observation keys.
                This arg can be passed during a call to :func:`~gym.make` (see
                example below).

                .. warning::
                  It may be the case that using ``info_keys`` makes a spec empty
                  because the content has been moved to the info dictionary.
                  Gym does not like empty ``Dict`` in the specs, so this empty
                  content should be removed with :class:`~torchrl.envs.transforms.RemoveEmptySpecs`.

            backend (str, optional): the backend. Can be either `"gym"` or `"gymnasium"`
                or any other backend compatible with :class:`~torchrl.envs.libs.gym.set_gym_backend`.
            to_numpy (bool, optional): if ``True``, the result of calls to `step` and
                `reset` will be mapped to numpy arrays. Defaults to ``False``
                (results are tensors).
                This arg can be passed during a call to :func:`~gym.make` (see
                example below).
            reward_threshold (:obj:`float`, optional): [Gym kwarg] The reward threshold
                considered to have learnt an environment.
            nondeterministic (bool, optional): [Gym kwarg If the environment is nondeterministic
                (even with knowledge of the initial seed and all actions). Defaults to
                ``False``.
            max_episode_steps (int, optional): [Gym kwarg] The maximum number
                of episodes steps before truncation. Used by the Time Limit wrapper.
            order_enforce (bool, optional): [Gym >= 0.14] Whether the order
                enforcer wrapper should be applied to ensure users run functions
                in the correct order.
                Defaults to ``True``.
            autoreset (bool, optional): [Gym >= 0.14 and <1.0.0] Whether the autoreset wrapper
                should be added such that reset does not need to be called.
                Defaults to ``False``.
            disable_env_checker: [Gym >= 0.14] Whether the environment
                checker should be disabled for the environment. Defaults to ``False``.
            apply_api_compatibility: [Gym >= 0.26 and <1.0.0] If to apply the `StepAPICompatibility` wrapper.
                Defaults to ``False``.
            **kwargs: arbitrary keyword arguments which are passed to the environment constructor.

        .. note::
            TorchRL's environment do not have the concept of an ``"info"`` dictionary,
            as ``TensorDict`` offers all the storage requirements deemed necessary
            in most training settings. Still, you can use the ``info_keys`` argument to
            have a fine grained control over what is deemed to be considered
            as an observation and what should be seen as info.

        Examples:
            >>> # Register the "cheetah" env from DMControl with the "run" task
            >>> from torchrl.envs import DMControlEnv
            >>> import torch
            >>> DMControlEnv.register_gym("DMC-cheetah-v0", to_numpy=False, backend="gym", env_name="cheetah", task_name="run")
            >>> import gym
            >>> envgym = gym.make("DMC-cheetah-v0")
            >>> envgym.seed(0)
            >>> torch.manual_seed(0)
            >>> envgym.reset()
            ({'position': tensor([-0.0855,  0.0215, -0.0881, -0.0412, -0.1101,  0.0080,  0.0254,  0.0424],
                   dtype=torch.float64), 'velocity': tensor([ 1.9609e-02, -1.9776e-04, -1.6347e-03,  3.3842e-02,  2.5338e-02,
                     3.3064e-02,  1.0381e-04,  7.6656e-05,  1.0204e-02],
                   dtype=torch.float64)}, {})
            >>> envgym.step(envgym.action_space.sample())
            ({'position': tensor([-0.0833,  0.0275, -0.0612, -0.0770, -0.1256,  0.0082,  0.0186,  0.0476],
                   dtype=torch.float64), 'velocity': tensor([ 0.2221,  0.2256,  0.5930,  2.6937, -3.5865, -1.5479,  0.0187, -0.6825,
                     0.5224], dtype=torch.float64)}, tensor([0.0018], dtype=torch.float64), tensor([False]), tensor([False]), {})
            >>> # same environment with observation stacked
            >>> from torchrl.envs import CatTensors
            >>> envgym = gym.make("DMC-cheetah-v0", transform=CatTensors(in_keys=["position", "velocity"], out_key="observation"))
            >>> envgym.reset()
            ({'observation': tensor([-0.1005,  0.0335, -0.0268,  0.0133, -0.0627,  0.0074, -0.0488, -0.0353,
                    -0.0075, -0.0069,  0.0098, -0.0058,  0.0033, -0.0157, -0.0004, -0.0381,
                    -0.0452], dtype=torch.float64)}, {})
            >>> # same environment with numpy observations
            >>> envgym = gym.make("DMC-cheetah-v0", transform=CatTensors(in_keys=["position", "velocity"], out_key="observation"), to_numpy=True)
            >>> envgym.reset()
            ({'observation': array([-0.11355747,  0.04257728,  0.00408397,  0.04155852, -0.0389733 ,
                   -0.01409826, -0.0978704 , -0.08808327,  0.03970837,  0.00535434,
                   -0.02353762,  0.05116226,  0.02788907,  0.06848346,  0.05154399,
                    0.0371798 ,  0.05128025])}, {})
            >>> # If gymnasium is installed, we can register the environment there too.
            >>> DMControlEnv.register_gym("DMC-cheetah-v0", to_numpy=False, backend="gymnasium", env_name="cheetah", task_name="run")
            >>> import gymnasium
            >>> envgym = gymnasium.make("DMC-cheetah-v0")
            >>> envgym.seed(0)
            >>> torch.manual_seed(0)
            >>> envgym.reset()
            ({'position': tensor([-0.0855,  0.0215, -0.0881, -0.0412, -0.1101,  0.0080,  0.0254,  0.0424],
                   dtype=torch.float64), 'velocity': tensor([ 1.9609e-02, -1.9776e-04, -1.6347e-03,  3.3842e-02,  2.5338e-02,
                     3.3064e-02,  1.0381e-04,  7.6656e-05,  1.0204e-02],
                   dtype=torch.float64)}, {})

        .. note::
            This feature also works for stateless environments (eg, :class:`~torchrl.envs.BraxEnv`).

                >>> import gymnasium
                >>> import torch
                >>> from tensordict import TensorDict
                >>> from torchrl.envs import BraxEnv, SelectTransform
                >>>
                >>> # get action for dydactic purposes
                >>> env = BraxEnv("ant", batch_size=[2])
                >>> env.set_seed(0)
                >>> torch.manual_seed(0)
                >>> td = env.rollout(10)
                >>>
                >>> actions = td.get("action")
                >>>
                >>> # register env
                >>> env.register_gym("Brax-Ant-v0", env_name="ant", batch_size=[2], info_keys=["state"])
                >>> gym_env = gymnasium.make("Brax-Ant-v0")
                >>> gym_env.seed(0)
                >>> torch.manual_seed(0)
                >>>
                >>> gym_env.reset()
                >>> obs = []
                >>> for i in range(10):
                ...     obs, reward, terminated, truncated, info = gym_env.step(td[..., i].get("action"))


        """
        from torchrl.envs.libs.gym import gym_backend, set_gym_backend

        if backend is None:
            backend = gym_backend()

        with set_gym_backend(backend):
            return cls._register_gym(
                id=id,
                entry_point=entry_point,
                transform=transform,
                info_keys=info_keys,
                to_numpy=to_numpy,
                reward_threshold=reward_threshold,
                nondeterministic=nondeterministic,
                max_episode_steps=max_episode_steps,
                order_enforce=order_enforce,
                autoreset=autoreset,
                disable_env_checker=disable_env_checker,
                apply_api_compatibility=apply_api_compatibility,
                **kwargs,
            )

    _GYM_UNRECOGNIZED_KWARG = (
        "The keyword argument {} is not compatible with gym version {}"
    )

    @implement_for("gym", "0.26", None, class_method=True)
    def _register_gym(
        cls,
        id,
        entry_point: Callable | None = None,
        transform: Transform | None = None,  # noqa: F821
        info_keys: list[NestedKey] | None = None,
        to_numpy: bool = False,
        reward_threshold: float | None = None,
        nondeterministic: bool = False,
        max_episode_steps: int | None = None,
        order_enforce: bool = True,
        autoreset: bool | None = None,
        disable_env_checker: bool = False,
        apply_api_compatibility: bool = False,
        **kwargs,
    ):
        import gym
        from torchrl.envs.libs._gym_utils import _TorchRLGymWrapper

        if entry_point is None:
            entry_point = cls
        entry_point = partial(
            _TorchRLGymWrapper,
            entry_point=entry_point,
            info_keys=info_keys,
            to_numpy=to_numpy,
            transform=transform,
            **kwargs,
        )
        return gym.register(
            id=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
            order_enforce=order_enforce,
            autoreset=bool(autoreset),
            disable_env_checker=disable_env_checker,
            apply_api_compatibility=apply_api_compatibility,
        )

    @implement_for("gym", "0.25", "0.26", class_method=True)
    def _register_gym(  # noqa: F811
        cls,
        id,
        entry_point: Callable | None = None,
        transform: Transform | None = None,  # noqa: F821
        info_keys: list[NestedKey] | None = None,
        to_numpy: bool = False,
        reward_threshold: float | None = None,
        nondeterministic: bool = False,
        max_episode_steps: int | None = None,
        order_enforce: bool = True,
        autoreset: bool | None = None,
        disable_env_checker: bool = False,
        apply_api_compatibility: bool = False,
        **kwargs,
    ):
        import gym

        if apply_api_compatibility is not False:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format(
                    "apply_api_compatibility", gym.__version__
                )
            )
        from torchrl.envs.libs._gym_utils import _TorchRLGymWrapper

        if entry_point is None:
            entry_point = cls
        entry_point = partial(
            _TorchRLGymWrapper,
            entry_point=entry_point,
            info_keys=info_keys,
            to_numpy=to_numpy,
            transform=transform,
            **kwargs,
        )
        return gym.register(
            id=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
            order_enforce=order_enforce,
            autoreset=bool(autoreset),
            disable_env_checker=disable_env_checker,
        )

    @implement_for("gym", "0.24", "0.25", class_method=True)
    def _register_gym(  # noqa: F811
        cls,
        id,
        entry_point: Callable | None = None,
        transform: Transform | None = None,  # noqa: F821
        info_keys: list[NestedKey] | None = None,
        to_numpy: bool = False,
        reward_threshold: float | None = None,
        nondeterministic: bool = False,
        max_episode_steps: int | None = None,
        order_enforce: bool = True,
        autoreset: bool | None = None,
        disable_env_checker: bool = False,
        apply_api_compatibility: bool = False,
        **kwargs,
    ):
        import gym

        if apply_api_compatibility is not False:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format(
                    "apply_api_compatibility", gym.__version__
                )
            )
        if disable_env_checker is not False:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format(
                    "disable_env_checker", gym.__version__
                )
            )
        from torchrl.envs.libs._gym_utils import _TorchRLGymWrapper

        if entry_point is None:
            entry_point = cls
        entry_point = partial(
            _TorchRLGymWrapper,
            entry_point=entry_point,
            info_keys=info_keys,
            to_numpy=to_numpy,
            transform=transform,
            **kwargs,
        )
        return gym.register(
            id=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
            order_enforce=order_enforce,
            autoreset=bool(autoreset),
        )

    @implement_for("gym", "0.21", "0.24", class_method=True)
    def _register_gym(  # noqa: F811
        cls,
        id,
        entry_point: Callable | None = None,
        transform: Transform | None = None,  # noqa: F821
        info_keys: list[NestedKey] | None = None,
        to_numpy: bool = False,
        reward_threshold: float | None = None,
        nondeterministic: bool = False,
        max_episode_steps: int | None = None,
        order_enforce: bool = True,
        autoreset: bool | None = None,
        disable_env_checker: bool = False,
        apply_api_compatibility: bool = False,
        **kwargs,
    ):
        import gym

        if apply_api_compatibility is not False:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format(
                    "apply_api_compatibility", gym.__version__
                )
            )
        if disable_env_checker is not False:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format(
                    "disable_env_checker", gym.__version__
                )
            )
        if autoreset is not None:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format("autoreset", gym.__version__)
            )
        from torchrl.envs.libs._gym_utils import _TorchRLGymWrapper

        if entry_point is None:
            entry_point = cls
        entry_point = partial(
            _TorchRLGymWrapper,
            entry_point=entry_point,
            info_keys=info_keys,
            to_numpy=to_numpy,
            transform=transform,
            **kwargs,
        )
        return gym.register(
            id=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
            order_enforce=order_enforce,
        )

    @implement_for("gym", None, "0.21", class_method=True)
    def _register_gym(  # noqa: F811
        cls,
        id,
        entry_point: Callable | None = None,
        transform: Transform | None = None,  # noqa: F821
        info_keys: list[NestedKey] | None = None,
        to_numpy: bool = False,
        reward_threshold: float | None = None,
        nondeterministic: bool = False,
        max_episode_steps: int | None = None,
        order_enforce: bool = True,
        autoreset: bool | None = None,
        disable_env_checker: bool = False,
        apply_api_compatibility: bool = False,
        **kwargs,
    ):
        import gym
        from torchrl.envs.libs._gym_utils import _TorchRLGymWrapper

        if order_enforce is not True:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format("order_enforce", gym.__version__)
            )
        if disable_env_checker is not False:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format(
                    "disable_env_checker", gym.__version__
                )
            )
        if autoreset is not None:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format("autoreset", gym.__version__)
            )
        if apply_api_compatibility is not False:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format(
                    "apply_api_compatibility", gym.__version__
                )
            )
        if entry_point is None:
            entry_point = cls
        entry_point = partial(
            _TorchRLGymWrapper,
            entry_point=entry_point,
            info_keys=info_keys,
            to_numpy=to_numpy,
            transform=transform,
            **kwargs,
        )
        return gym.register(
            id=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
        )

    @implement_for("gymnasium", None, "1.0.0", class_method=True)
    def _register_gym(  # noqa: F811
        cls,
        id,
        entry_point: Callable | None = None,
        transform: Transform | None = None,  # noqa: F821
        info_keys: list[NestedKey] | None = None,
        to_numpy: bool = False,
        reward_threshold: float | None = None,
        nondeterministic: bool = False,
        max_episode_steps: int | None = None,
        order_enforce: bool = True,
        autoreset: bool | None = None,
        disable_env_checker: bool = False,
        apply_api_compatibility: bool = False,
        **kwargs,
    ):
        import gymnasium
        from torchrl.envs.libs._gym_utils import _TorchRLGymnasiumWrapper

        if entry_point is None:
            entry_point = cls

        entry_point = partial(
            _TorchRLGymnasiumWrapper,
            entry_point=entry_point,
            info_keys=info_keys,
            to_numpy=to_numpy,
            transform=transform,
            **kwargs,
        )
        return gymnasium.register(
            id=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
            order_enforce=order_enforce,
            autoreset=bool(autoreset),
            disable_env_checker=disable_env_checker,
            apply_api_compatibility=apply_api_compatibility,
        )

    @implement_for("gymnasium", "1.1.0", class_method=True)
    def _register_gym(  # noqa: F811
        cls,
        id,
        entry_point: Callable | None = None,
        transform: Transform | None = None,  # noqa: F821
        info_keys: list[NestedKey] | None = None,
        to_numpy: bool = False,
        reward_threshold: float | None = None,
        nondeterministic: bool = False,
        max_episode_steps: int | None = None,
        order_enforce: bool = True,
        autoreset: bool | None = None,
        disable_env_checker: bool = False,
        apply_api_compatibility: bool = False,
        **kwargs,
    ):
        import gymnasium
        from torchrl.envs.libs._gym_utils import _TorchRLGymnasiumWrapper

        if autoreset is not None:
            raise TypeError(
                f"the autoreset argument is deprecated in gymnasium>=1.0. Got autoreset={autoreset}"
            )
        if entry_point is None:
            entry_point = cls

        entry_point = partial(
            _TorchRLGymnasiumWrapper,
            entry_point=entry_point,
            info_keys=info_keys,
            to_numpy=to_numpy,
            transform=transform,
            **kwargs,
        )
        if apply_api_compatibility is not False:
            raise TypeError(
                cls._GYM_UNRECOGNIZED_KWARG.format(
                    "apply_api_compatibility", gymnasium.__version__
                )
            )
        return gymnasium.register(
            id=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
            order_enforce=order_enforce,
            disable_env_checker=disable_env_checker,
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "EnvBase.forward is not implemented. If you ended here during a call to `ParallelEnv(...)`, please use "
            "a constructor such as `ParallelEnv(num_env, lambda env=env: env)` instead. "
            "Batched envs require constructors because environment instances may not always be serializable."
        )

    @abc.abstractmethod
    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        raise NotImplementedError

    @abc.abstractmethod
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        raise NotImplementedError

    def reset(
        self,
        tensordict: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        """Resets the environment.

        As for step and _step, only the private method :obj:`_reset` should be overwritten by EnvBase subclasses.

        Args:
            tensordict (TensorDictBase, optional): tensordict to be used to contain the resulting new observation.
                In some cases, this input can also be used to pass argument to the reset function.
            kwargs (optional): other arguments to be passed to the native
                reset function.

        Returns:
            a tensordict (or the input tensordict, if any), modified in place with the resulting observations.

        .. note:: `reset` should not be overwritten by :class:`~torchrl.envs.EnvBase` subclasses. The method to
            modify is :meth:`~torchrl.envs.EnvBase._reset`.

        """
        if tensordict is not None:
            self._assert_tensordict_shape(tensordict)

        select_reset_only = kwargs.pop("select_reset_only", False)
        if select_reset_only and tensordict is not None:
            # When making rollouts with step_and_maybe_reset, it can happen that a tensordict has
            # keys that are used by reset to optionally set the reset state (eg, the fen in chess). If that's the
            # case and we don't throw them away here, reset will just be a no-op (put the env in the state reached
            # during the previous step).
            # Therefore, maybe_reset tells reset to temporarily hide the non-reset keys.
            # To make step_and_maybe_reset handle custom reset states, some version of TensorDictPrimer should be used.
            tensordict_reset = self._reset(
                tensordict.select(*self.reset_keys, strict=False), **kwargs
            )
        else:
            tensordict_reset = self._reset(tensordict, **kwargs)
        # We assume that this is done properly
        # if reset.device != self.device:
        #     reset = reset.to(self.device, non_blocking=True)
        if tensordict_reset is tensordict:
            raise RuntimeError(
                "EnvBase._reset should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty()) "
                "inside _reset before writing new tensors onto this new instance."
            )
        if not isinstance(tensordict_reset, TensorDictBase):
            raise RuntimeError(
                f"env._reset returned an object of type {type(tensordict_reset)} but a TensorDict was expected."
            )
        return self._reset_proc_data(tensordict, tensordict_reset)

    def _reset_proc_data(self, tensordict, tensordict_reset):
        self._complete_done(self.full_done_spec, tensordict_reset)
        self._reset_check_done(tensordict, tensordict_reset)
        if tensordict is not None:
            return _update_during_reset(tensordict_reset, tensordict, self.reset_keys)
        return tensordict_reset

    def _reset_check_done(self, tensordict, tensordict_reset):
        """Checks the done status after reset.

        If _reset signals were passed, we check that the env is not done for these
        indices.

        We also check that the input tensordict contained ``"done"``s if the
        reset is partial and incomplete.

        """
        # we iterate over (reset_key, (done_key, truncated_key)) and check that all
        # values where reset was true now have a done set to False.
        # If no reset was present, all done and truncated must be False
        for reset_key, done_key_group in zip(self.reset_keys, self.done_keys_groups):
            reset_value = (
                tensordict.get(reset_key, default=None)
                if tensordict is not None
                else None
            )
            if reset_value is not None:
                for done_key in done_key_group:
                    done_val = tensordict_reset.get(done_key)
                    if (
                        done_val.any()
                        and done_val[reset_value].any()
                        and not self._allow_done_after_reset
                    ):
                        raise RuntimeError(
                            f"Env done entry '{done_key}' was (partially) True after reset on specified '_reset' dimensions. This is not allowed."
                        )
                    if (
                        done_key not in tensordict.keys(True)
                        and done_val[~reset_value].any()
                    ):
                        warnings.warn(
                            f"A partial `'_reset'` key has been passed to `reset` ({reset_key}), "
                            f"but the corresponding done_key ({done_key}) was not present in the input "
                            f"tensordict. "
                            f"This is discouraged, since the input tensordict should contain "
                            f"all the data not being reset."
                        )
                        # we set the done val to tensordict, to make sure that
                        # _update_during_reset does not pad the value
                        tensordict.set(done_key, done_val)
            elif not self._allow_done_after_reset:
                for done_key in done_key_group:
                    if tensordict_reset.get(done_key).any():
                        raise RuntimeError(
                            f"The done entry '{done_key}' was (partially) True after a call to reset() in env {self}."
                        )

    def numel(self) -> int:
        return prod(self.batch_size)

    def set_seed(
        self, seed: int | None = None, static_seed: bool = False
    ) -> int | None:
        """Sets the seed of the environment and returns the next seed to be used (which is the input seed if a single environment is present).

        Args:
            seed (int): seed to be set. The seed is set only locally in the environment. To handle the global seed,
                see :func:`~torch.manual_seed`.
            static_seed (bool, optional): if ``True``, the seed is not incremented.
                Defaults to False

        Returns:
            integer representing the "next seed": i.e. the seed that should be
            used for another environment if created concomitantly to this environment.

        """
        self._set_seed(seed)
        if seed is not None and not static_seed:
            new_seed = seed_generator(seed)
            seed = new_seed
        return seed

    @abc.abstractmethod
    def _set_seed(self, seed: int | None):
        raise NotImplementedError

    def set_state(self):
        raise NotImplementedError

    def _assert_tensordict_shape(self, tensordict: TensorDictBase) -> None:
        if (
            self.batch_locked or self.batch_size != ()
        ) and tensordict.batch_size != self.batch_size:
            raise RuntimeError(
                f"Expected a tensordict with shape==env.batch_size, "
                f"got {tensordict.batch_size} and {self.batch_size}"
            )

    def all_actions(self, tensordict: TensorDictBase | None = None) -> TensorDictBase:
        """Generates all possible actions from the action spec.

        This only works in environments with fully discrete actions.

        Args:
            tensordict (TensorDictBase, optional): If given, :meth:`~.reset`
                is called with this tensordict.

        Returns:
            a tensordict object with the "action" entry updated with a batch of
            all possible actions. The actions are stacked together in the
            leading dimension.
        """
        if tensordict is not None:
            self.reset(tensordict)

        return self.full_action_spec.enumerate(use_mask=True)

    def rand_action(self, tensordict: TensorDictBase | None = None):
        """Performs a random action given the action_spec attribute.

        Args:
            tensordict (TensorDictBase, optional): tensordict where the resulting action should be written.

        Returns:
            a tensordict object with the "action" entry updated with a random
            sample from the action-spec.

        """
        shape = torch.Size([])
        if not self.batch_locked:
            if not self.batch_size and tensordict is not None:
                # if we can't infer the batch-size from the env, take it from tensordict
                shape = tensordict.shape
            elif not self.batch_size:
                # if tensordict wasn't provided, we assume empty batch size
                shape = torch.Size([])
            elif tensordict.shape != self.batch_size:
                # if tensordict is not None and the env has a batch size, their shape must match
                raise RuntimeError(
                    "The input tensordict and the env have a different batch size: "
                    f"env.batch_size={self.batch_size} and tensordict.batch_size={tensordict.shape}. "
                    f"Non batch-locked environment require the env batch-size to be either empty or to"
                    f" match the tensordict one."
                )
        # We generate the action from the full_action_spec
        r = self.input_spec["full_action_spec"].rand(shape)
        if tensordict is None:
            return r
        tensordict.update(r)
        return tensordict

    def rand_step(self, tensordict: TensorDictBase | None = None) -> TensorDictBase:
        """Performs a random step in the environment given the action_spec attribute.

        Args:
            tensordict (TensorDictBase, optional): tensordict where the resulting info should be written.

        Returns:
            a tensordict object with the new observation after a random step in the environment. The action will
            be stored with the "action" key.

        """
        tensordict = self.rand_action(tensordict)
        return self.step(tensordict)

    @property
    def specs(self) -> Composite:
        """Returns a Composite container where all the environment are present.

        This feature allows one to create an environment, retrieve all of the specs in a single data container and then
        erase the environment from the workspace.

        """
        return Composite(
            output_spec=self.output_spec,
            input_spec=self.input_spec,
            shape=self.batch_size,
        )

    @property
    @_cache_value
    def _has_dynamic_specs(self) -> bool:
        return _has_dynamic_specs(self.specs)

    def rollout(
        self,
        max_steps: int,
        policy: Callable[[TensorDictBase], TensorDictBase] | None = None,
        callback: Callable[[TensorDictBase, ...], Any] | None = None,
        *,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool | None = None,
        break_when_all_done: bool | None = None,
        return_contiguous: bool | None = False,
        tensordict: TensorDictBase | None = None,
        set_truncated: bool = False,
        out=None,
        trust_policy: bool = False,
    ) -> TensorDictBase:
        """Executes a rollout in the environment.

        The function will return as soon as any of the contained environments
        reaches any of the done states.

        Args:
            max_steps (int): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before max_steps have been executed.
            policy (callable, optional): callable to be called to compute the desired action.
                If no policy is provided, actions will be called using :obj:`env.rand_step()`.
                The policy can be any callable that reads either a tensordict or
                the entire sequence of observation entries __sorted as__ the ``env.observation_spec.keys()``.
                Defaults to `None`.
            callback (Callable[[TensorDict], Any], optional): function to be called at each iteration with the given
                TensorDict. Defaults to ``None``. The output of ``callback`` will not be collected, it is the user
                responsibility to save any result within the callback call if data needs to be carried over beyond
                the call to ``rollout``.

        Keyword Args:
            auto_reset (bool, optional): if ``True``, the contained environments will be reset before starting the
                rollout. If ``False``, then the rollout will continue from a previous state, which requires the
                ``tensordict`` argument to be passed with the previous rollout. Default is ``True``.
            auto_cast_to_device (bool, optional): if ``True``, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is ``False``.
            break_when_any_done (bool): if ``True``, break when any of the contained environments reaches any of the
                done states. If ``False``, then the done environments are reset automatically. Default is ``True``.

                .. seealso:: The :ref:`Partial resets <ref_partial_resets>` of the documentation gives more
                    information about partial resets.

            break_when_all_done (bool, optional): if ``True``, break if all of the contained environments reach any
                of the done states. If ``False``, break if at least one environment reaches any of the done states.
                Default is ``False``.

                .. seealso:: The :ref:`Partial steps <ref_partial_steps>` of the documentation gives more
                    information about partial resets.

            return_contiguous (bool): if False, a LazyStackedTensorDict will be returned. Default is `True` if
                the env does not have dynamic specs, otherwise `False`.
            tensordict (TensorDict, optional): if ``auto_reset`` is False, an initial
                tensordict must be provided. Rollout will check if this tensordict has done flags and reset the
                environment in those dimensions (if needed).
                This normally should not occur if ``tensordict`` is the output of a reset, but can occur
                if ``tensordict`` is the last step of a previous rollout.
                A ``tensordict`` can also be provided when ``auto_reset=True`` if metadata need to be passed
                to the ``reset`` method, such as a batch-size or a device for stateless environments.
            set_truncated (bool, optional): if ``True``, ``"truncated"`` and ``"done"`` keys will be set to
                ``True`` after completion of the rollout. If no ``"truncated"`` is found within the
                ``done_spec``, an exception is raised.
                Truncated keys can be set through ``env.add_truncated_keys``.
                Defaults to ``False``.
            trust_policy (bool, optional): if ``True``, a non-TensorDictModule policy will be trusted to be
                assumed to be compatible with the collector. This defaults to ``True`` for CudaGraphModules
                and ``False`` otherwise.

        Returns:
            TensorDict object containing the resulting trajectory.

        The data returned will be marked with a "time" dimension name for the last
        dimension of the tensordict (at the ``env.ndim`` index).

        ``rollout`` is quite handy to display what the data structure of the
        environment looks like.

        Examples:
            >>> # Using rollout without a policy
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> from torchrl.envs.transforms import TransformedEnv, StepCounter
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), StepCounter(max_steps=20))
            >>> rollout = env.rollout(max_steps=1000)
            >>> print(rollout)
            TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                    done: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            step_count: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                            truncated: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([20]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                    step_count: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    truncated: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([20]),
                device=cpu,
                is_shared=False)
            >>> print(rollout.names)
            ['time']
            >>> # with envs that contain more dimensions
            >>> from torchrl.envs import SerialEnv
            >>> env = SerialEnv(3, lambda: TransformedEnv(GymEnv("Pendulum-v1"), StepCounter(max_steps=20)))
            >>> rollout = env.rollout(max_steps=1000)
            >>> print(rollout)
            TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                    done: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([3, 20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            step_count: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                            truncated: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([3, 20]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([3, 20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                    step_count: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    truncated: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([3, 20]),
                device=cpu,
                is_shared=False)
            >>> print(rollout.names)
            [None, 'time']

        Using a policy (a regular :class:`~torch.nn.Module` or a :class:`~tensordict.nn.TensorDictModule`)
        is also easy:

        Examples:
            >>> from torch import nn
            >>> env = GymEnv("CartPole-v1", categorical_action_encoding=True)
            >>> class ArgMaxModule(nn.Module):
            ...     def forward(self, values):
            ...         return values.argmax(-1)
            >>> n_obs = env.observation_spec["observation"].shape[-1]
            >>> n_act = env.action_spec.n
            >>> # A deterministic policy
            >>> policy = nn.Sequential(
            ...     nn.Linear(n_obs, n_act),
            ...     ArgMaxModule())
            >>> env.rollout(max_steps=10, policy=policy)
            TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
                    done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([10]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([10]),
                device=cpu,
                is_shared=False)
            >>> # Under the hood, rollout will wrap the policy in a TensorDictModule
            >>> # To speed things up we can do that ourselves
            >>> from tensordict.nn import TensorDictModule
            >>> policy = TensorDictModule(policy, in_keys=list(env.observation_spec.keys()), out_keys=["action"])
            >>> env.rollout(max_steps=10, policy=policy)
            TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
                    done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([10]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([10]),
                device=cpu,
                is_shared=False)


        In some instances, contiguous tensordict cannot be obtained because
        they cannot be stacked. This can happen when the data returned at
        each step may have a different shape, or when different environments
        are executed together. In that case, ``return_contiguous=False``
        will cause the returned tensordict to be a lazy stack of tensordicts:

        Examples of non-contiguous rollout:
            >>> rollout = env.rollout(4, return_contiguous=False)
            >>> print(rollout)
            LazyStackedTensorDict(
                fields={
                    action: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                    done: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: LazyStackedTensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([3, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            step_count: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                            truncated: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([3, 4]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([3, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                    step_count: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    truncated: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([3, 4]),
                device=cpu,
                is_shared=False)
                >>> print(rollout.names)
                [None, 'time']

        Rollouts can be used in a loop to emulate data collection.
        To do so, you need to pass as input the last tensordict coming from the previous rollout after calling
        :func:`~torchrl.envs.utils.step_mdp` on it.

        Examples of data collection rollouts:
            >>> from torchrl.envs import GymEnv, step_mdp
            >>> env = GymEnv("CartPole-v1")
            >>> epochs = 10
            >>> input_td = env.reset()
            >>> for i in range(epochs):
            ...     rollout_td = env.rollout(
            ...         max_steps=100,
            ...         break_when_any_done=False,
            ...         auto_reset=False,
            ...         tensordict=input_td,
            ...     )
            ...     input_td = step_mdp(
            ...         rollout_td[..., -1],
            ...     )

        """
        if break_when_any_done is None:  # True by default
            if break_when_all_done:  # all overrides
                break_when_any_done = False
            else:
                break_when_any_done = True
        if break_when_all_done is None:
            # There is no case where break_when_all_done is True by default
            break_when_all_done = False
        if break_when_all_done and break_when_any_done:
            raise TypeError(
                "Cannot have both break_when_all_done and break_when_any_done True at the same time."
            )
        if return_contiguous is None:
            return_contiguous = not self._has_dynamic_specs
        if policy is not None:
            policy = _make_compatible_policy(
                policy,
                self.observation_spec,
                env=self,
                fast_wrap=True,
                trust_policy=trust_policy,
            )
            if auto_cast_to_device:
                try:
                    policy_device = next(policy.parameters()).device
                except (StopIteration, AttributeError):
                    policy_device = None
            else:
                policy_device = None
        else:
            policy = self.rand_action
            policy_device = None

        env_device = self.device

        if auto_reset:
            tensordict = self.reset(tensordict)
        elif tensordict is None:
            raise RuntimeError("tensordict must be provided when auto_reset is False")
        else:
            tensordict = self.maybe_reset(tensordict)

        kwargs = {
            "tensordict": tensordict,
            "auto_cast_to_device": auto_cast_to_device,
            "max_steps": max_steps,
            "policy": policy,
            "policy_device": policy_device,
            "env_device": env_device,
            "callback": callback,
        }
        if break_when_any_done or break_when_all_done:
            tensordicts = self._rollout_stop_early(
                break_when_all_done=break_when_all_done,
                break_when_any_done=break_when_any_done,
                **kwargs,
            )
        else:
            tensordicts = self._rollout_nonstop(**kwargs)
        batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        if return_contiguous:
            try:
                out_td = torch.stack(tensordicts, len(batch_size), out=out)
            except RuntimeError as err:
                if (
                    "The shapes of the tensors to stack is incompatible" in str(err)
                    and self._has_dynamic_specs
                ):
                    raise RuntimeError(
                        "The environment specs are dynamic. Call rollout with return_contiguous=False."
                    )
                raise
        else:
            out_td = LazyStackedTensorDict.maybe_dense_stack(
                tensordicts, len(batch_size), out=out
            )
        if set_truncated:
            found_truncated = False
            for key in self.done_keys:
                if _ends_with(key, "truncated"):
                    val = out_td.get(("next", key))
                    done = out_td.get(("next", _replace_last(key, "done")))
                    val[(slice(None),) * (out_td.ndim - 1) + (-1,)] = True
                    out_td.set(("next", key), val)
                    out_td.set(("next", _replace_last(key, "done")), val | done)
                    found_truncated = True
            if not found_truncated:
                raise RuntimeError(
                    "set_truncated was set to True but no truncated key could be found. "
                    "Make sure a 'truncated' entry was set in the environment "
                    "full_done_keys using `env.add_truncated_keys()`."
                )

        out_td.refine_names(..., "time")
        return out_td

    @_maybe_unlock
    def add_truncated_keys(self) -> EnvBase:
        """Adds truncated keys to the environment."""
        i = 0
        for key in self.done_keys:
            i += 1
            truncated_key = _replace_last(key, "truncated")
            self.full_done_spec[truncated_key] = self.full_done_spec[key].clone()
        if i == 0:
            raise KeyError(f"Couldn't find done keys. done_spec={self.full_done_specs}")

        return self

    def step_mdp(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        """Advances the environment state by one step using the provided `next_tensordict`.

        This method updates the environment's state by transitioning from the current
        state to the next, as defined by the `next_tensordict`. The resulting tensordict
        includes updated observations and any other relevant state information, with
        keys managed according to the environment's specifications.

        Internally, this method utilizes a precomputed :class:`~torchrl.envs.utils._StepMDP` instance to efficiently
        handle the transition of state, observation, action, reward, and done keys. The
        :class:`~torchrl.envs.utils._StepMDP` class optimizes the process by precomputing the keys to include and
        exclude, reducing runtime overhead during repeated calls. The :class:`~torchrl.envs.utils._StepMDP` instance
        is created with `exclude_action=False`, meaning that action keys are retained in
        the root tensordict.

        Args:
            next_tensordict (TensorDictBase): A tensordict containing the state of the
                environment at the next time step. This tensordict should include keys
                for observations, actions, rewards, and done flags, as defined by the
                environment's specifications.

        Returns:
            TensorDictBase: A new tensordict representing the environment state after
            advancing by one step.

        .. note:: The method ensures that the environment's key specifications are validated
              against the provided `next_tensordict`, issuing warnings if discrepancies
              are found.

        .. note:: This method is designed to work efficiently with environments that have
              consistent key specifications, leveraging the `_StepMDP` class to minimize
              overhead.

        Example:
            >>> from torchrl.envs import GymEnv
            >>> env = GymEnv("Pendulum-1")
            >>> data = env.reset()
            >>> for i in range(10):
            ...     # compute action
            ...     env.rand_action(data)
            ...     # Perform action
            ...     next_data = env.step(reset_data)
            ...     data = env.step_mdp(next_data)
        """
        return self._step_mdp(next_tensordict)

    @property
    @_cache_value
    def _step_mdp(self):
        return _StepMDP(self, exclude_action=False)

    def _rollout_stop_early(
        self,
        *,
        break_when_any_done,
        break_when_all_done,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
    ):
        # Get the sync func
        if auto_cast_to_device:
            sync_func = _get_sync_func(policy_device, env_device)
        tensordicts = []
        partial_steps = True
        for i in range(max_steps):
            if auto_cast_to_device:
                if policy_device is not None:
                    tensordict = tensordict.to(policy_device, non_blocking=True)
                    sync_func()
                else:
                    tensordict.clear_device_()
            # In case policy(..) does not modify in-place - no-op for TensorDict and related
            tensordict.update(policy(tensordict))
            if auto_cast_to_device:
                if env_device is not None:
                    tensordict = tensordict.to(env_device, non_blocking=True)
                    sync_func()
                else:
                    tensordict.clear_device_()
            tensordict = self.step(tensordict)
            td_append = tensordict.copy()
            if break_when_all_done:
                if partial_steps is not True and not partial_steps.all():
                    # At least one step is partial
                    td_append.pop("_step", None)
                    td_append = torch.where(
                        partial_steps.view(td_append.shape), td_append, tensordicts[-1]
                    )

            tensordicts.append(td_append)

            if i == max_steps - 1:
                # we don't truncate as one could potentially continue the run
                break
            tensordict = self._step_mdp(tensordict)

            if break_when_any_done:
                # done and truncated are in done_keys
                # We read if any key is done.
                any_done = _terminated_or_truncated(
                    tensordict,
                    full_done_spec=self.output_spec["full_done_spec"],
                    key=None,
                )
                if any_done:
                    break
            else:
                # Write the '_step' entry, indicating which step is to be undertaken
                _terminated_or_truncated(
                    tensordict,
                    full_done_spec=self.output_spec["full_done_spec"],
                    key="_neg_step",
                    write_full_false=False,
                )
                # This is what differentiates _step and _reset: we need to flip _step False -> True
                partial_step_curr = tensordict.pop("_neg_step", None)
                if partial_step_curr is not None:
                    partial_step_curr = ~partial_step_curr
                    partial_steps = partial_steps & partial_step_curr
                if partial_steps is not True:
                    if not partial_steps.any():
                        break
                    # Write the final _step entry
                    tensordict.set("_step", partial_steps)

            if callback is not None:
                callback(self, tensordict)
        return tensordicts

    def _rollout_nonstop(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
    ):
        if auto_cast_to_device:
            sync_func = _get_sync_func(policy_device, env_device)
        tensordicts = []
        tensordict_ = tensordict
        for i in range(max_steps):
            if auto_cast_to_device:
                if policy_device is not None:
                    tensordict_ = tensordict_.to(policy_device, non_blocking=True)
                    sync_func()
                else:
                    tensordict_.clear_device_()
            # In case policy(..) does not modify in-place - no-op for TensorDict and related
            tensordict_.update(policy(tensordict_))
            if auto_cast_to_device:
                if env_device is not None:
                    tensordict_ = tensordict_.to(env_device, non_blocking=True)
                    sync_func()
                else:
                    tensordict_.clear_device_()
            if i == max_steps - 1:
                tensordict = self.step(tensordict_)
            else:
                tensordict, tensordict_ = self.step_and_maybe_reset(tensordict_)
            tensordicts.append(tensordict)
            if i == max_steps - 1:
                # we don't truncate as one could potentially continue the run
                break
            if callback is not None:
                callback(self, tensordict)

        return tensordicts

    def step_and_maybe_reset(
        self, tensordict: TensorDictBase
    ) -> tuple[TensorDictBase, TensorDictBase]:
        """Runs a step in the environment and (partially) resets it if needed.

        Args:
            tensordict (TensorDictBase): an input data structure for the :meth:`step`
                method.

        This method allows to easily code non-stopping rollout functions.

        Examples:
            >>> from torchrl.envs import ParallelEnv, GymEnv
            >>> def rollout(env, n):
            ...     data_ = env.reset()
            ...     result = []
            ...     for i in range(n):
            ...         data, data_ = env.step_and_maybe_reset(data_)
            ...         result.append(data)
            ...     return torch.stack(result)
            >>> env = ParallelEnv(2, lambda: GymEnv("CartPole-v1"))
            >>> print(rollout(env, 2))
            TensorDict(
                fields={
                    done: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([2, 2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            terminated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            truncated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([2, 2]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([2, 2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    terminated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    truncated: Tensor(shape=torch.Size([2, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([2, 2]),
                device=cpu,
                is_shared=False)
        """
        if tensordict.device != self.device:
            tensordict = tensordict.to(self.device)
        tensordict = self.step(tensordict)
        # done and truncated are in done_keys
        # We read if any key is done.
        tensordict_ = self._step_mdp(tensordict)
        tensordict_ = self.maybe_reset(tensordict_)
        return tensordict, tensordict_

    @property
    @_cache_value
    def _simple_done(self):
        key_set = set(self.full_done_spec.keys())
        _simple_done = key_set == {
            "done",
            "truncated",
            "terminated",
        } or key_set == {"done", "terminated"}
        return _simple_done

    def any_done(self, tensordict: TensorDictBase) -> bool:
        """Checks if the tensordict is in a "done" state (or if an element of the batch is).

        Writes the result under the `"_reset"` entry.

        Returns: a bool indicating whether there is an element in the tensordict that is marked
            as done.

        .. note:: The tensordict passed should be a `"next"` tensordict or equivalent -- i.e., it should not
            contain a `"next"` value.

        """
        if self._simple_done:
            done = tensordict._get_str("done", default=None)
            if done is not None:
                any_done = done.any()
            else:
                any_done = False
            if any_done:
                tensordict._set_str(
                    "_reset",
                    done.clone(),
                    validated=True,
                    inplace=False,
                    non_blocking=False,
                )
        else:
            any_done = _terminated_or_truncated(
                tensordict,
                full_done_spec=self.output_spec["full_done_spec"],
                key="_reset",
            )
        return any_done

    def maybe_reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Checks the done keys of the input tensordict and, if needed, resets the environment where it is done.

        Args:
            tensordict (TensorDictBase): a tensordict coming from the output of :func:`~torchrl.envs.utils.step_mdp`.

        Returns:
            A tensordict that is identical to the input where the environment was
            not reset and contains the new reset data where the environment was reset.

        """
        any_done = self.any_done(tensordict)
        if any_done:
            tensordict = self.reset(tensordict, select_reset_only=True)
        return tensordict

    def empty_cache(self):
        """Erases all the cached values.

        For regular envs, the key lists (reward, done etc) are cached, but in some cases
        they may change during the execution of the code (eg, when adding a transform).

        """
        self._cache.clear()

    @property
    @_cache_value
    def reset_keys(self) -> list[NestedKey]:
        """Returns a list of reset keys.

        Reset keys are keys that indicate partial reset, in batched, multitask or multiagent
        settings. They are structured as ``(*prefix, "_reset")`` where ``prefix`` is
        a (possibly empty) tuple of strings pointing to a tensordict location
        where a done state can be found.

        Keys are sorted by depth in the data tree.
        """
        reset_keys = sorted(
            (
                _replace_last(done_key, "_reset")
                for (done_key, *_) in self.done_keys_groups
            ),
            key=_repr_by_depth,
        )
        return reset_keys

    @property
    def _filtered_reset_keys(self):
        """Returns only the effective reset keys, discarding nested resets if they're not being used."""
        reset_keys = self.reset_keys
        result = []

        def _root(key):
            if isinstance(key, str):
                return ()
            return key[:-1]

        roots = []
        for reset_key in reset_keys:
            cur_root = _root(reset_key)
            for root in roots:
                if cur_root[: len(root)] == root:
                    break
            else:
                roots.append(cur_root)
                result.append(reset_key)
        return result

    @property
    @_cache_value
    def done_keys_groups(self):
        """A list of done keys, grouped as the reset keys.

        This is a list of lists. The outer list has the length of reset keys, the
        inner lists contain the done keys (eg, done and truncated) that can
        be read to determine a reset when it is absent.
        """
        # done keys, sorted as reset keys
        done_keys_group = []
        roots = set()
        fds = self.full_done_spec
        for done_key in self.done_keys:
            root_name = done_key[:-1] if isinstance(done_key, tuple) else ()
            root = fds[root_name] if root_name else fds
            n = len(roots)
            roots.add(root_name)
            if len(roots) - n:
                done_keys_group.append(
                    [
                        unravel_key(root_name + (key,))
                        for key in root.keys(include_nested=False, leaves_only=True)
                    ]
                )
        return done_keys_group

    def _select_observation_keys(self, tensordict: TensorDictBase) -> Iterator[str]:
        for key in tensordict.keys():
            if key.rfind("observation") >= 0:
                yield key

    def close(self, *, raise_if_closed: bool = True):
        self.is_closed = True

    def __del__(self):
        # if del occurs before env has been set up, we don't want a recursion
        # error
        if "is_closed" in self.__dict__ and not self.is_closed:
            try:
                self.close()
            except Exception:
                # a TypeError will typically be raised if the env is deleted when the program ends.
                # In the future, insignificant changes to the close method may change the error type.
                # We excplicitely assume that any error raised during closure in
                # __del__ will not affect the program.
                pass

    @_maybe_unlock
    def to(self, device: DEVICE_TYPING) -> EnvBase:
        device = _make_ordinal_device(torch.device(device))
        if device == self.device:
            return self
        self.__dict__["_input_spec"] = self.input_spec.to(device)
        self.__dict__["_output_spec"] = self.output_spec.to(device)
        self._device = device
        return super().to(device)

    def fake_tensordict(self) -> TensorDictBase:
        """Returns a fake tensordict with key-value pairs that match in shape, device and dtype what can be expected during an environment rollout."""
        state_spec = self.state_spec
        observation_spec = self.observation_spec
        action_spec = self.input_spec["full_action_spec"]
        # instantiates reward_spec if needed
        _ = self.full_reward_spec
        reward_spec = self.output_spec["full_reward_spec"]
        full_done_spec = self.output_spec["full_done_spec"]

        fake_obs = observation_spec.zero()
        fake_reward = reward_spec.zero()
        fake_done = full_done_spec.zero()

        fake_state = state_spec.zero()
        fake_action = action_spec.zero()

        if any(
            isinstance(val, LazyStackedTensorDict) for val in fake_action.values(True)
        ):
            fake_input = fake_action.update(fake_state)
        else:
            fake_input = fake_state.update(fake_action)

        # the input and output key may match, but the output prevails
        # Hence we generate the input, and override using the output
        fake_in_out = fake_input.update(fake_obs)

        next_output = fake_obs.clone()
        next_output.update(fake_reward)
        next_output.update(fake_done)
        fake_in_out.update(fake_done.clone())
        if "next" not in fake_in_out.keys():
            fake_in_out.set("next", next_output)
        else:
            fake_in_out.get("next").update(next_output)

        fake_in_out.batch_size = self.batch_size
        fake_in_out = fake_in_out.to(self.device)
        return fake_in_out


class _EnvWrapper(EnvBase):
    """Abstract environment wrapper class.

    Unlike EnvBase, _EnvWrapper comes with a :obj:`_build_env` private method that will be called upon instantiation.
    Interfaces with other libraries should be coded using _EnvWrapper.

    It is possible to directly query attributed from the nested environment it its name does not conflict with
    an attribute of the wrapper:
        >>> env = SomeWrapper(...)
        >>> custom_attribute0 = env._env.custom_attribute
        >>> custom_attribute1 = env.custom_attribute
        >>> assert custom_attribute0 is custom_attribute1  # should return True

    """

    git_url: str = ""
    available_envs: dict[str, Any] = {}
    libname: str = ""

    def __init__(
        self,
        *args,
        device: DEVICE_TYPING = None,
        batch_size: torch.Size | None = None,
        allow_done_after_reset: bool = False,
        spec_locked: bool = True,
        **kwargs,
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            allow_done_after_reset=allow_done_after_reset,
            spec_locked=spec_locked,
        )
        if len(args):
            raise ValueError(
                "`_EnvWrapper.__init__` received a non-empty args list of arguments. "
                "Make sure only keywords arguments are used when calling `super().__init__`."
            )

        frame_skip = kwargs.get("frame_skip", 1)
        if "frame_skip" in kwargs:
            del kwargs["frame_skip"]
        self.frame_skip = frame_skip
        # this value can be changed if frame_skip is passed during env construction
        self.wrapper_frame_skip = frame_skip

        self._constructor_kwargs = kwargs
        self._check_kwargs(kwargs)
        self._convert_actions_to_numpy = kwargs.pop("convert_actions_to_numpy", True)
        self._env = self._build_env(**kwargs)  # writes the self._env attribute
        self._make_specs(self._env)  # writes the self._env attribute
        self.is_closed = False
        self._init_env()  # runs all the steps to have a ready-to-use env

    def _sync_device(self):
        sync_func = self.__dict__.get("_sync_device_val")
        if sync_func is None:
            device = self.device
            if device.type != "cuda":
                if torch.cuda.is_available():
                    self._sync_device_val = torch.cuda.synchronize
                elif torch.backends.mps.is_available():
                    self._sync_device_val = torch.cuda.synchronize
                elif device.type == "cpu":
                    self._sync_device_val = _do_nothing
            else:
                self._sync_device_val = _do_nothing
            return self._sync_device
        return sync_func

    @abc.abstractmethod
    def _check_kwargs(self, kwargs: dict):
        raise NotImplementedError

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(
                attr
            )  # make sure that appropriate exceptions are raised

        elif attr.startswith("__"):
            raise AttributeError(
                "passing built-in private methods is "
                f"not permitted with type {type(self)}. "
                f"Got attribute {attr}."
            )

        elif "_env" in self.__dir__():
            env = self.__getattribute__("_env")
            return getattr(env, attr)
        super().__getattr__(attr)

        raise AttributeError(
            f"env not set in {self.__class__.__name__}, cannot access {attr}"
        )

    @abc.abstractmethod
    def _init_env(self) -> int | None:
        """Runs all the necessary steps such that the environment is ready to use.

        This step is intended to ensure that a seed is provided to the environment (if needed) and that the environment
        is reset (if needed). For instance, DMControl envs require the env to be reset before being used, but Gym envs
        don't.

        Returns:
            the resulting seed

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_env(self, **kwargs) -> gym.Env:  # noqa: F821
        """Creates an environment from the target library and stores it with the `_env` attribute.

        When overwritten, this function should pass all the required kwargs to the env instantiation method.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _make_specs(self, env: gym.Env) -> None:  # noqa: F821
        raise NotImplementedError

    def close(self, *, raise_if_closed: bool = True) -> None:
        """Closes the contained environment if possible."""
        self.is_closed = True
        try:
            self._env.close()
        except AttributeError:
            pass


def make_tensordict(
    env: _EnvWrapper,
    policy: Callable[[TensorDictBase, ...], TensorDictBase] | None = None,
) -> TensorDictBase:
    """Returns a zeroed-tensordict with fields matching those required for a full step (action selection and environment step) in the environment.

    Args:
        env (_EnvWrapper): environment defining the observation, action and reward space;
        policy (Callable, optional): policy corresponding to the environment.

    """
    with torch.no_grad():
        tensordict = env.reset()
        if policy is not None:
            tensordict.update(policy(tensordict))
        else:
            tensordict.set("action", env.action_spec.rand(), inplace=False)
        tensordict = env.step(tensordict)
        return tensordict.zero_()


def _get_sync_func(policy_device, env_device):
    if torch.cuda.is_available():
        # Look for a specific device
        if policy_device is not None and policy_device.type == "cuda":
            if env_device is None or env_device.type == "cuda":
                return torch.cuda.synchronize
            return partial(torch.cuda.synchronize, device=policy_device)
        if env_device is not None and env_device.type == "cuda":
            if policy_device is None:
                return torch.cuda.synchronize
            return partial(torch.cuda.synchronize, device=env_device)
        return torch.cuda.synchronize
    if torch.backends.mps.is_available():
        return torch.mps.synchronize
    return _do_nothing


def _do_nothing():
    return


def _has_dynamic_specs(spec: Composite):
    from tensordict.base import _NESTED_TENSORS_AS_LISTS

    return any(
        any(s == -1 for s in spec.shape)
        for spec in spec.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
    )


def _tensor_to_spec(name, leaf, leaf_compare=None, *, stack):
    if not (isinstance(leaf, torch.Tensor) or is_tensor_collection(leaf)):
        stack[name] = NonTensor(shape=())
        return
    elif is_non_tensor(leaf):
        stack[name] = NonTensor(shape=leaf.shape)
        return
    shape = leaf.shape
    if leaf_compare is not None:
        shape_compare = leaf_compare.shape
        shape = [s0 if s0 == s1 else -1 for s0, s1 in zip(shape, shape_compare)]
    stack[name] = Unbounded(shape, device=leaf.device, dtype=leaf.dtype)
