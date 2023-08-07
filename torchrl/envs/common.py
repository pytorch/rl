# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tensordict.tensordict import TensorDictBase

from torchrl._utils import prod, seed_generator

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.utils import get_available_libraries, step_mdp

LIBRARIES = get_available_libraries()


def _tensor_to_np(t):
    return t.detach().cpu().numpy()


dtype_map = {
    torch.float: np.float32,
    torch.double: np.float64,
    torch.bool: bool,
}


class EnvMetaData:
    """A class for environment meta-data storage and passing in multiprocessed settings."""

    def __init__(
        self,
        tensordict: TensorDictBase,
        specs: CompositeSpec,
        batch_size: torch.Size,
        env_str: str,
        device: torch.device,
        batch_locked: bool = True,
    ):
        self.device = device
        self.tensordict = tensordict
        self.specs = specs
        self.batch_size = batch_size
        self.env_str = env_str
        self.batch_locked = batch_locked

    @property
    def tensordict(self):
        return self._tensordict.to(self.device)

    @property
    def specs(self):
        return self._specs.to(self.device)

    @tensordict.setter
    def tensordict(self, value: TensorDictBase):
        self._tensordict = value.to("cpu")

    @specs.setter
    def specs(self, value: CompositeSpec):
        self._specs = value.to("cpu")

    @staticmethod
    def metadata_from_env(env) -> EnvMetaData:
        tensordict = env.fake_tensordict().clone()
        tensordict.set("_reset", torch.zeros_like(tensordict.get(env.done_key)))

        specs = env.specs.to("cpu")

        batch_size = env.batch_size
        env_str = str(env)
        device = env.device
        specs = specs.to("cpu")
        batch_locked = env.batch_locked
        return EnvMetaData(tensordict, specs, batch_size, env_str, device, batch_locked)

    def expand(self, *size: int) -> EnvMetaData:
        tensordict = self.tensordict.expand(*size).to_tensordict()
        batch_size = torch.Size(list(size))
        return EnvMetaData(
            tensordict,
            self.specs.expand(*size),
            batch_size,
            self.env_str,
            self.device,
            self.batch_locked,
        )

    def clone(self):
        return EnvMetaData(
            self.tensordict.clone(),
            self.specs.clone(),
            torch.Size([*self.batch_size]),
            deepcopy(self.env_str),
            self.device,
            self.batch_locked,
        )

    def to(self, device: DEVICE_TYPING) -> EnvMetaData:
        tensordict = self.tensordict.contiguous().to(device)
        specs = self.specs.to(device)
        return EnvMetaData(
            tensordict, specs, self.batch_size, self.env_str, device, self.batch_locked
        )


class EnvBase(nn.Module, metaclass=abc.ABCMeta):
    """Abstract environment parent class.

    Properties:
        observation_spec (CompositeSpec): sampling spec of the observations. Must be a
            :class:`torchrl.data.CompositeSpec` instance. The keys listed in the
            spec are directly accessible after reset.
            In TorchRL, even though they are not properly speaking "observations"
            all info, states, results of transforms etc. are stored in the
            observation_spec. Therefore, "observation_spec" should be thought as
            a generic data container for environment outputs that are not done
            or reward data.
        reward_spec (TensorSpec): the (leaf) spec of the reward. If the reward
            is nested within a tensordict, its location can be accessed via
            the ``reward_key`` attribute:

                >>> # accessing reward spec:
                >>> reward_spec = env.reward_spec
                >>> reward_spec = env.output_spec['_reward_spec'][env.reward_key]
                >>> # accessing reward:
                >>> reward = env.fake_tensordict()[('next', *env.reward_key)]

        done_spec (TensorSpec): the (leaf) spec of the done. If the done
            is nested within a tensordict, its location can be accessed via
            the ``done_key`` attribute.

                >>> # accessing done spec:
                >>> done_spec = env.done_spec
                >>> done_spec = env.output_spec['_done_spec'][env.done_key]
                >>> # accessing done:
                >>> done = env.fake_tensordict()[('next', *env.done_key)]

        action_spec (TensorSpec): the ampling spec of the actions. This attribute
            is contained in input_spec.

                >>> # accessing action spec:
                >>> action_spec = env.action_spec
                >>> action_spec = env.input_spec['_action_spec'][env.action_key]
                >>> # accessing action:
                >>> action = env.fake_tensordict()[env.action_key]

        output_spec (CompositeSpec): The container for all output specs (reward,
            done and observation).
        input_spec (CompositeSpec): the container for all input specs (actions
            and possibly others).
        batch_size (torch.Size): number of environments contained in the instance;
        device (torch.device): device where the env input and output are expected to live
        run_type_checks (bool): if ``True``, the observation and reward dtypes
            will be compared against their respective spec and an exception
            will be raised if they don't match.
            Defaults to False.

    .. note::
      The usage of ``done_key``, ``reward_key`` and ``action_key`` is aimed at
      facilitating the custom placement of done, reward and action data within
      the tensordict structures produced and read by the environment.
      In most cases, these attributes can be ignored and the default values
      (``"done"``, ``"reward"`` and ``"action"``) can be used.

    Methods:
        step (TensorDictBase -> TensorDictBase): step in the environment
        reset (TensorDictBase, optional -> TensorDictBase): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (TensorDictBase, optional -> TensorDictBase): random step given the action spec
        rollout (Callable, ... -> TensorDictBase): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = GymEnv("Pendulum-v1")
        >>> env.batch_size  # how many envs are run at once
        torch.Size([])
        >>> env.input_spec
        CompositeSpec(
            action: BoundedTensorSpec(
                shape=torch.Size([1]),
                space=ContinuousBox(
                    minimum=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                    maximum=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous), device=cpu, shape=torch.Size([]))
        >>> env.action_spec
        BoundedTensorSpec(
            shape=torch.Size([1]),
            space=ContinuousBox(
                minimum=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                maximum=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> env.observation_spec
        CompositeSpec(
            observation: BoundedTensorSpec(
                shape=torch.Size([3]),
                space=ContinuousBox(
                    minimum=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                    maximum=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous), device=cpu, shape=torch.Size([]))
        >>> env.reward_spec
        UnboundedContinuousTensorSpec(
            shape=torch.Size([1]),
            space=None,
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> env.done_spec
        DiscreteTensorSpec(
            shape=torch.Size([1]),
            space=DiscreteBox(n=2),
            device=cpu,
            dtype=torch.bool,
            domain=discrete)
        >>> # the output_spec contains all the expected outputs
        >>> env.output_spec
        CompositeSpec(
            observation: CompositeSpec(
                observation: BoundedTensorSpec(
                    shape=torch.Size([3]),
                    space=ContinuousBox(
                        minimum=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                        maximum=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([])),
            reward: CompositeSpec(
                reward: UnboundedContinuousTensorSpec(
                    shape=torch.Size([1]),
                    space=None,
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([])),
            done: CompositeSpec(
                done: DiscreteTensorSpec(
                    shape=torch.Size([1]),
                    space=DiscreteBox(n=2),
                    device=cpu,
                    dtype=torch.bool,
                    domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))

    """

    def __init__(
        self,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
        run_type_checks: bool = False,
    ):
        self.__dict__["_done_key"] = None
        self.__dict__["_reward_key"] = None
        self.__dict__["_action_key"] = None
        if device is not None:
            self.__dict__["_device"] = torch.device(device)
            output_spec = self.__dict__.get("_output_spec", None)
            if output_spec is not None:
                self.__dict__["_output_spec"] = output_spec.to(self.device)
            input_spec = self.__dict__.get("_input_spec", None)
            if input_spec is not None:
                self.__dict__["_input_spec"] = input_spec.to(self.device)

        super().__init__()
        self.dtype = dtype_map.get(dtype, dtype)
        if "is_closed" not in self.__dir__():
            self.is_closed = True
        if batch_size is not None:
            # we want an error to be raised if we pass batch_size but
            # it's already been set
            self.batch_size = torch.Size(batch_size)
        self._run_type_checks = run_type_checks

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
        if key in ("_input_spec", "_observation_spec", "_action_spec", "_reward_spec"):
            raise AttributeError(
                "To set an environment spec, please use `env.observation_spec = obs_spec` (without the leading"
                " underscore)."
            )
        return super().__setattr__(key, value)

    @property
    def batch_locked(self) -> bool:
        """Whether the environnement can be used with a batch size different from the one it was initialized with or not.

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
        _batch_size = getattr(self, "_batch_size", None)
        if _batch_size is None:
            _batch_size = self._batch_size = torch.Size([])
        return _batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        self._batch_size = torch.Size(value)
        if (
            hasattr(self, "output_spec")
            and self.output_spec.shape[: len(value)] != value
        ):
            self.output_spec.unlock_()
            self.output_spec.shape = value
            self.output_spec.lock_()
        if hasattr(self, "input_spec") and self.input_spec.shape[: len(value)] != value:
            self.input_spec.unlock_()
            self.input_spec.shape = value
            self.input_spec.lock_()

    @property
    def device(self) -> torch.device:
        device = self.__dict__.get("_device", None)
        if device is None:
            device = self.__dict__["_device"] = torch.device("cpu")
        return device

    @device.setter
    def device(self, value: torch.device) -> None:
        device = self.__dict__.get("_device", None)
        if device is None:
            self.__dict__["_device"] = value
            return
        raise RuntimeError("device cannot be set. Call env.to(device) instead.")

    def ndimension(self):
        return len(self.batch_size)

    @property
    def ndim(self):
        return self.ndimension()

    # Parent specs: input and output spec.
    @property
    def input_spec(self) -> TensorSpec:
        input_spec = self.__dict__.get("_input_spec", None)
        if input_spec is None:
            input_spec = CompositeSpec(
                _state_spec=None,
                shape=self.batch_size,
                device=self.device,
            ).lock_()
            self.__dict__["_input_spec"] = input_spec
        return input_spec

    @input_spec.setter
    def input_spec(self, value: TensorSpec) -> None:
        raise RuntimeError("input_spec is protected.")

    @property
    def output_spec(self) -> TensorSpec:
        output_spec = self.__dict__.get("_output_spec", None)
        if output_spec is None:
            output_spec = CompositeSpec(
                shape=self.batch_size,
                device=self.device,
            ).lock_()
            self.__dict__["_output_spec"] = output_spec
        return output_spec

    @output_spec.setter
    def output_spec(self, value: TensorSpec) -> None:
        raise RuntimeError("output_spec is protected.")

    # Action spec
    def _get_action_key(self):
        keys = self.input_spec["_action_spec"].keys(True, True)
        for key in keys:
            # the first key is the action
            if not isinstance(key, tuple):
                key = (key,)
            break
        else:
            raise AttributeError("Could not find action spec")
        self.__dict__["_action_key"] = key
        return key

    @property
    def action_key(self):
        """The action key of an environment.

        By default, non-nested keys are stored in the 'action' key.

        If the action is in a nested tensordict, this property will return its
        location.
        """
        out = self._action_key
        if out is None:
            out = self._get_action_key()
        return out

    # Action spec: action specs belong to input_spec
    @property
    def action_spec(self) -> TensorSpec:
        """The ``action`` leaf spec.

        This property will always return the leaf spec of the action attribute,
        which can be accessed in a typical rollout via

            >>> fake_td = env.fake_tensordict()  # a typical tensordict
            >>> action = fake_td[env.action_key]

        This property is mutable.
        """
        try:
            action_spec = self.input_spec["_action_spec"]
        except (KeyError, AttributeError):
            raise KeyError("Failed to find the action_spec.")
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
    def action_spec(self, value: TensorSpec) -> None:
        try:
            self.input_spec.unlock_()
            device = self.input_spec.device
            try:
                delattr(self, "_action_key")
            except AttributeError:
                pass

            if isinstance(value, CompositeSpec):
                for _ in value.values(True, True):  # noqa: B007
                    break
                else:
                    raise RuntimeError(
                        "An empty CompositeSpec was passed for the action spec. "
                        "This is currently not permitted."
                    )
            else:
                value = CompositeSpec(
                    action=value.to(device), shape=self.batch_size, device=device
                )

            self.input_spec["_action_spec"] = value.to(device)
            self._get_action_key()
        finally:
            self.input_spec.lock_()

    # Reward spec
    def _get_reward_key(self):
        keys = self.output_spec["_reward_spec"].keys(True, True)
        for key in keys:
            # the first key is the reward
            if not isinstance(key, tuple):
                key = (key,)
            break
        else:
            raise AttributeError("Could not find reward spec")
        self.__dict__["_reward_key"] = key
        return key

    @property
    def reward_key(self):
        """The reward key of an environment.

        By default, non-nested keys are stored in the ``'reward'`` entry.

        If the reward is in a nested tensordict, this property will return its
        location.
        """
        out = self._reward_key
        if out is None:
            out = self._get_reward_key()
        return out

    # Done spec: reward specs belong to output_spec
    @property
    def reward_spec(self) -> TensorSpec:
        """The ``reward`` leaf spec.

        This property will always return the leaf spec of the reward attribute,
        which can be accessed in a typical rollout via

            >>> fake_td = env.fake_tensordict()  # a typical tensordict
            >>> reward = fake_td[("next", *env.reward_key)]

        This property is mutable.
        """
        try:
            reward_spec = self.output_spec["_reward_spec"]
        except (KeyError, AttributeError):
            # populate the "reward" entry
            # this will be raised if there is not _reward_spec (unlikely) or no reward_key
            # Since output_spec is lazily populated with an empty composite spec for
            # reward_spec, the second case is much more likely to occur.
            self.reward_spec = out = UnboundedContinuousTensorSpec(
                shape=(*self.batch_size, 1),
                device=self.device,
            )
            reward_spec = self.output_spec["_reward_spec"]
        finally:
            try:
                out = reward_spec[self.reward_key]
            except KeyError:
                # the key may have changed
                raise KeyError(
                    "The reward_key attribute seems to have changed. "
                    "This occurs when a reward_spec is updated without "
                    "calling `env.reward_spec = new_spec`. "
                    "Make sure you rely on this  type of command "
                    "to set the reward and other specs."
                )

        return out

    @reward_spec.setter
    def reward_spec(self, value: TensorSpec) -> None:
        try:
            self.output_spec.unlock_()
            device = self.output_spec.device
            try:
                delattr(self, "_reward_key")
            except AttributeError:
                pass
            if not hasattr(value, "shape"):
                raise TypeError(
                    f"reward_spec of type {type(value)} do not have a shape "
                    f"attribute."
                )
            if value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    "The value of spec.shape must match the env batch size."
                )
            if isinstance(value, CompositeSpec):
                for nestedval in value.values(True, True):  # noqa: B007
                    break
                else:
                    raise RuntimeError(
                        "An empty CompositeSpec was passed for the reward spec. "
                        "This is currently not permitted."
                    )
            else:
                nestedval = value
                value = CompositeSpec(
                    reward=value.to(device), shape=self.batch_size, device=device
                )
            if len(nestedval.shape) == 0:
                raise RuntimeError(
                    "the reward_spec shape cannot be empty (this error"
                    " usually comes from trying to set a reward_spec"
                    " with a null number of dimensions. Try using a multidimensional"
                    " spec instead, for instance with a singleton dimension at the tail)."
                )
            self.output_spec["_reward_spec"] = value.to(device)
            self._get_reward_key()
        finally:
            self.output_spec.lock_()

    # done spec
    def _get_done_key(self):
        keys = self.output_spec["_done_spec"].keys(True, True)
        for key in keys:
            # the first key is the reward
            if not isinstance(key, tuple):
                key = (key,)
            break
        else:
            raise AttributeError(
                f"Could not find done spec: {self.output_spec['_done_spec']}"
            )
        self.__dict__["_done_key"] = key
        return key

    @property
    def done_key(self):
        """The done key of an environment.

        By default, non-nested keys are stored in the ``'done'`` entry.

        If the done is in a nested tensordict, this property will return its
        location.
        """
        out = self._done_key
        if out is None:
            out = self._get_done_key()
        return out

    # Done spec: done specs belong to output_spec
    @property
    def done_spec(self) -> TensorSpec:
        """The ``done`` leaf spec.

        This property will always return the leaf spec of the done attribute,
        which can be accessed in a typical rollout via

            >>> fake_td = env.fake_tensordict()  # a typical tensordict
            >>> done = fake_td[("next", *env.done_key)]

        This property is mutable.
        """
        try:
            done_spec = self.output_spec["_done_spec"]
        except (KeyError, AttributeError):
            # populate the "done" entry
            # this will be raised if there is not _done_spec (unlikely) or no done_key
            # Since output_spec is lazily populated with an empty composite spec for
            # done_spec, the second case is much more likely to occur.
            self.done_spec = DiscreteTensorSpec(
                n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device
            )
            done_spec = self.output_spec["_done_spec"]
        finally:
            try:
                out = done_spec[self.done_key]
            except KeyError:
                # the key may have changed
                raise KeyError(
                    "The done_key attribute seems to have changed. "
                    "This occurs when a done_spec is updated without "
                    "calling `env.done_spec = new_spec`. "
                    "Make sure you rely on this  type of command "
                    "to set the done and other specs."
                )

        return out

    @done_spec.setter
    def done_spec(self, value: TensorSpec) -> None:
        try:
            self.output_spec.unlock_()
            device = self.output_spec.device
            try:
                delattr(self, "_done_key")
            except AttributeError:
                pass
            if not hasattr(value, "shape"):
                raise TypeError(
                    f"done_spec of type {type(value)} do not have a shape "
                    f"attribute."
                )
            if value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    "The value of spec.shape must match the env batch size."
                )
            if isinstance(value, CompositeSpec):
                for nestedval in value.values(True, True):  # noqa: B007
                    break
                else:
                    raise RuntimeError(
                        "An empty CompositeSpec was passed for the done spec. "
                        "This is currently not permitted."
                    )
            else:
                nestedval = value
                value = CompositeSpec(
                    done=value.to(device), shape=self.batch_size, device=device
                )
            if len(nestedval.shape) == 0:
                raise RuntimeError(
                    "the done_spec shape cannot be empty (this error"
                    " usually comes from trying to set a done_spec"
                    " with a null number of dimensions. Try using a multidimensional"
                    " spec instead, for instance with a singleton dimension at the tail)."
                )
            if len(list(value.keys())) == 0:
                raise RuntimeError
            self.output_spec["_done_spec"] = value.to(device)
            self._get_done_key()
        finally:
            self.output_spec.lock_()

    # observation spec: observation specs belong to output_spec
    @property
    def observation_spec(self) -> CompositeSpec:
        observation_spec = self.output_spec["_observation_spec"]
        if observation_spec is None:
            observation_spec = CompositeSpec(shape=self.batch_size, device=self.device)
            self.output_spec.unlock_()
            self.output_spec["_observation_spec"] = observation_spec
            self.output_spec.lock_()
        return observation_spec

    @observation_spec.setter
    def observation_spec(self, value: TensorSpec) -> None:
        try:
            self.output_spec.unlock_()
            device = self.output_spec.device
            if not isinstance(value, CompositeSpec):
                raise TypeError("The type of an observation_spec must be Composite.")
            elif value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                )
            if value.shape[: len(self.batch_size)] != self.batch_size:
                raise ValueError(
                    f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                )
            self.output_spec["_observation_spec"] = value.to(device)
        finally:
            self.output_spec.lock_()

    # state spec: state specs belong to input_spec
    @property
    def state_spec(self) -> CompositeSpec:
        state_spec = self.input_spec["_state_spec"]
        if state_spec is None:
            state_spec = CompositeSpec(shape=self.batch_size, device=self.device)
            self.input_spec.unlock_()
            self.input_spec["_state_spec"] = state_spec
            self.input_spec.lock_()
        return state_spec

    @state_spec.setter
    def state_spec(self, value: CompositeSpec) -> None:
        try:
            self.input_spec.unlock_()
            if value is None:
                self.input_spec["_state_spec"] = CompositeSpec(
                    device=self.device, shape=self.batch_size
                )
            else:
                device = self.input_spec.device
                if not isinstance(value, CompositeSpec):
                    raise TypeError("The type of an state_spec must be Composite.")
                elif value.shape[: len(self.batch_size)] != self.batch_size:
                    raise ValueError(
                        f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                    )
                if value.shape[: len(self.batch_size)] != self.batch_size:
                    raise ValueError(
                        f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
                    )
                self.input_spec["_state_spec"] = value.to(device)
        finally:
            self.input_spec.lock_()

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Makes a step in the environment.

        Step accepts a single argument, tensordict, which usually carries an 'action' key which indicates the action
        to be taken.
        Step will call an out-place private method, _step, which is the method to be re-written by EnvBase subclasses.

        Args:
            tensordict (TensorDictBase): Tensordict containing the action to be taken.

        Returns:
            the input tensordict, modified in place with the resulting observations, done state and reward
            (+ others if needed).

        """
        # sanity check
        self._assert_tensordict_shape(tensordict)

        tensordict_out = self._step(tensordict)
        # this tensordict should contain a "next" key
        try:
            next_tensordict_out = tensordict_out.get("next")
        except KeyError:
            raise RuntimeError(
                "The value returned by env._step must be a tensordict where the "
                "values at t+1 have been written under a 'next' entry. This "
                f"tensordict couldn't be found in the output, got: {tensordict_out}."
            )
        if tensordict_out is tensordict:
            raise RuntimeError(
                "EnvBase._step should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _step before writing new tensors onto this new instance."
            )

        # TODO: Refactor this using reward spec
        reward = next_tensordict_out.get(self.reward_key)
        # unsqueeze rewards if needed
        # the input tensordict may have more leading dimensions than the batch_size
        # e.g. in model-based contexts.
        batch_size = self.batch_size
        dims = len(batch_size)
        leading_batch_size = (
            next_tensordict_out.batch_size[:-dims]
            if dims
            else next_tensordict_out.shape
        )
        expected_reward_shape = torch.Size(
            [*leading_batch_size, *self.reward_spec.shape]
        )
        actual_reward_shape = reward.shape
        if actual_reward_shape != expected_reward_shape:
            reward = reward.view(expected_reward_shape)
            next_tensordict_out.set(self.reward_key, reward)

        # TODO: Refactor this using done spec
        done = next_tensordict_out.get(self.done_key)
        # unsqueeze done if needed
        expected_done_shape = torch.Size([*leading_batch_size, *self.done_spec.shape])
        actual_done_shape = done.shape
        if actual_done_shape != expected_done_shape:
            done = done.view(expected_done_shape)
            next_tensordict_out.set(self.done_key, done)
        tensordict_out.set("next", next_tensordict_out)

        if self.run_type_checks:
            for key in self._select_observation_keys(tensordict_out):
                obs = tensordict_out.get(key)
                self.observation_spec.type_check(obs, key)

            if (
                next_tensordict_out.get(self.reward_key).dtype
                is not self.reward_spec.dtype
            ):
                raise TypeError(
                    f"expected reward.dtype to be {self.reward_spec.dtype} "
                    f"but got {tensordict_out.get(self.reward_key).dtype}"
                )

            if next_tensordict_out.get(self.done_key).dtype is not self.done_spec.dtype:
                raise TypeError(
                    f"expected done.dtype to be torch.bool but got {tensordict_out.get(self.done_key).dtype}"
                )
        # tensordict could already have a "next" key
        tensordict.update(tensordict_out)

        return tensordict

    def _get_in_keys_to_exclude(self, tensordict):
        if self._cache_in_keys is None:
            self._cache_in_keys = list(
                set(self.input_spec.keys(True)).intersection(
                    tensordict.keys(True, True)
                )
            )
        return self._cache_in_keys

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError("EnvBase.forward is not implemented")

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
        tensordict: Optional[TensorDictBase] = None,
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

        """
        if tensordict is not None and "_reset" in tensordict.keys():
            self._assert_tensordict_shape(tensordict)
            _reset = tensordict.get("_reset")
            if _reset.shape[-len(self.done_spec.shape) :] != self.done_spec.shape:
                raise RuntimeError(
                    "_reset flag in tensordict should follow env.done_spec"
                )
        else:
            _reset = None

        tensordict_reset = self._reset(tensordict, **kwargs)
        if tensordict_reset.device != self.device:
            tensordict_reset = tensordict_reset.to(self.device)
        if tensordict_reset is tensordict:
            raise RuntimeError(
                "EnvBase._reset should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _reset before writing new tensors onto this new instance."
            )
        if not isinstance(tensordict_reset, TensorDictBase):
            raise RuntimeError(
                f"env._reset returned an object of type {type(tensordict_reset)} but a TensorDict was expected."
            )

        if len(self.batch_size):
            leading_dim = tensordict_reset.shape[: -len(self.batch_size)]
        else:
            leading_dim = tensordict_reset.shape
        if self.done_spec is not None and self.done_key not in tensordict_reset.keys(
            True, True
        ):
            tensordict_reset.set(
                self.done_key,
                self.done_spec.zero(leading_dim),
            )

        if (_reset is None and tensordict_reset.get(self.done_key).any()) or (
            _reset is not None and tensordict_reset.get(self.done_key)[_reset].any()
        ):
            raise RuntimeError(
                f"Env {self} was done after reset on specified '_reset' dimensions. This is (currently) not allowed."
            )
        if tensordict is not None:
            tensordict.update(tensordict_reset)
        else:
            tensordict = tensordict_reset
        return tensordict

    def numel(self) -> int:
        return prod(self.batch_size)

    def set_seed(
        self, seed: Optional[int] = None, static_seed: bool = False
    ) -> Optional[int]:
        """Sets the seed of the environment and returns the next seed to be used (which is the input seed if a single environment is present).

        Args:
            seed (int): seed to be set
            static_seed (bool, optional): if ``True``, the seed is not incremented.
                Defaults to False

        Returns:
            integer representing the "next seed": i.e. the seed that should be
            used for another environment if created concomittently to this environment.

        """
        if seed is not None:
            torch.manual_seed(seed)
        self._set_seed(seed)
        if seed is not None and not static_seed:
            new_seed = seed_generator(seed)
            seed = new_seed
        return seed

    @abc.abstractmethod
    def _set_seed(self, seed: Optional[int]):
        raise NotImplementedError

    def set_state(self):
        raise NotImplementedError

    def _assert_tensordict_shape(self, tensordict: TensorDictBase) -> None:
        if (
            self.batch_locked or self.batch_size != torch.Size([])
        ) and tensordict.batch_size != self.batch_size:
            raise RuntimeError(
                f"Expected a tensordict with shape==env.shape, "
                f"got {tensordict.batch_size} and {self.batch_size}"
            )

    def rand_action(self, tensordict: Optional[TensorDictBase] = None):
        """Performs a random action given the action_spec attribute.

        Args:
            tensordict (TensorDictBase, optional): tensordict where the resulting action should be written.

        Returns:
            a tensordict object with the "action" entry updated with a random
            sample from the action-spec.

        """
        shape = torch.Size([])
        if not self.batch_locked and not self.batch_size and tensordict is not None:
            shape = tensordict.shape
        elif not self.batch_locked and not self.batch_size:
            shape = torch.Size([])
        elif not self.batch_locked and tensordict.shape != self.batch_size:
            raise RuntimeError(
                "The input tensordict and the env have a different batch size: "
                f"env.batch_size={self.batch_size} and tensordict.batch_size={tensordict.shape}. "
                f"Non batch-locked environment require the env batch-size to be either empty or to"
                f" match the tensordict one."
            )
        r = self.input_spec["_action_spec"].rand(shape)
        if tensordict is None:
            return r
        tensordict.update(r)
        return tensordict

    def rand_step(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
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
    def specs(self) -> CompositeSpec:
        """Returns a Composite container where all the environment are present.

        This feature allows one to create an environment, retrieve all of the specs in a single data container and then
        erase the environment from the workspace.

        """
        return CompositeSpec(
            output_spec=self.output_spec,
            input_spec=self.input_spec,
            shape=self.batch_size,
        ).lock_()

    def rollout(
        self,
        max_steps: int,
        policy: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        callback: Optional[Callable[[TensorDictBase, ...], TensorDictBase]] = None,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool = True,
        return_contiguous: bool = True,
        tensordict: Optional[TensorDictBase] = None,
    ) -> TensorDictBase:
        """Executes a rollout in the environment.

        The function will stop as soon as one of the contained environments
        returns done=True.

        Args:
            max_steps (int): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before max_steps have been executed.
            policy (callable, optional): callable to be called to compute the desired action. If no policy is provided,
                actions will be called using :obj:`env.rand_step()`
                default = None
            callback (callable, optional): function to be called at each iteration with the given TensorDict.
            auto_reset (bool, optional): if ``True``, resets automatically the environment
                if it is in a done state when the rollout is initiated.
                Default is ``True``.
            auto_cast_to_device (bool, optional): if ``True``, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is ``False``.
            break_when_any_done (bool): breaks if any of the done state is True. If False, a reset() is
                called on the sub-envs that are done. Default is True.
            return_contiguous (bool): if False, a LazyStackedTensorDict will be returned. Default is True.
            tensordict (TensorDict, optional): if auto_reset is False, an initial
                tensordict must be provided.

        Returns:
            TensorDict object containing the resulting trajectory.

        The data returned will be marked with a "time" dimension name for the last
        dimension of the tensordict (at the ``env.ndim`` index).

        Examples:
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

        In some instances, contiguous tensordict cannot be obtained because
        they cannot be stacked. This can happen when the data returned at
        each step may have a different shape, or when different environments
        are executed together. In that case, ``return_contiguous=False``
        will cause the returned tensordict to be a lazy stack of tensordicts:

        Examples:
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

        """
        try:
            policy_device = next(policy.parameters()).device
        except (StopIteration, AttributeError):
            policy_device = self.device

        env_device = self.device

        if auto_reset:
            if tensordict is not None:
                raise RuntimeError(
                    "tensordict cannot be provided when auto_reset is True"
                )
            tensordict = self.reset()
        elif tensordict is None:
            raise RuntimeError("tensordict must be provided when auto_reset is False")
        if policy is None:

            def policy(td):
                self.rand_action(td)
                return td

        tensordicts = []
        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict = tensordict.to(policy_device, non_blocking=True)
            tensordict = policy(tensordict)
            if auto_cast_to_device:
                tensordict = tensordict.to(env_device, non_blocking=True)
            tensordict = self.step(tensordict)

            tensordicts.append(tensordict.clone(False))
            done = tensordict.get(("next", self.done_key))
            truncated = tensordict.get(
                ("next", "truncated"),
                default=torch.zeros((), device=done.device, dtype=torch.bool),
            )
            done = done | truncated
            if (break_when_any_done and done.any()) or i == max_steps - 1:
                break
            tensordict = step_mdp(
                tensordict,
                keep_other=True,
                exclude_action=True,
                exclude_reward=True,
                reward_key=self.reward_key,
                action_key=self.action_key,
                done_key=self.done_key,
            )
            if not break_when_any_done and done.any():
                _reset = done.clone()
                tensordict.set("_reset", _reset)
                self.reset(tensordict)

            if callback is not None:
                callback(self, tensordict)

        batch_size = self.batch_size if tensordict is None else tensordict.batch_size

        out_td = torch.stack(tensordicts, len(batch_size))
        if return_contiguous:
            out_td = out_td.contiguous()
        out_td.refine_names(..., "time")
        return out_td

    def _select_observation_keys(self, tensordict: TensorDictBase) -> Iterator[str]:
        for key in tensordict.keys():
            if key.rfind("observation") >= 0:
                yield key

    def close(self):
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

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        device = torch.device(device)
        if device == self.device:
            return self
        self.__dict__["_input_spec"] = self.input_spec.to(device).lock_()
        self.__dict__["_output_spec"] = self.output_spec.to(device).lock_()
        self._device = device
        return super().to(device)

    def fake_tensordict(self) -> TensorDictBase:
        """Returns a fake tensordict with key-value pairs that match in shape, device and dtype what can be expected during an environment rollout."""
        state_spec = self.state_spec
        observation_spec = self.observation_spec
        action_spec = self.input_spec["_action_spec"]
        # instantiates reward_spec if needed
        _ = self.reward_spec
        reward_spec = self.output_spec["_reward_spec"]
        # instantiates done_spec if needed
        _ = self.done_spec
        done_spec = self.output_spec["_done_spec"]

        fake_obs = observation_spec.zero()

        fake_state = state_spec.zero()
        fake_action = action_spec.zero()
        fake_input = fake_state.update(fake_action)

        # the input and output key may match, but the output prevails
        # Hence we generate the input, and override using the output
        fake_in_out = fake_input.update(fake_obs)

        fake_reward = reward_spec.zero()
        fake_done = done_spec.zero()

        next_output = fake_obs.clone()
        next_output.update(fake_reward)
        next_output.update(fake_done)
        fake_in_out.update(fake_done.clone())

        fake_td = fake_in_out.set("next", next_output)
        fake_td.batch_size = self.batch_size
        fake_td = fake_td.to(self.device)
        return fake_td


class _EnvWrapper(EnvBase, metaclass=abc.ABCMeta):
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
    available_envs: Dict[str, Any] = {}
    libname: str = ""

    def __init__(
        self,
        *args,
        dtype: Optional[np.dtype] = None,
        device: DEVICE_TYPING = "cpu",
        batch_size: Optional[torch.Size] = None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        if len(args):
            raise ValueError(
                "`_EnvWrapper.__init__` received a non-empty args list of arguments."
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
        self._env = self._build_env(**kwargs)  # writes the self._env attribute
        self._make_specs(self._env)  # writes the self._env attribute
        self.is_closed = False
        self._init_env()  # runs all the steps to have a ready-to-use env

    @abc.abstractmethod
    def _check_kwargs(self, kwargs: Dict):
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
    def _init_env(self) -> Optional[int]:
        """Runs all the necessary steps such that the environment is ready to use.

        This step is intended to ensure that a seed is provided to the environment (if needed) and that the environment
        is reset (if needed). For instance, DMControl envs require the env to be reset before being used, but Gym envs
        don't.

        Returns:
            the resulting seed

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_env(self, **kwargs) -> "gym.Env":  # noqa: F821
        """Creates an environment from the target library and stores it with the `_env` attribute.

        When overwritten, this function should pass all the required kwargs to the env instantiation method.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _make_specs(self, env: "gym.Env") -> None:  # noqa: F821
        raise NotImplementedError

    def close(self) -> None:
        """Closes the contained environment if possible."""
        self.is_closed = True
        try:
            self._env.close()
        except AttributeError:
            pass


def make_tensordict(
    env: _EnvWrapper,
    policy: Optional[Callable[[TensorDictBase, ...], TensorDictBase]] = None,
) -> TensorDictBase:
    """Returns a zeroed-tensordict with fields matching those required for a full step (action selection and environment step) in the environment.

    Args:
        env (_EnvWrapper): environment defining the observation, action and reward space;
        policy (Callable, optional): policy corresponding to the environment.

    """
    with torch.no_grad():
        tensordict = env.reset()
        if policy is not None:
            tensordict = policy(tensordict)
        else:
            tensordict.set("action", env.action_spec.rand(), inplace=False)
        tensordict = env.step(tensordict)
        return tensordict.zero_()
