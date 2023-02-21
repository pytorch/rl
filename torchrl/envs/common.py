# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from copy import deepcopy
from numbers import Number
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, TensorSpec

from .._utils import prod, seed_generator
from ..data.utils import DEVICE_TYPING

from .utils import get_available_libraries, step_mdp

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
    def build_metadata_from_env(env) -> EnvMetaData:
        tensordict = env.fake_tensordict().clone()
        specs = {
            "input_spec": env.input_spec,
            "observation_spec": env.observation_spec,
            "reward_spec": env.reward_spec,
        }
        specs = CompositeSpec(**specs, shape=env.batch_size).to("cpu")

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


class Specs:
    """Container for action, observation and reward specs.

    This class allows one to create an environment, retrieve all of the specs
    in a single data container (and access them in one place) before erasing
    the environment from the workspace.

    Args:
        env (EnvBase): environment from which the specs have to be read.

    """

    _keys = {
        "action_spec",
        "observation_spec",
        "reward_spec",
        "input_spec",
        "from_pixels",
    }

    def __init__(self, env: EnvBase):
        self.env = env

    def __getitem__(self, item: str) -> Any:
        if item not in self._keys:
            raise KeyError(f"item must be one of {self._keys}")
        return getattr(self.env, item)

    def keys(self) -> Sequence[str]:
        return self._keys

    def build_tensordict(
        self, next_observation: bool = True, log_prob: bool = False
    ) -> TensorDictBase:
        """Returns a TensorDict with empty tensors of the desired shape.

        Args:
            next_observation (bool, optional): if False, the observation returned
                will be of the current step only (no :obj:`"next"` nested tensordict will be present).
                Default is True.
            log_prob (bool, optional): If True, a log_prob key-value pair will be added
                to the tensordict.

        Returns: A tensordict populated according to the env specs.

        """
        # build a tensordict from specs
        td = TensorDict({}, batch_size=torch.Size([]), _run_checks=False)
        action_placeholder = torch.zeros(
            self["action_spec"].shape, dtype=self["action_spec"].dtype
        )
        if not isinstance(self["observation_spec"], CompositeSpec):
            raise RuntimeError("observation_spec is expected to be of Composite type.")
        else:
            for (key, item) in self["observation_spec"].items():
                observation_placeholder = torch.zeros(item.shape, dtype=item.dtype)
                if next_observation:
                    td.update({"next": {key: observation_placeholder}})
                td.set(
                    key,
                    observation_placeholder.clone(),
                )

        reward_placeholder = torch.zeros(
            self["reward_spec"].shape, dtype=self["reward_spec"].dtype
        )
        done_placeholder = torch.zeros_like(reward_placeholder, dtype=torch.bool)

        td.set("action", action_placeholder)
        td.set("reward", reward_placeholder)

        if log_prob:
            td.set(
                "log_prob",
                torch.zeros_like(reward_placeholder, dtype=torch.float32),
            )  # we assume log_prob to be of type float32
        td.set("done", done_placeholder)
        return td


class EnvBase(nn.Module, metaclass=abc.ABCMeta):
    """Abstract environment parent class.

    Properties:
        - observation_spec (CompositeSpec): sampling spec of the observations;
        - action_spec (TensorSpec): sampling spec of the actions;
        - input_spec (CompositeSpec): sampling spec of the actions and/or other inputs;
        - reward_spec (TensorSpec): sampling spec of the rewards;
        - batch_size (torch.Size): number of environments contained in the instance;
        - device (torch.device): device where the env input and output are expected to live
        - run_type_checks (bool): if True, the observation and reward dtypes
            will be compared against their respective spec and an exception
            will be raised if they don't match.
            Defaults to False.

    Methods:
        step (TensorDictBase -> TensorDictBase): step in the environment
        reset (TensorDictBase, optional -> TensorDictBase): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (TensorDictBase, optional -> TensorDictBase): random step given the action spec
        rollout (Callable, ... -> TensorDictBase): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)

    """

    def __init__(
        self,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
        run_type_checks: bool = False,
    ):
        super().__init__()
        if device is not None:
            self.device = torch.device(device)
        self.dtype = dtype_map.get(dtype, dtype)
        if "is_closed" not in self.__dir__():
            self.is_closed = True
        if "_input_spec" not in self.__dir__():
            self.__dict__["_input_spec"] = None
        if "_reward_spec" not in self.__dir__():
            self.__dict__["_reward_spec"] = None
        if "_observation_spec" not in self.__dir__():
            self.__dict__["_observation_spec"] = None
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
    def batch_size(self) -> TensorSpec:
        if ("_batch_size" not in self.__dir__()) and (
            "_batch_size" not in self.__class__.__dict__
        ):
            self._batch_size = torch.Size([])
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        self._batch_size = torch.Size(value)

    @property
    def action_spec(self) -> TensorSpec:
        return self.input_spec["action"]

    @action_spec.setter
    def action_spec(self, value: TensorSpec) -> None:
        if self._input_spec is None:
            self.input_spec = CompositeSpec(action=value, shape=self.batch_size)
        else:
            self.input_spec["action"] = value

    @property
    def input_spec(self) -> TensorSpec:
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value: TensorSpec) -> None:
        if not isinstance(value, CompositeSpec):
            raise TypeError("The type of an input_spec must be Composite.")
        if value.shape[: len(self.batch_size)] != self.batch_size:
            raise ValueError(
                f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
            )
        self.__dict__["_input_spec"] = value

    @property
    def reward_spec(self) -> TensorSpec:
        return self._reward_spec

    @reward_spec.setter
    def reward_spec(self, value: TensorSpec) -> None:
        if not hasattr(value, "shape"):
            raise TypeError(
                f"reward_spec of type {type(value)} do not have a shape " f"attribute."
            )
        if value.shape[: len(self.batch_size)] != self.batch_size:
            raise ValueError("The value of spec.shape must match the env batch size.")
        if len(value.shape) == 0:
            raise RuntimeError(
                "the reward_spec shape cannot be empty (this error"
                " usually comes from trying to set a reward_spec"
                " with a null number of dimensions. Try using a multidimensional"
                " spec instead, for instance with a singleton dimension at the tail)."
            )
        self.__dict__["_reward_spec"] = value

    @property
    def observation_spec(self) -> TensorSpec:
        return self._observation_spec

    @observation_spec.setter
    def observation_spec(self, value: TensorSpec) -> None:
        if not isinstance(value, CompositeSpec):
            raise TypeError("The type of an observation_spec must be Composite.")
        elif value.shape[: len(self.batch_size)] != self.batch_size:
            raise ValueError(
                f"The value of spec.shape ({value.shape}) must match the env batch size ({self.batch_size})."
            )
        self.__dict__["_observation_spec"] = value

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

        tensordict.lock()  # make sure _step does not modify the tensordict
        tensordict_out = self._step(tensordict)
        if tensordict_out is tensordict:
            raise RuntimeError(
                "EnvBase._step should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _step before writing new tensors onto this new instance."
            )
        tensordict.unlock()

        obs_keys = self.observation_spec.keys(nested_keys=False)
        # we deliberately do not update the input values, but we want to keep track of
        # new keys considered as "input" by inverse transforms.
        in_keys = self._get_in_keys_to_exclude(tensordict)
        tensordict_out_select = tensordict_out.select(*obs_keys)
        tensordict_out = tensordict_out.exclude(*obs_keys, *in_keys)
        tensordict_out.set("next", tensordict_out_select)

        reward = tensordict_out.get("reward")
        # unsqueeze rewards if needed
        # the input tensordict may have more leading dimensions than the batch_size
        # e.g. in model-based contexts.
        batch_size = self.batch_size
        dims = len(batch_size)
        leading_batch_size = (
            tensordict_out.batch_size[:-dims] if dims else tensordict_out.shape
        )
        expected_reward_shape = torch.Size(
            [*leading_batch_size, *self.reward_spec.shape]
        )
        actual_reward_shape = reward.shape
        if actual_reward_shape != expected_reward_shape:
            reward = reward.view(expected_reward_shape)
            tensordict_out.set("reward", reward)

        done = tensordict_out.get("done")
        # unsqueeze done if needed
        expected_done_shape = torch.Size([*leading_batch_size, *batch_size, 1])
        actual_done_shape = done.shape
        if actual_done_shape != expected_done_shape:
            done = done.view(expected_done_shape)
            tensordict_out.set("done", done)

        if self.run_type_checks:
            for key in self._select_observation_keys(tensordict_out):
                obs = tensordict_out.get(key)
                self.observation_spec.type_check(obs, key)

            if tensordict_out.get("reward").dtype is not self.reward_spec.dtype:
                raise TypeError(
                    f"expected reward.dtype to be {self.reward_spec.dtype} "
                    f"but got {tensordict_out.get('reward').dtype}"
                )

            if tensordict_out.get("done").dtype is not torch.bool:
                raise TypeError(
                    f"expected done.dtype to be torch.bool but got {tensordict_out.get('done').dtype}"
                )
        tensordict.update(tensordict_out, inplace=self._inplace_update)

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
        else:
            _reset = None

        tensordict_reset = self._reset(tensordict, **kwargs)

        done = tensordict_reset.get("done", None)
        if done is not None:
            # unsqueeze done if needed
            # the input tensordict may have more leading dimensions than the batch_size
            # e.g. in model-based contexts.
            batch_size = self.batch_size
            dims = len(batch_size)
            leading_batch_size = (
                tensordict_reset.batch_size[:-dims] if dims else tensordict_reset.shape
            )
            expected_done_shape = torch.Size([*leading_batch_size, *batch_size, 1])
            actual_done_shape = done
            if actual_done_shape != expected_done_shape:
                done = done.view(expected_done_shape)
                tensordict_reset.set("done", done)
        else:
            tensordict_reset.set(
                "done",
                torch.zeros(
                    *tensordict_reset.batch_size,
                    1,
                    dtype=torch.bool,
                    device=self.device,
                ),
            )

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

        if (_reset is None and tensordict_reset.get("done").any()) or (
            _reset is not None and tensordict_reset.get("done")[_reset].any()
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
            static_seed (bool, optional): if True, the seed is not incremented.
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
        if tensordict.batch_size != self.batch_size and (
            self.batch_locked or self.batch_size != torch.Size([])
        ):
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
        if tensordict is None:
            tensordict = TensorDict(
                {}, device=self.device, batch_size=self.batch_size, _run_checks=False
            )

        if not self.batch_locked and not self.batch_size:
            shape = tensordict.shape
        elif not self.batch_locked and tensordict.shape != self.batch_size:
            raise RuntimeError(
                "The input tensordict and the env have a different batch size: "
                f"env.batch_size={self.batch_size} and tensordict.batch_size={tensordict.shape}. "
                f"Non batch-locked environment require the env batch-size to be either empty or to"
                f" match the tensordict one."
            )
        action = self.action_spec.rand(shape)
        tensordict.set("action", action)
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
    def specs(self) -> Specs:
        """Returns a Specs container where all the environment specs are contained.

        This feature allows one to create an environment, retrieve all of the specs in a single data container and then
        erase the environment from the workspace.

        """
        return Specs(self)

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
            auto_reset (bool, optional): if True, resets automatically the environment
                if it is in a done state when the rollout is initiated.
                Default is :obj:`True`.
            auto_cast_to_device (bool, optional): if True, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is :obj:`False`.
            break_when_any_done (bool): breaks if any of the done state is True. If False, a reset() is
                called on the sub-envs that are done. Default is True.
            return_contiguous (bool): if False, a LazyStackedTensorDict will be returned. Default is True.
            tensordict (TensorDict, optional): if auto_reset is False, an initial
                tensordict must be provided.

        Returns:
            TensorDict object containing the resulting trajectory.

        """
        try:
            policy_device = next(policy.parameters()).device
        except AttributeError:
            policy_device = "cpu"

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
                tensordict = tensordict.to(policy_device)
            tensordict = policy(tensordict)
            if auto_cast_to_device:
                tensordict = tensordict.to(env_device)
            tensordict = self.step(tensordict)

            tensordicts.append(tensordict.clone())
            if (
                break_when_any_done and tensordict.get("done").any()
            ) or i == max_steps - 1:
                break
            tensordict = step_mdp(
                tensordict,
                keep_other=True,
                exclude_reward=False,
                exclude_action=False,
                exclude_done=False,
            )
            if not break_when_any_done and tensordict.get("done").any():
                tensordict["_reset"] = tensordict.get(
                    "done",
                    torch.ones(
                        *tensordict.shape,
                        1,
                        dtype=torch.bool,
                    ),
                ).squeeze(-1)
                self.reset(tensordict)

            if callback is not None:
                callback(self, tensordict)

        batch_size = self.batch_size if tensordict is None else tensordict.batch_size

        out_td = torch.stack(tensordicts, len(batch_size))
        if return_contiguous:
            return out_td.contiguous()
        return out_td

    def _select_observation_keys(self, tensordict: TensorDictBase) -> Iterator[str]:
        for key in tensordict.keys():
            if key.rfind("observation") >= 0:
                yield key

    def _to_tensor(
        self,
        value: Union[dict, bool, float, torch.Tensor, np.ndarray],
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[torch.Tensor, dict]:
        if device is None:
            device = self.device

        if isinstance(value, dict):
            return {
                _key: self._to_tensor(_value, dtype=dtype, device=device)
                for _key, _value in value.items()
            }
        elif isinstance(value, (bool, Number)):
            value = np.array(value)

        if dtype is None and self.dtype is not None:
            dtype = self.dtype
        elif dtype is not None:
            dtype = dtype_map.get(dtype, dtype)
        else:
            dtype = value.dtype

        if not isinstance(value, torch.Tensor):
            if dtype is not None:
                try:
                    value = value.astype(dtype)
                except TypeError:
                    raise Exception(
                        "dtype must be a numpy-compatible dtype. Got {dtype}"
                    )
            value = torch.as_tensor(value, device=device)
        else:
            value = value.to(device)
        return value

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
        self.reward_spec = self.reward_spec.to(device)
        self.observation_spec = self.observation_spec.to(device)
        self.input_spec = self.input_spec.to(device)
        self.device = device
        return super().to(device)

    def fake_tensordict(self) -> TensorDictBase:
        """Returns a fake tensordict with key-value pairs that match in shape, device and dtype what can be expected during an environment rollout."""
        input_spec = self.input_spec
        observation_spec = self.observation_spec
        fake_obs = observation_spec.zero()
        fake_input = input_spec.zero()
        # the input and output key may match, but the output prevails
        # Hence we generate the input, and override using the output
        fake_in_out = fake_input.clone().update(fake_obs)
        reward_spec = self.reward_spec
        fake_reward = reward_spec.zero()
        fake_td = TensorDict(
            {
                **fake_in_out,
                "next": fake_obs.clone(),
                "reward": fake_reward,
                "done": torch.zeros(
                    (*self.batch_size, 1), dtype=torch.bool, device=self.device
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
            _run_checks=True,  # this method should not be run very often. This facilitates debugging
        )
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
