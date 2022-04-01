# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections import OrderedDict
from numbers import Number
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import numpy as np
import torch

from torchrl.data import CompositeSpec, TensorDict
from ..data.tensordict.tensordict import _TensorDict
from ..data.utils import DEVICE_TYPING
from .utils import get_available_libraries, step_tensor_dict

LIBRARIES = get_available_libraries()


def _tensor_to_np(t):
    return t.detach().cpu().numpy()


dtype_map = {
    torch.float: np.float32,
    torch.double: np.float64,
    torch.bool: bool,
}

__all__ = ["Specs", "GymLikeEnv", "make_tensor_dict"]


class Specs:
    """Container for action, observation and reward specs.

    This class allows one to create an environment, retrieve all of the specs
    in a single data container (and access them in one place) before erasing
    the environment from the workspace.

    Args:
        env (_EnvClass): environment from which the specs have to be read.

    """

    _keys = {"action_spec", "observation_spec", "reward_spec", "from_pixels"}

    def __init__(self, env: _EnvClass):
        self.env = env

    def __getitem__(self, item: str) -> Any:
        if item not in self._keys:
            raise KeyError(f"item must be one of {self._keys}")
        return getattr(self.env, item)

    def keys(self) -> dict:
        return self._keys

    def build_tensor_dict(
        self, next_observation: bool = True, log_prob: bool = False
    ) -> _TensorDict:
        """returns a TensorDict with empty tensors of the desired shape"""
        # build a tensordict from specs
        td = TensorDict({}, batch_size=torch.Size([]))
        action_placeholder = torch.zeros(
            self["action_spec"].shape, dtype=self["action_spec"].dtype
        )
        if not isinstance(self["observation_spec"], CompositeSpec):
            observation_placeholder = torch.zeros(
                self["observation_spec"].shape,
                dtype=self["observation_spec"].dtype,
            )
            td.set("observation", observation_placeholder)
        else:
            for i, key in enumerate(self["observation_spec"]):
                item = self["observation_spec"][key]
                observation_placeholder = torch.zeros(item.shape, dtype=item.dtype)
                td.set(f"observation_{key}", observation_placeholder)
                if next_observation:
                    td.set(
                        f"next_observation_{key}",
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


class _EnvClass:
    """
    Abstract environment parent class for TorchRL.

    Properties:
        - observation_spec (TensorSpec): sampling spec of the observations;
        - action_spec (TensorSpec): sampling spec of the actions;
        - reward_spec (TensorSpec): sampling spec of the rewards;
        - batch_size (torch.Size): number of environments contained in the instance;
        - device (torch.device): device where the env input and output are expected to live
        - is_done (torch.Tensor): boolean value(s) indicating if the environment has reached a done state since the
            last reset
        - current_tensordict (_TensorDict): last tensordict returned by `reset` or `step`.

    Methods:
        step (_TensorDict -> _TensorDict): step in the environment
        reset (_TensorDict, optional -> _TensorDict): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (_TensorDict, optional -> _TensorDict): random step given the action spec
        rollout (Callable, ... -> _TensorDict): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)

    """

    action_spec = None
    reward_spec = None
    observation_spec = None
    from_pixels: bool
    device = torch.device("cpu")
    batch_size = torch.Size([])

    def __init__(
        self,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
    ):
        self.device = device
        self.dtype = dtype_map.get(dtype, dtype)
        self._is_done = torch.zeros(self.batch_size, device=device)
        self._cache = dict()

    def step(self, tensor_dict: _TensorDict) -> _TensorDict:
        """Makes a step in the environment.
        Step accepts a single argument, tensor_dict, which usually carries an 'action' key which indicates the action
        to be taken.
        Step will call an out-place private method, _step, which is the method to be re-written by _EnvClass subclasses.

        Args:
            tensor_dict (_TensorDict): Tensordict containing the action to be taken.

        Returns:
            the input tensor_dict, modified in place with the resulting observations, done state and reward
            (+ others if needed).

        """

        # sanity check
        if tensor_dict.get("action").dtype is not self.action_spec.dtype:
            raise TypeError(
                f"expected action.dtype to be {self.action_spec.dtype} "
                f"but got {tensor_dict.get('action').dtype}"
            )

        tensor_dict_out = self._step(tensor_dict)

        if tensor_dict_out is tensor_dict:
            raise RuntimeError(
                "_EnvClass._step should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _step before writing new tensors onto this new instance."
            )
        self.is_done = tensor_dict_out.get("done")
        self._current_tensordict = step_tensor_dict(tensor_dict_out)

        for key in self._select_observation_keys(tensor_dict_out):
            obs = tensor_dict_out.get(key)
            self.observation_spec.type_check(obs, key)

        if tensor_dict_out.get("reward").dtype is not self.reward_spec.dtype:
            raise TypeError(
                f"expected reward.dtype to be {self.reward_spec.dtype} "
                f"but got {tensor_dict_out.get('reward').dtype}"
            )

        if tensor_dict_out.get("done").dtype is not torch.bool:
            raise TypeError(
                f"expected done.dtype to be torch.bool but got {tensor_dict_out.get('done').dtype}"
            )
        tensor_dict.update(tensor_dict_out, inplace=True)

        del tensor_dict_out
        return tensor_dict

    def state_dict(self, destination: Optional[OrderedDict] = None) -> OrderedDict:
        if destination is not None:
            return destination
        return OrderedDict()

    def load_state_dict(self, state_dict: OrderedDict, **kwargs) -> None:
        pass

    def eval(self) -> _EnvClass:
        return self

    def train(self, mode: bool = True) -> _EnvClass:
        return self

    def _step(
        self,
        tensor_dict: _TensorDict,
    ) -> _TensorDict:
        raise NotImplementedError

    def _reset(self, tensor_dict: _TensorDict, **kwargs) -> _TensorDict:
        raise NotImplementedError

    def reset(self, tensor_dict: Optional[_TensorDict] = None, **kwargs) -> _TensorDict:
        """Resets the environment.
        As for step and _step, only the private method `_reset` should be overwritten by _EnvClass subclasses.

        Args:
            tensor_dict (_TensorDict, optional): tensor_dict to be used to contain the resulting new observation.
                In some cases, this input can also be used to pass argument to the reset function.
            kwargs (optional): other arguments to be passed to the native
                reset function.
        Returns:
            a tensor_dict (or the input tensor_dict, if any), modified in place with the resulting observations.

        """
        # if tensor_dict is None:
        #     tensor_dict = self.specs.build_tensor_dict()
        if tensor_dict is None:
            tensor_dict = TensorDict({}, device=self.device, batch_size=self.batch_size)
        tensor_dict_reset = self._reset(tensor_dict, **kwargs)
        if tensor_dict_reset is tensor_dict:
            raise RuntimeError(
                "_EnvClass._reset should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _reset before writing new tensors onto this new instance."
            )

        self._current_tensordict = tensor_dict_reset
        self.is_done = tensor_dict_reset.get(
            "done",
            torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
        )
        if self.is_done:
            raise RuntimeError(
                f"Env {self} was done after reset. This is (currently) not allowed."
            )
        if tensor_dict is not None:
            tensor_dict.update(tensor_dict_reset)
        else:
            tensor_dict = tensor_dict_reset
        del tensor_dict_reset
        return tensor_dict

    @property
    def current_tensordict(self) -> _TensorDict:
        """Returns the last tensordict encountered after calling `reset` or `step`."""
        try:
            return self._current_tensordict
        except AttributeError:
            print(
                f"env {self} does not have a _current_tensordict attribute. Consider calling reset() before step()."
            )

    def numel(self) -> int:
        return math.prod(self.batch_size)

    def set_seed(self, seed: int) -> int:
        """Sets the seed of the environment and returns the last seed used (
        which is the input seed if a single environment is present)

        Args:
            seed: integer

        Returns:
            integer representing the "final seed" in case the environment has
            a non-empty batch. This feature makes sure that the same seed
            won't be used for two different environments.

        """
        raise NotImplementedError

    def set_state(self):
        raise NotImplementedError

    def _assert_tensordict_shape(self, tensor_dict: _TensorDict) -> None:
        if tensor_dict.batch_size != self.batch_size:
            raise RuntimeError(
                f"Expected a tensor_dict with shape==env.shape, "
                f"got {tensor_dict.batch_size} and {self.batch_size}"
            )

    def is_done_get_fn(self) -> bool:
        return self._is_done.all()

    def is_done_set_fn(self, val: bool) -> None:
        self._is_done = val

    is_done = property(is_done_get_fn, is_done_set_fn)

    def rand_step(self, tensor_dict: Optional[_TensorDict] = None) -> _TensorDict:
        """Performs a random step in the environment given the action_spec attribute.

        Args:
            tensor_dict (_TensorDict, optional): tensordict where the resulting info should be written.

        Returns:
            a tensordict object with the new observation after a random step in the environment. The action will
            be stored with the "action" key.

        """
        if tensor_dict is None:
            tensor_dict = self.current_tensordict
        action = self.action_spec.rand(self.batch_size)
        tensor_dict.set("action", action)
        return self.step(tensor_dict)

    @property
    def specs(self) -> Specs:
        """

        Returns a Specs container where all the environment specs are contained.
        This feature allows one to create an environment, retrieve all of the specs in a single data container and then
        erase the environment from the workspace.

        """
        return Specs(self)

    def rollout(
        self,
        policy: Optional[Callable[[_TensorDict], _TensorDict]] = None,
        n_steps: int = 1,
        callback: Optional[Callable[[_TensorDict, ...], _TensorDict]] = None,
        auto_reset: bool = True,
    ) -> _TensorDict:
        """

        Args:
            policy (callable, optional): callable to be called to compute the desired action. If no policy is provided,
                actions will be called using `env.rand_step()`
                default = None
            n_steps (int, optional): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before n_steps have been executed.
                default = 1
            callback (callable, optional): function to be called at each iteration with the given TensorDict.
            auto_reset (bool): if True, resets automatically the environment if it is in a done state when the rollout
                is initiated.
                default = True.

        Returns:
            TensorDict object containing the resulting trajectory.

        """
        try:
            policy_device = next(policy.parameters()).device
        except AttributeError:
            policy_device = "cpu"

        if auto_reset:
            tensor_dict = self.reset()
        else:
            # tensor_dict = (
            #     self.specs.build_tensor_dict().expand(*self.batch_size).contiguous()
            # )
            tensor_dict = self.current_tensordict.clone()

        if policy is None:

            def policy(td):
                return td.set("action", self.action_spec.rand(self.batch_size))

        tensor_dicts = []
        if not self.is_done:
            for i in range(n_steps):
                td = tensor_dict.to(policy_device)
                td = policy(td)
                tensor_dict = td.to("cpu")

                tensor_dict = self.step(tensor_dict.clone())

                tensor_dicts.append(tensor_dict.clone())
                if tensor_dict.get("done").all() or i == n_steps - 1:
                    break
                tensor_dict = step_tensor_dict(tensor_dict)

                if callback is not None:
                    callback(self, tensor_dict)
        else:
            raise Exception("reset env before calling rollout!")

        out_td = torch.stack(tensor_dicts, len(self.batch_size))
        return out_td

    def _select_observation_keys(self, tensor_dict: _TensorDict) -> Iterator[str]:
        for key in tensor_dict.keys():
            if key.rfind("observation") >= 0:
                yield key

    def _to_tensor(
        self,
        value: Union[dict, bool, float, torch.Tensor, np.ndarray],
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[torch.Tensor, dict]:

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

        if device is None:
            device = self.device

        if not isinstance(value, torch.Tensor):
            if dtype is not None:
                try:
                    value = value.astype(dtype)
                except TypeError:
                    raise Exception(
                        "dtype must be a numpy-compatible dtype. Got {dtype}"
                    )
            value = torch.from_numpy(value)
            if device != "cpu":
                value = value.to(device)
        else:
            value = value.to(device)
        # if dtype is not None:
        #     value = value.to(dtype)
        return value

    def close(self):
        pass

    def __del__(self):
        self.close()


class _EnvWrapper(_EnvClass):
    """Abstract environment wrapper class.

    Unlike _EnvClass, _EnvWrapper comes with a `_build_env` private method that will be called upon instantiation.
    Interfaces with other libraries should be coded using _EnvWrapper.
    """

    git_url: str = ""
    available_envs: dict = {}
    libname: str = ""

    def __init__(
        self,
        envname: str,
        taskname: str = "",
        frame_skip: int = 1,
        dtype: Optional[np.dtype] = None,
        device: DEVICE_TYPING = "cpu",
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            dtype=dtype,
        )
        self.envname = envname
        self.taskname = taskname

        self.frame_skip = frame_skip
        self.wrapper_frame_skip = frame_skip  # this value can be changed if frame_skip is passed during env construction

        self.constructor_kwargs = kwargs
        if not (
            (envname in self.available_envs)
            and (
                taskname in self.available_envs[envname]
                if isinstance(self.available_envs, dict)
                else True
            )
        ):
            raise RuntimeError(
                f"{envname} with task {taskname} is unknown in {self.libname}"
            )
        self._build_env(envname, taskname, **kwargs)  # writes the self._env attribute
        self._init_env(seed=seed)  # runs all the steps to have a ready-to-use env

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(
                attr
            )  # make sure that appropriate exceptions are raised

        elif "_env" in self.__dir__():
            env = self.__getattribute__("_env")
            return getattr(env, attr)

        raise AttributeError(
            f"env not set in {self.__class__.__name__}, cannot access {attr}"
        )

    def _init_env(self, seed: Optional[int] = None) -> Optional[int]:
        """Runs all the necessary steps such that the environment is ready to use.

        This step is intended to ensure that a seed is provided to the environment (if needed) and that the environment
        is reset (if needed). For instance, DMControl envs require the env to be reset before being used, but Gym envs
        don't.

        Args:
            seed (int, optional): seed to be set, if any.

        Returns:
            the resulting seed

        """

        raise NotImplementedError

    def _build_env(
        self, envname: str, taskname: Optional[str] = None, **kwargs
    ) -> None:
        """Creates an environment from the target library and stores it with the `_env` attribute.

        When overwritten, this function should pass all the required kwargs to the env instantiation method.

        Args:
            envname (str): name of the environment
            taskname: (str, optional): task to be performed, if any.


        """
        raise NotImplementedError

    def close(self) -> None:
        """Closes the contained environment if possible."""
        self.is_closed = True
        try:
            self._env.close()
        except AttributeError:
            pass


class GymLikeEnv(_EnvWrapper):
    """
    A gym-like env is an environment.


    A `GymLikeEnv` has a `.step()` method with the following signature:

        ``env.step(action: np.ndarray) -> Tuple[Union[np.ndarray, dict], double, bool, *info]``

    where the outputs are the observation, reward and done state respectively.
    In this implementation, the info output is discarded.

    By default, the first output is written at the "next_observation" key-value pair in the output tensordict, unless
    the first output is a dictionary. In that case, each observation output will be put at the corresponding
    "next_observation_{key}" location.

    It is also expected that env.reset() returns an observation similar to the one observed after a step is completed.
    """

    def _step(self, tensor_dict: _TensorDict) -> _TensorDict:
        action = tensor_dict.get("action")
        action_np = self.action_spec.to_numpy(action)

        reward = 0.0
        for _ in range(self.wrapper_frame_skip):
            obs, _reward, done, *info = self._output_transform(
                self._env.step(action_np)
            )
            if _reward is None:
                _reward = 0.0
            reward += _reward
            if done:
                break

        obs_dict = self._read_obs(obs)

        if reward is None:
            reward = np.nan
        reward = self._to_tensor(reward, dtype=self.reward_spec.dtype)
        done = self._to_tensor(done, dtype=torch.bool)
        self._is_done = done
        self._current_tensordict = obs_dict

        tensor_dict_out = TensorDict({}, batch_size=tensor_dict.batch_size)
        for key, value in obs_dict.items():
            tensor_dict_out.set(f"next_{key}", value)
        tensor_dict_out.set("reward", reward)
        tensor_dict_out.set("done", done)
        return tensor_dict_out

    def set_seed(self, seed: Optional[int] = None) -> Optional[int]:
        if seed is not None:
            torch.manual_seed(seed)
        return self._set_seed(seed)

    def _set_seed(self, seed: Optional[int]) -> Optional[int]:
        raise NotImplementedError

    def _reset(
        self, tensor_dict: Optional[_TensorDict] = None, **kwargs
    ) -> _TensorDict:
        obs, *_ = self._output_transform((self._env.reset(**kwargs),))
        tensor_dict_out = TensorDict(
            source=self._read_obs(obs), batch_size=self.batch_size
        )
        self._is_done = torch.zeros(1, dtype=torch.bool)
        tensor_dict_out.set("done", self._is_done)
        return tensor_dict_out

    def _read_obs(self, observations: torch.Tensor) -> dict:
        observations = self.observation_spec.encode(observations)
        if isinstance(observations, dict):
            obs_dict = {f"observation_{key}": obs for key, obs in observations.items()}
        else:
            obs_dict = {"observation": observations}
        obs_dict = self._to_tensor(obs_dict)
        return obs_dict

    def _output_transform(self, step_outputs_tuple: Tuple) -> Tuple:
        """To be overwritten when step_outputs differ from Tuple[Observation: Union[np.ndarray, dict], reward: Number, done:Bool]"""
        if not isinstance(step_outputs_tuple, tuple):
            raise TypeError(
                f"Expected step_outputs_tuple type to be Tuple but got {type(step_outputs_tuple)}"
            )
        return step_outputs_tuple

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.envname}, task={self.taskname if self.taskname else None}, batch_size={self.batch_size})"


def make_tensor_dict(
    env: _EnvClass,
    policy: Optional[Callable[[_TensorDict, ...], _TensorDict]] = None,
) -> _TensorDict:
    """
    Returns a zeroed-tensordict with fields matching those required for a full step
    (action selection and environment step) in the environment

    Args:
        env (_EnvWrapper): environment defining the observation, action and reward space;
        policy (Callable, optional): policy corresponding to the environment.

    """
    with torch.no_grad():
        tensor_dict = env.reset()
        if policy is not None:
            tensor_dict = tensor_dict.unsqueeze(0)
            tensor_dict = policy(tensor_dict.to(next(policy.parameters()).device))
            tensor_dict = tensor_dict.squeeze(0)
        else:
            tensor_dict.set("action", env.action_spec.rand(), inplace=False)
        tensor_dict = env.step(tensor_dict.to("cpu"))
        return tensor_dict.zero_()
