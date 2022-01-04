from __future__ import annotations

from numbers import Number
from typing import Optional, Tuple, Callable, Union, Any, Iterator

import numpy as np
import torch
from torch import nn

from torchrl.data import CompositeSpec
from torchrl.data import TensorDict
from .utils import get_available_libraries, set_exploration_mode, step_tensor_dict
from ..data.tensordict.tensordict import _TensorDict
from ..data.utils import DEVICE_TYPING

LIBRARIES = get_available_libraries()


def _tensor_to_np(t):
    return t.detach().cpu().numpy()


dtype_map = {
    torch.float: np.float32,
    torch.double: np.float64,
    torch.bool: bool,
}

__all__ = ["Specs", "GymLikeEnv"]


class Specs:
    _keys = {"action_spec", "observation_spec", "reward_spec", "from_pixels"}

    def __init__(self, env: _EnvClass):
        self.env = env

    def __getitem__(self, item: str) -> Any:
        assert item in self._keys, f"item must be one of {self._keys}"
        return getattr(self.env, item)

    def keys(self) -> dict:
        return self._keys

    def build_tensor_dict(
            self, next_observation: bool = True, log_prob: bool = False
    ) -> _TensorDict:
        """returns a TensorDict with empty tensors of the desired shape"""
        # build a tensordict from specs
        td = TensorDict(batch_size=torch.Size([]))
        action_placeholder = torch.zeros(
            self["action_spec"].shape,
            dtype=self["action_spec"].dtype
        )
        if not isinstance(self["observation_spec"], CompositeSpec):
            observation_placeholder = torch.zeros(
                self["observation_spec"].shape,
                dtype=self["observation_spec"].dtype
            )
            td.set(f"observation",
                   observation_placeholder)
        else:
            for i, key in enumerate(self["observation_spec"]):
                item = self["observation_spec"][key]
                observation_placeholder = torch.zeros(
                    item.shape, dtype=item.dtype
                )
                td.set(f"observation_{key}",
                       observation_placeholder)
                if next_observation:
                    td.set(f"next_observation_{key}",
                           observation_placeholder.clone())

        reward_placeholder = torch.zeros(
            self["reward_spec"].shape, dtype=self["reward_spec"].dtype
        )
        done_placeholder = torch.zeros_like(reward_placeholder, dtype=torch.bool)

        td.set("action", action_placeholder)
        td.set("reward", reward_placeholder)

        if log_prob:
            td.set(
                "log_prob", torch.zeros_like(reward_placeholder, dtype=torch.float32)
            )  # we assume log_prob to be of type float32
        td.set("done", done_placeholder)
        return td


class _EnvClass:
    action_spec = None
    reward_spec = None
    observation_spec = None
    from_pixels = False
    device = "cpu"
    batch_size = torch.Size([])

    def __init__(
            self, device: DEVICE_TYPING = "cpu", dtype: Optional[Union[torch.dtype, np.dtype]] = None,
    ):
        self.device = device
        self.dtype = dtype_map.get(dtype, dtype)
        self._is_done = torch.zeros(self.batch_size, device=device)
        self._cache = dict()

    def step(self, tensor_dict: _TensorDict) -> _TensorDict:
        #  sanity check
        assert (
                tensor_dict.get("action").dtype is self.action_spec.dtype
        ), f"expected action.dtype to be {self.action_spec.dtype} but got {tensor_dict.get('action').dtype}"

        tensor_dict_out = self._step(tensor_dict)
        assert tensor_dict_out is not tensor_dict, "_EnvClass._step should return outplace changes to the input " \
                                                   "tensordict. Consider emptying the TensorDict first " \
                                                   "(tensordict.empty() or tensordict.select())"
        self.is_done = tensor_dict_out.get("done")
        self._current_tensordict = step_tensor_dict(tensor_dict_out)

        for key in self._select_observation_keys(tensor_dict_out):
            obs = tensor_dict_out.get(key)
            self.observation_spec.type_check(obs, key)

        assert (
                tensor_dict_out.get("reward").dtype is self.reward_spec.dtype
        ), f"expected reward.dtype to be {self.reward_spec.dtype} but got {tensor_dict_out.get('reward').dtype}"

        assert (
                tensor_dict_out.get("done").dtype is torch.bool
        ), f"expected done.dtype to be torch.bool but got {tensor_dict_out.get('done').dtype}"

        tensor_dict.update(tensor_dict_out, inplace=True)
        return tensor_dict

    def _step(self, tensor_dict: _TensorDict, ) -> _TensorDict:
        raise NotImplementedError

    def _reset(self, tensor_dict: _TensorDict) -> _TensorDict:
        raise NotImplementedError

    def reset(self, tensor_dict: Optional[_TensorDict] = None) -> _TensorDict:
        """resets the environment and writes the initial observation into reset_observation"""
        # if tensor_dict is None:
        #     tensor_dict = self.specs.build_tensor_dict()
        if tensor_dict is None:
            tensor_dict = TensorDict(device=self.device, batch_size=self.batch_size)
        tensor_dict_reset = self._reset(tensor_dict)
        self._current_tensordict = tensor_dict_reset
        self.is_done = tensor_dict_reset.get("done", torch.zeros(self.batch_size, dtype=torch.bool, device=self.device))
        if tensor_dict is not None:
            tensor_dict.update(tensor_dict_reset)
        else:
            tensor_dict = tensor_dict_reset
        return tensor_dict

    @property
    def current_tensordict(self) -> _TensorDict:
        return self._current_tensordict

    def set_seed(self, seed: int) -> int:
        """
        Set the seed of the environment and returns the last seed used (which is the input seed if a single environment
        is present)

        Args:
            seed: integer

        Returns: integer representing the "final seed" in case the environment has a non-empty batch. This feature
         makes sure that the same seed won't be used for two different environments.

        """
        raise NotImplementedError

    def set_state(self):
        raise NotImplementedError

    def _assert_tensordict_shape(self, tensor_dict: _TensorDict) -> None:
        assert tensor_dict.batch_size == self.batch_size, f"Expected a tensor_dict with shape==env.shape, " \
                                                          f"got {tensor_dict.batch_size} and {self.batch_size}"

    def is_done_get_fn(self) -> bool:
        return self._is_done.all()

    def is_done_set_fn(self, val: bool) -> None:
        self._is_done = val

    is_done = property(is_done_get_fn, is_done_set_fn)

    def rand_step(self, tensor_dict: Optional[_TensorDict] = None) -> _TensorDict:
        if tensor_dict is None:
            tensor_dict = self.current_tensordict  # TensorDict(batch_size=self.batch_size)
        action = self.action_spec.rand(self.batch_size)
        tensor_dict.set("action", action)
        return self.step(tensor_dict)

    @property
    def specs(self) -> Specs:
        return Specs(self)

    def rollout(
            self, policy: Optional[Callable] = None, n_steps: int = 1, callback: Optional[Callable] = None,
            auto_reset: bool = True, explore: bool = True,
    ) -> _TensorDict:
        with set_exploration_mode(explore):
            try:
                policy_device = next(policy.parameters()).device
            except:
                policy_device = "cpu"

            if auto_reset:
                tensor_dict = self.reset()
            else:
                tensor_dict = self.specs.build_tensor_dict().expand(*self.batch_size).contiguous()
                tensor_dict.update(self.current_tensordict)

            if policy is None:
                policy = lambda td: td.set(
                    "action", self.action_spec.rand(self.batch_size)
                )

            tensor_dicts = []
            if not self.is_done:
                for i in range(n_steps):
                    td = tensor_dict.to(policy_device)
                    td = policy(td)
                    tensor_dict = td.to("cpu")

                    tensor_dict = self.step(tensor_dict)
                    tensor_dicts.append(tensor_dict.clone())
                    if tensor_dict.get("done").all() or i == n_steps - 1:
                        break
                    tensor_dict = step_tensor_dict(tensor_dict)

                    if callback is not None:
                        callback(self, tensor_dict)
            else:
                raise Exception(f"reset env before calling rollout!")
            out_td = torch.stack(tensor_dicts, len(self.batch_size))
            return out_td

    def _select_observation_keys(self, tensor_dict: _TensorDict) -> Iterator[str]:
        for key in tensor_dict.keys():
            if key.rfind("observation") >= 0:
                yield key

    def return_current_tensordict(self) -> _TensorDict:
        return self.current_tensordict

    def close(self) -> None:
        self.is_closed = True
        try:
            self.env.close()
        except:
            pass

    def _to_tensor(self, value: Union[dict, bool, Number, torch.Tensor, np.ndarray],
                   device: Optional[DEVICE_TYPING] = None,
                   dtype: Optional[torch.dtype] = None) -> Union[torch.Tensor, dict]:

        if isinstance(value, dict):
            return {_key: self._to_tensor(_value, dtype=dtype, device=device) for _key, _value in value.items()}
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
        if value.ndimension() == 0:
            value = value.view(1)
        return value


class _EnvWrapper(_EnvClass):
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
            **kwargs,
    ):
        super().__init__(
            device=device, dtype=dtype,
        )
        self.envname = envname
        self.taskname = taskname

        self.frame_skip = frame_skip
        self.wrapper_frame_skip = frame_skip  # this value can be changed if frame_skip is passed during env construction

        self.constructor_kwargs = kwargs
        assert (envname in self.available_envs) and (
            taskname in self.available_envs[envname]
            if isinstance(self.available_envs, dict)
            else True
        ), f"{envname} with task {taskname} is unknown in {self.libname}"
        self._build_env(envname, taskname, **kwargs)

    def _build_env(self, envname: str, taskname: str) -> None:
        raise NotImplementedError


class GymLikeEnv(_EnvWrapper):
    """
    A gym-like env is an environment that has a .step() function with the following signature:
         ` env.step(action: np.ndarray) -> Tuple[Union[np.ndarray, dict], double, bool, *info]`
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
            obs, _reward, done, *info = self._output_transform(self.env.step(action_np))
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
        self._last_obs_dict = obs_dict

        tensor_dict_out = TensorDict(batch_size=tensor_dict.batch_size)
        for key, value in obs_dict.items():
            tensor_dict_out.set(f"next_{key}", value)
        tensor_dict_out.set("reward", reward)
        tensor_dict_out.set("done", done)
        return tensor_dict_out

    def _reset(self, tensor_dict: Optional[_TensorDict] = None) -> _TensorDict:
        obs, *_ = self._output_transform((self.env.reset(),))
        tensor_dict_out = TensorDict(source=self._read_obs(obs), batch_size=self.batch_size)
        self._is_done = torch.zeros(1, dtype=torch.bool)
        tensor_dict_out.set("done", self._is_done)
        return tensor_dict_out

    def _read_obs(self, observations: torch.Tensor) -> dict:
        observations = self.observation_spec.encode(observations)
        if isinstance(observations, dict):
            obs_dict = {
                f"observation_{key}": obs for key, obs in observations.items()
            }
        else:
            obs_dict = {"observation": observations}
        obs_dict = self._to_tensor(obs_dict)
        return obs_dict

    def _output_transform(self, step_outputs_tuple: Tuple) -> Tuple:
        """To be overwritten when step_outputs differ from Tuple[Observation: Union[np.ndarray, dict], reward: Number, done:Bool]"""
        assert isinstance(step_outputs_tuple, tuple)
        return step_outputs_tuple

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.envname}, task={self.taskname if self.taskname else None}, batch_size={self.batch_size})"


def make_tensor_dict(env: _EnvWrapper, actor: Optional[nn.Module] = None) -> _TensorDict:
    """Returns a zeroed-tensordict with fields matching those required for a full step
    (action selection and environment step) in the environment
    """
    with torch.no_grad():
        tensor_dict = env.reset()
        if actor is not None:
            tensor_dict = tensor_dict.unsqueeze(0)
            tensor_dict = actor(tensor_dict.to(next(actor.parameters()).device))
            tensor_dict = tensor_dict.squeeze(0)
        else:
            tensor_dict.set("action", env.action_spec.rand(), inplace=False)
        tensor_dict = env.step(tensor_dict.to("cpu"))
        return tensor_dict.zero_()