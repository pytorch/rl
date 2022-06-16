from __future__ import annotations

from typing import Optional, Union, Tuple

import numpy as np
import torch

from torchrl.data import TensorDict
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import step_tensordict


class GymLikeEnv(_EnvWrapper):
    info_keys = []

    """
    A gym-like env is an environment whose behaviour is similar to gym environments in what
    common methods (specifically reset and step) are expected to do.


    A `GymLikeEnv` has a `.step()` method with the following signature:

        ``env.step(action: np.ndarray) -> Tuple[Union[np.ndarray, dict], double, bool, *info]``

    where the outputs are the observation, reward and done state respectively.
    In this implementation, the info output is discarded (but specific keys can be read
    by updating the `"info_keys"` class attribute).

    By default, the first output is written at the "next_observation" key-value pair in the output tensordict, unless
    the first output is a dictionary. In that case, each observation output will be put at the corresponding
    "next_observation_{key}" location.

    It is also expected that env.reset() returns an observation similar to the one observed after a step is completed.
    """

    def _step(self, tensordict: _TensorDict) -> _TensorDict:
        action = tensordict.get("action")
        action_np = self.action_spec.to_numpy(action, safe=False)

        reward = 0.0
        for _ in range(self.wrapper_frame_skip):
            obs, _reward, done, *info = self._output_transform(
                self._env.step(action_np)
            )
            if _reward is None:
                _reward = 0.0
            reward += _reward
            # TODO: check how to deal with np arrays
            if (isinstance(done, torch.Tensor) and done.all()) or done:  # or any?
                break

        obs_dict = self._read_obs(obs)

        if reward is None:
            reward = np.nan
        reward = self._to_tensor(reward, dtype=self.reward_spec.dtype)
        done = self._to_tensor(done, dtype=torch.bool)
        self.is_done = done

        tensordict_out = TensorDict(
            obs_dict, batch_size=tensordict.batch_size, device=self.device
        )
        tensordict_out.set("reward", reward)
        tensordict_out.set("done", done)
        for key in self.info_keys:
            data = info[0][key]
            tensordict_out.set(key, data)

        self.current_tensordict = step_tensordict(tensordict_out)
        return tensordict_out

    def _reset(self, tensordict: Optional[_TensorDict] = None, **kwargs) -> _TensorDict:
        obs, *_ = self._output_transform((self._env.reset(**kwargs),))
        tensordict_out = TensorDict(
            source=self._read_obs(obs),
            batch_size=self.batch_size,
            device=self.device,
        )
        self._is_done = torch.zeros(self.batch_size, dtype=torch.bool)
        tensordict_out.set("done", self._is_done)
        return tensordict_out

    def _read_obs(self, observations: Union[dict, torch.Tensor, np.ndarray]) -> dict:
        if isinstance(observations, dict):
            observations = {"next_" + key: value for key, value in observations.items()}
        if not isinstance(observations, (TensorDict, dict)):
            key = list(self.observation_spec.keys())[0]
            observations = {key: observations}
        observations = self.observation_spec.encode(observations)
        return observations

    def _output_transform(self, step_outputs_tuple: Tuple) -> Tuple:
        """To be overwritten when step_outputs differ from Tuple[Observation: Union[np.ndarray, dict], reward: Number, done:Bool]"""
        if not isinstance(step_outputs_tuple, tuple):
            raise TypeError(
                f"Expected step_outputs_tuple type to be Tuple but got {type(step_outputs_tuple)}"
            )
        return step_outputs_tuple

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )
