# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import itertools
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

from torchrl.data.tensor_specs import TensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs.common import _EnvWrapper


class BaseInfoDictReader(metaclass=abc.ABCMeta):
    """Base class for info-readers."""

    @abc.abstractmethod
    def __call__(
        self, info_dict: Dict[str, Any], tensordict: TensorDictBase
    ) -> TensorDictBase:
        raise NotImplementedError

    @abc.abstractproperty
    def info_spec(self) -> Dict[str, TensorSpec]:
        raise NotImplementedError


class default_info_dict_reader(BaseInfoDictReader):
    """Default info-key reader.

    In cases where keys can be directly written to a tensordict (mostly if they abide to the
    tensordict shape), one simply needs to indicate the keys to be registered during
    instantiation.

    Examples:
        >>> from torchrl.envs.libs.gym import GymWrapper
        >>> from torchrl.envs import default_info_dict_reader
        >>> reader = default_info_dict_reader(["my_info_key"])
        >>> # assuming "some_env-v0" returns a dict with a key "my_info_key"
        >>> env = GymWrapper(gym.make("some_env-v0"))
        >>> env.set_info_dict_reader(info_dict_reader=reader)
        >>> tensordict = env.reset()
        >>> tensordict = env.rand_step(tensordict)
        >>> assert "my_info_key" in tensordict.keys()

    """

    def __init__(
        self,
        keys: List[str] = None,
        spec: Union[Sequence[TensorSpec], Dict[str, TensorSpec]] = None,
    ):
        if keys is None:
            keys = []
        self.keys = keys

        if isinstance(spec, Sequence):
            if len(spec) != len(self.keys):
                raise ValueError(
                    "If specifying specs for info keys with a sequence, the "
                    "length of the sequence must match the number of keys"
                )
            self._info_spec = dict(zip(self.keys, spec))
        else:
            if spec is None:
                spec = {}

            self._info_spec = {
                key: spec.get(key, UnboundedContinuousTensorSpec()) for key in self.keys
            }

    def __call__(
        self, info_dict: Dict[str, Any], tensordict: TensorDictBase
    ) -> TensorDictBase:
        if not isinstance(info_dict, dict) and len(self.keys):
            warnings.warn(
                f"Found an info_dict of type {type(info_dict)} "
                f"but expected type or subtype `dict`."
            )
        for key in self.keys:
            if key in info_dict:
                tensordict[key] = info_dict[key]
        return tensordict

    @property
    def info_spec(self) -> Dict[str, TensorSpec]:
        return self._info_spec


class GymLikeEnv(_EnvWrapper):
    """A gym-like env is an environment.

    Its behaviour is similar to gym environments in what common methods (specifically reset and step) are expected to do.

    A :obj:`GymLikeEnv` has a :obj:`.step()` method with the following signature:

        ``env.step(action: np.ndarray) -> Tuple[Union[np.ndarray, dict], double, bool, *info]``

    where the outputs are the observation, reward and done state respectively.
    In this implementation, the info output is discarded (but specific keys can be read
    by updating info_dict_reader, see :obj:`set_info_dict_reader` class method).

    By default, the first output is written at the "observation" key-value pair in the output tensordict, unless
    the first output is a dictionary. In that case, each observation output will be put at the corresponding
    :obj:`f"{key}"` location for each :obj:`f"{key}"` of the dictionary.

    It is also expected that env.reset() returns an observation similar to the one observed after a step is completed.
    """

    _info_dict_reader: List[BaseInfoDictReader]

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._info_dict_reader = []
        return super().__new__(cls, *args, _batch_locked=True, **kwargs)

    def read_action(self, action):
        """Reads the action obtained from the input TensorDict and transforms it in the format expected by the contained environment.

        Args:
            action (Tensor or TensorDict): an action to be taken in the environment

        Returns: an action in a format compatible with the contained environment.

        """
        return self.action_spec.to_numpy(action, safe=False)

    def read_done(
        self,
        terminated: bool | None = None,
        truncated: bool | None = None,
        done: bool | None = None,
    ) -> Tuple[bool | np.ndarray, bool | np.ndarray, bool | np.ndarray, bool]:
        """Done state reader.

        In torchrl, a `"done"` signal means that a trajectory has reach its end,
        either because it has been interrupted or because it is terminated.
        Truncated means the episode has been interrupted early.
        Terminated means the task is finished, the episode is completed.

        Args:
            terminated (np.ndarray, boolean or other format): completion state
                obtained from the environment.
                ``"terminated"`` equates to ``"termination"`` in gymnasium:
                the signal that the environment has reached the end of the
                episode, any data coming after this should be considered as nonsensical.
                Defaults to ``None``.
            truncated (bool or None): early truncation signal.
                Defaults to ``None``.
            done (bool or None): end-of-trajectory signal.
                This should be the fallback value of envs which do not specify
                if the ``"done"`` entry points to a ``"terminated"`` or
                ``"truncated"``.
                Defaults to ``None``.

        Returns: a tuple with 4 boolean / tensor values,
            - a terminated state,
            - a truncated state,
            - a done state,
            - a boolean value indicating whether the frame_skip loop should be broken.

        """
        if truncated is not None and done is None:
            done = truncated | terminated
        elif truncated is None and done is None:
            done = terminated
        do_break = done.any() if not isinstance(done, bool) else done
        if isinstance(done, bool):
            done = [done]
            if terminated is not None:
                terminated = [terminated]
            if truncated is not None:
                truncated = [truncated]
        return (
            terminated,
            truncated,
            done,
            do_break.any() if not isinstance(do_break, bool) else do_break,
        )

    def read_reward(self, reward):
        """Reads the reward and maps it to the reward space.

        Args:
            reward (torch.Tensor or TensorDict): reward to be mapped.

        """
        return self.reward_spec.encode(reward, ignore_device=True)

    def read_obs(
        self, observations: Union[Dict[str, Any], torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """Reads an observation from the environment and returns an observation compatible with the output TensorDict.

        Args:
            observations (observation under a format dictated by the inner env): observation to be read.

        """
        if isinstance(observations, dict):
            if "state" in observations and "observation" not in observations:
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                observations["observation"] = observations.pop("state")
        if not isinstance(observations, (TensorDict, dict)):
            (key,) = itertools.islice(self.observation_spec.keys(True, True), 1)
            observations = {key: observations}
        for key, val in observations.items():
            observations[key] = self.observation_spec[key].encode(
                val, ignore_device=True
            )
        # observations = self.observation_spec.encode(observations, ignore_device=True)
        return observations

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        action_np = self.read_action(action)

        reward = 0
        for _ in range(self.wrapper_frame_skip):
            obs, _reward, terminated, truncated, done, info = self._output_transform(
                self._env.step(action_np)
            )
            if isinstance(obs, list) and len(obs) == 1:
                # Until gym 0.25.2 we had rendered frames returned in lists of length 1
                obs = obs[0]

            if _reward is None:
                _reward = self.reward_spec.zero()

            reward = reward + _reward

            terminated, truncated, done, do_break = self.read_done(
                terminated=terminated, truncated=truncated, done=done
            )
            if do_break:
                break

        reward = self.read_reward(reward)
        obs_dict = self.read_obs(obs)

        if reward is None:
            reward = torch.tensor(np.nan).expand(self.reward_spec.shape)

        obs_dict[self.reward_key] = reward

        # if truncated/terminated is not in the keys, we just don't pass it even if it
        # is defined.
        if terminated is None:
            terminated = done
        if truncated is not None and "truncated" in self.done_keys:
            obs_dict["truncated"] = truncated
        obs_dict["done"] = done
        obs_dict["terminated"] = terminated

        tensordict_out = TensorDict(
            obs_dict, batch_size=tensordict.batch_size, device=self.device
        )

        if self.info_dict_reader and info is not None:
            if not isinstance(info, dict):
                warnings.warn(
                    f"Expected info to be a dictionary but got a {type(info)} with values {str(info)[:100]}."
                )
            else:
                for info_dict_reader in self.info_dict_reader:
                    out = info_dict_reader(info, tensordict_out)
                    if out is not None:
                        tensordict_out = out
        # tensordict_out = tensordict_out.to(self.device, non_blocking=True)
        return tensordict_out

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        obs, info = self._reset_output_transform(self._env.reset(**kwargs))

        source = self.read_obs(obs)

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
        )
        if self.info_dict_reader and info is not None:
            for info_dict_reader in self.info_dict_reader:
                out = info_dict_reader(info, tensordict_out)
                if out is not None:
                    tensordict_out = out
        elif info is None and self.info_dict_reader:
            # populate the reset with the items we have not seen from info
            for key, item in self.observation_spec.items(True, True):
                if key not in tensordict_out.keys(True, True):
                    tensordict_out[key] = item.zero()
        tensordict_out = tensordict_out.to(self.device, non_blocking=True)
        return tensordict_out

    @abc.abstractmethod
    def _output_transform(
        self, step_outputs_tuple: Tuple
    ) -> Tuple[
        Any,
        float | np.ndarray,
        bool | np.ndarray | None,
        bool | np.ndarray | None,
        bool | np.ndarray | None,
        dict,
    ]:
        """A method to read the output of the env step.

        Must return a tuple: (obs, reward, terminated, truncated, done, info).
        If only one end-of-trajectory is passed, it is interpreted as ``"truncated"``.
        An attempt to retrieve ``"truncated"`` from the info dict is also undertaken.
        If 2 are passed (like in gymnasium), we interpret them as ``"terminated",
        "truncated"`` (``"truncated"`` meaning that the trajectory has been
        interrupted early), and ``"done"`` is the union of the two,
        ie. the unspecified end-of-trajectory signal.

        These three concepts have different usage:

          - ``"terminated"`` indicated the final stage of a Markov Decision
            Process. It means that one should not pay attention to the
            upcoming observations (eg., in value functions) as they should be
            regarded as not valid.
          - ``"truncated"`` means that the environment has reached a stage where
            we decided to stop the collection for some reason but the next
            observation should not be discarded. If it were not for this
            arbitrary decision, the collection could have proceeded further.
          - ``"done"`` is either one or the other. It is to be interpreted as
            "a reset should be called before the next step is undertaken".

        """
        ...

    @abc.abstractmethod
    def _reset_output_transform(self, reset_outputs_tuple: Tuple) -> Tuple:
        ...

    def set_info_dict_reader(self, info_dict_reader: BaseInfoDictReader) -> GymLikeEnv:
        """Sets an info_dict_reader function.

        This function should take as input an
        info_dict dictionary and the tensordict returned by the step function, and
        write values in an ad-hoc manner from one to the other.

        Args:
            info_dict_reader (callable): a callable taking a input dictionary and
                output tensordict as arguments. This function should modify the
                tensordict in-place.

        Returns: the same environment with the dict_reader registered.

        Examples:
            >>> from torchrl.envs import default_info_dict_reader
            >>> from torchrl.envs.libs.gym import GymWrapper
            >>> reader = default_info_dict_reader(["my_info_key"])
            >>> # assuming "some_env-v0" returns a dict with a key "my_info_key"
            >>> env = GymWrapper(gym.make("some_env-v0")).set_info_dict_reader(info_dict_reader=reader)
            >>> tensordict = env.reset()
            >>> tensordict = env.rand_step(tensordict)
            >>> assert "my_info_key" in tensordict.keys()

        """
        self.info_dict_reader.append(info_dict_reader)
        if isinstance(info_dict_reader, BaseInfoDictReader):
            # if we have a BaseInfoDictReader, we know what the specs will be
            # In other cases (eg, RoboHive) we will need to figure it out empirically.
            for info_key, spec in info_dict_reader.info_spec.items():
                self.observation_spec[info_key] = spec.to(self.device)
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )

    @property
    def info_dict_reader(self):
        return self._info_dict_reader

    @info_dict_reader.setter
    def info_dict_reader(self, value: callable):
        warnings.warn(
            f"Please use {type(self)}.set_info_dict_reader method to set a new info reader. "
            f"This method will append a reader to the list of existing readers (if any). "
            f"Setting info_dict_reader directly will be soon deprecated.",
            category=DeprecationWarning,
        )
        self._info_dict_reader.append(value)
