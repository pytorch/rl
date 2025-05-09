# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import functools
import re
import warnings
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np
import torch
from tensordict import NonTensorData, TensorDict, TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.data.tensor_specs import Composite, NonTensor, TensorSpec, Unbounded
from torchrl.envs.common import _EnvWrapper, _maybe_unlock, EnvBase

T = TypeVar("T", bound=EnvBase)


class BaseInfoDictReader(metaclass=abc.ABCMeta):
    """Base class for info-readers."""

    @abc.abstractmethod
    def __call__(
        self, info_dict: dict[str, Any], tensordict: TensorDictBase
    ) -> TensorDictBase:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def info_spec(self) -> dict[str, TensorSpec]:
        raise NotImplementedError


class default_info_dict_reader(BaseInfoDictReader):
    """Default info-key reader.

    Args:
        keys (list of keys, optional): If provided, the list of keys to get from
            the info dictionary. Defaults to all keys.
        spec (List[TensorSpec], Dict[str, TensorSpec] or Composite, optional):
            If a list of specs is provided, each spec will be matched to its
            correspondent key to form a :class:`torchrl.data.Composite`.
            If not provided, a composite spec with :class:`~torchrl.data.Unbounded`
            specs will lazyly be created.
        ignore_private (bool, optional): If ``True``, private infos (starting with
            an underscore) will be ignored. Defaults to ``True``.

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
        keys: list[str] | None = None,
        spec: Sequence[TensorSpec] | dict[str, TensorSpec] | Composite | None = None,
        ignore_private: bool = True,
    ):
        self.ignore_private = ignore_private
        self._lazy = False
        if keys is None:
            self._lazy = True
        self.keys = keys

        if spec is None and keys is None:
            _info_spec = None
        elif spec is None:
            _info_spec = Composite({key: Unbounded(()) for key in keys}, shape=[])
        elif not isinstance(spec, Composite):
            if self.keys is not None and len(spec) != len(self.keys):
                raise ValueError(
                    "If specifying specs for info keys with a sequence, the "
                    "length of the sequence must match the number of keys"
                )
            if isinstance(spec, dict):
                _info_spec = Composite(spec, shape=[])
            else:
                _info_spec = Composite(
                    {key: spec for key, spec in zip(keys, spec)}, shape=[]
                )
        else:
            _info_spec = spec.clone()
        self._info_spec = _info_spec

    def __call__(
        self, info_dict: dict[str, Any], tensordict: TensorDictBase
    ) -> TensorDictBase:
        if not isinstance(info_dict, (dict, TensorDictBase)) and len(self.keys):
            warnings.warn(
                f"Found an info_dict of type {type(info_dict)} "
                f"but expected type or subtype `dict`."
            )
        keys = self.keys
        if keys is None:
            keys = info_dict.keys()
            if self.ignore_private:
                keys = [key for key in keys if not key.startswith("_")]
            self.keys = keys
        # create an info_spec only if there is none
        info_spec = None if self.info_spec is not None else Composite()
        for key in keys:
            if key in info_dict:
                val = info_dict[key]
                if val.dtype == np.dtype("O"):
                    val = np.stack(val)
                tensordict.set(key, val)
                if info_spec is not None:
                    val = tensordict.get(key)
                    info_spec[key] = Unbounded(
                        val.shape, device=val.device, dtype=val.dtype
                    )
            elif self.info_spec is not None:
                if key in self.info_spec:
                    # Fill missing with 0s
                    tensordict.set(key, self.info_spec[key].zero())
            else:
                raise KeyError(f"The key {key} could not be found or inferred.")
        # set the info spec if there wasn't any - this should occur only once in this class
        if info_spec is not None:
            if tensordict.device is not None:
                info_spec = info_spec.to(tensordict.device)
            self._info_spec = info_spec
        return tensordict

    def reset(self):
        self.keys = None
        self._info_spec = None

    @property
    def info_spec(self) -> dict[str, TensorSpec]:
        return self._info_spec


class GymLikeEnv(_EnvWrapper):
    """A gym-like env is an environment.

    Its behavior is similar to gym environments in what common methods (specifically reset and step) are expected to do.

    A :obj:`GymLikeEnv` has a :obj:`.step()` method with the following signature:

        ``env.step(action: np.ndarray) -> Tuple[Union[np.ndarray, dict], double, bool, *info]``

    where the outputs are the observation, reward and done state respectively.
    In this implementation, the info output is discarded (but specific keys can be read
    by updating info_dict_reader, see :meth:`set_info_dict_reader` method).

    By default, the first output is written at the "observation" key-value pair in the output tensordict, unless
    the first output is a dictionary. In that case, each observation output will be put at the corresponding
    :obj:`f"{key}"` location for each :obj:`f"{key}"` of the dictionary.

    It is also expected that env.reset() returns an observation similar to the one observed after a step is completed.
    """

    _info_dict_reader: list[BaseInfoDictReader]

    @classmethod
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, _batch_locked=True, **kwargs)
        self._info_dict_reader = []

        return self

    def fast_encoding(self, mode: bool = True) -> T:
        """Skips several checks during encoding of the environment output to accelerate the execution of the environment.

        Args:
            mode (bool, optional): the memoization mode. If ``True``, input checks will be executed only once and then
                the encoding pipeline will be pre-recorded.

        .. seealso:: :meth:`~torchrl.data.TensorSpec.memoize_cache`.

        Example:
            >>> from torchrl.envs import GymEnv
            >>> from torch.utils.benchmark import Timer
            >>>
            >>> env = GymEnv("Pendulum-v1")
            >>> t = Timer("env.rollout(1000, break_when_any_done=False)", globals=globals(), num_threads=32).adaptive_autorange()
            >>> m = t.median
            >>> print(f"Speed without memoizing: {1000/t.median: 4.4f}fps")
            Speed without memoizing:  10141.5742fps
            >>>
            >>> env.fast_encoding()
            >>> t = Timer("env.rollout(1000, break_when_any_done=False)", globals=globals(), num_threads=32).adaptive_autorange()
            >>> m = t.median
            >>> print(f"Speed with memoizing: {1000/t.median: 4.4f}fps")
            Speed with memoizing:  10576.8388fps

        """
        self.specs.memoize_encode(mode=mode)
        if mode:
            if type(self).read_obs is not GymLikeEnv.read_obs:
                raise RuntimeError(
                    "Cannot use fast_encoding as the read_obs method has been overwritten."
                )
            if type(self).read_reward is not GymLikeEnv.read_reward:
                raise RuntimeError(
                    "Cannot use fast_encoding as the read_reward method has been overwritten."
                )

        if mode:
            self.read_reward = self._read_reward_memo
            self.read_obs = self._read_obs_memo
        else:
            self.read_reward = self._read_reward_eager
            self.read_obs = self._read_obs_eager

    def read_action(self, action):
        """Reads the action obtained from the input TensorDict and transforms it in the format expected by the contained environment.

        Args:
            action (Tensor or TensorDict): an action to be taken in the environment

        Returns: an action in a format compatible with the contained environment.

        """
        action_spec = self.full_action_spec
        action_keys = self.action_keys
        if len(action_keys) == 1:
            action_spec = action_spec[action_keys[0]]
        return action_spec.to_numpy(action, safe=False)

    def read_done(
        self,
        terminated: bool | None = None,
        truncated: bool | None = None,
        done: bool | None = None,
    ) -> tuple[bool | np.ndarray, bool | np.ndarray, bool | np.ndarray, bool]:
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
            torch.as_tensor(terminated),
            torch.as_tensor(truncated),
            torch.as_tensor(done),
            do_break.any() if not isinstance(do_break, bool) else do_break,
        )

    _read_reward: Callable[[Any], Any] | None = None

    def read_reward(self, reward):
        """Reads the reward and maps it to the reward space.

        Args:
            reward (torch.Tensor or TensorDict): reward to be mapped.

        """
        return self._read_reward_eager(reward)

    def _read_reward_eager(self, reward):
        if isinstance(reward, int) and reward == 0:
            return self.reward_spec.zero()
        reward = self.reward_spec.encode(reward, ignore_device=True)

        if reward is None:
            reward = torch.tensor(np.nan).expand(self.reward_spec.shape)

        return reward

    def _read_reward_memo(self, reward):
        func = self._read_reward
        if func is not None:
            return func(reward)
        funcs = []
        if isinstance(reward, int) and reward == 0:

            def process_zero(reward):
                return self.reward_spec.zero()

            funcs.append(process_zero)
        else:

            def encode_reward(reward):
                return self.reward_spec.encode(reward, ignore_device=True)

            funcs.append(encode_reward)

        if reward is None:

            def check_none(reward):
                return torch.tensor(np.nan).expand(self.reward_spec.shape)

            funcs.append(check_none)

        if len(funcs) == 1:
            self._read_reward = funcs[0]
        else:
            self._read_reward = functools.partial(
                functools.reduce, lambda x, f: f(x), funcs
            )
        return self._read_reward(reward)

    def read_obs(
        self, observations: dict[str, Any] | torch.Tensor | np.ndarray
    ) -> dict[str, Any]:
        """Reads an observation from the environment and returns an observation compatible with the output TensorDict.

        Args:
            observations (observation under a format dictated by the inner env): observation to be read.

        """
        return self._read_obs_eager(observations)

    def _read_obs_eager(
        self, observations: dict[str, Any] | torch.Tensor | np.ndarray
    ) -> dict[str, Any]:
        if isinstance(observations, dict):
            if "state" in observations and "observation" not in observations:
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                observations["observation"] = observations.pop("state")
        if not isinstance(observations, Mapping):
            for key, spec in self.observation_spec.items(True, True):
                observations_dict = {}
                observations_dict[key] = spec.encode(observations, ignore_device=True)
                # we don't check that there is only one spec because obs spec also
                # contains the data spec of the info dict.
                break
            else:
                raise RuntimeError("Could not find any element in observation_spec.")
            observations = observations_dict
        else:
            for key, val in observations.items():
                if isinstance(self.observation_spec[key], NonTensor):
                    observations[key] = NonTensorData(val)
                else:
                    observations[key] = self.observation_spec[key].encode(
                        val, ignore_device=True
                    )
        return observations

    _read_obs: Callable[[Any], Any] | None = None

    def _read_obs_memo(
        self, observations: dict[str, Any] | torch.Tensor | np.ndarray
    ) -> dict[str, Any]:
        func = self._read_obs
        if func is not None:
            return func(observations)
        funcs = []
        if isinstance(observations, (dict, Mapping)):
            if "state" in observations and "observation" not in observations:

                def process_dict_pop(observations):
                    observations["observation"] = observations.pop("state")
                    return observations

                funcs.append(process_dict_pop)
            for key in observations.keys():
                if isinstance(self.observation_spec[key], NonTensor):

                    def process_dict(observations, key=key):
                        observations[key] = NonTensorData(observations[key])
                        return observations

                else:

                    def process_dict(observations, key=key):
                        observations[key] = self.observation_spec[key].encode(
                            observations[key], ignore_device=True
                        )
                        return observations

                funcs.append(process_dict)
        else:
            key = next(iter(self.observation_spec.keys(True, True)), None)
            if key is None:
                raise RuntimeError("Could not find any element in observation_spec.")
            spec = self.observation_spec[key]

            def process_non_dict(observations, spec=spec):
                return {key: spec.encode(observations, ignore_device=True)}

            funcs.append(process_non_dict)
        if len(funcs) == 1:
            self._read_obs = funcs[0]
        else:
            self._read_obs = functools.partial(
                functools.reduce, lambda x, f: f(x), funcs
            )
        return self._read_obs(observations)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if len(self.action_keys) == 1:
            # Use brackets to get non-tensor data
            action = tensordict[self.action_key]
        else:
            action = tensordict.select(*self.action_keys).to_dict()
        if self._convert_actions_to_numpy:
            action = self.read_action(action)

        reward = 0
        for _ in range(self.wrapper_frame_skip):
            step_result = self._env.step(action)
            (
                obs,
                _reward,
                terminated,
                truncated,
                done,
                info_dict,
            ) = self._output_transform(step_result)

            if _reward is not None:
                reward = reward + _reward
            terminated, truncated, done, do_break = self.read_done(
                terminated=terminated, truncated=truncated, done=done
            )
            if do_break:
                break

        reward = self.read_reward(reward)
        obs_dict = self.read_obs(obs)
        obs_dict[self.reward_key] = reward

        # if truncated/terminated is not in the keys, we just don't pass it even if it
        # is defined.
        if terminated is None:
            terminated = done.clone()
        if truncated is not None:
            obs_dict["truncated"] = truncated
        obs_dict["done"] = done
        obs_dict["terminated"] = terminated
        validated = self.validated
        if not validated:
            tensordict_out = TensorDict(obs_dict, batch_size=tensordict.batch_size)
            if validated is None:
                # check if any value has to be recast to something else. If not, we can safely
                # build the tensordict without running checks
                self.validated = all(
                    val is tensordict_out.get(key)
                    for key, val in TensorDict(obs_dict, []).items(True, True)
                )
        else:
            tensordict_out = TensorDict._new_unsafe(
                obs_dict,
                batch_size=tensordict.batch_size,
            )
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device)

        if self.info_dict_reader and info_dict is not None:
            if not isinstance(info_dict, dict):
                warnings.warn(
                    f"Expected info to be a dictionary but got a {type(info_dict)} with values {str(info_dict)[:100]}."
                )
            else:
                for info_dict_reader in self.info_dict_reader:
                    out = info_dict_reader(info_dict, tensordict_out)
                    if out is not None:
                        tensordict_out = out
        return tensordict_out

    @property
    def validated(self):
        return self.__dict__.get("_validated", None)

    @validated.setter
    def validated(self, value):
        self.__dict__["_validated"] = value

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if (
            tensordict is not None
            and "_reset" in tensordict
            and not tensordict["_reset"].all()
        ):
            raise RuntimeError("Partial resets are not handled at this level.")
        obs, info = self._reset_output_transform(self._env.reset(**kwargs))

        source = self.read_obs(obs)

        # _new_unsafe cannot be used because it won't wrap non-tensor correctly
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
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device)
        return tensordict_out

    @abc.abstractmethod
    def _output_transform(
        self, step_outputs_tuple: tuple
    ) -> tuple[
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
    def _reset_output_transform(self, reset_outputs_tuple: tuple) -> tuple:
        ...

    @_maybe_unlock
    def set_info_dict_reader(
        self,
        info_dict_reader: BaseInfoDictReader | None = None,
        ignore_private: bool = True,
    ) -> GymLikeEnv:
        """Sets an info_dict_reader function.

        This function should take as input an
        info_dict dictionary and the tensordict returned by the step function, and
        write values in an ad-hoc manner from one to the other.

        Args:
            info_dict_reader (Callable[[Dict], TensorDict], optional): a callable
                taking a input dictionary and output tensordict as arguments.
                This function should modify the tensordict in-place. If none is
                provided, :class:`~torchrl.envs.gym_like.default_info_dict_reader`
                will be used.
            ignore_private (bool, optional): If ``True``, private infos (starting with
                an underscore) will be ignored. Defaults to ``True``.

        Returns: the same environment with the dict_reader registered.

        .. note::
          Automatically registering an info_dict reader should be done via
          :meth:`auto_register_info_dict`, which will ensure that the env
          specs are properly constructed.

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
        if info_dict_reader is None:
            info_dict_reader = default_info_dict_reader(ignore_private=ignore_private)
        self.info_dict_reader.append(info_dict_reader)
        if isinstance(info_dict_reader, BaseInfoDictReader):
            # if we have a BaseInfoDictReader, we know what the specs will be
            # In other cases (eg, RoboHive) we will need to figure it out empirically.
            if (
                isinstance(info_dict_reader, default_info_dict_reader)
                and info_dict_reader.info_spec is None
            ):
                torchrl_logger.info(
                    "The info_dict_reader does not have specs. The only way to palliate to this issue automatically "
                    "is to run a dummy rollout and gather the specs automatically. "
                    "To silence this message, provide the specs directly to your spec reader."
                )
                # Gym does not guarantee that reset passes all info
                self.reset()
                info_dict_reader.reset()
                self.rand_step()
                self.reset()

            self.observation_spec.update(info_dict_reader.info_spec)

        return self

    def auto_register_info_dict(
        self,
        ignore_private: bool = True,
        *,
        info_dict_reader: BaseInfoDictReader = None,
    ) -> EnvBase:
        """Automatically registers the info dict and appends :class:`~torch.envs.transforms.TensorDictPrimer` instances if needed.

        If no info_dict_reader is provided, it is assumed that all the information contained in the info dict can
        be registered as numerical values within the tensordict.

        This method returns a (possibly transformed) environment where we make sure that
        the :func:`torchrl.envs.utils.check_env_specs` succeeds, whether
        the info is filled at reset time.

        .. note:: This method requires running a few iterations in the environment to
          manually check that the behavior matches expectations.

        Args:
            ignore_private (bool, optional): If ``True``, private infos (starting with
                an underscore) will be ignored. Defaults to ``True``.

        Keyword Args:
            info_dict_reader (BaseInfoDictReader, optional): the info_dict_reader, if it is known in advance.
                Unlike :meth:`set_info_dict_reader`, this method will create the primers necessary to get
                :func:`~torchrl.envs.utils.check_env_specs` to run.

        Examples:
            >>> from torchrl.envs import GymEnv
            >>> env = GymEnv("HalfCheetah-v4")
            >>> # registers the info dict reader
            >>> env.auto_register_info_dict()
            GymEnv(env=HalfCheetah-v4, batch_size=torch.Size([]), device=cpu)
            >>> env.rollout(3)
            TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([3, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                    done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            observation: Tensor(shape=torch.Size([3, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                            reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                            reward_ctrl: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False),
                            reward_run: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False),
                            terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            x_position: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False),
                            x_velocity: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False)},
                        batch_size=torch.Size([3]),
                        device=cpu,
                        is_shared=False),
                    observation: Tensor(shape=torch.Size([3, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                    reward_ctrl: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False),
                    reward_run: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False),
                    terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    x_position: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False),
                    x_velocity: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False)},
                batch_size=torch.Size([3]),
                device=cpu,
                is_shared=False)

        """
        from torchrl.envs import check_env_specs, TensorDictPrimer, TransformedEnv

        if self.info_dict_reader:
            raise RuntimeError("The environment already has an info-dict reader.")
        self.set_info_dict_reader(
            ignore_private=ignore_private, info_dict_reader=info_dict_reader
        )
        try:
            check_env_specs(self)
            return self
        except (AssertionError, RuntimeError) as err:
            patterns = [
                "The keys of the specs and data do not match",
                "The sets of keys in the tensordicts to stack are exclusive",
            ]
            for pattern in patterns:
                if re.search(pattern, str(err)):
                    result = TransformedEnv(
                        self, TensorDictPrimer(self.info_dict_reader[0].info_spec)
                    )
                    check_env_specs(result)
                    return result
            raise err

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )

    @property
    def info_dict_reader(self):
        return self._info_dict_reader
