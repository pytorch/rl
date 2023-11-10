# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import warnings
from copy import copy
from types import ModuleType
from typing import Dict, List, Tuple
from warnings import warn

import numpy as np
import torch
from packaging import version

from tensordict import TensorDictBase

from torchrl._utils import implement_for
from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs.batched_envs import CloudpickleWrapper
from torchrl.envs.common import _EnvPostInit

from torchrl.envs.gym_like import (
    BaseInfoDictReader,
    default_info_dict_reader,
    GymLikeEnv,
)

from torchrl.envs.utils import _classproperty

try:
    from torch.utils._contextlib import _DecoratorContextManager
except ModuleNotFoundError:
    from torchrl._utils import _DecoratorContextManager

DEFAULT_GYM = None
IMPORT_ERROR = None
# check gym presence without importing it
_has_gym = importlib.util.find_spec("gym") is not None
if not _has_gym:
    _has_gym = importlib.util.find_spec("gymnasium") is not None

_has_mo = importlib.util.find_spec("mo_gymnasium") is not None
_has_sb3 = importlib.util.find_spec("stable_baselines3") is not None


class set_gym_backend(_DecoratorContextManager):
    """Sets the gym-backend to a certain value.

    Args:
        backend (python module, string or callable returning a module): the
            gym backend to use. Use a string or callable whenever you wish to
            avoid importing gym at loading time.

    Examples:
        >>> import gym
        >>> import gymnasium
        >>> with set_gym_backend("gym"):
        ...     assert gym_backend() == gym
        >>> with set_gym_backend(lambda: gym):
        ...     assert gym_backend() == gym
        >>> with set_gym_backend(gym):
        ...     assert gym_backend() == gym
        >>> with set_gym_backend("gymnasium"):
        ...     assert gym_backend() == gymnasium
        >>> with set_gym_backend(lambda: gymnasium):
        ...     assert gym_backend() == gymnasium
        >>> with set_gym_backend(gymnasium):
        ...     assert gym_backend() == gymnasium

    This class can also be used as a function decorator.

    Examples:
        >>> @set_gym_backend("gym")
        ... def fun():
        ...     gym = gym_backend()
        ...     print(gym)
        >>> fun()
        <module 'gym' from '/path/to/env/site-packages/gym/__init__.py'>
        >>> @set_gym_backend("gymnasium")
        ... def fun():
        ...     gym = gym_backend()
        ...     print(gym)
        >>> fun()
        <module 'gymnasium' from '/path/to/env/site-packages/gymnasium/__init__.py'>


    """

    def __init__(self, backend):
        self.backend = backend

    def _call(self):
        """Sets the backend as default."""
        global DEFAULT_GYM
        DEFAULT_GYM = self.backend
        found_setter = False
        for setter in copy(implement_for._setters):
            check_module = (
                callable(setter.module_name)
                and setter.module_name.__name__ == self.backend.__name__
            ) or setter.module_name == self.backend.__name__
            check_version = setter.check_version(
                self.backend.__version__, setter.from_version, setter.to_version
            )
            if check_module and check_version:
                setter.module_set()
                found_setter = True
        # we keep only the setters we need. This is safe because a copy is saved under self._setters_saved
        if not found_setter:
            raise ImportError(
                f"could not set anything related to gym backend "
                f"{self.backend.__name__} with version={self.backend.__version__}. "
                f"Check that the gym versions match!"
            )

    def set(self):
        """Irreversibly sets the gym backend in the script."""
        self._call()

    def __enter__(self):
        # we save a complete list of setters as well as whether they should be set.
        # we want the full list becasue we want to be able to nest the calls to set_gym_backend.
        # we also want to keep track of which ones are set to reproduce what was set before.
        self._setters_saved = copy(implement_for._implementations)
        self._call()

    def __exit__(self, exc_type, exc_val, exc_tb):
        implement_for.reset(setters_dict=self._setters_saved)
        delattr(self, "_setters_saved")

    def clone(self):
        # override this method if your children class takes __init__ parameters
        return self.__class__(self.backend)

    @property
    def backend(self):
        if isinstance(self._backend, str):
            return importlib.import_module(self._backend)
        elif callable(self._backend):
            return self._backend()
        return self._backend

    @backend.setter
    def backend(self, value):
        self._backend = value


def gym_backend(submodule=None):
    """Returns the gym backend, or a sumbodule of it.

    Args:
        submodule (str): the submodule to import. If ``None``, the backend
            itself is returned.

    Examples:
        >>> import mo_gymnasium
        >>> with set_gym_backend("gym"):
        ...     wrappers = gym_backend('wrappers')
        ...     print(wrappers)
        >>> with set_gym_backend("gymnasium"):
        ...     wrappers = gym_backend('wrappers')
        ...     print(wrappers)
    """
    global IMPORT_ERROR
    global DEFAULT_GYM
    if DEFAULT_GYM is None:
        try:
            # rule of thumbs: gymnasium precedes
            import gymnasium as gym
        except ImportError as err:
            IMPORT_ERROR = err
            try:
                import gym as gym
            except ImportError as err:
                IMPORT_ERROR = err
                gym = None
        DEFAULT_GYM = gym
    if submodule is not None:
        if not submodule.startswith("."):
            submodule = "." + submodule
            submodule = importlib.import_module(submodule, package=DEFAULT_GYM.__name__)
            return submodule
    return DEFAULT_GYM


__all__ = ["GymWrapper", "GymEnv"]


def _gym_to_torchrl_spec_transform(
    spec,
    dtype=None,
    device="cpu",
    categorical_action_encoding=False,
    remap_state_to_observation: bool = True,
) -> TensorSpec:
    """Maps the gym specs to the TorchRL specs.

    Args:
        spec: the gym space to transform
        dtype: a dtype to use for the spec. Defaults to`spec.dtype`.
        device: the device for the spec. Defaults to "cpu".
        categorical_action_encoding: whether discrete spaces should be mapped to categorical or one-hot.
            Defaults to one-hot.
        remap_state_to_observation: whether to rename the 'state' key of Dict specs to "observation". Default is true.

    """
    gym_spaces = gym_backend("spaces")
    if isinstance(spec, gym_spaces.tuple.Tuple):
        return torch.stack(
            [
                _gym_to_torchrl_spec_transform(
                    s,
                    device=device,
                    categorical_action_encoding=categorical_action_encoding,
                    remap_state_to_observation=remap_state_to_observation,
                )
                for s in spec
            ],
            0,
        )
    if isinstance(spec, gym_spaces.discrete.Discrete):
        action_space_cls = (
            DiscreteTensorSpec
            if categorical_action_encoding
            else OneHotDiscreteTensorSpec
        )
        dtype = (
            numpy_to_torch_dtype_dict[spec.dtype]
            if categorical_action_encoding
            else torch.long
        )
        return action_space_cls(spec.n, device=device, dtype=dtype)
    elif isinstance(spec, gym_spaces.multi_binary.MultiBinary):
        return BinaryDiscreteTensorSpec(
            spec.n, device=device, dtype=numpy_to_torch_dtype_dict[spec.dtype]
        )
    elif isinstance(spec, gym_spaces.multi_discrete.MultiDiscrete):
        if len(spec.nvec.shape) == 1 and len(np.unique(spec.nvec)) > 1:
            dtype = (
                numpy_to_torch_dtype_dict[spec.dtype]
                if categorical_action_encoding
                else torch.long
            )

            return (
                MultiDiscreteTensorSpec(spec.nvec, device=device, dtype=dtype)
                if categorical_action_encoding
                else MultiOneHotDiscreteTensorSpec(
                    spec.nvec, device=device, dtype=dtype
                )
            )

        return torch.stack(
            [
                _gym_to_torchrl_spec_transform(
                    spec[i],
                    device=device,
                    categorical_action_encoding=categorical_action_encoding,
                    remap_state_to_observation=remap_state_to_observation,
                )
                for i in range(len(spec.nvec))
            ],
            0,
        )
    elif isinstance(spec, gym_spaces.Box):
        shape = spec.shape
        if not len(shape):
            shape = torch.Size([1])
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        low = torch.tensor(spec.low, device=device, dtype=dtype)
        high = torch.tensor(spec.high, device=device, dtype=dtype)
        is_unbounded = low.isinf().all() and high.isinf().all()
        return (
            UnboundedContinuousTensorSpec(shape, device=device, dtype=dtype)
            if is_unbounded
            else BoundedTensorSpec(
                low,
                high,
                shape,
                dtype=dtype,
                device=device,
            )
        )
    elif isinstance(spec, (Dict,)):
        spec_out = {}
        for k in spec.keys():
            key = k
            if (
                remap_state_to_observation
                and k == "state"
                and "observation" not in spec.keys()
            ):
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                key = "observation"
            spec_out[key] = _gym_to_torchrl_spec_transform(
                spec[k],
                device=device,
                categorical_action_encoding=categorical_action_encoding,
                remap_state_to_observation=remap_state_to_observation,
            )
        # the batch-size must be set later
        return CompositeSpec(spec_out)
    elif isinstance(spec, gym_spaces.dict.Dict):
        return _gym_to_torchrl_spec_transform(
            spec.spaces,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            remap_state_to_observation=remap_state_to_observation,
        )
    else:
        raise NotImplementedError(
            f"spec of type {type(spec).__name__} is currently unaccounted for"
        )


def _get_envs(to_dict=False) -> List:
    if not _has_gym:
        raise ImportError("Gym(nasium) could not be found in your virtual environment.")
    envs = _get_gym_envs()
    envs = list(envs)
    envs = sorted(envs)
    return envs


@implement_for("gym", None, "0.26.0")
def _get_gym_envs():  # noqa: F811
    gym = gym_backend()
    return gym.envs.registration.registry.env_specs.keys()


@implement_for("gym", "0.26.0", None)
def _get_gym_envs():  # noqa: F811
    gym = gym_backend()
    return gym.envs.registration.registry.keys()


@implement_for("gymnasium", "0.27.0", None)
def _get_gym_envs():  # noqa: F811
    gym = gym_backend()
    return gym.envs.registration.registry.keys()


def _is_from_pixels(env):
    gym = gym_backend()
    observation_spec = env.observation_space
    try:
        PixelObservationWrapper = gym_backend(
            "wrappers.pixel_observation.PixelObservationWrapper"
        )
    except ModuleNotFoundError:

        class PixelObservationWrapper:
            pass

    from torchrl.envs.libs.utils import (
        GymPixelObservationWrapper as LegacyPixelObservationWrapper,
    )

    if isinstance(observation_spec, (Dict,)):
        if "pixels" in set(observation_spec.keys()):
            return True
    if isinstance(observation_spec, (gym.spaces.dict.Dict,)):
        if "pixels" in set(observation_spec.spaces.keys()):
            return True
    elif (
        isinstance(observation_spec, gym.spaces.Box)
        and (observation_spec.low == 0).all()
        and (observation_spec.high == 255).all()
        and observation_spec.low.shape[-1] == 3
        and observation_spec.low.ndim == 3
    ):
        return True
    elif isinstance(env, (LegacyPixelObservationWrapper, PixelObservationWrapper)):
        return True
    return False


class _AsyncMeta(_EnvPostInit):
    def __call__(cls, *args, **kwargs):
        instance: GymWrapper = super().__call__(*args, **kwargs)

        # before gym 0.22, there was no final_observation
        if instance._is_batched:
            gym_backend = instance.get_library_name(instance._env)
            from torchrl.envs.transforms.transforms import (
                TransformedEnv,
                VecGymEnvTransform,
            )

            if _has_sb3:
                from stable_baselines3.common.vec_env.base_vec_env import VecEnv

                if isinstance(instance._env, VecEnv):
                    backend = "sb3"
                else:
                    backend = "gym"
            else:
                backend = "gym"

            # we need 3 checks: the backend is not sb3 (if so, gymnasium is used),
            # it is gym and not gymnasium and the version is before 0.22.0
            add_info_dict = True
            if backend == "gym" and gym_backend == "gym":  # check gym against gymnasium
                import gym

                if version.parse(gym.__version__) < version.parse("0.22.0"):
                    warn(
                        "A batched gym environment is being wrapped in a GymWrapper with gym version < 0.22. "
                        "This implies that the next-observation is wrongly tracked (as the batched environment auto-resets "
                        "and discards the true next observation to return the result of the step). "
                        "This isn't compatible with TorchRL API and should be used with caution.",
                        category=UserWarning,
                    )
                    add_info_dict = False
            if add_info_dict:
                instance.set_info_dict_reader(
                    terminal_obs_reader(instance.observation_spec, backend=backend)
                )
            return TransformedEnv(instance, VecGymEnvTransform())
        return instance


class GymWrapper(GymLikeEnv, metaclass=_AsyncMeta):
    """OpenAI Gym environment wrapper.

    Examples:
        >>> env = gym.make("Pendulum-v0")
        >>> env = GymWrapper(env)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    git_url = "https://github.com/openai/gym"
    libname = "gym"

    @staticmethod
    def get_library_name(env) -> str:
        """Given a gym environment, returns the backend name (either gym or gymnasium).

        This can be used to set the appropriate backend when needed:

        Examples:
            >>> env = gymnasium.make("Pendulum-v1")
            >>> with set_gym_backend(env):
            ...    env = GymWrapper(env)

        :class:`~GymWrapper` and similar use this method to set their method
        to the right backend during instantiation.

        """
        try:
            import gym

            if isinstance(env.action_space, gym.spaces.space.Space):
                return "gym"
        except ImportError:
            pass
        try:
            import gymnasium

            if isinstance(env.action_space, gymnasium.spaces.space.Space):
                return "gymnasium"
        except ImportError:
            pass
        raise ImportError(
            f"Could not find the library of env {env}. Please file an issue on torchrl github repo."
        )

    def __init__(self, env=None, categorical_action_encoding=False, **kwargs):
        if env is not None:
            kwargs["env"] = env
        self._seed_calls_reset = None
        self._categorical_action_encoding = categorical_action_encoding
        if "env" in kwargs:
            if "EnvCompatibility" in str(
                kwargs["env"]
            ):  # a hacky way of knowing if EnvCompatibility is part of the wrappers of env
                raise ValueError(
                    "GymWrapper does not support the gym.wrapper.compatibility.EnvCompatibility wrapper. "
                    "If this feature is needed, detail your use case in an issue of "
                    "https://github.com/pytorch/rl/issues."
                )
            libname = self.get_library_name(kwargs["env"])
            with set_gym_backend(libname):
                super().__init__(**kwargs)
        else:
            super().__init__(**kwargs)
        self._post_init()

    def _post_init(self):
        # writes the functions that are gym-version specific to the instance
        # once and for all. This is aimed at avoiding the need of decorating code
        # with set_gym_backend + allowing for parallel execution (which would
        # be troublesome when both an old version of gym and recent gymnasium
        # are present within the same virtual env).
        #
        # These calls seemingly do nothing but they actually get rid of the @implement_for decorator.
        # We execute them within the set_gym_backend context manager to make sure we get
        # the right implementation.
        #
        # This method is executed by the metaclass of GymWrapper.
        with set_gym_backend(self.get_library_name(self._env)):
            self._reset_output_transform = self._reset_output_transform
            self._output_transform = self._output_transform

    @property
    def _is_batched(self):
        if _has_sb3:
            from stable_baselines3.common.vec_env.base_vec_env import VecEnv

            tuple_of_classes = (VecEnv,)
        else:
            tuple_of_classes = ()
        return isinstance(
            self._env, tuple_of_classes + (gym_backend("vector").VectorEnv,)
        )

    @implement_for("gym", None, "0.27")
    def _get_batch_size(self, env):
        if hasattr(env, "num_envs"):
            batch_size = torch.Size([env.num_envs, *self.batch_size])
        else:
            batch_size = self.batch_size
        return batch_size

    @implement_for("gymnasium", "0.27", None)  # gymnasium wants the unwrapped env
    def _get_batch_size(self, env):  # noqa: F811
        if hasattr(env, "num_envs"):
            batch_size = torch.Size([env.unwrapped.num_envs, *self.batch_size])
        else:
            batch_size = self.batch_size
        return batch_size

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not (hasattr(env, "action_space") and hasattr(env, "observation_space")):
            raise TypeError("env is not of type 'gym.Env'.")

    def _build_env(
        self,
        env,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ) -> "gym.core.Env":  # noqa: F821
        self.batch_size = self._get_batch_size(env)

        env_from_pixels = _is_from_pixels(env)
        from_pixels = from_pixels or env_from_pixels
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        if from_pixels and not env_from_pixels:
            try:
                PixelObservationWrapper = gym_backend(
                    "wrappers.pixel_observation.PixelObservationWrapper"
                )
                if isinstance(env, PixelObservationWrapper):
                    raise TypeError(
                        "PixelObservationWrapper cannot be used to wrap an environment"
                        "that is already a PixelObservationWrapper instance."
                    )
            except ModuleNotFoundError:
                pass
            env = self._build_gym_env(env, pixels_only)
        return env

    def read_action(self, action):
        action = super().read_action(action)
        if (
            isinstance(self.action_spec, (OneHotDiscreteTensorSpec, DiscreteTensorSpec))
            and action.size == 1
        ):
            # some envs require an integer for indexing
            action = int(action)
        return action

    @implement_for("gym", None, "0.19.0")
    def _build_gym_env(self, env, pixels_only):  # noqa: F811
        from .utils import GymPixelObservationWrapper as PixelObservationWrapper

        return PixelObservationWrapper(env, pixels_only=pixels_only)

    @implement_for("gym", "0.19.0", "0.26.0")
    def _build_gym_env(self, env, pixels_only):  # noqa: F811
        pixel_observation = gym_backend("wrappers.pixel_observation")
        return pixel_observation.PixelObservationWrapper(env, pixels_only=pixels_only)

    @implement_for("gym", "0.26.0", None)
    def _build_gym_env(self, env, pixels_only):  # noqa: F811
        compatibility = gym_backend("wrappers.compatibility")
        pixel_observation = gym_backend("wrappers.pixel_observation")

        if env.render_mode:
            return pixel_observation.PixelObservationWrapper(
                env, pixels_only=pixels_only
            )

        warnings.warn(
            "Environments provided to GymWrapper that need to be wrapped in PixelObservationWrapper "
            "should be created with `gym.make(env_name, render_mode=mode)` where possible,"
            'where mode is either "rgb_array" or any other supported mode.'
        )
        # resetting as 0.26 comes with a very 'nice' OrderEnforcing wrapper
        env = compatibility.EnvCompatibility(env)
        env.reset()
        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as LegacyPixelObservationWrapper,
        )

        return LegacyPixelObservationWrapper(env, pixels_only=pixels_only)

    @implement_for("gymnasium", "0.27.0", None)
    def _build_gym_env(self, env, pixels_only):  # noqa: F811
        compatibility = gym_backend("wrappers.compatibility")
        pixel_observation = gym_backend("wrappers.pixel_observation")

        if env.render_mode:
            return pixel_observation.PixelObservationWrapper(
                env, pixels_only=pixels_only
            )

        warnings.warn(
            "Environments provided to GymWrapper that need to be wrapped in PixelObservationWrapper "
            "should be created with `gym.make(env_name, render_mode=mode)` where possible,"
            'where mode is either "rgb_array" or any other supported mode.'
        )
        # resetting as 0.26 comes with a very 'nice' OrderEnforcing wrapper
        env = compatibility.EnvCompatibility(env)
        env.reset()
        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as LegacyPixelObservationWrapper,
        )

        return LegacyPixelObservationWrapper(env, pixels_only=pixels_only)

    @_classproperty
    def available_envs(cls):
        if not _has_gym:
            return
        yield from _get_envs()

    @property
    def lib(self) -> ModuleType:
        return gym_backend()

    def _set_seed(self, seed: int) -> int:  # noqa: F811
        if self._seed_calls_reset is None:
            # Determine basing on gym version whether `reset` is called when setting seed.
            self._set_seed_initial(seed)
        elif self._seed_calls_reset:
            self.reset(seed=seed)
        else:
            self._env.seed(seed=seed)

        return seed

    @implement_for("gym", None, "0.15.0")
    def _set_seed_initial(self, seed: int) -> None:  # noqa: F811
        self._seed_calls_reset = False
        self._env.seed(seed)

    @implement_for("gym", "0.15.0", "0.19.0")
    def _set_seed_initial(self, seed: int) -> None:  # noqa: F811
        self._seed_calls_reset = False
        self._env.seed(seed=seed)

    @implement_for("gym", "0.19.0", None)
    def _set_seed_initial(self, seed: int) -> None:  # noqa: F811
        try:
            self.reset(seed=seed)
            self._seed_calls_reset = True
        except TypeError as err:
            warnings.warn(
                f"reset with seed kwarg returned an exception: {err}.\n"
                f"Calling env.seed from now on."
            )
            self._seed_calls_reset = False
            self._env.seed(seed=seed)

    @implement_for("gymnasium", "0.27.0", None)
    def _set_seed_initial(self, seed: int) -> None:  # noqa: F811
        try:
            self.reset(seed=seed)
            self._seed_calls_reset = True
        except TypeError as err:
            warnings.warn(
                f"reset with seed kwarg returned an exception: {err}.\n"
                f"Calling env.seed from now on."
            )
            self._seed_calls_reset = False
            self._env.seed(seed=seed)

    def _make_specs(self, env: "gym.Env", batch_size=None) -> None:  # noqa: F821
        action_spec = _gym_to_torchrl_spec_transform(
            env.action_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        observation_spec = _gym_to_torchrl_spec_transform(
            env.observation_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if not isinstance(observation_spec, CompositeSpec):
            if self.from_pixels:
                observation_spec = CompositeSpec(
                    pixels=observation_spec, shape=self.batch_size
                )
            else:
                observation_spec = CompositeSpec(
                    observation=observation_spec, shape=self.batch_size
                )
        elif observation_spec.shape[: len(self.batch_size)] != self.batch_size:
            observation_spec.shape = self.batch_size

        if hasattr(env, "reward_space") and env.reward_space is not None:
            reward_spec = _gym_to_torchrl_spec_transform(
                env.reward_space,
                device=self.device,
                categorical_action_encoding=self._categorical_action_encoding,
            )
        else:
            reward_spec = UnboundedContinuousTensorSpec(
                shape=[1],
                device=self.device,
            )
        if batch_size is not None:
            action_spec = action_spec.expand(*batch_size, *action_spec.shape)
            reward_spec = reward_spec.expand(*batch_size, *reward_spec.shape)
            observation_spec = observation_spec.expand(
                *batch_size, *observation_spec.shape
            )
        self.done_spec = self._make_done_spec()
        self.action_spec = action_spec
        if reward_spec.shape[: len(self.batch_size)] != self.batch_size:
            self.reward_spec = reward_spec.expand(*self.batch_size, *reward_spec.shape)
        else:
            self.reward_spec = reward_spec
        self.observation_spec = observation_spec

    @implement_for("gym", None, "0.26")
    def _make_done_spec(self):  # noqa: F811
        return CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "terminated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "truncated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
            },
            shape=self.batch_size,
        )

    @implement_for("gym", "0.26", None)
    def _make_done_spec(self):  # noqa: F811
        return CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "terminated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "truncated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
            },
            shape=self.batch_size,
        )

    @implement_for("gymnasium", "0.27", None)
    def _make_done_spec(self):  # noqa: F811
        return CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "terminated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "truncated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
            },
            shape=self.batch_size,
        )

    @implement_for("gym", None, "0.26")
    def _reset_output_transform(self, reset_data):  # noqa: F811
        return reset_data, None

    @implement_for("gym", "0.26", None)
    def _reset_output_transform(self, reset_data):  # noqa: F811
        return reset_data

    @implement_for("gymnasium", "0.27", None)
    def _reset_output_transform(self, reset_data):  # noqa: F811
        return reset_data

    @implement_for("gym", None, "0.24")
    def _output_transform(self, step_outputs_tuple):  # noqa: F811
        observations, reward, done, info = step_outputs_tuple
        if self._is_batched:
            # info needs to be flipped
            info = _flip_info_tuple(info)
        # The variable naming follows torchrl's convention here.
        # A done is interpreted the union of terminated and truncated.
        # (as in earlier versions of gym).
        truncated = info.pop("TimeLimit.truncated", False)
        if not isinstance(done, bool) and isinstance(truncated, bool):
            # if bool is an array, make truncated an array
            truncated = [truncated] * len(done)
            truncated = np.array(truncated)
        elif not isinstance(truncated, bool):
            # make sure it's a boolean np.array
            truncated = np.array(truncated, dtype=np.dtype("bool"))
        terminated = done & ~truncated
        if not isinstance(terminated, np.ndarray):
            # if it's not a ndarray, we must return bool
            # since it's not a bool, we make it so
            terminated = bool(terminated)
        return (observations, reward, terminated, truncated, done, info)

    @implement_for("gym", "0.24", "0.26")
    def _output_transform(self, step_outputs_tuple):  # noqa: F811
        observations, reward, done, info = step_outputs_tuple
        # The variable naming follows torchrl's convention here.
        # A done is interpreted the union of terminated and truncated.
        # (as in earlier versions of gym).
        truncated = info.pop("TimeLimit.truncated", False)
        if not isinstance(done, bool) and isinstance(truncated, bool):
            # if bool is an array, make truncated an array
            truncated = [truncated] * len(done)
            truncated = np.array(truncated)
        elif not isinstance(truncated, bool):
            # make sure it's a boolean np.array
            truncated = np.array(truncated, dtype=np.dtype("bool"))
        terminated = done & ~truncated
        if not isinstance(terminated, np.ndarray):
            # if it's not a ndarray, we must return bool
            # since it's not a bool, we make it so
            terminated = bool(terminated)
        return (observations, reward, terminated, truncated, done, info)

    @implement_for("gym", "0.26", None)
    def _output_transform(self, step_outputs_tuple):  # noqa: F811
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        return (
            observations,
            reward,
            terminated,
            truncated,
            terminated | truncated,
            info,
        )

    @implement_for("gymnasium", "0.27", None)
    def _output_transform(self, step_outputs_tuple):  # noqa: F811
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        return (
            observations,
            reward,
            terminated,
            truncated,
            terminated | truncated,
            info,
        )

    def _init_env(self):
        self.reset()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )

    def rebuild_with_kwargs(self, **new_kwargs):
        self._constructor_kwargs.update(new_kwargs)
        self._env = self._build_env(**self._constructor_kwargs)
        self._make_specs(self._env)

    @property
    def info_dict_reader(self):
        if not self._info_dict_reader:
            self._info_dict_reader.append(default_info_dict_reader())
        return self._info_dict_reader

    @info_dict_reader.setter
    def info_dict_reader(self, value: callable):
        self._info_dict_reader = value

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if self._is_batched:
            # batched (aka 'vectorized') env reset is a bit special: envs are
            # automatically reset. What we do here is just to check if _reset
            # is present. If it is not, we just reset. Otherwise we just skip.
            if tensordict is None:
                return super()._reset(tensordict)
            reset = tensordict.get("_reset", None)
            if reset is None:
                return super()._reset(tensordict)
            elif reset is not None:
                return tensordict.exclude("_reset")
        return super()._reset(tensordict, **kwargs)


ACCEPTED_TYPE_ERRORS = {
    "render_mode": "__init__() got an unexpected keyword argument 'render_mode'",
    "frame_skip": "unexpected keyword argument 'frameskip'",
}


class GymEnv(GymWrapper):
    """OpenAI Gym environment wrapper.

    Examples:
        >>> env = GymEnv(env_name="Pendulum-v0", frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    def __init__(self, env_name, **kwargs):
        kwargs["env_name"] = env_name
        self._set_gym_args(kwargs)
        super().__init__(**kwargs)

    @implement_for("gym", None, "0.24.0")
    def _set_gym_args(self, kwargs) -> None:  # noqa: F811
        disable_env_checker = kwargs.pop("disable_env_checker", None)
        if disable_env_checker is not None:
            raise RuntimeError(
                "disable_env_checker should only be set if gym version is > 0.24"
            )

    @implement_for("gym", "0.24.0", None)
    def _set_gym_args(  # noqa: F811
        self,
        kwargs,
    ) -> None:
        kwargs.setdefault("disable_env_checker", True)

    @implement_for("gymnasium", "0.27.0", None)
    def _set_gym_args(  # noqa: F811
        self,
        kwargs,
    ) -> None:
        kwargs.setdefault("disable_env_checker", True)

    def _async_env(self, *args, **kwargs):
        return gym_backend("vector").AsyncVectorEnv(*args, **kwargs)

    def _build_env(
        self,
        env_name: str,
        **kwargs,
    ) -> "gym.core.Env":  # noqa: F821
        if not _has_gym:
            raise RuntimeError(
                f"gym not found, unable to create {env_name}. "
                f"Consider downloading and installing gym from"
                f" {self.git_url}"
            )
        from_pixels = kwargs.pop("from_pixels", False)
        self._set_gym_default(kwargs, from_pixels)
        pixels_only = kwargs.pop("pixels_only", True)
        num_envs = kwargs.pop("num_envs", 0)
        made_env = False
        kwargs["frameskip"] = self.frame_skip
        self.wrapper_frame_skip = 1
        while not made_env:
            # env.__init__ may not be compatible with all the kwargs that
            # have been preset. We iterate through the various solutions
            # to find the config that works.
            try:
                with warnings.catch_warnings(record=True) as w:
                    # we catch warnings as they may cause silent bugs
                    env = self.lib.make(env_name, **kwargs)
                    if len(w) and "frameskip" in str(w[-1].message):
                        raise TypeError("unexpected keyword argument 'frameskip'")
                made_env = True
            except TypeError as err:
                if ACCEPTED_TYPE_ERRORS["frame_skip"] in str(err):
                    # we can disable this, not strictly indispensable to know
                    # warn(
                    #     "Discarding frameskip arg. This will be taken care of by TorchRL env wrapper."
                    # )
                    self.wrapper_frame_skip = kwargs.pop("frameskip")
                elif ACCEPTED_TYPE_ERRORS["render_mode"] in str(err):
                    warn("Discarding render_mode from the env constructor.")
                    kwargs.pop("render_mode")
                else:
                    raise err
        env = super()._build_env(env, pixels_only=pixels_only, from_pixels=from_pixels)
        if num_envs > 0:
            try:
                env = self._async_env([CloudpickleWrapper(lambda: env)] * num_envs)
            except RuntimeError:
                # It would fail if the environment is not pickable. In that case,
                # delegating environment instantiation to each subprocess as a fallback.
                env = self._async_env(
                    [lambda: self.lib.make(env_name, **kwargs)] * num_envs
                )
            self.batch_size = torch.Size([num_envs, *self.batch_size])
        return env

    @implement_for("gym", None, "0.25.1")
    def _set_gym_default(self, kwargs, from_pixels: bool) -> None:  # noqa: F811
        # Do nothing for older gym versions.
        pass

    @implement_for("gym", "0.25.1", None)
    def _set_gym_default(self, kwargs, from_pixels: bool) -> None:  # noqa: F811
        if from_pixels:
            kwargs.setdefault("render_mode", "rgb_array")

    @implement_for("gymnasium", "0.27.0", None)
    def _set_gym_default(self, kwargs, from_pixels: bool) -> None:  # noqa: F811
        if from_pixels:
            kwargs.setdefault("render_mode", "rgb_array")

    @property
    def env_name(self):
        return self._constructor_kwargs["env_name"]

    def _check_kwargs(self, kwargs: Dict):
        if "env_name" not in kwargs:
            raise TypeError("Expected 'env_name' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, batch_size={self.batch_size}, device={self.device})"


class MOGymWrapper(GymWrapper):
    """FARAMA MO-Gymnasium environment wrapper.

    Examples:
        >>> import mo_gymnasium as mo_gym
        >>> env = MOGymWrapper(mo_gym.make('minecart-v0'), frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    git_url = "https://github.com/Farama-Foundation/MO-Gymnasium"
    libname = "mo-gymnasium"

    _make_specs = set_gym_backend("gymnasium")(GymEnv._make_specs)


class MOGymEnv(GymEnv):
    """FARAMA MO-Gymnasium environment wrapper.

    Examples:
        >>> env = MOGymEnv(env_name="minecart-v0", frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    git_url = "https://github.com/Farama-Foundation/MO-Gymnasium"
    libname = "mo-gymnasium"

    @property
    def lib(self) -> ModuleType:
        if _has_mo:
            import mo_gymnasium as mo_gym

            return mo_gym
        else:
            try:
                import mo_gymnasium  # noqa: F401
            except ImportError as err:
                raise ImportError("MO-gymnasium not found, check installation") from err

    _make_specs = set_gym_backend("gymnasium")(GymEnv._make_specs)


class terminal_obs_reader(BaseInfoDictReader):
    """Terminal observation reader for 'vectorized' gym environments.

    When running envs in parallel, Gym(nasium) writes the result of the true call
    to `step` in `"final_observation"` entry within the `info` dictionary.

    This breaks the natural flow and makes single-processed and multiprocessed envs
    incompatible.

    This class reads the info obs, removes the `"final_observation"` from
    the env and writes its content in the data.

    Next, a :class:`torchrl.envs.VecGymEnvTransform` transform will reorganise the
    data by caching the result of the (implicit) reset and swap the true next
    observation with the reset one. At reset time, the true reset data will be
    replaced.

    Args:
        observation_spec (CompositeSpec): The observation spec of the gym env.
        backend (str, optional): the backend of the env. One of `"sb3"` for
            stable-baselines3 or `"gym"` for gym/gymnasium.

    .. note:: In general, this class should not be handled directly. It is
        created whenever a vectorized environment is placed within a :class:`GymWrapper`.

    """

    backend_key = {
        "sb3": "terminal_observation",
        "gym": "final_observation",
    }

    def __init__(self, observation_spec: CompositeSpec, backend, name="final"):
        self.name = name
        self._info_spec = CompositeSpec(
            {(self.name, key): item.clone() for key, item in observation_spec.items()},
            shape=observation_spec.shape,
        )
        self.backend = backend

    @property
    def info_spec(self):
        return self._info_spec

    def _read_obs(self, obs, key, tensor, index):
        if obs is None:
            return
        if isinstance(obs, np.ndarray):
            # Simplest case: there is one observation,
            # presented as a np.ndarray. The key should be pixels or observation.
            # We just write that value at its location in the tensor
            tensor[index] = torch.as_tensor(obs, device=tensor.device)
        elif isinstance(obs, dict):
            if key not in obs:
                raise KeyError(
                    f"The observation {key} could not be found in the final observation dict."
                )
            subobs = obs[key]
            if subobs is not None:
                # if the obs is a dict, we expect that the key points also to
                # a value in the obs. We retrieve this value and write it in the
                # tensor
                tensor[index] = torch.as_tensor(subobs, device=tensor.device)

        elif isinstance(obs, (list, tuple)):
            # tuples are stacked along the first dimension when passing gym spaces
            # to torchrl specs. As such, we can simply stack the tuple and set it
            # at the relevant index (assuming stacking can be achieved)
            tensor[index] = torch.as_tensor(obs, device=tensor.device)
        else:
            raise NotImplementedError(
                f"Observations of type {type(obs)} are not supported yet."
            )

    def __call__(self, info_dict, tensordict):
        terminal_obs = info_dict.get(self.backend_key[self.backend], None)
        for key, item in self.info_spec.items(True, True):
            final_obs = item.zero()
            if terminal_obs is not None:
                for i, obs in enumerate(terminal_obs):
                    self._read_obs(obs, key[-1], final_obs, index=i)
            tensordict.set(key, final_obs)
        return tensordict


def _flip_info_tuple(info: Tuple[Dict]) -> Dict[str, tuple]:
    # In Gym < 0.24, batched envs returned tuples of dict, and not dict of tuples.
    # We patch this by flipping the tuple -> dict order.
    info_example = set(info[0])
    for item in info[1:]:
        info_example = info_example.union(item)
    result = {}
    for key in info_example:
        result[key] = tuple(_info.get(key, None) for _info in info)
    return result
