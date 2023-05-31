# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import warnings
from copy import copy
from types import ModuleType
from typing import Dict, List
from warnings import warn

import torch

try:
    from torch.utils._contextlib import _DecoratorContextManager
except ModuleNotFoundError:
    from torchrl._utils import _DecoratorContextManager

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

from torchrl.envs.gym_like import default_info_dict_reader, GymLikeEnv
from torchrl.envs.utils import _classproperty

DEFAULT_GYM = None
IMPORT_ERROR = None
# check gym presence without importing it
_has_gym = importlib.util.find_spec("gym") is not None
if not _has_gym:
    _has_gym = importlib.util.find_spec("gymnasium") is not None

_has_mo = importlib.util.find_spec("mo_gymnasium") is not None


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
        global DEFAULT_GYM
        DEFAULT_GYM = self.backend
        # implement_for.reset()
        setters = copy(implement_for._setters)
        found_setter = False
        for setter in setters:
            check_module = (
                callable(setter.module_name)
                and setter.module_name.__name__ == self.backend.__name__
            ) or setter.module_name == self.backend.__name__
            check_version = setter.check_version(
                self.backend.__version__, setter.from_version, setter.to_version
            )
            if check_module and check_version:
                setter(setter.fn)
                found_setter = True
        if not found_setter:
            raise ImportError(
                f"could not set anything related to gym backend "
                f"{self.backend.__name__} with version={self.backend.__version__}."
            )

    def __enter__(self):
        self._setters = copy(implement_for._setters)
        self._call()

    def __exit__(self, exc_type, exc_val, exc_tb):
        implement_for.reset(setters=self._setters)
        delattr(self, "_setters")

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
    spec, dtype=None, device="cpu", categorical_action_encoding=False
) -> TensorSpec:
    """Maps the gym specs to the TorchRL specs.

    By convention, 'state' keys of Dict specs will be renamed "observation" to match the
    default TorchRL keys.

    """
    gym = gym_backend()
    if isinstance(spec, gym.spaces.tuple.Tuple):
        raise NotImplementedError("gym.spaces.tuple.Tuple mapping not yet implemented")
    if isinstance(spec, gym.spaces.discrete.Discrete):
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
    elif isinstance(spec, gym.spaces.multi_binary.MultiBinary):
        return BinaryDiscreteTensorSpec(
            spec.n, device=device, dtype=numpy_to_torch_dtype_dict[spec.dtype]
        )
    elif isinstance(spec, gym.spaces.multi_discrete.MultiDiscrete):
        dtype = (
            numpy_to_torch_dtype_dict[spec.dtype]
            if categorical_action_encoding
            else torch.long
        )
        return (
            MultiDiscreteTensorSpec(spec.nvec, device=device, dtype=dtype)
            if categorical_action_encoding
            else MultiOneHotDiscreteTensorSpec(spec.nvec, device=device, dtype=dtype)
        )
    elif isinstance(spec, gym.spaces.Box):
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
            if k == "state" and "observation" not in spec.keys():
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                key = "observation"
            spec_out[key] = _gym_to_torchrl_spec_transform(
                spec[k],
                device=device,
                categorical_action_encoding=categorical_action_encoding,
            )
        return CompositeSpec(**spec_out)
    elif isinstance(spec, gym.spaces.dict.Dict):
        return _gym_to_torchrl_spec_transform(
            spec.spaces,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
        )
    else:
        raise NotImplementedError(
            f"spec of type {type(spec).__name__} is currently unaccounted for"
        )


def _get_envs(to_dict=False) -> List:
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


class GymWrapper(GymLikeEnv):
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
    def get_library_name(env):
        # try gym
        try:
            import gym

            if isinstance(env.action_space, gym.spaces.space.Space):
                return gym
        except ImportError:
            pass
        try:
            import gymnasium

            if isinstance(env.action_space, gymnasium.spaces.space.Space):
                return gymnasium
        except ImportError:
            pass
        raise RuntimeError(
            f"Could not find the library of env {env}. Please file an issue on torchrl github repo."
        )

    def __init__(self, env=None, categorical_action_encoding=False, **kwargs):
        if env is not None:
            kwargs["env"] = env
        self._seed_calls_reset = None
        self._categorical_action_encoding = categorical_action_encoding
        if "env" in kwargs:
            with set_gym_backend(self.get_library_name(kwargs["env"])):
                super().__init__(**kwargs)
        else:
            super().__init__(**kwargs)

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
    def available_envs(cls) -> List[str]:
        return _get_envs()

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

    @implement_for("gym", None, "0.19.0")
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

    def _make_specs(self, env: "gym.Env") -> None:  # noqa: F821
        self.action_spec = _gym_to_torchrl_spec_transform(
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
                observation_spec = CompositeSpec(pixels=observation_spec)
            else:
                observation_spec = CompositeSpec(observation=observation_spec)
        self.observation_spec = observation_spec
        if hasattr(env, "reward_space") and env.reward_space is not None:
            self.reward_spec = _gym_to_torchrl_spec_transform(
                env.reward_space,
                device=self.device,
                categorical_action_encoding=self._categorical_action_encoding,
            )
        else:
            self.reward_spec = UnboundedContinuousTensorSpec(
                shape=[1],
                device=self.device,
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
        if self._info_dict_reader is None:
            self._info_dict_reader = default_info_dict_reader()
        return self._info_dict_reader

    @info_dict_reader.setter
    def info_dict_reader(self, value: callable):
        self._info_dict_reader = value


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
        from_pixels = kwargs.get("from_pixels", False)
        self._set_gym_default(kwargs, from_pixels)
        if "from_pixels" in kwargs:
            del kwargs["from_pixels"]
        pixels_only = kwargs.get("pixels_only", True)
        if "pixels_only" in kwargs:
            del kwargs["pixels_only"]
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
        return super()._build_env(env, pixels_only=pixels_only, from_pixels=from_pixels)

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
