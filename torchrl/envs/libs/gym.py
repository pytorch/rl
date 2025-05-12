# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
import importlib
import warnings
from copy import copy
from types import ModuleType
from typing import Dict
from warnings import warn

import numpy as np
import torch
from packaging import version
from tensordict import TensorDict, TensorDictBase
from torch.utils._pytree import tree_map

from torchrl._utils import implement_for
from torchrl.data.tensor_specs import (
    _minmax_dtype,
    Binary,
    Bounded,
    Categorical,
    Composite,
    MultiCategorical,
    MultiOneHot,
    NonTensor,
    OneHot,
    TensorSpec,
    Unbounded,
)
from torchrl.data.utils import numpy_to_torch_dtype_dict, torch_to_numpy_dtype_dict
from torchrl.envs.batched_envs import CloudpickleWrapper
from torchrl.envs.common import _EnvPostInit
from torchrl.envs.gym_like import default_info_dict_reader, GymLikeEnv
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
_has_minigrid = importlib.util.find_spec("minigrid") is not None


GYMNASIUM_1_ERROR = """RuntimeError: TorchRL does not support gymnasium 1.0 versions due to incompatible
changes in the Gym API.
Using gymnasium 1.0 with TorchRL would require significant modifications to your code and may result in:
* Inaccurate step counting, as the auto-reset feature can cause unpredictable numbers of steps to be executed.
* Potential data corruption, as the environment may require/produce garbage data during reset steps.
* Trajectory overlap during data collection.
* Increased computational overhead, as the library would need to handle the additional complexity of auto-resets.
* Manual filtering and boilerplate code to mitigate these issues, which would compromise the modularity and ease of
use of TorchRL.
To maintain the integrity and efficiency of our library, we cannot support this version of gymnasium at this time.
If you need to use gymnasium 1.0, we recommend exploring alternative solutions or waiting for future updates
to TorchRL and gymnasium that may address this compatibility issue.
For more information, please refer to discussion https://github.com/pytorch/rl/discussions/2483 in torchrl.
"""


def _minigrid_lib():
    assert _has_minigrid, "minigrid not found"
    import minigrid

    return minigrid


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
        found_setters = collections.defaultdict(lambda: False)
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
            elif check_module:
                found_setter = False
            else:
                found_setter = None
            if found_setter is not None:
                found_setters[setter.func_name] = (
                    found_setters[setter.func_name] or found_setter
                )
        # we keep only the setters we need. This is safe because a copy is saved under self._setters_saved
        for func_name, found_setter in found_setters.items():
            if not found_setter:
                raise ImportError(
                    f"could not set anything related to gym backend "
                    f"{self.backend.__name__} with version={self.backend.__version__} for the function with name {func_name}. "
                    f"Check that the gym versions match!"
                )

    def set(self):
        """Irreversibly sets the gym backend in the script."""
        self._call()

    def __enter__(self):
        # we save a complete list of setters as well as whether they should be set.
        # we want the full list because we want to be able to nest the calls to set_gym_backend.
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


# Define a dictionary to store conversion functions for each spec type
class _ConversionRegistry(collections.UserDict):
    def __getitem__(self, cls):
        if cls not in super().keys():
            # We want to find the closest parent
            parents = {}
            for k in self.keys():
                if not isinstance(k, str):
                    parents[k] = k
                    continue
                try:
                    space_cls = gym_backend("spaces")
                    for sbsp in k.split("."):
                        space_cls = getattr(space_cls, sbsp)
                except AttributeError:
                    # Some specs may be too recent
                    continue
                parents[space_cls] = k
            mro = cls.mro()
            for base in mro:
                for p in parents:
                    if issubclass(base, p):
                        return self[parents[p]]
            else:
                raise KeyError(
                    f"No conversion tool could be found with the gym space {cls}. "
                    f"You can register your own with `torchrl.envs.libs.register_gym_spec_conversion.`"
                )
        return super().__getitem__(cls)


_conversion_registry = _ConversionRegistry()


def register_gym_spec_conversion(spec_type):
    """Decorator to register a conversion function for a specific spec type.

    The method must have the following signature:

        >>> @register_gym_spec_conversion("spec.name")
        ... def convert_specname(
        ...     spec,
        ...     dtype=None,
        ...     device=None,
        ...     categorical_action_encoding=None,
        ...     remap_state_to_observation=None,
        ...     batch_size=None,
        ... ):

    where `gym(nasium).spaces.spec.name` is the location of the spec in gym.

    If the spec type is accessible, this will also work:

        >>> @register_gym_spec_conversion(SpecType)
        ... def convert_specname(
        ...     spec,
        ...     dtype=None,
        ...     device=None,
        ...     categorical_action_encoding=None,
        ...     remap_state_to_observation=None,
        ...     batch_size=None,
        ... ):

    ..note:: The wrapped function can be simplified, and unused kwargs can be wrapped in `**kwargs`.

    """

    def decorator(conversion_func):
        _conversion_registry[spec_type] = conversion_func
        return conversion_func

    return decorator


def _gym_to_torchrl_spec_transform(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=False,
    remap_state_to_observation: bool = True,
    batch_size: tuple = (),
) -> TensorSpec:
    """Maps the gym specs to the TorchRL specs.

    Args:
        spec (gym.spaces member): the gym space to transform.
        dtype (torch.dtype): a dtype to use for the spec.
            Defaults to`spec.dtype`.
        device (torch.device): the device for the spec.
            Defaults to ``None`` (no device for composite and default device for specs).
        categorical_action_encoding (bool): whether discrete spaces should be mapped to categorical or one-hot.
            Defaults to ``False`` (one-hot).
        remap_state_to_observation (bool): whether to rename the 'state' key of
            Dict specs to "observation". Default is true.
        batch_size (torch.Size): batch size to which expand the spec. Defaults to
            ``torch.Size([])``.
    """
    if batch_size:
        return _gym_to_torchrl_spec_transform(
            spec,
            dtype=dtype,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            remap_state_to_observation=remap_state_to_observation,
            batch_size=None,
        ).expand(batch_size)

    # Get the conversion function from the registry
    conversion_func = _conversion_registry[type(spec)]
    # Call the conversion function with the provided arguments
    return conversion_func(
        spec,
        dtype=dtype,
        device=device,
        categorical_action_encoding=categorical_action_encoding,
        remap_state_to_observation=remap_state_to_observation,
        batch_size=batch_size,
    )


# Register conversion functions for each spec type
@register_gym_spec_conversion("tuple.Tuple")
def convert_tuple_spec(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=None,
    remap_state_to_observation=None,
    batch_size=None,
):
    # Implementation for Tuple spec type
    result = torch.stack(
        [
            _gym_to_torchrl_spec_transform(
                s,
                device=device,
                categorical_action_encoding=categorical_action_encoding,
                remap_state_to_observation=remap_state_to_observation,
            )
            for s in spec
        ],
        dim=0,
    )
    return result


@register_gym_spec_conversion("discrete.Discrete")
def convert_discrete_spec(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=None,
    remap_state_to_observation=None,
    batch_size=None,
):
    # Implementation for Discrete spec type
    action_space_cls = Categorical if categorical_action_encoding else OneHot
    dtype = (
        numpy_to_torch_dtype_dict[spec.dtype]
        if categorical_action_encoding
        else torch.long
    )
    return action_space_cls(spec.n, device=device, dtype=dtype)


@register_gym_spec_conversion("multi_binary.MultiBinary")
def convert_multi_binary_spec(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=None,
    remap_state_to_observation=None,
    batch_size=None,
):
    # Implementation for MultiBinary spec type
    return Binary(spec.n, device=device, dtype=numpy_to_torch_dtype_dict[spec.dtype])


@register_gym_spec_conversion("multi_discrete.MultiDiscrete")
def convert_multidiscrete_spec(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=None,
    remap_state_to_observation=None,
    batch_size=None,
):
    if len(spec.nvec.shape) == 1 and len(np.unique(spec.nvec)) > 1:
        dtype = (
            numpy_to_torch_dtype_dict[spec.dtype]
            if categorical_action_encoding
            else torch.long
        )

        return (
            MultiCategorical(spec.nvec, device=device, dtype=dtype)
            if categorical_action_encoding
            else MultiOneHot(spec.nvec, device=device, dtype=dtype)
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


@register_gym_spec_conversion("Box")
def convert_box_spec(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=None,
    remap_state_to_observation=None,
    batch_size=None,
):
    shape = spec.shape
    if not len(shape):
        shape = torch.Size([1])
    if dtype is None:
        dtype = numpy_to_torch_dtype_dict[spec.dtype]
    low = torch.as_tensor(spec.low, device=device, dtype=dtype)
    high = torch.as_tensor(spec.high, device=device, dtype=dtype)
    is_unbounded = low.isinf().all() and high.isinf().all()

    minval, maxval = _minmax_dtype(dtype)
    minval = torch.as_tensor(minval).to(low.device, dtype)
    maxval = torch.as_tensor(maxval).to(low.device, dtype)
    is_unbounded = is_unbounded or (
        torch.isclose(low, torch.as_tensor(minval, dtype=dtype)).all()
        and torch.isclose(high, torch.as_tensor(maxval, dtype=dtype)).all()
    )
    return (
        Unbounded(shape, device=device, dtype=dtype)
        if is_unbounded
        else Bounded(
            low,
            high,
            shape,
            dtype=dtype,
            device=device,
        )
    )


@register_gym_spec_conversion("Sequence")
def convert_sequence_spec(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=None,
    remap_state_to_observation=None,
    batch_size=None,
):
    if not hasattr(spec, "stack"):
        # gym does not have a stack attribute in sequence
        raise ValueError(
            "gymnasium should be used whenever a Sequence is present, as it needs to be stacked. "
            "If you need the gym backend at all price, please raise an issue on the TorchRL GitHub repository."
        )
    if not getattr(spec, "stack", False):
        raise ValueError(
            "Sequence spaces must have the stack argument set to ``True``. "
        )
    space = spec.feature_space
    out = _gym_to_torchrl_spec_transform(space, device=device, dtype=dtype)
    out = out.unsqueeze(0)
    out.make_neg_dim(0)
    return out


@register_gym_spec_conversion(Dict)
def convert_dict_spec(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=None,
    remap_state_to_observation=None,
    batch_size=None,
):
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
            batch_size=batch_size,
        )
    # the batch-size must be set later
    return Composite(spec_out, device=device)


@register_gym_spec_conversion("Text")
def convert_text_soec(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=None,
    remap_state_to_observation=None,
    batch_size=None,
):
    return NonTensor((), device=device, example_data="a string")


@register_gym_spec_conversion("dict.Dict")
def convert_dict_spec2(
    spec,
    dtype=None,
    device=None,
    categorical_action_encoding=None,
    remap_state_to_observation=None,
    batch_size=None,
):
    return _gym_to_torchrl_spec_transform(
        spec.spaces,
        device=device,
        categorical_action_encoding=categorical_action_encoding,
        remap_state_to_observation=remap_state_to_observation,
        batch_size=batch_size,
    )


@implement_for("gym", None, "0.18")
def _box_convert(spec, gym_spaces, shape):
    low = spec.low.detach().unique().cpu().item()
    high = spec.high.detach().unique().cpu().item()
    return gym_spaces.Box(low=low, high=high, shape=shape)


@implement_for("gym", "0.18")
def _box_convert(spec, gym_spaces, shape):  # noqa: F811
    low = spec.low.detach().cpu().numpy()
    high = spec.high.detach().cpu().numpy()
    return gym_spaces.Box(low=low, high=high, shape=shape)


@implement_for("gymnasium", None, "1.0.0")
def _box_convert(spec, gym_spaces, shape):  # noqa: F811
    low = spec.low.detach().cpu().numpy()
    high = spec.high.detach().cpu().numpy()
    return gym_spaces.Box(low=low, high=high, shape=shape)


@implement_for("gymnasium", "1.0.0", "1.1.0")
def _box_convert(spec, gym_spaces, shape):  # noqa: F811
    raise ImportError(GYMNASIUM_1_ERROR)


@implement_for("gymnasium", "1.1.0")
def _box_convert(spec, gym_spaces, shape):  # noqa: F811
    low = spec.low.detach().cpu().numpy()
    high = spec.high.detach().cpu().numpy()
    return gym_spaces.Box(low=low, high=high, shape=shape)


@implement_for("gym", "0.21", None)
def _multidiscrete_convert(gym_spaces, spec):
    return gym_spaces.multi_discrete.MultiDiscrete(
        spec.nvec, dtype=torch_to_numpy_dtype_dict[spec.dtype]
    )


@implement_for("gymnasium", None, "1.0.0")
def _multidiscrete_convert(gym_spaces, spec):  # noqa: F811
    return gym_spaces.multi_discrete.MultiDiscrete(
        spec.nvec, dtype=torch_to_numpy_dtype_dict[spec.dtype]
    )


@implement_for("gymnasium", "1.0.0", "1.1.0")
def _multidiscrete_convert(gym_spaces, spec):  # noqa: F811
    raise ImportError(GYMNASIUM_1_ERROR)


@implement_for("gymnasium", "1.1.0")
def _multidiscrete_convert(gym_spaces, spec):  # noqa: F811
    return gym_spaces.multi_discrete.MultiDiscrete(
        spec.nvec, dtype=torch_to_numpy_dtype_dict[spec.dtype]
    )


@implement_for("gym", None, "0.21")
def _multidiscrete_convert(gym_spaces, spec):  # noqa: F811
    return gym_spaces.multi_discrete.MultiDiscrete(spec.nvec)


def _torchrl_to_gym_spec_transform(
    spec,
    categorical_action_encoding=False,
) -> TensorSpec:
    """Maps TorchRL specs to gym spaces.

    Args:
        spec: the torchrl spec to transform.
        categorical_action_encoding: whether discrete spaces should be mapped to categorical or one-hot.
            Defaults to one-hot.

    """
    gym_spaces = gym_backend("spaces")
    shape = spec.shape
    if any(s == -1 for s in spec.shape):
        if spec.shape[0] == -1:
            spec = spec.clone()
            spec = spec[0]
            return gym_spaces.Sequence(_torchrl_to_gym_spec_transform(spec), stack=True)
        else:
            return gym_spaces.Tuple(
                tuple(_torchrl_to_gym_spec_transform(spec) for spec in spec.unbind(0))
            )
    if isinstance(spec, MultiCategorical):
        return _multidiscrete_convert(gym_spaces, spec)
    if isinstance(spec, MultiOneHot):
        return gym_spaces.multi_discrete.MultiDiscrete(spec.nvec)
    if isinstance(spec, Binary):
        return gym_spaces.multi_binary.MultiBinary(spec.shape[-1])
    if isinstance(spec, Categorical):
        return gym_spaces.discrete.Discrete(
            spec.n
        )  # dtype=torch_to_numpy_dtype_dict[spec.dtype])
    if isinstance(spec, OneHot):
        return gym_spaces.discrete.Discrete(spec.n)
    if isinstance(spec, Unbounded):
        minval, maxval = _minmax_dtype(spec.dtype)
        return gym_spaces.Box(
            low=minval,
            high=maxval,
            shape=shape,
            dtype=torch_to_numpy_dtype_dict[spec.dtype],
        )
    if isinstance(spec, Unbounded):
        minval, maxval = _minmax_dtype(spec.dtype)
        return gym_spaces.Box(
            low=minval,
            high=maxval,
            shape=shape,
            dtype=torch_to_numpy_dtype_dict[spec.dtype],
        )
    if isinstance(spec, Bounded):
        return _box_convert(spec, gym_spaces, shape)
    if isinstance(spec, Composite):
        # remove batch size
        while spec.shape:
            spec = spec[0]
        return gym_spaces.Dict(
            **{
                key: _torchrl_to_gym_spec_transform(
                    val,
                    categorical_action_encoding=categorical_action_encoding,
                )
                for key, val in spec.items()
            }
        )
    else:
        raise NotImplementedError(
            f"spec of type {type(spec).__name__} is currently unaccounted for"
        )


def _get_envs(to_dict=False) -> list:
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


@implement_for("gymnasium", None, "1.0.0")
def _get_gym_envs():  # noqa: F811
    gym = gym_backend()
    return gym.envs.registration.registry.keys()


@implement_for("gymnasium", "1.0.0", "1.1.0")
def _get_gym_envs():  # noqa: F811
    raise ImportError(GYMNASIUM_1_ERROR)


@implement_for("gymnasium", "1.1.0")
def _get_gym_envs():  # noqa: F811
    gym = gym_backend()
    return gym.envs.registration.registry.keys()


def _is_from_pixels(env):
    observation_spec = env.observation_space
    try:
        PixelObservationWrapper = gym_backend(
            "wrappers.pixel_observation"
        ).PixelObservationWrapper
    except ModuleNotFoundError:

        class PixelObservationWrapper:
            pass

    from torchrl.envs.libs.utils import (
        GymPixelObservationWrapper as LegacyPixelObservationWrapper,
    )

    gDict = gym_backend("spaces").dict.Dict
    Box = gym_backend("spaces").Box

    if isinstance(observation_spec, (Dict,)):
        if "pixels" in set(observation_spec.keys()):
            return True
    if isinstance(observation_spec, (gDict,)):
        if "pixels" in set(observation_spec.spaces.keys()):
            return True
    elif (
        isinstance(observation_spec, Box)
        and (observation_spec.low == 0).all()
        and (observation_spec.high == 255).all()
        and observation_spec.low.shape[-1] == 3
        and observation_spec.low.ndim == 3
    ):
        return True
    else:
        while True:
            if isinstance(
                env, (LegacyPixelObservationWrapper, PixelObservationWrapper)
            ):
                return True
            if hasattr(env, "env"):
                env = env.env
            else:
                break
    return False


class _GymAsyncMeta(_EnvPostInit):
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
                    backend = gym_backend
            else:
                backend = gym_backend

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
            if gym_backend == "gymnasium":
                import gymnasium

                if version.parse(gymnasium.__version__) >= version.parse("1.1.0"):
                    add_info_dict = (
                        instance._env.autoreset_mode
                        != gymnasium.vector.AutoresetMode.DISABLED
                    )
                    if not add_info_dict:
                        return instance
            if add_info_dict:
                # register terminal_obs_reader
                instance.auto_register_info_dict(
                    info_dict_reader=terminal_obs_reader(
                        instance.observation_spec, backend=backend
                    )
                )
            return TransformedEnv(instance, VecGymEnvTransform())
        return instance


class GymWrapper(GymLikeEnv, metaclass=_GymAsyncMeta):
    """OpenAI Gym environment wrapper.

    Works across `gymnasium <https://gymnasium.farama.org/>`_ and `OpenAI/gym <https://github.com/openai/gym>`_.

    Args:
        env (gym.Env): the environment to wrap. Batched environments (:class:`~stable_baselines3.common.vec_env.base_vec_env.VecEnv`
            or :class:`gym.VectorEnv`) are supported and the environment batch-size
            will reflect the number of environments executed in parallel.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.

    Keyword Args:
        from_pixels (bool, optional): if ``True``, an attempt to return the pixel
            observations from the env will be performed. By default, these observations
            will be written under the ``"pixels"`` entry.
            The method being used varies
            depending on the gym version and may involve a ``wrappers.pixel_observation.PixelObservationWrapper``.
            Defaults to ``False``.
        pixels_only (bool, optional): if ``True``, only the pixel observations will
            be returned (by default under the ``"pixels"`` entry in the output tensordict).
            If ``False``, observations (eg, states) and pixels will be returned
            whenever ``from_pixels=True``. Defaults to ``True``.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Should match the leading dimensions of all observations, done states,
            rewards, actions and infos.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.
        convert_actions_to_numpy (bool, optional): if ``True``, actions will be
            converted from tensors to numpy arrays and moved to CPU before being passed to the
            env step function. Set this to ``False`` if the environment is evaluated
            on GPU, such as IsaacLab.
            Defaults to ``True``.

    Attributes:
        available_envs (List[str]): a list of environments to build.

    .. note::
        If an attribute cannot be found, this class will attempt to retrieve it from
        the nested env:

            >>> from torchrl.envs import GymWrapper
            >>> import gymnasium as gym
            >>> env = GymWrapper(gym.make("Pendulum-v1"))
            >>> print(env.spec.max_episode_steps)
            200

    Examples:
        >>> import gymnasium as gym
        >>> from torchrl.envs import GymWrapper
        >>> base_env = gym.make("Pendulum-v1")
        >>> env = GymWrapper(base_env)
        >>> td = env.rand_step()
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['ALE/Adventure-ram-v5', 'ALE/Adventure-v5', 'ALE/AirRaid-ram-v5', 'ALE/AirRaid-v5', 'ALE/Alien-ram-v5', 'ALE/Alien-v5',

    .. note::
        info dictionaries will be read using :class:`~torchrl.envs.gym_like.default_info_dict_reader`
        if no other reader is provided. To provide another reader, refer to
        :meth:`set_info_dict_reader`. To automatically register the info_dict
        content, refer to :meth:`torchrl.envs.GymLikeEnv.auto_register_info_dict`.
        For parallel (Vectorized) environments, the info dictionary reader is automatically set and should
        not be set manually.

    .. note:: Gym spaces are not completely covered.
        The following spaces are accounted for provided that they can be represented by a torch.Tensor, a nested tensor
        and/or within a tensordict:

        - spaces.Box
        - spaces.Sequence
        - spaces.Tuple
        - spaces.Discrete
        - spaces.MultiBinary
        - spaces.MultiDiscrete
        - spaces.Dict

        Some considerations should be made when working with gym spaces. For instance, a tuple of spaces
        can only be supported if the spaces are semantically identical (same dtype and same number of dimensions).
        Ragged dimension can be supported through :func:`~torch.nested.nested_tensor`, but then there should be only
        one level of tuple and data should be stacked along the first dimension (as nested_tensors can only be
        stacked along the first dimension).

        Check the example in examples/envs/gym_conversion_examples.py to know more!

    """

    git_url = "https://github.com/openai/gym"
    libname = "gym"

    @_classproperty
    def available_envs(cls):
        if not _has_gym:
            return []
        return list(_get_envs())

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
        self._seed_calls_reset = None
        self._categorical_action_encoding = categorical_action_encoding
        if env is not None:
            try:
                env_str = str(env)
            except TypeError:
                # MiniGrid has a bug where the __str__ method fails
                pass
            else:
                if (
                    "EnvCompatibility" in env_str
                ):  # a hacky way of knowing if EnvCompatibility is part of the wrappers of env
                    raise ValueError(
                        "GymWrapper does not support the gym.wrapper.compatibility.EnvCompatibility wrapper. "
                        "If this feature is needed, detail your use case in an issue of "
                        "https://github.com/pytorch/rl/issues."
                    )
            libname = self.get_library_name(env)
            self._validate_env(env)
            with set_gym_backend(libname):
                kwargs["env"] = env
                super().__init__(**kwargs)
        else:
            super().__init__(**kwargs)
        self._post_init()

    @implement_for("gymnasium", "1.1.0")
    def _validate_env(self, env):
        autoreset_mode = getattr(env, "autoreset_mode", None)
        if autoreset_mode is not None:
            from gymnasium.vector import AutoresetMode

            if autoreset_mode not in (AutoresetMode.DISABLED, AutoresetMode.SAME_STEP):
                raise RuntimeError(
                    "The auto-reset mode must be one of SAME_STEP or DISABLED (which is preferred). Got "
                    f"autoreset_mode={autoreset_mode}."
                )

    @implement_for("gym", None, "1.1.0")
    def _validate_env(self, env):  # noqa
        pass

    @implement_for("gymnasium", None, "1.1.0")
    def _validate_env(self, env):  # noqa
        pass

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

    @implement_for("gym")
    def _get_batch_size(self, env):
        if hasattr(env, "num_envs"):
            batch_size = torch.Size([env.num_envs, *self.batch_size])
        else:
            batch_size = self.batch_size
        return batch_size

    @implement_for("gymnasium", None, "1.0.0")  # gymnasium wants the unwrapped env
    def _get_batch_size(self, env):  # noqa: F811
        env_unwrapped = env.unwrapped
        if hasattr(env_unwrapped, "num_envs"):
            batch_size = torch.Size([env_unwrapped.num_envs, *self.batch_size])
        else:
            batch_size = self.batch_size
        return batch_size

    @implement_for("gymnasium", "1.0.0", "1.1.0")
    def _get_batch_size(self, env):  # noqa: F811
        raise ImportError(GYMNASIUM_1_ERROR)

    @implement_for("gymnasium", "1.1.0")  # gymnasium wants the unwrapped env
    def _get_batch_size(self, env):  # noqa: F811
        env_unwrapped = env.unwrapped
        if hasattr(env_unwrapped, "num_envs"):
            batch_size = torch.Size([env_unwrapped.num_envs, *self.batch_size])
        else:
            batch_size = self.batch_size
        return batch_size

    def _check_kwargs(self, kwargs: dict):
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
    ) -> gym.core.Env:  # noqa: F821
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
                        "PixelObservationWrapper cannot be used to wrap an environment "
                        "that is already a PixelObservationWrapper instance."
                    )
            except ModuleNotFoundError:
                pass
            env = self._build_gym_env(env, pixels_only)
        return env

    def read_action(self, action):
        action = super().read_action(action)
        if isinstance(self.action_spec, (OneHot, Categorical)) and action.size == 1:
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

    @implement_for("gymnasium", "1.0.0", "1.1.0")
    def _build_gym_env(self, env, pixels_only):  # noqa: F811
        raise ImportError(GYMNASIUM_1_ERROR)

    @implement_for("gymnasium", None, "1.0.0")
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

    @implement_for("gymnasium", "1.1.0")
    def _build_gym_env(self, env, pixels_only):  # noqa: F811
        wrappers = gym_backend("wrappers")

        if env.render_mode:
            return wrappers.AddRenderObservation(env, render_only=pixels_only)

        warnings.warn(
            "Environments provided to GymWrapper that need to be wrapped in PixelObservationWrapper "
            "should be created with `gym.make(env_name, render_mode=mode)` where possible,"
            'where mode is either "rgb_array" or any other supported mode.'
        )
        env.reset()
        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as LegacyPixelObservationWrapper,
        )

        return LegacyPixelObservationWrapper(env, pixels_only=pixels_only)

    @property
    def lib(self) -> ModuleType:
        return gym_backend()

    def _set_seed(self, seed: int | None) -> None:  # noqa: F811
        if self._seed_calls_reset is None:
            # Determine basing on gym version whether `reset` is called when setting seed.
            self._set_seed_initial(seed)
        elif self._seed_calls_reset:
            self.reset(seed=seed)
        else:
            self._env.seed(seed=seed)

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
            try:
                self._env.seed(seed=seed)
            except AttributeError as err2:
                raise err from err2

    @implement_for("gymnasium", "1.0.0", "1.1.0")
    def _set_seed_initial(self, seed: int) -> None:  # noqa: F811
        raise ImportError(GYMNASIUM_1_ERROR)

    @implement_for("gymnasium", None, "1.0.0")
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

    @implement_for("gymnasium", "1.1.0")
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

    @implement_for("gym")
    def _reward_space(self, env):
        if hasattr(env, "reward_space") and env.reward_space is not None:
            return env.reward_space

    @implement_for("gymnasium", "1.0.0", "1.1.0")
    def _reward_space(self, env):  # noqa: F811
        raise ImportError(GYMNASIUM_1_ERROR)

    @implement_for("gymnasium", None, "1.0.0")
    def _reward_space(self, env):  # noqa: F811
        env = env.unwrapped
        if hasattr(env, "reward_space") and env.reward_space is not None:
            rs = env.reward_space
            return rs

    @implement_for("gymnasium", "1.1.0")
    def _reward_space(self, env):  # noqa: F811
        env = env.unwrapped
        if hasattr(env, "reward_space") and env.reward_space is not None:
            rs = env.reward_space
            return rs

    def _make_specs(self, env: gym.Env, batch_size=None) -> None:  # noqa: F821
        # If batch_size is provided, we se it to tell what batch size must be used
        # instead of self.batch_size
        cur_batch_size = self.batch_size if batch_size is None else torch.Size([])
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
        if not isinstance(observation_spec, Composite):
            if self.from_pixels:
                observation_spec = Composite(
                    pixels=observation_spec, shape=cur_batch_size
                )
            else:
                observation_spec = Composite(
                    observation=observation_spec, shape=cur_batch_size
                )
        elif observation_spec.shape[: len(cur_batch_size)] != cur_batch_size:
            observation_spec.shape = cur_batch_size

        reward_space = self._reward_space(env)
        if reward_space is not None:
            reward_spec = _gym_to_torchrl_spec_transform(
                reward_space,
                device=self.device,
                categorical_action_encoding=self._categorical_action_encoding,
            )
        else:
            reward_spec = Unbounded(
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
        if reward_spec.shape[: len(cur_batch_size)] != cur_batch_size:
            self.reward_spec = reward_spec.expand(*cur_batch_size, *reward_spec.shape)
        else:
            self.reward_spec = reward_spec
        self.observation_spec = observation_spec

    @implement_for("gym", None, "0.26")
    def _make_done_spec(self):  # noqa: F811
        return Composite(
            {
                "done": Categorical(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "terminated": Categorical(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "truncated": Categorical(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
            },
            shape=self.batch_size,
        )

    @implement_for("gym", "0.26", None)
    def _make_done_spec(self):  # noqa: F811
        return Composite(
            {
                "done": Categorical(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "terminated": Categorical(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "truncated": Categorical(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
            },
            shape=self.batch_size,
        )

    @implement_for("gymnasium", "0.27", None)
    def _make_done_spec(self):  # noqa: F811
        return Composite(
            {
                "done": Categorical(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "terminated": Categorical(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
                "truncated": Categorical(
                    2, dtype=torch.bool, device=self.device, shape=(*self.batch_size, 1)
                ),
            },
            shape=self.batch_size,
        )

    @implement_for("gym", None, "0.26")
    def _reset_output_transform(self, reset_data):  # noqa: F811
        if (
            isinstance(reset_data, tuple)
            and len(reset_data) == 2
            and isinstance(reset_data[1], dict)
        ):
            return reset_data
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

        if isinstance(observations, list) and len(observations) == 1:
            # Until gym 0.25.2 we had rendered frames returned in lists of length 1
            observations = observations[0]

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

        if isinstance(observations, list) and len(observations) == 1:
            # Until gym 0.25.2 we had rendered frames returned in lists of length 1
            observations = observations[0]

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
        pass
        # init_reset = self.init_reset
        # if init_reset is None:
        #     warnings.warn(f"init_env is None in the {type(self).__name__} constructor. The current "
        #                   f"default behavior is to reset the gym env as soon as it's wrapped in the "
        #                   f"class (init_reset=True), but from v0.9 this will be changed to False. "
        #                   f"To adapt for these changes, pass init_reset to your constructor.", category=FutureWarning)
        #     init_reset = True
        # if init_reset:
        #     self._env.reset()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )

    def rebuild_with_kwargs(self, **new_kwargs):
        self._constructor_kwargs.update(new_kwargs)
        self._env = self._build_env(**self._constructor_kwargs)
        self._make_specs(self._env)

    @implement_for("gym")
    def _replace_reset(self, reset, kwargs):
        return kwargs

    @implement_for("gymnasium", None, "1.1.0")
    def _replace_reset(self, reset, kwargs):  # noqa
        return kwargs

    # From gymnasium 1.1.0, AutoresetMode.DISABLED is like resets in torchrl
    @implement_for("gymnasium", "1.1.0")
    def _replace_reset(self, reset, kwargs):  # noqa
        import gymnasium as gym

        if self._env.autoreset_mode == gym.vector.AutoresetMode.DISABLED:
            options = {"reset_mask": reset.view(self.batch_size).numpy()}
            kwargs.setdefault("options", {}).update(options)
        return kwargs

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if self._is_batched:
            # batched (aka 'vectorized') env reset is a bit special: envs are
            # automatically reset. What we do here is just to check if _reset
            # is present. If it is not, we just reset. Otherwise, we just skip.
            if tensordict is None:
                return super()._reset(tensordict, **kwargs)
            reset = tensordict.get("_reset", None)
            kwargs = self._replace_reset(reset, kwargs)
            if reset is not None:
                # we must copy the tensordict because the transform
                # expects a tuple (tensordict, tensordict_reset) where the
                # first still carries a _reset
                tensordict = tensordict.exclude("_reset")
            if reset is None or reset.all() or "options" in kwargs:
                result = super()._reset(tensordict, **kwargs)
                return result
            else:
                return tensordict
        return super()._reset(tensordict, **kwargs)


ACCEPTED_TYPE_ERRORS = {
    "render_mode": "__init__() got an unexpected keyword argument 'render_mode'",
    "frame_skip": "unexpected keyword argument 'frameskip'",
}


class GymEnv(GymWrapper):
    """OpenAI Gym environment wrapper constructed by environment ID directly.

    Works across `gymnasium <https://gymnasium.farama.org/>`_ and `OpenAI/gym <https://github.com/openai/gym>`_.

    Args:
        env_name (str): the environment id registered in `gym.registry`.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.

    Keyword Args:
        num_envs (int, optional): the number of envs to run in parallel. Defaults to
            ``None`` (a single env is to be run). :class:`~gym.vector.AsyncVectorEnv`
            will be used by default.
        disable_env_checker (bool, optional): for gym > 0.24 only. If ``True`` (default
            for these versions), the environment checker won't be run.
        from_pixels (bool, optional): if ``True``, an attempt to return the pixel
            observations from the env will be performed. By default, these observations
            will be written under the ``"pixels"`` entry.
            The method being used varies
            depending on the gym version and may involve a ``wrappers.pixel_observation.PixelObservationWrapper``.
            Defaults to ``False``.
        pixels_only (bool, optional): if ``True``, only the pixel observations will
            be returned (by default under the ``"pixels"`` entry in the output tensordict).
            If ``False``, observations (eg, states) and pixels will be returned
            whenever ``from_pixels=True``. Defaults to ``False``.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Should match the leading dimensions of all observations, done states,
            rewards, actions and infos.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs (List[str]): the list of envs that can be built.

    .. note::
        If an attribute cannot be found, this class will attempt to retrieve it from
        the nested env:

            >>> from torchrl.envs import GymEnv
            >>> env = GymEnv("Pendulum-v1")
            >>> print(env.spec.max_episode_steps)
            200


        If a use-case is not covered by TorchRL, please submit an issue on GitHub.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> env = GymEnv("Pendulum-v1")
        >>> td = env.rand_step()
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['ALE/Adventure-ram-v5', 'ALE/Adventure-v5', 'ALE/AirRaid-ram-v5', 'ALE/AirRaid-v5', 'ALE/Alien-ram-v5', 'ALE/Alien-v5',

    .. note::
        If both `OpenAI/gym` and `gymnasium` are present in the virtual environment,
        one can swap backend using :func:`~torchrl.envs.libs.gym.set_gym_backend`:

            >>> from torchrl.envs import set_gym_backend, GymEnv
            >>> with set_gym_backend("gym"):
            ...     env = GymEnv("Pendulum-v1")
            ...     print(env._env)
            <class 'gym.wrappers.time_limit.TimeLimit'>
            >>> with set_gym_backend("gymnasium"):
            ...     env = GymEnv("Pendulum-v1")
            ...     print(env._env)
            <class 'gymnasium.wrappers.time_limit.TimeLimit'>

    .. note::
        info dictionaries will be read using :class:`~torchrl.envs.gym_like.default_info_dict_reader`
        if no other reader is provided. To provide another reader, refer to
        :meth:`set_info_dict_reader`. To automatically register the info_dict
        content, refer to :meth:`torchrl.envs.GymLikeEnv.auto_register_info_dict`.

    .. note:: Gym spaces are not completely covered.
        The following spaces are accounted for provided that they can be represented by a torch.Tensor, a nested tensor
        and/or within a tensordict:

        - spaces.Box
        - spaces.Sequence
        - spaces.Tuple
        - spaces.Discrete
        - spaces.MultiBinary
        - spaces.MultiDiscrete
        - spaces.Dict

        Some considerations should be made when working with gym spaces. For instance, a tuple of spaces
        can only be supported if the spaces are semantically identical (same dtype and same number of dimensions).
        Ragged dimension can be supported through :func:`~torch.nested.nested_tensor`, but then there should be only
        one level of tuple and data should be stacked along the first dimension (as nested_tensors can only be
        stacked along the first dimension).

        Check the example in examples/envs/gym_conversion_examples.py to know more!

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

    @implement_for("gymnasium", "1.0.0", "1.1.0")
    def _set_gym_args(  # noqa: F811
        self,
        kwargs,
    ) -> None:
        raise ImportError(GYMNASIUM_1_ERROR)

    @implement_for("gymnasium", None, "1.0.0")
    def _set_gym_args(  # noqa: F811
        self,
        kwargs,
    ) -> None:
        kwargs.setdefault("disable_env_checker", True)

    @implement_for("gymnasium", "1.1.0")
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
    ) -> gym.core.Env:  # noqa: F821
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

    def _check_kwargs(self, kwargs: dict):
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

    """

    git_url = "https://github.com/Farama-Foundation/MO-Gymnasium"
    libname = "mo-gymnasium"

    _make_specs = set_gym_backend("gymnasium")(GymEnv._make_specs)

    @_classproperty
    def available_envs(cls):
        if not _has_mo:
            return []
        return [
            "deep-sea-treasure-v0",
            "deep-sea-treasure-concave-v0",
            "resource-gathering-v0",
            "fishwood-v0",
            "breakable-bottles-v0",
            "fruit-tree-v0",
            "water-reservoir-v0",
            "four-room-v0",
            "mo-mountaincar-v0",
            "mo-mountaincarcontinuous-v0",
            "mo-lunar-lander-v2",
            "minecart-v0",
            "mo-highway-v0",
            "mo-highway-fast-v0",
            "mo-supermario-v0",
            "mo-reacher-v4",
            "mo-hopper-v4",
            "mo-halfcheetah-v4",
        ]


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

    available_envs = MOGymWrapper.available_envs

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


class terminal_obs_reader(default_info_dict_reader):
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
        observation_spec (Composite): The observation spec of the gym env.
        backend (str, optional): the backend of the env. One of `"sb3"` for
            stable-baselines3 or `"gym"` for gym/gymnasium.

    .. note:: In general, this class should not be handled directly. It is
        created whenever a vectorized environment is placed within a :class:`GymWrapper`.

    """

    backend_key = {
        "sb3": "terminal_observation",
        "gym": "final_observation",
        "gymnasium": "final_obs",
    }
    backend_info_key = {
        "sb3": "terminal_info",
        "gym": "final_info",
        "gymnasium": "final_info",
    }

    def __init__(self, observation_spec: Composite, backend, name="final"):
        super().__init__()
        self.name = name
        self._obs_spec = observation_spec.clone()
        self.backend = backend
        self._final_validated = False

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
        if isinstance(obs, torch.Tensor):
            # Simplest case: there is one observation,
            # presented as a np.ndarray. The key should be pixels or observation.
            # We just write that value at its location in the tensor
            tensor[index] = obs.to(device=tensor.device)
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
        # TODO: This is a tad slow, we iterate over each sub-env and call spec.zero() at each step.
        #  In theory we could spare that whole thing but we need to run it once at the beginning if specs
        #  of the info reader are not passed as we need to observe the data to infer the spec.
        #  We should find a way to avoid this call altogether is no env is resetting.
        def replace_none(nparray):
            if not isinstance(nparray, np.ndarray) or nparray.dtype != np.dtype("O"):
                return nparray
            is_none = np.array([info is None for info in nparray])
            if is_none.any():
                # Then it is a final observation and we delegate the registration to the appropriate reader
                nz = (~is_none).nonzero()[0][0]
                zero_like = tree_map(lambda x: np.zeros_like(x), nparray[nz])
                for idx in is_none.nonzero()[0]:
                    nparray[idx] = zero_like
            return tree_map(lambda *x: np.stack(x), *nparray)

        info_dict = tree_map(replace_none, info_dict)
        # convert info_dict to a tensordict
        info_dict = TensorDict(info_dict)
        # get the terminal observation
        terminal_obs = info_dict.pop(self.backend_key[self.backend], None)
        # get the terminal info dict
        terminal_info = info_dict.pop(self.backend_info_key[self.backend], None)

        if terminal_info is None:
            terminal_info = {}

        super().__call__(info_dict, tensordict)
        if not self._final_validated:
            self.info_spec[self.name] = self._obs_spec.update(self.info_spec)
            self._final_validated = True

        final_info = terminal_info.copy()
        if terminal_obs is not None:
            final_info["observation"] = terminal_obs

        for key in self.info_spec[self.name].keys():
            spec = self.info_spec[self.name, key]

            final_obs_buffer = spec.zero()
            terminal_obs = final_info.get(key, None)
            if terminal_obs is not None:
                for i, obs in enumerate(terminal_obs):
                    # writes final_obs inplace with terminal_obs content
                    self._read_obs(obs, key, final_obs_buffer, index=i)
            tensordict.set((self.name, key), final_obs_buffer)
        return tensordict

    def reset(self):
        super().reset()
        self._final_validated = False


def _flip_info_tuple(info: tuple[dict]) -> dict[str, tuple]:
    # In Gym < 0.24, batched envs returned tuples of dict, and not dict of tuples.
    # We patch this by flipping the tuple -> dict order.
    info_example = set(info[0])
    for item in info[1:]:
        info_example = info_example.union(item)
    result = {}
    for key in info_example:
        result[key] = tuple(_info.get(key, None) for _info in info)
    return result
