# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import overload, TypeVar

import torch
from tensordict import is_tensor_collection
from tensordict.base import TensorDictBase

from torchrl.data.tensor_specs import DEVICE_TYPING, TensorSpec
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.transforms import Transform

T = TypeVar("T")


@overload
def _maybe_to_device(r: tuple, device: DEVICE_TYPING) -> tuple:
    ...


@overload
def _maybe_to_device(r: list, device: DEVICE_TYPING) -> list:
    ...


@overload
def _maybe_to_device(r: dict, device: DEVICE_TYPING) -> dict:
    ...


@overload
def _maybe_to_device(r: TensorDictBase, device: DEVICE_TYPING) -> TensorDictBase:
    ...


@overload
def _maybe_to_device(r: T, device: DEVICE_TYPING) -> T:
    ...


def _maybe_to_device(r, device):
    if isinstance(r, tuple):
        return tuple(_maybe_to_device(r_i, device) for r_i in r)
    if isinstance(r, list):
        return [_maybe_to_device(r_i, device) for r_i in r]
    if isinstance(r, dict):
        return {k: _maybe_to_device(v, device) for k, v in r.items()}
    if hasattr(r, "to"):
        return r.to(device)
    return r


@overload
def _maybe_clear_device(r: tuple) -> tuple:
    ...


@overload
def _maybe_clear_device(r: list) -> list:
    ...


@overload
def _maybe_clear_device(r: dict) -> dict:
    ...


@overload
def _maybe_clear_device(r: TensorDictBase) -> TensorDictBase:
    ...


@overload
def _maybe_clear_device(r: T) -> T:
    ...


def _maybe_clear_device(r):
    if isinstance(r, tuple):
        return tuple(_maybe_clear_device(r_i) for r_i in r)
    if isinstance(r, list):
        return [_maybe_clear_device(r_i) for r_i in r]
    if isinstance(r, dict):
        return {k: _maybe_clear_device(v) for k, v in r.items()}
    if is_tensor_collection(r) or isinstance(r, TensorSpec):
        r = r.clone()
        r = r.cpu().clear_device_()
    return r


def _map_input_output_device(func: Callable):
    """Decorator that maps inputs to CPU and outputs to the local device.

    This decorator ensures that:
    1. All inputs are moved to CPU before being sent to the remote Ray actor
    2. All outputs are moved to the local device (if set) after receiving from the Ray actor

    Args:
        func: The method to decorate

    Returns:
        The decorated method
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        args = _maybe_clear_device(args)
        kwargs = _maybe_clear_device(kwargs)
        r = func(self, *args, **kwargs)
        if hasattr(self, "_device"):
            if self._device is not None:
                r = _maybe_to_device(r, self._device)
            else:
                r = _maybe_clear_device(r)
        return r

    return wrapper


class RayTransform(Transform, ABC):
    """Base class for transforms that delegate operations to Ray remote actors.

    This class provides a framework for creating transforms that offload their operations
    to Ray remote actors, enabling:
    - Resource isolation and dedicated CPU/GPU allocation
    - Shared state across multiple environment instances
    - Distributed computation for expensive operations

    The class automatically handles:
    - Ray actor lifecycle management (creation, reuse, cleanup)
    - Device mapping between local client and remote actor contexts
    - Transparent method delegation with proper error handling
    - Local management of parent/container relationships

    Subclasses only need to implement `_create_actor()` to specify how their
    specific Ray actor should be created and configured.

    Args:
        num_cpus: CPU cores to allocate to the Ray actor
        num_gpus: GPU devices to allocate to the Ray actor
        device: Local device for tensor operations (client-side)
        actor_name: Optional name for actor reuse across instances
        **kwargs: Additional arguments passed to Transform base class

    Example:
        ```python
        class MyRayTransform(RayTransform):
            def _create_actor(self, **kwargs):
                RemoteClass = self._ray.remote(num_cpus=self._num_cpus)(MyClass)
                return RemoteClass.remote(**kwargs)
        ```
    """

    @property
    def _ray(self):
        ray = self.__dict__.get("_ray_val", None)
        if ray is not None:
            return ray
        # Import ray here to avoid requiring it as a dependency
        try:
            import ray
        except ImportError:
            raise ImportError(
                "Ray is required for RayTransform. Install with: pip install ray"
            )
        self.__dict__["_ray_val"] = ray
        return ray

    @_ray.setter
    def _ray(self, value):
        self.__dict__["_ray_val"] = value

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_ray_val", None)
        return state

    def __init__(
        self,
        *,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        device: DEVICE_TYPING | None = None,
        actor_name: str | None = None,
        **kwargs,
    ):
        """Initialize the RayTransform.

        Args:
            num_cpus: Number of CPUs to allocate to the Ray actor
            num_gpus: Number of GPUs to allocate to the Ray actor
            device: Local device for tensor operations
            actor_name: Name of the Ray actor (for reuse)
            **kwargs: Additional arguments passed to Transform
        """
        super().__init__(
            in_keys=kwargs.get("in_keys", []), out_keys=kwargs.get("out_keys", [])
        )

        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        self._device = device
        self._actor_name = actor_name
        self._actor = None

        # Initialize the Ray actor
        self._initialize_actor(**kwargs)

    def _initialize_actor(self, **kwargs):
        """Initialize the Ray actor, either by reusing existing or creating new."""
        # First attempt to get the actor if it already exists
        if self._actor_name is not None:
            try:
                existing_actor = self._ray.get_actor(self._actor_name)
                self._actor = existing_actor
                return
            except ValueError:
                pass

        # Create new actor
        self._actor = self._create_actor(**kwargs)

    @abstractmethod
    def _create_actor(self, **kwargs):
        """Create and return a Ray actor.

        This method should be implemented by subclasses to create the specific
        Ray actor needed for their operations.

        Args:
            **kwargs: Additional arguments for actor creation

        Returns:
            The created Ray actor
        """

    # Container management - handled locally, not delegated to remote actor
    def set_container(self, container: Transform | EnvBase) -> None:
        """Set the container for this transform. This is handled locally."""
        result = super().set_container(container)

        # After setting the container locally, provide batch size information to the remote actor
        # This ensures the remote actor has the right batch size for proper shape handling
        if self.parent is not None:
            parent_batch_size = self.parent.batch_size

            # Set the batch size directly on the remote actor to override its initialization
            self._ray.get(self._actor._set_attr.remote("batch_size", parent_batch_size))

            # Also disable validation on the remote actor since we'll handle consistency locally
            self._ray.get(self._actor._set_attr.remote("_validated", True))

        return result

    def reset_parent(self) -> None:
        """Reset the parent. This is handled locally."""
        return super().reset_parent()

    def clone(self):
        """Clone the transform."""
        # Use the parent's clone method to properly copy all Transform attributes
        new_instance = super().clone()
        # Then copy our specific Ray attributes to share the same actor
        new_instance._actor = self._actor
        new_instance._ray = self._ray
        new_instance._device = getattr(self, "_device", None)
        new_instance._num_cpus = self._num_cpus
        new_instance._num_gpus = self._num_gpus
        new_instance._actor_name = self._actor_name
        return new_instance

    def empty_cache(self):
        """Empty cache."""
        super().empty_cache()
        return self._ray.get(self._actor.empty_cache.remote())

    @property
    def container(self) -> EnvBase | None:
        """Returns the env containing the transform. This is handled locally."""
        return super().container

    @property
    def parent(self) -> EnvBase | None:
        """Returns the parent env of the transform. This is handled locally."""
        return super().parent

    @property
    def base_env(self):
        """Returns the base environment. This traverses the parent chain locally."""
        return (
            getattr(self.parent, "base_env", None) if self.parent is not None else None
        )

    def __repr__(self):
        """String representation."""
        try:
            if hasattr(self, "_actor") and self._actor is not None:
                return self._ray.get(self._actor.__repr__.remote())
            else:
                return f"{self.__class__.__name__}(actor=None)"
        except Exception:
            return f"{self.__class__.__name__}(actor={getattr(self, '_actor', 'None')})"

    # Properties - access via generic attribute getter since Ray doesn't support direct property access
    @property
    def device(self):
        """Get device property."""
        return getattr(self, "_device", None)

    @device.setter
    def device(self, value):
        """Set device property."""
        raise NotImplementedError(
            f"device setter is not implemented for {self.__class__.__name__}. Use transform.to() instead."
        )

    # TensorDictPrimer methods
    def init(self, tensordict: TensorDictBase | None):
        """Initialize."""
        return self._ray.get(self._actor.init.remote(tensordict))

    @_map_input_output_device
    def _reset_func(
        self, tensordict: TensorDictBase | None, tensordict_reset: TensorDictBase | None
    ) -> TensorDictBase | None:
        """Reset function."""
        result = self._ray.get(
            self._actor._reset_func.remote(tensordict, tensordict_reset)
        )
        return result

    @_map_input_output_device
    def _reset(
        self, tensordict: TensorDictBase | None, tensordict_reset: TensorDictBase | None
    ) -> TensorDictBase | None:
        """Reset method for TensorDictPrimer."""
        return self._ray.get(self._actor._reset.remote(tensordict, tensordict_reset))

    @_map_input_output_device
    def _reset_env_preprocess(
        self, tensordict: TensorDictBase | None
    ) -> TensorDictBase | None:
        """Reset environment preprocess - crucial for call_before_env_reset=True."""
        return self._ray.get(self._actor._reset_env_preprocess.remote(tensordict))

    def close(self):
        """Close the transform."""
        return self._ray.get(self._actor.close.remote())

    @_map_input_output_device
    def _apply_transform(self, obs: torch.Tensor | None) -> torch.Tensor | None:
        """Apply transform."""
        return self._ray.get(self._actor._apply_transform.remote(obs))

    @_map_input_output_device
    def _call(self, next_tensordict: TensorDictBase | None) -> TensorDictBase | None:
        """Call method."""
        return self._ray.get(self._actor._call.remote(next_tensordict))

    @_map_input_output_device
    def forward(self, tensordict: TensorDictBase | None) -> TensorDictBase | None:
        """Forward pass."""
        return self._ray.get(self._actor.forward.remote(tensordict))

    @_map_input_output_device
    def _inv_apply_transform(
        self, state: TensorDictBase | None
    ) -> TensorDictBase | None:
        """Inverse apply transform."""
        return self._ray.get(self._actor._inv_apply_transform.remote(state))

    @_map_input_output_device
    def _inv_call(self, tensordict: TensorDictBase | None) -> TensorDictBase | None:
        """Inverse call."""
        return self._ray.get(self._actor._inv_call.remote(tensordict))

    @_map_input_output_device
    def inv(self, tensordict: TensorDictBase | None) -> TensorDictBase | None:
        """Inverse."""
        return self._ray.get(self._actor.inv.remote(tensordict))

    @_map_input_output_device
    def _step(
        self, tensordict: TensorDictBase | None, next_tensordict: TensorDictBase | None
    ) -> TensorDictBase | None:
        """Step method."""
        return self._ray.get(self._actor._step.remote(tensordict, next_tensordict))

    def transform_env_device(self, device):
        """Transform environment device."""
        return self._ray.get(self._actor.transform_env_device.remote(device))

    def transform_env_batch_size(self, batch_size):
        """Transform environment batch size."""
        return self._ray.get(self._actor.transform_env_batch_size.remote(batch_size))

    @_map_input_output_device
    def transform_output_spec(self, output_spec):
        """Transform output spec."""
        return self._ray.get(self._actor.transform_output_spec.remote(output_spec))

    @_map_input_output_device
    def transform_input_spec(self, input_spec):
        """Transform input spec."""
        return self._ray.get(self._actor.transform_input_spec.remote(input_spec))

    @_map_input_output_device
    def transform_observation_spec(self, observation_spec):
        """Transform observation spec."""
        return self._ray.get(
            self._actor.transform_observation_spec.remote(observation_spec)
        )

    @_map_input_output_device
    def transform_reward_spec(self, reward_spec):
        """Transform reward spec."""
        return self._ray.get(self._actor.transform_reward_spec.remote(reward_spec))

    @_map_input_output_device
    def transform_done_spec(self, done_spec):
        """Transform done spec."""
        return self._ray.get(self._actor.transform_done_spec.remote(done_spec))

    @_map_input_output_device
    def transform_action_spec(self, action_spec):
        """Transform action spec."""
        return self._ray.get(self._actor.transform_action_spec.remote(action_spec))

    @_map_input_output_device
    def transform_state_spec(self, state_spec):
        """Transform state spec."""
        return self._ray.get(self._actor.transform_state_spec.remote(state_spec))

    def dump(self, **kwargs):
        """Dump method."""
        return self._ray.get(self._actor.dump.remote(**kwargs))

    def set_missing_tolerance(self, mode=False):
        """Set missing tolerance."""
        return self._ray.get(self._actor.set_missing_tolerance.remote(mode))

    @property
    def missing_tolerance(self):
        """Get missing tolerance."""
        return self._ray.get(self._actor.missing_tolerance.remote())

    @property
    def primers(self):
        """Get primers."""
        return self._ray.get(self._actor.__getattribute__.remote("primers"))

    @primers.setter
    def primers(self, value):
        """Set primers."""
        self.__dict__["_primers"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor._set_attr.remote("primers", value))

    def to(self, *args, **kwargs):
        """Move to device."""
        # Parse the device from args/kwargs like torch does
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self._device = device
        # Don't delegate to remote actor - just register device locally
        return super().to(*args, **kwargs)

    # Properties that should be accessed from the remote actor
    @property
    def in_keys(self):
        """Get in_keys property."""
        return self._ray.get(self._actor.__getattribute__.remote("in_keys"))

    @in_keys.setter
    def in_keys(self, value):
        """Set in_keys property."""
        self.__dict__["_in_keys"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor._set_attr.remote("in_keys", value))

    @property
    def out_keys(self):
        """Get out_keys property."""
        return self._ray.get(self._actor.__getattribute__.remote("out_keys"))

    @out_keys.setter
    def out_keys(self, value):
        """Set out_keys property."""
        self.__dict__["_out_keys"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor._set_attr.remote("out_keys", value))

    @property
    def in_keys_inv(self):
        """Get in_keys_inv property."""
        return self._ray.get(self._actor.__getattribute__.remote("in_keys_inv"))

    @in_keys_inv.setter
    def in_keys_inv(self, value):
        """Set in_keys_inv property."""
        self.__dict__["_in_keys_inv"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor._set_attr.remote("in_keys_inv", value))

    @property
    def out_keys_inv(self):
        """Get out_keys_inv property."""
        return self._ray.get(self._actor.__getattribute__.remote("out_keys_inv"))

    @out_keys_inv.setter
    def out_keys_inv(self, value):
        """Set out_keys_inv property."""
        self.__dict__["_out_keys_inv"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor._set_attr.remote("out_keys_inv", value))

    # Generic attribute access for any remaining attributes
    def __getattr__(self, name):
        """Get attribute from the remote actor.

        This method should only be called for attributes that don't exist locally
        and should be delegated to the remote actor (inward-facing).

        Outward-facing attributes (parent, container, base_env, etc.) should be handled
        by the Transform base class and never reach this method.
        """
        # Upward-facing attributes that should never be delegated to remote actor
        upward_attrs = {"parent", "container", "base_env", "_parent", "_container"}

        if name in upward_attrs:
            # These should be handled by the local Transform implementation
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Only delegate to remote actor if we're sure this is an inward-facing attribute
        # and the actor is properly initialized
        actor = self.__dict__.get("_actor", None)
        if actor is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Only delegate specific DataLoadingPrimer methods/attributes to the remote actor
        # This is a whitelist approach to be more conservative
        delegated_methods = {
            # DataLoadingPrimer methods that should be called on the remote actor
            "_call",
            "_reset",
            "_inv_call",
            "forward",
            "inv",
            "_apply_transform",
            "_inv_apply_transform",
            "_reset_func",
            "init",  # TensorDictPrimer specific methods
            "primers",
            "dataloader",  # Properties
            # Add other specific methods that should be delegated as needed
        }

        if name in delegated_methods:
            try:
                result = self._ray.get(getattr(actor, name).remote())
                # If it's a method, wrap it to make remote calls
                if callable(result):
                    return lambda *args, **kwargs: self._ray.get(
                        getattr(actor, name).remote(*args, **kwargs)
                    )
                return result
            except (AttributeError, TypeError):
                # If that fails, it might be a callable method
                try:
                    remote_method = getattr(actor, name)
                    return lambda *args, **kwargs: self._ray.get(
                        remote_method.remote(*args, **kwargs)
                    )
                except AttributeError:
                    pass

        # If not in our whitelist, don't delegate to remote actor
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        """Set attribute on the remote actor or locally."""
        # Local attributes that should never be delegated to remote actor
        local_attrs = {
            "_actor",
            "_ray",
            "_parent",
            "_container",
            "_missing_tolerance",
            "_in_keys",
            "_out_keys",
            "_in_keys_inv",
            "_out_keys_inv",
            "in_keys",
            "out_keys",
            "in_keys_inv",
            "out_keys_inv",
            "_modules",
            "_parameters",
            "_buffers",
            "_device",
        }

        if name in local_attrs:
            super().__setattr__(name, value)
        else:
            # Try to set on remote actor for other attributes
            try:
                if hasattr(self, "_actor") and self._actor is not None:
                    self._ray.get(self._actor._set_attr.remote(name, value))
                else:
                    super().__setattr__(name, value)
            except Exception:
                # Fall back to local setting for attributes that can't be set remotely
                super().__setattr__(name, value)


class _RayServiceMetaClass(type):
    """Metaclass that enables dynamic class selection based on use_ray_service parameter.

    This metaclass allows a class to dynamically return either itself or a Ray-based
    alternative class when instantiated with use_ray_service=True.

    Usage:
        >>> class MyRayClass():
        ...     def __init__(self, **kwargs):
        ...         ...
        ...
        >>> class MyClass(metaclass=_RayServiceMetaClass):
        ...     _RayServiceClass = MyRayClass
        ...
        ...     def __init__(self, use_ray_service=False, **kwargs):
        ...         # Regular implementation
        ...         pass
        ...
        >>> # Returns MyClass instance
        >>> obj1 = MyClass(use_ray_service=False)
        >>>
        >>> # Returns MyRayClass instance
        >>> obj2 = MyClass(use_ray_service=True)
    """

    def __call__(cls, *args, use_ray_service=False, **kwargs):
        if use_ray_service:
            if not hasattr(cls, "_RayServiceClass"):
                raise ValueError(
                    f"Class {cls.__name__} does not have a _RayServiceClass attribute"
                )
            return cls._RayServiceClass(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)
