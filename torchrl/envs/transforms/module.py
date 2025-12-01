# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import overload, TYPE_CHECKING

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl._utils import logger as torchrl_logger

from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.transforms.ray_service import _RayServiceMetaClass, RayTransform
from torchrl.envs.transforms.transforms import Transform

if TYPE_CHECKING:
    from torchrl.weight_update import WeightSyncScheme

__all__ = ["ModuleTransform", "RayModuleTransform"]


class RayModuleTransform(RayTransform):
    """Ray-based ModuleTransform for distributed processing.

    This transform creates a Ray actor that wraps a ModuleTransform,
    allowing module execution in a separate Ray worker process.

    Args:
        weight_sync_scheme: Optional weight synchronization scheme for updating
            the module's weights from a parent collector. When provided, the scheme
            is initialized on the receiver side (the Ray actor) and can receive
            weight updates via torch.distributed.
        **kwargs: Additional arguments passed to RayTransform and ModuleTransform.

    Example:
        >>> from torchrl.weight_update import RayModuleTransformScheme
        >>> scheme = RayModuleTransformScheme()
        >>> transform = RayModuleTransform(module=my_module, weight_sync_scheme=scheme)
        >>> # The scheme can then be registered with a collector for weight updates
    """

    def __init__(self, *, weight_sync_scheme=None, **kwargs):
        self._weight_sync_scheme = weight_sync_scheme
        super().__init__(**kwargs)

        # After actor is created, initialize the scheme on the receiver side
        if weight_sync_scheme is not None:
            # Store transform reference in the scheme for sender initialization
            weight_sync_scheme._set_transform(self)

            weight_sync_scheme.init_on_sender()

            # Initialize receiver in the actor
            torchrl_logger.debug(
                "Setting up weight sync scheme on sender -- sender will do the remote call"
            )
            weight_sync_scheme.connect()

    @property
    def in_keys(self):
        return self._ray.get(self._actor._getattr.remote("in_keys"))

    @property
    def out_keys(self):
        return self._ray.get(self._actor._getattr.remote("out_keys"))

    def _create_actor(self, **kwargs):
        import ray

        remote = self._ray.remote(ModuleTransform)
        ray_kwargs = {}
        num_gpus = self._num_gpus
        if num_gpus is not None:
            ray_kwargs["num_gpus"] = num_gpus
        num_cpus = self._num_cpus
        if num_cpus is not None:
            ray_kwargs["num_cpus"] = num_cpus
        actor_name = self._actor_name
        if actor_name is not None:
            ray_kwargs["name"] = actor_name
        if ray_kwargs:
            remote = remote.options(**ray_kwargs)
        actor = remote.remote(**kwargs)
        # wait till the actor is ready
        ray.get(actor._ready.remote())
        return actor

    @overload
    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        ...

    @overload
    def update_weights(self, params: TensorDictBase) -> None:
        ...

    def update_weights(self, *args, **kwargs) -> None:
        import ray

        if self._update_weights_method == "tensordict":
            try:
                td = kwargs.get("params", args[0])
            except IndexError:
                raise ValueError("params must be provided")
            return ray.get(self._actor._update_weights_tensordict.remote(params=td))
        elif self._update_weights_method == "state_dict":
            try:
                state_dict = kwargs.get("state_dict", args[0])
            except IndexError:
                raise ValueError("state_dict must be provided")
            return ray.get(
                self._actor._update_weights_state_dict.remote(state_dict=state_dict)
            )
        else:
            raise ValueError(
                f"Invalid update_weights_method: {self._update_weights_method}"
            )


class ModuleTransform(Transform, metaclass=_RayServiceMetaClass):
    """A transform that wraps a module.

    Keyword Args:
        module (TensorDictModuleBase): The module to wrap. Exclusive with `module_factory`. At least one of `module` or `module_factory` must be provided.
        module_factory (Callable[[], TensorDictModuleBase]): The factory to create the module. Exclusive with `module`. At least one of `module` or `module_factory` must be provided.
        no_grad (bool, optional): Whether to use gradient computation. Default is `False`.
        inverse (bool, optional): Whether to use the inverse of the module. Default is `False`.
        device (torch.device, optional): The device to use. Default is `None`.
        use_ray_service (bool, optional): Whether to use Ray service. Default is `False`.
        num_gpus (int, optional): The number of GPUs to use if using Ray. Default is `None`.
        num_cpus (int, optional): The number of CPUs to use if using Ray. Default is `None`.
        actor_name (str, optional): The name of the actor to use. Default is `None`. If an actor name is provided and
            an actor with this name already exists, the existing actor will be used.
        observation_spec_transform (TensorSpec or Callable[[TensorSpec], TensorSpec]): either a new spec for the observation
            after it has been transformed by the module, or a function that modifies the existing spec.
            Defaults to `None` (observation specs remain unchanged).
        done_spec_transform (TensorSpec or Callable[[TensorSpec], TensorSpec]): either a new spec for the done
            after it has been transformed by the module, or a function that modifies the existing spec.
            Defaults to `None` (done specs remain unchanged).
        reward_spec_transform (TensorSpec or Callable[[TensorSpec], TensorSpec]): either a new spec for the reward
            after it has been transformed by the module, or a function that modifies the existing spec.
            Defaults to `None` (reward specs remain unchanged).
        state_spec_transform (TensorSpec or Callable[[TensorSpec], TensorSpec]): either a new spec for the state
            after it has been transformed by the module, or a function that modifies the existing spec.
            Defaults to `None` (state specs remain unchanged).
        action_spec_transform (TensorSpec or Callable[[TensorSpec], TensorSpec]): either a new spec for the action
            after it has been transformed by the module, or a function that modifies the existing spec.
            Defaults to `None` (action specs remain unchanged).
    """

    _RayServiceClass = RayModuleTransform

    def __init__(
        self,
        *,
        module: TensorDictModuleBase | None = None,
        module_factory: Callable[[], TensorDictModuleBase] | None = None,
        no_grad: bool = False,
        inverse: bool = False,
        device: torch.device | None = None,
        use_ray_service: bool = False,  # noqa
        actor_name: str | None = None,  # noqa
        num_gpus: int | None = None,
        num_cpus: int | None = None,
        observation_spec_transform: TensorSpec
        | Callable[[TensorSpec], TensorSpec]
        | None = None,
        action_spec_transform: TensorSpec
        | Callable[[TensorSpec], TensorSpec]
        | None = None,
        reward_spec_transform: TensorSpec
        | Callable[[TensorSpec], TensorSpec]
        | None = None,
        done_spec_transform: TensorSpec
        | Callable[[TensorSpec], TensorSpec]
        | None = None,
        state_spec_transform: TensorSpec
        | Callable[[TensorSpec], TensorSpec]
        | None = None,
    ):
        super().__init__()
        if module is None and module_factory is None:
            raise ValueError(
                "At least one of `module` or `module_factory` must be provided."
            )
        if module is not None and module_factory is not None:
            raise ValueError(
                "Only one of `module` or `module_factory` must be provided."
            )
        self.module = module if module is not None else module_factory()
        self.no_grad = no_grad
        self.inverse = inverse
        self.device = device
        self.observation_spec_transform = observation_spec_transform
        self.action_spec_transform = action_spec_transform
        self.reward_spec_transform = reward_spec_transform
        self.done_spec_transform = done_spec_transform
        self.state_spec_transform = state_spec_transform

    @property
    def in_keys(self) -> list[str]:
        return self._in_keys()

    def _in_keys(self):
        return self.module.in_keys if not self.inverse else []

    @in_keys.setter
    def in_keys(self, value: list[str] | None):
        if value is not None:
            raise RuntimeError(f"in_keys {value} cannot be set for ModuleTransform")

    @property
    def out_keys(self) -> list[str]:
        return self._out_keys()

    def _out_keys(self):
        return self.module.out_keys if not self.inverse else []

    @property
    def in_keys_inv(self) -> list[str]:
        return self._in_keys_inv()

    def _in_keys_inv(self):
        return self.module.out_keys if self.inverse else []

    @in_keys_inv.setter
    def in_keys_inv(self, value: list[str]):
        if value is not None:
            raise RuntimeError(f"in_keys_inv {value} cannot be set for ModuleTransform")

    @property
    def out_keys_inv(self) -> list[str]:
        return self._out_keys_inv()

    def _out_keys_inv(self):
        return self.module.in_keys if self.inverse else []

    @out_keys_inv.setter
    def out_keys_inv(self, value: list[str] | None):
        if value is not None:
            raise RuntimeError(
                f"out_keys_inv {value} cannot be set for ModuleTransform"
            )

    @out_keys.setter
    def out_keys(self, value: list[str] | None):
        if value is not None:
            raise RuntimeError(f"out_keys {value} cannot be set for ModuleTransform")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.inverse:
            return tensordict
        with torch.no_grad() if self.no_grad else nullcontext():
            with (
                tensordict.to(self.device)
                if self.device is not None
                else nullcontext(tensordict)
            ) as td:
                return self.module(td)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.inverse:
            return tensordict
        with torch.no_grad() if self.no_grad else nullcontext():
            with (
                tensordict.to(self.device)
                if self.device is not None
                else nullcontext(tensordict)
            ) as td:
                return self.module(td)

    def _update_weights_tensordict(self, params: TensorDictBase) -> None:
        params.to_module(self.module)

    def _update_weights_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state_dict)

    def _init_weight_sync_scheme(self, scheme: WeightSyncScheme, model_id: str) -> None:
        """Initialize weight sync scheme on the receiver side (called in Ray actor).

        This method is called by RayModuleTransform after the actor is created
        to set up the receiver side of the weight synchronization scheme.

        Args:
            scheme: The weight sync scheme instance (e.g., RayModuleTransformScheme).
            model_id: Identifier for the model being synchronized.
        """
        torchrl_logger.debug(f"Initializing weight sync scheme for {model_id=}")
        scheme.init_on_receiver(model_id=model_id, context=self)
        torchrl_logger.debug(f"Setup weight sync scheme for {model_id=}")
        scheme._setup_connection_and_weights_on_receiver_impl()
        self._weight_sync_scheme = scheme

    def _receive_weights_scheme(self):
        self._weight_sync_scheme.receive()

    def _debug_scheme(self) -> dict:
        """Debug method to inspect scheme state on the receiver."""
        if not hasattr(self, "_weight_sync_scheme") or self._weight_sync_scheme is None:
            return {"error": "No scheme"}
        s = self._weight_sync_scheme
        return {
            "initialized_on_receiver": getattr(s, "_initialized_on_receiver", False),
            "initialized_on_sender": getattr(s, "_initialized_on_sender", False),
            "synchronized_on_receiver": getattr(s, "synchronized_on_receiver", False),
            "synchronized_on_sender": getattr(s, "synchronized_on_sender", False),
            "dist_initialized": getattr(s, "_dist_initialized", False),
            "has_model": s.model is not None if hasattr(s, "model") else False,
        }

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if self.observation_spec_transform is not None:
            if isinstance(self.observation_spec_transform, TensorSpec):
                return self.observation_spec_transform
            else:
                return self.observation_spec_transform(observation_spec)
        return observation_spec

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        if self.action_spec_transform is not None:
            if isinstance(self.action_spec_transform, TensorSpec):
                return self.action_spec_transform
            else:
                return self.action_spec_transform(action_spec)
        return action_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if self.reward_spec_transform is not None:
            if isinstance(self.reward_spec_transform, TensorSpec):
                return self.reward_spec_transform
            else:
                return self.reward_spec_transform(reward_spec)
        return reward_spec

    def transform_done_spec(self, done_spec: TensorSpec) -> TensorSpec:
        if self.done_spec_transform is not None:
            if isinstance(self.done_spec_transform, TensorSpec):
                return self.done_spec_transform
            else:
                return self.done_spec_transform(done_spec)
        return done_spec

    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        if self.state_spec_transform is not None:
            if isinstance(self.state_spec_transform, TensorSpec):
                return self.state_spec_transform
            else:
                return self.state_spec_transform(state_spec)
        return state_spec
