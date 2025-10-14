# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from contextlib import nullcontext
from typing import overload

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.transforms.ray_service import _RayServiceMetaClass, RayTransform
from torchrl.envs.transforms.transforms import Transform


__all__ = ["ModuleTransform", "RayModuleTransform"]


class RayModuleTransform(RayTransform):
    """Ray-based ModuleTransform for distributed processing.

    This transform creates a Ray actor that wraps a ModuleTransform,
    allowing module execution in a separate Ray worker process.
    """

    def _create_actor(self, **kwargs):
        return self._ray.remote(ModuleTransform).remote(**kwargs)

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
        actor_name (str, optional): The name of the actor to use. Default is `None`. If an actor name is provided and
            an actor with this name already exists, the existing actor will be used.

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
        use_ray_service: bool = False,
        actor_name: str | None = None,
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
