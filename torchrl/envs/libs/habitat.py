# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools

import torch

from torchrl.data import DEVICE_TYPING
from torchrl.envs import EnvBase
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import classproperty

IMPORT_ERR = None
try:
    import habitat
    import habitat.gym  # noqa

    _has_habitat = True
except ImportError as err:
    _has_habitat = False
    IMPORT_ERR = err


def _wrap_import_error(fun):
    @functools.wraps(fun)
    def new_fun(*args, **kwargs):
        if not _has_habitat:
            raise ImportError(
                "Habitat could not be loaded. Consider installing "
                "it or solving the import bugs (see attached error message). "
                "Refer to TorchRL's knowledge base in the documentation to "
                "debug habitat installation."
            ) from IMPORT_ERR
        return fun(*args, **kwargs)

    return new_fun


@_wrap_import_error
def _get_available_envs():
    for env in GymEnv.available_envs:
        if env.startswith("Habitat"):
            yield env


class HabitatEnv(GymEnv):
    """A wrapper for habitat envs.

    This class currently serves as placeholder and compatibility security.
    It behaves exactly like the GymEnv wrapper.

    """

    @_wrap_import_error
    def __init__(self, env_name, disable_env_checker=None, **kwargs):
        device_num = torch.device(kwargs.pop("device", 0)).index
        kwargs["override_options"] = [
            f"habitat.simulator.habitat_sim_v0.gpu_device_id={device_num}",
        ]
        super().__init__(
            env_name=env_name, disable_env_checker=disable_env_checker, **kwargs
        )

    @classproperty
    def available_envs(cls):
        yield from _get_available_envs()

    def _build_gym_env(self, env, pixels_only):
        if self.from_pixels:
            env.reset()
        return super()._build_gym_env(env, pixels_only)

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("The device must be of type cuda for Habitat.")
        device_num = device.index
        kwargs = {"override_options": []}
        for arg in self._constructor_kwargs.get("override_options", []):
            if arg.startswith("habitat.simulator.habitat_sim_v0.gpu_device_id"):
                arg = f"habitat.simulator.habitat_sim_v0.gpu_device_id={device_num}"
                kwargs["override_options"].append(arg)
            else:
                kwargs["override_options"].append(arg)

        self._env.close()
        del self._env
        self.rebuild_with_kwargs(**kwargs)
        return super().to(device)
