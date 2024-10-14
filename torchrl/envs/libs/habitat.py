# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import importlib.util

import torch
from torchrl._utils import _make_ordinal_device
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import _classproperty

_has_habitat = importlib.util.find_spec("habitat") is not None


def _wrap_import_error(fun):
    @functools.wraps(fun)
    def new_fun(*args, **kwargs):
        if not _has_habitat:
            raise ImportError(
                "Habitat could not be loaded. Consider installing "
                "it or solving the import bugs (see attached error message). "
                "Refer to TorchRL's knowledge base in the documentation to "
                "debug habitat installation."
            )
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

    Doc: https://aihabitat.org/docs/

    GitHub: https://github.com/facebookresearch/habitat-lab

    URL: https://aihabitat.org/habitat3/

    Paper: https://ai.meta.com/static-resource/habitat3

    Args:
        env_name (str): The environment to execute.
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
        device (torch.device, optional): if provided, the device on which the simulation
            will occur. Defaults to ``torch.device("cuda:0")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Should match the leading dimensions of all observations, done states,
            rewards, actions and infos.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs (List[str]): a list of environments to build.

    Examples:
        >>> from torchrl.envs import HabitatEnv
        >>> env = HabitatEnv("HabitatRenderPick-v0", from_pixels=True)
        >>> env.rollout(3)

    """

    @_wrap_import_error
    @set_gym_backend("gym")
    def __init__(self, env_name, **kwargs):
        import habitat  # noqa
        import habitat.gym  # noqa

        device_num = torch.device(kwargs.pop("device", 0)).index
        kwargs["override_options"] = [
            f"habitat.simulator.habitat_sim_v0.gpu_device_id={device_num}",
            "habitat.simulator.concur_render=False",
        ]
        super().__init__(env_name=env_name, **kwargs)

    @_classproperty
    def available_envs(cls):
        if not _has_habitat:
            return []
        return list(_get_available_envs())

    def _build_gym_env(self, env, pixels_only):
        if self.from_pixels:
            env.reset()
        return super()._build_gym_env(env, pixels_only)

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        device = _make_ordinal_device(torch.device(device))
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
