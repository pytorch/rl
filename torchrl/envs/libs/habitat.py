# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import importlib.util

import torch
from torchrl._utils import _make_ordinal_device
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.libs.gym import _GymAsyncMeta, GymEnv, set_gym_backend
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


class _HabitatMeta(_GymAsyncMeta):
    """Metaclass for HabitatEnv that returns a lazy ParallelEnv when num_workers > 1."""

    def __call__(cls, *args, num_workers: int | None = None, **kwargs):
        # Extract num_workers from explicit kwarg or kwargs dict
        if num_workers is None:
            num_workers = kwargs.pop("num_workers", 1)
        else:
            kwargs.pop("num_workers", None)

        num_workers = int(num_workers)
        if getattr(cls, "__name__", None) == "HabitatEnv" and num_workers > 1:
            from torchrl.envs import ParallelEnv

            env_name = args[0] if len(args) >= 1 else kwargs.get("env_name")
            env_kwargs = {k: v for k, v in kwargs.items() if k != "env_name"}

            # Extract device - can be a single device or a list of devices
            device = env_kwargs.pop("device", 0)

            # Handle device as list for per-worker GPU assignment
            if isinstance(device, (list, tuple)):
                if len(device) != num_workers:
                    raise ValueError(
                        f"Length of device list ({len(device)}) must match "
                        f"num_workers ({num_workers})"
                    )
                devices = device
            else:
                devices = [device] * num_workers

            # We intentionally don't use EnvCreator here to avoid instantiating
            # a HabitatEnv in the local process, which can be expensive.
            # Create per-worker partials with different devices.
            make_envs = [
                functools.partial(cls, env_name, num_workers=1, device=d, **env_kwargs)
                for d in devices
            ]
            return ParallelEnv(num_workers, make_envs)

        return super().__call__(*args, **kwargs)


class HabitatEnv(GymEnv, metaclass=_HabitatMeta):
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
        device (torch.device or list of torch.device, optional): if provided, the device
            on which the simulation will occur. When ``num_workers > 1``, this can be a
            list of devices (one per worker) to distribute environments across multiple
            GPUs. Defaults to ``torch.device("cuda:0")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Should match the leading dimensions of all observations, done states,
            rewards, actions and infos.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.
        num_workers (int, optional): if provided and greater than 1, a
            :class:`torchrl.envs.ParallelEnv` will be instantiated with
            ``num_workers`` copies of ``HabitatEnv``. Defaults to ``1``.

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
