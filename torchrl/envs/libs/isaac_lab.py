# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Literal

import torch
from tensordict import NestedKey
from torchrl.data.tensor_specs import Bounded, Unbounded
from torchrl.envs.libs.gym import GymWrapper

_has_isaaclab = importlib.util.find_spec("isaaclab") is not None
_has_isaaclab_newton = importlib.util.find_spec("isaaclab_newton") is not None
_has_isaaclab_ov = importlib.util.find_spec("isaaclab_ov") is not None


class IsaacLabWrapper(GymWrapper):
    """A wrapper for IsaacLab environments.

    Args:
        env (scripts_isaaclab.envs.ManagerBasedRLEnv or equivalent): the environment instance to wrap.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.
        native_autoreset (bool, optional): if ``True``, keeps Isaac Lab's native
            auto-reset observations in the collector hot path and avoids the
            synthetic reset call in :meth:`~torchrl.envs.EnvBase.step_and_maybe_reset`.
            The terminal ``"next"`` observation remains unavailable and is
            marked with ``NaN``; the native reset observation is cloned into
            the next root observation.
            Defaults to ``False``.
        from_tiled_camera (bool, optional): if ``True``, reads pixels from an
            Isaac Lab tiled camera sensor and writes them under ``pixels_key``.
            This is the recommended headless rendering path. Defaults to
            ``False``.
        tiled_camera_name (str, optional): Name of the sensor in
            ``env.scene.sensors``. Defaults to ``"tiled_camera"``.
        tiled_camera_data_type (str, optional): Camera data type to read.
            Defaults to ``"rgb"``.
        pixels_key (NestedKey, optional): TensorDict key where pixels are
            written. Defaults to ``"pixels"``.
        pixels_dtype (torch.dtype, optional): dtype used for the output pixels.
            If ``torch.uint8`` is requested and the camera returns floating
            point data, values are scaled from ``[0, 1]`` to ``[0, 255]``.
            Defaults to ``torch.uint8``.
        pixels_channels (int, optional): Number of channels to keep from the
            camera output. Defaults to ``3``.

    For other arguments, see the :class:`torchrl.envs.GymWrapper` documentation.

    Refer to `the Isaac Lab doc for installation instructions <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>`_.

    Example:
        >>> # This code block ensures that the Isaac app is started in headless mode
        >>> from scripts_isaaclab.app import AppLauncher
        >>> import argparse

        >>> parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
        >>> AppLauncher.add_app_launcher_args(parser)
        >>> args_cli, hydra_args = parser.parse_known_args(["--headless"])
        >>> app_launcher = AppLauncher(args_cli)

        >>> # Imports and env
        >>> import gymnasium as gym
        >>> import isaaclab_tasks  # noqa: F401
        >>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
        >>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

        >>> env = gym.make("Isaac-Ant-v0", cfg=AntEnvCfg())
        >>> env = IsaacLabWrapper(env)

    """

    def __init__(
        self,
        env: isaaclab.envs.ManagerBasedRLEnv,  # noqa: F821
        *,
        categorical_action_encoding: bool = False,
        allow_done_after_reset: bool = True,
        convert_actions_to_numpy: bool = False,
        device: torch.device | None = None,
        native_autoreset: bool = False,
        from_tiled_camera: bool = False,
        tiled_camera_name: str = "tiled_camera",
        tiled_camera_data_type: str = "rgb",
        pixels_key: NestedKey = "pixels",
        pixels_dtype: torch.dtype | None = torch.uint8,
        pixels_channels: int | None = 3,
        **kwargs,
    ):
        self.from_tiled_camera = from_tiled_camera
        self.tiled_camera_name = tiled_camera_name
        self.tiled_camera_data_type = tiled_camera_data_type
        self.pixels_key = pixels_key
        self.pixels_dtype = pixels_dtype
        self.pixels_channels = pixels_channels
        if device is None:
            device = torch.device("cuda:0")
        super().__init__(
            env,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs,
        )

    @staticmethod
    def add_tiled_camera_config(
        env_cfg,
        *,
        sensor_name: str = "tiled_camera",
        data_type: str = "rgb",
        renderer_backend: Literal["isaac_rtx", "newton_warp", "ovrtx"] | None = None,
        renderer_cfg: object | None = None,
        width: int = 320,
        height: int = 240,
        pos: tuple[float, float, float] = (-7.0, 0.0, 3.0),
        rot: tuple[float, float, float, float] = (0.9945, 0.0, 0.1045, 0.0),
        convention: Literal["opengl", "ros", "world"] = "world",
        focal_length: float = 24.0,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.955,
        clipping_range: tuple[float, float] = (0.1, 100.0),
        render_interval: int | None = None,
    ):
        """Attach an Isaac Lab :class:`~isaaclab.sensors.TiledCameraCfg`.

        This helper mutates an Isaac Lab environment config before the env is
        instantiated, making headless RGB capture available through
        :class:`IsaacLabWrapper` with ``from_tiled_camera=True``.

        Args:
            env_cfg: Isaac Lab environment config to mutate.
            sensor_name (str, optional): Name used in ``env.scene.sensors``.
                Defaults to ``"tiled_camera"``.
            data_type (str, optional): Camera output data type. Defaults to
                ``"rgb"``.
            renderer_backend (str, optional): Renderer backend to use for the
                tiled camera. ``"isaac_rtx"`` keeps Isaac Lab's default RTX
                renderer, ``"newton_warp"`` uses the Isaac Lab Newton Warp
                renderer, and ``"ovrtx"`` uses the OVRTX renderer. Defaults to
                ``None``, which keeps the Isaac Lab default.
            renderer_cfg (object, optional): Explicit renderer configuration
                object to pass to :class:`~isaaclab.sensors.TiledCameraCfg`.
                This cannot be combined with ``renderer_backend``.
            width (int, optional): Camera image width. Defaults to ``320``.
            height (int, optional): Camera image height. Defaults to ``240``.
            pos (tuple of float, optional): Camera position offset.
            rot (tuple of float, optional): Camera orientation offset.
            convention (str, optional): Orientation convention. Defaults to
                ``"world"``.
            focal_length (float, optional): Pinhole camera focal length.
            focus_distance (float, optional): Pinhole camera focus distance.
            horizontal_aperture (float, optional): Pinhole camera aperture.
            clipping_range (tuple of float, optional): Near and far clipping
                planes.
            render_interval (int, optional): Simulation render interval. If
                omitted, the existing config value is kept.

        Returns:
            The mutated ``env_cfg``.

        Examples:
            >>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
            >>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
            >>> cfg = IsaacLabWrapper.add_tiled_camera_config(AntEnvCfg())
        """
        if not _has_isaaclab:
            raise ImportError("Isaac Lab is required to add a tiled camera config.")
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg

        if renderer_backend is not None and renderer_cfg is not None:
            raise ValueError(
                "Only one of renderer_backend or renderer_cfg can be provided."
            )
        if renderer_backend == "newton_warp":
            if not _has_isaaclab_newton:
                raise ImportError(
                    "Isaac Lab Newton is required to use "
                    "renderer_backend='newton_warp'."
                )
            from isaaclab_newton.renderers import NewtonWarpRendererCfg

            renderer_cfg = NewtonWarpRendererCfg()
        elif renderer_backend == "ovrtx":
            if not _has_isaaclab_ov:
                raise ImportError(
                    "Isaac Lab OVRTX is required to use renderer_backend='ovrtx'."
                )
            from isaaclab_ov.renderers import OVRTXRendererCfg

            renderer_cfg = OVRTXRendererCfg()

        camera_kwargs = {}
        if renderer_cfg is not None:
            camera_kwargs["renderer_cfg"] = renderer_cfg
        camera_cfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=pos,
                rot=rot,
                convention=convention,
            ),
            data_types=[data_type],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=focus_distance,
                horizontal_aperture=horizontal_aperture,
                clipping_range=clipping_range,
            ),
            width=width,
            height=height,
            **camera_kwargs,
        )
        setattr(env_cfg.scene, sensor_name, camera_cfg)
        if render_interval is not None:
            env_cfg.sim.render_interval = render_interval
        return env_cfg

    def seed(self, seed: int | None):
        self._set_seed(seed)

    def _make_specs(self, env, batch_size=None) -> None:
        super()._make_specs(env, batch_size=batch_size)
        if not self.from_tiled_camera:
            return
        camera = self._get_tiled_camera()
        cfg = camera.cfg
        channels = self.pixels_channels
        if channels is None:
            if self.tiled_camera_data_type == "rgb":
                channels = 3
            else:
                channels = camera.data.output[self.tiled_camera_data_type].shape[-1]
        dtype = self.pixels_dtype
        if dtype is None:
            dtype = camera.data.output[self.tiled_camera_data_type].dtype
        shape = (*self.batch_size, cfg.height, cfg.width, channels)
        if dtype == torch.uint8:
            pixels_spec = Bounded(0, 255, shape=shape, dtype=dtype, device=self.device)
        else:
            pixels_spec = Unbounded(shape=shape, dtype=dtype, device=self.device)
        self.observation_spec[self.pixels_key] = pixels_spec

    def _get_tiled_camera(self):
        env = self._env.unwrapped
        scene = env.scene
        try:
            return scene.sensors[self.tiled_camera_name]
        except KeyError as err:
            raise KeyError(
                f"Could not find tiled camera sensor {self.tiled_camera_name!r} "
                "in env.scene.sensors. Add one to the Isaac Lab config before "
                "calling gym.make, for example with "
                "IsaacLabWrapper.add_tiled_camera_config(...)."
            ) from err

    def _read_tiled_camera_pixels(self) -> torch.Tensor:
        pixels = self._get_tiled_camera().data.output[self.tiled_camera_data_type]
        pixels = torch.as_tensor(pixels, device=self.device)
        if self.pixels_channels is not None:
            pixels = pixels[..., : self.pixels_channels]
        if self.pixels_dtype is not None and pixels.dtype != self.pixels_dtype:
            if self.pixels_dtype == torch.uint8 and pixels.dtype.is_floating_point:
                pixels = pixels.mul(255).clamp(0, 255)
            pixels = pixels.to(self.pixels_dtype)
        return pixels

    def _add_tiled_camera_pixels(self, observations):
        if not self.from_tiled_camera:
            return observations
        if isinstance(observations, Mapping):
            observations = dict(observations)
        else:
            observations = {"observation": observations}
        observations[self.pixels_key] = self._read_tiled_camera_pixels()
        return observations

    def _reset_output_transform(self, reset_data):  # noqa: F811
        observations, info = super()._reset_output_transform(reset_data)
        return self._add_tiled_camera_pixels(observations), info

    def _output_transform(self, step_outputs_tuple):  # noqa: F811
        # IsaacLab will modify the `terminated` and `truncated` tensors
        #  in-place. We clone them here to make sure data doesn't inadvertently get modified.
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        observations = self._add_tiled_camera_pixels(observations)
        done = terminated | truncated
        reward = reward.unsqueeze(-1)  # to get to (num_envs, 1)
        return (
            observations,
            reward,
            terminated.clone(),
            truncated.clone(),
            done.clone(),
            info,
        )
