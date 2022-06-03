import torch
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    NdBoundedTensorSpec,
)
from torchrl.envs import GymEnv
from torchrl.envs.libs.gym import _has_gym, _gym_to_torchrl_spec_transform
from mj_envs.envs.relay_kitchen import *


class MJEnv(GymEnv):
    info_keys = ["time", "rwd_dense", "rwd_sparse", "solved"]

    def _build_env(
        self,
        envname: str,
        taskname: str,
        from_pixels: bool = False,
        pixels_only: bool = False,
        **kwargs,
    ) -> "gym.core.Env":
        self.pixels_only = pixels_only
        try:
            render_device = int(str(self.device)[-1]) + 1
        except ValueError:
            render_device = 0
        print(f"rendering device: {render_device}, device is {self.device}")
        if not _has_gym:
            raise RuntimeError(
                f"gym not found, unable to create {envname}. "
                f"Consider downloading and installing dm_control from"
                f" {self.git_url}"
            )
        if not ((taskname == "") or (taskname is None)):
            raise ValueError(
                f"gym does not support taskname, received {taskname} instead."
            )
        try:
            env = self.lib.make(
                envname, frameskip=self.frame_skip, device_id=render_device, **kwargs
            )
            self.wrapper_frame_skip = 1
        except TypeError as err:
            if "unexpected keyword argument 'frameskip" not in str(err):
                raise TypeError(err)
            env = self.lib.make(envname)
            self.wrapper_frame_skip = self.frame_skip
        self._env = env

        self.from_pixels = from_pixels
        self.render_device = render_device

        self.action_spec = _gym_to_torchrl_spec_transform(self._env.action_space)
        self.observation_spec = _gym_to_torchrl_spec_transform(
            self._env.observation_space
        )
        if not isinstance(self.observation_spec, CompositeSpec):
            self.observation_spec = CompositeSpec(
                next_observation=self.observation_spec
            )
        if from_pixels:
            self.cameras = kwargs.get("cameras", ["left_cam", "right_cam", "top_cam"])

            self.observation_spec["next_pixels"] = NdBoundedTensorSpec(
                torch.zeros(
                    len(self.cameras),
                    480,
                    640,
                    3,
                    device=self.device,
                    dtype=torch.uint8,
                ),
                255
                * torch.ones(
                    len(self.cameras),
                    480,
                    640,
                    3,
                    device=self.device,
                    dtype=torch.uint8,
                ),
                torch.Size(torch.Size([len(self.cameras), 480, 640, 3])),
                dtype=torch.uint8,
            )

        self.reward_spec = UnboundedContinuousTensorSpec(
            device=self.device,
        )  # default

    def _step(self, td):
        td = super()._step(td)
        if self.from_pixels:
            img = self._env.render_camera_offscreen(
                sim=self._env.sim, cameras=self.cameras, device_id=self.render_device
            )
            img = torch.Tensor(img).squeeze(0)
            td.set("next_pixels", img)
        return td

    def _reset(self, td=None, **kwargs):
        td = super()._reset(td, **kwargs)
        if self.from_pixels:
            img = self._env.render_camera_offscreen(
                sim=self._env.sim, cameras=self.cameras, device_id=self.render_device
            )
            img = torch.Tensor(img).squeeze(0)
            td.set("next_pixels", img)
        return td

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        try:
            out.render_device = int(str(out.device)[-1]) + 1
        except ValueError:
            out.render_device = 0
        self._build_env(self.envname, self.taskname, **self.constructor_kwargs)
        return out
