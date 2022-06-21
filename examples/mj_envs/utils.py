# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import traceback

import torch
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    NdBoundedTensorSpec,
)
from torchrl.envs import GymEnv
from torchrl.envs.libs.gym import _has_gym, _gym_to_torchrl_spec_transform
from torchrl.trainers.helpers.envs import LIBS
from mj_envs.envs.relay_kitchen import *
from mj_envs.envs.env_variants import register_env_variant
from envs_reg import *


class MJEnv(GymEnv):
    info_keys = ["time", "rwd_dense", "rwd_sparse", "solved"]

    def _build_env(
        self,
        env_name: str,
        from_pixels: bool = False,
        pixels_only: bool = False,
        **kwargs,
    ) -> "gym.core.Env":
        self.pixels_only = pixels_only
        try:
            render_device = int(str(self.device)[-1])
        except ValueError:
            render_device = 0
        print(f"rendering device: {render_device}, device is {self.device}")

        # traceback.print_stack()

        if not _has_gym:
            raise RuntimeError(
                f"gym not found, unable to create {env_name}. "
                f"Consider downloading and installing dm_control from"
                f" {self.git_url}"
            )
        try:
            env = self.lib.make(
                env_name, frameskip=self.frame_skip, device_id=render_device, **kwargs
            )
            self.wrapper_frame_skip = 1
        except TypeError as err:
            if "unexpected keyword argument 'frameskip" not in str(err):
                raise TypeError(err)
            env = self.lib.make(env_name)
            self.wrapper_frame_skip = self.frame_skip

        self.from_pixels = from_pixels
        self.render_device = render_device
        return env

    def _make_specs(self, env: "gym.Env") -> None:
        self.action_spec = _gym_to_torchrl_spec_transform(
            env.action_space, device=self.device
        )
        self.observation_spec = _gym_to_torchrl_spec_transform(
            env.observation_space,
            device=self.device,
        )
        if not isinstance(self.observation_spec, CompositeSpec):
            self.observation_spec = CompositeSpec(
                next_observation=self.observation_spec
            )
        env_name = self._constructor_kwargs["env_name"]
        if self.from_pixels:
            self.cameras = self._constructor_kwargs.get(
                "cameras",
                ["left_cam", "right_cam", "top_cam"]
                if "franka" in env_name
                else ["left_cam", "right_cam"],
            )

            self.observation_spec["next_pixels"] = NdBoundedTensorSpec(
                torch.zeros(
                    len(self.cameras),
                    244,  # working with 640
                    244,  # working with 480
                    3,
                    device=self.device,
                    dtype=torch.uint8,
                ),
                255
                * torch.ones(
                    len(self.cameras),
                    244,
                    244,
                    3,
                    device=self.device,
                    dtype=torch.uint8,
                ),
                torch.Size(torch.Size([len(self.cameras), 244, 244, 3])),
                dtype=torch.uint8,
                device=self.device,
            )

        self.reward_spec = UnboundedContinuousTensorSpec(
            device=self.device,
        )  # default

    def _step(self, td):
        td = super()._step(td)
        if self.from_pixels:
            img = self._env.render_camera_offscreen(
                sim=self._env.sim,
                cameras=self.cameras,
                device_id=self.render_device,
                width=244,
                height=244,  # working with 640 / 480
            )
            img = torch.Tensor(img).squeeze(0)
            td.set("next_pixels", img)
        return td

    def _reset(self, td=None, **kwargs):
        td = super()._reset(td, **kwargs)
        if self.from_pixels:
            img = self._env.render_camera_offscreen(
                sim=self._env.sim,
                cameras=self.cameras,
                device_id=self.render_device,
                width=244,
                height=244,
            )
            img = torch.Tensor(img).squeeze(0)
            td.set("next_pixels", img)
        return td

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        try:
            render_device = int(str(out.device)[-1])
        except ValueError:
            render_device = 0
        if render_device != self.render_device:
            out._build_env(**self._constructor_kwargs)
        # self._build_env(**self._constructor_kwargs)
        return out


LIBS["mj_envs"] = MJEnv
