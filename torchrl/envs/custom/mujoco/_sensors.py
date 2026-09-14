# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Internal MicroDuck sensor assembly shared by locomotion and game scenes."""

from __future__ import annotations

import importlib
import math
from collections import deque
from collections.abc import Sequence
from copy import copy
from typing import Any

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Binary, Composite, Unbounded

_has_mujoco = importlib.util.find_spec("mujoco") is not None


def _align_microduck_cameras(model: Any):
    """Convert head-camera sites (+X forward, +Z up) to MuJoCo optical axes.

    The pinned asset's camera quaternion looks backward into its own head.
    Retain its lens position and rigid mounting body, taking orientation from
    the matching site. Custom cameras without that site remain unchanged.
    """
    mujoco = importlib.import_module("mujoco")
    for index in range(model.ncam):
        camera = model.camera(index)
        if camera.name.rsplit("/", 1)[-1] != "head_camera":
            continue
        try:
            site = model.site(camera.name)
        except KeyError:
            continue
        if model.site_bodyid[site.id] != model.cam_bodyid[index]:
            raise ValueError("MicroDuck camera and matching site must share a body.")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, site.quat)
        rotation = rotation.reshape(3, 3)
        # Camera right=-site Y, up=site Z, back=-site X.
        optical = np.column_stack((-rotation[:, 1], rotation[:, 2], -rotation[:, 0]))
        mujoco.mju_mat2Quat(camera.quat, optical.ravel())


class _MicroDuckSensors:
    """Ideal trunk IMU/encoder readings and a sampled, delayed head camera.

    The 53-value vector removes simulator linear velocity from the legacy
    layout: gravity (3), gyro (3), command (2), joint positions (14), joint
    velocities (14), gait clock (3), previous motor action (14). Gravity and
    gyro are ideal trunk-frame estimates; joint velocity is an ideal encoder
    derivative. No world pose, heading, contact or base-velocity estimate is
    supplied. Noise settings are explicit simulation perturbations, not a
    measured hardware calibration.

    The stock IMX219 profile uses approximately 62 degrees horizontal FOV at
    16:9. Square policy images are the centred crop, retaining the corresponding
    vertical FOV. The lens position is preserved and the matching head-camera
    site's axes define its outward view. Camera samples use
    simulator time at 30 Hz, with explicit age/validity and optional delay or
    dropout; a missing first frame is black/invalid. Native cameras are ordered
    exactly as the supplied names. Spectator rendering is independent.
    """

    DIM = 53
    SCHEMA = "microduck-proprioception-v1"

    def __init__(
        self,
        env: Any,
        *,
        vision: bool = False,
        camera_names: Sequence[str] = ("head_camera",),
        image_size: int = 64,
        camera_fps: float = 30.0,
        camera_hfov: float = 62.0,
        camera_delay_s: float = 0.0,
        camera_dropout: float = 0.0,
        gyro_noise_std: float = 0.0,
        joint_position_noise_std: float = 0.0,
        joint_velocity_noise_std: float = 0.0,
    ):
        if (
            image_size < 32
            or not math.isfinite(camera_fps)
            or camera_fps <= 0
            or not 0 < camera_hfov < 180
        ):
            raise ValueError("Camera size, rate and field of view must be valid.")
        if (
            not 0 <= camera_dropout <= 1
            or any(
                not math.isfinite(value)
                for value in (
                    camera_delay_s,
                    gyro_noise_std,
                    joint_position_noise_std,
                    joint_velocity_noise_std,
                )
            )
            or min(
                camera_delay_s,
                gyro_noise_std,
                joint_position_noise_std,
                joint_velocity_noise_std,
            )
            < 0
        ):
            raise ValueError(
                "Sensor noise/delay must be nonnegative and dropout in [0, 1]."
            )
        if vision and env.backend_name != "mujoco":
            raise ValueError(
                "MicroDuck sensor cameras currently require native MuJoCo."
            )
        self.env = env
        self.vision = vision
        self.image_size, self.camera_fps = image_size, camera_fps
        self.camera_delay_s, self.camera_dropout = camera_delay_s, camera_dropout
        self.noise = (
            gyro_noise_std,
            joint_position_noise_std,
            joint_velocity_noise_std,
        )
        self.camera_names = tuple(camera_names)
        self.camera_ids = []
        if vision:
            model = env._backend.mj_model
            _align_microduck_cameras(model)
            vfov = math.degrees(
                2 * math.atan(math.tan(math.radians(camera_hfov) / 2) * 9 / 16)
            )
            for name in self.camera_names:
                camera = model.camera(name)
                self.camera_ids.append(camera.id)
                model.cam_fovy[camera.id] = vfov
        self.reset()

    def reset(self):
        self._next_capture = 0.0
        self._pending = deque()
        self._pixels = self._stamp = self._valid = None

    def clone_for(self, env):
        # Native camera history belongs to its simulator snapshot. In particular,
        # stepping a clone must not render from or mutate its parent's cache.
        sensor = copy(self)
        sensor.env = env
        sensor._pending = deque(
            (stamp, pixels.clone(), valid.clone())
            for stamp, pixels, valid in self._pending
        )
        for name in ("_pixels", "_stamp", "_valid"):
            value = getattr(self, name)
            setattr(sensor, name, None if value is None else value.clone())
        return sensor

    def add_specs(self, spec: Composite):
        shape = spec["observation"].shape[:-1]
        spec["proprioception"] = Unbounded(
            (*shape, self.DIM), dtype=spec["observation"].dtype, device=spec.device
        )
        if self.vision:
            spec["camera_pixels"] = Unbounded(
                (*shape, self.image_size, self.image_size, 3),
                dtype=torch.uint8,
                device=spec.device,
            )
            spec["camera_age"] = Unbounded((*shape, 1), device=spec.device)
            spec["camera_valid"] = Binary(
                n=1, shape=(*shape, 1), dtype=torch.bool, device=spec.device
            )

    def update(self, td: TensorDictBase):
        observation = td["observation"]
        proprioception = torch.cat((observation[..., :6], observation[..., 9:56]), -1)
        for section, std in zip((slice(3, 6), slice(8, 22), slice(22, 36)), self.noise):
            if std:
                values = proprioception[..., section]
                values.add_(
                    torch.randn(
                        values.shape,
                        generator=self.env.rng,
                        device=values.device,
                        dtype=values.dtype,
                    )
                    * std
                )
        td["proprioception"] = proprioception
        if not self.vision:
            return td
        # Native MuJoCo has one simulator per worker; each camera adds an agent
        # dimension only when the observations themselves have that dimension.
        now = (
            float(self.env._step_count[0])
            * self.env.frame_skip
            * self.env._backend.timestep
        )
        if self._pixels is None:
            shape = observation.shape[:-1]
            self._pixels = torch.zeros(
                (*shape, self.image_size, self.image_size, 3),
                dtype=torch.uint8,
                device=observation.device,
            )
            self._stamp = torch.zeros((*shape, 1), device=observation.device)
            self._valid = torch.zeros(
                (*shape, 1), dtype=torch.bool, device=observation.device
            )
        if now + 1e-8 >= self._next_capture:
            images = torch.stack(
                [
                    self.env._backend.render(
                        camera_id=camera, width=self.image_size, height=self.image_size
                    )
                    for camera in self.camera_ids
                ],
                dim=1,
            )
            if observation.ndim == 2:
                images = images.squeeze(1)
            received = (
                torch.rand(
                    self._valid.shape, generator=self.env.rng, device=observation.device
                )
                >= self.camera_dropout
            )
            self._pending.append((now, images, received))
            self._next_capture = (
                math.floor(now * self.camera_fps + 1e-8) + 1
            ) / self.camera_fps
        while self._pending and self._pending[0][0] + self.camera_delay_s <= now + 1e-8:
            stamp, images, received = self._pending.popleft()
            self._pixels = torch.where(received[..., None, None], images, self._pixels)
            self._stamp = torch.where(received, stamp, self._stamp)
            self._valid = self._valid | received
        td["camera_pixels"] = self._pixels.clone()
        td["camera_age"] = torch.full_like(self._stamp, now) - self._stamp
        td["camera_valid"] = self._valid.clone()
        return td
