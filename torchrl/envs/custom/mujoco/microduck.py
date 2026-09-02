# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Commanded-velocity locomotion task for the MicroDuck biped.

MicroDuck is a small open-hardware bipedal robot by Pollen Robotics. The
walking MJCF and its meshes live in the ``microduck_rl`` repository and are
not vendored here: :class:`MicroDuckEnv` locates a local checkout or an
installed ``mjlab_microduck`` package and loads the same model on any of the
three :class:`~torchrl.envs.MujocoEnv` physics backends.

Reward
    Nonzero commands receive an alive bonus plus the body-frame velocity along
    the commanded direction; a zero command receives a Gaussian
    velocity-tracking reward and a nominal-pose term. Uprightness and height
    terms stabilize the gait, small costs discourage lateral drift, roll/yaw
    rate, joint velocity and action rate, and a fall costs a fixed penalty.

Termination
    A physical fall (low base height or tilted torso) or a non-finite state.
"""

from __future__ import annotations

import importlib.util
import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import torch
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Binary, Bounded, Composite, Unbounded
from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.envs.custom.mujoco.base import MujocoEnv


def _projected_gravity(quaternion: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame gravity into the body frame for wxyz quaternions."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x.square() + y.square())),
        ),
        dim=-1,
    )


def _body_frame_linear_velocity(
    quaternion: torch.Tensor,
    world_velocity: torch.Tensor,
) -> torch.Tensor:
    """Rotate a world-frame velocity into the body frame for wxyz quaternions."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w = quaternion[..., :1]
    vector = -quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(vector, world_velocity, dim=-1)
    return world_velocity + w * twice_cross + torch.cross(vector, twice_cross, dim=-1)


def _body_forward_vector(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the body x-axis expressed in the world frame."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        dim=-1,
    )


@contextmanager
def _low_cost_collision_scene(scene_path: Path) -> Iterator[Path]:
    """Replace detailed collision meshes with tight box proxies at load time.

    The upstream walking asset reuses render meshes for the feet and the
    self-collision geoms. Accelerated MuJoCo implementations expand every pair
    of convex-hull edges, which makes the two roughly 10,000-edge soles
    prohibitively expensive to compile or step in a batch. Visual meshes stay
    untouched; only geoms in the ``collision`` or ``self_collision_only``
    classes are replaced, and MuJoCo performs the box fitting so its mesh
    centering and principal-axis transforms remain part of the geom pose.

    Self-contained MJCF files without an ``<include>`` are yielded unchanged so
    small fixtures and custom MicroDuck-compatible files keep working.
    """
    scene_tree = ET.parse(scene_path)
    include = scene_tree.getroot().find("include")
    if include is None or include.get("file") is None:
        yield scene_path
        return

    robot_path = (scene_path.parent / include.get("file")).resolve()
    if not robot_path.is_file():
        yield scene_path
        return

    robot_tree = ET.parse(robot_path)
    robot_root = robot_tree.getroot()
    proxy_count = 0
    for geom in robot_root.iter("geom"):
        if geom.get("class") not in {"collision", "self_collision_only"}:
            continue
        if geom.get("mesh") is None:
            continue
        geom.set("type", "box")
        proxy_count += 1

    if not proxy_count:
        yield scene_path
        return

    compiler = robot_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        robot_root.insert(0, compiler)
    compiler.set("fitaabb", "true")
    for attribute in ("meshdir", "texturedir"):
        directory = compiler.get(attribute)
        if directory is not None and not Path(directory).is_absolute():
            compiler.set(attribute, str((robot_path.parent / directory).resolve()))

    with TemporaryDirectory(prefix="torchrl-microduck-") as directory:
        patched_robot = Path(directory) / robot_path.name
        patched_scene = Path(directory) / scene_path.name
        robot_tree.write(patched_robot, encoding="unicode")
        scene_tree.write(patched_scene, encoding="unicode")
        yield patched_scene


class MicroDuckEnv(MujocoEnv):
    r"""Commanded longitudinal-velocity locomotion task for the MicroDuck biped.

    The action is a normalized offset around the actuator targets of the MJCF
    ``STAND`` keyframe. The observation concatenates projected gravity (3),
    base angular velocity (3), measured and commanded body-frame longitudinal
    velocity (2), joint-position error (14), joint velocity (14), the sine,
    cosine and ramp of a fixed-frequency gait clock (3), and the previous
    action (14). The command is also exposed under the
    ``commanded_x_velocity`` key so evaluation can read it directly.

    MuJoCo stores free-joint linear velocity in the world frame and angular
    velocity in the body frame; the task rotates the linear velocity into the
    body frame before computing the observation and the reward.

    The MJCF is resolved from ``microduck_root``, then from the
    ``MICRODUCK_RL_ROOT`` environment variable, then from an installed
    ``mjlab_microduck`` package. A ``microduck_rl`` checkout, its package
    directory, or the ``scene_walk.xml`` file itself are all accepted.

    Args:
        microduck_root: ``microduck_rl`` checkout, ``mjlab_microduck`` package
            directory, or path to ``scene_walk.xml``. Defaults to the
            :attr:`ROOT_ENV_VAR` environment variable or the installed package.
        backend: MuJoCo physics backend. Defaults to ``"mujoco"``, which was
            the fastest backend for this model on CPU in eager mode.
        commanded_x_velocity: fixed body-frame longitudinal velocity command in
            m/s, or a sequence sampled uniformly at every reset. A command may
            also be provided under ``commanded_x_velocity`` in the reset
            TensorDict; the key is part of the env's ``state_spec`` so it is
            honored through :class:`~torchrl.envs.TransformedEnv` as well.
            Defaults to a forward-only ``0.03`` m/s command.
        action_scale: position-target offset in radians for a unit normalized
            action. Defaults to ``0.35``.
        diagnostics: if ``True``, add each reward component and pose
            diagnostics to the observation spec under ``diagnostic_*`` keys.
            Off by default because it roughly doubles the per-step task cost.
        low_cost_collisions: if ``True`` (default), replace the collision-class
            meshes with box proxies at load time. The unmodified meshes make
            the ``mjx`` and ``mujoco-torch`` backends run out of memory.
        gait_frequency_hz: frequency of the gait clock exposed in the
            observation. Defaults to ``1.8913``.
        gait_phase_offset: phase of the gait clock at the first step, in
            radians. Defaults to ``-1.5237``.
        gait_ramp_duration_s: duration over which the gait ramp feature grows
            from zero to one after a reset. Defaults to ``0.4``.
        max_episode_steps: truncation horizon. Defaults to ``500``.
        \*\*kwargs: forwarded to :class:`~torchrl.envs.MujocoEnv`. ``xml_path``
            and ``patch_xml`` are not accepted.

    Examples:
        >>> from torchrl.envs import MicroDuckEnv  # doctest: +SKIP
        >>> env = MicroDuckEnv(  # doctest: +SKIP
        ...     "/path/to/microduck_rl", num_envs=4, parallel=False
        ... )
        >>> td = env.rollout(10)  # doctest: +SKIP

    Reference:
        Pollen Robotics, MicroDuck (https://github.com/pollen-robotics/microduck)
        and its mjlab training environments
        (https://github.com/pollen-robotics/microduck_rl).
    """

    DEFAULT_BACKEND: ClassVar[BackendName] = "mujoco"
    FRAME_SKIP = 4
    RESET_NOISE_SCALE = 0.02
    ROOT_ENV_VAR: ClassVar[str] = "MICRODUCK_RL_ROOT"
    SCENE_FILE: ClassVar[str] = "scene_walk.xml"
    NUM_JOINTS: ClassVar[int] = 14
    JOINT_NAMES: ClassVar[tuple[str, ...]] = (
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    )
    FOOT_GEOMS: ClassVar[tuple[str, str]] = (
        "left_foot_collision",
        "right_foot_collision",
    )
    FOOT_SITES: ClassVar[tuple[str, str]] = ("left_foot", "right_foot")
    GAIT_PHASE_START: ClassVar[int] = 3 + 3 + 2 + NUM_JOINTS * 2
    OBSERVATION_DIM: ClassVar[int] = GAIT_PHASE_START + 3 + NUM_JOINTS
    VELOCITY_TRACKING_STD: ClassVar[float] = 0.25
    STAND_POSE_COMMAND_STD: ClassVar[float] = 0.01
    FORWARD_VELOCITY_REWARD_SCALE: ClassVar[float] = 5.0
    FORWARD_VELOCITY_CLAMP: ClassVar[float] = 0.2
    FALL_PENALTY: ClassVar[float] = 10.0
    MIN_HEIGHT_RATIO: ClassVar[float] = 0.55
    MIN_UPRIGHT: ClassVar[float] = 0.35
    REWARD_COMPONENTS: ClassVar[tuple[str, ...]] = (
        "velocity_tracking",
        "upright",
        "height",
        "pose",
        "lateral_velocity",
        "roll_yaw_rate",
        "joint_velocity",
        "action_rate",
        "termination",
    )
    POSE_DIAGNOSTICS: ClassVar[tuple[str, ...]] = (
        "height",
        "upright",
        "pitch",
        "roll",
        "body_velocity_x",
        "body_velocity_y",
        "body_velocity_z",
        "action_saturation_fraction",
        "target_clamp_fraction",
        "action_rate_rms",
    )
    FAILURE_DIAGNOSTICS: ClassVar[tuple[str, ...]] = ("height", "upright", "nonfinite")

    def __init__(
        self,
        microduck_root: str | Path | None = None,
        *,
        backend: BackendName = "mujoco",
        commanded_x_velocity: float | Sequence[float] = (0.03,),
        action_scale: float = 0.35,
        diagnostics: bool = False,
        low_cost_collisions: bool = True,
        gait_frequency_hz: float = 1.8913,
        gait_phase_offset: float = -1.5237,
        gait_ramp_duration_s: float = 0.4,
        max_episode_steps: int = 500,
        **kwargs: Any,
    ) -> None:
        for forbidden in ("xml_path", "patch_xml"):
            if forbidden in kwargs:
                raise ValueError(
                    f"MicroDuckEnv loads the MicroDuck MJCF itself; pass "
                    f"microduck_root=... instead of {forbidden}=..."
                )
        if not math.isfinite(gait_frequency_hz) or gait_frequency_hz <= 0:
            raise ValueError("gait_frequency_hz must be finite and positive.")
        if not math.isfinite(gait_phase_offset):
            raise ValueError("gait_phase_offset must be finite.")
        if not math.isfinite(gait_ramp_duration_s) or gait_ramp_duration_s < 0:
            raise ValueError("gait_ramp_duration_s must be finite and non-negative.")
        if not math.isfinite(action_scale) or action_scale <= 0:
            raise ValueError("action_scale must be finite and positive.")
        command_values = torch.as_tensor(commanded_x_velocity, dtype=torch.float64)
        if command_values.ndim == 0:
            command_values = command_values.unsqueeze(0)
        if command_values.ndim != 1 or command_values.numel() == 0:
            raise ValueError(
                "commanded_x_velocity must be a scalar or a non-empty 1-D sequence."
            )
        if not torch.isfinite(command_values).all():
            raise ValueError("commanded_x_velocity values must be finite.")

        self.scene_path = self.resolve_scene(microduck_root)
        self.action_scale = float(action_scale)
        self.diagnostics = bool(diagnostics)
        self.low_cost_collisions = bool(low_cost_collisions)
        self.gait_frequency_hz = float(gait_frequency_hz)
        self.gait_phase_offset = float(gait_phase_offset)
        self.gait_ramp_duration_s = float(gait_ramp_duration_s)
        physics_scene = (
            _low_cost_collision_scene(self.scene_path)
            if self.low_cost_collisions
            else nullcontext(self.scene_path)
        )
        with physics_scene as scene:
            super().__init__(
                xml_path=scene,
                patch_xml=False,
                backend=backend,
                max_episode_steps=max_episode_steps,
                **kwargs,
            )
        self._configure_from_model()
        self._command_values = command_values.to(device=self.device, dtype=self.dtype)
        self._commanded_x_velocity = torch.zeros(
            self.num_envs, 1, dtype=self.dtype, device=self.device
        )
        self._previous_action = torch.zeros(
            self.num_envs, self.NUM_JOINTS, dtype=self.dtype, device=self.device
        )
        self._observation_action = self._previous_action.clone()
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.NUM_JOINTS),
            dtype=self.dtype,
            device=self.device,
        )
        # Declaring the command as state lets a reset TensorDict carry it
        # through TransformedEnv, which only forwards reset and state keys.
        self.state_spec = Composite(
            commanded_x_velocity=Unbounded(
                shape=(self.num_envs, 1), dtype=self.dtype, device=self.device
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Asset resolution and model metadata
    # ------------------------------------------------------------------

    @classmethod
    def resolve_scene(cls, microduck_root: str | Path | None = None) -> Path:
        """Locate MicroDuck's ``scene_walk.xml``.

        Args:
            microduck_root: ``microduck_rl`` checkout, ``mjlab_microduck``
                package directory, or the scene XML itself. When omitted, the
                :attr:`ROOT_ENV_VAR` environment variable and an installed
                ``mjlab_microduck`` package are tried in that order.

        Returns:
            The absolute path to the scene XML.

        Raises:
            FileNotFoundError: if the scene cannot be located.
        """
        candidates: list[Path] = []
        if microduck_root is not None:
            candidates.append(Path(microduck_root).expanduser())
        else:
            env_root = os.environ.get(cls.ROOT_ENV_VAR)
            if env_root:
                candidates.append(Path(env_root).expanduser())
            spec = importlib.util.find_spec("mjlab_microduck")
            if spec is not None and spec.origin is not None:
                candidates.append(Path(spec.origin).resolve().parent)
        suffixes = (
            Path(cls.SCENE_FILE),
            Path("robot", "microduck", cls.SCENE_FILE),
            Path("mjlab_microduck", "robot", "microduck", cls.SCENE_FILE),
            Path("src", "mjlab_microduck", "robot", "microduck", cls.SCENE_FILE),
        )
        attempted: list[Path] = []
        for candidate in candidates:
            paths = (
                (candidate,)
                if candidate.suffix == ".xml"
                else tuple(candidate / suffix for suffix in suffixes)
            )
            for path in paths:
                attempted.append(path)
                if path.is_file():
                    return path.resolve()
        detail = "\n".join(f"  - {path}" for path in attempted) or "  (nothing)"
        raise FileNotFoundError(
            f"Could not find MicroDuck's {cls.SCENE_FILE}. Pass microduck_root=..., "
            f"set {cls.ROOT_ENV_VAR}, or install mjlab_microduck. Tried:\n{detail}"
        )

    def _configure_from_model(self) -> None:
        import mujoco

        model = self._backend.mj_model
        if (model.nq, model.nv, model.nu) != (21, 20, self.NUM_JOINTS):
            raise ValueError(
                "Expected the 14-actuator MicroDuck walking model with "
                f"(nq, nv, nu)=(21, 20, 14), got {(model.nq, model.nv, model.nu)}."
            )
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if key_id < 0:
            raise ValueError("The MicroDuck MJCF must define a `STAND` keyframe.")
        joint_ids = torch.as_tensor(model.actuator_trnid[:, 0].copy()).long()
        if (joint_ids < 0).any():
            raise ValueError("Every MicroDuck actuator must target a joint.")
        home_ctrl = torch.as_tensor(model.key_ctrl[key_id].copy())
        joint_limited = torch.as_tensor(
            model.jnt_limited[joint_ids.numpy()].copy()
        ).bool()
        joint_range = torch.as_tensor(model.jnt_range[joint_ids.numpy()].copy())
        joint_low = torch.where(joint_limited, joint_range[:, 0], home_ctrl - torch.pi)
        joint_high = torch.where(joint_limited, joint_range[:, 1], home_ctrl + torch.pi)
        self._home_qpos = torch.as_tensor(model.key_qpos[key_id].copy()).to(
            device=self.device, dtype=self.dtype
        )
        self._home_ctrl = home_ctrl.to(device=self.device, dtype=self.dtype)
        self._joint_low = joint_low.to(device=self.device, dtype=self.dtype)
        self._joint_high = joint_high.to(device=self.device, dtype=self.dtype)
        self._target_height = self._home_qpos[2].clone()

    # ------------------------------------------------------------------
    # Contact helpers
    # ------------------------------------------------------------------

    def foot_contacts(self) -> torch.Tensor:
        """Return a ``(num_envs, 2)`` boolean tensor of left/right foot contact."""
        return self.geom_contacts(self.FOOT_GEOMS)

    def foot_heights(self) -> torch.Tensor:
        """Return a ``(num_envs, 2)`` tensor with the left/right foot site heights."""
        return self.site_positions(self.FOOT_SITES)[..., 2]

    # ------------------------------------------------------------------
    # Specs and observations
    # ------------------------------------------------------------------

    def _make_obs_spec(self) -> Composite:
        spec = Composite(
            observation=Unbounded(
                shape=(self.num_envs, self.OBSERVATION_DIM),
                dtype=self.dtype,
                device=self.device,
            ),
            commanded_x_velocity=Unbounded(
                shape=(self.num_envs, 1),
                dtype=self.dtype,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )
        if not self.diagnostics:
            return spec
        for name in self.REWARD_COMPONENTS:
            spec[f"diagnostic_reward_{name}"] = Unbounded(
                shape=(self.num_envs, 1), dtype=self.dtype, device=self.device
            )
        for name in self.POSE_DIAGNOSTICS:
            spec[f"diagnostic_{name}"] = Unbounded(
                shape=(self.num_envs, 1), dtype=self.dtype, device=self.device
            )
        for name in self.FAILURE_DIAGNOSTICS:
            spec[f"diagnostic_{name}_failure"] = Binary(
                n=1, shape=(self.num_envs, 1), dtype=torch.bool, device=self.device
            )
        return spec

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        observation = super()._build_obs_dict(state)
        observation["commanded_x_velocity"] = self._commanded_x_velocity.clone()
        if self.diagnostics:
            observation.update(self._diagnostics(state, self._observation_action))
        return observation

    def _gait_clock(self) -> tuple[torch.Tensor, torch.Tensor]:
        elapsed_time = self._step_count.to(self.dtype) * (
            self.frame_skip * self._backend.timestep
        )
        phase = (
            self.gait_phase_offset
            + 2.0 * math.pi * self.gait_frequency_hz * elapsed_time
        )
        if self.gait_ramp_duration_s > 0:
            ramp = (elapsed_time / self.gait_ramp_duration_s).clamp(max=1.0)
        else:
            ramp = torch.ones_like(elapsed_time)
        return phase, ramp

    def _make_obs(self, state: TensorDictBase) -> torch.Tensor:
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        body_velocity = _body_frame_linear_velocity(qpos[..., 3:7], qvel[..., :3])
        phase, ramp = self._gait_clock()
        return torch.cat(
            (
                _projected_gravity(qpos[..., 3:7]),
                qvel[..., 3:6],
                body_velocity[..., :1],
                self._commanded_x_velocity,
                qpos[..., 7:] - self._home_qpos[7:],
                qvel[..., 6:],
                phase.sin().unsqueeze(-1),
                phase.cos().unsqueeze(-1),
                ramp.unsqueeze(-1),
                self._observation_action,
            ),
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Reset and commands
    # ------------------------------------------------------------------

    def _sample_initial_state(
        self,
        n: int,
        tensordict: TensorDictBase | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del tensordict
        qpos = self._home_qpos.unsqueeze(0).expand(n, -1).clone()
        qpos = qpos.to(dtype=self._backend.qpos0.dtype)
        qvel = torch.zeros(
            n, self._backend.nv, dtype=self._backend.qvel0.dtype, device=self.device
        )
        if self.reset_noise_scale > 0:
            noise = self.reset_noise_scale
            qpos[..., :2] += torch.empty_like(qpos[..., :2]).uniform_(
                -noise, noise, generator=self.rng
            )
            qpos[..., 7:] += torch.empty_like(qpos[..., 7:]).uniform_(
                -noise, noise, generator=self.rng
            )
            qvel += torch.empty_like(qvel).uniform_(-noise, noise, generator=self.rng)
        return qpos, qvel

    def _sample_command(self, tensordict: TensorDictBase | None) -> torch.Tensor:
        if tensordict is not None and "commanded_x_velocity" in tensordict.keys():
            command = tensordict["commanded_x_velocity"].to(
                device=self.device, dtype=self.dtype
            )
            if command.shape == self.batch_size:
                command = command.unsqueeze(-1)
            if command.shape != self.batch_size + torch.Size([1]):
                raise ValueError(
                    "A reset commanded_x_velocity must have shape "
                    f"{tuple(self.batch_size + torch.Size([1]))}, got "
                    f"{tuple(command.shape)}."
                )
            if not torch.isfinite(command).all():
                raise ValueError("A reset commanded_x_velocity must be finite.")
            return command
        indices = torch.randint(
            self._command_values.numel(),
            (self.num_envs,),
            generator=self.rng,
            device=self.device,
        )
        return self._command_values[indices].unsqueeze(-1)

    def _on_reset_all(self, tensordict: TensorDictBase | None = None) -> None:
        self._previous_action.zero_()
        self._observation_action.zero_()
        self._commanded_x_velocity = self._sample_command(tensordict)

    def _on_reset_mask(
        self,
        mask: torch.Tensor,
        tensordict: TensorDictBase | None = None,
    ) -> None:
        mask = mask.unsqueeze(-1) if mask.ndim == 1 else mask
        self._previous_action = torch.where(
            mask, torch.zeros_like(self._previous_action), self._previous_action
        )
        self._observation_action = torch.where(
            mask, torch.zeros_like(self._observation_action), self._observation_action
        )
        self._commanded_x_velocity = torch.where(
            mask, self._sample_command(tensordict), self._commanded_x_velocity
        )

    # ------------------------------------------------------------------
    # Dynamics, reward, termination
    # ------------------------------------------------------------------

    def _prepare_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(-1.0, 1.0)
        target = self._home_ctrl + self.action_scale * action
        return target.clamp(self._joint_low, self._joint_high)

    def _fallen(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        upright = -_projected_gravity(qpos[..., 3:7])[..., 2]
        finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
        return (
            (qpos[..., 2] < self.MIN_HEIGHT_RATIO * self._target_height)
            | (upright < self.MIN_UPRIGHT)
            | ~finite
        )

    def _reward_components(
        self,
        next_state: TensorDictBase,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        qpos = next_state["qpos"].to(self.dtype)
        qvel = next_state["qvel"].to(self.dtype)
        projected_gravity = _projected_gravity(qpos[..., 3:7])
        body_velocity = _body_frame_linear_velocity(qpos[..., 3:7], qvel[..., :3])
        command = self._commanded_x_velocity.squeeze(-1)
        tracking_reward = 2.0 * torch.exp(
            -((body_velocity[..., 0] - command) / self.VELOCITY_TRACKING_STD).square()
        )
        directional_velocity = command.sign() * body_velocity[..., 0]
        locomotion_reward = 1.0 + self.FORWARD_VELOCITY_REWARD_SCALE * (
            directional_velocity.clamp(
                -self.FORWARD_VELOCITY_CLAMP, self.FORWARD_VELOCITY_CLAMP
            )
        )
        velocity_reward = torch.where(
            command.abs() > 1e-6, locomotion_reward, tracking_reward
        )
        upright = (-projected_gravity[..., 2]).clamp(-1.0, 1.0)
        upright_reward = torch.exp(-4.0 * (1.0 - upright).square())
        height_reward = torch.exp(
            -((qpos[..., 2] - self._target_height) / 0.03).square()
        )
        pose_reward = torch.exp(
            -((qpos[..., 7:] - self._home_qpos[7:]) / 0.35).square().mean(dim=-1)
        )
        stand_pose_gate = torch.exp(-(command / self.STAND_POSE_COMMAND_STD).square())
        fallen = self._fallen(qpos, qvel)
        components = {
            "velocity_tracking": velocity_reward,
            "upright": 0.5 * upright_reward,
            "height": 0.25 * height_reward,
            "pose": 0.5 * stand_pose_gate * pose_reward,
            "lateral_velocity": -0.1 * body_velocity[..., 1].square(),
            "roll_yaw_rate": -0.02 * qvel[..., [3, 5]].square().mean(dim=-1),
            "joint_velocity": -0.002 * qvel[..., 6:].square().mean(dim=-1),
            "action_rate": -0.02 * (action - self._previous_action).square().mean(-1),
            "termination": -self.FALL_PENALTY * fallen.to(self.dtype),
        }
        return {
            f"diagnostic_reward_{name}": value.unsqueeze(-1)
            for name, value in components.items()
        }

    def _diagnostics(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        quaternion = qpos[..., 3:7]
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = quaternion.unbind(-1)
        pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
        upright = -_projected_gravity(quaternion)[..., 2]
        body_velocity = _body_frame_linear_velocity(quaternion, qvel[..., :3])
        target = self._home_ctrl + self.action_scale * action.clamp(-1.0, 1.0)
        target_clamped = (target < self._joint_low) | (target > self._joint_high)
        finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
        diagnostics = self._reward_components(state, action)
        diagnostics.update(
            {
                "diagnostic_height": qpos[..., 2:3],
                "diagnostic_upright": upright.unsqueeze(-1),
                "diagnostic_pitch": pitch.unsqueeze(-1),
                "diagnostic_roll": roll.unsqueeze(-1),
                "diagnostic_body_velocity_x": body_velocity[..., 0:1],
                "diagnostic_body_velocity_y": body_velocity[..., 1:2],
                "diagnostic_body_velocity_z": body_velocity[..., 2:3],
                "diagnostic_action_saturation_fraction": (action.abs() >= 0.99)
                .to(self.dtype)
                .mean(dim=-1, keepdim=True),
                "diagnostic_target_clamp_fraction": target_clamped.to(self.dtype).mean(
                    dim=-1, keepdim=True
                ),
                "diagnostic_action_rate_rms": (action - self._previous_action)
                .square()
                .mean(dim=-1, keepdim=True)
                .sqrt(),
                "diagnostic_height_failure": (
                    qpos[..., 2] < self.MIN_HEIGHT_RATIO * self._target_height
                ).unsqueeze(-1),
                "diagnostic_upright_failure": (upright < self.MIN_UPRIGHT).unsqueeze(
                    -1
                ),
                "diagnostic_nonfinite_failure": (~finite).unsqueeze(-1),
            }
        )
        return diagnostics

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        return torch.stack(
            tuple(self._reward_components(next_state, action).values())
        ).sum(dim=0)

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        qpos = next_state["qpos"].to(self.dtype)
        qvel = next_state["qvel"].to(self.dtype)
        return self._fallen(qpos, qvel).unsqueeze(-1)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._observation_action = tensordict["action"].to(self.dtype)
        result = super()._step(tensordict)
        self._previous_action = self._observation_action.clone()
        return result

    # ------------------------------------------------------------------
    # Snapshot indexing
    # ------------------------------------------------------------------

    def _index_extra_state(self, index: slice | torch.Tensor) -> dict[str, Any]:
        return {
            "previous_action": self._previous_action[index].clone(),
            "commanded_x_velocity": self._commanded_x_velocity[index].clone(),
        }

    def _load_indexed_extra_state(self, state: dict[str, Any]) -> None:
        self._previous_action = state["previous_action"].clone()
        self._observation_action = self._previous_action.clone()
        self._commanded_x_velocity = state["commanded_x_velocity"].clone()

    def _set_indexed_extra_state(
        self,
        index: slice | torch.Tensor,
        source: MujocoEnv,
    ) -> None:
        if not isinstance(source, MicroDuckEnv):
            raise TypeError(
                "MicroDuckEnv snapshots can only be restored from a MicroDuckEnv."
            )
        self._previous_action[index] = source._previous_action.to(self.device)
        self._observation_action[index] = source._observation_action.to(self.device)
        self._commanded_x_velocity[index] = source._commanded_x_velocity.to(self.device)
