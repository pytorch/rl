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
    A locomotion reward in the style of the mjlab velocity tasks, with every
    term weighted per second and multiplied by the control period: Gaussian
    tracking of the commanded body-frame velocity (lateral velocity tracked to
    zero) and of a zero yaw rate, a Gaussian uprightness term, a nominal-pose
    term that is tight when standing and loose when walking, and three
    contact-based gait terms that are active only under a nonzero command:
    rewarded foot air time inside a swing-duration window, the height of the
    clock's swing foot toward a clearance target, correct single support with
    respect to the gait clock, and a penalty for keeping both feet planted. Standing still under a
    nonzero command therefore earns clearly less than stepping, which a
    from-scratch policy otherwise settles into. Small costs discourage vertical
    and roll/pitch base motion, joint velocity and action rate. A fall costs a
    fixed penalty.

Termination
    A physical fall (low base height or tilted torso) or a non-finite state.
"""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import torch
from tensordict import TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.data.tensor_specs import Binary, Bounded, Composite, Unbounded
from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.envs.custom.mujoco.base import _MujocoMeta, MujocoEnv

MICRODUCK_RL_COMMIT = "d424a0c899f6b33cbd3daeb279913134349c0b63"
MICRODUCK_RL_ARCHIVE_URL = (
    "https://github.com/pollen-robotics/microduck_rl/archive/{commit}.zip"
)


def _download_microduck_rl(root: Path, commit: str, *, force: bool) -> Path:
    """Fetch the pinned ``microduck_rl`` archive into ``root/microduck_rl-<commit>``.

    The archive is extracted into a temporary directory next to the target and
    moved into place atomically, so a concurrent caller either finds the
    complete checkout or performs the download itself.
    """
    target = root / f"microduck_rl-{commit}"
    if target.exists() and force:
        shutil.rmtree(target)
    if target.exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    url = MICRODUCK_RL_ARCHIVE_URL.format(commit=commit)
    torchrl_logger.info("Downloading the MicroDuck assets from %s to %s", url, target)
    with TemporaryDirectory(prefix="microduck_rl-", dir=root) as tmp:
        archive = Path(tmp) / "microduck_rl.zip"
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        extracted = Path(tmp) / f"microduck_rl-{commit}"
        try:
            extracted.replace(target)
        except OSError:
            if not target.exists():
                raise
    return target


class _MicroDuckMeta(_MujocoMeta):
    """Resolve (and if requested download) the assets once, before batching.

    :class:`~torchrl.envs.custom.mujoco.base._MujocoMeta` builds one env per
    worker for the native backend; resolving the scene here means the workers
    receive a local path and never download concurrently.
    """

    def __call__(
        cls,
        microduck_root: str | Path | None = None,
        *args: Any,
        root: str | Path | None = None,
        download: bool | str = False,
        **kwargs: Any,
    ):
        scene = cls.resolve_scene(microduck_root, root=root, download=download)
        return super().__call__(scene, *args, **kwargs)


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


@dataclass(frozen=True)
class MicroDuckTask:
    """Task parameters of :class:`MicroDuckEnv`.

    Commands, reset distribution, actuation scale, gait clock, observation and
    reward options live here, so a task is one object rather than a dozen
    constructor arguments. Build one directly or start from a preset such as
    :meth:`MicroDuckEnv.tracking_task`, :meth:`MicroDuckEnv.standing_task` or
    :meth:`MicroDuckEnv.speed_range_task` and pass field overrides.

    Args:
        commanded_x_velocity (float or Sequence[float], optional): body-frame
            longitudinal velocity command in m/s. Every reset draws one value
            uniformly from the sequence for each env (a scalar is a fixed
            command, and repeating a value weights the draw); the command
            stays constant until the next reset. A ``commanded_x_velocity``
            entry of shape ``(num_envs, 1)`` or ``(num_envs,)`` in the reset
            TensorDict overrides the draw; the key is in the env's
            ``state_spec`` so :class:`~torchrl.envs.TransformedEnv` forwards
            it. Defaults to ``(0.03,)``. Ignored when ``command_range`` is
            given.
        command_range (tuple[float, float], optional): ``(low, high)`` interval
            in m/s from which the command is drawn uniformly at every reset
            instead, for training over a continuous speed range.
        warm_start_velocity (tuple[float, float], optional): ``(low, high)``
            forward speed interval in m/s. At reset, a ``warm_start_fraction``
            of the environments start already moving along their heading at a
            speed drawn from it, so an untrained policy experiences locomotion
            states early.
        warm_start_fraction (float, optional): fraction of resets that receive
            the warm start. Defaults to ``0.0``.
        joint_reset_noise_scale (float, optional): uniform noise added to the
            joint positions at reset, in radians. Defaults to the env's
            ``reset_noise_scale``. Larger values start episodes in diverse,
            off-balance poses, including single-support ones, which a
            from-scratch policy otherwise rarely visits.
        action_scale (float, optional): position-target offset in radians for
            a unit normalized action. Defaults to ``0.35``.
        gait_frequency_hz (float, optional): frequency of the gait clock
            exposed in the observation, at zero command. Defaults to
            ``1.8913``.
        gait_frequency_per_mps (float, optional): increase of the gait clock
            frequency per m/s of commanded speed, so the cadence rewarded by
            the single-support term follows the command. Defaults to ``0.0``
            (fixed clock).
        gait_phase_offset (float, optional): phase of the gait clock at the
            first step, in radians. Defaults to ``-1.5237``.
        gait_ramp_duration_s (float, optional): duration over which the gait
            ramp feature grows from zero to one after a reset. Defaults to
            ``0.4``.
        observe_lateral_velocity (bool, optional): if ``True``, append the
            body-frame lateral and vertical velocities to the observation,
            which gives the lateral tracking term an input. Defaults to
            ``False``.
        reward_scales (Mapping[str, float], optional): reward attribute names
            of :class:`MicroDuckEnv` such as ``"TRACKING_WEIGHT"`` or
            ``"TRACKING_STD"`` mapped to values that override the class
            defaults on the env instance.
        compute_reward (bool, optional): if ``False``, the env writes a zero
            reward and leaves the reward to a transform, which can read the
            observation or the ``diagnostics`` keys. Defaults to ``True``.
        diagnostics (bool, optional): if ``True``, add each reward component
            and pose diagnostics to the observation spec under
            ``diagnostic_*`` keys. Off by default because it roughly doubles
            the per-step task cost.

    Examples:
        Start from a preset, override a field, and hand the task to the env:

        >>> from dataclasses import replace
        >>> from torchrl.envs import MicroDuckEnv, MicroDuckTask
        >>> task = MicroDuckEnv.speed_range_task(0.1, 0.3, action_scale=1.0)
        >>> task.command_range, task.gait_frequency_per_mps, task.action_scale
        ((0.1, 0.3), 5.0, 1.0)
        >>> env = MicroDuckEnv(download=True, task=task, num_envs=4)  # doctest: +SKIP
        >>> rollout = env.rollout(20)  # doctest: +SKIP
        >>> rollout["commanded_x_velocity"][:, 0, 0]  # one command per env, drawn in [0.1, 0.3]  # doctest: +SKIP
        tensor([0.2731, 0.1207, 0.2942, 0.1685])

        The same task with the reward turned off, for a transform to fill in:

        >>> env = MicroDuckEnv(download=True, task=replace(task, compute_reward=False))  # doctest: +SKIP
        >>> env.rollout(5)["next", "reward"].sum()  # doctest: +SKIP
        tensor(0.)

        A task built field by field is equivalent to the presets:

        >>> MicroDuckTask(commanded_x_velocity=0.2) == MicroDuckEnv.tracking_task(0.2)
        True
    """

    commanded_x_velocity: float | Sequence[float] = (0.03,)
    command_range: tuple[float, float] | None = None
    warm_start_velocity: tuple[float, float] | None = None
    warm_start_fraction: float = 0.0
    joint_reset_noise_scale: float | None = None
    action_scale: float = 0.35
    gait_frequency_hz: float = 1.8913
    gait_frequency_per_mps: float = 0.0
    gait_phase_offset: float = -1.5237
    gait_ramp_duration_s: float = 0.4
    observe_lateral_velocity: bool = False
    reward_scales: Mapping[str, float] = field(default_factory=dict)
    compute_reward: bool = True
    diagnostics: bool = False

    def __post_init__(self):
        if not math.isfinite(self.gait_frequency_hz) or self.gait_frequency_hz <= 0:
            raise ValueError("gait_frequency_hz must be finite and positive.")
        if (
            not math.isfinite(self.gait_frequency_per_mps)
            or self.gait_frequency_per_mps < 0
        ):
            raise ValueError("gait_frequency_per_mps must be finite and non-negative.")
        if not math.isfinite(self.gait_phase_offset):
            raise ValueError("gait_phase_offset must be finite.")
        if (
            not math.isfinite(self.gait_ramp_duration_s)
            or self.gait_ramp_duration_s < 0
        ):
            raise ValueError("gait_ramp_duration_s must be finite and non-negative.")
        if not math.isfinite(self.action_scale) or self.action_scale <= 0:
            raise ValueError("action_scale must be finite and positive.")
        commands = torch.as_tensor(self.commanded_x_velocity, dtype=torch.float64)
        if commands.ndim == 0:
            commands = commands.unsqueeze(0)
        if commands.ndim != 1 or commands.numel() == 0:
            raise ValueError(
                "commanded_x_velocity must be a scalar or a non-empty 1-D sequence."
            )
        if not torch.isfinite(commands).all():
            raise ValueError("commanded_x_velocity values must be finite.")
        object.__setattr__(self, "commanded_x_velocity", tuple(commands.tolist()))
        for name in ("command_range", "warm_start_velocity"):
            interval = getattr(self, name)
            if interval is None:
                continue
            if (
                len(interval) != 2
                or not all(math.isfinite(v) for v in interval)
                or interval[0] > interval[1]
            ):
                raise ValueError(f"{name} must be a finite (low, high) pair.")
            object.__setattr__(self, name, tuple(float(v) for v in interval))
        if not 0.0 <= self.warm_start_fraction <= 1.0:
            raise ValueError("warm_start_fraction must be in [0, 1].")
        if self.warm_start_fraction > 0 and self.warm_start_velocity is None:
            raise ValueError("warm_start_fraction requires warm_start_velocity.")
        if self.joint_reset_noise_scale is not None and (
            not math.isfinite(self.joint_reset_noise_scale)
            or self.joint_reset_noise_scale < 0
        ):
            raise ValueError("joint_reset_noise_scale must be finite and non-negative.")
        for name, value in self.reward_scales.items():
            if not math.isfinite(value):
                raise ValueError(f"reward_scales[{name!r}] must be finite.")
        object.__setattr__(self, "reward_scales", dict(self.reward_scales))


class MicroDuckEnv(MujocoEnv, metaclass=_MicroDuckMeta):
    r"""Commanded longitudinal-velocity locomotion task for the MicroDuck biped.

    The action is a normalized offset around the actuator targets of the MJCF
    ``STAND`` keyframe, applied at 50 Hz. The observation concatenates
    projected gravity (3), base angular velocity (3), measured and commanded
    body-frame longitudinal velocity (2), joint-position error (14), joint
    velocity (14), the sine, cosine and ramp of a fixed-frequency gait clock
    (3), and the previous action (14). The command is also exposed under the
    ``commanded_x_velocity`` key so evaluation can read it directly.

    The reward follows the mjlab velocity-task recipe (see the module
    docstring); its weights are class attributes so a subclass can retune
    them, and ``reward_scales`` overrides them on one instance. Foot contacts
    and heights come from :meth:`foot_contacts` and :meth:`foot_heights`, so
    the gait terms work on every backend.

    MuJoCo stores free-joint linear velocity in the world frame and angular
    velocity in the body frame; the task rotates the linear velocity into the
    body frame before computing the observation and the reward.

    The MJCF is not vendored. It is resolved from ``microduck_root``, then from
    the ``MICRODUCK_RL_ROOT`` environment variable, then from an installed
    ``mjlab_microduck`` package, then from a checkout of the pinned upstream
    commit under ``root``, which ``download=True`` fetches when absent. Any
    revision of ``microduck_rl`` works through the first three options; the pin
    only fixes what ``download`` fetches, so the joint layout, the ``STAND``
    keyframe and the foot geom and site names the task relies on, all checked
    at load time, are known to match.

    Args:
        microduck_root (str or Path, optional): ``microduck_rl`` checkout,
            ``mjlab_microduck`` package directory, or path to
            ``scene_walk.xml``. Defaults to the :attr:`ROOT_ENV_VAR`
            environment variable, the installed package, or a download under
            ``root``.

    Keyword Args:
        task (MicroDuckTask, optional): commands, reset distribution, action
            scale, gait clock, observation and reward options. Defaults to
            :meth:`tracking_task`, a fixed ``0.03`` m/s forward command; see
            :class:`MicroDuckTask` for every field and :meth:`standing_task`
            and :meth:`speed_range_task` for the other presets.
        root (str or Path, optional): directory holding downloaded
            ``microduck_rl`` checkouts. Defaults to
            ``~/.cache/torchrl/microduck``.
        download (bool or ``"force"``, optional): whether to download commit
            :data:`MICRODUCK_RL_COMMIT` of ``microduck_rl`` into ``root`` when
            no other source resolves. Defaults to ``False``, in which case a
            missing asset raises an error describing every option. ``"force"``
            re-downloads even when the checkout is present.
        backend (str, optional): ``"mujoco"``, ``"mjx"`` or ``"mujoco-torch"``.
            Defaults to ``"mujoco"``, the fastest backend for this model on CPU
            in eager mode. The native backend batches ``num_envs`` simulators
            with :class:`~torchrl.envs.SerialEnv`, or with
            :class:`~torchrl.envs.ParallelEnv` when ``parallel=True``; the
            other two batch inside the simulator.
        low_cost_collisions (bool, optional): if ``True`` (default), replace
            the collision-class meshes with box proxies at load time. The
            unmodified meshes make the ``mjx`` and ``mujoco-torch`` backends
            run out of memory.
        max_episode_steps (int, optional): truncation horizon. Defaults to
            ``500``.
        \*\*kwargs: forwarded to :class:`~torchrl.envs.MujocoEnv`:
            ``num_envs``, ``device``, ``seed``, ``reset_noise_scale``,
            ``from_pixels``, ``render_width``, ``render_height``, ``camera_id``,
            ``compile_step`` and so on. ``xml_path`` and ``patch_xml`` are not
            accepted.

    Examples:
        Fetch the assets once and roll out a random policy that receives a
        different speed command at every reset:

        >>> from torchrl.envs import MicroDuckEnv
        >>> env = MicroDuckEnv(  # doctest: +SKIP
        ...     download=True, task=MicroDuckEnv.tracking_task((0.1, 0.2, 0.3))
        ... )
        >>> rollout = env.rollout(50)  # doctest: +SKIP
        >>> rollout["observation"].shape[-1], rollout["commanded_x_velocity"][0, 0]  # doctest: +SKIP
        (53, tensor([0.2000]))

        Scale up: batch 16 native simulators in worker processes, or run
        thousands inside MJX or ``mujoco-torch`` (optionally compiled) on a
        GPU. The task code is the same on every backend.

        >>> env = MicroDuckEnv(download=True, num_envs=16, parallel=True)  # doctest: +SKIP
        >>> env = MicroDuckEnv(download=True, backend="mjx", num_envs=1024, device="cuda")  # doctest: +SKIP
        >>> env = MicroDuckEnv(  # doctest: +SKIP
        ...     download=True, backend="mujoco-torch", num_envs=1024, device="cuda", compile_step=True
        ... )

        Pick the task: balance in place, track a speed range with a gait clock
        that follows the command (with a wider action scale for training from
        scratch), or pin the speed of an evaluation episode at reset.

        >>> from tensordict import TensorDict
        >>> env = MicroDuckEnv(download=True, task=MicroDuckEnv.standing_task())  # doctest: +SKIP
        >>> env = MicroDuckEnv(  # doctest: +SKIP
        ...     download=True, task=MicroDuckEnv.speed_range_task(0.1, 0.3, action_scale=1.0)
        ... )
        >>> td = env.reset(TensorDict(commanded_x_velocity=torch.full((1, 1), 0.25), batch_size=[1]))  # doctest: +SKIP
        >>> td["commanded_x_velocity"]  # doctest: +SKIP
        tensor([[0.2500]])

        Record a video with the standard recorder transform: the env renders
        offscreen into a ``"pixels"`` observation and the recorder writes an
        mp4 under ``./microduck/videos``.

        >>> from torchrl.envs import TransformedEnv
        >>> from torchrl.record import CSVLogger, VideoRecorder
        >>> env = TransformedEnv(  # doctest: +SKIP
        ...     MicroDuckEnv(download=True, from_pixels=True, render_width=480, render_height=360),
        ...     VideoRecorder(CSVLogger("microduck", video_format="mp4"), tag="rollout"),
        ... )
        >>> env.rollout(200)  # doctest: +SKIP
        >>> env.transform.dump()  # doctest: +SKIP

        Look inside the reward, retune it, or replace it: ``diagnostics=True``
        exposes every term in the observation, ``reward_scales`` changes the
        weights, and ``compute_reward=False`` leaves the reward to a transform.

        >>> env = MicroDuckEnv(  # doctest: +SKIP
        ...     download=True,
        ...     task=MicroDuckEnv.tracking_task(diagnostics=True, reward_scales={"TRACKING_WEIGHT": 4.0}),
        ... )
        >>> env.rollout(10)["next", "diagnostic_reward_tracking"].shape  # doctest: +SKIP
        torch.Size([1, 10, 1])
        >>> from torchrl.envs import Transform
        >>> class ForwardSpeedReward(Transform):
        ...     def _step(self, tensordict, next_tensordict):
        ...         next_tensordict["reward"] = next_tensordict["observation"][..., 6:7]
        ...         return next_tensordict
        >>> env = TransformedEnv(  # doctest: +SKIP
        ...     MicroDuckEnv(download=True, task=MicroDuckEnv.tracking_task(compute_reward=False)),
        ...     ForwardSpeedReward(),
        ... )

    Reference:
        Pollen Robotics, MicroDuck (https://github.com/pollen-robotics/microduck)
        and its mjlab training environments
        (https://github.com/pollen-robotics/microduck_rl).
    """

    DEFAULT_BACKEND: ClassVar[BackendName] = "mujoco"
    FRAME_SKIP = 10
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
    """Observation size without the optional lateral and vertical velocities."""
    # Reward weights are per second and multiplied by the control period, like
    # the mjlab velocity tasks. Positive terms are Gaussians in [0, 1].
    COMMAND_THRESHOLD: ClassVar[float] = 0.01
    TRACKING_WEIGHT: ClassVar[float] = 2.0
    TRACKING_STD: ClassVar[float] = 0.1
    YAW_RATE_WEIGHT: ClassVar[float] = 1.0
    YAW_RATE_STD: ClassVar[float] = 0.5**0.5
    UPRIGHT_WEIGHT: ClassVar[float] = 2.0
    UPRIGHT_STD: ClassVar[float] = 0.05**0.5
    POSE_WEIGHT: ClassVar[float] = 1.0
    POSE_STD_STANDING: ClassVar[float] = 0.1
    POSE_STD_WALKING: ClassVar[float] = 0.5
    AIR_TIME_WEIGHT: ClassVar[float] = 3.0
    AIR_TIME_WINDOW: ClassVar[tuple[float, float]] = (0.125, 0.3)
    SWING_HEIGHT_WEIGHT: ClassVar[float] = 2.0
    SWING_TARGET_HEIGHT: ClassVar[float] = 0.02
    PHASE_CONTACT_WEIGHT: ClassVar[float] = 3.0
    DOUBLE_SUPPORT_WEIGHT: ClassVar[float] = -1.0
    ANG_VEL_XY_WEIGHT: ClassVar[float] = -0.05
    LIN_VEL_Z_WEIGHT: ClassVar[float] = -2.0
    ACTION_RATE_WEIGHT: ClassVar[float] = -0.1
    JOINT_VELOCITY_WEIGHT: ClassVar[float] = -0.001
    FALL_PENALTY: ClassVar[float] = 4.0
    MIN_HEIGHT_RATIO: ClassVar[float] = 0.55
    MIN_UPRIGHT: ClassVar[float] = 0.35
    REWARD_COMPONENTS: ClassVar[tuple[str, ...]] = (
        "tracking",
        "yaw_rate",
        "upright",
        "pose",
        "air_time",
        "swing_height",
        "phase_contact",
        "double_support",
        "ang_vel_xy",
        "lin_vel_z",
        "action_rate",
        "joint_velocity",
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
        "left_foot_contact",
        "right_foot_contact",
        "left_foot_height",
        "right_foot_height",
    )
    FAILURE_DIAGNOSTICS: ClassVar[tuple[str, ...]] = ("height", "upright", "nonfinite")

    def __init__(
        self,
        microduck_root: str | Path | None = None,
        *,
        task: MicroDuckTask | None = None,
        root: str | Path | None = None,
        download: bool | str = False,
        backend: BackendName = "mujoco",
        low_cost_collisions: bool = True,
        max_episode_steps: int = 500,
        **kwargs: Any,
    ) -> None:
        for forbidden in ("xml_path", "patch_xml"):
            if forbidden in kwargs:
                raise ValueError(
                    f"MicroDuckEnv loads the MicroDuck MJCF itself; pass "
                    f"microduck_root=... instead of {forbidden}=..."
                )
        self.task = self.tracking_task() if task is None else task
        self.scene_path = self.resolve_scene(
            microduck_root, root=root, download=download
        )
        self.command_range = self.task.command_range
        self.warm_start_velocity = self.task.warm_start_velocity
        self.warm_start_fraction = self.task.warm_start_fraction
        self._joint_reset_noise_scale = self.task.joint_reset_noise_scale
        self.action_scale = self.task.action_scale
        self.diagnostics = self.task.diagnostics
        self.low_cost_collisions = bool(low_cost_collisions)
        self.gait_frequency_hz = self.task.gait_frequency_hz
        self.gait_frequency_per_mps = self.task.gait_frequency_per_mps
        self.gait_phase_offset = self.task.gait_phase_offset
        self.gait_ramp_duration_s = self.task.gait_ramp_duration_s
        self.observe_lateral_velocity = self.task.observe_lateral_velocity
        for name, value in self.task.reward_scales.items():
            if not name.isupper() or not isinstance(
                getattr(type(self), name, None), float
            ):
                raise ValueError(
                    f"reward_scales key {name!r} is not a float reward attribute of "
                    f"{type(self).__name__}."
                )
            setattr(self, name, float(value))
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
        self._command_values = torch.tensor(
            self.task.commanded_x_velocity, dtype=self.dtype, device=self.device
        )
        self._commanded_x_velocity = torch.zeros(
            self.num_envs, 1, dtype=self.dtype, device=self.device
        )
        self._previous_action = torch.zeros(
            self.num_envs, self.NUM_JOINTS, dtype=self.dtype, device=self.device
        )
        self._observation_action = self._previous_action.clone()
        self._feet_air_time = torch.zeros(
            self.num_envs, 2, dtype=self.dtype, device=self.device
        )
        self._touchdown_air_time = torch.zeros_like(self._feet_air_time)
        self._contacts = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self._foot_heights = torch.zeros_like(self._feet_air_time)
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

    @classmethod
    def tracking_task(
        cls, commanded_x_velocity: float | Sequence[float] = (0.03,), **overrides: Any
    ) -> MicroDuckTask:
        """Track a forward speed drawn from ``commanded_x_velocity`` at every reset.

        This is the default task. ``overrides`` set any other
        :class:`MicroDuckTask` field.
        """
        return MicroDuckTask(commanded_x_velocity=commanded_x_velocity, **overrides)

    @classmethod
    def standing_task(cls, **overrides: Any) -> MicroDuckTask:
        """Balance in place under a zero command.

        The zero command turns the gait terms off, leaving velocity tracking
        toward zero, posture, uprightness and the regularization costs.
        """
        return MicroDuckTask(commanded_x_velocity=0.0, **overrides)

    @classmethod
    def speed_range_task(
        cls, low: float = 0.1, high: float = 0.3, **overrides: Any
    ) -> MicroDuckTask:
        """Track a speed drawn uniformly from ``[low, high]`` at every reset.

        The gait clock runs at 1 Hz plus 5 Hz per m/s of command so the
        rewarded cadence follows the speed, which is what lets a policy trained
        from scratch modulate its speed with the command.
        """
        return MicroDuckTask(
            command_range=(float(low), float(high)),
            **{"gait_frequency_hz": 1.0, "gait_frequency_per_mps": 5.0, **overrides},
        )

    @property
    def joint_reset_noise_scale(self) -> float:
        """Uniform joint-position noise applied at reset, in radians."""
        if self._joint_reset_noise_scale is None:
            return self.reset_noise_scale
        return float(self._joint_reset_noise_scale)

    # ------------------------------------------------------------------
    # Asset resolution and model metadata
    # ------------------------------------------------------------------

    @classmethod
    def resolve_scene(
        cls,
        microduck_root: str | Path | None = None,
        *,
        root: str | Path | None = None,
        download: bool | str = False,
    ) -> Path:
        """Locate MicroDuck's ``scene_walk.xml``.

        Args:
            microduck_root: ``microduck_rl`` checkout, ``mjlab_microduck``
                package directory, or the scene XML itself. When omitted, the
                :attr:`ROOT_ENV_VAR` environment variable, an installed
                ``mjlab_microduck`` package and a checkout of
                :data:`MICRODUCK_RL_COMMIT` under ``root`` are tried in that
                order.
            root: directory holding downloaded checkouts. Defaults to
                ``~/.cache/torchrl/microduck``.
            download: download the pinned commit into ``root`` when nothing
                else resolves; ``"force"`` re-downloads it.

        Returns:
            The absolute path to the scene XML.

        Raises:
            FileNotFoundError: if the scene cannot be located and ``download``
                is ``False``.
        """
        cache_root = (
            Path("~/.cache/torchrl/microduck").expanduser()
            if root is None
            else Path(root).expanduser()
        )
        if download == "force":
            candidates = [
                _download_microduck_rl(cache_root, MICRODUCK_RL_COMMIT, force=True)
            ]
        elif microduck_root is not None:
            candidates = [Path(microduck_root).expanduser()]
        else:
            candidates = []
            env_root = os.environ.get(cls.ROOT_ENV_VAR)
            if env_root:
                candidates.append(Path(env_root).expanduser())
            spec = importlib.util.find_spec("mjlab_microduck")
            if spec is not None and spec.origin is not None:
                candidates.append(Path(spec.origin).resolve().parent)
            candidates.append(cache_root / f"microduck_rl-{MICRODUCK_RL_COMMIT}")
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
        if download and microduck_root is None:
            checkout = _download_microduck_rl(
                cache_root, MICRODUCK_RL_COMMIT, force=False
            )
            return cls.resolve_scene(checkout)
        detail = "\n".join(f"  - {path}" for path in attempted) or "  (nothing)"
        raise FileNotFoundError(
            f"Could not find MicroDuck's {cls.SCENE_FILE}. Either pass "
            "microduck_root=<microduck_rl checkout>, set the "
            f"{cls.ROOT_ENV_VAR} environment variable, install the "
            "mjlab_microduck package, or pass download=True to fetch commit "
            f"{MICRODUCK_RL_COMMIT[:9]} of pollen-robotics/microduck_rl into "
            f"{cache_root}. Tried:\n{detail}"
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

    @property
    def observation_dim(self) -> int:
        """Size of the ``observation`` vector for this instance."""
        return self.OBSERVATION_DIM + (2 if self.observe_lateral_velocity else 0)

    def _make_obs_spec(self) -> Composite:
        spec = Composite(
            observation=Unbounded(
                shape=(self.num_envs, self.observation_dim),
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
        frequency = (
            self.gait_frequency_hz
            + self.gait_frequency_per_mps * self._commanded_x_velocity.squeeze(-1).abs()
        )
        phase = self.gait_phase_offset + 2.0 * math.pi * frequency * elapsed_time
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
        parts = [
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
        ]
        if self.observe_lateral_velocity:
            parts.append(body_velocity[..., 1:3])
        return torch.cat(parts, dim=-1)

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
            qvel += torch.empty_like(qvel).uniform_(-noise, noise, generator=self.rng)
        joint_noise = self.joint_reset_noise_scale
        if joint_noise > 0:
            qpos[..., 7:] += torch.empty_like(qpos[..., 7:]).uniform_(
                -joint_noise, joint_noise, generator=self.rng
            )
        if self.warm_start_fraction > 0:
            low, high = self.warm_start_velocity
            speed = torch.empty(n, dtype=qvel.dtype, device=self.device).uniform_(
                low, high, generator=self.rng
            )
            selected = (
                torch.rand(n, generator=self.rng, device=self.device)
                < self.warm_start_fraction
            )
            heading = _body_forward_vector(qpos[..., 3:7].to(self.dtype))
            heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            warm = speed.unsqueeze(-1) * heading.to(qvel.dtype)
            qvel[..., :3] = torch.where(selected.unsqueeze(-1), warm, qvel[..., :3])
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
        if self.command_range is not None:
            low, high = self.command_range
            return torch.empty(
                self.num_envs, 1, dtype=self.dtype, device=self.device
            ).uniform_(low, high, generator=self.rng)
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
        self._feet_air_time.zero_()
        self._touchdown_air_time.zero_()
        self._refresh_contacts()

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
        self._feet_air_time = torch.where(
            mask, torch.zeros_like(self._feet_air_time), self._feet_air_time
        )
        self._touchdown_air_time = torch.where(
            mask, torch.zeros_like(self._touchdown_air_time), self._touchdown_air_time
        )
        self._refresh_contacts()

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

    def _refresh_contacts(self) -> None:
        self._contacts = self.foot_contacts()
        self._foot_heights = self.foot_heights().to(self.dtype)

    def _update_gait_state(self) -> None:
        """Advance the per-foot air-time bookkeeping after a physics step."""
        self._refresh_contacts()
        dt = self.frame_skip * self._backend.timestep
        airborne_before = self._feet_air_time > 0
        touchdown = self._contacts & airborne_before
        self._touchdown_air_time = torch.where(
            touchdown, self._feet_air_time, torch.zeros_like(self._feet_air_time)
        )
        self._feet_air_time = torch.where(
            self._contacts,
            torch.zeros_like(self._feet_air_time),
            self._feet_air_time + dt,
        )

    def _reward_components(
        self,
        next_state: TensorDictBase,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        qpos = next_state["qpos"].to(self.dtype)
        qvel = next_state["qvel"].to(self.dtype)
        dt = self.frame_skip * self._backend.timestep
        quaternion = qpos[..., 3:7]
        body_velocity = _body_frame_linear_velocity(quaternion, qvel[..., :3])
        command = self._commanded_x_velocity.squeeze(-1)
        moving = command.abs() > self.COMMAND_THRESHOLD
        upright = (-_projected_gravity(quaternion)[..., 2]).clamp(-1.0, 1.0)
        tilt = torch.acos(upright)

        tracking_error = (body_velocity[..., 0] - command).square() + body_velocity[
            ..., 1
        ].square()
        tracking = torch.exp(-tracking_error / self.TRACKING_STD**2)
        yaw_rate = torch.exp(-qvel[..., 5].square() / self.YAW_RATE_STD**2)
        upright_reward = torch.exp(-tilt.square() / self.UPRIGHT_STD**2)
        pose_std = torch.where(
            moving,
            torch.full_like(command, self.POSE_STD_WALKING),
            torch.full_like(command, self.POSE_STD_STANDING),
        )
        pose = torch.exp(
            -(qpos[..., 7:] - self._home_qpos[7:]).square().mean(dim=-1)
            / pose_std.square()
        )
        window_low, window_high = self.AIR_TIME_WINDOW
        air_time = (
            (self._touchdown_air_time - window_low)
            .clamp(0.0, window_high - window_low)
            .sum(dim=-1)
        )
        phase, _ = self._gait_clock()
        directed_sin = command.sign() * phase.sin()
        # The clock's swing foot is rewarded for any lift toward the clearance
        # target, contact or not, so the incentive to step is dense.
        swing_foot = torch.stack((directed_sin > 0, directed_sin <= 0), dim=-1)
        swing_height = (
            (self._foot_heights / self.SWING_TARGET_HEIGHT).clamp(0.0, 1.0) * swing_foot
        ).sum(dim=-1)
        # Left foot swings while the directed clock is positive. Credit is
        # given only for correct single support, so standing on both feet
        # earns nothing here.
        expected_contact = torch.stack((directed_sin <= 0, directed_sin > 0), dim=-1)
        phase_contact = (self._contacts == expected_contact).all(dim=-1).to(self.dtype)
        double_support = self._contacts.all(dim=-1).to(self.dtype)
        gait_gate = (moving & (upright >= self.MIN_UPRIGHT)).to(self.dtype)

        fallen = self._fallen(qpos, qvel)
        components = {
            "tracking": self.TRACKING_WEIGHT * tracking,
            "yaw_rate": self.YAW_RATE_WEIGHT * yaw_rate,
            "upright": self.UPRIGHT_WEIGHT * upright_reward,
            "pose": self.POSE_WEIGHT * pose,
            "air_time": self.AIR_TIME_WEIGHT * air_time * gait_gate,
            "swing_height": self.SWING_HEIGHT_WEIGHT * swing_height * gait_gate,
            "phase_contact": self.PHASE_CONTACT_WEIGHT * phase_contact * gait_gate,
            "double_support": self.DOUBLE_SUPPORT_WEIGHT * double_support * gait_gate,
            "ang_vel_xy": self.ANG_VEL_XY_WEIGHT * qvel[..., 3:5].square().sum(-1),
            "lin_vel_z": self.LIN_VEL_Z_WEIGHT * body_velocity[..., 2].square(),
            "action_rate": self.ACTION_RATE_WEIGHT
            * (action - self._previous_action).square().sum(dim=-1),
            "joint_velocity": self.JOINT_VELOCITY_WEIGHT
            * qvel[..., 6:].square().sum(-1),
        }
        components = {name: dt * value for name, value in components.items()}
        components["termination"] = -self.FALL_PENALTY * fallen.to(self.dtype)
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
                "diagnostic_left_foot_contact": self._contacts[..., 0:1].to(self.dtype),
                "diagnostic_right_foot_contact": self._contacts[..., 1:2].to(
                    self.dtype
                ),
                "diagnostic_left_foot_height": self._foot_heights[..., 0:1],
                "diagnostic_right_foot_height": self._foot_heights[..., 1:2],
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
        self._update_gait_state()
        if not self.task.compute_reward:
            return torch.zeros(self.num_envs, 1, dtype=self.dtype, device=self.device)
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
            "feet_air_time": self._feet_air_time[index].clone(),
        }

    def _load_indexed_extra_state(self, state: dict[str, Any]) -> None:
        self._previous_action = state["previous_action"].clone()
        self._observation_action = self._previous_action.clone()
        self._commanded_x_velocity = state["commanded_x_velocity"].clone()
        self._feet_air_time = state["feet_air_time"].clone()
        self._touchdown_air_time = torch.zeros_like(self._feet_air_time)
        self._refresh_contacts()

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
        self._feet_air_time[index] = source._feet_air_time.to(self.device)
        self._refresh_contacts()
