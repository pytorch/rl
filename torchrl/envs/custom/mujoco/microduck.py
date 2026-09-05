# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Locomotion tasks for the MicroDuck biped: stand, walk, sidestep, jump.

MicroDuck is a small open-hardware bipedal robot by Pollen Robotics. The
walking MJCF and its meshes live in the ``microduck_rl`` repository and are
not vendored here: :class:`MicroDuckEnv` locates a local checkout or an
installed ``mjlab_microduck`` package and loads the same model on any of the
three :class:`~torchrl.envs.MujocoEnv` physics backends.

Tasks
    A task is data: :class:`MicroDuckTask` is a tensorclass holding the
    command box, the reset distribution, the gait clock and the reward
    weights and parameters. The env takes a library of tasks and every env of
    the batch holds one row of it, picked at reset either from the library's
    ``weight`` field or from a ``task_id`` carried by the reset TensorDict.

Reward
    A locomotion reward in the style of the mjlab velocity tasks, computed as
    a matrix of registered terms times each env's weight row, with every
    per-second term multiplied by the control period: Gaussian tracking of
    the commanded planar body-frame velocity (tighter across the commanded
    direction than along it, so diagonal motion earns less than an on-axis
    error of the same size) and of a zero yaw rate, a
    Gaussian uprightness term, a nominal-pose term, contact-based gait terms
    (foot air time inside a swing window, swing-foot height toward a
    clearance target, correct single support with respect to the gait clock,
    a penalty for keeping both feet planted; their credit scales with the
    progress along the command), a progress term linear in the velocity along
    the command, hop terms (a vertical-velocity rhythm on the task clock,
    upward velocity while planted, base height gained while both feet are
    airborne), small costs on vertical and
    roll/pitch base motion, joint velocity and action rate, and a fixed fall
    penalty. A term is off when its weight is zero; the presets set the
    weights, and :meth:`MicroDuckEnv.register_reward` adds user terms.

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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import torch
from tensordict import NestedKey, tensorclass, TensorDict, TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.data.tensor_specs import Binary, Bounded, Categorical, Composite, Unbounded
from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.envs.custom.mujoco.base import _MujocoMeta, MujocoEnv
from torchrl.envs.transforms.transforms import Transform

MICRODUCK_RL_COMMIT = "d424a0c899f6b33cbd3daeb279913134349c0b63"
MICRODUCK_RL_ARCHIVE_URL = (
    "https://github.com/pollen-robotics/microduck_rl/archive/{commit}.zip"
)

# A reward term maps (features, params) to a (num_envs,) tensor; see
# MicroDuckEnv.register_reward.
RewardTerm = Callable[[TensorDictBase, TensorDictBase], torch.Tensor]


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


def _body_lateral_vector(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the body y-axis (left) expressed in the world frame."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z + w * x),
        ),
        dim=-1,
    )


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


@tensorclass
class MicroDuckTask:
    """One locomotion task of :class:`MicroDuckEnv`, as data.

    A task is a row of tensors and a name: the command box, the reset
    distribution, the gait clock, the reward weights and parameters, and a
    label. A library is a stack of tasks, and every env of a
    :class:`MicroDuckEnv` batch holds one row of the library for the duration
    of an episode. Gathering rows with ``library[task_id]`` copies the name
    entries like the tensors, which needs ``tensordict>=0.14.1``. Build tasks with the presets
    (:meth:`MicroDuckEnv.tracking_task`, :meth:`MicroDuckEnv.standing_task`,
    :meth:`MicroDuckEnv.speed_range_task`, :meth:`MicroDuckEnv.sidestep_task`,
    :meth:`MicroDuckEnv.jump_task`), which fill every field and accept
    overrides, and stack them with :func:`torch.stack` or by handing a
    sequence to the env. Stacking is the structural validation: every task
    must carry the full reward weight vector and every parameter key.

    Args:
        command_low (torch.Tensor): lower corner of the planar command box
            ``(vx, vy)`` in m/s, shape ``(2,)``. A fixed command is a box with
            ``command_low == command_high``.
        command_high (torch.Tensor): upper corner of the command box, shape
            ``(2,)``. The command is drawn uniformly in the box at reset and
            held for the episode.
        warm_start_velocity (torch.Tensor): ``(low, high)`` speed interval in
            m/s, shape ``(2,)``. At reset, a ``warm_start_fraction`` of the
            envs whose command is nonzero start already moving along the
            commanded direction at a speed drawn from it.
        warm_start_fraction (torch.Tensor): fraction of resets that receive
            the warm start, scalar in ``[0, 1]``.
        joint_reset_noise_scale (torch.Tensor): uniform noise added to the
            joint positions at reset, in radians, scalar.
        gait_frequency_hz (torch.Tensor): frequency of the gait clock exposed
            in the observation at zero command, scalar.
        gait_frequency_per_mps (torch.Tensor): increase of the clock frequency
            per m/s of commanded speed, scalar.
        reward_weights (torch.Tensor): one weight per registered reward term,
            in the order of :attr:`MicroDuckEnv.REWARD_TERMS`, shape
            ``(num_terms,)``. A zero weight turns the term off; an all-zero
            row is "no built-in reward", for a transform to fill in.
        params (TensorDict): scalar parameters read by the reward terms, one
            entry per key of :attr:`MicroDuckEnv.REWARD_PARAMS` (for example
            ``tracking_std`` or ``pose_std``).
        weight (torch.Tensor): relative weight of the task when the env draws
            a task per env at reset, scalar and non-negative.
        name (str): label of the task, for logging and evaluation; the presets
            derive it from their arguments (``"tracking+0.20"``,
            ``"sidestep-0.15"``, ``"jump"``) and accept ``name=`` to override.

    Examples:
        Two tasks stacked into a library, one row per env picked at reset:

        >>> import torch
        >>> from torchrl.envs import MicroDuckEnv
        >>> library = torch.stack(
        ...     [MicroDuckEnv.tracking_task(0.2, weight=2.0), MicroDuckEnv.jump_task()]
        ... )
        >>> library.shape, library.command_high[:, 0], library.weight
        (torch.Size([2]), tensor([0.2000, 0.0000]), tensor([2., 1.]))
        >>> list(library.name)
        ['tracking+0.20', 'jump']
        >>> env = MicroDuckEnv(download=True, tasks=library, num_envs=4)  # doctest: +SKIP
        >>> rollout = env.rollout(20)  # doctest: +SKIP
        >>> rollout["task_id"][:, 0, 0], rollout["command"][:, 0]  # doctest: +SKIP
        (tensor([0, 1, 0, 0]), tensor([[0.2, 0.0], [0.0, 0.0], [0.2, 0.0], [0.2, 0.0]]))

        Overrides retune one row: reward weights by term name, term
        parameters and reset fields by name.

        >>> task = MicroDuckEnv.tracking_task(
        ...     0.2, reward_weights={"tracking": 4.0}, tracking_std=0.2, warm_start_fraction=0.5,
        ...     warm_start_velocity=(0.1, 0.3),
        ... )
        >>> task.reward_weights[list(MicroDuckEnv.REWARD_TERMS).index("tracking")]
        tensor(4.)
        >>> task.params["tracking_std"], task.warm_start_fraction
        (tensor(0.2000), tensor(0.5000))
    """

    command_low: torch.Tensor
    command_high: torch.Tensor
    warm_start_velocity: torch.Tensor
    warm_start_fraction: torch.Tensor
    joint_reset_noise_scale: torch.Tensor
    gait_frequency_hz: torch.Tensor
    gait_frequency_per_mps: torch.Tensor
    reward_weights: torch.Tensor
    params: TensorDict
    weight: torch.Tensor
    name: str


@dataclass(frozen=True)
class _RegisteredTerm:
    fn: RewardTerm
    weight: float
    per_second: bool


class MicroDuckEnv(MujocoEnv, metaclass=_MicroDuckMeta):
    r"""Locomotion tasks for the MicroDuck biped: stand, walk, sidestep, jump.

    The action is a normalized offset around the actuator targets of the MJCF
    ``STAND`` keyframe, applied at 50 Hz. The observation concatenates
    projected gravity (3), base angular velocity (3), body-frame linear
    velocity (3), the planar command ``(vx, vy)`` (2), joint-position error
    (14), joint velocity (14), the sine, cosine and ramp of the gait clock
    (3), and the previous action (14). The command and the index of the env's
    task in the library are also exposed under the ``command`` and ``task_id``
    keys; task parameters are not in the observation, an embedding of the id
    stands for them, and a transform can gather ``env.tasks[task_id]`` when
    they are needed.

    The env holds a library of :class:`MicroDuckTask` rows in :attr:`tasks`
    (``env.tasks.name`` lists their labels).
    At every reset, the envs being reset pick a row: the ``task_id`` entry of
    the reset TensorDict when present (``(num_envs, 1)`` or ``(num_envs,)``
    integers), otherwise a draw weighted by the tasks' ``weight`` field with
    the env's generator. The row sets the command box, the warm start, the
    joint reset noise, the gait clock and the reward for the episode. A
    ``command`` entry in the reset TensorDict pins the command inside the
    row's box. Both keys are in the ``state_spec`` so
    :class:`~torchrl.envs.TransformedEnv` forwards them; see
    :class:`MicroDuckTaskSampler` for weighted or curriculum mixtures.

    The reward is a matrix of registered terms times each env's weight row;
    :meth:`register_reward` adds terms and :attr:`REWARD_TERMS` lists them.
    Foot contacts and heights come from :meth:`foot_contacts` and
    :meth:`foot_heights`, so the gait terms work on every backend.

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
        tasks (MicroDuckTask or Sequence[MicroDuckTask], optional): the task
            library: one task, a sequence of tasks or a stacked
            :class:`MicroDuckTask` of shape ``(num_tasks,)``. A single task is
            a library of one. Defaults to :meth:`tracking_task`, a fixed
            ``0.03`` m/s forward command.
        action_scale (float, optional): position-target offset in radians for
            a unit normalized action. Defaults to ``0.35``.
        diagnostics (bool, optional): if ``True``, add each weighted reward
            term and pose diagnostics to the observation spec under
            ``diagnostic_*`` keys. Off by default because it roughly doubles
            the per-step task cost.
        root (str or Path, optional): directory holding downloaded
            ``microduck_rl`` checkouts. Defaults to
            ``~/.cache/torchrl/microduck``.
        download (bool or ``"force"``, optional): whether to download commit
            :data:`MICRODUCK_RL_COMMIT` of ``microduck_rl`` into ``root`` when
            no other source resolves. Defaults to ``False``, in which case a
            missing asset raises an error describing every option. ``"force"``
            re-downloads even when the checkout is present.
        backend (str, optional): ``"mujoco-torch"`` (default) and ``"mjx"``
            vectorize the ``num_envs`` simulators inside the simulator, which
            is how the env is meant to run at scale on an accelerator.
            ``"mujoco"`` runs the official C bindings, one simulator per
            worker process with :class:`~torchrl.envs.ParallelEnv` (or in one
            process with :class:`~torchrl.envs.SerialEnv` when
            ``parallel=False``); it is the fallback for CPU-only machines.
            Native workers each receive the library and draw their own task
            ids.
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
        Fetch the assets once and roll out a random policy on a two-task
        library; each env picks a task at reset and holds it:

        >>> import torch
        >>> from torchrl.envs import MicroDuckEnv
        >>> env = MicroDuckEnv(  # doctest: +SKIP
        ...     download=True,
        ...     tasks=[MicroDuckEnv.tracking_task(0.2), MicroDuckEnv.sidestep_task(0.15)],
        ...     num_envs=4,
        ... )
        >>> rollout = env.rollout(50)  # doctest: +SKIP
        >>> rollout["observation"].shape[-1], rollout["task_id"][:, 0, 0]  # doctest: +SKIP
        (56, tensor([1, 0, 0, 1]))
        >>> rollout["command"][:, 0]  # doctest: +SKIP
        tensor([[0.0000, 0.1500], [0.2000, 0.0000], [0.2000, 0.0000], [0.0000, 0.1500]])

        Scale up: run thousands of vectorized simulators inside
        ``mujoco-torch`` (optionally compiled) or MJX on a GPU, or fall back
        to 16 native simulators in worker processes on a CPU-only machine. The
        task code is the same on every backend.

        >>> env = MicroDuckEnv(download=True, num_envs=1024, device="cuda", compile_step=True)  # doctest: +SKIP
        >>> env = MicroDuckEnv(download=True, backend="mjx", num_envs=1024, device="cuda")  # doctest: +SKIP
        >>> env = MicroDuckEnv(download=True, backend="mujoco", num_envs=16, parallel=True)  # doctest: +SKIP

        Pick the task per env at reset, or pin the command of an evaluation
        episode inside its box:

        >>> from tensordict import TensorDict
        >>> env = MicroDuckEnv(  # doctest: +SKIP
        ...     download=True,
        ...     tasks=[MicroDuckEnv.standing_task(), MicroDuckEnv.speed_range_task(0.1, 0.3)],
        ...     num_envs=2,
        ... )
        >>> td = env.reset(TensorDict(task_id=torch.tensor([[1], [0]]), batch_size=[2]))  # doctest: +SKIP
        >>> td["task_id"][:, 0], td["command"][1]  # doctest: +SKIP
        (tensor([1, 0]), tensor([0., 0.]))
        >>> td = env.reset(  # doctest: +SKIP
        ...     TensorDict(task_id=torch.tensor([1, 1]), command=torch.tensor([[0.25, 0.0], [0.1, 0.0]]), batch_size=[2])
        ... )
        >>> td["command"][:, 0]  # doctest: +SKIP
        tensor([0.2500, 0.1000])

        Weighted mixtures: the ``weight`` field of each task sets its share of
        the env's own draw; :class:`MicroDuckTaskSampler` writes ``task_id``
        at reset for weights that change during training.

        >>> from torchrl.envs import MicroDuckTaskSampler, TransformedEnv
        >>> library = [MicroDuckEnv.standing_task(weight=0.5), MicroDuckEnv.jump_task(weight=2.0)]
        >>> env = TransformedEnv(  # doctest: +SKIP
        ...     MicroDuckEnv(download=True, backend="mujoco", num_envs=16, tasks=library),
        ...     MicroDuckTaskSampler([0.0, 1.0]),  # every reset picks the jump task
        ... )

        Record a video with the standard recorder transform: the env renders
        offscreen into a ``"pixels"`` observation and the recorder writes an
        mp4 under ``./microduck/videos``.

        >>> from torchrl.record import CSVLogger, VideoRecorder
        >>> env = TransformedEnv(  # doctest: +SKIP
        ...     MicroDuckEnv(download=True, from_pixels=True, render_width=480, render_height=360),
        ...     VideoRecorder(CSVLogger("microduck", video_format="mp4"), tag="rollout"),
        ... )
        >>> env.rollout(200)  # doctest: +SKIP
        >>> env.transform.dump()  # doctest: +SKIP

        Look inside the reward, retune it, or replace it: ``diagnostics=True``
        exposes every weighted term in the observation, a task's
        ``reward_weights`` retune it, an all-zero weight row leaves the reward
        to a transform, and :meth:`register_reward` adds a term that every
        task can weight.

        >>> env = MicroDuckEnv(  # doctest: +SKIP
        ...     download=True, diagnostics=True, tasks=MicroDuckEnv.tracking_task(reward_weights={"tracking": 4.0})
        ... )
        >>> env.rollout(10)["next", "diagnostic_reward_tracking"].shape  # doctest: +SKIP
        torch.Size([1, 10, 1])
        >>> @MicroDuckEnv.register_reward("heading", heading_std=0.3)
        ... def heading(features, params):
        ...     return torch.exp(-features["angular_velocity"][..., 2].square() / params["heading_std"].square())
        >>> task = MicroDuckEnv.tracking_task(0.2, reward_weights={"heading": 1.0}, heading_std=0.5)
        >>> env = MicroDuckEnv(download=True, tasks=task)  # doctest: +SKIP

    Reference:
        Pollen Robotics, MicroDuck (https://github.com/pollen-robotics/microduck)
        and its mjlab training environments
        (https://github.com/pollen-robotics/microduck_rl).
    """

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
    BODY_VELOCITY_START: ClassVar[int] = 6
    """Index of the body-frame linear velocity ``(vx, vy, vz)`` in the observation."""
    COMMAND_START: ClassVar[int] = 9
    """Index of the planar command ``(vx, vy)`` in the observation."""
    GAIT_PHASE_START: ClassVar[int] = COMMAND_START + 2 + NUM_JOINTS * 2
    """Index of the gait clock ``(sin, cos, ramp)`` in the observation."""
    OBSERVATION_DIM: ClassVar[int] = GAIT_PHASE_START + 3 + NUM_JOINTS
    COMMAND_THRESHOLD: ClassVar[float] = 0.01
    """Planar command speed under which a task counts as standing."""
    GAIT_FREQUENCY_HZ: ClassVar[float] = 1.8913
    GAIT_PHASE_OFFSET: ClassVar[float] = -1.5237
    """Phase of the gait clock at the first step, in radians."""
    GAIT_RAMP_DURATION_S: ClassVar[float] = 0.4
    """Duration over which the gait ramp feature grows from zero to one after a reset."""
    POSE_STD_STANDING: ClassVar[float] = 0.1
    POSE_STD_MOVING: ClassVar[float] = 0.5
    JUMP_WEIGHT: ClassVar[float] = 10.0
    LAUNCH_WEIGHT: ClassVar[float] = 30.0
    HOP_RHYTHM_WEIGHT: ClassVar[float] = 1.0
    HOP_FREQUENCY_HZ: ClassVar[float] = 2.0
    GAIT_TERMS: ClassVar[tuple[str, ...]] = (
        "air_time",
        "swing_height",
        "phase_contact",
        "double_support",
    )
    """Reward terms that shape stepping; the standing and jump presets turn them off."""
    FALL_PENALTY: ClassVar[float] = 4.0
    MIN_HEIGHT_RATIO: ClassVar[float] = 0.55
    MIN_UPRIGHT: ClassVar[float] = 0.35
    REWARD_TERMS: ClassVar[dict[str, _RegisteredTerm]] = {}
    """Registered reward terms by name, in weight-vector order."""
    REWARD_PARAMS: ClassVar[dict[str, float]] = {}
    """Default value of every term parameter a :class:`MicroDuckTask` carries."""
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
        tasks: MicroDuckTask | Sequence[MicroDuckTask] | None = None,
        action_scale: float = 0.35,
        diagnostics: bool = False,
        root: str | Path | None = None,
        download: bool | str = False,
        backend: BackendName = "mujoco-torch",
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
        if not math.isfinite(action_scale) or action_scale <= 0:
            raise ValueError("action_scale must be finite and positive.")
        self.tasks = self.stack_tasks(tasks)
        self.action_scale = float(action_scale)
        self.diagnostics = bool(diagnostics)
        self.scene_path = self.resolve_scene(
            microduck_root, root=root, download=download
        )
        self.low_cost_collisions = bool(low_cost_collisions)
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
        self.tasks = self.tasks.to(self.device).to(self.dtype)
        # Per-env task rows, task ids and commands; refreshed at reset.
        self._task_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._task = self.tasks[self._task_id]
        self._command = torch.zeros(
            self.num_envs, 2, dtype=self.dtype, device=self.device
        )
        # Task rows and command drawn while sampling the initial state, consumed
        # by the reset hooks so the warm start and the command agree.
        self._pending: tuple[torch.Tensor, MicroDuckTask, torch.Tensor] | None = None
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
        # Declaring the task id and the command as state lets a reset
        # TensorDict carry them through TransformedEnv, which only forwards
        # reset and state keys.
        self.state_spec = Composite(
            task_id=Categorical(
                n=self.tasks.shape[0],
                shape=(self.num_envs, 1),
                dtype=torch.long,
                device=self.device,
            ),
            command=Unbounded(
                shape=(self.num_envs, 2), dtype=self.dtype, device=self.device
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Tasks: library, presets and reward registry
    # ------------------------------------------------------------------

    @classmethod
    def stack_tasks(
        cls, tasks: MicroDuckTask | Sequence[MicroDuckTask] | None
    ) -> MicroDuckTask:
        """Normalize one task, a sequence or a stacked task into a ``(num_tasks,)`` library.

        Stacking checks that every task carries the same fields, the full
        reward weight vector and every parameter key; the boxes, weights and
        fractions are then range checked.
        """
        if tasks is None:
            tasks = cls.tracking_task()
        if isinstance(tasks, MicroDuckTask):
            library = tasks.unsqueeze(0) if tasks.batch_dims == 0 else tasks
        else:
            tasks = list(tasks)
            if not tasks or not all(isinstance(task, MicroDuckTask) for task in tasks):
                raise TypeError(
                    "tasks must be a MicroDuckTask or a non-empty sequence of them."
                )
            library = torch.stack(tasks)
        if library.batch_dims != 1:
            raise ValueError(
                f"A task library has one batch dimension, got shape {library.shape}."
            )
        library = library.contiguous()
        cls._validate_rows(library)
        if library.weight.sum() <= 0:
            raise ValueError("Task weights must not all be zero.")
        return library

    @classmethod
    def _validate_rows(cls, rows: MicroDuckTask) -> None:
        """Range and finiteness checks shared by :meth:`make_task` and :meth:`stack_tasks`."""
        num_terms = len(cls.REWARD_TERMS)
        if rows.reward_weights.shape[-1:] != (num_terms,):
            raise ValueError(
                f"reward_weights must have one entry per registered term "
                f"({num_terms}: {tuple(cls.REWARD_TERMS)}), got shape "
                f"{tuple(rows.reward_weights.shape[1:])}. Build the tasks after "
                "registering every term."
            )
        missing = set(cls.REWARD_PARAMS) - set(rows.params.keys())
        if missing:
            raise ValueError(f"tasks are missing the reward params {sorted(missing)}.")
        for field in (
            "command_low",
            "command_high",
            "warm_start_velocity",
            "warm_start_fraction",
            "joint_reset_noise_scale",
            "gait_frequency_hz",
            "gait_frequency_per_mps",
            "reward_weights",
            "weight",
        ):
            if not torch.isfinite(getattr(rows, field)).all():
                raise ValueError(f"Task field {field!r} must be finite.")
        for key, value in rows.params.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Reward param {key!r} must be finite.")
        if rows.command_low.shape[-1:] != (2,) or rows.command_high.shape[-1:] != (2,):
            raise ValueError("command_low and command_high are planar (vx, vy) boxes.")
        if (rows.command_low > rows.command_high).any():
            raise ValueError("Every task needs command_low <= command_high.")
        if (
            rows.warm_start_velocity.shape[-1:] != (2,)
            or (
                rows.warm_start_velocity[..., 0] > rows.warm_start_velocity[..., 1]
            ).any()
        ):
            raise ValueError("warm_start_velocity must be a (low, high) pair.")
        if (rows.warm_start_velocity < 0).any():
            raise ValueError("warm_start_velocity speeds must be non-negative.")
        fraction = rows.warm_start_fraction
        if ((fraction < 0) | (fraction > 1)).any():
            raise ValueError("warm_start_fraction must be in [0, 1].")
        if (rows.joint_reset_noise_scale < 0).any():
            raise ValueError("joint_reset_noise_scale must be non-negative.")
        if (rows.gait_frequency_hz <= 0).any():
            raise ValueError("gait_frequency_hz must be positive.")
        if (rows.gait_frequency_per_mps < 0).any():
            raise ValueError("gait_frequency_per_mps must be non-negative.")
        if (rows.weight < 0).any():
            raise ValueError("Task weights must be non-negative.")
        if not all(isinstance(name, str) and name for name in rows.name):
            raise ValueError("Every task needs a non-empty string name.")

    @classmethod
    def register_reward(
        cls,
        name: str,
        *,
        weight: float = 0.0,
        per_second: bool = True,
        **params: float,
    ) -> Callable[[RewardTerm], RewardTerm]:
        r"""Register a reward term that every task weights.

        Used as a decorator on a function ``(features, params) -> Tensor`` of
        shape ``(num_envs,)``. ``features`` is the step's feature TensorDict
        with entries ``body_velocity`` (body frame, ``(num_envs, 3)``),
        ``angular_velocity`` (3), ``upright`` (cosine of the tilt), ``base_height``,
        ``standing_height``, ``joint_error`` (14), ``joint_velocity`` (14),
        ``action`` (14), ``previous_action`` (14), ``contacts`` (bool, 2),
        ``foot_heights`` (2), ``touchdown_air_time`` (2), ``gait_phase``
        (radians), ``command`` (2) and ``fallen`` (bool). ``params`` is the
        per-env TensorDict of task parameters, each of shape ``(num_envs,)``.

        Args:
            name (str): term name; the diagnostics key is
                ``diagnostic_reward_<name>``.

        Keyword Args:
            weight (float, optional): default weight of the term in every
                preset. Defaults to ``0.0``, so an existing task ignores the
                term until its ``reward_weights`` name it.
            per_second (bool, optional): if ``True`` (default), the term is a
                rate and is multiplied by the control period, like the mjlab
                velocity tasks. ``False`` for one-off terms such as the fall
                penalty.
            \*\*params: default values of parameters the term reads from
                ``params``; the presets carry them and accept overrides by
                name. A parameter name may be registered by one term only.

        Tasks built before a registration have a shorter weight vector and
        are rejected by the env, so register terms before building tasks.

        Examples:
            >>> import torch
            >>> from torchrl.envs import MicroDuckEnv
            >>> @MicroDuckEnv.register_reward("still_head", still_head_std=1.0)
            ... def still_head(features, params):
            ...     head = features["joint_velocity"][..., 5:9].square().sum(-1)
            ...     return torch.exp(-head / params["still_head_std"].square())
            >>> task = MicroDuckEnv.standing_task(reward_weights={"still_head": 0.5})
            >>> task.reward_weights[-1], task.params["still_head_std"]
            (tensor(0.5000), tensor(1.))
        """
        if name in cls.REWARD_TERMS:
            raise ValueError(f"A reward term named {name!r} is already registered.")
        clash = set(params) & set(cls.REWARD_PARAMS)
        if clash:
            raise ValueError(f"Reward params already registered: {sorted(clash)}.")
        for key, value in params.items():
            if not math.isfinite(value):
                raise ValueError(f"Reward param {key!r} must be finite.")

        def decorator(fn: RewardTerm) -> RewardTerm:
            cls.REWARD_TERMS[name] = _RegisteredTerm(fn, float(weight), per_second)
            cls.REWARD_PARAMS.update(
                {key: float(value) for key, value in params.items()}
            )
            return fn

        return decorator

    @classmethod
    def make_task(
        cls,
        command_low: Sequence[float],
        command_high: Sequence[float],
        *,
        name: str,
        weight: float = 1.0,
        reward_weights: Mapping[str, float] | None = None,
        **overrides: Any,
    ) -> MicroDuckTask:
        """Build a :class:`MicroDuckTask` from a command box, a name and overrides.

        The presets call this with their box, their name and their weight and
        parameter choices. ``reward_weights`` maps term names to weights that
        replace the registered defaults; ``overrides`` set reset and clock
        fields (``warm_start_velocity``, ``warm_start_fraction``,
        ``joint_reset_noise_scale``, ``gait_frequency_hz``,
        ``gait_frequency_per_mps``) or term parameters (any key of
        :attr:`REWARD_PARAMS`) by name.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("A task needs a non-empty string name.")
        fields: dict[str, Any] = {
            "warm_start_velocity": (0.0, 0.0),
            "warm_start_fraction": 0.0,
            "joint_reset_noise_scale": cls.RESET_NOISE_SCALE,
            "gait_frequency_hz": cls.GAIT_FREQUENCY_HZ,
            "gait_frequency_per_mps": 0.0,
        }
        params = dict(cls.REWARD_PARAMS)
        for key, value in overrides.items():
            if key in fields:
                fields[key] = value
            elif key in params:
                params[key] = float(value)
            else:
                raise ValueError(
                    f"Unknown task field {key!r}; expected one of "
                    f"{sorted(fields)} or a reward param in {sorted(params)}."
                )
        weights = {name: term.weight for name, term in cls.REWARD_TERMS.items()}
        unknown = set(reward_weights or {}) - set(weights)
        if unknown:
            raise ValueError(
                f"reward_weights name unregistered terms {sorted(unknown)}; "
                f"registered: {tuple(weights)}."
            )
        weights.update(reward_weights or {})
        warm_low, warm_high = fields["warm_start_velocity"]
        task = MicroDuckTask(
            command_low=torch.tensor(command_low, dtype=torch.float32),
            command_high=torch.tensor(command_high, dtype=torch.float32),
            warm_start_velocity=torch.tensor(
                (warm_low, warm_high), dtype=torch.float32
            ),
            warm_start_fraction=torch.tensor(
                float(fields["warm_start_fraction"]), dtype=torch.float32
            ),
            joint_reset_noise_scale=torch.tensor(
                float(fields["joint_reset_noise_scale"]), dtype=torch.float32
            ),
            gait_frequency_hz=torch.tensor(
                float(fields["gait_frequency_hz"]), dtype=torch.float32
            ),
            gait_frequency_per_mps=torch.tensor(
                float(fields["gait_frequency_per_mps"]), dtype=torch.float32
            ),
            reward_weights=torch.tensor(
                [float(weights[name]) for name in cls.REWARD_TERMS],
                dtype=torch.float32,
            ),
            params=TensorDict(
                {
                    key: torch.tensor(value, dtype=torch.float32)
                    for key, value in params.items()
                },
                batch_size=[],
            ),
            weight=torch.tensor(float(weight), dtype=torch.float32),
            name=name,
            batch_size=[],
        )
        cls._validate_rows(task.unsqueeze(0))
        return task

    @classmethod
    def tracking_task(
        cls, speed: float = 0.03, *, weight: float = 1.0, **overrides: Any
    ) -> MicroDuckTask:
        """Walk at a fixed forward speed in m/s (negative walks backward).

        This is the default task. The gait terms are on and the pose term
        loose unless ``speed`` is below :attr:`COMMAND_THRESHOLD`, in which
        case the task is :meth:`standing_task`. ``overrides`` go to
        :meth:`make_task`.
        """
        if abs(float(speed)) <= cls.COMMAND_THRESHOLD:
            return cls.standing_task(weight=weight, **overrides)
        return cls.make_task(
            (float(speed), 0.0),
            (float(speed), 0.0),
            weight=weight,
            **{
                "name": f"tracking{float(speed):+.2f}",
                "pose_std": cls.POSE_STD_MOVING,
                **overrides,
            },
        )

    @classmethod
    def standing_task(cls, *, weight: float = 1.0, **overrides: Any) -> MicroDuckTask:
        """Balance in place under a zero command.

        The gait terms are off and the pose term tight, leaving velocity
        tracking toward zero, posture, uprightness and the regularization
        costs.
        """
        reward_weights = dict.fromkeys(cls.GAIT_TERMS, 0.0)
        reward_weights.update(overrides.pop("reward_weights", None) or {})
        return cls.make_task(
            (0.0, 0.0),
            (0.0, 0.0),
            weight=weight,
            reward_weights=reward_weights,
            **{"name": "standing", "pose_std": cls.POSE_STD_STANDING, **overrides},
        )

    @classmethod
    def speed_range_task(
        cls,
        low: float = 0.1,
        high: float = 0.3,
        *,
        weight: float = 1.0,
        **overrides: Any,
    ) -> MicroDuckTask:
        """Track a forward speed drawn uniformly from ``[low, high]`` at every reset.

        The gait clock runs at 1 Hz plus 5 Hz per m/s of command so the
        rewarded cadence follows the speed, which is what lets a policy trained
        from scratch modulate its speed with the command. The gait terms stay
        on over the whole range, so a range that spans zero rewards stepping
        in place at low commands.
        """
        return cls.make_task(
            (float(low), 0.0),
            (float(high), 0.0),
            weight=weight,
            **{
                "name": f"speed_range{float(low):+.2f}..{float(high):+.2f}",
                "gait_frequency_hz": 1.0,
                "gait_frequency_per_mps": 5.0,
                "pose_std": cls.POSE_STD_MOVING,
                **overrides,
            },
        )

    @classmethod
    def sidestep_task(
        cls, speed: float = 0.15, *, weight: float = 1.0, **overrides: Any
    ) -> MicroDuckTask:
        """Walk sideways at ``speed`` m/s, to the left (positive) or the right.

        The gait clock and the contact terms are the same as for forward
        walking; only the tracked velocity component changes.
        """
        return cls.make_task(
            (0.0, float(speed)),
            (0.0, float(speed)),
            weight=weight,
            **{
                "name": f"sidestep{float(speed):+.2f}",
                "pose_std": cls.POSE_STD_MOVING,
                **overrides,
            },
        )

    @classmethod
    def jump_task(cls, *, weight: float = 1.0, **overrides: Any) -> MicroDuckTask:
        """Hop in place under a zero command.

        Three terms shape the hop, in the order a policy discovers it.
        ``hop_rhythm`` (weight 1) pays, linearly up to the
        ``hop_velocity_amplitude`` of 0.1 m/s, for vertical base velocity in
        phase with the task clock (2 Hz), which starts a crouch-and-extend
        cycle from standing; ``launch`` (weight 30) pays for upward base
        velocity while both feet are planted, linearly up to
        ``launch_velocity_scale`` (0.5 m/s, take-off speed), which speeds the
        extension up until the feet leave the ground; ``jump`` (weight 10)
        pays for base height gained above the standing height while both feet
        are off the ground, in full from ``jump_target_height`` (5 mm) on, so
        the first real hop is reinforced hard. The gait terms and the
        vertical-velocity cost are off and the pose term is loose so the robot
        can crouch and extend.

        The MicroDuck servos (0.55 N m/rad, clipped at 0.96 N m, 0.74 kg
        robot) allow only a small hop. With these weights, a policy resumed
        from a walking one learned 2 Hz hops of about 2 cm, airborne 15% of
        the time, within 6M transitions with the jump row sampled three times
        as often as the others (``weight=3.0``); a dominant rhythm term
        instead produced a bob with the feet never leaving the ground.
        """
        reward_weights = dict.fromkeys(cls.GAIT_TERMS, 0.0)
        reward_weights.update(
            {
                "jump": cls.JUMP_WEIGHT,
                "launch": cls.LAUNCH_WEIGHT,
                "hop_rhythm": cls.HOP_RHYTHM_WEIGHT,
                "lin_vel_z": 0.0,
            }
        )
        reward_weights.update(overrides.pop("reward_weights", None) or {})
        return cls.make_task(
            (0.0, 0.0),
            (0.0, 0.0),
            weight=weight,
            reward_weights=reward_weights,
            **{
                "name": "jump",
                "pose_std": cls.POSE_STD_MOVING,
                "gait_frequency_hz": cls.HOP_FREQUENCY_HZ,
                **overrides,
            },
        )

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

    def _make_obs_spec(self) -> Composite:
        spec = Composite(
            observation=Unbounded(
                shape=(self.num_envs, self.OBSERVATION_DIM),
                dtype=self.dtype,
                device=self.device,
            ),
            command=Unbounded(
                shape=(self.num_envs, 2),
                dtype=self.dtype,
                device=self.device,
            ),
            task_id=Categorical(
                n=self.tasks.shape[0],
                shape=(self.num_envs, 1),
                dtype=torch.long,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )
        if not self.diagnostics:
            return spec
        for name in self.REWARD_TERMS:
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
        observation["command"] = self._command.clone()
        observation["task_id"] = self._task_id.unsqueeze(-1).clone()
        if self.diagnostics:
            observation.update(self._diagnostics(state, self._observation_action))
        return observation

    def _gait_clock(self) -> tuple[torch.Tensor, torch.Tensor]:
        elapsed_time = self._step_count.to(self.dtype) * (
            self.frame_skip * self._backend.timestep
        )
        frequency = (
            self._task.gait_frequency_hz
            + self._task.gait_frequency_per_mps * self._command.norm(dim=-1)
        )
        phase = self.GAIT_PHASE_OFFSET + 2.0 * math.pi * frequency * elapsed_time
        ramp = (elapsed_time / self.GAIT_RAMP_DURATION_S).clamp(max=1.0)
        return phase, ramp

    def _make_obs(self, state: TensorDictBase) -> torch.Tensor:
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        body_velocity = _body_frame_linear_velocity(qpos[..., 3:7], qvel[..., :3])
        phase, ramp = self._gait_clock()
        parts = [
            _projected_gravity(qpos[..., 3:7]),
            qvel[..., 3:6],
            body_velocity,
            self._command,
            qpos[..., 7:] - self._home_qpos[7:],
            qvel[..., 6:],
            phase.sin().unsqueeze(-1),
            phase.cos().unsqueeze(-1),
            ramp.unsqueeze(-1),
            self._observation_action,
        ]
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Reset: task pick-up, command and initial state
    # ------------------------------------------------------------------

    def _sample_initial_state(
        self,
        n: int,
        tensordict: TensorDictBase | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_id = self._sample_task_id(tensordict)
        task = self.tasks[task_id]
        command = self._sample_command(task, tensordict)
        self._pending = (task_id, task, command)
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
        joint_noise = task.joint_reset_noise_scale.to(qpos.dtype).unsqueeze(-1)
        unit_noise = (
            2.0
            * torch.rand(
                n,
                self.NUM_JOINTS,
                generator=self.rng,
                device=self.device,
                dtype=qpos.dtype,
            )
            - 1.0
        )
        qpos[..., 7:] += unit_noise * joint_noise
        # Warm start: push a fraction of the envs along the commanded planar
        # direction, expressed in the world frame through the body's forward
        # and left axes.
        low, high = task.warm_start_velocity.to(qvel.dtype).unbind(-1)
        speed = low + torch.rand(
            n, generator=self.rng, device=self.device, dtype=qvel.dtype
        ) * (high - low)
        planar_speed = command.norm(dim=-1)
        selected = (
            torch.rand(n, generator=self.rng, device=self.device, dtype=self.dtype)
            < task.warm_start_fraction
        ) & (planar_speed > self.COMMAND_THRESHOLD)
        unit = command / planar_speed.clamp_min(1e-8).unsqueeze(-1)
        quaternion = qpos[..., 3:7].to(self.dtype)
        world = unit[..., :1] * _body_forward_vector(quaternion) + unit[
            ..., 1:2
        ] * _body_lateral_vector(quaternion)
        warm = speed.unsqueeze(-1) * world.to(qvel.dtype)
        qvel[..., :3] = torch.where(selected.unsqueeze(-1), warm, qvel[..., :3])
        return qpos, qvel

    def _sample_task_id(self, tensordict: TensorDictBase | None) -> torch.Tensor:
        num_tasks = self.tasks.shape[0]
        if tensordict is not None and "task_id" in tensordict.keys():
            task_id = tensordict["task_id"].to(self.device)
            if task_id.shape == self.batch_size + torch.Size([1]):
                task_id = task_id.squeeze(-1)
            if task_id.shape != self.batch_size or task_id.is_floating_point():
                raise ValueError(
                    "A reset task_id must be an integer tensor of shape "
                    f"{tuple(self.batch_size)} or {tuple(self.batch_size + torch.Size([1]))}, "
                    f"got {task_id.dtype} of shape {tuple(task_id.shape)}."
                )
            task_id = task_id.long()
            if ((task_id < 0) | (task_id >= num_tasks)).any():
                raise ValueError(
                    f"A reset task_id must index the {num_tasks} tasks of the library."
                )
            return task_id
        return torch.multinomial(
            self.tasks.weight, self.num_envs, replacement=True, generator=self.rng
        )

    def _sample_command(
        self, task: MicroDuckTask, tensordict: TensorDictBase | None
    ) -> torch.Tensor:
        if tensordict is not None and "command" in tensordict.keys():
            command = tensordict["command"].to(device=self.device, dtype=self.dtype)
            if command.shape != self.batch_size + torch.Size([2]):
                raise ValueError(
                    "A reset command must have shape "
                    f"{tuple(self.batch_size + torch.Size([2]))} for (vx, vy), "
                    f"got {tuple(command.shape)}."
                )
            if not torch.isfinite(command).all():
                raise ValueError("A reset command must be finite.")
            return command
        fraction = torch.rand(
            self.num_envs, 2, generator=self.rng, device=self.device, dtype=self.dtype
        )
        return task.command_low + fraction * (task.command_high - task.command_low)

    def _consume_pending(
        self, tensordict: TensorDictBase | None
    ) -> tuple[torch.Tensor, MicroDuckTask, torch.Tensor]:
        """Return the task rows drawn with the initial state, or draw them now.

        A reset to a provided ``qpos``/``qvel`` snapshot skips
        :meth:`_sample_initial_state`, so nothing is pending in that case.
        """
        pending = self._pending
        self._pending = None
        if pending is not None:
            return pending
        task_id = self._sample_task_id(tensordict)
        task = self.tasks[task_id]
        return task_id, task, self._sample_command(task, tensordict)

    def _on_reset_all(self, tensordict: TensorDictBase | None = None) -> None:
        self._previous_action.zero_()
        self._observation_action.zero_()
        self._task_id, self._task, self._command = self._consume_pending(tensordict)
        self._feet_air_time.zero_()
        self._touchdown_air_time.zero_()
        self._refresh_contacts()

    def _on_reset_mask(
        self,
        mask: torch.Tensor,
        tensordict: TensorDictBase | None = None,
    ) -> None:
        mask = mask.squeeze(-1) if mask.ndim == 2 else mask
        column = mask.unsqueeze(-1)
        task_id, task, command = self._consume_pending(tensordict)
        self._previous_action = torch.where(
            column, torch.zeros_like(self._previous_action), self._previous_action
        )
        self._observation_action = torch.where(
            column, torch.zeros_like(self._observation_action), self._observation_action
        )
        # Only the rows being reset take the new task, id and command.
        self._task_id = torch.where(mask, task_id, self._task_id)
        self._task = task.where(mask, self._task)
        self._command = torch.where(column, command, self._command)
        self._feet_air_time = torch.where(
            column, torch.zeros_like(self._feet_air_time), self._feet_air_time
        )
        self._touchdown_air_time = torch.where(
            column, torch.zeros_like(self._touchdown_air_time), self._touchdown_air_time
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

    def reward_features(
        self, state: TensorDictBase, action: torch.Tensor
    ) -> TensorDictBase:
        """Return the per-step features every reward term reads.

        ``state`` is a simulator state (``qpos``/``qvel``) and ``action`` the
        normalized action that led to it; the contact bookkeeping is the
        env's current one. See :meth:`register_reward` for the entries.
        """
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        quaternion = qpos[..., 3:7]
        phase, _ = self._gait_clock()
        return TensorDict(
            {
                "body_velocity": _body_frame_linear_velocity(quaternion, qvel[..., :3]),
                "angular_velocity": qvel[..., 3:6],
                "upright": (-_projected_gravity(quaternion)[..., 2]).clamp(-1.0, 1.0),
                "base_height": qpos[..., 2],
                "standing_height": self._target_height.expand(qpos.shape[0]),
                "joint_error": qpos[..., 7:] - self._home_qpos[7:],
                "joint_velocity": qvel[..., 6:],
                "action": action.to(self.dtype),
                "previous_action": self._previous_action,
                "contacts": self._contacts,
                "foot_heights": self._foot_heights,
                "touchdown_air_time": self._touchdown_air_time,
                "gait_phase": phase,
                "command": self._command,
                "fallen": self._fallen(qpos, qvel),
            },
            batch_size=qpos.shape[:1],
            device=self.device,
        )

    def _reward_terms(self, features: TensorDictBase) -> torch.Tensor:
        """Evaluate every registered term: ``(num_envs, num_terms)``, unweighted."""
        dt = self.frame_skip * self._backend.timestep
        params = self._task.params
        columns = [
            term.fn(features, params).to(self.dtype) * (dt if term.per_second else 1.0)
            for term in self.REWARD_TERMS.values()
        ]
        return torch.stack(columns, dim=-1)

    def _reward_components(
        self,
        next_state: TensorDictBase,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        weighted = (
            self._reward_terms(self.reward_features(next_state, action))
            * self._task.reward_weights
        )
        return {
            f"diagnostic_reward_{name}": weighted[..., index : index + 1]
            for index, name in enumerate(self.REWARD_TERMS)
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
        terms = self._reward_terms(self.reward_features(next_state, action))
        return (terms * self._task.reward_weights).sum(dim=-1, keepdim=True)

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
            "task_id": self._task_id[index].clone(),
            "command": self._command[index].clone(),
            "feet_air_time": self._feet_air_time[index].clone(),
        }

    def _load_indexed_extra_state(self, state: dict[str, Any]) -> None:
        self._previous_action = state["previous_action"].clone()
        self._observation_action = self._previous_action.clone()
        self._task_id = state["task_id"].clone()
        self._task = self.tasks[self._task_id]
        self._command = state["command"].clone()
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
        self._task_id[index] = source._task_id.to(self.device)
        self._task = self.tasks[self._task_id]
        self._command[index] = source._command.to(self.device)
        self._feet_air_time[index] = source._feet_air_time.to(self.device)
        self._refresh_contacts()


# ----------------------------------------------------------------------
# Built-in reward terms. Weights are per second and multiplied by the control
# period, like the mjlab velocity tasks; positive terms are Gaussians in [0, 1].
# ----------------------------------------------------------------------


def _gait_gate(features: TensorDictBase) -> torch.Tensor:
    """Contact terms only pay while the torso is upright enough to be stepping."""
    return (features["upright"] >= MicroDuckEnv.MIN_UPRIGHT).to(
        features["upright"].dtype
    )


def _progress_fraction(features: TensorDictBase) -> torch.Tensor:
    """Body velocity along the commanded direction as a fraction of the command.

    Clipped to ``[-1, 1]``; zero under a command below the standing threshold,
    which has no direction.
    """
    command = features["command"]
    speed = command.norm(dim=-1)
    unit = command / speed.clamp_min(1e-8).unsqueeze(-1)
    along = (features["body_velocity"][..., :2] * unit).sum(-1)
    fraction = (along / speed.clamp_min(1e-8)).clamp(-1.0, 1.0)
    return torch.where(
        speed > MicroDuckEnv.COMMAND_THRESHOLD, fraction, torch.zeros_like(fraction)
    )


def _stepping_gate(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    """Gait credit scaled by progress along the command, with a floor.

    Stepping in place keeps ``gait_progress_floor`` of the credit so a policy
    can still leave the standing optimum; full credit needs the body to move
    where the command points, so drifting sideways or forward under a
    sidestep command no longer collects the gait terms for free.
    """
    floor = params["gait_progress_floor"]
    progress = _progress_fraction(features).clamp_min(0.0)
    return _gait_gate(features) * (floor + (1.0 - floor) * progress)


def _directed_clock(features: TensorDictBase) -> torch.Tensor:
    """Sine of the gait clock, mirrored for backward commands.

    Forward and sideways gaits swing the left foot on the positive half of the
    clock; walking backward mirrors the pattern.
    """
    direction = torch.where(features["command"][..., 0] < 0, -1.0, 1.0)
    return direction.to(features["gait_phase"].dtype) * features["gait_phase"].sin()


@MicroDuckEnv.register_reward(
    "tracking", weight=2.0, tracking_std=0.1, tracking_off_axis_std=0.05
)
def _tracking(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    # Anisotropic Gaussian on the planar velocity error: the width across the
    # commanded direction is tighter than along it, so moving diagonally earns
    # less than an on-axis error of the same size. A zero command has no axis
    # and falls back to the isotropic Gaussian.
    command = features["command"]
    error = features["body_velocity"][..., :2] - command
    speed = command.norm(dim=-1)
    unit = command / speed.clamp_min(1e-8).unsqueeze(-1)
    along = (error * unit).sum(-1)
    total = error.square().sum(-1)
    across = (total - along.square()).clamp_min(0.0)
    anisotropic = (
        along.square() / params["tracking_std"].square()
        + across / params["tracking_off_axis_std"].square()
    )
    isotropic = total / params["tracking_std"].square()
    return torch.exp(
        -torch.where(speed > MicroDuckEnv.COMMAND_THRESHOLD, anisotropic, isotropic)
    )


@MicroDuckEnv.register_reward("yaw_rate", weight=1.0, yaw_rate_std=0.5**0.5)
def _yaw_rate(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    return torch.exp(
        -features["angular_velocity"][..., 2].square() / params["yaw_rate_std"].square()
    )


@MicroDuckEnv.register_reward("upright", weight=2.0, upright_std=0.05**0.5)
def _upright(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    tilt = torch.acos(features["upright"])
    return torch.exp(-tilt.square() / params["upright_std"].square())


@MicroDuckEnv.register_reward("pose", weight=1.0, pose_std=0.5)
def _pose(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    return torch.exp(
        -features["joint_error"].square().mean(dim=-1) / params["pose_std"].square()
    )


@MicroDuckEnv.register_reward("progress", weight=2.0)
def _progress(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    # Linear in the velocity along the command, so the gradient toward moving
    # where asked does not vanish at zero speed the way the Gaussian does, and
    # moving against the command costs.
    return _progress_fraction(features)


@MicroDuckEnv.register_reward(
    "air_time",
    weight=3.0,
    air_time_min=0.125,
    air_time_max=0.3,
    gait_progress_floor=0.5,
)
def _air_time(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    window = (params["air_time_max"] - params["air_time_min"]).unsqueeze(-1)
    credited = torch.minimum(
        (
            features["touchdown_air_time"] - params["air_time_min"].unsqueeze(-1)
        ).clamp_min(0.0),
        window,
    )
    return credited.sum(dim=-1) * _stepping_gate(features, params)


@MicroDuckEnv.register_reward("swing_height", weight=2.0, swing_target_height=0.02)
def _swing_height(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    # The clock's swing foot is rewarded for any lift toward the clearance
    # target, contact or not, so the incentive to step is dense.
    directed_sin = _directed_clock(features)
    swing_foot = torch.stack((directed_sin > 0, directed_sin <= 0), dim=-1)
    lift = (
        features["foot_heights"] / params["swing_target_height"].unsqueeze(-1)
    ).clamp(0.0, 1.0)
    return (lift * swing_foot).sum(dim=-1) * _stepping_gate(features, params)


@MicroDuckEnv.register_reward("phase_contact", weight=3.0)
def _phase_contact(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    # Left foot swings while the directed clock is positive. Credit is given
    # only for correct single support, so standing on both feet earns nothing.
    directed_sin = _directed_clock(features)
    expected = torch.stack((directed_sin <= 0, directed_sin > 0), dim=-1)
    correct = (features["contacts"] == expected).all(dim=-1)
    return correct.to(directed_sin.dtype) * _stepping_gate(features, params)


@MicroDuckEnv.register_reward("double_support", weight=-1.0)
def _double_support(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    planted = features["contacts"].all(dim=-1).to(features["upright"].dtype)
    return planted * _stepping_gate(features, params)


@MicroDuckEnv.register_reward("ang_vel_xy", weight=-0.05)
def _ang_vel_xy(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    return features["angular_velocity"][..., :2].square().sum(-1)


@MicroDuckEnv.register_reward("lin_vel_z", weight=-2.0)
def _lin_vel_z(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    return features["body_velocity"][..., 2].square()


@MicroDuckEnv.register_reward("action_rate", weight=-0.1)
def _action_rate(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    return (features["action"] - features["previous_action"]).square().sum(dim=-1)


@MicroDuckEnv.register_reward("joint_velocity", weight=-0.001)
def _joint_velocity(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    return features["joint_velocity"].square().sum(-1)


@MicroDuckEnv.register_reward("hop_rhythm", weight=0.0, hop_velocity_amplitude=0.1)
def _hop_rhythm(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    # Vertical base velocity in phase with the task clock (up on the positive
    # half of the cosine, down on the negative half), linear in the speed as a
    # fraction of the amplitude and clipped at one. A Gaussian on a reference
    # velocity paid standing still a third of its value and nothing for a
    # small bob; this term pays nothing for standing still and grows with
    # every bit of crouch-and-extend motion on the beat.
    beat = features["gait_phase"].cos().sign()
    fraction = (
        features["body_velocity"][..., 2] * beat / params["hop_velocity_amplitude"]
    )
    return fraction.clamp(-1.0, 1.0) * _gait_gate(features)


@MicroDuckEnv.register_reward("launch", weight=0.0, launch_velocity_scale=0.5)
def _launch(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    # Upward base velocity while both feet are planted, as a fraction of
    # ``launch_velocity_scale``: the take-off of a hop, which the airborne
    # gated jump term cannot see. The scale sits well above the rhythm
    # amplitude so a faster extension keeps paying up to take-off speed.
    planted = features["contacts"].all(dim=-1).to(features["upright"].dtype)
    upward = (
        features["body_velocity"][..., 2] / params["launch_velocity_scale"]
    ).clamp(0.0, 1.0)
    return upward * planted * _gait_gate(features)


@MicroDuckEnv.register_reward("jump", weight=0.0, jump_target_height=0.005)
def _jump(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    # Height gained with both feet in the air, so hopping in place beats
    # standing tall on the toes.
    gain = (
        (features["base_height"] - features["standing_height"])
        / params["jump_target_height"]
    ).clamp(0.0, 1.0)
    airborne = (~features["contacts"].any(dim=-1)).to(gain.dtype)
    return gain * airborne * _gait_gate(features)


@MicroDuckEnv.register_reward(
    "termination", weight=-MicroDuckEnv.FALL_PENALTY, per_second=False
)
def _termination(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    return features["fallen"].to(features["upright"].dtype)


class MicroDuckTaskSampler(Transform):
    """Write a :class:`MicroDuckEnv` ``task_id`` per env at reset from a weighted mixture.

    The env draws a task per env on its own, weighted by the ``weight`` field
    of its library; this transform replaces that draw with weights of its own,
    for a mixture that changes during training (a curriculum) or to pin one
    task for evaluation. At every reset it draws an index per env with the
    given weights and writes it under ``task_id_key`` in the reset TensorDict,
    which :class:`~torchrl.envs.TransformedEnv` forwards to the env because
    the key is in its ``state_spec``. Partial resets only replace the tasks of
    the envs being reset; the env checks the indices against its library.

    Args:
        weights (Sequence[float] or torch.Tensor): one non-negative sampling
            weight per task of the env's library, in library order.

    Keyword Args:
        task_id_key (NestedKey, optional): key written in the reset
            TensorDict. Defaults to ``"task_id"``.
        seed (int, optional): seed of the sampler's generator. Inside a
            :class:`~torchrl.envs.TransformedEnv`, ``env.set_seed`` seeds it
            as well.

    Examples:
        >>> import torch
        >>> from torchrl.envs import MicroDuckEnv, MicroDuckTaskSampler, TransformedEnv
        >>> library = [MicroDuckEnv.standing_task(), MicroDuckEnv.tracking_task(0.2), MicroDuckEnv.jump_task()]
        >>> sampler = MicroDuckTaskSampler([1.0, 2.0, 1.0], seed=0)
        >>> sampler.sample(torch.Size([6]))[:, 0]
        tensor([2, 1, 1, 2, 1, 2])
        >>> env = TransformedEnv(  # doctest: +SKIP
        ...     MicroDuckEnv(download=True, num_envs=16, tasks=library), sampler
        ... )
        >>> env.rollout(50)["task_id"][:, 0, 0]  # one task per env  # doctest: +SKIP
        >>> sampler.probabilities.copy_(torch.tensor([0.0, 0.0, 1.0]))  # doctest: +SKIP
    """

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        *,
        task_id_key: NestedKey = "task_id",
        seed: int | None = None,
    ):
        super().__init__(in_keys=[], out_keys=[], in_keys_inv=[], out_keys_inv=[])
        weights = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
        if weights.numel() == 0 or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(
                "weights must be a non-empty sequence of non-negative values that "
                "do not all vanish."
            )
        self.task_id_key = task_id_key
        self.register_buffer("probabilities", weights / weights.sum())
        self.rng: torch.Generator | None = None
        self._set_seed(seed)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            self.rng = None
            return
        self.rng = torch.Generator(device=self.probabilities.device)
        self.rng.manual_seed(int(seed))

    def sample(self, batch_size: torch.Size) -> torch.Tensor:
        """Draw one task index per element of ``batch_size``, shape ``(*batch_size, 1)``."""
        n = int(torch.Size(batch_size).numel())
        index = torch.multinomial(
            self.probabilities, n, replacement=True, generator=self.rng
        )
        return index.reshape(*batch_size, 1)

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        batch_size = (
            self.parent.batch_size if self.parent is not None else torch.Size([])
        )
        if tensordict is None:
            tensordict = TensorDict(
                batch_size=batch_size, device=self.probabilities.device
            )
        tensordict.set(self.task_id_key, self.sample(batch_size).to(tensordict.device))
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return tensordict_reset

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return next_tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict
