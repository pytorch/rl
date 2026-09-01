# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Backend-neutral MicroDuck velocity task and a compact TorchRL PPO loop.

The MicroDuck MJCF is shared by all three :class:`~torchrl.envs.MujocoEnv`
backends. This example supplies a small commanded locomotion task around that
model: normalized joint-position actions, a 50-value proprioceptive
observation, and signed longitudinal-velocity tracking. A zero velocity command
requests standing, while positive and negative velocities command forward and
backward motion. It intentionally does not reproduce the richer walking task from
``microduck_rl`` (sensor history, delays, curricula, domain randomization, and
the BAM actuator model).

Run a short CPU training job from a TorchRL checkout::

    python examples/mujoco/ppo_microduck.py \
        --microduck-root /path/to/microduck_rl --smoke

Use ``--backend mjx`` or ``--backend mujoco-torch`` to keep the same task and
change only the simulator backend. The accompanying notebook demonstrates
native rendering and the interactive MuJoCo WASM viewer.
"""

from __future__ import annotations

import argparse
import functools as ft
import importlib.util
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    NormalParamExtractor,
    TensorDictModule,
    TensorDictSequential,
)
from torch import nn

from torchrl import torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import (
    Bounded,
    Composite,
    LazyTensorStorage,
    SliceSampler,
    TensorDictReplayBuffer,
    Unbounded,
)
from torchrl.envs import EnvBase, ExplorationType, MujocoEnv, set_exploration_type
from torchrl.envs.utils import step_mdp
from torchrl.modules import (
    GRUModule,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
    set_recurrent_mode,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import WandbLogger

_has_mujoco = importlib.util.find_spec("mujoco") is not None
_has_mujoco_torch = importlib.util.find_spec("mujoco_torch") is not None
_has_mjx = importlib.util.find_spec("mujoco.mjx") is not None
_has_psutil = importlib.util.find_spec("psutil") is not None
_psutil_process = None

Backend = Literal["mujoco", "mjx", "mujoco-torch"]
NUM_JOINTS = 14
OBSERVATION_DIM = 3 + 3 + 2 + NUM_JOINTS * 3
VELOCITY_TRACKING_STD = 0.25
STAND_POSE_COMMAND_STD = 0.1
DEFAULT_COMMANDS = (-0.3, 0.0, 0.3)
MAX_EPISODE_STEPS = 500
DEFAULT_REPLAY_CAPACITY = 16_384


class _CompleteTrajectoryReplayBuffer(TensorDictReplayBuffer):
    """Reject a trajectory rather than partially overwriting an on-policy batch."""

    def __init__(self, *args: Any, collection_capacity: int, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.collection_capacity = collection_capacity
        self.collection_full = False
        self._validate_single_trajectory = True

    def extend(
        self,
        data: TensorDictBase,
        *,
        update_priority: bool | None = None,
    ) -> torch.Tensor:
        if self._validate_single_trajectory:
            done = data.get(("next", "done"))
            if data.ndim != 1 or not bool(done[-1].all()) or bool(done[:-1].any()):
                raise RuntimeError(
                    "The collector must write one complete trajectory per replay "
                    "buffer extend call."
                )
            if (
                self.collection_full
                or len(self) + data.numel() > self.collection_capacity
            ):
                self.collection_full = True
                return torch.zeros((0, 1), dtype=torch.long)
        return super().extend(data, update_priority=update_priority)

    def replace_with_processed_batch(self, batch: TensorDictBase) -> None:
        """Replace raw collection data by the same data augmented with GAE targets."""
        self.empty()
        self._validate_single_trajectory = False
        try:
            self.extend(batch)
        finally:
            self._validate_single_trajectory = True
            self.collection_full = False

    def empty(self, empty_write_count: bool = True) -> None:
        super().empty(empty_write_count=empty_write_count)
        self.collection_full = False


def _metric_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().cpu()
    return float(value)


def _telemetry_metrics(device: torch.device) -> dict[str, float]:
    global _psutil_process
    metrics = {}
    if _has_psutil:
        import psutil

        if _psutil_process is None:
            _psutil_process = psutil.Process()
        memory = _psutil_process.memory_info()
        system_memory = psutil.virtual_memory()
        metrics.update(
            {
                "telemetry/process_rss_gb": memory.rss / 1e9,
                "telemetry/process_cpu_percent": _psutil_process.cpu_percent(),
                "telemetry/process_threads": float(_psutil_process.num_threads()),
                "telemetry/system_memory_percent": float(system_memory.percent),
                "telemetry/system_memory_available_gb": system_memory.available / 1e9,
            }
        )
    if device.type == "cuda":
        metrics.update(
            {
                "telemetry/device_allocated_gb": torch.cuda.memory_allocated(device)
                / 1e9,
                "telemetry/device_reserved_gb": torch.cuda.memory_reserved(device)
                / 1e9,
                "telemetry/device_max_allocated_gb": torch.cuda.max_memory_allocated(
                    device
                )
                / 1e9,
            }
        )
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        metrics["telemetry/device_allocated_gb"] = (
            torch.mps.current_allocated_memory() / 1e9
        )
        metrics["telemetry/device_driver_allocated_gb"] = (
            torch.mps.driver_allocated_memory() / 1e9
        )
    return metrics


def _collection_metrics(batch: TensorDictBase) -> tuple[dict[str, float], int]:
    rewards = batch.get(("next", "reward")).squeeze(-1)
    done = batch.get(("next", "done")).squeeze(-1).bool()
    terminated = batch.get(("next", "terminated")).squeeze(-1).bool()
    commands = batch["commanded_x_velocity"].squeeze(-1)
    measured_velocity = batch["observation"][..., 6]
    ends = done.nonzero(as_tuple=False).squeeze(-1).tolist()
    starts = [0, *(end + 1 for end in ends[:-1])]
    returns = torch.stack(
        [rewards[start : end + 1].sum() for start, end in zip(starts, ends)]
    )
    lengths = torch.tensor(
        [end - start + 1 for start, end in zip(starts, ends)], dtype=torch.float32
    )
    end_indices = torch.tensor(ends, dtype=torch.long, device=terminated.device)
    metrics = {
        "collection/reward_mean": _metric_float(rewards.mean()),
        "collection/reward_std": _metric_float(rewards.std(unbiased=False)),
        "collection/tracking_error_mean": _metric_float(
            (measured_velocity - commands).abs().mean()
        ),
        "episode/return_mean": _metric_float(returns.mean()),
        "episode/return_std": _metric_float(returns.std(unbiased=False)),
        "episode/length_mean": _metric_float(lengths.mean()),
        "episode/length_min": _metric_float(lengths.min()),
        "episode/length_max": _metric_float(lengths.max()),
        "episode/survival_rate": _metric_float(
            (~terminated[end_indices]).float().mean()
        ),
    }
    policy_scale = batch.get("scale")
    if policy_scale is not None:
        metrics.update(
            {
                "policy/scale_mean": _metric_float(policy_scale.mean()),
                "policy/scale_min": _metric_float(policy_scale.min()),
                "policy/scale_max": _metric_float(policy_scale.max()),
            }
        )
    return metrics, len(ends)


def resolve_microduck_scene(
    microduck_root: str | Path | None = None,
) -> Path:
    """Resolve MicroDuck's ``scene_walk.xml`` from a checkout or installation.

    Args:
        microduck_root: ``microduck_rl`` checkout, its Python package directory,
            or the scene XML itself. When omitted, an installed
            ``mjlab_microduck`` package and nearby ``microduck_rl`` checkouts are
            considered.

    Returns:
        Absolute path to ``scene_walk.xml``.

    Raises:
        FileNotFoundError: If the MicroDuck MJCF cannot be located.
    """
    candidates: list[Path] = []
    if microduck_root is not None:
        candidates.append(Path(microduck_root).expanduser())

    spec = importlib.util.find_spec("mjlab_microduck")
    if spec is not None and spec.origin is not None:
        candidates.append(Path(spec.origin).resolve().parent)

    cwd = Path.cwd()
    candidates.extend((cwd / "microduck_rl", cwd.parent / "microduck_rl"))
    suffixes = (
        Path("scene_walk.xml"),
        Path("robot/microduck/scene_walk.xml"),
        Path("mjlab_microduck/robot/microduck/scene_walk.xml"),
        Path("src/mjlab_microduck/robot/microduck/scene_walk.xml"),
    )
    attempted = []
    for candidate in candidates:
        paths = (
            (candidate,)
            if candidate.suffix == ".xml"
            else (candidate / suffix for suffix in suffixes)
        )
        for path in paths:
            attempted.append(path)
            if path.is_file():
                return path.resolve()
    detail = "\n".join(f"  - {path}" for path in attempted)
    raise FileNotFoundError(
        "Could not find MicroDuck's scene_walk.xml. Pass --microduck-root or "
        "install microduck_rl. Tried:\n" + detail
    )


@contextmanager
def _low_cost_collision_scene(scene_path: Path):
    """Replace detailed collision meshes with tight box proxies at load time.

    The upstream walking asset reuses render meshes for the feet and
    self-collision geoms. Accelerated MuJoCo implementations expand every pair
    of convex-hull edges, which makes the two roughly 10,000-edge soles
    prohibitively expensive to compile or step in a batch. Visual meshes stay
    untouched; only geoms explicitly assigned to the ``collision`` or
    ``self_collision_only`` classes are replaced.

    Direct, self-contained MJCF files without an ``<include>`` are yielded
    unchanged. This keeps small fixtures and custom MicroDuck-compatible files
    working without imposing the upstream asset layout.
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

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    robot_tree = ET.parse(robot_path)
    robot_root = robot_tree.getroot()
    proxy_count = 0
    for geom in robot_root.iter("geom"):
        if geom.get("class") not in {"collision", "self_collision_only"}:
            continue
        mesh_name = geom.get("mesh")
        if mesh_name is None:
            continue
        mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        if mesh_id < 0:
            raise ValueError(f"Could not resolve collision mesh {mesh_name!r}.")
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        vertices = torch.from_numpy(
            model.mesh_vert[vertex_start : vertex_start + vertex_count].copy()
        )
        lower = vertices.amin(dim=0)
        upper = vertices.amax(dim=0)
        center = (lower + upper) / 2
        half_size = ((upper - lower) / 2).clamp_min(1e-4)

        position = torch.tensor(
            [float(value) for value in geom.get("pos", "0 0 0").split()]
        )
        quaternion = torch.tensor(
            [float(value) for value in geom.get("quat", "1 0 0 0").split()]
        )
        quaternion = quaternion / quaternion.norm().clamp_min(1e-8)
        w = quaternion[0]
        vector = quaternion[1:]
        rotated_center = (
            center
            + 2 * w * torch.cross(vector, center, dim=0)
            + 2 * torch.cross(vector, torch.cross(vector, center, dim=0), dim=0)
        )
        position = position + rotated_center

        geom.set("type", "box")
        geom.set("pos", " ".join(f"{value:.9g}" for value in position.tolist()))
        geom.set("size", " ".join(f"{value:.9g}" for value in half_size.tolist()))
        del geom.attrib["mesh"]
        proxy_count += 1

    if not proxy_count:
        yield scene_path
        return

    compiler = robot_root.find("compiler")
    if compiler is not None:
        for attribute in ("meshdir", "texturedir"):
            directory = compiler.get(attribute)
            if directory is not None and not Path(directory).is_absolute():
                compiler.set(attribute, str((robot_path.parent / directory).resolve()))

    with TemporaryDirectory(prefix="torchrl-microduck-") as directory:
        directory = Path(directory)
        patched_robot = directory / robot_path.name
        patched_scene = directory / scene_path.name
        robot_tree.write(patched_robot, encoding="unicode")
        scene_tree.write(patched_scene, encoding="unicode")
        yield patched_scene


def _load_stand_metadata(scene_path: Path) -> tuple[torch.Tensor, ...]:
    if not _has_mujoco:
        raise ImportError(
            "MicroDuckVelocityEnv requires the `mujoco` package to read MJCF "
            "metadata, including when another physics backend is selected."
        )
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if (model.nq, model.nv, model.nu) != (21, 20, NUM_JOINTS):
        raise ValueError(
            "Expected the 14-actuator MicroDuck walking model with "
            f"(nq, nv, nu)=(21, 20, 14), got {(model.nq, model.nv, model.nu)}."
        )
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if key_id < 0:
        raise ValueError("MicroDuck MJCF must define a `STAND` keyframe.")

    home_qpos = torch.from_numpy(model.key_qpos[key_id].copy())
    home_ctrl = torch.from_numpy(model.key_ctrl[key_id].copy())
    joint_ids = torch.from_numpy(model.actuator_trnid[:, 0].copy()).long()
    if (joint_ids < 0).any():
        raise ValueError("Every MicroDuck actuator must target a joint.")
    joint_limited = torch.from_numpy(model.jnt_limited[joint_ids.numpy()].copy()).bool()
    joint_range = torch.from_numpy(model.jnt_range[joint_ids.numpy()].copy())
    joint_low = torch.where(joint_limited, joint_range[:, 0], home_ctrl - torch.pi)
    joint_high = torch.where(joint_limited, joint_range[:, 1], home_ctrl + torch.pi)
    return home_qpos, home_ctrl, joint_low, joint_high


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


class MicroDuckVelocityEnv(MujocoEnv):
    """A compact commanded-velocity task for the MicroDuck MJCF.

    Actions are normalized offsets around the actuator targets in the
    ``STAND`` keyframe. Observations concatenate projected gravity, base
    angular velocity, measured and commanded body-frame longitudinal velocity,
    joint-position error, joint velocity, and the previous action. The reward
    prioritizes signed velocity tracking while uprightness and target height
    stabilize the motion. A nominal-pose reward applies only near a zero
    command and smoothly vanishes before locomotion commands.

    Args:
        microduck_root: ``microduck_rl`` checkout, package directory, or
            ``scene_walk.xml`` path.
        backend: MuJoCo physics backend.
        commanded_x_velocity: A fixed body-frame longitudinal velocity command,
            or a sequence sampled uniformly at every reset. The default exact
            zero and signed commands train standing, forward, and backward
            motion in one task.
        action_scale: Maximum position-target offset in radians for a unit
            normalized action.
        kwargs: Forwarded to :class:`~torchrl.envs.MujocoEnv`.
    """

    FRAME_SKIP = 4
    RESET_NOISE_SCALE = 0.02

    def __init__(
        self,
        microduck_root: str | Path | None = None,
        *,
        backend: Backend = "mujoco",
        commanded_x_velocity: float | Sequence[float] = (-0.3, 0.0, 0.3),
        action_scale: float = 0.35,
        **kwargs: Any,
    ):
        scene_path = resolve_microduck_scene(microduck_root)
        home_qpos, home_ctrl, joint_low, joint_high = _load_stand_metadata(scene_path)
        with _low_cost_collision_scene(scene_path) as physics_scene:
            super().__init__(
                xml_path=physics_scene,
                patch_xml=False,
                backend=backend,
                **kwargs,
            )
        self.scene_path = scene_path
        self.action_scale = float(action_scale)
        self._home_qpos = home_qpos.to(device=self.device, dtype=self.dtype)
        self._home_ctrl = home_ctrl.to(device=self.device, dtype=self.dtype)
        self._joint_low = joint_low.to(device=self.device, dtype=self.dtype)
        self._joint_high = joint_high.to(device=self.device, dtype=self.dtype)
        self._target_height = self._home_qpos[2].clone()
        self._previous_action = torch.zeros(
            self.num_envs, NUM_JOINTS, dtype=self.dtype, device=self.device
        )
        self._observation_action = self._previous_action.clone()
        command_values = torch.as_tensor(
            commanded_x_velocity, dtype=self.dtype, device=self.device
        )
        if command_values.ndim == 0:
            command_values = command_values.unsqueeze(0)
        if command_values.ndim != 1 or command_values.numel() == 0:
            raise ValueError(
                "commanded_x_velocity must be a scalar or a non-empty 1-D sequence."
            )
        if not torch.isfinite(command_values).all():
            raise ValueError("commanded_x_velocity values must be finite.")
        self._command_values = command_values
        self._commanded_x_velocity = torch.zeros(
            self.num_envs, 1, dtype=self.dtype, device=self.device
        )
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, NUM_JOINTS),
            dtype=self.dtype,
            device=self.device,
        )

    def _make_obs_spec(self) -> Composite:
        return Composite(
            observation=Unbounded(
                shape=(self.num_envs, OBSERVATION_DIM),
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

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        observation = super()._build_obs_dict(state)
        observation["commanded_x_velocity"] = self._commanded_x_velocity.clone()
        return observation

    def _make_obs(self, state: TensorDictBase) -> torch.Tensor:
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        body_velocity = _body_frame_linear_velocity(qpos[..., 3:7], qvel[..., :3])
        return torch.cat(
            (
                _projected_gravity(qpos[..., 3:7]),
                qvel[..., 3:6],
                body_velocity[..., :1],
                self._commanded_x_velocity,
                qpos[..., 7:] - self._home_qpos[7:],
                qvel[..., 6:],
                self._observation_action,
            ),
            dim=-1,
        )

    def _sample_initial_state(
        self,
        n: int,
        tensordict: TensorDictBase | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del tensordict
        qpos = self._home_qpos.unsqueeze(0).expand(n, -1).clone()
        qvel = torch.zeros(
            n, self._backend.nv, dtype=self._backend.qvel0.dtype, device=self.device
        )
        qpos = qpos.to(dtype=self._backend.qpos0.dtype)
        if self.reset_noise_scale > 0:
            qpos[..., :2] += torch.empty_like(qpos[..., :2]).uniform_(
                -self.reset_noise_scale,
                self.reset_noise_scale,
                generator=self.rng,
            )
            qpos[..., 7:] += torch.empty_like(qpos[..., 7:]).uniform_(
                -self.reset_noise_scale,
                self.reset_noise_scale,
                generator=self.rng,
            )
            qvel += torch.empty_like(qvel).uniform_(
                -self.reset_noise_scale,
                self.reset_noise_scale,
                generator=self.rng,
            )
        return qpos, qvel

    def _sample_command(
        self,
        tensordict: TensorDictBase | None,
    ) -> torch.Tensor:
        if tensordict is not None and "commanded_x_velocity" in tensordict:
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

    def _prepare_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(-1.0, 1.0)
        target = self._home_ctrl + self.action_scale * action
        return target.clamp(self._joint_low, self._joint_high)

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        qpos = next_state["qpos"].to(self.dtype)
        qvel = next_state["qvel"].to(self.dtype)
        projected_gravity = _projected_gravity(qpos[..., 3:7])
        # MuJoCo free-joint linear velocities are world-frame. Angular
        # velocities are already expressed in the local body frame.
        body_velocity = _body_frame_linear_velocity(qpos[..., 3:7], qvel[..., :3])
        velocity_tracking_reward = torch.exp(
            -(
                (body_velocity[..., 0] - self._commanded_x_velocity.squeeze(-1))
                / VELOCITY_TRACKING_STD
            ).square()
        )
        upright = (-projected_gravity[..., 2]).clamp(-1.0, 1.0)
        upright_reward = torch.exp(-4.0 * (1.0 - upright).square())
        height_reward = torch.exp(
            -((qpos[..., 2] - self._target_height) / 0.03).square()
        )
        pose_reward = torch.exp(
            -((qpos[..., 7:] - self._home_qpos[7:]) / 0.35).square().mean(dim=-1)
        )
        stand_pose_gate = torch.exp(
            -(self._commanded_x_velocity.squeeze(-1) / STAND_POSE_COMMAND_STD).square()
        )
        lateral_velocity_cost = body_velocity[..., 1].square()
        roll_yaw_velocity_cost = qvel[..., [3, 5]].square().mean(dim=-1)
        joint_velocity_cost = qvel[..., 6:].square().mean(dim=-1)
        action_rate_cost = (action - self._previous_action).square().mean(dim=-1)
        return (
            2.0 * velocity_tracking_reward
            + 0.5 * upright_reward
            + 0.25 * height_reward
            + 0.5 * stand_pose_gate * pose_reward
            - 0.1 * lateral_velocity_cost
            - 0.02 * roll_yaw_velocity_cost
            - 0.002 * joint_velocity_cost
            - 0.02 * action_rate_cost
        ).unsqueeze(-1)

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        qpos = next_state["qpos"].to(self.dtype)
        qvel = next_state["qvel"].to(self.dtype)
        upright = -_projected_gravity(qpos[..., 3:7])[..., 2]
        finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
        return (
            (qpos[..., 2] < 0.55 * self._target_height) | (upright < 0.35) | ~finite
        ).unsqueeze(-1)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._observation_action = tensordict["action"].to(self.dtype)
        result = super()._step(tensordict)
        self._previous_action = self._observation_action.clone()
        return result

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
        if not isinstance(source, MicroDuckVelocityEnv):
            raise TypeError(
                "MicroDuckVelocityEnv snapshots can only be restored from the same task."
            )
        self._previous_action[index] = source._previous_action.to(self.device)
        self._observation_action[index] = source._observation_action.to(self.device)
        self._commanded_x_velocity[index] = source._commanded_x_velocity.to(self.device)


def make_env(
    microduck_root: str | Path | None = None,
    *,
    backend: Backend = "mujoco",
    commanded_x_velocity: float | Sequence[float] = (-0.3, 0.0, 0.3),
    num_envs: int = 4,
    device: torch.device | str = "cpu",
    seed: int = 0,
    parallel: bool = False,
) -> EnvBase:
    """Build a batched MicroDuck commanded-velocity task.

    Native MuJoCo uses :class:`~torchrl.envs.SerialEnv` by default for a
    notebook-friendly, multiprocessing-free batch. MJX and ``mujoco-torch``
    batch directly in their respective array frameworks.
    """
    kwargs = {
        "microduck_root": microduck_root,
        "backend": backend,
        "commanded_x_velocity": commanded_x_velocity,
        "num_envs": num_envs,
        "device": torch.device(device),
        "seed": seed,
        "max_episode_steps": 500,
    }
    if backend == "mujoco":
        kwargs["parallel"] = parallel
    return MicroDuckVelocityEnv(**kwargs)


def make_models(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    hidden_size: int = 128,
    initial_policy_scale: float = 0.2,
) -> tuple[ProbabilisticActor, ValueOperator, TensorDictSequential]:
    """Create a GRU actor and value head sharing the recurrent backbone.

    The actor starts with zero deterministic actions and a modest exploration
    scale so its initial rollouts stay near the ``STAND`` actuator targets. The
    third returned module is the complete recurrent value network used by GAE
    and PPO; the second is its non-shared value head.
    """
    if not math.isfinite(initial_policy_scale) or initial_policy_scale <= 0:
        raise ValueError("initial_policy_scale must be finite and positive.")
    device = torch.device(device)
    action_dim = env.action_spec_unbatched.shape[-1]
    embed = TensorDictModule(
        nn.Sequential(
            nn.LazyLinear(hidden_size, device=device),
            nn.Tanh(),
        ),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    gru = GRUModule(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        in_keys=["embed", "recurrent_state", "is_init"],
        out_keys=["gru_out", ("next", "recurrent_state")],
        device=device,
    )
    backbone = TensorDictSequential(embed, gru)
    actor_output = nn.Linear(hidden_size, 2 * action_dim, device=device)
    nn.init.zeros_(actor_output.weight)
    nn.init.zeros_(actor_output.bias)
    actor_head = TensorDictModule(
        nn.Sequential(
            actor_output,
            NormalParamExtractor(
                scale_mapping=f"biased_softplus_{initial_policy_scale}"
            ),
        ),
        in_keys=["gru_out"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=TensorDictSequential(backbone, actor_head),
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec_unbatched.space.low.to(device),
            "high": env.action_spec_unbatched.space.high.to(device),
        },
        return_log_prob=True,
    ).to(device)
    value_feature = TensorDictModule(
        nn.Identity(), in_keys=["gru_out"], out_keys=["value_gru_out"]
    )
    critic = ValueOperator(
        nn.Linear(hidden_size, 1, device=device),
        in_keys=["value_gru_out"],
    ).to(device)
    full_value = TensorDictSequential(backbone, value_feature, critic)
    with torch.no_grad():
        fake_tensordict = env.fake_tensordict().to(device)
        fake_tensordict.set(
            "is_init",
            torch.ones(
                *fake_tensordict.batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
        )
        fake_tensordict.set(
            "recurrent_state",
            torch.zeros(
                *fake_tensordict.batch_size,
                1,
                hidden_size,
                device=device,
            ),
        )
        actor(fake_tensordict)
        full_value(fake_tensordict)
    return actor, critic, full_value


def _prepare_render_recurrent_state(
    _module: nn.Module,
    inputs: tuple[TensorDictBase, ...],
    *,
    hidden_size: int,
) -> None:
    tensordict = inputs[0]
    observation = tensordict.get("observation")
    recurrent_state = tensordict.get("recurrent_state", None)
    is_init = recurrent_state is None
    if is_init:
        tensordict.set(
            "recurrent_state",
            observation.new_zeros(*tensordict.batch_size, 1, hidden_size),
        )
    tensordict.set(
        "is_init",
        torch.full(
            (*tensordict.batch_size, 1),
            is_init,
            dtype=torch.bool,
            device=observation.device,
        ),
    )


def make_render_policy(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    hidden_size: int = 128,
    initial_policy_scale: float = 0.2,
) -> ProbabilisticActor:
    """Build the recurrent actor expected by a MicroDuck ``rlrender`` checkpoint.

    ``rlrender`` loads the actor weights after calling this factory. The forward
    hook supplies the initial GRU state on reset and marks subsequent rollout
    steps as non-initial without changing the task or checkpoint state dict.
    """
    actor, _, _ = make_models(
        env,
        device=device,
        hidden_size=hidden_size,
        initial_policy_scale=initial_policy_scale,
    )
    actor.register_forward_pre_hook(
        ft.partial(_prepare_render_recurrent_state, hidden_size=hidden_size)
    )
    return actor


def _prepare_recurrent_reset(
    tensordict: TensorDictBase, policy: ProbabilisticActor
) -> TensorDictBase:
    """Add the zero recurrent state expected on a manually-driven reset."""
    if "is_init" in tensordict:
        return tensordict
    gru = next(
        (module for module in policy.modules() if isinstance(module, GRUModule)), None
    )
    if gru is None:
        return tensordict
    tensordict.set(
        "is_init",
        torch.ones(
            *tensordict.batch_size,
            1,
            dtype=torch.bool,
            device=tensordict.device,
        ),
    )
    tensordict.set(
        gru.in_keys[1],
        torch.zeros(
            *tensordict.batch_size,
            gru.gru.num_layers,
            gru.gru.hidden_size,
            device=tensordict.device,
        ),
    )
    return tensordict


@torch.no_grad()
def evaluate_policy(
    env: MicroDuckVelocityEnv,
    policy: ProbabilisticActor,
    *,
    commanded_x_velocities: Sequence[float] = (-0.3, 0.0, 0.3),
    seeds: Sequence[int] = (0, 1, 2),
    steps: int = 500,
) -> list[dict[str, float]]:
    """Evaluate deterministic fixed-command episodes from controlled seeds.

    Each result reports return, tracking error, survival, episode length, and
    displacement along the duck's initial heading. Keeping results separated
    by signed command prevents aggregate reward from hiding a wrong-way policy.

    Args:
        env: A single-environment task used only for evaluation.
        policy: Policy to evaluate with deterministic actions.
        commanded_x_velocities: Fixed signed commands to evaluate.
        seeds: Reset seeds evaluated for every command.
        steps: Maximum number of steps in each episode.

    Returns:
        One metric dictionary per command and seed.
    """
    if env.num_envs != 1:
        raise ValueError("MicroDuck evaluation expects a single environment.")
    if steps < 1 or not commanded_x_velocities or not seeds:
        raise ValueError("Evaluation commands, seeds, and steps must be non-empty.")

    policy_device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    results = []
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            for command in commanded_x_velocities:
                for seed in seeds:
                    env.set_seed(seed)
                    reset_input = TensorDict(
                        {
                            "commanded_x_velocity": torch.full(
                                (1, 1),
                                float(command),
                                dtype=env.dtype,
                                device=env.device,
                            )
                        },
                        batch_size=env.batch_size,
                        device=env.device,
                    )
                    tensordict = _prepare_recurrent_reset(
                        env.reset(reset_input), policy
                    )
                    start_state = env.get_state()
                    start_qpos = start_state["qpos"][0].to(env.dtype)
                    initial_forward = _body_forward_vector(start_qpos[3:7])
                    episode_return = 0.0
                    tracking_error = 0.0
                    terminated = False
                    episode_length = 0
                    for _ in range(steps):
                        policy_input = tensordict.to(policy_device)
                        policy(policy_input)
                        tensordict["action"] = policy_input["action"].to(env.device)
                        transition = env.step(tensordict)
                        body_x_velocity = transition["next", "observation"][..., 6]
                        episode_return += float(transition["next", "reward"].sum())
                        tracking_error += float(
                            (body_x_velocity - float(command)).abs().sum()
                        )
                        terminated = bool(transition["next", "terminated"].any())
                        done = bool(transition["next", "done"].any())
                        episode_length += 1
                        tensordict = step_mdp(transition, keep_other=True)
                        if done:
                            break
                    end_qpos = env.get_state()["qpos"][0].to(env.dtype)
                    displacement = torch.dot(
                        end_qpos[:2] - start_qpos[:2], initial_forward[:2]
                    )
                    results.append(
                        {
                            "commanded_x_velocity": float(command),
                            "seed": float(seed),
                            "episode_return": episode_return,
                            "tracking_error": tracking_error / episode_length,
                            "survived": float(not terminated),
                            "episode_length": float(episode_length),
                            "signed_displacement": float(displacement),
                        }
                    )
    finally:
        policy.train(was_training)
    return results


def _evaluation_metrics(evaluation: list[dict[str, float]]) -> dict[str, float]:
    metrics = {}
    fields = (
        "episode_return",
        "tracking_error",
        "survived",
        "episode_length",
        "signed_displacement",
    )
    for field in fields:
        metrics[f"evaluation/{field}"] = sum(row[field] for row in evaluation) / len(
            evaluation
        )
    for command in sorted({row["commanded_x_velocity"] for row in evaluation}):
        rows = [row for row in evaluation if row["commanded_x_velocity"] == command]
        command_name = f"{command:+.2f}".replace("+", "plus_").replace("-", "minus_")
        for field in fields:
            metrics[f"evaluation/{command_name}/{field}"] = sum(
                row[field] for row in rows
            ) / len(rows)
    return metrics


def train_ppo(
    env: EnvBase,
    actor: ProbabilisticActor,
    critic: ValueOperator,
    full_value: TensorDictSequential,
    *,
    total_transitions: int = 10_000_000,
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
    epochs: int = 10,
    minibatch_trajectories: int = 8,
    learning_rate: float = 1e-4,
    entropy_coeff: float = 1e-3,
    critic_coeff: float = 1.0,
    target_kl: float | None = None,
    anneal_learning_rate: bool = True,
    max_grad_norm: float = 1.0,
    evaluation_env: MicroDuckVelocityEnv | None = None,
    evaluation_interval: int | None = None,
    evaluation_commands: Sequence[float] = DEFAULT_COMMANDS,
    evaluation_seeds: Sequence[int] = (0, 1, 2),
    evaluation_steps: int = MAX_EPISODE_STEPS,
    best_checkpoint_path: str | Path | None = None,
    experiment_logger: WandbLogger | None = None,
) -> list[dict[str, float]]:
    """Train recurrent PPO from complete on-policy trajectories.

    A collector writes only finished episodes to a roughly 16K-transition
    replay buffer. ``SliceSampler`` draws whole episodes for recurrent PPO,
    the buffer is replayed for up to ``epochs`` passes, and is then erased
    before collecting with the updated policy. When ``target_kl`` is provided,
    an update stops after the first epoch whose mean approximate-KL magnitude
    exceeds that threshold. Collection stops after at least ``total_transitions``
    complete-trajectory transitions.
    """
    if (
        min(
            total_transitions,
            replay_capacity,
            epochs,
            minibatch_trajectories,
        )
        < 1
    ):
        raise ValueError("PPO transition and replay sizes must all be positive.")
    if replay_capacity < MAX_EPISODE_STEPS:
        raise ValueError(
            f"replay_capacity must be at least {MAX_EPISODE_STEPS} transitions."
        )
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive.")
    if max_grad_norm <= 0 or not math.isfinite(max_grad_norm):
        raise ValueError("max_grad_norm must be finite and positive.")
    if min(entropy_coeff, critic_coeff) < 0 or not all(
        math.isfinite(value) for value in (entropy_coeff, critic_coeff)
    ):
        raise ValueError("PPO loss coefficients must be finite and non-negative.")
    if target_kl is not None and (target_kl <= 0 or not math.isfinite(target_kl)):
        raise ValueError("target_kl must be finite and positive when provided.")
    if evaluation_interval is not None and evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive when provided.")
    if evaluation_interval is not None and evaluation_env is None:
        raise ValueError("evaluation_interval requires an evaluation_env.")
    if best_checkpoint_path is not None and evaluation_interval is None:
        raise ValueError("best_checkpoint_path requires periodic evaluation.")

    device = next(actor.parameters()).device
    sampler = SliceSampler(
        num_slices=minibatch_trajectories,
        end_key=("next", "done"),
        strict_length=False,
        cache_values=True,
    )
    replay_buffer = _CompleteTrajectoryReplayBuffer(
        storage=LazyTensorStorage(replay_capacity, ndim=1),
        sampler=sampler,
        collection_capacity=replay_capacity,
    )
    frames_per_batch = max(1, env.batch_size.numel())
    collector = Collector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        replay_buffer=replay_buffer,
        trajs_per_batch=1,
        trajs_per_write=1,
        storing_device="cpu",
        auto_register_policy_transforms=True,
    )
    advantage = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=full_value,
        average_gae=False,
        shifted=True,
        deactivate_vmap=True,
        device=device,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=full_value,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=entropy_coeff,
        critic_coeff=critic_coeff,
        loss_critic_type="smooth_l1",
        normalize_advantage=True,
    )
    parameters = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history = []
    collected_transitions = 0
    collection_iteration = 0
    best_evaluation_return = -float("inf")
    best_actor_state = None
    best_critic_state = None
    checkpoint_path = (
        Path(best_checkpoint_path) if best_checkpoint_path is not None else None
    )

    def run_evaluation() -> tuple[list[dict[str, float]], dict[str, float]]:
        evaluation = evaluate_policy(
            evaluation_env,
            actor,
            commanded_x_velocities=evaluation_commands,
            seeds=evaluation_seeds,
            steps=evaluation_steps,
        )
        for row in evaluation:
            torchrl_logger.info(
                "MicroDuck evaluation transitions=%d command=%+.3f seed=%d: "
                "return=%+.4f tracking_error=%.4f survived=%d "
                "length=%d displacement=%+.4f",
                collected_transitions,
                row["commanded_x_velocity"],
                int(row["seed"]),
                row["episode_return"],
                row["tracking_error"],
                int(row["survived"]),
                int(row["episode_length"]),
                row["signed_displacement"],
            )
        return evaluation, _evaluation_metrics(evaluation)

    if evaluation_interval is not None:
        initial_evaluation, initial_metrics = run_evaluation()
        best_evaluation_return = initial_metrics["evaluation/episode_return"]
        best_actor_state = deepcopy(actor.state_dict())
        best_critic_state = deepcopy(critic.state_dict())
        if experiment_logger is not None:
            experiment_logger.log_metrics(
                initial_metrics, step=0, override_global_step=True
            )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "transitions": 0,
                    "evaluation_return": best_evaluation_return,
                    "evaluation": initial_evaluation,
                    "actor": best_actor_state,
                    "critic": best_critic_state,
                },
                checkpoint_path,
            )

    collector_iterator = iter(collector)
    try:
        while collected_transitions < total_transitions:
            collection_iteration += 1
            collection_start = time.perf_counter()
            while not replay_buffer.collection_full:
                next(collector_iterator)
            collection_time = time.perf_counter() - collection_start
            collected = replay_buffer[:]
            collection_size = collected.numel()
            if not collection_size:
                raise RuntimeError(
                    "The complete-trajectory collector did not fit an episode in "
                    "the replay buffer."
                )
            previous_transitions = collected_transitions
            collected_transitions += collection_size
            metrics, num_trajectories = _collection_metrics(collected)
            metrics.update(
                {
                    "collection/iteration": float(collection_iteration),
                    "collection/transitions": float(collection_size),
                    "collection/trajectories": float(num_trajectories),
                    "collection/replay_fill_fraction": collection_size
                    / replay_capacity,
                    "throughput/inference_transitions_per_second": collection_size
                    / collection_time,
                    "throughput/collection_seconds": collection_time,
                }
            )
            with torch.no_grad(), set_recurrent_mode(True):
                processed = collected.to(device).clone().refine_names("time")
                advantage(processed)
            replay_buffer.replace_with_processed_batch(processed.cpu())

            current_learning_rate = learning_rate
            if anneal_learning_rate:
                current_learning_rate *= max(
                    0.0, 1.0 - previous_transitions / total_transitions
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_learning_rate
            updates_per_epoch = max(
                1, math.ceil(num_trajectories / minibatch_trajectories)
            )
            training_start = time.perf_counter()
            training_transitions = 0
            update_count = 0
            update_metrics: dict[str, float] = {}
            epochs_completed = 0
            stopped_on_kl = False
            for epoch in range(epochs):
                epoch_kl = 0.0
                for _ in range(updates_per_epoch):
                    sample = replay_buffer.sample(
                        minibatch_trajectories * MAX_EPISODE_STEPS
                    ).to(device)
                    sample.refine_names("time")
                    optimizer.zero_grad(set_to_none=True)
                    with set_recurrent_mode(True):
                        loss_values = loss_module(sample)
                    total_loss = (
                        loss_values["loss_objective"]
                        + loss_values["loss_critic"]
                        + loss_values["loss_entropy"]
                    )
                    total_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                    optimizer.step()
                    training_transitions += sample.numel()
                    update_count += 1
                    epoch_kl += _metric_float(loss_values["kl_approx"])
                    for key, value in loss_values.items():
                        metric_name = f"ppo/{key}"
                        update_metrics[metric_name] = update_metrics.get(
                            metric_name, 0.0
                        ) + _metric_float(value)
                    update_metrics["ppo/loss_total"] = update_metrics.get(
                        "ppo/loss_total", 0.0
                    ) + _metric_float(total_loss)
                    update_metrics["ppo/grad_norm"] = update_metrics.get(
                        "ppo/grad_norm", 0.0
                    ) + _metric_float(grad_norm)
                epochs_completed = epoch + 1
                if (
                    target_kl is not None
                    and abs(epoch_kl / updates_per_epoch) > target_kl
                ):
                    stopped_on_kl = True
                    break
            training_time = time.perf_counter() - training_start
            metrics.update(
                {key: value / update_count for key, value in update_metrics.items()}
            )
            metrics.update(
                {
                    "training/epochs": float(epochs_completed),
                    "training/epochs_configured": float(epochs),
                    "training/stopped_on_kl": float(stopped_on_kl),
                    "training/updates": float(update_count),
                    "training/learning_rate": current_learning_rate,
                    "throughput/training_transitions_per_second": training_transitions
                    / training_time,
                    "throughput/training_seconds": training_time,
                    "progress/transitions": float(collected_transitions),
                    "progress/target_transitions": float(total_transitions),
                }
            )
            metrics.update(_telemetry_metrics(device))

            should_evaluate = evaluation_interval is not None and (
                collection_iteration % evaluation_interval == 0
                or collected_transitions >= total_transitions
            )
            if should_evaluate:
                evaluation, evaluation_metrics = run_evaluation()
                metrics.update(evaluation_metrics)
                mean_return = evaluation_metrics["evaluation/episode_return"]
                if mean_return > best_evaluation_return:
                    best_evaluation_return = mean_return
                    best_actor_state = deepcopy(actor.state_dict())
                    best_critic_state = deepcopy(critic.state_dict())
                    if checkpoint_path is not None:
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "transitions": collected_transitions,
                                "evaluation_return": mean_return,
                                "evaluation": evaluation,
                                "actor": best_actor_state,
                                "critic": best_critic_state,
                            },
                            checkpoint_path,
                        )
            history.append(metrics)
            if experiment_logger is not None:
                experiment_logger.log_metrics(
                    metrics,
                    step=collected_transitions,
                    override_global_step=True,
                )
            torchrl_logger.info(
                "MicroDuck PPO transitions=%d/%d trajectories=%d "
                "reward=%+.4f return=%+.3f inference=%.0f/s training=%.0f/s",
                collected_transitions,
                total_transitions,
                num_trajectories,
                metrics["collection/reward_mean"],
                metrics["episode/return_mean"],
                metrics["throughput/inference_transitions_per_second"],
                metrics["throughput/training_transitions_per_second"],
            )
            replay_buffer.empty()
            collector.update_policy_weights_()
            collector.reset()
    finally:
        replay_buffer.empty()
        collector.shutdown()
    if best_actor_state is not None and best_critic_state is not None:
        actor.load_state_dict(best_actor_state)
        critic.load_state_dict(best_critic_state)
    return history


@torch.no_grad()
def collect_qpos_trajectory(
    env: MicroDuckVelocityEnv,
    policy: ProbabilisticActor,
    *,
    steps: int = 300,
    commanded_x_velocity: float = 0.0,
) -> TensorDict:
    """Collect a single-env qpos trajectory for native or WASM rendering."""
    if env.num_envs != 1:
        raise ValueError("qpos rendering expects a single MicroDuckVelocityEnv.")
    reset_input = TensorDict(
        {
            "commanded_x_velocity": torch.full(
                (1, 1),
                float(commanded_x_velocity),
                dtype=env.dtype,
                device=env.device,
            )
        },
        batch_size=env.batch_size,
        device=env.device,
    )
    tensordict = _prepare_recurrent_reset(env.reset(reset_input), policy)
    qpos = [env.get_state()["qpos"].squeeze(0).cpu()]
    for _ in range(steps):
        policy(tensordict)
        transition = env.step(tensordict)
        qpos.append(env.get_state()["qpos"].squeeze(0).cpu())
        tensordict = step_mdp(transition, keep_other=True)
        if bool(transition["next", "done"].any()):
            break
    trajectory = torch.stack(qpos)
    return TensorDict({"qpos": trajectory}, batch_size=(trajectory.shape[0],))


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck-root", type=Path)
    parser.add_argument(
        "--backend",
        choices=("mujoco", "mjx", "mujoco-torch"),
        default="mujoco",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-transitions", type=int, default=10_000_000)
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch-trajectories", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coeff", type=float, default=1e-3)
    parser.add_argument("--critic-coeff", type=float, default=1.0)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.01,
        help=(
            "Stop each PPO update after an epoch whose mean KL magnitude exceeds "
            "this value."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--anneal-learning-rate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--initial-policy-scale", type=float, default=0.2)
    parser.add_argument(
        "--commanded-x-velocity",
        action="append",
        type=float,
        dest="commanded_x_velocities",
        help=(
            "Velocity command to sample at reset. Repeat for multiple commands; "
            "defaults to -0.3, 0.0, and 0.3 m/s."
        ),
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=25,
        help="Run deterministic fixed-command evaluation every N collections.",
    )
    parser.add_argument("--evaluation-steps", type=int, default=500)
    parser.add_argument(
        "--best-checkpoint-path",
        type=Path,
        default=Path("microduck_ppo_best.pt"),
    )
    parser.add_argument("--wandb-project", default="torchrl-microduck-ppo")
    parser.add_argument(
        "--wandb-entity",
        help=(
            "Explicit W&B entity. Required whenever W&B logging is enabled to "
            "avoid falling back to an unintended default workspace."
        ),
    )
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    if args.smoke:
        args.total_transitions = 500
        args.replay_capacity = 600
        args.epochs = 1
        args.minibatch_trajectories = 1
        args.num_envs = 1
        args.evaluation_interval = None
        args.best_checkpoint_path = None
        args.wandb_mode = "disabled"
    if args.wandb_mode != "disabled" and not args.wandb_entity:
        raise ValueError(
            "W&B logging requires an explicit --wandb-entity to avoid writing "
            "to an unintended default workspace."
        )
    torch.manual_seed(args.seed)
    commands = args.commanded_x_velocities or DEFAULT_COMMANDS
    env = make_env(
        args.microduck_root,
        backend=args.backend,
        commanded_x_velocity=commands,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
    )
    evaluation_env = None
    experiment_logger = None
    try:
        actor, critic, full_value = make_models(
            env,
            device=args.device,
            hidden_size=args.hidden_size,
            initial_policy_scale=args.initial_policy_scale,
        )
        if args.evaluation_interval is not None:
            evaluation_env = MicroDuckVelocityEnv(
                args.microduck_root,
                backend=args.backend,
                commanded_x_velocity=commands,
                num_envs=1,
                device=torch.device(args.device),
                seed=args.seed,
                max_episode_steps=MAX_EPISODE_STEPS,
            )
        if args.wandb_mode != "disabled":
            experiment_logger = WandbLogger(
                exp_name=args.wandb_name
                or f"microduck-{args.backend}-seed-{args.seed}",
                project=args.wandb_project,
                entity=args.wandb_entity,
                offline=args.wandb_mode == "offline",
            )
            experiment_logger.log_hparams(
                {
                    "backend": args.backend,
                    "device": args.device,
                    "num_envs": args.num_envs,
                    "commands": list(commands),
                    "total_transitions": args.total_transitions,
                    "replay_capacity": args.replay_capacity,
                    "epochs": args.epochs,
                    "minibatch_trajectories": args.minibatch_trajectories,
                    "hidden_size": args.hidden_size,
                    "learning_rate": args.learning_rate,
                    "anneal_learning_rate": args.anneal_learning_rate,
                    "entropy_coeff": args.entropy_coeff,
                    "critic_coeff": args.critic_coeff,
                    "target_kl": args.target_kl,
                    "seed": args.seed,
                }
            )
        train_ppo(
            env,
            actor,
            critic,
            full_value,
            total_transitions=args.total_transitions,
            replay_capacity=args.replay_capacity,
            epochs=args.epochs,
            minibatch_trajectories=args.minibatch_trajectories,
            learning_rate=args.learning_rate,
            entropy_coeff=args.entropy_coeff,
            critic_coeff=args.critic_coeff,
            target_kl=args.target_kl,
            anneal_learning_rate=args.anneal_learning_rate,
            max_grad_norm=args.max_grad_norm,
            evaluation_env=evaluation_env,
            evaluation_interval=args.evaluation_interval,
            evaluation_commands=commands,
            evaluation_steps=args.evaluation_steps,
            best_checkpoint_path=args.best_checkpoint_path,
            experiment_logger=experiment_logger,
        )
    finally:
        if experiment_logger is not None:
            experiment_logger.experiment.finish()
        if evaluation_env is not None:
            evaluation_env.close()
        if not env.is_closed:
            env.close()


if __name__ == "__main__":
    main(parse_args())
