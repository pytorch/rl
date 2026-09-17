# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PPO on a MuJoCo Menagerie robot, or any MuJoCo model on GitHub: hold the home pose, or walk.

A short recipe around :class:`~torchrl.envs.MenagerieEnv`: a Gaussian MLP
actor and an MLP critic, a :class:`~torchrl.collectors.Collector` feeding
:class:`~torchrl.objectives.ClipPPOLoss` with GAE, and a checkpoint written
with :func:`~torchrl.render.save_render_checkpoint`. :func:`make_env` and
:func:`make_policy` are the factories ``rlrender`` imports to play it back.

Two tasks. ``hold_pose`` is the
:meth:`~torchrl.envs.MenagerieEnv.hold_pose_task` preset of the env and works
for any robot. ``walk`` is a velocity-tracking task for position-controlled
quadrupeds built from two transforms on the raw env state:
:class:`HomeOffsetActions` maps a normalized action to joint targets around the
home keyframe, and :class:`QuadrupedJoystick` assembles the observation, the
reward (the terms and weights of MuJoCo Playground's Go1 joystick task, see
https://github.com/google-deepmind/mujoco_playground) and the fall termination.

Train the Unitree Go2 to walk forward at 0.5 m/s from the ``mujoco-menagerie``
package cache (``--download`` fetches the model once; the MJX scene drives the
joints with position servos)::

    python examples/menagerie/ppo.py --task walk --robot unitree_go2 --entry scene_mjx \\
        --download --num-envs 6 --frames 20000000 --command-range 1.0 0.5 0.8

The policy tracks ``--command`` (0.5 m/s forward by default) at evaluation;
``--command-range`` draws a fresh command per episode during training, which
keeps the gait honest instead of specializing to one command.

A UR5e holding its pose from a local checkout, as a quick check::

    TORCHRL_MUJOCO_MENAGERIE_PATH=~/mujoco_menagerie \\
        python examples/menagerie/ppo.py --task hold_pose --robot universal_robots_ur5e --smoke

Any model in a GitHub repository instead of a Menagerie robot: ``--repo`` and
``--revision`` pin it through :class:`~torchrl.envs.GitHubModelSource`, and
``--entry`` is then the repository-relative XML::

    python examples/menagerie/ppo.py --task hold_pose --download \\
        --repo SouthColumn76/universal_robots_ur3e \\
        --revision 5f042ffca6b5885fd18f5448e17b71ab46274fa3 --entry ur3e.xml

Render the checkpoint as a video from the scene's first camera (``--fps 50``
is real time for the 20 ms control step), or as a notebook with a saved
rollout and a cell that collects a fresh one in the kernel. The checkpoint
records the training arguments, so the factories rebuild the same robot and
task; ``--env-kwargs`` overrides them and sets the render camera and size::

    rlrender --ckpt menagerie_ppo.ckpt \\
        --policy examples/menagerie/ppo.py:make_policy \\
        --env examples/menagerie/ppo.py:make_env \\
        --env-kwargs '{"camera_id": 0, "render_width": 640, "render_height": 480, "fixed_command": true}' \\
        --deterministic --from-pixels --render-backend pixels \\
        --max-steps 500 --fps 50 --format mp4 --out menagerie_ppo.mp4 --overwrite
    rlrender --ckpt menagerie_ppo.ckpt \\
        --policy examples/menagerie/ppo.py:make_policy \\
        --env examples/menagerie/ppo.py:make_env \\
        --deterministic --from-pixels --render-backend pixels \\
        --max-steps 500 --fps 50 --format ipynb --out menagerie_ppo.ipynb \\
        --notebook-rollout-mode both --overwrite
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import (
    Binary,
    Bounded,
    Composite,
    LazyTensorStorage,
    ReplayBuffer,
    SamplerWithoutReplacement,
    Unbounded,
)
from torchrl.envs import (
    CatTensors,
    Compose,
    EnvBase,
    GitHubModelSource,
    MenagerieEnv,
    MenagerieTask,
    MujocoModelEnv,
    ParallelEnv,
    Transform,
    TransformedEnv,
)
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.render import save_render_checkpoint

TASK_ARGS = (
    "task",
    "robot",
    "repo",
    "revision",
    "entry",
    "frame_skip",
    "max_episode_steps",
    "download",
    "fall_height",
    "alive_bonus",
    "control_cost",
    "command",
    "command_range",
    "action_scale",
    "feet_sites",
    "feet_geoms",
)

RENDER_RUNTIME_KWARGS = frozenset(
    {
        "spec",
        "config",
        "env_kwargs",
        "max_steps",
        "pixels_only",
        "camera",
        "render_mode",
    }
)

WALK_REWARD_WEIGHTS = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -0.5,
    "ang_vel_xy": -0.05,
    "orientation": -5.0,
    "pose": 0.5,
    "termination": -1.0,
    "torques": -0.0002,
    "energy": -0.001,
    "dof_pos_limits": -1.0,
    "action_rate": -0.01,
    "feet_slip": -0.1,
    "feet_clearance": -2.0,
    "feet_height": -0.2,
    "feet_air_time": 0.1,
    "feet_stuck": -2.0,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Training arguments; the task entries are recorded in the checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=("hold_pose", "walk"), default="hold_pose")
    parser.add_argument(
        "--robot", default="unitree_go2", help="Menagerie model directory."
    )
    parser.add_argument(
        "--entry",
        default=None,
        help="Menagerie entry point by stem (e.g. scene_mjx), or the repository-relative XML of --repo.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub owner/name to load instead of a Menagerie robot.",
    )
    parser.add_argument(
        "--revision", default=None, help="Commit, tag or branch of --repo."
    )
    parser.add_argument(
        "--frame-skip", type=int, default=10, help="Physics steps per action."
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=1000, help="Episode horizon."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Let the mujoco-menagerie package fetch the robot into its cache.",
    )
    hold = parser.add_argument_group("hold_pose")
    hold.add_argument(
        "--fall-height",
        type=float,
        default=None,
        help="Terminate when the floating base drops below this height (m).",
    )
    hold.add_argument("--alive-bonus", type=float, default=0.5)
    hold.add_argument("--control-cost", type=float, default=0.01)
    walk = parser.add_argument_group("walk")
    walk.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=(0.5, 0.0, 0.0),
        metavar=("VX", "VY", "WZ"),
        help="Body-frame velocity command: forward and lateral (m/s), yaw rate (rad/s).",
    )
    walk.add_argument(
        "--command-range",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "WZ"),
        help="If set, sample a command uniformly in +/- this range at every reset.",
    )
    walk.add_argument(
        "--action-scale",
        type=float,
        default=0.5,
        help="Joint target offset (rad) for a unit normalized action.",
    )
    walk.add_argument(
        "--feet-sites",
        nargs="+",
        default=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        help="Foot sites, for foot heights and velocities.",
    )
    walk.add_argument(
        "--feet-geoms",
        nargs="+",
        default=("FL", "FR", "RL", "RR"),
        help="Foot collision geoms, for contacts.",
    )
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--frames", type=int, default=20_000_000)
    parser.add_argument("--steps-per-batch", type=int, default=2048, help="Per env.")
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, nargs="+", default=(512, 256, 128))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ckpt", default="menagerie_ppo.ckpt")
    parser.add_argument(
        "--smoke", action="store_true", help="One tiny update on one env."
    )
    args = parser.parse_args(argv)
    if (args.repo is None) != (args.revision is None) or (
        args.repo is not None and args.entry is None
    ):
        parser.error(
            "--repo needs --revision and --entry (the repository-relative XML)."
        )
    args.command = tuple(float(v) for v in args.command)
    if args.command_range is not None:
        args.command_range = tuple(float(v) for v in args.command_range)
    args.feet_sites = tuple(args.feet_sites)
    args.feet_geoms = tuple(args.feet_geoms)
    args.hidden = tuple(int(v) for v in args.hidden)
    if args.smoke:
        args.num_envs = 1
        args.frames = 128
        args.steps_per_batch = 64
        args.minibatch = 32
        args.epochs = 1
    return args


# ----------------------------------------------------------------------
# The walk task, as transforms over the raw MenagerieEnv state
# ----------------------------------------------------------------------


def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame ``vec`` into the body frame of a ``wxyz`` quaternion."""
    w, xyz = quat[..., :1], quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec - w * t + torch.cross(xyz, t, dim=-1)


class HomeOffsetActions(Transform):
    """Normalized actions in ``[-1, 1]`` become joint targets around the home pose.

    The env receives ``home + action_scale * action`` while the normalized
    action stays in the collected data. ``_step`` sees the mapped targets, so
    it inverts the map to expose the previous normalized action as the
    ``prev_action`` observation for the policy and the action-rate cost.
    """

    def __init__(self, home: torch.Tensor, action_scale: float):
        super().__init__(in_keys_inv=["action"], out_keys_inv=["action"])
        self.register_buffer("home", home.clone())
        self.action_scale = float(action_scale)

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        return self.home + self.action_scale * action

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        targets = tensordict.get("action")
        next_tensordict.set("prev_action", (targets - self.home) / self.action_scale)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        shape = (*tensordict_reset.batch_size, self.home.numel())
        tensordict_reset.set("prev_action", self.home.new_zeros(shape))
        return tensordict_reset

    def transform_action_spec(self, action_spec: Composite) -> Composite:
        action_spec = action_spec.clone()
        old = action_spec["action"]
        action_spec["action"] = Bounded(
            low=-1.0, high=1.0, shape=old.shape, dtype=old.dtype, device=old.device
        )
        return action_spec

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec["prev_action"] = Unbounded(
            shape=(*observation_spec.shape, self.home.numel()),
            dtype=self.home.dtype,
            device=observation_spec.device,
        )
        return observation_spec


class QuadrupedJoystick(Transform):
    """Track a body-frame velocity command with a position-controlled quadruped.

    Observation (``observation``): base angular velocity, gravity in the body
    frame, joint angles minus the home pose, joint velocities, the previous
    action and the command. Reward: the terms and weights of MuJoCo
    Playground's Go1 joystick task (velocity tracking, base motion and
    orientation costs, a pose term, servo torque and energy costs, joint soft
    limits, action rate, foot slip, clearance and swing height, air time at
    touchdown, a termination cost), plus a ``feet_stuck`` cost on any foot
    kept in the air longer than ``max_air_time``, which closes the
    three-legged gait the reference terms leave open; summed, clipped at zero
    and multiplied by the control period. Termination: the base turns over. Foot contacts come
    from the env's ``geom_contacts``; the foot velocities are finite
    differences of the ``site_positions`` observation; the servo torques
    follow the actuators' affine gain and bias. The command, the air time, the
    last contact and the swing peak of every foot travel in the tensordict, so
    the transform holds no state of its own.

    Args:
        command (Sequence[float]): ``(vx, vy, wz)`` in body-frame m/s and rad/s,
            the command of every episode unless ``command_range`` is set.
        home_joints (Tensor): joint angles of the home pose, in ``qpos[7:]`` order.
        feet_geoms (Sequence[str]): foot collision geoms, in the order of the
            ``site_positions`` observation.
        dt (float): control period in seconds.
        action_scale (float): joint target offset for a unit action, to
            recover the servo torques.
        servo (Tensor): per-actuator ``(gain, bias_q, bias_qd, force_low,
            force_high)`` of the affine position servos, shaped ``(nu, 5)``.
        joint_limits (Tensor): per-joint ``(low, high)`` soft limits, shaped
            ``(nu, 2)``.
        command_range (Sequence[float], optional): if set, every reset draws
            the command uniformly in ``[-range, range]`` per component
            instead of using ``command``. Defaults to ``None``.
        weights (Mapping[str, float], optional): reward weights per term;
            defaults to :data:`WALK_REWARD_WEIGHTS`.
        tracking_sigma (float, optional): scale of the tracking terms. Defaults to ``0.25``.
        max_foot_height (float, optional): target swing height in meters. Defaults to ``0.1``.
        max_air_time (float, optional): air time in seconds beyond which a foot
            is charged the ``feet_stuck`` cost, growing to its full weight one
            second later. Defaults to ``0.5``.
    """

    def __init__(
        self,
        command: Sequence[float],
        home_joints: torch.Tensor,
        feet_geoms: Sequence[str],
        dt: float,
        action_scale: float,
        servo: torch.Tensor,
        joint_limits: torch.Tensor,
        *,
        command_range: Sequence[float] | None = None,
        weights: Mapping[str, float] | None = None,
        tracking_sigma: float = 0.25,
        max_foot_height: float = 0.1,
        max_air_time: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("command", torch.as_tensor(command, dtype=torch.float32))
        self.register_buffer(
            "command_range",
            None
            if command_range is None
            else torch.as_tensor(command_range, dtype=torch.float32),
        )
        self.register_buffer("home_joints", home_joints.clone())
        self.register_buffer("servo", servo.clone())
        self.register_buffer("joint_limits", joint_limits.clone())
        self.action_scale = float(action_scale)
        self.register_buffer(
            "pose_weight", torch.tensor([1.0, 1.0, 0.1] * (home_joints.numel() // 3))
        )
        self.feet_geoms = tuple(feet_geoms)
        self.dt = float(dt)
        self.weights = dict(WALK_REWARD_WEIGHTS if weights is None else weights)
        self.tracking_sigma = float(tracking_sigma)
        self.max_foot_height = float(max_foot_height)
        self.max_air_time = float(max_air_time)

    @property
    def observation_dim(self) -> int:
        return 3 + 3 + 2 * self.home_joints.numel() + self.home_joints.numel() + 3

    def _observation(self, next_tensordict: TensorDictBase) -> torch.Tensor:
        qpos = next_tensordict.get("qpos")
        qvel = next_tensordict.get("qvel")
        command = next_tensordict.get("command")
        gravity = quat_rotate_inverse(
            qpos[..., 3:7], qpos.new_tensor([0.0, 0.0, -1.0]).expand_as(qpos[..., :3])
        )
        return torch.cat(
            [
                qvel[..., 3:6] * 0.25,
                gravity,
                qpos[..., 7:] - self.home_joints,
                qvel[..., 6:] * 0.05,
                next_tensordict.get("prev_action"),
                command,
            ],
            dim=-1,
        )

    def _sample_command(self, batch_size: torch.Size) -> torch.Tensor:
        command = self.command.expand(*batch_size, 3).clone()
        if self.command_range is None:
            return command
        return (2.0 * torch.rand_like(command) - 1.0) * self.command_range

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        feet = tensordict_reset.get("site_positions")
        zeros = feet.new_zeros(feet.shape[:-1])
        tensordict_reset.set(
            "command", self._sample_command(tensordict_reset.batch_size)
        )
        tensordict_reset.set("observation", self._observation(tensordict_reset))
        tensordict_reset.set("feet_air_time", zeros)
        tensordict_reset.set("swing_peak", zeros.clone())
        tensordict_reset.set("last_contact", zeros.bool())
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        qpos = next_tensordict.get("qpos")
        qvel = next_tensordict.get("qvel")
        quat = qpos[..., 3:7]
        up = quat_rotate_inverse(
            quat, qpos.new_tensor([0.0, 0.0, 1.0]).expand_as(qpos[..., :3])
        )
        local_linvel = quat_rotate_inverse(quat, qvel[..., :3])
        gyro = qvel[..., 3:6]
        world_angvel = self._rotate(quat, gyro)
        joints = qpos[..., 7:]
        joint_vel = qvel[..., 6:]
        action = next_tensordict.get("prev_action")
        last_action = tensordict.get("prev_action")
        command = tensordict.get("command")
        cmd_norm = command.norm(dim=-1)
        targets = self.home_joints + self.action_scale * action
        gain, bias_q, bias_qd = self.servo[:, 0], self.servo[:, 1], self.servo[:, 2]
        torques = (gain * targets + bias_q * joints + bias_qd * joint_vel).clamp(
            self.servo[:, 3], self.servo[:, 4]
        )
        below = (self.joint_limits[:, 0] - joints).clamp_min(0.0)
        above = (joints - self.joint_limits[:, 1]).clamp_min(0.0)

        feet = next_tensordict.get("site_positions")
        feet_vel = (feet - tensordict.get("site_positions")) / self.dt
        contact = self.parent.base_env.geom_contacts(self.feet_geoms).to(feet.device)
        last_contact = tensordict.get("last_contact")
        air_time = tensordict.get("feet_air_time")
        first_contact = (air_time > 0.0) & (contact | last_contact)
        air_time = air_time + self.dt
        swing_peak = torch.maximum(tensordict.get("swing_peak"), feet[..., 2])
        fallen = up[..., 2] < 0.0
        foot_speed_xy = feet_vel[..., :2].square().sum(-1)

        terms = {
            "tracking_lin_vel": torch.exp(
                -(command[..., :2] - local_linvel[..., :2]).square().sum(-1)
                / self.tracking_sigma
            ),
            "tracking_ang_vel": torch.exp(
                -(command[..., 2] - gyro[..., 2]).square() / self.tracking_sigma
            ),
            "lin_vel_z": qvel[..., 2].square(),
            "ang_vel_xy": world_angvel[..., :2].square().sum(-1),
            "orientation": up[..., :2].square().sum(-1),
            "pose": torch.exp(
                -((joints - self.home_joints).square() * self.pose_weight).sum(-1)
            ),
            "termination": fallen.to(qpos.dtype),
            "torques": torques.square().sum(-1).sqrt() + torques.abs().sum(-1),
            "energy": (joint_vel.abs() * torques.abs()).sum(-1),
            "dof_pos_limits": (below + above).sum(-1),
            "action_rate": (action - last_action).square().sum(-1),
            "feet_slip": (foot_speed_xy * contact).sum(-1) * (cmd_norm > 0.01),
            "feet_clearance": (
                (feet[..., 2] - self.max_foot_height).abs() * foot_speed_xy.pow(0.25)
            ).sum(-1),
            "feet_height": (
                (swing_peak / self.max_foot_height - 1.0).square() * first_contact
            ).sum(-1)
            * (cmd_norm > 0.01),
            "feet_air_time": ((air_time - 0.1) * first_contact).sum(-1)
            * (cmd_norm > 0.01),
            "feet_stuck": (air_time - self.max_air_time).clamp(0.0, 1.0).sum(-1),
        }
        reward = sum(self.weights[name] * value for name, value in terms.items())
        next_tensordict.set("reward", (reward.clamp_min(0.0) * self.dt).unsqueeze(-1))
        next_tensordict.set(
            "terminated", next_tensordict.get("terminated") | fallen.unsqueeze(-1)
        )
        next_tensordict.set("done", next_tensordict.get("done") | fallen.unsqueeze(-1))
        next_tensordict.set("feet_air_time", air_time * ~contact)
        next_tensordict.set("swing_peak", swing_peak * ~contact)
        next_tensordict.set("last_contact", contact)
        next_tensordict.set("command", command)
        next_tensordict.set("observation", self._observation(next_tensordict))
        return next_tensordict

    @staticmethod
    def _rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        w, xyz = quat[..., :1], quat[..., 1:]
        t = 2.0 * torch.cross(xyz, vec, dim=-1)
        return vec + w * t + torch.cross(xyz, t, dim=-1)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        batch = observation_spec.shape
        feet = observation_spec["site_positions"].shape[-2]
        device = observation_spec.device
        observation_spec["observation"] = Unbounded(
            shape=(*batch, self.observation_dim), dtype=torch.float32, device=device
        )
        observation_spec["feet_air_time"] = Unbounded(
            shape=(*batch, feet), dtype=torch.float32, device=device
        )
        observation_spec["swing_peak"] = Unbounded(
            shape=(*batch, feet), dtype=torch.float32, device=device
        )
        observation_spec["last_contact"] = Binary(
            n=feet, shape=(*batch, feet), dtype=torch.bool, device=device
        )
        observation_spec["command"] = Unbounded(
            shape=(*batch, 3), dtype=torch.float32, device=device
        )
        return observation_spec


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------


def make_single_env(
    settings: Mapping[str, Any],
    *,
    fixed_command: bool = False,
    seed: int | None = None,
    device: torch.device | str | None = None,
    from_pixels: bool = False,
    camera_id: int = -1,
    render_width: int = 320,
    render_height: int = 240,
) -> TransformedEnv:
    """One env with its task transforms; ``settings`` holds the :data:`TASK_ARGS`.

    ``fixed_command`` keeps the walk task on ``settings["command"]`` even when
    a command range was recorded, which is what evaluation and rendering want.
    """
    walk = settings["task"] == "walk"
    if walk:
        task = MenagerieTask(site_names=settings["feet_sites"])
    else:
        task = MenagerieEnv.hold_pose_task(
            control_cost_weight=settings["control_cost"],
            alive_bonus=settings["alive_bonus"],
            terminate_below_height=settings["fall_height"],
        )
    env_kwargs = {
        "download": settings["download"],
        "task": task,
        "backend": "mujoco",
        "frame_skip": settings["frame_skip"],
        "seed": seed,
        "device": device,
        "max_episode_steps": settings["max_episode_steps"],
        "from_pixels": from_pixels,
        "camera_id": camera_id,
        "render_width": render_width,
        "render_height": render_height,
    }
    if settings["repo"] is not None:
        source = GitHubModelSource(
            settings["repo"], settings["revision"], settings["entry"]
        )
        env = MujocoModelEnv(source, **env_kwargs)
    else:
        env = MenagerieEnv(settings["robot"], entry=settings["entry"], **env_kwargs)
    if not walk:
        return TransformedEnv(
            env,
            CatTensors(in_keys=["qpos", "qvel"], out_key="observation", del_keys=False),
        )
    home = env.reset_state["qpos"][7:]
    if home.numel() != env.action_spec.shape[-1]:
        raise ValueError(
            "The walk task expects one position actuator per joint after the free "
            f"joint, got {env.action_spec.shape[-1]} actuators for {home.numel()} joints."
        )
    model = env.mj_model
    if (model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE).any():
        raise ValueError(
            "The walk task drives position servos, but this entry has torque or "
            "unbiased actuators; pick the position-controlled scene, e.g. "
            "entry='scene_mjx' for the Unitree Go2."
        )
    servo = torch.as_tensor(
        np.concatenate(
            [
                model.actuator_gainprm[:, :1],
                model.actuator_biasprm[:, 1:3],
                model.actuator_forcerange,
            ],
            axis=1,
        ),
        dtype=torch.float32,
    )
    joint_range = torch.as_tensor(model.jnt_range[1:], dtype=torch.float32)
    mid = joint_range.mean(dim=1, keepdim=True)
    half = 0.95 * (joint_range[:, 1:] - joint_range[:, :1]) / 2
    joint_limits = torch.cat([mid - half, mid + half], dim=1)
    return TransformedEnv(
        env,
        Compose(
            HomeOffsetActions(home, settings["action_scale"]),
            QuadrupedJoystick(
                settings["command"],
                home,
                settings["feet_geoms"],
                env.dt,
                settings["action_scale"],
                servo,
                joint_limits,
                command_range=None if fixed_command else settings["command_range"],
            ),
        ),
    )


def make_env(
    robot: str | None = None,
    *,
    num_envs: int = 1,
    fixed_command: bool = False,
    seed: int | None = None,
    device: torch.device | str | None = None,
    from_pixels: bool = False,
    camera_id: int = -1,
    render_width: int = 320,
    render_height: int = 240,
    checkpoint: Mapping[str, Any] | None = None,
    **task_overrides: Any,
) -> EnvBase:
    """Build the env, batched over worker processes when ``num_envs > 1``.

    ``rlrender`` calls this with the checkpoint. The task arguments
    (:data:`TASK_ARGS`, the ``--`` options of :func:`parse_args`) come from the
    checkpoint's recorded config, and explicit keyword arguments override
    them, e.g. ``--env-kwargs '{"command": [0.8, 0, 0]}'``; the keys
    ``rlrender`` forwards for its own use (:data:`RENDER_RUNTIME_KWARGS`) are
    ignored. An explicit ``robot`` clears a recorded ``repo``, so a checkpoint
    trained on a GitHub model can be re-targeted to a Menagerie robot. ``from_pixels``
    adds frames from ``camera_id`` (MuJoCo's free camera by default, ``0``
    for the first camera of the scene); ``fixed_command`` pins the walk
    command for evaluation.
    """
    unknown = set(task_overrides) - set(TASK_ARGS) - RENDER_RUNTIME_KWARGS
    if unknown:
        raise TypeError(
            f"Unknown task arguments {sorted(unknown)}; expected {TASK_ARGS}."
        )
    recorded = dict((checkpoint or {}).get("config") or {})
    defaults = vars(parse_args([]))
    settings = {key: recorded.get(key, defaults[key]) for key in TASK_ARGS}
    if robot is not None:
        settings["robot"] = robot
        settings["repo"] = settings["revision"] = None
    settings.update(
        {
            key: value
            for key, value in task_overrides.items()
            if key in TASK_ARGS and value is not None
        }
    )
    if (settings["repo"] is None) != (settings["revision"] is None) or (
        settings["repo"] is not None and settings["entry"] is None
    ):
        raise ValueError("repo needs revision and entry (the repository-relative XML).")
    render = {
        "fixed_command": fixed_command,
        "from_pixels": from_pixels,
        "camera_id": camera_id,
        "render_width": render_width,
        "render_height": render_height,
    }
    if num_envs == 1:
        return make_single_env(settings, seed=seed, device=device, **render)
    seeds = [None if seed is None else seed + index for index in range(num_envs)]
    return ParallelEnv(
        num_envs,
        partial(make_single_env, settings, device=device, **render),
        create_env_kwargs=[{"seed": worker_seed} for worker_seed in seeds],
    )


def make_policy(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    hidden: Sequence[int] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> ProbabilisticActor:
    """Gaussian MLP actor over ``observation``; ``rlrender`` loads the checkpoint into it."""
    if hidden is None:
        hidden = ((checkpoint or {}).get("config") or {}).get("hidden", (512, 256, 128))
    action_spec = env.action_spec_unbatched
    net = nn.Sequential(
        MLP(
            in_features=env.observation_spec["observation"].shape[-1],
            out_features=2 * action_spec.shape[-1],
            num_cells=list(hidden),
            activation_class=nn.SiLU,
        ),
        NormalParamExtractor(),
    )
    return ProbabilisticActor(
        TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"]),
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=True,
    ).to(device)


def make_critic(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    hidden: Sequence[int] = (512, 256, 128),
) -> ValueOperator:
    """MLP state-value critic over ``observation``."""
    return ValueOperator(
        MLP(
            in_features=env.observation_spec["observation"].shape[-1],
            out_features=1,
            num_cells=list(hidden),
            activation_class=nn.SiLU,
        ),
        in_keys=["observation"],
    ).to(device)


def train(args: argparse.Namespace) -> Path:
    """Run PPO and return the checkpoint path."""
    device = torch.device(args.device)
    env = make_env(
        **{key: getattr(args, key) for key in TASK_ARGS},
        num_envs=args.num_envs,
        seed=args.seed,
        device=device,
    )
    actor = make_policy(env, device=device, hidden=args.hidden)
    critic = make_critic(env, device=device, hidden=args.hidden)
    frames_per_batch = args.num_envs * args.steps_per_batch
    if frames_per_batch < args.minibatch:
        raise ValueError(
            f"minibatch={args.minibatch} exceeds the {frames_per_batch} frames "
            "collected per batch, so no update would run."
        )
    collector = Collector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=args.frames,
        device=device,
        auto_register_policy_transforms=True,
    )
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
    )
    advantage = GAE(
        gamma=args.gamma, lmbda=0.95, value_network=critic, average_gae=True
    )
    loss = ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=0.2,
        entropy_coeff=args.entropy,
        critic_coeff=1.0,
        loss_critic_type="smooth_l1",
    )
    optim = torch.optim.Adam(loss.parameters(), args.lr)
    frames = 0
    metrics: dict[str, float] = {}
    for batch in collector:
        for _ in range(args.epochs):
            advantage(batch)
            buffer.extend(batch.reshape(-1))
            for _ in range(batch.numel() // args.minibatch):
                losses = loss(buffer.sample(args.minibatch))
                (
                    losses["loss_objective"]
                    + losses["loss_critic"]
                    + losses["loss_entropy"]
                ).backward()
                torch.nn.utils.clip_grad_norm_(loss.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
        frames += batch.numel()
        metrics = {
            "reward": batch["next", "reward"].mean().item(),
            "terminated": batch["next", "terminated"].float().mean().item(),
            "forward_speed": batch["next", "qvel"][..., 0].mean().item(),
        }
        torchrl_logger.info(
            "frames %d reward/step %.4f terminated %.4f forward m/s %.3f",
            frames,
            metrics["reward"],
            metrics["terminated"],
            metrics["forward_speed"],
        )
        save_render_checkpoint(
            args.ckpt,
            actor,
            frames=frames,
            metrics=metrics,
            config=vars(args),
            extra={"critic_state_dict": critic.state_dict()},
            format="archive",
        )
    collector.shutdown()
    return Path(args.ckpt)


if __name__ == "__main__":
    train(parse_args())
