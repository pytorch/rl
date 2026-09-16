# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PPO on a MuJoCo Menagerie robot: hold the home pose, or walk.

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
        --download --num-envs 6 --frames 20000000

A UR5e holding its pose from a local checkout, as a quick check::

    TORCHRL_MUJOCO_MENAGERIE_PATH=~/mujoco_menagerie \\
        python examples/menagerie/ppo.py --task hold_pose --robot universal_robots_ur5e --smoke

Render the checkpoint as a video from the scene's first camera (``--fps 50``
is real time for the 20 ms control step), or as a notebook with a saved
rollout and a cell that collects a fresh one in the kernel. The checkpoint
records the training arguments, so the factories rebuild the same robot and
task; ``--env-kwargs`` overrides them and sets the render camera and size::

    rlrender --ckpt menagerie_ppo.ckpt \\
        --policy examples/menagerie/ppo.py:make_policy \\
        --env examples/menagerie/ppo.py:make_env \\
        --env-kwargs '{"camera_id": 0, "render_width": 640, "render_height": 480}' \\
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
from typing import Any, Literal

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
    MenagerieEnv,
    MenagerieTask,
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
    "entry",
    "frame_skip",
    "max_steps",
    "download",
    "fall_height",
    "alive_bonus",
    "control_cost",
    "command",
    "action_scale",
    "feet_sites",
    "feet_geoms",
)

WALK_REWARD_WEIGHTS = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -0.5,
    "ang_vel_xy": -0.05,
    "orientation": -5.0,
    "pose": 0.5,
    "termination": -1.0,
    "action_rate": -0.01,
    "feet_slip": -0.1,
    "feet_clearance": -2.0,
    "feet_height": -0.2,
    "feet_air_time": 0.1,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Training arguments; the task entries are recorded in the checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=("hold_pose", "walk"), default="hold_pose")
    parser.add_argument(
        "--robot", default="unitree_go2", help="Menagerie model directory."
    )
    parser.add_argument(
        "--entry", default=None, help="XML entry point, e.g. scene_mjx."
    )
    parser.add_argument(
        "--frame-skip", type=int, default=10, help="Physics steps per action."
    )
    parser.add_argument("--max-steps", type=int, default=1000, help="Episode horizon.")
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
    args.command = tuple(float(v) for v in args.command)
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
    orientation costs, a pose term, action rate, foot slip, clearance and
    swing height, air time at touchdown, a termination cost), summed, clipped
    at zero and multiplied by the control period. Termination: the base turns
    over. Foot contacts come from the env's ``geom_contacts``; the foot
    velocities are finite differences of the ``site_positions`` observation.
    Air time, the last contact and the swing peak of every foot travel in the
    tensordict, so the transform holds no state of its own.

    Args:
        command (Sequence[float]): ``(vx, vy, wz)`` in body-frame m/s and rad/s.
        home_joints (Tensor): joint angles of the home pose, in ``qpos[7:]`` order.
        feet_geoms (Sequence[str]): foot collision geoms, in the order of the
            ``site_positions`` observation.
        dt (float): control period in seconds.
        weights (Mapping[str, float], optional): reward weights per term;
            defaults to :data:`WALK_REWARD_WEIGHTS`.
        tracking_sigma (float, optional): scale of the tracking terms. Defaults to ``0.25``.
        max_foot_height (float, optional): target swing height in meters. Defaults to ``0.1``.
    """

    def __init__(
        self,
        command: Sequence[float],
        home_joints: torch.Tensor,
        feet_geoms: Sequence[str],
        dt: float,
        *,
        weights: Mapping[str, float] | None = None,
        tracking_sigma: float = 0.25,
        max_foot_height: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("command", torch.as_tensor(command, dtype=torch.float32))
        self.register_buffer("home_joints", home_joints.clone())
        self.register_buffer(
            "pose_weight", torch.tensor([1.0, 1.0, 0.1] * (home_joints.numel() // 3))
        )
        self.feet_geoms = tuple(feet_geoms)
        self.dt = float(dt)
        self.weights = dict(WALK_REWARD_WEIGHTS if weights is None else weights)
        self.tracking_sigma = float(tracking_sigma)
        self.max_foot_height = float(max_foot_height)

    @property
    def observation_dim(self) -> int:
        return 3 + 3 + 2 * self.home_joints.numel() + self.home_joints.numel() + 3

    def _observation(self, next_tensordict: TensorDictBase) -> torch.Tensor:
        qpos = next_tensordict.get("qpos")
        qvel = next_tensordict.get("qvel")
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
                self.command.expand(*qpos.shape[:-1], 3),
            ],
            dim=-1,
        )

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        feet = tensordict_reset.get("site_positions")
        zeros = feet.new_zeros(feet.shape[:-1])
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
        action = next_tensordict.get("prev_action")
        last_action = tensordict.get("prev_action")
        command = self.command.expand(*qpos.shape[:-1], 3)
        cmd_norm = command.norm(dim=-1)

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
        return observation_spec


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------


def make_single_env(
    settings: Mapping[str, Any],
    *,
    seed: int | None = None,
    device: torch.device | str | None = None,
    from_pixels: bool = False,
    camera_id: int = -1,
    render_width: int = 320,
    render_height: int = 240,
) -> TransformedEnv:
    """One env with its task transforms; ``settings`` holds the :data:`TASK_ARGS`."""
    walk = settings["task"] == "walk"
    if walk:
        task = MenagerieTask(site_names=settings["feet_sites"])
    else:
        task = MenagerieEnv.hold_pose_task(
            control_cost_weight=settings["control_cost"],
            alive_bonus=settings["alive_bonus"],
            terminate_below_height=settings["fall_height"],
        )
    env = MenagerieEnv(
        settings["robot"],
        entry=settings["entry"],
        download=settings["download"],
        task=task,
        backend="mujoco",
        frame_skip=settings["frame_skip"],
        seed=seed,
        device=device,
        max_episode_steps=settings["max_steps"],
        from_pixels=from_pixels,
        camera_id=camera_id,
        render_width=render_width,
        render_height=render_height,
    )
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
    return TransformedEnv(
        env,
        Compose(
            HomeOffsetActions(home, settings["action_scale"]),
            QuadrupedJoystick(
                settings["command"], home, settings["feet_geoms"], env.dt
            ),
        ),
    )


def make_env(
    robot: str | None = None,
    *,
    task: Literal["hold_pose", "walk"] | None = None,
    entry: str | None = None,
    frame_skip: int | None = None,
    max_steps: int | None = None,
    download: bool | None = None,
    fall_height: float | None = None,
    alive_bonus: float | None = None,
    control_cost: float | None = None,
    command: Sequence[float] | None = None,
    action_scale: float | None = None,
    feet_sites: Sequence[str] | None = None,
    feet_geoms: Sequence[str] | None = None,
    num_envs: int = 1,
    seed: int | None = None,
    device: torch.device | str | None = None,
    from_pixels: bool = False,
    camera_id: int = -1,
    render_width: int = 320,
    render_height: int = 240,
    checkpoint: Mapping[str, Any] | None = None,
) -> EnvBase:
    """Build the env, batched over worker processes when ``num_envs > 1``.

    ``rlrender`` calls this with the checkpoint. Explicit arguments win over
    the training arguments recorded in ``checkpoint``, which win over the
    defaults of :func:`parse_args`. ``from_pixels`` adds frames from
    ``camera_id`` (MuJoCo's free camera by default, ``0`` for the first camera
    of the scene), e.g. ``--env-kwargs '{"camera_id": 0, "render_width": 640}'``.
    """
    recorded = dict((checkpoint or {}).get("config") or {})
    defaults = vars(parse_args([]))
    settings = {key: recorded.get(key, defaults[key]) for key in TASK_ARGS}
    explicit = {
        "task": task,
        "robot": robot,
        "entry": entry,
        "frame_skip": frame_skip,
        "max_steps": max_steps,
        "download": download,
        "fall_height": fall_height,
        "alive_bonus": alive_bonus,
        "control_cost": control_cost,
        "command": command,
        "action_scale": action_scale,
        "feet_sites": feet_sites,
        "feet_geoms": feet_geoms,
    }
    settings.update(
        {key: value for key, value in explicit.items() if value is not None}
    )
    render = {
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
