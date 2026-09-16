# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PPO on a MuJoCo Menagerie robot holding its home pose.

A short recipe around :class:`~torchrl.envs.MenagerieEnv`: the
:meth:`~torchrl.envs.MenagerieEnv.hold_pose_task` preset, a Gaussian MLP actor
and an MLP critic, a :class:`~torchrl.collectors.Collector` feeding
:class:`~torchrl.objectives.ClipPPOLoss` with GAE, and a checkpoint written
with :func:`~torchrl.render.save_render_checkpoint`. :func:`make_env` and
:func:`make_policy` are the factories ``rlrender`` imports to play it back.

Train the Unitree Go2 to stand from the ``mujoco-menagerie`` package cache
(``--download`` fetches the model once). Its MJX scene drives the joints with
position servos, so the policy learns target angles::

    python examples/menagerie/ppo_hold_pose.py --robot unitree_go2 --entry scene_mjx \\
        --fall-height 0.15 --download --num-envs 4 --frames 200000

A UR5e from a local checkout, as a quick check::

    TORCHRL_MUJOCO_MENAGERIE_PATH=~/mujoco_menagerie \\
        python examples/menagerie/ppo_hold_pose.py --robot universal_robots_ur5e --smoke

Render the checkpoint as a video from the scene's first camera (``--fps 100``
is real time: the control step is 10 ms), or as a notebook with a saved
rollout and a cell that collects a fresh one in the kernel. The checkpoint
records the training arguments, so the factories rebuild the same robot and
task; ``--env-kwargs`` overrides them and sets the render camera and size::

    rlrender --ckpt menagerie_ppo.ckpt \\
        --policy examples/menagerie/ppo_hold_pose.py:make_policy \\
        --env examples/menagerie/ppo_hold_pose.py:make_env \\
        --env-kwargs '{"camera_id": 0, "render_width": 640, "render_height": 480}' \\
        --deterministic --from-pixels --render-backend pixels \\
        --max-steps 300 --fps 100 --format mp4 --out menagerie_ppo.mp4 --overwrite
    rlrender --ckpt menagerie_ppo.ckpt \\
        --policy examples/menagerie/ppo_hold_pose.py:make_policy \\
        --env examples/menagerie/ppo_hold_pose.py:make_env \\
        --deterministic --from-pixels --render-backend pixels \\
        --max-steps 300 --fps 100 --format ipynb --out menagerie_ppo.ipynb \\
        --notebook-rollout-mode both --overwrite
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import CatTensors, MenagerieEnv, TransformedEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.render import save_render_checkpoint

TASK_ARGS = (
    "robot",
    "entry",
    "fall_height",
    "alive_bonus",
    "control_cost",
    "max_steps",
    "download",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Training arguments; the robot and task entries are recorded in the checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--robot", default="unitree_go2", help="Menagerie model directory."
    )
    parser.add_argument(
        "--entry", default=None, help="XML entry point, e.g. scene_mjx."
    )
    parser.add_argument(
        "--fall-height",
        type=float,
        default=None,
        help="Terminate when the floating base drops below this height (m).",
    )
    parser.add_argument("--alive-bonus", type=float, default=0.5)
    parser.add_argument("--control-cost", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=300, help="Episode horizon.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Let the mujoco-menagerie package fetch the robot into its cache.",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--frames", type=int, default=200_000)
    parser.add_argument("--frames-per-batch", type=int, default=4096)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ckpt", default="menagerie_ppo.ckpt")
    parser.add_argument(
        "--smoke", action="store_true", help="One tiny update on one env."
    )
    args = parser.parse_args(argv)
    if args.smoke:
        args.num_envs = 1
        args.frames = 128
        args.frames_per_batch = 64
        args.minibatch = 32
        args.epochs = 1
    return args


def make_env(
    robot: str | None = None,
    *,
    entry: str | None = None,
    fall_height: float | None = None,
    alive_bonus: float | None = None,
    control_cost: float | None = None,
    max_steps: int | None = None,
    download: bool | None = None,
    num_envs: int = 1,
    parallel: bool = False,
    seed: int | None = None,
    device: torch.device | str | None = None,
    from_pixels: bool = False,
    camera_id: int = -1,
    render_width: int = 320,
    render_height: int = 240,
    checkpoint: Mapping[str, Any] | None = None,
) -> TransformedEnv:
    """Build the batched hold-pose env; ``rlrender`` calls this with the checkpoint.

    Explicit arguments win over the training arguments recorded in
    ``checkpoint``, which win over the defaults of :func:`parse_args`. The
    transform concatenates ``qpos`` and ``qvel`` into the ``observation`` the
    policy reads. ``from_pixels`` adds frames from ``camera_id`` (MuJoCo's free
    camera by default, ``0`` for the first camera of the scene) for
    ``rlrender``, e.g. ``--env-kwargs '{"camera_id": 0, "render_width": 640}'``.
    """
    recorded = dict((checkpoint or {}).get("config") or {})
    defaults = vars(parse_args([]))
    settings = {key: recorded.get(key, defaults[key]) for key in TASK_ARGS}
    explicit = {
        "robot": robot,
        "entry": entry,
        "fall_height": fall_height,
        "alive_bonus": alive_bonus,
        "control_cost": control_cost,
        "max_steps": max_steps,
        "download": download,
    }
    settings.update(
        {key: value for key, value in explicit.items() if value is not None}
    )
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
        num_envs=num_envs,
        parallel=parallel,
        seed=seed,
        device=device,
        max_episode_steps=settings["max_steps"],
        from_pixels=from_pixels,
        camera_id=camera_id,
        render_width=render_width,
        render_height=render_height,
    )
    return TransformedEnv(
        env, CatTensors(in_keys=["qpos", "qvel"], out_key="observation", del_keys=False)
    )


def make_policy(
    env: TransformedEnv,
    *,
    device: torch.device | str = "cpu",
    hidden: int | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> ProbabilisticActor:
    """Gaussian MLP actor over ``observation``; ``rlrender`` loads the checkpoint into it."""
    if hidden is None:
        hidden = int(((checkpoint or {}).get("config") or {}).get("hidden", 256))
    action_spec = env.action_spec_unbatched
    net = nn.Sequential(
        MLP(
            in_features=env.observation_spec["observation"].shape[-1],
            out_features=2 * action_spec.shape[-1],
            num_cells=[hidden, hidden],
            activation_class=nn.Tanh,
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
    env: TransformedEnv, *, device: torch.device | str = "cpu", hidden: int = 256
) -> ValueOperator:
    """MLP state-value critic over ``observation``."""
    return ValueOperator(
        MLP(
            in_features=env.observation_spec["observation"].shape[-1],
            out_features=1,
            num_cells=[hidden, hidden],
            activation_class=nn.Tanh,
        ),
        in_keys=["observation"],
    ).to(device)


def train(args: argparse.Namespace) -> Path:
    """Run PPO and return the checkpoint path."""
    device = torch.device(args.device)
    env = make_env(
        args.robot,
        entry=args.entry,
        fall_height=args.fall_height,
        alive_bonus=args.alive_bonus,
        control_cost=args.control_cost,
        max_steps=args.max_steps,
        download=args.download,
        num_envs=args.num_envs,
        parallel=args.num_envs > 1,
        seed=args.seed,
        device=device,
    )
    actor = make_policy(env, device=device, hidden=args.hidden)
    critic = make_critic(env, device=device, hidden=args.hidden)
    collector = Collector(
        env,
        actor,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.frames,
        device=device,
        auto_register_policy_transforms=True,
    )
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(args.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
    )
    advantage = GAE(gamma=0.99, lmbda=0.95, value_network=critic, average_gae=True)
    loss = ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=0.2,
        entropy_coeff=1e-3,
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
        }
        torchrl_logger.info(
            "frames %d reward/step %.3f terminated %.3f",
            frames,
            metrics["reward"],
            metrics["terminated"],
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
