# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Closed-form MicroDuck gait as a TensorDict policy, judged from foot contacts.

:class:`MicroDuckGaitActor` computes a walking gait from the
:class:`~torchrl.envs.MicroDuckEnv` observation alone, so the same module is the
baseline controller, the prior of the PPO actor in ``ppo_mujoco.py`` and the
``rlrender`` policy. :func:`gait_metrics` reads a rollout of an env built with
``diagnostics=True`` and counts swing phases from foot contacts, because a
planted-foot controller can move forward by pitching its torso.

Validate the default gait from a TorchRL checkout::

    python examples/microduck/heuristic_gait.py --download
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl import torchrl_logger
from torchrl.envs import MicroDuckEnv
from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.envs.custom.mujoco._skill_models import (
    MicroDuckGaitActor,
    MicroDuckGaitConfig,
)

MIN_SINGLE_SUPPORT_STEPS = 4
MIN_SWING_PHASES = 4
MAX_WALKING_PITCH = 0.2


def gait_metrics(rollout: TensorDictBase) -> TensorDict:
    """Measure one single-env rollout of a ``diagnostics=True`` MicroDuck env.

    A swing phase is a stretch of the gait clock during which the same foot is
    the swing foot; it counts when that foot was airborne while the other foot
    stayed planted for at least ``MIN_SINGLE_SUPPORT_STEPS`` steps. The rollout
    ``walking`` flag requires survival, ``MIN_SWING_PHASES`` such phases per
    foot, a torso pitch below ``MAX_WALKING_PITCH`` rad and a mean forward speed
    in the commanded direction.

    Returns:
        A scalar :class:`~tensordict.TensorDict` of float metrics: ``survived``,
        ``episode_length``, ``forward_speed`` (m/s, body frame), ``max_abs_pitch``,
        ``left_swing_phases``, ``right_swing_phases``,
        ``left_single_support_steps``, ``right_single_support_steps``,
        ``left_foot_height_max``, ``right_foot_height_max`` and ``walking``.
    """
    rollout = rollout.reshape(-1)
    observation = rollout["observation"]
    after = rollout["next"]
    direction = observation[0, MicroDuckEnv.COMMAND_START].sign()
    # The left foot swings while the directed gait clock is positive.
    swing_left = direction * observation[:, MicroDuckEnv.GAIT_PHASE_START] >= 0
    left_contact = after["diagnostic_left_foot_contact"][:, 0] > 0.5
    right_contact = after["diagnostic_right_foot_contact"][:, 0] > 0.5
    single_support = torch.where(
        swing_left, ~left_contact & right_contact, ~right_contact & left_contact
    )
    segment = torch.cat(
        (
            swing_left.new_zeros(1, dtype=torch.long),
            (swing_left[1:] != swing_left[:-1]).cumsum(0),
        )
    )
    num_segments = int(segment[-1]) + 1
    support_steps = torch.zeros(num_segments).index_add_(
        0, segment, single_support.float()
    )
    segment_swings_left = torch.zeros(num_segments, dtype=torch.bool)
    segment_swings_left[segment] = swing_left
    valid_phase = support_steps >= MIN_SINGLE_SUPPORT_STEPS
    left_swing_phases = (valid_phase & segment_swings_left).sum()
    right_swing_phases = (valid_phase & ~segment_swings_left).sum()

    pitch = torch.cat((observation[:1, 0], after["observation"][:, 0]))
    max_abs_pitch = pitch.clamp(-1.0, 1.0).asin().abs().max()
    survived = ~after["terminated"][-1].any()
    forward_speed = after["observation"][:, 6].mean()
    walking = (
        survived
        & (direction * forward_speed > 0)
        & (max_abs_pitch < MAX_WALKING_PITCH)
        & (left_swing_phases >= MIN_SWING_PHASES)
        & (right_swing_phases >= MIN_SWING_PHASES)
    )
    return TensorDict(
        {
            "survived": survived,
            "episode_length": torch.tensor(rollout.shape[-1]),
            "forward_speed": forward_speed,
            "max_abs_pitch": max_abs_pitch,
            "left_swing_phases": left_swing_phases,
            "right_swing_phases": right_swing_phases,
            "left_single_support_steps": (single_support & swing_left).sum(),
            "right_single_support_steps": (single_support & ~swing_left).sum(),
            "left_foot_height_max": after["diagnostic_left_foot_height"].max(),
            "right_foot_height_max": after["diagnostic_right_foot_height"].max(),
            "walking": walking,
        }
    ).float()


def make_render_policy(
    checkpoint: Mapping[str, Any] | None = None,
    policy_kwargs: Mapping[str, float] | None = None,
) -> MicroDuckGaitActor:
    """Build the gait actor for ``rlrender``.

    The checkpoint is the file written by ``--render-checkpoint``, a mapping
    with the gait parameters under ``"gait"``; ``--policy-kwargs`` fields
    override them. For instance::

        rlrender --ckpt microduck_gait.pt --no-auto-load-policy \\
            --policy examples/microduck/heuristic_gait.py:make_render_policy \\
            --policy-kwargs '{"frequency_hz": 2.0}' \\
            --env examples/microduck/heuristic_gait.py:make_env ...

    renders the saved gait with its clock sped up to 2 Hz.
    """
    gait = dict(checkpoint.get("gait") or {}) if isinstance(checkpoint, Mapping) else {}
    gait.update(policy_kwargs or {})
    return MicroDuckGaitActor(gait)


def make_env(
    microduck_root: str | Path | None = None,
    *,
    download: bool | str = False,
    backend: BackendName = "mujoco",
    speed: float = 0.03,
    seed: int = 0,
    reset_noise_scale: float = MicroDuckEnv.RESET_NOISE_SCALE,
    gait: MicroDuckGaitConfig | Mapping[str, float] | None = None,
    max_episode_steps: int = 500,
    camera_id: int = -1,
    render_width: int = 640,
    render_height: int = 480,
) -> MicroDuckEnv:
    """Build a single MicroDuck env whose gait clock matches ``gait``.

    Diagnostics are enabled so :func:`gait_metrics` can read foot contacts and
    heights from the rollout. Only environment arguments are accepted so
    ``rlrender`` can call this factory with its own keyword arguments.
    """
    gait = MicroDuckGaitActor(gait).config
    return MicroDuckEnv(
        microduck_root,
        download=download,
        backend=backend,
        tasks=MicroDuckEnv.tracking_task(speed, **gait.task_kwargs()),
        diagnostics=True,
        num_envs=1,
        seed=seed,
        reset_noise_scale=reset_noise_scale,
        max_episode_steps=max_episode_steps,
        camera_id=camera_id,
        render_width=render_width,
        render_height=render_height,
    )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck-root", type=Path)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the pinned microduck_rl assets when no checkout is found.",
    )
    parser.add_argument(
        "--backend", choices=("mujoco", "mjx", "mujoco-torch"), default="mujoco"
    )
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--reset-noise-scale", type=float, default=0.02)
    parser.add_argument("--commanded-x-velocity", type=float, default=0.03)
    parser.add_argument(
        "--render-checkpoint",
        type=Path,
        help="Save the gait parameters as an rlrender checkpoint.",
    )
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    actor = MicroDuckGaitActor()
    env = make_env(
        args.microduck_root,
        download=args.download,
        backend=args.backend,
        speed=args.commanded_x_velocity,
        reset_noise_scale=args.reset_noise_scale,
        gait=actor.config,
        max_episode_steps=args.steps,
    )
    metrics = []
    with torch.no_grad():
        for seed in range(args.num_seeds):
            env.set_seed(seed)
            metrics.append(
                gait_metrics(env.rollout(args.steps, actor, break_when_any_done=True))
            )
    env.close()
    metrics = torch.stack(metrics)
    summary = TensorDict(mean=metrics.mean(dim=0), min=metrics.min(dim=0).values)
    torchrl_logger.info("Gait config: %s", asdict(actor.config))
    torchrl_logger.info(
        "Gait metrics over %d seeds: %s",
        args.num_seeds,
        {
            "/".join(key): round(float(value), 4)
            for key, value in summary.items(include_nested=True, leaves_only=True)
        },
    )
    if args.render_checkpoint is not None:
        args.render_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"gait": asdict(actor.config)}, args.render_checkpoint)
        torchrl_logger.info(
            "Saved rlrender gait checkpoint to %s", args.render_checkpoint
        )


if __name__ == "__main__":
    main(parse_args())
