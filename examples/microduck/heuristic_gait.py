# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Closed-form MicroDuck gait: baseline controller, gait metrics, rlrender policy.

The gait combines a bilateral phase oscillator over the hip, knee, ankle and
lateral targets with proportional-derivative pitch feedback. It is computed
from the :class:`~torchrl.envs.MicroDuckEnv` observation alone, so the same
function drives this baseline, initializes the PPO actor in ``ppo_mujoco.py``
and serves as the ``rlrender`` policy.

Gait quality is measured from foot contacts rather than displacement: a
planted-foot controller can move forward by pitching its torso, so a rollout
only counts as walking when both feet alternate swing phases while the other
foot stays in single support, the torso pitch stays bounded, and the robot
moves in the commanded direction.

Validate the default gait from a TorchRL checkout::

    python examples/microduck/heuristic_gait.py --microduck-root /path/to/microduck_rl

Search around it::

    python examples/microduck/heuristic_gait.py --microduck-root /path/to/microduck_rl \\
        --search-candidates 128
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl import torchrl_logger
from torchrl.envs import MicroDuckEnv
from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.render import RenderPolicySpec

MIN_SINGLE_SUPPORT_STEPS = 4
MIN_SWING_PHASES = 4
MAX_WALKING_PITCH = 0.2


@dataclass(frozen=True)
class MicroDuckGaitConfig:
    """Parameters of the closed-form MicroDuck gait.

    Amplitudes are normalized actions: an amplitude of one is a full
    ``action_scale`` offset around the MJCF ``STAND`` actuator target. The
    oscillator drives mirrored leg joints half a cycle apart, and pitch
    feedback acts through the hip and ankle targets. The clock parameters are
    forwarded to :class:`~torchrl.envs.MicroDuckEnv` through :meth:`env_kwargs`
    so the gait phase read from the observation matches this configuration.
    """

    frequency_hz: float = 1.8913
    hip_amplitude: float = 0.999
    knee_amplitude: float = 0.9097
    ankle_amplitude: float = 0.0317
    lateral_amplitude: float = 0.9584
    lateral_phase_offset: float = -0.1624
    phase_offset: float = -1.5237
    pitch_kp: float = -9.9495
    pitch_kd: float = -0.6119
    ankle_pitch_kp: float = 10.9934
    ankle_pitch_kd: float = 0.6033
    ramp_duration_s: float = 0.4

    def env_kwargs(self) -> dict[str, float]:
        """Return the :class:`~torchrl.envs.MicroDuckEnv` gait-clock arguments."""
        return {
            "gait_frequency_hz": self.frequency_hz,
            "gait_phase_offset": self.phase_offset,
            "gait_ramp_duration_s": self.ramp_duration_s,
        }


def gait_action(config: MicroDuckGaitConfig, observation: torch.Tensor) -> torch.Tensor:
    """Compute the closed-form normalized action from a MicroDuck observation.

    The observation layout is the one produced by
    :class:`~torchrl.envs.MicroDuckEnv`: projected gravity, body angular
    velocity, measured and commanded longitudinal velocity, joint errors and
    velocities, the gait clock and the previous action. The command sign sets
    the walking direction; a zero command keeps only the balance feedback.

    Args:
        config: oscillator and pitch-feedback parameters.
        observation: tensor of shape ``(*, MicroDuckEnv.OBSERVATION_DIM)``.

    Returns:
        A tensor of normalized actions in ``[-1, 1]`` with shape ``(*, 14)``.
    """
    phase_start = MicroDuckEnv.GAIT_PHASE_START
    pitch = observation[..., 0:1].clamp(-1.0, 1.0).asin()
    pitch_rate = observation[..., 4:5]
    direction = observation[..., 7:8].sign()
    gait_sin = direction * observation[..., phase_start : phase_start + 1]
    gait_cos = direction * observation[..., phase_start + 1 : phase_start + 2]
    ramp = observation[..., phase_start + 2 : phase_start + 3]

    pitch_correction = (config.pitch_kp * pitch + config.pitch_kd * pitch_rate).clamp(
        -0.95, 0.95
    )
    ankle_pitch_correction = (
        config.ankle_pitch_kp * pitch + config.ankle_pitch_kd * pitch_rate
    ).clamp(-1.0, 1.0)
    gait_wave = ramp * gait_sin
    left_swing = ramp * gait_sin.clamp_min(0.0)
    right_swing = ramp * (-gait_sin).clamp_min(0.0)
    lateral_wave = ramp * (
        gait_sin * math.cos(config.lateral_phase_offset)
        + gait_cos * math.sin(config.lateral_phase_offset)
    )

    action = observation.new_zeros(*observation.shape[:-1], MicroDuckEnv.NUM_JOINTS)
    # A common hip-pitch oscillation produces opposite physical leg motion
    # because the left and right joints use opposite sign conventions.
    action[..., 2:3] = -pitch_correction - config.hip_amplitude * gait_wave
    action[..., 11:12] = pitch_correction - config.hip_amplitude * gait_wave
    action[..., 3:4] = config.knee_amplitude * left_swing
    action[..., 12:13] = -config.knee_amplitude * right_swing
    action[..., 4:5] = ankle_pitch_correction - config.ankle_amplitude * left_swing
    action[..., 13:14] = -ankle_pitch_correction + config.ankle_amplitude * right_swing
    action[..., 1:2] = config.lateral_amplitude * lateral_wave
    action[..., 10:11] = config.lateral_amplitude * lateral_wave
    return action.clamp(-1.0, 1.0)


def heading_xy(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the planar body x-axis of a wxyz quaternion in the world frame."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + w * z)), dim=-1
    )


class MicroDuckGaitPolicy:
    """TensorDict policy writing the closed-form gait action."""

    def __init__(self, config: MicroDuckGaitConfig):
        self.config = config

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.set("action", gait_action(self.config, tensordict["observation"]))
        return tensordict


def make_render_policy(spec: RenderPolicySpec) -> MicroDuckGaitPolicy:
    """Build the closed-form policy for an ``rlrender`` rollout.

    The checkpoint may contain a ``gait`` mapping with
    :class:`MicroDuckGaitConfig` fields; ``policy_kwargs`` override them.
    """
    gait: dict[str, Any] = {}
    checkpoint = spec.checkpoint
    if isinstance(checkpoint, Mapping) and checkpoint.get("gait") is not None:
        if not isinstance(checkpoint["gait"], Mapping):
            raise TypeError("The heuristic checkpoint `gait` entry must be a mapping.")
        gait.update(checkpoint["gait"])
    gait.update(spec.policy_kwargs)
    return MicroDuckGaitPolicy(MicroDuckGaitConfig(**gait))


def make_env(
    microduck_root: str | Path | None = None,
    *,
    backend: BackendName = "mujoco",
    commanded_x_velocity: float | Sequence[float] = 0.03,
    seed: int = 0,
    reset_noise_scale: float = MicroDuckEnv.RESET_NOISE_SCALE,
    gait: MicroDuckGaitConfig | Mapping[str, float] | None = None,
    **kwargs: Any,
) -> MicroDuckEnv:
    """Build a single MicroDuck env whose gait clock matches ``gait``."""
    if isinstance(gait, Mapping):
        gait = MicroDuckGaitConfig(**gait)
    elif gait is None:
        gait = MicroDuckGaitConfig()
    return MicroDuckEnv(
        microduck_root,
        backend=backend,
        commanded_x_velocity=commanded_x_velocity,
        num_envs=1,
        seed=seed,
        reset_noise_scale=reset_noise_scale,
        **gait.env_kwargs(),
        **kwargs,
    )


@dataclass(frozen=True)
class GaitTrial:
    """Contact-derived metrics of one fixed-seed rollout."""

    seed: int
    episode_length: int
    survived: bool
    signed_displacement: float
    average_forward_speed: float
    max_abs_pitch: float
    left_swing_phases: int
    right_swing_phases: int
    left_single_support_steps: int
    right_single_support_steps: int
    left_foot_height_max: float
    right_foot_height_max: float

    @property
    def walking(self) -> bool:
        """Whether the rollout alternated feet while upright and moving forward."""
        return (
            self.survived
            and self.signed_displacement > 0.0
            and self.max_abs_pitch < MAX_WALKING_PITCH
            and self.left_swing_phases >= MIN_SWING_PHASES
            and self.right_swing_phases >= MIN_SWING_PHASES
        )


@torch.no_grad()
def run_trial(
    env: MicroDuckEnv,
    config: MicroDuckGaitConfig,
    *,
    seed: int,
    steps: int,
    commanded_x_velocity: float,
) -> GaitTrial:
    """Roll the closed-form gait out for one seed and measure it from contacts."""
    if env.batch_size.numel() != 1:
        raise ValueError("Gait trials expect a single-environment MicroDuckEnv.")
    env.set_seed(seed)
    tensordict = env.reset(
        TensorDict(
            {
                "commanded_x_velocity": torch.full(
                    (1, 1), float(commanded_x_velocity), dtype=env.dtype
                )
            },
            batch_size=env.batch_size,
        )
    )
    start_state = env.get_state()
    start_qpos = start_state["qpos"][0].to(env.dtype)
    heading = heading_xy(start_qpos[3:7])
    direction = 1.0 if commanded_x_velocity >= 0 else -1.0
    phase_start = MicroDuckEnv.GAIT_PHASE_START

    max_abs_pitch = abs(float(tensordict["observation"][0, 0].clamp(-1, 1).asin()))
    heights = env.foot_heights()[0]
    left_height_max, right_height_max = float(heights[0]), float(heights[1])
    left_swing_phases = right_swing_phases = 0
    left_support = right_support = 0
    current_swing_is_left: bool | None = None
    support_run = support_run_max = 0
    survived = True
    episode_length = 0
    for _ in range(steps):
        episode_length += 1
        observation = tensordict["observation"]
        swing_is_left = bool(direction * observation[0, phase_start] >= 0.0)
        if current_swing_is_left is None:
            current_swing_is_left = swing_is_left
        elif swing_is_left != current_swing_is_left:
            if support_run_max >= MIN_SINGLE_SUPPORT_STEPS:
                if current_swing_is_left:
                    left_swing_phases += 1
                else:
                    right_swing_phases += 1
            current_swing_is_left = swing_is_left
            support_run = support_run_max = 0

        tensordict.set("action", gait_action(config, observation))
        tensordict = env.step(tensordict)["next"]

        left_contact, right_contact = env.foot_contacts()[0].tolist()
        single_support = (
            (not left_contact and right_contact)
            if swing_is_left
            else (not right_contact and left_contact)
        )
        if single_support:
            support_run += 1
            support_run_max = max(support_run_max, support_run)
            if swing_is_left:
                left_support += 1
            else:
                right_support += 1
        else:
            support_run = 0
        heights = env.foot_heights()[0]
        left_height_max = max(left_height_max, float(heights[0]))
        right_height_max = max(right_height_max, float(heights[1]))
        pitch = float(tensordict["observation"][0, 0].clamp(-1, 1).asin())
        max_abs_pitch = max(max_abs_pitch, abs(pitch))
        if bool(tensordict["terminated"].any()):
            survived = False
            break
        if bool(tensordict["done"].any()):
            break
    if (
        support_run_max >= MIN_SINGLE_SUPPORT_STEPS
        and current_swing_is_left is not None
    ):
        if current_swing_is_left:
            left_swing_phases += 1
        else:
            right_swing_phases += 1

    end_state = env.get_state()
    end_qpos = end_state["qpos"][0].to(env.dtype)
    displacement = float(torch.dot(end_qpos[:2] - start_qpos[:2], heading))
    elapsed = float(end_state["time"][0] - start_state["time"][0])
    return GaitTrial(
        seed=seed,
        episode_length=episode_length,
        survived=survived,
        signed_displacement=displacement,
        average_forward_speed=displacement / elapsed if elapsed > 0 else 0.0,
        max_abs_pitch=max_abs_pitch,
        left_swing_phases=left_swing_phases,
        right_swing_phases=right_swing_phases,
        left_single_support_steps=left_support,
        right_single_support_steps=right_support,
        left_foot_height_max=left_height_max,
        right_foot_height_max=right_height_max,
    )


def evaluate_gait(
    env: MicroDuckEnv,
    config: MicroDuckGaitConfig,
    *,
    seeds: Sequence[int],
    steps: int,
    commanded_x_velocity: float,
) -> list[GaitTrial]:
    """Run :func:`run_trial` for every seed."""
    return [
        run_trial(
            env,
            config,
            seed=seed,
            steps=steps,
            commanded_x_velocity=commanded_x_velocity,
        )
        for seed in seeds
    ]


def summarize(trials: Sequence[GaitTrial]) -> dict[str, float]:
    """Aggregate gait trials, ranking contact metrics before speed."""
    lengths = np.asarray([trial.episode_length for trial in trials])
    displacements = np.asarray([trial.signed_displacement for trial in trials])
    return {
        "survival_rate": float(np.mean([trial.survived for trial in trials])),
        "walking_success_rate": float(np.mean([trial.walking for trial in trials])),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_min": float(lengths.min()),
        "signed_displacement_mean": float(displacements.mean()),
        "signed_displacement_min": float(displacements.min()),
        "forward_speed_mean": float(
            np.mean([trial.average_forward_speed for trial in trials])
        ),
        "max_abs_pitch_max": float(max(trial.max_abs_pitch for trial in trials)),
        "left_swing_phases_min": float(min(t.left_swing_phases for t in trials)),
        "right_swing_phases_min": float(min(t.right_swing_phases for t in trials)),
        "left_single_support_steps_min": float(
            min(t.left_single_support_steps for t in trials)
        ),
        "right_single_support_steps_min": float(
            min(t.right_single_support_steps for t in trials)
        ),
        "left_foot_height_max_min": float(min(t.left_foot_height_max for t in trials)),
        "right_foot_height_max_min": float(
            min(t.right_foot_height_max for t in trials)
        ),
        "wrong_way_rollouts": float(np.count_nonzero(displacements <= 0.0)),
    }


def search_key(trials: Sequence[GaitTrial]) -> tuple[float, ...]:
    """Rank candidates by survival and bilateral stepping before speed."""
    summary = summarize(trials)
    return (
        summary["episode_length_min"],
        summary["survival_rate"],
        summary["episode_length_mean"],
        min(summary["left_swing_phases_min"], summary["right_swing_phases_min"]),
        summary["walking_success_rate"],
        -summary["max_abs_pitch_max"],
        summary["forward_speed_mean"],
    )


def search_configs(
    initial: MicroDuckGaitConfig,
    num_candidates: int,
    rng: np.random.Generator,
) -> list[MicroDuckGaitConfig]:
    """Sample gait configurations around ``initial`` (which stays first)."""
    configs = [initial]
    for _ in range(num_candidates - 1):
        values = rng.normal(size=11)
        configs.append(
            replace(
                initial,
                frequency_hz=float(
                    np.clip(initial.frequency_hz + 0.2 * values[0], 0.5, 3.0)
                ),
                hip_amplitude=float(
                    np.clip(initial.hip_amplitude + 0.08 * values[1], 0.4, 1.0)
                ),
                knee_amplitude=float(
                    np.clip(initial.knee_amplitude + 0.08 * values[2], 0.4, 1.0)
                ),
                ankle_amplitude=float(
                    np.clip(initial.ankle_amplitude + 0.03 * values[3], 0.0, 0.5)
                ),
                lateral_amplitude=float(
                    np.clip(initial.lateral_amplitude + 0.08 * values[4], 0.4, 1.0)
                ),
                lateral_phase_offset=float(
                    (initial.lateral_phase_offset + 0.2 * values[5] + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                ),
                phase_offset=float(
                    (initial.phase_offset + 0.3 * values[6] + math.pi) % (2.0 * math.pi)
                    - math.pi
                ),
                pitch_kp=float(initial.pitch_kp + values[7]),
                pitch_kd=float(initial.pitch_kd + 0.15 * values[8]),
                ankle_pitch_kp=float(initial.ankle_pitch_kp + 2.0 * values[9]),
                ankle_pitch_kd=float(initial.ankle_pitch_kd + 0.25 * values[10]),
            )
        )
    return configs


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck-root", type=Path)
    parser.add_argument(
        "--backend", choices=("mujoco", "mjx", "mujoco-torch"), default="mujoco"
    )
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--reset-noise-scale", type=float, default=0.02)
    parser.add_argument("--commanded-x-velocity", type=float, default=0.03)
    parser.add_argument("--search-candidates", type=int, default=0)
    parser.add_argument("--search-num-seeds", type=int, default=8)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument(
        "--render-checkpoint",
        type=Path,
        help="Save the selected gait as an rlrender checkpoint.",
    )
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    if min(args.num_seeds, args.steps) < 1:
        raise ValueError("num_seeds and steps must be positive.")
    if args.reset_noise_scale < 0 or not math.isfinite(args.reset_noise_scale):
        raise ValueError("reset_noise_scale must be finite and non-negative.")
    if not math.isfinite(args.commanded_x_velocity):
        raise ValueError("commanded_x_velocity must be finite.")
    if args.search_candidates < 0 or args.search_num_seeds < 1:
        raise ValueError(
            "search_candidates must be non-negative and search_num_seeds positive."
        )

    def evaluate(config: MicroDuckGaitConfig, seeds: Sequence[int]) -> list[GaitTrial]:
        env = make_env(
            args.microduck_root,
            backend=args.backend,
            commanded_x_velocity=args.commanded_x_velocity,
            reset_noise_scale=args.reset_noise_scale,
            gait=config,
            max_episode_steps=args.steps,
        )
        try:
            return evaluate_gait(
                env,
                config,
                seeds=seeds,
                steps=args.steps,
                commanded_x_velocity=args.commanded_x_velocity,
            )
        finally:
            env.close()

    config = MicroDuckGaitConfig()
    if args.search_candidates:
        candidates = search_configs(
            config, args.search_candidates, np.random.default_rng(args.search_seed)
        )
        ranked = sorted(
            (
                (search_key(evaluate(candidate, range(args.search_num_seeds))), index)
                for index, candidate in enumerate(candidates)
            ),
            reverse=True,
        )
        config = candidates[ranked[0][1]]
        torchrl_logger.info("Best searched config: %s", json.dumps(asdict(config)))

    if args.render_checkpoint is not None:
        args.render_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"gait": asdict(config)}, args.render_checkpoint)
        torchrl_logger.info(
            "Saved rlrender gait checkpoint to %s", args.render_checkpoint
        )

    trials = evaluate(config, range(args.num_seeds))
    torchrl_logger.info("Gait config: %s", json.dumps(asdict(config)))
    torchrl_logger.info("Evaluation: %s", json.dumps(summarize(trials), sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
