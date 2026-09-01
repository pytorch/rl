# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Search and evaluate a closed-form MicroDuck locomotion policy.

The policy combines a bilateral phase oscillator with proportional-derivative
pitch feedback. It has no neural network and uses no gradients. Its small
parameter vector can be searched directly from simulator feedback while fixed
reset seeds make regressions reproducible.

The search is survival constrained: candidates are ranked by worst-case and
mean episode length before forward speed. This prevents a forward fall from
winning merely because it briefly produces positive velocity.

Run the validated controller from a TorchRL checkout::

    python examples/mujoco/heuristic_microduck.py \
        --microduck-root /path/to/microduck_rl

Run another local search around it with::

    python examples/mujoco/heuristic_microduck.py \
        --microduck-root /path/to/microduck_rl \
        --search-candidates 128
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDictBase

from torchrl import torchrl_logger
from torchrl.render import RenderPolicySpec

if __package__:
    from examples.mujoco.ppo_microduck import (
        _low_cost_collision_scene,
        MAX_EPISODE_STEPS,
        MicroDuckVelocityEnv,
        resolve_microduck_scene,
    )
else:
    from ppo_microduck import (
        _low_cost_collision_scene,
        MAX_EPISODE_STEPS,
        MicroDuckVelocityEnv,
        resolve_microduck_scene,
    )

_has_mujoco = importlib.util.find_spec("mujoco") is not None

ACTION_SCALE = 0.35
MIN_HEIGHT_RATIO = 0.55
MIN_UPRIGHT = 0.35
CONTROL_DT = MicroDuckVelocityEnv.FRAME_SKIP * 0.002


@dataclass(frozen=True)
class MicroDuckGaitConfig:
    """Parameters of the closed-form MicroDuck gait.

    The four amplitudes are normalized actions, so an amplitude of one is an
    ``ACTION_SCALE``-radian offset around the MJCF ``STAND`` actuator target.
    The oscillator drives mirrored leg joints half a cycle apart. Pitch
    feedback is applied symmetrically to the two hip-pitch targets.

    Examples:
        >>> config = MicroDuckGaitConfig()
        >>> config.frequency_hz
        3.5477
    """

    frequency_hz: float = 3.5477
    hip_amplitude: float = -0.1837
    knee_amplitude: float = 0.0151
    ankle_amplitude: float = -0.0946
    lateral_amplitude: float = 0.0549
    phase_offset: float = 2.7925
    pitch_kp: float = -7.0
    pitch_kd: float = -0.56
    ramp_duration_s: float = 0.4


@dataclass(frozen=True)
class _TrialResult:
    seed: int
    episode_length: int
    survived: bool
    signed_displacement: float
    average_forward_speed: float
    tracking_reward: float


def _quaternion_pitch(quaternion: np.ndarray) -> float:
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    w, x, y, z = quaternion
    return math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))


def _quaternion_upright(quaternion: np.ndarray) -> float:
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    _, x, y, _ = quaternion
    return 1.0 - 2.0 * (x * x + y * y)


def _body_forward_vector(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    w, x, y, z = quaternion
    return np.asarray(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        )
    )


def microduck_heuristic_action(
    config: MicroDuckGaitConfig,
    qpos: np.ndarray,
    qvel: np.ndarray,
    elapsed_time_s: float,
) -> np.ndarray:
    """Compute one normalized action from the current physical state.

    Args:
        config: Oscillator and pitch-feedback parameters.
        qpos: MuJoCo generalized positions with shape ``(21,)``.
        qvel: MuJoCo generalized velocities with shape ``(20,)``.
        elapsed_time_s: Time since the beginning of the episode.

    Returns:
        A finite normalized action with shape ``(14,)`` in actuator order.

    Examples:
        >>> config = MicroDuckGaitConfig()
        >>> qpos = np.zeros(21); qpos[3] = 1.0
        >>> action = microduck_heuristic_action(config, qpos, np.zeros(20), 0.0)
        >>> action.shape
        (14,)
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.shape != (21,) or qvel.shape != (20,):
        raise ValueError(
            "MicroDuck heuristic state must have qpos shape (21,) and "
            f"qvel shape (20,), got {qpos.shape} and {qvel.shape}."
        )
    if not math.isfinite(elapsed_time_s) or elapsed_time_s < 0:
        raise ValueError("elapsed_time_s must be finite and non-negative.")

    pitch = _quaternion_pitch(qpos[3:7])
    pitch_correction = np.clip(
        config.pitch_kp * pitch + config.pitch_kd * qvel[4],
        -0.9,
        0.9,
    )
    phase = config.phase_offset + 2.0 * math.pi * config.frequency_hz * elapsed_time_s
    wave = math.sin(phase)
    if config.ramp_duration_s <= 0:
        ramp = 1.0
    else:
        ramp = min(elapsed_time_s / config.ramp_duration_s, 1.0)
    gait_wave = ramp * wave

    action = np.zeros(14, dtype=np.float64)
    # A common hip-pitch oscillation produces opposite physical leg motion
    # because the left and right joints use opposite sign conventions.
    action[2] = -pitch_correction - config.hip_amplitude * gait_wave
    action[11] = pitch_correction - config.hip_amplitude * gait_wave
    action[3] = config.knee_amplitude * ramp * max(wave, 0.0)
    action[12] = config.knee_amplitude * ramp * min(wave, 0.0)
    action[4] = config.ankle_amplitude * gait_wave
    action[13] = config.ankle_amplitude * gait_wave
    action[1] = config.lateral_amplitude * gait_wave
    action[10] = config.lateral_amplitude * gait_wave
    return np.clip(action, -1.0, 1.0)


@dataclass
class _MicroDuckHeuristicRenderPolicy:
    config: MicroDuckGaitConfig
    control_dt: float

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        observation = tensordict["observation"]
        pitch = tensordict.get("diagnostic_pitch", None)
        if pitch is None:
            pitch = observation[..., :1].clamp(-1.0, 1.0).asin()
        step_count = tensordict.get("step_count", None)
        if step_count is None:
            raise KeyError(
                "The MicroDuck heuristic RLRender policy requires `step_count`; "
                "pass --max-steps so rlrender adds a StepCounter transform."
            )

        pitch_rate = observation[..., 4:5]
        pitch_correction = (
            self.config.pitch_kp * pitch + self.config.pitch_kd * pitch_rate
        ).clamp(-0.9, 0.9)
        elapsed_time = step_count.to(observation.dtype) * self.control_dt
        phase = (
            self.config.phase_offset
            + 2.0 * math.pi * self.config.frequency_hz * elapsed_time
        )
        wave = phase.sin()
        if self.config.ramp_duration_s <= 0:
            ramp = torch.ones_like(wave)
        else:
            ramp = (elapsed_time / self.config.ramp_duration_s).clamp(max=1.0)
        gait_wave = ramp * wave

        action = observation.new_zeros(*observation.shape[:-1], 14)
        action[..., 2:3] = -pitch_correction - self.config.hip_amplitude * gait_wave
        action[..., 11:12] = pitch_correction - self.config.hip_amplitude * gait_wave
        action[..., 3:4] = self.config.knee_amplitude * ramp * wave.clamp_min(0.0)
        action[..., 12:13] = self.config.knee_amplitude * ramp * wave.clamp_max(0.0)
        action[..., 4:5] = self.config.ankle_amplitude * gait_wave
        action[..., 13:14] = self.config.ankle_amplitude * gait_wave
        action[..., 1:2] = self.config.lateral_amplitude * gait_wave
        action[..., 10:11] = self.config.lateral_amplitude * gait_wave
        tensordict.set("action", action.clamp(-1.0, 1.0))
        return tensordict


def make_render_policy(spec: RenderPolicySpec) -> _MicroDuckHeuristicRenderPolicy:
    """Build the closed-form policy for an ``rlrender`` rollout.

    The checkpoint may contain a ``gait`` mapping with
    :class:`MicroDuckGaitConfig` fields. ``policy_kwargs`` override those
    values and may additionally set ``control_dt``.

    Args:
        spec: RLRender policy construction context.

    Returns:
        A TensorDict policy that writes normalized MicroDuck actions.

    Examples:
        >>> from types import SimpleNamespace
        >>> spec = SimpleNamespace(checkpoint={}, policy_kwargs={})
        >>> policy = make_render_policy(spec)
        >>> policy.config.frequency_hz
        3.5477
    """
    gait: dict[str, Any] = {}
    checkpoint = spec.checkpoint
    if isinstance(checkpoint, Mapping) and checkpoint.get("gait") is not None:
        checkpoint_gait = checkpoint["gait"]
        if not isinstance(checkpoint_gait, Mapping):
            raise TypeError("The heuristic checkpoint `gait` entry must be a mapping.")
        gait.update(checkpoint_gait)
    gait.update(spec.policy_kwargs)
    control_dt = float(gait.pop("control_dt", CONTROL_DT))
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive.")
    return _MicroDuckHeuristicRenderPolicy(
        MicroDuckGaitConfig(**gait),
        control_dt,
    )


def _load_model(microduck_root: str | Path | None) -> Any:
    if not _has_mujoco:
        raise ImportError(
            "The heuristic MicroDuck example requires `mujoco`. Install the "
            "gym_continuous extra or run with `uv run --with mujoco`."
        )
    import mujoco

    scene_path = resolve_microduck_scene(microduck_root)
    with _low_cost_collision_scene(scene_path) as physics_scene:
        return mujoco.MjModel.from_xml_path(str(physics_scene))


def _stand_metadata(model: Any) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    import mujoco

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if key_id < 0:
        raise ValueError("MicroDuck MJCF must define a `STAND` keyframe.")
    joint_ids = model.actuator_trnid[:, 0]
    return (
        key_id,
        model.key_ctrl[key_id].copy(),
        model.jnt_range[joint_ids, 0].copy(),
        model.jnt_range[joint_ids, 1].copy(),
    )


def _run_trial(
    model: Any,
    config: MicroDuckGaitConfig,
    *,
    seed: int,
    steps: int,
    reset_noise_scale: float,
    commanded_x_velocity: float,
) -> _TrialResult:
    import mujoco

    key_id, home_ctrl, joint_low, joint_high = _stand_metadata(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    rng = np.random.default_rng(seed)
    data.qpos[:2] += rng.uniform(-reset_noise_scale, reset_noise_scale, 2)
    data.qpos[7:] += rng.uniform(-reset_noise_scale, reset_noise_scale, 14)
    data.qvel[:] += rng.uniform(-reset_noise_scale, reset_noise_scale, 20)
    mujoco.mj_forward(model, data)

    control_dt = MicroDuckVelocityEnv.FRAME_SKIP * model.opt.timestep
    initial_position = data.qpos[:3].copy()
    initial_forward = _body_forward_vector(data.qpos[3:7])
    tracking_reward = 0.0
    forward_distance = 0.0
    survived = True
    for step in range(steps):
        action = microduck_heuristic_action(
            config,
            data.qpos,
            data.qvel,
            step * control_dt,
        )
        data.ctrl[:] = np.clip(
            home_ctrl + ACTION_SCALE * action,
            joint_low,
            joint_high,
        )
        for _ in range(MicroDuckVelocityEnv.FRAME_SKIP):
            mujoco.mj_step(model, data)

        body_forward = _body_forward_vector(data.qpos[3:7])
        forward_speed = float(np.dot(data.qvel[:3], body_forward))
        forward_distance += control_dt * forward_speed
        tracking_reward += math.exp(
            -(((forward_speed - commanded_x_velocity) / 0.25) ** 2)
        )
        upright = _quaternion_upright(data.qpos[3:7])
        finite = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        if (
            data.qpos[2] < MIN_HEIGHT_RATIO * model.key_qpos[key_id, 2]
            or upright < MIN_UPRIGHT
            or not finite
        ):
            survived = False
            break

    episode_length = step + 1
    signed_displacement = float(
        np.dot(data.qpos[:3] - initial_position, initial_forward)
    )
    return _TrialResult(
        seed=seed,
        episode_length=episode_length,
        survived=survived,
        signed_displacement=signed_displacement,
        average_forward_speed=forward_distance / (episode_length * control_dt),
        tracking_reward=tracking_reward / episode_length,
    )


def _evaluate(
    model: Any,
    config: MicroDuckGaitConfig,
    *,
    seeds: Sequence[int],
    steps: int,
    reset_noise_scale: float,
    commanded_x_velocity: float,
) -> list[_TrialResult]:
    return [
        _run_trial(
            model,
            config,
            seed=seed,
            steps=steps,
            reset_noise_scale=reset_noise_scale,
            commanded_x_velocity=commanded_x_velocity,
        )
        for seed in seeds
    ]


def _summary(trials: Sequence[_TrialResult]) -> dict[str, float]:
    lengths = np.asarray([trial.episode_length for trial in trials])
    displacements = np.asarray([trial.signed_displacement for trial in trials])
    speeds = np.asarray([trial.average_forward_speed for trial in trials])
    return {
        "survival_rate": float(np.mean([trial.survived for trial in trials])),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_min": float(lengths.min()),
        "signed_displacement_mean": float(displacements.mean()),
        "signed_displacement_min": float(displacements.min()),
        "forward_speed_mean": float(speeds.mean()),
        "forward_speed_min": float(speeds.min()),
        "wrong_way_rollouts": float(np.count_nonzero(displacements <= 0.0)),
        "tracking_reward_mean": float(
            np.mean([trial.tracking_reward for trial in trials])
        ),
    }


def _search_key(trials: Sequence[_TrialResult]) -> tuple[float, ...]:
    summary = _summary(trials)
    return (
        summary["episode_length_min"],
        summary["survival_rate"],
        summary["episode_length_mean"],
        summary["forward_speed_mean"],
    )


def _search_configs(
    initial: MicroDuckGaitConfig,
    num_candidates: int,
    rng: np.random.Generator,
) -> list[MicroDuckGaitConfig]:
    configs = [initial]
    for _ in range(num_candidates - 1):
        values = rng.normal(size=6)
        configs.append(
            replace(
                initial,
                frequency_hz=float(
                    np.clip(initial.frequency_hz + 0.35 * values[0], 2.5, 4.5)
                ),
                hip_amplitude=float(initial.hip_amplitude + 0.05 * values[1]),
                knee_amplitude=float(
                    np.clip(initial.knee_amplitude + 0.025 * values[2], 0.0, 0.12)
                ),
                ankle_amplitude=float(initial.ankle_amplitude + 0.035 * values[3]),
                lateral_amplitude=float(initial.lateral_amplitude + 0.06 * values[4]),
                phase_offset=float(
                    (initial.phase_offset + 0.4 * values[5] + math.pi) % (2.0 * math.pi)
                    - math.pi
                ),
            )
        )
    return configs


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck-root", type=Path)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--reset-noise-scale", type=float, default=0.02)
    parser.add_argument("--commanded-x-velocity", type=float, default=0.3)
    parser.add_argument("--search-candidates", type=int, default=0)
    parser.add_argument("--search-num-seeds", type=int, default=8)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--render-checkpoint", type=Path)
    return parser.parse_args(args)


def _main(args: argparse.Namespace) -> None:
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

    model = _load_model(args.microduck_root)
    config = MicroDuckGaitConfig()
    if args.search_candidates:
        search_seeds = range(args.search_num_seeds)
        candidates = _search_configs(
            config,
            args.search_candidates,
            np.random.default_rng(args.search_seed),
        )
        ranked = []
        for candidate in candidates:
            trials = _evaluate(
                model,
                candidate,
                seeds=search_seeds,
                steps=args.steps,
                reset_noise_scale=args.reset_noise_scale,
                commanded_x_velocity=args.commanded_x_velocity,
            )
            ranked.append((_search_key(trials), candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        config = ranked[0][1]
        torchrl_logger.info("Best searched config: %s", json.dumps(asdict(config)))

    if args.render_checkpoint is not None:
        args.render_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"gait": asdict(config)}, args.render_checkpoint)
        torchrl_logger.info(
            "Saved RLRender gait checkpoint to %s", args.render_checkpoint
        )

    trials = _evaluate(
        model,
        config,
        seeds=range(args.num_seeds),
        steps=args.steps,
        reset_noise_scale=args.reset_noise_scale,
        commanded_x_velocity=args.commanded_x_velocity,
    )
    torchrl_logger.info("Gait config: %s", json.dumps(asdict(config)))
    torchrl_logger.info("Evaluation: %s", json.dumps(_summary(trials), sort_keys=True))


if __name__ == "__main__":
    _main(_parse_args())
