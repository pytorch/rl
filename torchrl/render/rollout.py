# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDictBase

from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.render.backends import (
    EnvRenderBackend,
    NullRenderBackend,
    TensorDictPixelsBackend,
)
from torchrl.render.config import FrameBundle, RenderConfig, RenderResult

__all__ = ["collect_render_rollouts"]


def collect_render_rollouts(
    env: Any, policy: Any, config: RenderConfig
) -> RenderResult:
    """Collects sequential render rollouts.

    Args:
        env: Environment returned by :func:`torchrl.render.make_render_env`.
        policy: TensorDict-compatible policy.
        config: Render configuration.

    Returns:
        A render result containing trajectories and in-memory frames.
    """
    max_steps = _resolve_max_steps(env, config)
    trajectories: list[TensorDictBase] = []
    all_frames: list[list[FrameBundle]] = []
    warnings: list[str] = []
    frame_backends = _make_backends(config)
    exploration = _exploration_type(config)
    for traj_index in range(config.num_trajs):
        td = _reset_env(env)
        trajectory_steps: list[TensorDictBase] = []
        trajectory_frames: list[FrameBundle] = []
        for step in range(max_steps):
            with torch.no_grad(), set_exploration_type(exploration):
                action_td = policy(td)
            next_td = _step_env(env, action_td)
            trajectory_steps.append(_trajectory_step(next_td, config))
            frame = _capture_frame(
                frame_backends,
                env,
                next_td,
                config,
                step=step,
                trajectory_index=traj_index,
            )
            if frame is not None:
                trajectory_frames.append(frame)
            if _is_done(next_td, config, env):
                break
            td = _step_mdp(next_td, env)
        if not trajectory_steps:
            raise RuntimeError("Environment produced an empty rollout.")
        trajectories.append(torch.stack(trajectory_steps, 0).contiguous())
        all_frames.append(trajectory_frames)
        if not trajectory_frames:
            warnings.append(
                f"No RGB frames were captured for trajectory {traj_index}. For Gymnasium "
                "environments, create the env with render_mode='rgb_array' or set --from-pixels."
            )
    metadata = _rollout_metadata(trajectories, all_frames, config, warnings)
    return RenderResult(
        artifact_path=None,
        trajectories=trajectories,
        frame_paths=[],
        metadata=metadata,
        warnings=warnings,
        frames=all_frames,
    )


def _make_backends(config: RenderConfig):
    if config.render_backend == "pixels":
        return [TensorDictPixelsBackend()]
    if config.render_backend == "env":
        return [EnvRenderBackend()]
    if config.render_backend == "null":
        return [NullRenderBackend()]
    return [TensorDictPixelsBackend(), EnvRenderBackend(), NullRenderBackend()]


def _capture_frame(
    backends,
    env: Any,
    tensordict: TensorDictBase,
    config: RenderConfig,
    *,
    step: int,
    trajectory_index: int,
) -> FrameBundle | None:
    for backend in backends:
        if not backend.supports(env, config):
            continue
        frame = backend.capture(
            env,
            tensordict,
            config,
            step=step,
            trajectory_index=trajectory_index,
        )
        if frame is not None:
            return frame
    return None


def _reset_env(env: Any) -> TensorDictBase:
    reset = getattr(env, "reset", None)
    if not callable(reset):
        raise TypeError("rlrender requires an environment with a reset() method.")
    td = reset()
    if not isinstance(td, TensorDictBase):
        raise TypeError(
            "rlrender MVP expects reset() to return a TensorDict. Wrap Gym/Gymnasium "
            "environments with TorchRL or return an EnvBase from the env factory."
        )
    return td


def _step_env(env: Any, tensordict: TensorDictBase) -> TensorDictBase:
    step = getattr(env, "step", None)
    if not callable(step):
        raise TypeError("rlrender requires an environment with a step() method.")
    next_td = step(tensordict)
    if not isinstance(next_td, TensorDictBase):
        raise TypeError("rlrender MVP expects step() to return a TensorDict.")
    return next_td


def _trajectory_step(
    tensordict: TensorDictBase, config: RenderConfig
) -> TensorDictBase:
    try:
        next_td = tensordict.get("next")
    except Exception:
        next_td = None
    if not isinstance(next_td, TensorDictBase):
        return tensordict.clone()
    step = next_td.clone()
    try:
        action = tensordict.get(config.action_key)
    except Exception:
        return step
    step.set(config.action_key, action.clone() if torch.is_tensor(action) else action)
    return step


def _step_mdp(tensordict: TensorDictBase, env: Any) -> TensorDictBase:
    if isinstance(env, EnvBase):
        return step_mdp(tensordict, exclude_reward=False, exclude_done=False)
    next_td = tensordict.get("next", None)
    if isinstance(next_td, TensorDictBase):
        return next_td.clone()
    return tensordict.clone()


def _is_done(tensordict: TensorDictBase, config: RenderConfig, env: Any) -> bool:
    keys = []
    done_keys = getattr(env, "done_keys", None)
    if done_keys:
        keys.extend(done_keys)
    keys.append(config.done_key)
    keys.append("done")
    for key in keys:
        for candidate in _done_candidates(key):
            try:
                value = tensordict.get(candidate)
            except Exception:
                continue
            if torch.as_tensor(value).bool().any().item():
                return True
    return False


def _done_candidates(key: Any) -> list[Any]:
    if isinstance(key, tuple):
        return [("next", *key), key]
    return [("next", key), key]


def _resolve_max_steps(env: Any, config: RenderConfig) -> int:
    if config.max_steps is not None:
        return config.max_steps
    candidates = [
        getattr(env, "max_steps", None),
        getattr(env, "max_episode_steps", None),
        getattr(getattr(env, "spec", None), "max_episode_steps", None),
        getattr(
            getattr(getattr(env, "base_env", None), "spec", None),
            "max_episode_steps",
            None,
        ),
    ]
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    raise ValueError(
        "rlrender could not infer a rollout horizon. Pass --max-steps or configure "
        "an environment with an explicit time limit."
    )


def _exploration_type(config: RenderConfig):
    mode = config.exploration_mode
    if mode is None:
        mode = "deterministic" if config.deterministic else "random"
    mapping = {
        "deterministic": ExplorationType.DETERMINISTIC,
        "mode": ExplorationType.MODE,
        "mean": ExplorationType.MEAN,
        "random": ExplorationType.RANDOM,
    }
    return mapping[mode]


def _rollout_metadata(
    trajectories: list[TensorDictBase],
    frames: list[list[FrameBundle]],
    config: RenderConfig,
    warnings: list[str],
) -> dict[str, Any]:
    traj_meta = []
    for index, trajectory in enumerate(trajectories):
        reward = _trajectory_return(trajectory, config)
        traj_meta.append(
            {
                "index": index,
                "num_steps": int(trajectory.shape[0]),
                "return": reward,
                "num_frames": len(frames[index]),
                "cameras": sorted(
                    {name for bundle in frames[index] for name in bundle.frames}
                ),
            }
        )
    return {
        "format": config.format,
        "num_trajs": len(trajectories),
        "max_steps": config.max_steps,
        "fps": config.fps,
        "render_backend": config.render_backend,
        "env_backend": config.env_backend,
        "trajectories": traj_meta,
        "warnings": list(warnings),
    }


def _trajectory_return(
    trajectory: TensorDictBase, config: RenderConfig
) -> float | None:
    for key in _reward_candidates(config.reward_key):
        try:
            reward = trajectory.get(key)
        except Exception:
            continue
        if torch.is_tensor(reward):
            return float(reward.detach().cpu().sum().item())
    return None


def _reward_candidates(key: Any) -> list[Any]:
    if isinstance(key, tuple):
        return [("next", *key), key]
    return [("next", key), key]
