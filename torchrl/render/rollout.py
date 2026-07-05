# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable

from typing import Any

import torch
from tensordict import TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
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

    :class:`~torchrl.envs.EnvBase` environments are rolled out through
    :meth:`~torchrl.envs.EnvBase.rollout`; other environments fall back to a
    duck-typed reset/step loop. Captured frames span the initial state through
    the terminal state of each trajectory.

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
    collect = (
        _collect_env_base_trajectory
        if isinstance(env, EnvBase)
        else _collect_duck_typed_trajectory
    )
    for traj_index in range(config.num_trajs):
        trajectory, trajectory_frames = collect(
            env,
            policy,
            config,
            max_steps=max_steps,
            backends=frame_backends,
            exploration=exploration,
            trajectory_index=traj_index,
        )
        trajectories.append(trajectory)
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


def _collect_env_base_trajectory(
    env: EnvBase,
    policy: Any,
    config: RenderConfig,
    *,
    max_steps: int,
    backends: list[Any],
    exploration: Any,
    trajectory_index: int,
) -> tuple[TensorDictBase, list[FrameBundle]]:
    frames: list[FrameBundle] = []
    capture = _frame_recorder(backends, env, config, frames, trajectory_index)
    reset_td = env.reset()
    capture(reset_td)
    with torch.no_grad(), set_exploration_type(exploration):
        rollout = env.rollout(
            max_steps,
            policy,
            tensordict=reset_td,
            auto_reset=False,
            break_when_any_done=True,
            return_contiguous=True,
            callback=lambda _env, tensordict: capture(tensordict),
        )
    if rollout.numel() == 0:
        raise RuntimeError("Environment produced an empty rollout.")
    # rollout() never invokes the callback on the final step, so the terminal
    # state is captured from the last step's "next" entry here.
    capture(rollout[..., -1])
    return _trajectory_from_rollout(rollout, config), frames


def _collect_duck_typed_trajectory(
    env: Any,
    policy: Any,
    config: RenderConfig,
    *,
    max_steps: int,
    backends: list[Any],
    exploration: Any,
    trajectory_index: int,
) -> tuple[TensorDictBase, list[FrameBundle]]:
    frames: list[FrameBundle] = []
    capture = _frame_recorder(backends, env, config, frames, trajectory_index)
    td = _reset_env(env)
    capture(td)
    trajectory_steps: list[TensorDictBase] = []
    for _ in range(max_steps):
        with torch.no_grad(), set_exploration_type(exploration):
            action_td = policy(td)
        next_td = _step_env(env, action_td)
        trajectory_steps.append(_trajectory_step(next_td, config))
        capture(next_td)
        if _is_done(next_td, config, env):
            break
        td = _step_mdp(next_td)
    if not trajectory_steps:
        raise RuntimeError("Environment produced an empty rollout.")
    return torch.stack(trajectory_steps, 0), frames


def _frame_recorder(
    backends: list[Any],
    env: Any,
    config: RenderConfig,
    frames: list[FrameBundle],
    trajectory_index: int,
) -> Callable[[TensorDictBase], None]:
    def capture(tensordict: TensorDictBase) -> None:
        frame = _capture_frame(
            backends,
            env,
            tensordict,
            config,
            step=len(frames),
            trajectory_index=trajectory_index,
        )
        if frame is not None:
            frames.append(frame)

    return capture


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
    for backend in list(backends):
        if not backend.supports(env, config):
            continue
        try:
            frame = backend.capture(
                env,
                tensordict,
                config,
                step=step,
                trajectory_index=trajectory_index,
            )
        except Exception as err:
            if config.render_backend != "auto":
                raise
            backends.remove(backend)
            torchrl_logger.warning(
                f"rlrender backend {backend.name!r} failed to capture a frame and "
                f"was disabled for this run: {err}"
            )
            continue
        if frame is not None:
            return frame
    return None


def _trajectory_from_rollout(
    rollout: TensorDictBase, config: RenderConfig
) -> TensorDictBase:
    trajectory = rollout.get("next").clone()
    try:
        action = rollout.get(config.action_key)
    except Exception:
        return trajectory
    trajectory.set(
        config.action_key, action.clone() if torch.is_tensor(action) else action
    )
    return trajectory


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


def _step_mdp(tensordict: TensorDictBase) -> TensorDictBase:
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
    keys.extend(["done", "terminated", "truncated"])
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
    return ExplorationType.from_str(mode)


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
                "num_steps": int(trajectory.shape[-1]),
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
