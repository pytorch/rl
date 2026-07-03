# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

from torchrl.envs import EnvBase, GymWrapper
from torchrl.envs.transforms import StepCounter
from torchrl.render.config import RenderConfig, RenderEnvSpec
from torchrl.render.import_utils import call_with_supported_kwargs, import_from_string

__all__ = ["add_step_counter", "make_render_env", "normalize_env", "seed_env"]


def make_render_env(config: RenderConfig) -> Any:
    """Builds and prepares an environment for rendering.

    Args:
        config: Render configuration.

    Returns:
        A TorchRL environment when wrapping is possible, otherwise the factory result.
    """
    factory = (
        import_from_string(config.env) if isinstance(config.env, str) else config.env
    )
    if not callable(factory):
        raise TypeError(
            f"Environment factory must be callable, got {type(factory).__name__}."
        )
    spec = RenderEnvSpec.from_config(config)
    kwargs = {
        "spec": spec,
        "device": spec.device,
        "seed": spec.seed,
        "max_steps": spec.max_steps,
        "from_pixels": spec.from_pixels,
        "pixels_only": spec.pixels_only,
        "camera": spec.camera,
        "render_mode": spec.render_mode,
        "env_kwargs": dict(spec.env_kwargs),
        "config": config,
        **spec.env_kwargs,
    }
    env = call_with_supported_kwargs(factory, spec, kwargs)
    env = normalize_env(env, config)
    seed_env(env, config.seed)
    if config.max_steps is not None:
        env = add_step_counter(env, config.max_steps)
    return env


def normalize_env(env: Any, config: RenderConfig) -> Any:
    """Normalizes external environments into TorchRL wrappers when feasible."""
    if isinstance(env, EnvBase):
        return env
    if config.env_backend in ("auto", "gym", "gymnasium"):
        try:
            return GymWrapper(
                env, from_pixels=config.from_pixels, pixels_only=config.pixels_only
            )
        except Exception:
            return env
    return env


def add_step_counter(env: Any, max_steps: int) -> Any:
    """Adds a :class:`~torchrl.envs.transforms.StepCounter` when supported."""
    if not isinstance(env, EnvBase):
        return env
    if _has_step_counter(env):
        return env
    return env.append_transform(StepCounter(max_steps=max_steps))


def seed_env(env: Any, seed: int | None) -> None:
    """Seeds an environment if it exposes a known seed method."""
    if seed is None:
        return
    set_seed = getattr(env, "set_seed", None)
    if callable(set_seed):
        set_seed(seed)
        return
    seed_method = getattr(env, "seed", None)
    if callable(seed_method):
        seed_method(seed)


def _has_step_counter(env: EnvBase) -> bool:
    transform = getattr(env, "transform", None)
    if transform is None:
        return False
    if isinstance(transform, StepCounter):
        return True
    try:
        return any(isinstance(item, StepCounter) for item in transform)
    except TypeError:
        return False
