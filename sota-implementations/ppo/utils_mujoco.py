# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import warnings
from typing import Any, Literal

import torch
import torch.nn
from tensordict import TensorDictBase
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule

from tensordict.utils import NestedKey
from torchrl.data import Unbounded
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    ObservationNorm,
    RewardSum,
    StepCounter,
    Transform,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder

_has_mujoco_playground = importlib.util.find_spec("mujoco_playground") is not None

_EnvBackend = Literal["gym", "mujoco_playground"]
_MujocoPlaygroundEnv: type | None = None


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


_MUJOCO_QPOS_DIMS = {
    "InvertedPendulum-v4": 2,
    "InvertedPendulum-v5": 2,
    "InvertedDoublePendulum-v4": 3,
    "InvertedDoublePendulum-v5": 3,
}


def make_env(
    env_name="HalfCheetah-v4",
    device="cpu",
    from_pixels: bool = False,
    *,
    backend: _EnvBackend = "gym",
    config_overrides: dict[str, Any] | None = None,
    normalize_observation: bool = True,
    vecnorm_stats: dict[str, torch.Tensor] | None = None,
    max_episode_steps: int | None = None,
    qpos_key: NestedKey | None = None,
):
    if backend == "gym":
        env = GymEnv(
            env_name, device=device, from_pixels=from_pixels, pixels_only=False
        )
    elif backend == "mujoco_playground":
        if not _has_mujoco_playground:
            raise ImportError(
                "mujoco_playground is required for env.backend=mujoco_playground. "
                "Run with `uv run --extra mujoco_playground ...`."
            )
        if from_pixels:
            raise ValueError(
                "from_pixels=True is not supported for MuJoCo Playground PPO envs."
            )
        MujocoPlaygroundEnv = _get_mujoco_playground_env()

        env = MujocoPlaygroundEnv(
            env_name,
            device=device,
            config_overrides=config_overrides,
        )
    else:
        raise ValueError(
            f"Unsupported PPO MuJoCo backend {backend!r}. "
            "Expected 'gym' or 'mujoco_playground'."
        )
    env = TransformedEnv(env)
    # The qpos transform must run before observation normalization so saved
    # qpos values are raw joint positions rather than whitened observations.
    if qpos_key is not None:
        env.append_transform(
            _MujocoQposTransform(
                qpos_dim=_qpos_dim(env_name),
                out_key=qpos_key,
            )
        )
    if normalize_observation:
        if vecnorm_stats is not None:
            env.append_transform(
                ObservationNorm(
                    loc=vecnorm_stats["loc"].to(device),
                    scale=vecnorm_stats["scale"].to(device),
                    in_keys=["observation"],
                    out_keys=["observation"],
                    standard_normal=True,
                )
            )
        else:
            env.append_transform(
                VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2)
            )
        env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=max_episode_steps))
    if _observation_dtype(env) is torch.float64:
        env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


def get_vecnorm_state(env) -> dict[str, torch.Tensor] | None:
    """Extracts frozen VecNorm observation statistics for rlrender checkpoints.

    Returns ``None`` when the environment holds no reachable VecNorm transform
    (e.g. observation normalization is disabled).
    """
    transform = getattr(env, "transform", None)
    if transform is None:
        return None
    try:
        transforms = list(transform)
    except TypeError:
        transforms = [transform]
    for item in transforms:
        if isinstance(item, VecNorm):
            loc = item.loc.get("observation", None)
            scale = item.scale.get("observation", None)
            if loc is None or scale is None:
                return None
            return {
                "loc": loc.detach().cpu().clone(),
                "scale": scale.detach().cpu().clone(),
            }
    return None


def _observation_dtype(env) -> torch.dtype | None:
    try:
        return env.observation_spec["observation"].dtype
    except Exception:
        return None


def _get_mujoco_playground_env() -> type:
    global _MujocoPlaygroundEnv
    if _MujocoPlaygroundEnv is None:
        from torchrl.envs.libs.mujoco_playground import MujocoPlaygroundEnv

        _MujocoPlaygroundEnv = MujocoPlaygroundEnv
    return _MujocoPlaygroundEnv


def make_render_env(spec: Any):
    """Builds a MuJoCo environment suitable for ``rlrender``.

    Environment defaults (name, backend, observation normalization, frozen
    VecNorm statistics) are read from the checkpoint metadata written by
    ``ppo_mujoco.py`` and can be overridden through ``--env-kwargs``. qpos
    extraction for MuJoCo-WASM playback is enabled automatically for the env
    names listed in ``_MUJOCO_QPOS_DIMS``.
    """
    checkpoint = (
        spec.checkpoint if isinstance(getattr(spec, "checkpoint", None), dict) else {}
    )
    env_name = spec.env_kwargs.get(
        "env_name", checkpoint.get("env_name", "InvertedPendulum-v4")
    )
    backend = spec.env_kwargs.get("backend", checkpoint.get("env_backend", "gym"))
    config_overrides = spec.env_kwargs.get(
        "config_overrides", checkpoint.get("env_config_overrides")
    )
    normalize_observation = bool(
        spec.env_kwargs.get(
            "normalize_observation", checkpoint.get("normalize_observation", False)
        )
    )
    vecnorm_stats = checkpoint.get("vecnorm")
    if normalize_observation and vecnorm_stats is None:
        warnings.warn(
            "Rendering with normalize_observation=True but the checkpoint does not "
            "contain VecNorm statistics under 'vecnorm'. Observations will be "
            "normalized with running statistics initialized for rendering and will not "
            "match training. Save checkpoints with VecNorm metadata or "
            "pass normalize_observation=False in --env-kwargs.",
            stacklevel=2,
        )
    default_qpos_key = (
        "qpos" if backend == "gym" and env_name in _MUJOCO_QPOS_DIMS else None
    )
    qpos_key = spec.env_kwargs.get("qpos_key", default_qpos_key)
    max_episode_steps = spec.env_kwargs.get(
        "max_episode_steps", checkpoint.get("max_episode_steps")
    )
    return make_env(
        env_name,
        device=spec.device,
        from_pixels=spec.from_pixels,
        backend=backend,
        config_overrides=config_overrides,
        normalize_observation=normalize_observation,
        vecnorm_stats=vecnorm_stats,
        max_episode_steps=max_episode_steps,
        qpos_key=qpos_key,
    )


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_models_state(proof_environment, device):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec_unbatched.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low.to(device),
        "high": proof_environment.action_spec_unbatched.space.high.to(device),
        "tanh_loc": False,
    }

    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=[64, 64],
        device=device,
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            proof_environment.action_spec_unbatched.shape[-1], scale_lb=1e-8
        ).to(device),
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
        device=device,
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module


def make_ppo_models(
    env_name,
    device,
    *,
    backend: _EnvBackend = "gym",
    config_overrides: dict[str, Any] | None = None,
    normalize_observation: bool = True,
    max_episode_steps: int | None = None,
):
    proof_environment = make_env(
        env_name,
        device=device,
        backend=backend,
        config_overrides=config_overrides,
        normalize_observation=normalize_observation,
        max_episode_steps=max_episode_steps,
    )
    actor, critic = make_ppo_models_state(proof_environment, device=device)
    return actor, critic


def make_render_policy(spec: Any):
    """Builds the PPO policy module for ``rlrender`` checkpoint loading."""
    checkpoint = spec.checkpoint if isinstance(spec.checkpoint, dict) else {}
    env_name = spec.policy_kwargs.get(
        "env_name",
        checkpoint.get("env_name", "InvertedPendulum-v4"),
    )
    normalize_observation = bool(
        spec.policy_kwargs.get(
            "normalize_observation", checkpoint.get("normalize_observation", False)
        )
    )
    backend = spec.policy_kwargs.get("backend", checkpoint.get("env_backend", "gym"))
    config_overrides = spec.policy_kwargs.get(
        "config_overrides", checkpoint.get("env_config_overrides")
    )
    max_episode_steps = spec.policy_kwargs.get(
        "max_episode_steps", checkpoint.get("max_episode_steps")
    )
    actor, _ = make_ppo_models(
        env_name,
        device=spec.device,
        backend=backend,
        config_overrides=config_overrides,
        normalize_observation=normalize_observation,
        max_episode_steps=max_episode_steps,
    )
    return actor


class _MujocoQposTransform(Transform):
    def __init__(
        self,
        *,
        qpos_dim: int,
        in_key: NestedKey = "observation",
        out_key: NestedKey = "qpos",
    ) -> None:
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.qpos_dim = qpos_dim

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        data = self.parent.base_env._env.unwrapped.data
        return torch.as_tensor(
            data.qpos.copy(),
            dtype=observation.dtype,
            device=observation.device,
        )

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        observation_spec = observation_spec.clone()
        source_spec = observation_spec[self.in_keys[0]]
        observation_spec[self.out_keys[0]] = Unbounded(
            shape=(*source_spec.shape[:-1], self.qpos_dim),
            dtype=source_spec.dtype,
            device=source_spec.device,
        )
        return observation_spec


def _qpos_dim(env_name: str) -> int:
    try:
        return _MUJOCO_QPOS_DIMS[env_name]
    except KeyError as err:
        raise ValueError(
            f"MuJoCo-WASM qpos extraction is not configured for {env_name!r}. "
            "Pass a supported env_name or extend _MUJOCO_QPOS_DIMS."
        ) from err


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def eval_model(actor, test_env, num_episodes=3, max_steps=10_000_000):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=max_steps,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        if reward.numel() == 0:
            reward = td_test["next", "reward"].sum().view(1)
        test_rewards.append(reward.cpu())
        test_env.apply(dump_video)
    del td_test
    return torch.cat(test_rewards, 0).mean()
