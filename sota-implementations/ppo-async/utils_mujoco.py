# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for async PPO MuJoCo experiments.

Contains environment factories, model construction, and postproc callables
used by the training loops.
"""
from __future__ import annotations

import logging

import torch
import torch.nn

from mujoco_torch.zoo import ENVS
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import (
    ClipTransform,
    ExplorationType,
    ObservationNorm,
    RandomTruncationTransform,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

log = logging.getLogger(__name__)


# ── Environment factories ──────────────────────────────────────────────

# Number of random rollout steps used to compute ObservationNorm stats.
_OBS_NORM_INIT_STEPS = 1000


def compute_obs_norm_stats(env_name):
    """Compute ObservationNorm loc/scale from a small proof env.

    Returns (loc, scale) tensors on CPU.  These are passed into make_env
    and make_eval_env so that the (potentially compiled, high-batch-size)
    production envs never need to run init_stats themselves.
    """
    proof = ENVS[env_name](
        num_envs=1, device="cpu", dtype=torch.float32, compile_step=False
    )
    proof = TransformedEnv(proof)
    obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
    proof.append_transform(obs_norm)
    obs_norm.init_stats(_OBS_NORM_INIT_STEPS)
    loc, scale = obs_norm.loc.clone(), obs_norm.scale.clone()
    del proof
    return loc, scale


def make_env(
    env_name="halfcheetah",
    device="cpu",
    num_envs=4096,
    compile=True,
    obs_norm_loc=None,
    obs_norm_scale=None,
):
    """Create a batched MuJoCo env using mujoco-torch.

    Returns a single env with batch_size=[num_envs], where all envs run
    in parallel via torch.vmap (GPU) or sequential (CPU).

    When obs_norm_loc/scale are provided, ObservationNorm uses those fixed
    stats (avoids running init_stats on the production env).
    """
    compile_kwargs = {"mode": "default"} if compile else None
    env = ENVS[env_name](
        num_envs=num_envs,
        device=device,
        dtype=torch.float32,
        compile_step=compile,
        compile_kwargs=compile_kwargs,
    )
    env = TransformedEnv(env)
    if obs_norm_loc is not None and obs_norm_scale is not None:
        env.append_transform(
            ObservationNorm(
                loc=obs_norm_loc,
                scale=obs_norm_scale,
                in_keys=["observation"],
                standard_normal=True,
            )
        )
    else:
        env.append_transform(
            ObservationNorm(in_keys=["observation"], standard_normal=True)
        )
        env.transform[-1].init_stats(_OBS_NORM_INIT_STEPS)
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(
        RandomTruncationTransform(prob=0.05, max_horizon=1000, first_episode_prob=1.0)
    )
    return env


def make_eval_env(
    env_name,
    device,
    num_eval_envs,
    max_steps=1000,
    obs_norm_loc=None,
    obs_norm_scale=None,
):
    """Env factory for the Evaluator.

    Creates a compiled GPU-batched mujoco-torch env. The StepCounter
    uses max_steps so that episodes terminate — the Evaluator skips adding
    its own max_frames_per_traj when it sees an existing step_count.
    """
    env = ENVS[env_name](
        num_envs=num_eval_envs,
        device=device,
        dtype=torch.float32,
        compile_step=True,
        compile_kwargs={"mode": "default"},
    )
    env = TransformedEnv(env)
    if obs_norm_loc is not None and obs_norm_scale is not None:
        env.append_transform(
            ObservationNorm(
                loc=obs_norm_loc,
                scale=obs_norm_scale,
                in_keys=["observation"],
                standard_normal=True,
            )
        )
    else:
        env.append_transform(
            ObservationNorm(in_keys=["observation"], standard_normal=True)
        )
        env.transform[-1].init_stats(_OBS_NORM_INIT_STEPS)
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=max_steps))
    return env


# ── Model factories ────────────────────────────────────────────────────


def make_ppo_models(env_name, device, num_envs=1):
    """Build PPO actor and critic networks.

    Uses a small mujoco-torch env as a proof environment to read
    observation/action specs. The proof env is discarded after construction.
    """
    proof_environment = make_env(
        env_name, device="cpu", num_envs=num_envs, compile=False
    )
    actor, critic = _make_ppo_models_state(proof_environment, device=device)
    del proof_environment
    return actor, critic


def _make_ppo_models_state(proof_environment, device):
    input_shape = proof_environment.observation_spec["observation"].shape
    num_outputs = proof_environment.action_spec_unbatched.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low.to(device),
        "high": proof_environment.action_spec_unbatched.space.high.to(device),
        "tanh_loc": False,
    }

    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,
        num_cells=[256, 256, 256, 256],
        device=device,
    )
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            proof_environment.action_spec_unbatched.shape[-1], scale_lb=1e-8
        ).to(device),
    )

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

    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[256, 256, 256, 256],
        device=device,
    )
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    value_module = ValueOperator(value_mlp, in_keys=["observation"])
    return policy_module, value_module


# ── Postproc callables (module-level for pickle compatibility) ─────────


class ActorWithCritic(torch.nn.Module):
    """Wrapper that holds both actor and critic but only runs the actor.

    Ensures ``update_policy_weights_()`` syncs both modules to workers,
    while keeping per-step collection lightweight (actor only). The critic is
    used by the postproc for batched GAE computation.
    """

    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, td):
        return self.actor(td)


class WorkerGAEPostproc:
    """Postproc for worker GAE mode: critic + GAE, flatten, stamp policy_version.

    The adv_module's value_network (critic) is synced to the collector device
    automatically via a dedicated SharedMemWeightSyncScheme registered with
    model_id="postproc.adv_module". No manual .to() calls needed.
    """

    def __init__(self, adv_module, version_counter):
        self.adv_module = adv_module
        self.version_counter = version_counter

    def __call__(self, data):
        with torch.no_grad():
            data = self.adv_module(data)
        data_flat = data.reshape(-1)
        data_flat["policy_version"] = torch.full(
            (data_flat.shape[0],),
            float(self.version_counter.value),
            device=data_flat.device,
        )
        return data_flat


class LearnerPostproc:
    """Postproc for start+learner mode: flatten, stamp policy_version."""

    def __init__(self, version_counter):
        self.version_counter = version_counter

    def __call__(self, data):
        data_flat = data.reshape(-1)
        data_flat["policy_version"] = torch.full(
            (data_flat.shape[0],),
            float(self.version_counter.value),
            device=data_flat.device,
        )
        return data_flat
