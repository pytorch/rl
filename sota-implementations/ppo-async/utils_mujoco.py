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
from tensordict.base import TensorDictBase
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import (
    ClipTransform,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNormV2,
)
from torchrl.envs.transforms import Transform
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

log = logging.getLogger(__name__)


# ── NaN diagnostic guard ──────────────────────────────────────────────


class NanDumpTransform(Transform):
    """Detect non-finite raw env observations and dump the full step context.

    Inserted *before* VecNormV2 so it sees the raw physics output.
    On first NaN detection it saves the TensorDict (action + resulting
    observation for all envs) to ``dump_path`` and raises RuntimeError.
    """

    def __init__(self, dump_path: str = "/root/nan_dump.pt"):
        super().__init__()
        self.dump_path = dump_path
        self._dumped = False

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        obs = next_tensordict.get("observation", None)
        if obs is not None and not self._dumped and not torch.isfinite(obs).all():
            self._dumped = True
            non_finite = ~torch.isfinite(obs)
            bad_envs = non_finite.flatten(1).any(dim=-1)  # [num_envs]
            n_bad_vals = non_finite.sum().item()
            n_bad_envs = bad_envs.sum().item()

            # Build dump: input td (has action) + output td (has obs)
            dump = tensordict.detach().cpu().clone()
            dump["next"] = next_tensordict.detach().cpu().clone()
            dump["_nan_env_mask"] = bad_envs.detach().cpu()
            torch.save(dump, self.dump_path)
            log.error(
                "NaN in raw env observation: %d/%d values across %d/%d envs. "
                "Dumped to %s",
                n_bad_vals,
                obs.numel(),
                n_bad_envs,
                obs.shape[0],
                self.dump_path,
            )
            raise RuntimeError(
                f"NaN in raw env observation: {n_bad_vals}/{obs.numel()} values "
                f"across {n_bad_envs}/{obs.shape[0]} envs. "
                f"Dumped to {self.dump_path}"
            )
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return tensordict_reset

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict


# ── Episode decorrelation ─────────────────────────────────────────────


class RandomTruncationTransform(Transform):
    """Randomly truncate episodes to decorrelate synchronized batched envs.

    With 4096 batched envs sharing ``max_episode_steps``, all envs hit
    truncation at nearly the same step, flooding the buffer with
    low-reward start-of-episode data in waves.  This transform breaks
    that synchronisation.

    On the **first reset** every env receives a horizon drawn from
    ``Uniform(1, max_horizon)`` so they immediately spread across
    different phases.  On **subsequent resets**, with probability
    ``prob`` a random horizon from ``Uniform(min_horizon, max_horizon)``
    is assigned; otherwise the full ``max_horizon`` is used.

    Must be placed **after** ``StepCounter`` in the transform chain.
    """

    def __init__(
        self, prob: float = 0.5, min_horizon: int = 500, max_horizon: int = 1000
    ):
        super().__init__()
        self.prob = prob
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self._horizons: torch.Tensor | None = None
        self._initialized = False

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        step_count = next_tensordict.get("step_count", None)
        if step_count is None or self._horizons is None:
            return next_tensordict

        should_truncate = step_count >= self._horizons
        if should_truncate.any():
            truncated = next_tensordict.get(
                "truncated", torch.zeros_like(should_truncate)
            )
            done = next_tensordict.get("done", torch.zeros_like(should_truncate))
            next_tensordict.set("truncated", truncated | should_truncate)
            next_tensordict.set("done", done | should_truncate)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        step_count = tensordict_reset.get("step_count", None)
        if step_count is None:
            return tensordict_reset

        if not self._initialized:
            # First reset: uniform spread for immediate decorrelation
            self._horizons = torch.randint(
                1,
                self.max_horizon + 1,
                step_count.shape,
                device=step_count.device,
            )
            self._initialized = True
            return tensordict_reset

        # Resample horizons for envs that just reset
        reset_mask = tensordict.get("_reset", None)
        if reset_mask is not None:
            mask = reset_mask.view_as(self._horizons).bool()
            if mask.any():
                n = int(mask.sum())
                new_h = torch.randint(
                    self.min_horizon,
                    self.max_horizon + 1,
                    (n,),
                    device=self._horizons.device,
                )
                # With probability (1 - prob), keep the full episode length
                keep_full = torch.rand(n, device=self._horizons.device) > self.prob
                new_h[keep_full] = self.max_horizon
                self._horizons[mask] = new_h.view_as(self._horizons[mask])
        return tensordict_reset

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict


# ── Environment factories ──────────────────────────────────────────────


def make_shared_vecnorm_data(env_name):
    """Create shared-memory VecNormV2 state for cross-process sharing.

    Returns a TensorDict with {loc, var, count} in shared memory.
    Both the training collector and evaluator should reference this so
    they share identical observation normalization statistics.
    """
    from tensordict import TensorDict

    proof = ENVS[env_name](
        num_envs=1, device="cpu", dtype=torch.float32, compile_step=False
    )
    proof = TransformedEnv(proof)
    proof.append_transform(
        VecNormV2(
            in_keys=["observation"], decay=0.99999, eps=1e-2, reduce_batch_dims=True
        )
    )
    # Reset triggers VecNormV2 lazy init (shapes come from data)
    proof.reset()
    vecnorm = proof.transform[0]
    shared = TensorDict(
        loc=vecnorm._loc.clone(),
        var=vecnorm._var.clone(),
        count=vecnorm._count.clone(),
    ).share_memory_()
    del proof
    return shared


def make_env(
    env_name="halfcheetah",
    device="cpu",
    num_envs=4096,
    compile=True,
    shared_vecnorm=None,
):
    """Create a batched MuJoCo env using mujoco-torch.

    Returns a single env with batch_size=[num_envs], where all envs run
    in parallel via torch.vmap (GPU) or sequential (CPU).
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
    # NaN guard on raw physics output — before VecNormV2 so we see un-normalised obs
    env.append_transform(NanDumpTransform(dump_path="/root/nan_dump.pt"))
    env.append_transform(
        VecNormV2(
            in_keys=["observation"],
            decay=0.99999,
            eps=1e-2,
            shared_data=shared_vecnorm,
            reduce_batch_dims=True,
        )
    )
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(RandomTruncationTransform(prob=0.5, max_horizon=1000))
    return env


def make_eval_env(env_name, device, num_eval_envs, max_steps=1000, shared_vecnorm=None):
    """Env factory for the Evaluator.

    Creates a compiled GPU-batched mujoco-torch env. The StepCounter
    uses max_steps so that episodes terminate — the Evaluator skips adding
    its own max_frames_per_traj when it sees an existing step_count.

    When shared_vecnorm is provided, the VecNormV2 is frozen — it reads
    the training collector's normalization stats without updating them.
    """
    env = ENVS[env_name](
        num_envs=num_eval_envs,
        device=device,
        dtype=torch.float32,
        compile_step=True,
        compile_kwargs={"mode": "default"},
    )
    env = TransformedEnv(env)
    vecnorm = VecNormV2(
        in_keys=["observation"],
        decay=0.99999,
        eps=1e-2,
        shared_data=shared_vecnorm,
        reduce_batch_dims=True,
    )
    if shared_vecnorm is not None:
        vecnorm.freeze()
    env.append_transform(vecnorm)
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
