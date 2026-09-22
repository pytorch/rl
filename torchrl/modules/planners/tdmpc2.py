# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from contextlib import contextmanager
from numbers import Real
from typing import TYPE_CHECKING

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase

from ..models.tdmpc2 import symlog_two_hot_decode, TdMpc2QEnsemble

if TYPE_CHECKING:
    from ...envs.transforms import TensorDictPrimer


def _tdmpc2_next_key(key: NestedKey) -> NestedKey:
    """Return the transition ``next`` counterpart of a TensorDict key."""
    if isinstance(key, tuple):
        if key and key[0] == "next":
            return key
        return ("next", *key)
    return ("next", key)


class TdMpc2Planner(TensorDictModuleBase):
    """Select actions with the latent-space TD-MPC2 planner.

    The planner combines policy-prior trajectories with sampled action
    sequences, scores them with the world model and Q ensemble, and refits a
    Gaussian distribution to the elite trajectories. Actions are expected to
    be normalized to ``[-1, 1]``.

    The fitted action mean is written under ``("next", prev_mean_key)`` so a
    collector can carry the warm-start state between environment steps.

    Args:
        world_model: TD-MPC2 world model exposing ``encoder``, ``dynamics``,
            and ``reward_head`` TensorDict modules.
        policy_prior: TD-MPC2 policy-prior TensorDict module.
        q_ensemble: TD-MPC2 distributional Q ensemble exposing ``reduce``.
        horizon: Number of imagined action steps. Two additional search
            iterations are used when ``action_dim >= 20``.
        discount: Scalar discount used for imagined rewards and Q bootstrap.
        num_samples: Number of candidate action trajectories.
        num_elites: Number of candidates used for Gaussian refitting.
        num_pi_trajs: Number of fixed policy-prior trajectories.
        iterations: Number of elite-refitting iterations.
        min_std: Minimum fitted action standard deviation.
        max_std: Initial and maximum action standard deviation.
        temperature: Elite score temperature.
        observation_key: Observation key consumed by the planner.
        action_key: Action key written by the planner.
        is_init_key: Per-environment reset indicator.
        prev_mean_key: Private root key carrying the previous fitted mean.
        action_dim: Optional action dimension. It is inferred from the
            configured TD-MPC2 policy prior when omitted.
    """

    def __init__(
        self,
        world_model: TensorDictModuleBase,
        policy_prior: TensorDictModuleBase,
        q_ensemble: TdMpc2QEnsemble,
        *,
        horizon: int = 3,
        discount: float = 0.99,
        num_samples: int = 512,
        num_elites: int = 64,
        num_pi_trajs: int = 24,
        iterations: int = 6,
        min_std: float = 0.05,
        max_std: float = 2.0,
        temperature: float = 0.5,
        observation_key: NestedKey | None = None,
        action_key: NestedKey | None = None,
        is_init_key: NestedKey = "is_init",
        prev_mean_key: NestedKey = "_tdmpc2_prev_mean",
        action_dim: int | None = None,
    ) -> None:
        super().__init__()
        integer_fields = {
            "horizon": horizon,
            "num_samples": num_samples,
            "num_elites": num_elites,
            "num_pi_trajs": num_pi_trajs,
            "iterations": iterations,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}.")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if not 0 <= num_pi_trajs <= num_samples:
            raise ValueError(
                "num_pi_trajs must be in [0, num_samples], got "
                f"{num_pi_trajs} for num_samples={num_samples}."
            )
        if not 1 <= num_elites <= num_samples:
            raise ValueError(
                "num_elites must be in [1, num_samples], got "
                f"{num_elites} for num_samples={num_samples}."
            )
        if (
            not isinstance(discount, Real)
            or not math.isfinite(discount)
            or not 0 <= discount <= 1
        ):
            raise ValueError(f"discount must be in [0, 1], got {discount}.")
        if (
            not isinstance(min_std, Real)
            or not isinstance(max_std, Real)
            or not math.isfinite(min_std)
            or not math.isfinite(max_std)
            or min_std < 0
            or max_std < min_std
        ):
            raise ValueError(
                f"expected 0 <= min_std <= max_std, got {min_std}, {max_std}."
            )
        if (
            not isinstance(temperature, Real)
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError(
                f"temperature must be finite and non-negative, got {temperature}."
            )
        try:
            latent_key = world_model.encoder.out_keys[0]
            world_model_observation_key = world_model.encoder.in_keys[0]
            world_model_action_key = world_model.dynamics.in_keys[1]
            next_latent_key = world_model.dynamics.out_keys[0]
            reward_logits_key = world_model.reward_head.out_keys[0]
            policy_latent_key = policy_prior.in_keys[0]
            policy_action_key = policy_prior.out_keys[0]
            q_latent_key, q_action_key = q_ensemble.in_keys
        except (AttributeError, IndexError, ValueError) as err:
            raise TypeError(
                "TdMpc2Planner expects TensorDict-native world-model, policy "
                "prior, and Q-ensemble components."
            ) from err
        if getattr(world_model, "done_head", None) is not None:
            raise NotImplementedError(
                "TdMpc2Planner does not yet support imagined termination."
            )
        if (
            observation_key is not None
            and observation_key != world_model_observation_key
        ):
            raise ValueError(
                "observation_key must match the world-model encoder input key, "
                f"got {observation_key!r} and {world_model_observation_key!r}."
            )
        if action_key is None:
            action_key = policy_action_key
        if not isinstance(action_key, (str, tuple)):
            raise TypeError(f"action_key must be a TensorDict key, got {action_key!r}.")
        prev_mean_leaf = (
            prev_mean_key[-1] if isinstance(prev_mean_key, tuple) else prev_mean_key
        )
        if not isinstance(prev_mean_leaf, str) or not prev_mean_leaf.startswith("_"):
            raise ValueError(
                "prev_mean_key must be private (its final key component must "
                f"start with '_'), got {prev_mean_key!r}."
            )
        if policy_latent_key != latent_key or q_latent_key != latent_key:
            raise ValueError(
                "TD-MPC2 world-model latent key must match the policy-prior and "
                "Q-ensemble latent keys, got "
                f"{latent_key!r}, {policy_latent_key!r}, and {q_latent_key!r}."
            )
        if (
            policy_action_key != world_model_action_key
            or q_action_key != world_model_action_key
        ):
            raise ValueError(
                "TD-MPC2 action keys must match across the world model, policy "
                "prior, and Q ensemble, got "
                f"{world_model_action_key!r}, {policy_action_key!r}, and "
                f"{q_action_key!r}."
            )

        if action_dim is None:
            policy_module = getattr(policy_prior, "module", None)
            action_dim = getattr(policy_module, "action_dim", None)
            network = getattr(policy_module, "network", None)
            try:
                if action_dim is None:
                    policy_output_dim = network[-1].out_features
                    if policy_output_dim % 2:
                        raise ValueError
                    action_dim = policy_output_dim // 2
            except (AttributeError, IndexError, TypeError, ValueError) as err:
                raise ValueError(
                    "action_dim must be provided when it cannot be inferred from "
                    "the policy-prior network."
                ) from err
        if isinstance(action_dim, bool) or not isinstance(action_dim, int):
            raise TypeError(f"action_dim must be an integer, got {action_dim!r}.")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")

        self._world_model = world_model
        self._policy_prior = policy_prior
        self._q_ensemble = q_ensemble
        self.horizon = int(horizon)
        self.discount = float(discount)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.num_pi_trajs = int(num_pi_trajs)
        self.iterations = int(iterations) + 2 * int(action_dim >= 20)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.temperature = float(temperature)
        self.action_dim = int(action_dim)
        self.observation_key = (
            world_model_observation_key if observation_key is None else observation_key
        )
        self.action_key = action_key
        self.is_init_key = is_init_key
        self.prev_mean_key = prev_mean_key
        self.next_prev_mean_key = _tdmpc2_next_key(prev_mean_key)
        self.latent_key = latent_key
        self.world_model_observation_key = world_model_observation_key
        self.world_model_action_key = world_model_action_key
        self.next_latent_key = next_latent_key
        self.reward_logits_key = reward_logits_key
        self.policy_latent_key = policy_latent_key
        self.policy_action_key = policy_action_key
        self.q_latent_key = q_latent_key
        self.q_action_key = q_action_key
        self.in_keys = [self.observation_key, self.is_init_key, self.prev_mean_key]
        self.out_keys = [self.action_key, self.next_prev_mean_key]

    @property
    def world_model(self) -> TensorDictModuleBase:
        """Return the live world model borrowed from the learner."""
        return self._world_model

    @property
    def policy_prior(self) -> TensorDictModuleBase:
        """Return the live policy prior borrowed from the learner."""
        return self._policy_prior

    @property
    def q_ensemble(self) -> TdMpc2QEnsemble:
        """Return the live Q ensemble borrowed from the learner."""
        return self._q_ensemble

    def make_tensordict_primer(self) -> TensorDictPrimer:
        """Create the primer needed to carry the planner warm-start state."""
        from ...data.tensor_specs import Unbounded
        from ...envs.transforms import TensorDictPrimer

        return TensorDictPrimer(
            {self.prev_mean_key: Unbounded(shape=(self.horizon, self.action_dim))},
            expand_specs=True,
        )

    @contextmanager
    def _inference_components(self):
        roots = []
        seen = set()
        for module in (self.world_model, self.policy_prior, self.q_ensemble):
            if id(module) not in seen:
                roots.append(module)
                seen.add(id(module))
        modules = []
        seen_modules = set()
        for root in roots:
            for module in root.modules():
                if id(module) not in seen_modules:
                    modules.append(module)
                    seen_modules.add(id(module))
        training = [module.training for module in modules]
        try:
            for module in roots:
                module.eval()
            yield
        finally:
            for module, was_training in zip(modules, training):
                module.train(was_training)

    def _policy_action(self, latent: torch.Tensor) -> torch.Tensor:
        policy_td = TensorDict({}, batch_size=latent.shape[:-1], device=latent.device)
        policy_td.set(self.policy_latent_key, latent)
        self.policy_prior(policy_td)
        return policy_td.get(self.policy_action_key)

    def _step(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_td = TensorDict({}, batch_size=latent.shape[:-1], device=latent.device)
        step_td.set(self.latent_key, latent)
        step_td.set(self.world_model_action_key, action)
        self.world_model.step(step_td)
        return step_td.get(self.next_latent_key), step_td.get(
            _tdmpc2_next_key(self.reward_logits_key)
        )

    def _next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Advance latent state without evaluating the reward head."""
        step_td = TensorDict({}, batch_size=latent.shape[:-1], device=latent.device)
        step_td.set(self.latent_key, latent)
        step_td.set(self.world_model_action_key, action)
        self.world_model.dynamics(step_td)
        return step_td.get(self.next_latent_key)

    def _q_value(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_td = TensorDict({}, batch_size=latent.shape[:-1], device=latent.device)
        q_td.set(self.q_latent_key, latent)
        q_td.set(self.q_action_key, action)
        self.q_ensemble.reduce(q_td, reduction="avg", source="online")
        return q_td.get(self.q_ensemble.q_value_key)

    def _estimate_value(
        self, latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        value = latent.new_zeros((*latent.shape[:-1], 1))
        discount = latent.new_tensor(1.0)
        current_latent = latent
        for index in range(self.horizon):
            current_latent, reward_logits = self._step(current_latent, actions[index])
            reward = symlog_two_hot_decode(
                reward_logits,
                self.q_ensemble.vmin,
                self.q_ensemble.vmax,
                self.q_ensemble.num_bins,
            )
            value = value + discount * reward
            discount = discount * self.discount
        terminal_action = self._policy_action(current_latent)
        return value + discount * self._q_value(current_latent, terminal_action)

    def _plan_single(
        self,
        latent: torch.Tensor,
        previous_mean: torch.Tensor | None,
        is_init: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = latent.device
        dtype = latent.dtype
        mean = torch.zeros(self.horizon, self.action_dim, device=device, dtype=dtype)
        if previous_mean is not None and not is_init:
            mean[:-1] = previous_mean[1:]
        std = torch.full(
            (self.horizon, self.action_dim),
            self.max_std,
            device=device,
            dtype=dtype,
        )

        if self.num_pi_trajs:
            policy_actions = torch.empty(
                self.horizon,
                self.num_pi_trajs,
                self.action_dim,
                device=device,
                dtype=dtype,
            )
            policy_latent = latent.repeat(self.num_pi_trajs, 1)
            for index in range(self.horizon - 1):
                policy_actions[index] = self._policy_action(policy_latent)
                policy_latent = self._next_latent(policy_latent, policy_actions[index])
            policy_actions[-1] = self._policy_action(policy_latent)
        else:
            policy_actions = None

        candidate_latent = latent.repeat(self.num_samples, 1)
        actions = torch.empty(
            self.horizon,
            self.num_samples,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        if policy_actions is not None:
            actions[:, : self.num_pi_trajs] = policy_actions

        for _ in range(self.iterations):
            sample_shape = (
                self.horizon,
                self.num_samples - self.num_pi_trajs,
                self.action_dim,
            )
            noise = torch.randn(sample_shape, device=device, dtype=dtype)
            sampled_actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(
                -1, 1
            )
            actions[:, self.num_pi_trajs :] = sampled_actions

            value = self._estimate_value(candidate_latent, actions).nan_to_num(0)
            elite_indices = torch.topk(value.squeeze(1), self.num_elites, dim=0).indices
            elite_value = value[elite_indices]
            elite_actions = actions[:, elite_indices]

            max_value = elite_value.max(0).values
            score = torch.exp(self.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (
                score.sum(0) + 1e-9
            )
            std = (
                (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)).pow(2)).sum(
                    dim=1
                )
                / (score.sum(0) + 1e-9)
            ).sqrt()
            std = std.clamp(self.min_std, self.max_std)

        score = score.squeeze(1)
        gumbels = -torch.empty_like(score).exponential_().log()
        elite_index = (score.log() + gumbels).argmax(-1)
        selected = torch.index_select(elite_actions, 1, elite_index.reshape(1)).squeeze(
            1
        )
        action = selected[0]
        from ...envs.utils import exploration_type, ExplorationType

        interaction = exploration_type()
        if interaction is None:
            use_exploration_noise = self.training
        else:
            use_exploration_noise = interaction is ExplorationType.RANDOM
        if use_exploration_noise:
            action = action + std[0] * torch.randn(
                self.action_dim, device=device, dtype=dtype
            )
        return action.clamp(-1, 1), mean

    @torch.no_grad()
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Plan actions for a TensorDict and write the action/state outputs.

        Args:
            tensordict: TensorDict containing the observation, optional
                ``is_init`` reset indicator, and optional previous mean.

        Returns:
            The input TensorDict with the planned action and the fitted mean
            under ``("next", prev_mean_key)``.
        """
        observation = tensordict.get(self.observation_key)
        batch_shape = observation.shape[:-1]
        flat_observation = observation.reshape(-1, observation.shape[-1])
        encode_td = TensorDict(
            {}, batch_size=[flat_observation.shape[0]], device=observation.device
        )
        encode_td.set(self.world_model_observation_key, flat_observation)
        with self._inference_components():
            self.world_model.encode(encode_td)
            latent = encode_td.get(self.latent_key)

            try:
                previous_mean = tensordict.get(self.prev_mean_key)
            except KeyError:
                previous_mean = None
            if previous_mean is not None:
                previous_mean = previous_mean.reshape(-1, self.horizon, self.action_dim)
            try:
                is_init = tensordict.get(self.is_init_key)
            except KeyError:
                is_init = None
            if is_init is None:
                is_init = torch.zeros(
                    flat_observation.shape[0],
                    dtype=torch.bool,
                    device=observation.device,
                )
            else:
                is_init = is_init.reshape(-1).to(torch.bool)

            actions = []
            fitted_means = []
            for index in range(latent.shape[0]):
                previous = None if previous_mean is None else previous_mean[index]
                action, fitted_mean = self._plan_single(
                    latent[index], previous, bool(is_init[index])
                )
                actions.append(action)
                fitted_means.append(fitted_mean)
            action = torch.stack(actions).reshape(*batch_shape, self.action_dim)
            fitted_mean = torch.stack(fitted_means).reshape(
                *batch_shape, self.horizon, self.action_dim
            )

        tensordict.set(self.action_key, action)
        tensordict.set(self.next_prev_mean_key, fitted_mean)
        return tensordict
