# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn import dispatch, TensorDictModuleBase

from ..modules import TdMpc2QEnsemble
from .common import LossModule


def symlog_two_hot(
    value: torch.Tensor,
    vmin: float,
    vmax: float,
    num_bins: int,
) -> torch.Tensor:
    """Encode scalars with TD-MPC2's symlog-space two-hot representation.

    The categorical support is uniformly spaced in symlog space. The final
    dimension of ``value`` is interpreted as a singleton event dimension and
    the returned tensor has ``num_bins`` categories in its final dimension.

    Args:
        value: Scalar targets with shape ``[..., 1]``.
        vmin: Lower endpoint of the symlog-space support.
        vmax: Upper endpoint of the symlog-space support.
        num_bins: Number of categorical support points.

    Returns:
        A tensor containing the two-hot targets.
    """
    if num_bins <= 1:
        raise ValueError(f"num_bins must be greater than 1, got {num_bins}.")
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin.")

    transformed = torch.sign(value) * torch.log1p(value.abs())
    if transformed.shape[-1] == 1:
        transformed = transformed.squeeze(-1)
    transformed = transformed.clamp(vmin, vmax)
    bin_size = (vmax - vmin) / (num_bins - 1)
    position = (transformed - vmin) / bin_size
    index = position.floor().long()
    offset = position - index
    encoded = torch.zeros(
        *transformed.shape,
        num_bins,
        dtype=value.dtype,
        device=value.device,
    )
    encoded.scatter_(-1, index.unsqueeze(-1), (1 - offset).unsqueeze(-1))
    encoded.scatter_(
        -1,
        ((index + 1) % num_bins).unsqueeze(-1),
        offset.unsqueeze(-1),
    )
    return encoded


def symlog_two_hot_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    vmin: float,
    vmax: float,
    num_bins: int,
) -> torch.Tensor:
    """Compute cross-entropy between logits and symlog two-hot targets.

    Args:
        logits (torch.Tensor): Unnormalized categorical predictions with
            categories in the final dimension.
        target (torch.Tensor): Scalar target values with shape ``[..., 1]``.
        vmin (float): Lower endpoint of the symlog-space support.
        vmax (float): Upper endpoint of the symlog-space support.
        num_bins (int): Number of categorical support points.

    Returns:
        Per-sample cross-entropy with a trailing singleton event dimension.

    The returned tensor retains all non-category dimensions and has a trailing
    singleton event dimension, matching TD-MPC2 reward and value losses.
    """
    encoded = symlog_two_hot(target, vmin, vmax, num_bins)
    return -(encoded * F.log_softmax(logits, dim=-1)).sum(-1, keepdim=True)


class _TdMpc2RunningScale(nn.Module):
    """Track the robust value scale used by the TD-MPC2 policy objective."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("value", torch.ones(1))
        self.register_buffer("percentiles", torch.tensor([5.0, 95.0]))

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        """Update the scale from a batch of scalar values."""
        value = value.detach().reshape(-1, value.shape[-1])
        if value.shape[0] == 0:
            return
        if self.value.device != value.device or self.value.dtype != value.dtype:
            self.value = self.value.to(device=value.device, dtype=value.dtype)
        sorted_value = value.sort(dim=0).values
        positions = (
            self.percentiles.to(device=value.device, dtype=value.dtype)
            * (value.shape[0] - 1)
            / 100
        )
        lower = positions.floor().long()
        upper = positions.ceil().long()
        weight_upper = positions - lower
        percentiles = (
            sorted_value[lower] * (1 - weight_upper).unsqueeze(-1)
            + sorted_value[upper] * weight_upper.unsqueeze(-1)
        ).squeeze(-1)
        new_value = (percentiles[1] - percentiles[0]).clamp_min(1.0)
        self.value.lerp_(new_value.reshape_as(self.value), self.tau)

    def forward(self, value: torch.Tensor, *, update: bool = False) -> torch.Tensor:
        """Scale values, optionally updating the running estimate first."""
        if update:
            self.update(value)
        return value / self.value


def _tdmpc2_next_key(key: NestedKey) -> NestedKey:
    """Return the transition ``next`` counterpart of a TensorDict key."""
    if isinstance(key, tuple):
        if key and key[0] == "next":
            return key
        return ("next", *key)
    return ("next", key)


class TdMpc2Loss(LossModule):
    """Compute the TD-MPC2 model-learning objective.

    Args:
        world_model: TensorDict-native encoder, dynamics, and reward model.
        policy_prior: TensorDict module producing sampled actions and policy
            statistics from a latent state.
        q_ensemble: Distributional Q-function ensemble.
        horizon: Number of transitions in each sampled sequence.
        discount: Scalar discount factor used for TD targets.
        rho: Temporal weighting factor for model and actor losses.
        consistency_coef: Weight of latent consistency loss.
        reward_coef: Weight of distributional reward loss.
        value_coef: Weight of distributional value loss.
        entropy_coef: Entropy coefficient in the actor objective.
        scale_tau: Exponential averaging factor for the running value scale.
        observation_key: Current observation key.
        action_key: Action key.
        reward_key: Reward key under ``"next"``.
        terminated_key: Termination key under ``"next"``.

    .. seealso::
        :class:`~torchrl.trainers.algorithms.configs.TdMpc2LossConfig`,
        `TD-MPC2: Scalable, Robust World Models for Continuous Control
        <https://arxiv.org/abs/2310.16828>`_.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.objectives import TdMpc2Loss
        >>> # The three TensorDict-native components are built from the
        >>> # corresponding TD-MPC2 configuration classes.
        >>> loss = TdMpc2Loss(world_model, policy_prior, q_ensemble)  # doctest: +SKIP
    """

    @dataclass
    class _AcceptedKeys:
        """TensorDict entries consumed by the TD-MPC2 objective."""

        observation: NestedKey = "observation"
        action: NestedKey = "action"
        reward: NestedKey = "reward"
        terminated: NestedKey = "terminated"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys

    def __init__(
        self,
        world_model: TensorDictModuleBase,
        policy_prior: TensorDictModuleBase,
        q_ensemble: TdMpc2QEnsemble,
        *,
        horizon: int = 3,
        discount: float = 0.99,
        rho: float = 0.5,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        entropy_coef: float = 1e-4,
        scale_tau: float = 0.01,
        observation_key: NestedKey = "observation",
        action_key: NestedKey = "action",
        reward_key: NestedKey = "reward",
        terminated_key: NestedKey = "terminated",
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        if not 0 <= discount <= 1:
            raise ValueError(f"discount must be in [0, 1], got {discount}.")
        if not 0 <= rho <= 1:
            raise ValueError(f"rho must be in [0, 1], got {rho}.")
        for name, value in {
            "consistency_coef": consistency_coef,
            "reward_coef": reward_coef,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if not 0 < scale_tau <= 1:
            raise ValueError(f"scale_tau must be in (0, 1], got {scale_tau}.")

        self.world_model = world_model
        self.policy_prior = policy_prior
        self.q_ensemble = q_ensemble
        self.horizon = int(horizon)
        self.consistency_coef = float(consistency_coef)
        self.reward_coef = float(reward_coef)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.register_buffer("discount", torch.as_tensor(float(discount)))
        self.rho = float(rho)
        self.scale = _TdMpc2RunningScale(scale_tau)

        self._tensor_keys = self._AcceptedKeys(
            observation=observation_key,
            action=action_key,
            reward=reward_key,
            terminated=terminated_key,
        )
        try:
            self.latent_key = world_model.encoder.out_keys[0]
            self.next_latent_key = _tdmpc2_next_key(self.latent_key)
            self.reward_logits_key = world_model.reward_head.out_keys[0]
            self.next_reward_logits_key = _tdmpc2_next_key(self.reward_logits_key)
            self.world_model_observation_key = world_model.encoder.in_keys[0]
            self.world_model_action_key = world_model.dynamics.in_keys[1]
        except (AttributeError, IndexError) as err:
            raise TypeError(
                "world_model must expose encoder, dynamics, and reward_head "
                "TensorDict modules."
            ) from err
        self.policy_latent_key = policy_prior.in_keys[0]
        self.policy_action_key = policy_prior.out_keys[0]
        if len(policy_prior.out_keys) < 5:
            raise ValueError(
                "policy_prior must expose action, mean, log_std, entropy, and "
                "scaled_entropy outputs."
            )
        self.policy_entropy_key = policy_prior.out_keys[3]
        self.policy_scaled_entropy_key = policy_prior.out_keys[4]
        self.q_logits_key = q_ensemble.out_keys_source[0]

    @property
    def in_keys(self) -> list[NestedKey]:
        """Return the canonical current and next transition keys."""
        return [
            self.tensor_keys.observation,
            self.tensor_keys.action,
            _tdmpc2_next_key(self.tensor_keys.observation),
            _tdmpc2_next_key(self.tensor_keys.reward),
            _tdmpc2_next_key(self.tensor_keys.terminated),
        ]

    @property
    def out_keys(self) -> list[NestedKey]:
        """Return the scalar loss and diagnostic keys written by ``forward``."""
        return [
            "loss_consistency",
            "loss_reward",
            "loss_value",
        ]

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        """TD-MPC2 does not use TorchRL's value-estimator interface."""

    @staticmethod
    def _make_td(
        batch_size: torch.Size,
        values: Sequence[tuple[NestedKey, torch.Tensor]],
    ) -> TensorDict:
        td = TensorDict({}, batch_size=batch_size)
        for key, value in values:
            td.set(key, value)
        return td

    def _sequence_tensors(
        self, sample: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample.ndim < 1:
            raise ValueError(
                "TdMpc2Loss expects a time dimension in the TensorDict batch shape."
            )
        if sample.batch_size[-1] != self.horizon:
            raise ValueError(
                f"Expected a final time dimension of {self.horizon}, got "
                f"{sample.batch_size[-1]}."
            )
        keys = self.tensor_keys
        observation = sample.get(keys.observation)
        action = sample.get(keys.action)
        next_observation = sample.get(_tdmpc2_next_key(keys.observation))
        reward = sample.get(_tdmpc2_next_key(keys.reward))
        terminated = sample.get(_tdmpc2_next_key(keys.terminated))
        missing = [
            name
            for name, value in (
                ("observation", observation),
                ("action", action),
                ("next/observation", next_observation),
                ("next/reward", reward),
                ("next/terminated", terminated),
            )
            if value is None
        ]
        if missing:
            raise KeyError(f"Missing TD-MPC2 sequence keys: {missing}.")
        if reward.ndim == sample.ndim:
            reward = reward.unsqueeze(-1)
        if terminated.ndim == sample.ndim:
            terminated = terminated.unsqueeze(-1)
        terminated = terminated.to(dtype=reward.dtype)
        expected_batch = sample.batch_size
        for name, value in (
            ("observation", observation),
            ("action", action),
            ("next/observation", next_observation),
            ("reward", reward),
            ("terminated", terminated),
        ):
            if torch.Size(value.shape[: sample.ndim]) != expected_batch:
                raise ValueError(
                    f"{name} has batch shape {value.shape[: sample.ndim]}, "
                    f"expected {expected_batch}."
                )
        for name, value in (("reward", reward), ("terminated", terminated)):
            if value.shape[sample.ndim :] != torch.Size([1]):
                raise ValueError(
                    f"{name} must have a singleton event dimension after the "
                    f"batch dimensions, got shape {value.shape}."
                )
        return observation, action, next_observation, reward, terminated

    def _loss_mask(self, sample: TensorDictBase) -> torch.Tensor | None:
        """Return the validity mask for the sample, if one is available."""
        mask = None
        for mask_key in self._loss_mask_keys():
            candidate = sample.get(mask_key, default=None)
            if candidate is None:
                continue
            if candidate.ndim == sample.ndim + 1 and candidate.shape[-1] == 1:
                candidate = candidate.squeeze(-1)
            if candidate.shape != sample.batch_size:
                raise ValueError(
                    f"Loss mask {mask_key!r} has shape {candidate.shape}, expected "
                    f"{sample.batch_size}."
                )
            candidate = candidate.to(dtype=torch.bool)
            mask = candidate if mask is None else mask & candidate
        return mask

    @staticmethod
    def _masked_transition_mean(
        value: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reduce per-transition values while ignoring invalid positions."""
        if mask is None:
            return value.mean()
        mask = mask.to(dtype=torch.bool)
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(value)
        return torch.where(
            mask, value, torch.zeros_like(value)
        ).sum() / mask.sum().clamp_min(1)

    def _encode(self, observation: torch.Tensor) -> torch.Tensor:
        batch_size = torch.Size(observation.shape[:-1])
        td = self._make_td(
            batch_size, [(self.world_model_observation_key, observation)]
        )
        return self.world_model.encode(td).get(self.latent_key)

    def _policy(self, latent: torch.Tensor) -> TensorDictBase:
        td = self._make_td(
            torch.Size(latent.shape[:-1]), [(self.policy_latent_key, latent)]
        )
        return self.policy_prior(td)

    def _q_logits(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        *,
        source: Literal["online", "detached", "target"],
    ) -> torch.Tensor:
        td = self._make_td(
            torch.Size(latent.shape[:-1]),
            [
                (self.q_ensemble.in_keys[0], latent),
                (self.q_ensemble.in_keys[1], action),
            ],
        )
        return self.q_ensemble(td, source=source).get(self.q_logits_key)

    def _q_value(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        *,
        reduction: Literal["min", "avg"],
        source: Literal["online", "detached", "target"],
    ) -> torch.Tensor:
        td = self._make_td(
            torch.Size(latent.shape[:-1]),
            [
                (self.q_ensemble.in_keys[0], latent),
                (self.q_ensemble.in_keys[1], action),
            ],
        )
        return self.q_ensemble.reduce(td, reduction=reduction, source=source).get(
            self.q_ensemble.q_value_key
        )

    @torch.no_grad()
    def _td_targets(
        self,
        next_latent: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        policy_td = self._policy(next_latent)
        next_action = policy_td.get(self.policy_action_key)
        next_value = self._q_value(
            next_latent, next_action, reduction="min", source="target"
        )
        return reward + self.discount * (1 - terminated.to(reward.dtype)) * next_value

    def _rollout(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._encode(observation.select(-2, 0))
        zs = [z]
        reward_predictions = []
        for index in range(self.horizon):
            current_action = action.select(-2, index)
            step_td = self._make_td(
                torch.Size(z.shape[:-1]),
                [(self.latent_key, z), (self.world_model_action_key, current_action)],
            )
            step_td = self.world_model.step(step_td)
            z = step_td.get(self.next_latent_key)
            zs.append(z)
            reward_predictions.append(step_td.get(self.next_reward_logits_key))
        latent_sequence = torch.stack(zs, dim=-2)
        reward_predictions = torch.stack(reward_predictions, dim=-2)
        q_predictions = self._q_logits(
            latent_sequence[..., :-1, :], action, source="online"
        )
        return latent_sequence, reward_predictions, q_predictions

    def _model_loss_components(
        self, sample: TensorDictBase
    ) -> tuple[torch.Tensor, TensorDictBase]:
        (
            observation,
            action,
            next_observation,
            reward,
            terminated,
        ) = self._sequence_tensors(sample)
        loss_mask = self._loss_mask(sample)
        with torch.no_grad():
            next_latent = self._encode(next_observation)
            td_targets = self._td_targets(next_latent, reward, terminated)

        latent_sequence, reward_predictions, q_predictions = self._rollout(
            observation, action
        )
        consistency_loss = latent_sequence.new_zeros(())
        reward_loss = latent_sequence.new_zeros(())
        value_loss = latent_sequence.new_zeros(())
        rho = latent_sequence.new_tensor(self.rho).pow(
            torch.arange(self.horizon, device=latent_sequence.device)
        )
        for index in range(self.horizon):
            transition_mask = None if loss_mask is None else loss_mask[..., index]
            consistency_per_transition = F.mse_loss(
                latent_sequence.select(-2, index + 1),
                next_latent.select(-2, index),
                reduction="none",
            ).mean(-1)
            consistency_loss = (
                consistency_loss
                + self._masked_transition_mean(
                    consistency_per_transition,
                    transition_mask,
                )
                * rho[index]
            )
            reward_per_transition = symlog_two_hot_cross_entropy(
                reward_predictions.select(-2, index),
                reward.select(-2, index),
                self.q_ensemble.vmin,
                self.q_ensemble.vmax,
                self.q_ensemble.num_bins,
            ).squeeze(-1)
            reward_loss = (
                reward_loss
                + self._masked_transition_mean(
                    reward_per_transition,
                    transition_mask,
                )
                * rho[index]
            )
            value_target = (
                td_targets.select(-2, index)
                .unsqueeze(-2)
                .expand(
                    *td_targets.select(-2, index).shape[:-1],
                    self.q_ensemble.num_q,
                    td_targets.shape[-1],
                )
            )
            value_per_transition = (
                symlog_two_hot_cross_entropy(
                    q_predictions.select(-3, index),
                    value_target,
                    self.q_ensemble.vmin,
                    self.q_ensemble.vmax,
                    self.q_ensemble.num_bins,
                )
                .squeeze(-1)
                .mean(-1)
            )
            value_loss = (
                value_loss
                + self._masked_transition_mean(
                    value_per_transition,
                    transition_mask,
                )
                * self.q_ensemble.num_q
                * rho[index]
            )
        consistency_loss = consistency_loss / self.horizon
        reward_loss = reward_loss / self.horizon
        value_loss = value_loss / (self.horizon * self.q_ensemble.num_q)
        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
        )
        metadata = TensorDict(
            {
                "loss_consistency": consistency_loss,
                "loss_reward": reward_loss,
                "loss_value": value_loss,
                "actor_latents": latent_sequence.detach(),
            },
            batch_size=[],
        )
        return total_loss, metadata

    def model_loss(self, sample: TensorDictBase) -> tuple[torch.Tensor, TensorDictBase]:
        """Compute the model, reward, value, and consistency objectives.

        Args:
            sample: Canonical batch-major transition sequence with final batch
                dimension equal to ``horizon``.

        Returns:
            The weighted model loss and detached latent/model metadata for the
            subsequent actor phase.
        """
        return self._model_loss_components(sample)

    def _actor_loss_from_latents(
        self, latent_sequence: torch.Tensor
    ) -> tuple[torch.Tensor, TensorDictBase]:
        policy_td = self._policy(latent_sequence)
        action = policy_td.get(self.policy_action_key)
        q_value = self._q_value(
            latent_sequence, action, reduction="avg", source="detached"
        )
        self.scale.update(q_value.select(-2, 0).reshape(-1, q_value.shape[-1]))
        normalized_q = self.scale(q_value)
        entropy = policy_td.get(self.policy_entropy_key)
        scaled_entropy = policy_td.get(self.policy_scaled_entropy_key)
        temporal_rho = latent_sequence.new_tensor(self.rho).pow(
            torch.arange(latent_sequence.shape[-2], device=latent_sequence.device)
        )
        actor_loss = -(self.entropy_coef * scaled_entropy + normalized_q)
        temporal_rho = temporal_rho.reshape((1,) * (actor_loss.ndim - 2) + (-1, 1))
        actor_loss = (actor_loss * temporal_rho).mean()
        metadata = TensorDict(
            {
                "pi_entropy": entropy.detach(),
                "pi_scaled_entropy": scaled_entropy.detach(),
                "pi_scale": self.scale.value.detach().clone(),
            },
            batch_size=[],
        )
        return actor_loss, metadata

    def actor_loss_from_latents(
        self, latent_sequence: torch.Tensor
    ) -> tuple[torch.Tensor, TensorDictBase]:
        """Compute the actor objective from detached imagined latents."""
        return self._actor_loss_from_latents(latent_sequence.detach())

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Return the independent weighted model-loss components."""
        _, metadata = self._model_loss_components(tensordict)
        return TensorDict(
            {
                "loss_consistency": self.consistency_coef
                * metadata.get("loss_consistency"),
                "loss_reward": self.reward_coef * metadata.get("loss_reward"),
                "loss_value": self.value_coef * metadata.get("loss_value"),
            },
            batch_size=[],
        )
