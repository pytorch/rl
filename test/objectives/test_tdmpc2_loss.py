# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.modules import TdMpc2QEnsemble, WorldModel
from torchrl.objectives import TdMpc2Loss


class TestTdMpc2Loss:
    _OBSERVATION_DIM = 5
    _ACTION_DIM = 2
    _LATENT_DIM = 8
    _NUM_BINS = 5
    _HORIZON = 3

    class _ConcatLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, latent, action):
            return self.linear(torch.cat((latent, action), dim=-1))

    class _Policy(nn.Module):
        def __init__(self, latent_dim, action_dim):
            super().__init__()
            self.network = nn.Linear(latent_dim, action_dim)

        def forward(self, latent):
            action = torch.tanh(self.network(latent))
            zeros = latent.new_zeros(*latent.shape[:-1], 1)
            return action, action, torch.zeros_like(action), zeros, zeros

    @classmethod
    def _make_loss(cls):
        observation_dim = cls._OBSERVATION_DIM
        action_dim = cls._ACTION_DIM
        latent_dim = cls._LATENT_DIM
        num_bins = cls._NUM_BINS

        encoder = TensorDictModule(
            nn.Linear(observation_dim, latent_dim),
            in_keys=["observation"],
            out_keys=["latent"],
        )
        dynamics = TensorDictModule(
            cls._ConcatLinear(latent_dim + action_dim, latent_dim),
            in_keys=["latent", "action"],
            out_keys=[("next", "latent")],
        )
        reward_head = TensorDictModule(
            cls._ConcatLinear(latent_dim + action_dim, num_bins),
            in_keys=["latent", "action"],
            out_keys=[("next", "reward_logits")],
        )
        world_model = WorldModel(encoder, dynamics, reward_head)

        policy_prior = TensorDictModule(
            cls._Policy(latent_dim, action_dim),
            in_keys=["latent"],
            out_keys=["action", "mean", "log_std", "entropy", "scaled_entropy"],
        )
        q_ensemble = TdMpc2QEnsemble(
            [nn.Linear(latent_dim + action_dim, num_bins) for _ in range(2)],
            num_bins=num_bins,
            vmin=-10.0,
            vmax=10.0,
        )
        return TdMpc2Loss(world_model, policy_prior, q_ensemble, horizon=cls._HORIZON)

    def test_forward(self):
        observation_dim, action_dim = self._OBSERVATION_DIM, self._ACTION_DIM
        loss = self._make_loss()
        sample = TensorDict(
            {
                "observation": torch.randn(2, 3, observation_dim),
                "action": torch.randn(2, 3, action_dim),
                ("next", "observation"): torch.randn(2, 3, observation_dim),
                ("next", "reward"): torch.randn(2, 3, 1),
                ("next", "terminated"): torch.zeros(2, 3, 1, dtype=torch.bool),
            },
            batch_size=[2, 3],
        )

        with torch.no_grad():
            output = loss(sample)

        expected_keys = {"loss_consistency", "loss_reward", "loss_value"}
        assert set(output.keys()) == expected_keys
        assert set(loss.out_keys) == expected_keys
        assert all(output[key].shape == torch.Size([]) for key in expected_keys)

    def test_model_loss(self):
        observation_dim, action_dim = self._OBSERVATION_DIM, self._ACTION_DIM
        loss = self._make_loss()
        sample = TensorDict(
            {
                "observation": torch.randn(2, 3, observation_dim),
                "action": torch.randn(2, 3, action_dim),
                ("next", "observation"): torch.randn(2, 3, observation_dim),
                ("next", "reward"): torch.randn(2, 3, 1),
                ("next", "terminated"): torch.zeros(2, 3, 1, dtype=torch.bool),
            },
            batch_size=[2, 3],
        )

        model_loss, metadata = loss.model_loss(sample)

        expected_loss = (
            loss.consistency_coef * metadata["loss_consistency"]
            + loss.reward_coef * metadata["loss_reward"]
            + loss.value_coef * metadata["loss_value"]
        )
        torch.testing.assert_close(model_loss, expected_loss)
        assert model_loss.shape == torch.Size([])
        assert model_loss.requires_grad
        for key in ("loss_consistency", "loss_reward", "loss_value"):
            assert metadata[key].requires_grad
        assert not metadata["actor_latents"].requires_grad

        model_loss.backward()

    def test_actor_loss_from_latents(self):
        latent_sequence = torch.randn(
            2,
            self._HORIZON + 1,
            self._LATENT_DIM,
            requires_grad=True,
        )
        loss = self._make_loss()

        actor_loss, metadata = loss.actor_loss_from_latents(latent_sequence)

        assert actor_loss.shape == torch.Size([])
        assert actor_loss.requires_grad
        assert set(metadata.keys()) == {
            "pi_entropy",
            "pi_scaled_entropy",
            "pi_scale",
        }
        for key in metadata.keys():
            assert not metadata[key].requires_grad

        actor_loss.backward()

        assert latent_sequence.grad is None

    def test_actor_latents(self):
        observed = {}

        class _IdentityScale:
            value = torch.ones(())

            def update(self, value):
                pass

            def __call__(self, value):
                return value

        class _FakeLoss:
            policy_action_key = "action"
            policy_entropy_key = "entropy"
            policy_scaled_entropy_key = "scaled_entropy"
            entropy_coef = 1e-4
            rho = 0.5
            scale = _IdentityScale()

            def _policy(self, latent):
                observed["shape"] = latent.shape
                batch_size = latent.shape[:-1]
                return TensorDict(
                    {
                        "action": torch.zeros(*batch_size, 2),
                        "entropy": torch.zeros(*batch_size, 1),
                        "scaled_entropy": torch.zeros(*batch_size, 1),
                    },
                    batch_size=batch_size,
                )

            def _q_value(self, latent, action, **kwargs):
                return torch.zeros(*latent.shape[:-1], 1)

        latent_sequence = torch.randn(3, 5, 7)
        TdMpc2Loss._actor_loss_from_latents(_FakeLoss(), latent_sequence)

        assert observed["shape"] == latent_sequence.shape

    def test_td_targets(self):
        class _FakeLoss:
            discount = torch.tensor(0.97)
            policy_action_key = "action"

            def _policy(self, latent):
                return TensorDict(
                    {"action": torch.zeros(*latent.shape[:-1], 2)},
                    batch_size=latent.shape[:-1],
                )

            def _q_value(self, latent, action, **kwargs):
                return torch.full((*latent.shape[:-1], 1), 2.0)

        next_latent = torch.randn(2, 7)
        reward = torch.tensor([[1.0], [3.0]])
        terminated = torch.tensor([[False], [True]])

        targets = TdMpc2Loss._td_targets(_FakeLoss(), next_latent, reward, terminated)

        torch.testing.assert_close(targets[0], reward[0] + 0.97 * 2.0)
        torch.testing.assert_close(targets[1], reward[1])


if __name__ == "__main__":
    pytest.main()
