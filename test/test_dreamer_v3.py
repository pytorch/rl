# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DreamerV3 loss modules and RSSM components.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from test_objectives import LossModuleTestBase
from torch import nn

from torchrl.data import Unbounded
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import SafeSequential, WorldModelWrapper
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.model_based import DreamerActor
from torchrl.modules.models.model_based_v3 import RSSMPriorV3
from torchrl.modules.models.models import MLP
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.dreamer_v3 import (
    _default_bins,
    categorical_kl_balanced,
    symexp,
    symlog,
    two_hot_decode,
    two_hot_encode,
)
from torchrl.objectives.utils import ValueEstimators
from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv


@pytest.mark.parametrize("device", get_default_devices())
class TestDreamerV3(LossModuleTestBase):  # type: ignore[misc]
    img_size = (64, 64)
    # Compact sizes to keep tests fast
    num_cats = 4
    num_classes = 4
    state_dim = num_cats * num_classes  # 16
    rnn_hidden_dim = 8
    action_dim = 3
    num_reward_bins = 16  # small for tests; paper uses 255

    def _create_world_model_data(self):
        B, T = 2, 3
        return TensorDict(
            {
                "state": torch.zeros(B, T, self.state_dim),
                "belief": torch.zeros(B, T, self.rnn_hidden_dim),
                "pixels": torch.rand(B, T, 3, *self.img_size),
                "action": torch.randn(B, T, self.action_dim),
                "next": {
                    "pixels": torch.rand(B, T, 3, *self.img_size),
                    "reward": torch.randn(B, T, 1),
                    "done": torch.zeros(B, T, dtype=torch.bool),
                    "terminated": torch.zeros(B, T, dtype=torch.bool),
                },
            },
            [B, T],
        )

    def _create_actor_data(self):
        B, T = 2, 3
        return TensorDict(
            {
                "state": torch.randn(B, T, self.state_dim),
                "belief": torch.randn(B, T, self.rnn_hidden_dim),
                "reward": torch.randn(B, T, 1),
            },
            [B, T],
        )

    def _create_value_data(self):
        N = 6  # 2 * 3
        return TensorDict(
            {
                "state": torch.randn(N, self.state_dim),
                "belief": torch.randn(N, self.rnn_hidden_dim),
                "lambda_target": torch.randn(N, 1),
            },
            [N],
        )

    def _create_world_model(self, reward_two_hot=True):
        """Minimal stub world model that produces all keys DreamerV3ModelLoss expects."""

        class _StubWorldModel(nn.Module):
            def __init__(
                self_,
                num_cats,
                num_classes,
                rnn_hidden_dim,
                num_reward_bins,
                reward_two_hot,
            ):
                super().__init__()
                state_dim = num_cats * num_classes
                # pixel encoder → reco
                self_.encoder = nn.LazyConv2d(8, 4, stride=2)
                self_.decoder = nn.LazyConvTranspose2d(3, 4, stride=2)
                # prior / posterior MLP stubs
                self_.prior_net = nn.Linear(
                    state_dim + rnn_hidden_dim, num_cats * num_classes
                )
                self_.posterior_net = nn.LazyLinear(num_cats * num_classes)
                # reward head
                out_r = num_reward_bins if reward_two_hot else 1
                self_.reward_net = nn.LazyLinear(out_r)
                self_.num_cats = num_cats
                self_.num_classes = num_classes
                self_.reward_two_hot = reward_two_hot

            def forward(self_, tensordict):
                B, T = tensordict.shape
                state = tensordict["state"]  # [B, T, state_dim]
                belief = tensordict["belief"]  # [B, T, rnn_hidden]

                # prior logits
                prior_in = torch.cat([state, belief], dim=-1)
                prior_flat = self_.prior_net(prior_in)
                prior_logits = prior_flat.view(B, T, self_.num_cats, self_.num_classes)

                # posterior logits (lazy — accepts anything)
                post_flat = self_.posterior_net(prior_in)
                posterior_logits = post_flat.view(
                    B, T, self_.num_cats, self_.num_classes
                )

                # reco pixels (tiny decode — just needs right shape)
                next_pixels = tensordict["next", "pixels"]  # [B, T, 3, H, W]
                flat_pix = next_pixels.flatten(0, 1)  # [B*T, 3, H, W]
                enc = torch.relu(self_.encoder(flat_pix))
                reco_flat = torch.sigmoid(self_.decoder(enc))
                _, C, H, W = reco_flat.shape
                reco_pixels = reco_flat.view(B, T, C, H, W)

                # reward prediction
                reward_in = torch.cat([state, belief], dim=-1)
                reward_pred = self_.reward_net(reward_in)  # [B, T, out_r]

                tensordict.set(("next", "prior_logits"), prior_logits)
                tensordict.set(("next", "posterior_logits"), posterior_logits)
                tensordict.set(("next", "reco_pixels"), reco_pixels)
                tensordict.set(("next", "reward"), reward_pred)
                return tensordict

        stub = _StubWorldModel(
            self.num_cats,
            self.num_classes,
            self.rnn_hidden_dim,
            self.num_reward_bins,
            reward_two_hot,
        )
        # warm-up lazy layers
        with torch.no_grad():
            stub(self._create_world_model_data())
        return stub

    def _create_mb_env(self):
        mock_env = TransformedEnv(
            ContinuousActionConvMockEnv(pixel_shape=[3, *self.img_size])
        )
        default_dict = {
            "state": Unbounded(self.state_dim),
            "belief": Unbounded(self.rnn_hidden_dim),
        }
        mock_env.append_transform(
            TensorDictPrimer(random=False, default_value=0, **default_dict)
        )
        rssm_prior = RSSMPriorV3(
            action_spec=mock_env.action_spec,
            hidden_dim=self.rnn_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            action_dim=mock_env.action_spec.shape[0],
        )
        transition_model = SafeSequential(
            TensorDictModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=["_", "state", "belief"],
            )
        )
        reward_model = TensorDictModule(
            MLP(out_features=1, depth=1, num_cells=8),
            in_keys=["state", "belief"],
            out_keys=["reward"],
        )
        model_based_env = DreamerEnv(
            world_model=WorldModelWrapper(transition_model, reward_model),
            prior_shape=torch.Size([self.state_dim]),
            belief_shape=torch.Size([self.rnn_hidden_dim]),
        )
        model_based_env.set_specs_from_env(mock_env)
        with torch.no_grad():
            model_based_env.rollout(3)
        return model_based_env

    def _create_actor_model(self):
        mock_env = TransformedEnv(
            ContinuousActionConvMockEnv(pixel_shape=[3, *self.img_size])
        )
        actor_module = DreamerActor(
            out_features=mock_env.action_spec.shape[0],
            depth=1,
            num_cells=8,
        )
        actor_model = ProbabilisticTensorDictSequential(
            TensorDictModule(
                actor_module,
                in_keys=["state", "belief"],
                out_keys=["loc", "scale"],
            ),
            ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                default_interaction_type=InteractionType.RANDOM,
                distribution_class=TanhNormal,
            ),
        )
        with torch.no_grad():
            td = TensorDict(
                {
                    "state": torch.randn(1, 2, self.state_dim),
                    "belief": torch.randn(1, 2, self.rnn_hidden_dim),
                },
                batch_size=[1],
            )
            actor_model(td)
        return actor_model

    def _create_value_model(self, out_features=1):
        value_model = TensorDictModule(
            MLP(out_features=out_features, depth=1, num_cells=8),
            in_keys=["state", "belief"],
            out_keys=["state_value"],
        )
        with torch.no_grad():
            td = TensorDict(
                {
                    "state": torch.randn(1, 2, self.state_dim),
                    "belief": torch.randn(1, 2, self.rnn_hidden_dim),
                },
                batch_size=[1],
            )
            value_model(td)
        return value_model

    # ------------------------------------------------------------------ #
    # Required by LossModuleTestBase
    # ------------------------------------------------------------------ #

    def test_reset_parameters_recursive(self, device):
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_fn = DreamerV3ModelLoss(world_model, num_reward_bins=self.num_reward_bins)
        self.reset_parameters_recursive_test(loss_fn)

    # ------------------------------------------------------------------ #
    # Utility tests
    # ------------------------------------------------------------------ #

    def test_dreamer_v3_symlog_invertibility(self, device):
        x = torch.tensor([-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0], device=device)
        reconstructed = symexp(symlog(x))
        assert torch.allclose(
            reconstructed, x, atol=1e-4
        ), f"symexp(symlog(x)) ≠ x: {reconstructed}"

    def test_dreamer_v3_two_hot_roundtrip(self, device):
        bins = _default_bins(self.num_reward_bins).to(device)
        vals = torch.linspace(-15.0, 15.0, 9, device=device)
        encoded = two_hot_encode(vals, bins)
        # Each row must be a valid probability distribution
        assert torch.allclose(encoded.sum(-1), torch.ones(9, device=device), atol=1e-5)
        decoded = two_hot_decode(torch.log(encoded + 1e-8), bins)
        assert torch.allclose(
            decoded, vals, atol=0.5
        ), f"two_hot round-trip error too large: {(decoded - vals).abs().max()}"

    # ------------------------------------------------------------------ #
    # World model loss tests
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("reward_two_hot", [True, False])
    @pytest.mark.parametrize(
        "lambda_kl,lambda_reco,lambda_reward", [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)]
    )
    def test_dreamer_v3_model_loss_output_keys(
        self, device, reward_two_hot, lambda_kl, lambda_reco, lambda_reward
    ):
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=reward_two_hot).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            lambda_kl=lambda_kl,
            lambda_reco=lambda_reco,
            lambda_reward=lambda_reward,
            reward_two_hot=reward_two_hot,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        for key in ("loss_model_kl", "loss_model_reco", "loss_model_reward"):
            assert key in loss_td.keys(), f"Missing {key}"
            assert loss_td[key].shape == torch.Size([1])

    def test_dreamer_v3_model_loss_backward(self, device):
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        total_loss = sum(
            loss_td[k]
            for k in ("loss_model_kl", "loss_model_reco", "loss_model_reward")
        )
        total_loss.backward()
        grad_total = sum(
            p.grad.pow(2).sum().item()
            for p in loss_module.parameters()
            if p.grad is not None
        )
        assert grad_total > 0, "All gradients are zero after backward"
        for name, p in loss_module.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf grad in {name}"

    def test_dreamer_v3_kl_balanced_gradients(self, device):
        """Both prior_logits and posterior_logits must receive gradients (KL balancing)."""
        prior_logits = torch.randn(
            2, 3, self.num_cats, self.num_classes, requires_grad=True, device=device
        )
        posterior_logits = torch.randn(
            2, 3, self.num_cats, self.num_classes, requires_grad=True, device=device
        )
        kl = categorical_kl_balanced(
            posterior_logits, prior_logits, alpha=0.8, free_bits=1.0
        )
        kl.backward()
        assert (
            prior_logits.grad is not None and prior_logits.grad.norm() > 0
        ), "prior_logits has no gradient — KL balancing broken"
        assert (
            posterior_logits.grad is not None and posterior_logits.grad.norm() > 0
        ), "posterior_logits has no gradient — KL balancing broken"

    def test_dreamer_v3_model_tensor_keys(self, device):
        world_model = self._create_world_model()
        loss_fn = DreamerV3ModelLoss(world_model, num_reward_bins=self.num_reward_bins)
        default_keys = {
            "reward": "reward",
            "true_reward": "true_reward",
            "prior_logits": "prior_logits",
            "posterior_logits": "posterior_logits",
            "pixels": "pixels",
            "reco_pixels": "reco_pixels",
        }
        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

    # ------------------------------------------------------------------ #
    # Actor loss tests
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("imagination_horizon", [3, 5])
    @pytest.mark.parametrize("discount_loss", [True, False])
    @pytest.mark.parametrize(
        "td_est",
        [ValueEstimators.TD0, ValueEstimators.TD1, ValueEstimators.TDLambda, None],
    )
    def test_dreamer_v3_actor_loss(
        self, device, imagination_horizon, discount_loss, td_est
    ):
        tensordict = self._create_actor_data().to(device)
        mb_env = self._create_mb_env().to(device)
        actor_model = self._create_actor_model().to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            mb_env,
            imagination_horizon=imagination_horizon,
            discount_loss=discount_loss,
        )
        if td_est is not None:
            loss_module.make_value_estimator(td_est)
        loss_td, fake_data = loss_module(tensordict.reshape(-1))
        assert "loss_actor" in loss_td.keys()
        assert loss_td["loss_actor"].ndim == 0 or loss_td["loss_actor"].numel() == 1
        loss_td["loss_actor"].backward()
        grad_total = sum(
            p.grad.pow(2).sum().item()
            for p in loss_module.parameters()
            if p.grad is not None
        )
        assert grad_total > 0, "All gradients are zero after actor backward"

    # ------------------------------------------------------------------ #
    # Value loss tests
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("discount_loss", [True, False])
    def test_dreamer_v3_value_loss_symlog_mse(self, device, discount_loss):
        tensordict = self._create_value_data().to(device)
        value_model = self._create_value_model(out_features=1).to(device)
        loss_module = DreamerV3ValueLoss(
            value_model,
            value_loss="symlog_mse",
            discount_loss=discount_loss,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_value" in loss_td.keys()
        loss_td["loss_value"].backward()
        grad_total = sum(
            p.grad.pow(2).sum().item()
            for p in loss_module.parameters()
            if p.grad is not None
        )
        assert (
            grad_total > 0
        ), "All gradients are zero after value (symlog_mse) backward"

    @pytest.mark.parametrize("discount_loss", [True, False])
    def test_dreamer_v3_value_loss_two_hot(self, device, discount_loss):
        tensordict = self._create_value_data().to(device)
        # Value model must output logits over bins
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(
            device
        )
        loss_module = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            discount_loss=discount_loss,
            num_value_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_value" in loss_td.keys()
        loss_td["loss_value"].backward()
        grad_total = sum(
            p.grad.pow(2).sum().item()
            for p in loss_module.parameters()
            if p.grad is not None
        )
        assert grad_total > 0, "All gradients are zero after value (two_hot) backward"

    def test_dreamer_v3_value_invalid_loss_type(self, device):
        value_model = self._create_value_model()
        with pytest.raises(ValueError, match="symlog_mse.*two_hot"):
            DreamerV3ValueLoss(value_model, value_loss="bad_loss_type")
