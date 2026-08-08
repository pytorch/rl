# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DreamerV3 loss modules and RSSM components.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from _objectives_common import LossModuleTestBase
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)
from torch import nn

from torchrl.data import Unbounded
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import SafeSequential, SymExpTwoHot, WorldModelWrapper
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.model_based import DreamerActor
from torchrl.modules.models.model_based_v3 import (
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)
from torchrl.modules.models.models import MLP
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.dreamer_v3 import (
    _default_bins,
    _match_trailing_dim,
    categorical_kl_balanced,
    categorical_kl_terms,
    symexp,
    symlog,
    two_hot_cross_entropy,
    two_hot_decode,
    two_hot_encode,
)
from torchrl.objectives.utils import SoftUpdate, ValueEstimators
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
                self_.reward_decoder = SymExpTwoHot(num_reward_bins)
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
                if self_.reward_two_hot:
                    tensordict.set(("next", "reward_logits"), reward_pred)
                    reward_pred = self_.reward_decoder(reward_pred)
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
        value_head = TensorDictModule(
            MLP(out_features=out_features, depth=1, num_cells=8),
            in_keys=["state", "belief"],
            out_keys=["state_value" if out_features == 1 else "state_value_logits"],
        )
        if out_features == 1:
            value_model = value_head
        else:
            value_model = TensorDictSequential(
                value_head,
                TensorDictModule(
                    SymExpTwoHot(out_features),
                    in_keys=["state_value_logits"],
                    out_keys=["state_value"],
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

    def test_dreamer_v3_two_hot_official_support(self, device):
        bins = _default_bins(5, device=device)
        expected = torch.tensor(
            [-485165184.0, -22025.4648, 0.0, 22025.4648, 485165184.0],
            device=device,
        )
        torch.testing.assert_close(bins, expected, rtol=1e-6, atol=1e-4)
        assert torch.equal(bins, -bins.flip(0))

        even_bins = _default_bins(4, device=device)
        assert torch.equal(even_bins, -even_bins.flip(0))
        expected_even = symexp(torch.linspace(-20, 20, 4, device=device))
        torch.testing.assert_close(even_bins, expected_even)

    def test_dreamer_v3_two_hot_golden_encode_loss(self, device):
        two_hot = SymExpTwoHot(5).to(device)
        midpoint = (two_hot.bins[1] + two_hot.bins[2]) / 2
        target = torch.stack(
            (
                two_hot.bins[0] - 1,
                midpoint,
                two_hot.bins[-1] + 1,
            )
        )
        encoded = two_hot.encode(target)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )
        torch.testing.assert_close(encoded, expected)

        logits = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0]], device=device)
        loss = two_hot_cross_entropy(logits, midpoint.reshape(1), two_hot.bins)
        torch.testing.assert_close(
            loss, torch.tensor([2.4519143], device=device), rtol=1e-6, atol=1e-6
        )

    def test_dreamer_v3_two_hot_golden_decode(self, device):
        two_hot = SymExpTwoHot(5).to(device)
        uniform = torch.zeros(3, 5, device=device)
        assert torch.equal(two_hot.decode(uniform), torch.zeros(3, device=device))

        logits = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0]], device=device)
        decoded = two_hot.decode(logits)
        torch.testing.assert_close(
            decoded,
            torch.tensor([-36122512.0], device=device),
            rtol=2e-6,
            atol=2.0,
        )

    def test_dreamer_v3_two_hot_module_state_and_compile(self, device):
        two_hot = SymExpTwoHot(5).to(device)
        logits = torch.randn(4, 5, device=device)
        expected = two_hot(logits)
        restored = SymExpTwoHot(5).to(device)
        restored.load_state_dict(two_hot.state_dict())
        torch.testing.assert_close(restored(logits), expected)

        compiled = torch.compile(restored, fullgraph=True)
        torch.testing.assert_close(compiled(logits), expected)

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

    @pytest.mark.parametrize("free_bits", [0.0, 0.5])
    def test_dreamer_v3_kl_balanced_gradients(self, device, free_bits):
        """Both prior_logits and posterior_logits must receive gradients (KL balancing).

        Run with free_bits=0 (no clamp) and free_bits=0.5 (typical) to confirm
        that gradient flow survives the per-categorical free-bits clamp.
        """
        # Larger logits make per-categorical KL exceed any modest free_bits,
        # ensuring the clamp does not zero out the gradient on every element.
        prior_logits = (
            torch.randn(2, 3, self.num_cats, self.num_classes, device=device) * 2.0
        ).requires_grad_(True)
        posterior_logits = (
            torch.randn(2, 3, self.num_cats, self.num_classes, device=device) * 2.0
        ).requires_grad_(True)
        kl = categorical_kl_balanced(
            posterior_logits, prior_logits, alpha=0.8, free_bits=free_bits
        )
        kl.backward()
        assert (
            prior_logits.grad is not None and prior_logits.grad.norm() > 0
        ), "prior_logits has no gradient - KL balancing broken"
        assert (
            posterior_logits.grad is not None and posterior_logits.grad.norm() > 0
        ), "posterior_logits has no gradient - KL balancing broken"

    def test_dreamer_v3_kl_balanced_free_bits_clamp(self, device):
        """When the per-categorical KL is below ``free_bits``, the loss is the
        clamp value and its gradient is zero. When most categoricals are above,
        the gradient must still flow (per-categorical clamp, not mean clamp)."""
        # Two near-identical distributions: KL is essentially zero and gets
        # clamped to free_bits => gradient must be exactly zero everywhere.
        base = torch.randn(2, 3, self.num_cats, self.num_classes, device=device)
        prior_logits = base.clone().requires_grad_(True)
        posterior_logits = base.clone().requires_grad_(True)
        free_bits = 0.5
        kl = categorical_kl_balanced(
            posterior_logits, prior_logits, alpha=0.8, free_bits=free_bits
        )
        # Loss equals the clamp floor: alpha * fb + (1 - alpha) * fb = fb.
        assert kl.item() == pytest.approx(free_bits, abs=1e-5)
        kl.backward()
        assert prior_logits.grad.abs().max().item() == pytest.approx(0.0, abs=1e-6)
        assert posterior_logits.grad.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_dreamer_v3_reference_kl_fixture_and_gradients(self, device):
        posterior_logits = torch.tensor(
            [[[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0]]],
            device=device,
            requires_grad=True,
        )
        prior_logits = torch.tensor(
            [[[0.0, 1.0, -1.0], [1.0, -0.5, 0.5]]],
            device=device,
            requires_grad=True,
        )

        dynamics, representation = categorical_kl_terms(
            posterior_logits,
            prior_logits,
            free_nats=0.0,
            unimix=0.01,
        )
        assert dynamics.item() == pytest.approx(1.9163513, abs=1e-6)
        assert representation.item() == pytest.approx(1.9163513, abs=1e-6)

        dynamics.backward(retain_graph=True)
        assert posterior_logits.grad is None
        assert prior_logits.grad is not None and prior_logits.grad.norm() > 0
        prior_logits.grad = None
        representation.backward()
        assert posterior_logits.grad is not None and posterior_logits.grad.norm() > 0
        assert prior_logits.grad is None

    def test_dreamer_v3_reference_kl_aggregates_before_free_nats(self, device):
        logits = torch.randn(3, 4, 8, device=device, requires_grad=True)
        dynamics, representation = categorical_kl_terms(
            logits,
            logits,
            free_nats=1.0,
            unimix=0.01,
        )
        assert dynamics.item() == pytest.approx(1.0)
        assert representation.item() == pytest.approx(1.0)

    def test_dreamer_v3_model_loss_reference_kl_keys(self, device):
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            kl_mode="separate",
            lambda_dynamic=1.0,
            lambda_representation=0.1,
            unimix=0.01,
            free_bits=0.0,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_model_kl" not in loss_td.keys()
        assert "loss_model_dynamic" in loss_td.keys()
        assert "loss_model_representation" in loss_td.keys()
        dynamic = loss_td["loss_model_dynamic"]
        representation = loss_td["loss_model_representation"]
        assert dynamic.shape == torch.Size([1])
        assert representation.shape == torch.Size([1])
        (dynamic + representation).backward()

    def test_dreamer_v3_model_tensor_keys(self, device):
        world_model = self._create_world_model()
        loss_fn = DreamerV3ModelLoss(world_model, num_reward_bins=self.num_reward_bins)
        default_keys = {
            "reward": "reward",
            "reward_logits": "reward_logits",
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

    def test_dreamer_v3_continuation_lambda_and_weights(self, device):
        class _ConstantContinuation(nn.Module):
            def forward(self_, state, belief):
                return torch.full_like(state[..., :1], 0.5)

        continuation_model = TensorDictModule(
            _ConstantContinuation(),
            in_keys=["state", "belief"],
            out_keys=["continuation"],
        ).to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            value_model,
            self._create_mb_env().to(device),
            continuation_model=continuation_model,
            imagination_horizon=3,
            discount_loss=True,
        )
        loss_module.make_value_estimator(ValueEstimators.TDLambda, gamma=1.0, lmbda=0.5)

        reward = torch.tensor([[[1.0], [2.0], [3.0]]], device=device)
        value = torch.tensor([[[10.0], [20.0], [30.0]]], device=device)
        continuation = torch.full_like(reward, 0.5)
        torch.testing.assert_close(
            loss_module.lambda_target(reward, value, continuation),
            torch.tensor([[[6.375], [11.5], [18.0]]], device=device),
        )

        _, fake_data = loss_module(self._create_actor_data().to(device).reshape(-1))
        expected_weight = torch.tensor([1.0, 0.5, 0.25], device=device)
        torch.testing.assert_close(
            fake_data["discount_weight"][0, :, 0], expected_weight
        )
        torch.testing.assert_close(
            fake_data["next", "continuation"],
            torch.full_like(fake_data["next", "continuation"], 0.5),
        )

        value_loss = DreamerV3ValueLoss(
            value_model,
            discount_loss=True,
            actor_loss=loss_module,
        )
        value_loss(fake_data.detach())

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

    def test_dreamer_v3_categorical_value_exposes_decoded_value(self, device):
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(
            device
        )
        tensordict = self._create_value_data().to(device)
        value_model(tensordict)
        assert tensordict["state_value_logits"].shape[-1] == self.num_reward_bins
        assert tensordict["state_value"].shape[-1] == 1

        actor_loss = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            value_model,
            self._create_mb_env().to(device),
            imagination_horizon=3,
        )
        actor_loss.make_value_estimator(ValueEstimators.TDLambda)
        loss_td, fake_data = actor_loss(
            self._create_actor_data().to(device).reshape(-1)
        )
        assert loss_td["loss_actor"].ndim == 0
        assert fake_data["lambda_target"].shape[-1] == 1

    def test_dreamer_v3_legacy_logits_keys_warn(self, device):
        class LegacyWorldModel(nn.Module):
            def __init__(self_, world_model):
                super().__init__()
                self_.world_model = world_model

            def forward(self_, tensordict):
                tensordict = self_.world_model(tensordict)
                logits = tensordict.pop(("next", "reward_logits"))
                tensordict.set(("next", "reward"), logits)
                return tensordict

        world_model = LegacyWorldModel(self._create_world_model()).to(device)
        model_loss = DreamerV3ModelLoss(
            world_model, num_reward_bins=self.num_reward_bins
        )
        with pytest.warns(DeprecationWarning, match="removed in v0.16"):
            model_loss(self._create_world_model_data().to(device))

        legacy_value = TensorDictModule(
            MLP(out_features=self.num_reward_bins, depth=1, num_cells=8),
            in_keys=["state", "belief"],
            out_keys=["state_value"],
        ).to(device)
        value_loss = DreamerV3ValueLoss(
            legacy_value,
            value_loss="two_hot",
            num_value_bins=self.num_reward_bins,
        )
        with pytest.warns(DeprecationWarning, match="removed in v0.16"):
            value_loss(self._create_value_data().to(device))

    def test_dreamer_v3_nested_logits_keys(self, device):
        class NestedWorldModel(nn.Module):
            def __init__(self_, world_model):
                super().__init__()
                self_.world_model = world_model

            def forward(self_, tensordict):
                tensordict = self_.world_model(tensordict)
                tensordict.rename_key_(
                    ("next", "reward_logits"),
                    ("next", "predictions", "reward_logits"),
                )
                return tensordict

        model_loss = DreamerV3ModelLoss(
            NestedWorldModel(self._create_world_model()).to(device),
            num_reward_bins=self.num_reward_bins,
        )
        model_loss.set_keys(reward_logits=("predictions", "reward_logits"))
        model_loss(self._create_world_model_data().to(device))

        value_model = TensorDictSequential(
            TensorDictModule(
                MLP(out_features=self.num_reward_bins, depth=1, num_cells=8),
                in_keys=["state", "belief"],
                out_keys=[("predictions", "value_logits")],
            ),
            TensorDictModule(
                SymExpTwoHot(self.num_reward_bins),
                in_keys=[("predictions", "value_logits")],
                out_keys=[("predictions", "value")],
            ),
        ).to(device)
        value_loss = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            num_value_bins=self.num_reward_bins,
        )
        value_loss.set_keys(
            value=("predictions", "value"),
            value_logits=("predictions", "value_logits"),
        )
        value_loss(self._create_value_data().to(device))

    def test_dreamer_v3_sota_shares_imagination_parameters(self, device):
        repo_root = Path(__file__).parents[2]
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/dreamer_v3.py",
            run_name="dreamer_v3_test",
        )
        cfg = OmegaConf.load(repo_root / "sota-implementations/dreamer_v3/config.yaml")
        cfg.networks.num_reward_bins = self.num_reward_bins
        (world_model, prior, reward_head, reward_decoder, continuation_head,) = example[
            "build_world_model"
        ](cfg=cfg, obs_dim=3, action_dim=self.action_dim)
        imagination_model = example["build_imagination_model"](
            prior_net=prior,
            reward_net=reward_head,
            reward_decoder=reward_decoder,
        ).to(device)
        continuation_model = example["build_continuation_model"](
            continuation_net=continuation_head
        ).to(device)
        world_model = world_model.to(device)
        observation = torch.tensor(
            [[[0.0, 1.0, -3.0], [2.0, -1.0, 0.5]]], device=device
        )
        world_input = TensorDict(
            {
                "state": torch.zeros(1, 2, self.state_dim, device=device),
                "belief": torch.zeros(1, 2, cfg.networks.rnn_hidden_dim, device=device),
                "action": torch.zeros(1, 2, self.action_dim, device=device),
                "next": {"observation": observation},
            },
            [1, 2],
        )
        world_model(world_input)
        torch.testing.assert_close(
            world_input["next", "symlog_observation"], symlog(observation)
        )
        shared_parameters = tuple(prior.parameters()) + tuple(reward_head.parameters())
        world_parameters = tuple(world_model.parameters())
        imagination_parameters = tuple(imagination_model.parameters())
        assert all(
            any(parameter is candidate for candidate in world_parameters)
            and any(parameter is candidate for candidate in imagination_parameters)
            for parameter in shared_parameters
        )
        assert all(
            any(parameter is candidate for candidate in world_parameters)
            and any(
                parameter is candidate for candidate in continuation_model.parameters()
            )
            for parameter in continuation_head.parameters()
        )

        reward_td = TensorDict(
            {
                "state": torch.randn(2, self.state_dim, device=device),
                "belief": torch.randn(2, cfg.networks.rnn_hidden_dim, device=device),
            },
            [2],
        )
        imagination_model.get_reward_operator()(reward_td)
        assert reward_td["reward_logits"].shape == (2, self.num_reward_bins)
        assert reward_td["reward"].shape == (2, 1)
        continuation_model(reward_td)
        assert reward_td["continuation"].shape == (2, 1)

        parameter = nn.Parameter(torch.tensor([3.0, 4.0], device=device))
        parameter.grad = torch.tensor([30.0, 40.0], device=device)
        example["adaptive_grad_clip_"]([parameter], clip=0.3)
        assert parameter.grad.norm().item() == pytest.approx(1.5)

    def test_dreamer_v3_value_invalid_loss_type(self, device):
        value_model = self._create_value_model()
        with pytest.raises(ValueError, match="symlog_mse.*two_hot"):
            DreamerV3ValueLoss(value_model, value_loss="bad_loss_type")

    def test_dreamer_v3_slow_critic_regularization_and_update(self, device):
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(
            device
        )
        loss_module = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            discount_loss=False,
            num_value_bins=self.num_reward_bins,
            slow_critic_regularization=1.0,
        ).to(device)
        updater = SoftUpdate(loss_module, tau=0.02)
        tensordict = self._create_value_data().to(device)

        online_td = tensordict.select(*value_model.in_keys, strict=False)
        with loss_module.value_model_params.to_module(
            loss_module.value_model, preserve_module_state=False
        ):
            loss_module.value_model(online_td)
        target_td = tensordict.select(*value_model.in_keys, strict=False)
        with torch.no_grad(), loss_module.target_value_model_params.to_module(
            loss_module.value_model, preserve_module_state=False
        ):
            loss_module.value_model(target_td)
        expected_slow_loss = two_hot_cross_entropy(
            online_td["state_value_logits"],
            target_td["state_value"].squeeze(-1),
            loss_module.value_bins,
        ).mean()

        loss_td, _ = loss_module(tensordict)
        torch.testing.assert_close(loss_td["value_slow_loss"], expected_slow_loss)
        loss_td["loss_value"].backward()
        assert any(
            parameter.grad is not None
            for parameter in loss_module.value_model_params.values(True, True)
            if parameter.requires_grad
        )
        assert all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in loss_module.target_value_model_params.values(True, True)
        )

        source = next(
            parameter
            for parameter in loss_module.value_model_params.values(True, True)
            if parameter.requires_grad
        )
        target = next(
            parameter
            for parameter in loss_module.target_value_model_params.values(True, True)
            if parameter.shape == source.shape
        )
        target_before = target.clone()
        with torch.no_grad():
            source.add_(1.0)
        updater.step()
        torch.testing.assert_close(target, target_before.lerp(source.detach(), 0.02))

    def test_dreamer_v3_slow_critic_checkpoint_and_online_bootstrap(self, device):
        value_model = self._create_value_model(out_features=self.num_reward_bins).to(
            device
        )
        actor_loss = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            value_model,
            self._create_mb_env().to(device),
        )
        value_loss = DreamerV3ValueLoss(
            value_model,
            value_loss="two_hot",
            num_value_bins=self.num_reward_bins,
            actor_loss=actor_loss,
            slow_critic_regularization=1.0,
        ).to(device)
        SoftUpdate(value_loss, tau=0.02)

        actor_parameters = tuple(actor_loss.__dict__["value_model"].parameters())
        online_parameters = tuple(value_loss.value_model_params.values(True, True))
        target_parameters = tuple(
            value_loss.target_value_model_params.values(True, True)
        )
        assert all(
            any(parameter is online for online in online_parameters)
            for parameter in actor_parameters
        )
        assert all(
            all(parameter is not target for target in target_parameters)
            for parameter in actor_parameters
        )

        checkpoint = {
            key: value.detach().clone()
            for key, value in value_loss.state_dict().items()
        }
        target_keys = [key for key in checkpoint if key.startswith("target_value")]
        assert target_keys
        expected_target = tuple(parameter.clone() for parameter in target_parameters)
        with torch.no_grad():
            for parameter in target_parameters:
                parameter.add_(10.0)
        value_loss.load_state_dict(checkpoint)
        for actual, expected in zip(
            value_loss.target_value_model_params.values(True, True),
            expected_target,
        ):
            torch.testing.assert_close(actual, expected)

    # ------------------------------------------------------------------ #
    # RSSM component tests
    # ------------------------------------------------------------------ #

    def test_rssm_posterior_v3_forward_shapes_and_grads(self, device):
        B = 4
        obs_embed_dim = 16
        posterior = RSSMPosteriorV3(
            hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            rnn_hidden_dim=self.rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
        ).to(device)

        belief = torch.randn(B, self.rnn_hidden_dim, device=device, requires_grad=True)
        obs_embed = torch.randn(B, obs_embed_dim, device=device, requires_grad=True)

        logits, state = posterior(belief, obs_embed)
        assert logits.shape == (B, self.num_cats, self.num_classes)
        assert state.shape == (B, self.state_dim)
        # one-hot forward: each categorical sums to 1
        state_grid = state.view(B, self.num_cats, self.num_classes)
        assert torch.allclose(
            state_grid.sum(-1), torch.ones(B, self.num_cats, device=device), atol=1e-5
        )

        # Straight-through: gradients must flow back through logits to belief/obs.
        # NOTE: ``state.sum()`` is mathematically constant w.r.t. the logits — every
        # row of the softmax inside the STE sums to 1, so any sum-reduction over
        # the full ``state`` has zero gradient through softmax (uniform incoming
        # gradient cancels exactly in the softmax Jacobian). Whether the resulting
        # belief/obs grads are exactly 0.0 or a tiny float-roundoff residue depends
        # on the runtime — leading to flakiness across Python/torch versions.
        # Use random per-element weights so the gradient signal through softmax
        # is non-degenerate.
        torch.manual_seed(0)
        weights = torch.randn_like(state)
        (state * weights).sum().backward()
        assert belief.grad is not None and belief.grad.abs().sum() > 0
        assert obs_embed.grad is not None and obs_embed.grad.abs().sum() > 0

    def test_rssm_rollout_v3_forward(self, device):
        B, T = 2, 4
        obs_embed_dim = 12
        action_dim = self.action_dim

        prior_net = RSSMPriorV3(
            action_shape=torch.Size([action_dim]),
            hidden_dim=self.rnn_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            action_dim=action_dim,
        ).to(device)
        posterior_net = RSSMPosteriorV3(
            hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            rnn_hidden_dim=self.rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
        ).to(device)

        rssm_prior = TensorDictModule(
            prior_net,
            in_keys=["state", "belief", "action"],
            out_keys=[
                ("next", "prior_logits"),
                ("next", "state"),
                ("next", "belief"),
            ],
        )
        rssm_posterior = TensorDictModule(
            posterior_net,
            in_keys=[("next", "belief"), ("next", "encoded_latents")],
            out_keys=[("next", "posterior_logits"), ("next", "state")],
        )
        rollout = RSSMRolloutV3(rssm_prior, rssm_posterior)

        td = TensorDict(
            {
                "state": torch.zeros(B, T, self.state_dim, device=device),
                "belief": torch.zeros(B, T, self.rnn_hidden_dim, device=device),
                "action": torch.randn(B, T, action_dim, device=device),
                "next": {
                    "encoded_latents": torch.randn(B, T, obs_embed_dim, device=device),
                },
            },
            [B, T],
        )
        out = rollout(td)
        assert out.shape == (B, T)
        prior_logits = out.get(("next", "prior_logits"))
        post_logits = out.get(("next", "posterior_logits"))
        assert prior_logits.shape == (B, T, self.num_cats, self.num_classes)
        assert post_logits.shape == (B, T, self.num_cats, self.num_classes)

        reset = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        reset[:, 2] = True
        td_a = td.clone().set("is_init", reset)
        td_b = td.clone().set("is_init", reset)
        td_b["action"][:, :2] = torch.randn_like(td_b["action"][:, :2])
        td_b["next", "encoded_latents"][:, :2] = torch.randn_like(
            td_b["next", "encoded_latents"][:, :2]
        )
        torch.manual_seed(0)
        out_a = rollout(td_a)
        torch.manual_seed(0)
        out_b = rollout(td_b)
        for key in (
            ("next", "prior_logits"),
            ("next", "posterior_logits"),
            ("next", "state"),
            ("next", "belief"),
        ):
            torch.testing.assert_close(out_a[key][:, 2:], out_b[key][:, 2:])

    # ------------------------------------------------------------------ #
    # Coverage for previously untested branches
    # ------------------------------------------------------------------ #

    def test_dreamer_v3_model_loss_reco_l1(self, device):
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            reco_loss="l1",
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_model_reco" in loss_td.keys()
        loss_td["loss_model_reco"].backward()

    def test_dreamer_v3_model_loss_no_continue_default(self, device):
        """With ``lambda_continue=0`` (default), no ``loss_model_continue`` key is emitted."""
        tensordict = self._create_world_model_data().to(device)
        world_model = self._create_world_model(reward_two_hot=True).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_model_continue" not in loss_td.keys()

    def test_dreamer_v3_model_loss_continue(self, device):
        """Exercises the lambda_continue > 0 branch with a continue head."""
        B, T = 2, 3
        base_td = self._create_world_model_data().to(device)

        class _StubWithContinue(nn.Module):
            def __init__(self_, base):
                super().__init__()
                self_.base = base
                self_.continue_head = nn.Linear(
                    self.state_dim + self.rnn_hidden_dim, 1
                ).to(device)

            def forward(self_, td):
                td = self_.base(td)
                cat_in = torch.cat([td["state"], td["belief"]], dim=-1)
                td.set(
                    ("next", "continue_pred"),
                    self_.continue_head(cat_in).squeeze(-1),
                )
                return td

        world_model = _StubWithContinue(self._create_world_model()).to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            lambda_continue=1.0,
            continue_target_scale=0.75,
            num_reward_bins=self.num_reward_bins,
        )
        # state/belief in the default data are zeros, so the continue_head
        # weight gradient is always zero (W*0 = 0). Use non-zero inputs so
        # the BCE gradient reaches both weight and bias.
        base_td["state"] = torch.randn_like(base_td["state"])
        base_td["belief"] = torch.randn_like(base_td["belief"])
        # seed a mix of done / not-done so the BCE target is non-degenerate
        base_td["next", "done"][0, 0] = True
        loss_td, model_out = loss_module(base_td)
        assert "loss_model_continue" in loss_td.keys()
        target = (~base_td["next", "terminated"]).float() * 0.75
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            model_out["next", "continue_pred"], target
        )
        torch.testing.assert_close(loss_td["loss_model_continue"].squeeze(), expected)
        loss_td["loss_model_continue"].backward()
        assert world_model.continue_head.weight.grad.abs().sum() > 0
        assert base_td.shape == (B, T)

    def _create_actor_model_with_log_prob(self):
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
                return_log_prob=True,
                log_prob_key="action_log_prob",
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

    def test_dreamer_v3_actor_loss_reinforce(self, device):
        """REINFORCE branch: log_prob * sg(advantage) path must be exercised."""
        tensordict = self._create_actor_data().to(device)
        mb_env = self._create_mb_env().to(device)
        actor_model = self._create_actor_model_with_log_prob().to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            mb_env,
            imagination_horizon=3,
            use_reinforce=True,
        )
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_td, _ = loss_module(tensordict.reshape(-1))
        assert "loss_actor" in loss_td.keys()
        loss_td["loss_actor"].backward()
        actor_grad = sum(
            p.grad.pow(2).sum().item()
            for p in actor_model.parameters()
            if p.grad is not None
        )
        assert actor_grad > 0, "REINFORCE path produced no actor gradients"

    def test_dreamer_v3_reinforce_return_normalization(self, device):
        actor_model = self._create_actor_model_with_log_prob().to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            self._create_mb_env().to(device),
            imagination_horizon=3,
            discount_loss=False,
            entropy_bonus=0.0,
            use_reinforce=True,
        ).to(device)
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_module.return_low.fill_(-2.0)
        loss_module.return_high.fill_(8.0)
        loss_module.eval()

        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        baseline_td = fake_data.select(*value_model.in_keys, strict=False)
        value_model(baseline_td)
        advantage = (fake_data["lambda_target"] - baseline_td["state_value"]).detach()
        log_prob = _match_trailing_dim(
            fake_data["action_log_prob"], fake_data["lambda_target"]
        )
        expected = -(log_prob * advantage / 10.0).sum((-2, -1)).mean()
        torch.testing.assert_close(loss_td["loss_actor"], expected)
        torch.testing.assert_close(
            loss_td["return_scale"], torch.tensor(10.0, device=device)
        )

        compiled_scale = torch.compile(loss_module._return_scale, fullgraph=True)
        torch.testing.assert_close(
            compiled_scale(fake_data["lambda_target"]),
            torch.tensor(10.0, device=device),
        )

    def test_dreamer_v3_return_statistics_checkpoint(self, device):
        loss_module = DreamerV3ActorLoss(
            self._create_actor_model_with_log_prob().to(device),
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
            imagination_horizon=3,
            entropy_bonus=0.0,
            use_reinforce=True,
            return_normalization_rate=0.01,
        ).to(device)
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        expected_low, expected_high = torch.quantile(
            fake_data["lambda_target"].detach(),
            torch.tensor([0.05, 0.95], device=device),
        )
        torch.testing.assert_close(loss_module.return_low, 0.01 * expected_low)
        torch.testing.assert_close(loss_module.return_high, 0.01 * expected_high)
        torch.testing.assert_close(loss_td["return_low"], loss_module.return_low)
        torch.testing.assert_close(loss_td["return_high"], loss_module.return_high)
        torch.testing.assert_close(
            loss_td["return_scale"],
            (loss_module.return_high - loss_module.return_low).clamp_min(1.0),
        )

        checkpoint = {
            key: value.detach().clone()
            for key, value in loss_module.state_dict().items()
        }
        expected_statistics = (
            loss_module.return_low.clone(),
            loss_module.return_high.clone(),
        )
        loss_module.return_low.zero_()
        loss_module.return_high.zero_()
        loss_module.load_state_dict(checkpoint)
        torch.testing.assert_close(loss_module.return_low, expected_statistics[0])
        torch.testing.assert_close(loss_module.return_high, expected_statistics[1])

        loss_module.eval()
        loss_module(self._create_actor_data().to(device).reshape(-1))
        torch.testing.assert_close(loss_module.return_low, expected_statistics[0])
        torch.testing.assert_close(loss_module.return_high, expected_statistics[1])

    def test_dreamer_v3_value_loss_sync_gamma(self, device):
        """sync_gamma_with_actor_loss must pull gamma from the actor's value estimator."""
        mb_env = self._create_mb_env().to(device)
        actor_model = self._create_actor_model().to(device)
        value_model = self._create_value_model().to(device)
        actor_loss = DreamerV3ActorLoss(actor_model, value_model, mb_env)
        actor_loss.make_value_estimator(ValueEstimators.TDLambda, gamma=0.95, lmbda=0.9)

        value_loss = DreamerV3ValueLoss(value_model, gamma=0.99)
        assert value_loss.gamma == 0.99
        value_loss.sync_gamma_with_actor_loss(actor_loss)
        assert value_loss.gamma == pytest.approx(0.95)

    # ------------------------------------------------------------------ #
    # End-to-end model-loss test with the real RSSM pair (no stub)
    # ------------------------------------------------------------------ #

    def test_dreamer_v3_model_loss_real_rssm(self, device):
        """DreamerV3ModelLoss against the real RSSMPriorV3 + RSSMPosteriorV3 wiring."""
        B, T = 2, 3
        obs_embed_dim = 16

        prior_net = RSSMPriorV3(
            action_shape=torch.Size([self.action_dim]),
            hidden_dim=self.rnn_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            action_dim=self.action_dim,
        ).to(device)
        posterior_net = RSSMPosteriorV3(
            hidden_dim=self.rnn_hidden_dim,
            num_categoricals=self.num_cats,
            num_classes=self.num_classes,
            rnn_hidden_dim=self.rnn_hidden_dim,
            obs_embed_dim=obs_embed_dim,
        ).to(device)

        class _EndToEndWorldModel(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.encoder = nn.Sequential(
                    nn.LazyConv2d(8, 4, stride=2),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.LazyLinear(obs_embed_dim),
                )
                self_.decoder = nn.Sequential(
                    nn.LazyLinear(3 * 64 * 64),
                    nn.Unflatten(-1, (3, 64, 64)),
                )
                self_.reward_head = nn.LazyLinear(self.num_reward_bins)
                self_.reward_decoder = SymExpTwoHot(self.num_reward_bins)
                self_.prior = prior_net
                self_.posterior = posterior_net
                self_.num_cats = self.num_cats
                self_.num_classes = self.num_classes

            def forward(self_, td):
                B_, T_ = td.shape
                state = td["state"]
                belief = td["belief"]
                action = td["action"]

                prior_logits, _, next_belief = self_.prior(
                    state.flatten(0, 1), belief.flatten(0, 1), action.flatten(0, 1)
                )
                prior_logits = prior_logits.view(
                    B_, T_, self_.num_cats, self_.num_classes
                )
                next_belief = next_belief.view(B_, T_, -1)

                next_pixels = td["next", "pixels"]
                pix_flat = next_pixels.flatten(0, 1)
                obs_embed = self_.encoder(pix_flat)

                post_logits, post_state = self_.posterior(
                    next_belief.flatten(0, 1), obs_embed
                )
                post_logits = post_logits.view(
                    B_, T_, self_.num_cats, self_.num_classes
                )

                reco_flat = self_.decoder(post_state)
                reco_pixels = reco_flat.view(B_, T_, 3, 64, 64)

                reward_pred = self_.reward_head(post_state).view(
                    B_, T_, self.num_reward_bins
                )

                td.set(("next", "prior_logits"), prior_logits)
                td.set(("next", "posterior_logits"), post_logits)
                td.set(("next", "reco_pixels"), reco_pixels)
                td.set(("next", "reward_logits"), reward_pred)
                td.set(("next", "reward"), self_.reward_decoder(reward_pred))
                return td

        world_model = _EndToEndWorldModel().to(device)
        tensordict = self._create_world_model_data().to(device)
        # warm-up lazy layers
        with torch.no_grad():
            world_model(tensordict.clone())

        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
        )
        loss_td, _ = loss_module(tensordict)
        total = (
            loss_td["loss_model_kl"]
            + loss_td["loss_model_reco"]
            + loss_td["loss_model_reward"]
        )
        total.backward()
        # both the real prior and posterior nets must receive gradients
        prior_grad = sum(
            p.grad.pow(2).sum().item()
            for p in prior_net.parameters()
            if p.grad is not None
        )
        posterior_grad = sum(
            p.grad.pow(2).sum().item()
            for p in posterior_net.parameters()
            if p.grad is not None
        )
        assert prior_grad > 0, "Real prior received no gradient"
        assert posterior_grad > 0, "Real posterior received no gradient"
        assert B == 2 and T == 3
