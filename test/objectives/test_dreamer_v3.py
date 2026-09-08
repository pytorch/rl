# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DreamerV3 loss modules and RSSM components.

Reference: https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

import copy
import functools as ft
import importlib.util
import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from _objectives_common import LossModuleTestBase
from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)
from tensordict.utils import assert_close
from torch import nn
from torchrl.checkpoint import Checkpoint, CheckpointRotation
from torchrl.data import (
    Bounded,
    Categorical,
    Composite,
    OneHot,
    SliceSampler,
    StreamingSliceSampler,
    Unbounded,
)
from torchrl.envs import EnvBase
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import SafeSequential, SymExpTwoHot, WorldModelWrapper
from torchrl.modules.distributions.continuous import IndependentNormal, TanhNormal
from torchrl.modules.models.model_based import (
    DreamerActor,
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
    _replay_value_target,
    categorical_kl_balanced,
    categorical_kl_terms,
    symexp,
    symlog,
    two_hot_cross_entropy,
    two_hot_decode,
    two_hot_encode,
)
from torchrl.objectives.utils import HardUpdate, SoftUpdate, ValueEstimators
from torchrl.testing import get_default_devices, PENDULUM_VERSIONED
from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv
from torchrl.trainers import Trainer
from torchrl.trainers.algorithms import DreamerV3Optimizer

_has_hydra = importlib.util.find_spec("hydra") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None
_has_hoptorch = importlib.util.find_spec("hoptorch") is not None
_has_gym = (
    importlib.util.find_spec("gymnasium") is not None
    or importlib.util.find_spec("gym") is not None
)
_compile_backend = "eager" if os.name == "nt" else "inductor"


class _DreamerV3TestEnv(EnvBase):
    def __init__(self, *, seed, env_index, num_envs, pixels=False, discrete=False):
        super().__init__()
        assert seed is not None and 0 <= env_index < num_envs
        self.index = env_index
        self.steps = 0
        self.pixels = pixels
        self.observation_spec = Composite(
            {
                ("sensors", "vector"): Unbounded((3,)),
                ("episode", "milestones"): Categorical(2, shape=(2,), dtype=torch.bool),
            }
        )
        if pixels:
            self.observation_spec["sensors", "image"] = Bounded(
                0, 255, (1, 8, 8), dtype=torch.uint8
            )
        self.action_spec = OneHot(3) if discrete else Bounded(-1, 1, (1,))
        self.reward_spec = Unbounded((1,))
        self.done_spec = Categorical(2, shape=(1,), dtype=torch.bool)

    def _observation(self):
        data = TensorDict(
            {
                ("sensors", "vector"): torch.tensor([self.index, self.steps, 1.0]),
                ("episode", "milestones"): torch.tensor(
                    [self.steps >= 1, self.steps >= 3]
                ),
            },
            [],
        )
        if self.pixels:
            data["sensors", "image"] = torch.full(
                (1, 8, 8), self.steps * 20, dtype=torch.uint8
            )
        return data

    def _reset(self, tensordict=None, **kwargs):
        self.steps = 0
        return self._observation()

    def _step(self, tensordict):
        self.steps += 1
        data = self._observation()
        data["reward"] = torch.ones(1)
        data["done"] = torch.tensor([self.steps >= 3])
        data["terminated"] = data["done"].clone()
        return data

    def _set_seed(self, seed):
        return seed


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

    def _small_sota_config(
        self,
        example_dir: Path,
        *,
        compile_train_step: bool,
        cudagraph_train_step: bool,
        mixed_precision: bool = False,
    ):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(example_dir / "config.yaml")
        cfg.env.name = PENDULUM_VERSIONED()
        cfg.networks.rnn_hidden_dim = 8
        cfg.networks.num_categoricals = 2
        cfg.networks.num_classes = 2
        cfg.networks.num_blocks = 2
        cfg.networks.hidden_dim = 8
        cfg.networks.num_reward_bins = 16
        cfg.networks.num_value_bins = 16
        cfg.networks.encoder_layers = 1
        cfg.networks.decoder_layers = 1
        cfg.networks.reward_layers = 1
        cfg.networks.actor_layers = 1
        cfg.networks.value_layers = 1
        cfg.replay_buffer.batch_size = 2
        cfg.replay_buffer.seq_len = 3
        cfg.optimization.imagination_horizon = 3
        cfg.optimization.continuation_horizon = 3
        cfg.optimization.warmup_steps = 0
        cfg.optimization.mixed_precision = mixed_precision
        cfg.optimization.compile_rssm = "scan"
        cfg.optimization.rssm_scan_unroll = 1
        cfg.optimization.compile_train_step = compile_train_step
        cfg.optimization.cudagraph_train_step = cudagraph_train_step
        return cfg

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
        logits = torch.linspace(-0.5, 0.5, 20, device=device).reshape(4, 5)
        expected = two_hot(logits)
        restored = SymExpTwoHot(5).to(device)
        restored.load_state_dict(two_hot.state_dict())
        torch.testing.assert_close(restored(logits), expected)

        compiled = torch.compile(restored, backend=_compile_backend, fullgraph=True)
        torch.testing.assert_close(compiled(logits), expected, rtol=1e-5, atol=1e-5)

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

    def test_dreamer_v3_model_loss_compile_preserves_input(self, device):
        tensordict = self._create_world_model_data().to(device)
        reward = tensordict["next", "reward"].clone()
        loss_module = DreamerV3ModelLoss(
            self._create_world_model(reward_two_hot=True).to(device),
            num_reward_bins=self.num_reward_bins,
        )
        compiled_loss = torch.compile(loss_module, backend="eager")

        for _ in range(2):
            compiled_loss(tensordict)
            torch.testing.assert_close(tensordict["next", "reward"], reward)

    def test_dreamer_v3_model_loss_sums_only_event_dims(self, device):
        batch_size, event_size = (2, 3), 4
        target = torch.ones(*batch_size, event_size, device=device)
        logits = torch.zeros(
            *batch_size, self.num_cats, self.num_classes, device=device
        )
        tensordict = TensorDict(
            {
                "next": {
                    "pixels": target,
                    "reco_pixels": torch.zeros_like(target),
                    "prior_logits": logits,
                    "posterior_logits": logits.clone(),
                    "reward": torch.zeros(*batch_size, 1, device=device),
                }
            },
            batch_size,
        )
        world_model = TensorDictModule(
            torch.zeros_like,
            in_keys=[("next", "true_reward")],
            out_keys=[("next", "reward")],
        )
        loss_td, _ = DreamerV3ModelLoss(
            world_model, reward_two_hot=False, free_bits=0.0, global_average=False
        )(tensordict)
        expected = event_size * symlog(torch.tensor(1.0, device=device)).square()
        torch.testing.assert_close(loss_td["loss_model_reco"].squeeze(), expected)

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

    @pytest.mark.parametrize("detach_output", [True, False])
    def test_dreamer_v3_model_loss_detach_output(self, device, detach_output):
        world_model = self._create_world_model().to(device)
        loss_module = DreamerV3ModelLoss(
            world_model,
            num_reward_bins=self.num_reward_bins,
            detach_output=detach_output,
        )
        _, features = loss_module(self._create_world_model_data().to(device))
        posterior = features["next", "posterior_logits"]
        assert posterior.requires_grad is not detach_output

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

    @pytest.mark.parametrize("entropy_bonus", [0.0, 3e-4])
    def test_dreamer_v3_actor_entropy(self, device, entropy_bonus):
        actor_model = self._create_actor_model().to(device)
        actor_model[-1].distribution_class = IndependentNormal
        loss_module = DreamerV3ActorLoss(
            actor_model,
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
            imagination_horizon=3,
            entropy_bonus=entropy_bonus,
        )
        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        if entropy_bonus:
            distribution = actor_model.get_dist(
                fake_data.select(*actor_model.in_keys).detach()
            )
            expected = (
                fake_data["discount_weight"] * distribution.entropy().unsqueeze(-1)
            ).mean()
        else:
            expected = loss_td["loss_actor"].new_zeros(())
        torch.testing.assert_close(loss_td["actor_entropy"], expected)
        assert not loss_td["actor_entropy"].requires_grad
        loss_td["loss_actor"].backward()
        assert any(p.grad is not None for p in actor_model.parameters())

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_dreamer_v3_actor_loss_cuda_graph(self, device):
        device = torch.device(device)
        if device.type != "cuda":
            pytest.skip("CUDA graph test only runs for the CUDA parametrization")

        tensordict = self._create_actor_data().to(device).reshape(-1)
        actor_model = self._create_actor_model().to(device)
        # The DMC reproduction uses IndependentNormal, whose constructor does
        # not materialize action bounds from the host during graph capture.
        actor_model[-1].distribution_class = IndependentNormal
        continuation_model = TensorDictModule(
            nn.Sequential(nn.Linear(self.state_dim, 1), nn.Sigmoid()).to(device),
            in_keys=["state"],
            out_keys=["continuation"],
        )
        loss_module = DreamerV3ActorLoss(
            actor_model,
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
            continuation_model=continuation_model,
            imagination_horizon=3,
        )
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_module.to(device)

        warmup_stream = torch.cuda.Stream(device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                loss_module(tensordict)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss_td, fake_data = loss_module(tensordict)
        graph.replay()
        torch.cuda.synchronize(device)

        assert torch.isfinite(loss_td["loss_actor"])
        assert fake_data.shape == (tensordict.shape[0], 3)

    def test_dreamer_v3_continuation_lambda_and_weights(self, device):
        class _ConstantContinuation(nn.Module):
            def forward(self_, state, belief):
                return state[..., :1] * 0 + 0.5

        actor_model = self._create_actor_model_with_log_prob().to(device)
        continuation_model = TensorDictModule(
            _ConstantContinuation(),
            in_keys=["state", "belief"],
            out_keys=["continuation"],
        ).to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            self._create_mb_env().to(device),
            continuation_model=continuation_model,
            imagination_horizon=3,
            discount_loss=True,
            entropy_bonus=0.0,
            use_reinforce=True,
            return_normalization=False,
        )
        loss_module.make_value_estimator(ValueEstimators.TDLambda, gamma=1.0, lmbda=0.5)

        reward = torch.tensor([[[1.0], [2.0], [3.0]]], device=device)
        value = torch.tensor([[[10.0], [20.0], [30.0]]], device=device)
        continuation = torch.full_like(reward, 0.5)
        torch.testing.assert_close(
            loss_module.lambda_target(reward, value, continuation),
            torch.tensor([[[6.375], [11.5], [18.0]]], device=device),
        )

        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        # The initial state is weighted by its own continuation probability.
        expected_weight = torch.tensor([0.5, 0.25, 0.125], device=device)
        torch.testing.assert_close(
            fake_data["discount_weight"][0, :, 0], expected_weight
        )
        torch.testing.assert_close(
            fake_data["next", "continuation"],
            torch.full_like(fake_data["next", "continuation"], 0.5),
        )
        assert not fake_data["discount_weight"].requires_grad
        actor_parameters = tuple(actor_model.parameters())
        actual_gradients = torch.autograd.grad(
            loss_td["loss_actor"], actor_parameters, retain_graph=True
        )

        actor_inputs = fake_data.select(*actor_model.in_keys, strict=False).detach()
        distribution = actor_model.get_dist(actor_inputs)
        log_prob = distribution.log_prob(fake_data["action"].detach())
        log_prob = _match_trailing_dim(log_prob, fake_data["lambda_target"])
        baseline_td = fake_data.select(*value_model.in_keys, strict=False)
        value_model(baseline_td)
        advantage = (fake_data["lambda_target"] - baseline_td["state_value"]).detach()
        expected_loss = -(fake_data["discount_weight"] * log_prob * advantage).mean()
        expected_gradients = torch.autograd.grad(expected_loss, actor_parameters)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)

        value_loss = DreamerV3ValueLoss(
            value_model,
            discount_loss=True,
            actor_loss=loss_module,
        )
        value_loss(fake_data.detach())

    # ------------------------------------------------------------------ #
    # Value loss tests
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("compiled", [False, True])
    def test_dreamer_v3_replay_value_target(self, device, compiled):
        reward = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=device)
        bootstrap = torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=device)
        done = torch.zeros_like(reward, dtype=torch.bool)
        terminated = torch.zeros_like(done)

        target_fn = _replay_value_target
        if compiled:
            target_fn = torch.compile(target_fn, backend="eager", fullgraph=True)
        target = target_fn(
            reward,
            done,
            terminated,
            bootstrap,
            horizon=2.0,
            lmbda=0.5,
        )
        torch.testing.assert_close(
            target, torch.tensor([[9.8125, 15.25, 23.0]], device=device)
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_dreamer_v3_replay_value_target_cuda_graph(self, device):
        device = torch.device(device)
        if device.type != "cuda":
            pytest.skip("CUDA graph test only runs for the CUDA parametrization")

        reward = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=device)
        bootstrap = torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=device)
        done = torch.zeros_like(reward, dtype=torch.bool)
        terminated = torch.zeros_like(done)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            target = _replay_value_target(
                reward,
                done,
                terminated,
                bootstrap,
                horizon=2.0,
                lmbda=0.5,
            )
        graph.replay()
        torch.cuda.synchronize(device)

        torch.testing.assert_close(
            target, torch.tensor([[9.8125, 15.25, 23.0]], device=device)
        )

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_dreamer_v3_replay_value_loss_nested_keys(self, device, reduction):
        batch, time_steps = 2, 4
        state = torch.randn(
            batch,
            time_steps,
            self.state_dim,
            device=device,
            requires_grad=True,
        )
        replay = TensorDict(
            {
                "state": state,
                "belief": torch.randn(
                    batch, time_steps, self.rnn_hidden_dim, device=device
                ),
                "first_return": torch.randn(batch, time_steps, device=device),
                "next": {
                    "replay": {
                        "reward": torch.randn(batch, time_steps, device=device),
                        "done": torch.zeros(
                            batch, time_steps, dtype=torch.bool, device=device
                        ),
                        "terminated": torch.zeros(
                            batch, time_steps, dtype=torch.bool, device=device
                        ),
                    }
                },
            },
            [batch, time_steps],
        )
        value_loss = DreamerV3ValueLoss(
            self._create_value_model().to(device), reduction=reduction
        ).to(device)
        value_loss.set_keys(
            reward=("replay", "reward"),
            done=("replay", "done"),
            terminated=("replay", "terminated"),
            bootstrap="first_return",
        )

        loss = value_loss.replay_value_loss(replay)["loss_replay_value"]
        expected_shape = (batch, time_steps - 1) if reduction == "none" else ()
        assert loss.shape == expected_shape
        loss.sum().backward()
        assert state.grad is not None and state.grad.abs().sum() > 0

    @pytest.mark.parametrize("discount_loss", [True, False])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_dreamer_v3_value_loss_symlog_mse(self, device, discount_loss, reduction):
        tensordict = self._create_value_data().to(device)
        value_model = self._create_value_model(out_features=1).to(device)
        loss_module = DreamerV3ValueLoss(
            value_model,
            value_loss="symlog_mse",
            discount_loss=discount_loss,
            reduction=reduction,
        )
        loss_td, _ = loss_module(tensordict)
        assert "loss_value" in loss_td.keys()
        expected_shape = tensordict.batch_size if reduction == "none" else ()
        assert loss_td["loss_value"].shape == expected_shape
        loss_td["loss_value"].sum().backward()
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

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf),
        reason="requires hydra and omegaconf",
    )
    def test_dreamer_v3_sota_shares_imagination_parameters(self, device, monkeypatch):
        from omegaconf import OmegaConf

        repo_root = Path(__file__).parents[2]
        example_dir = repo_root / "sota-implementations/dreamer_v3"
        monkeypatch.syspath_prepend(str(example_dir))
        example = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/train.py",
            run_name="dreamer_v3_test",
        )
        cfg = OmegaConf.load(repo_root / "sota-implementations/dreamer_v3/config.yaml")
        cfg.networks.num_reward_bins = self.num_reward_bins
        (world_model, prior, reward_net, reward_decoder, continuation_net,) = example[
            "build_world_model"
        ](cfg=cfg, obs_dim=3, action_dim=self.action_dim)
        posterior = world_model[1].rssm_posterior.module
        imagination_model = example["build_imagination_model"](
            prior_net=prior,
            reward_net=reward_net,
            reward_decoder=reward_decoder,
        ).to(device)
        continuation_model = example["build_continuation_model"](
            continuation_net=continuation_net
        ).to(device)
        actor_model = example["build_actor"](cfg=cfg, action_dim=self.action_dim).to(
            device
        )
        real_actor = example["build_real_world_actor"](
            world_model=world_model,
            actor_model=actor_model,
        ).to(device)
        world_model = world_model.to(device)
        assert (
            prior.rnn_to_prior_projector[0].out_features
            == posterior.obs_rnn_to_post_projector[0].out_features
            == cfg.networks.hidden_dim
        )
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
        world_input = world_model(world_input)
        torch.testing.assert_close(
            world_input["next", "symlog_observation"], symlog(observation)
        )
        torch.testing.assert_close(
            symlog(world_input["next", "reco_pixels"]),
            world_input["next", "reco_symlog_observation"],
        )
        shared_parameters = tuple(prior.parameters()) + tuple(reward_net.parameters())
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
            for parameter in continuation_net.parameters()
        )

        observation = torch.tensor(
            [[0.25, -0.75, 1.5]], device=device, requires_grad=True
        )
        real_input = TensorDict(
            {
                "observation": observation,
                "state": torch.zeros(1, self.state_dim, device=device),
                "belief": torch.zeros(1, cfg.networks.rnn_hidden_dim, device=device),
                "previous_action": torch.zeros(1, self.action_dim, device=device),
                "is_init": torch.zeros(1, 1, dtype=torch.bool, device=device),
            },
            [1],
        )
        real_actor(real_input)
        observation_gradient = torch.autograd.grad(
            real_input["loc"].sum(), observation
        )[0]
        assert observation_gradient.abs().sum() > 0
        assert ("next", "belief") in real_input.keys(include_nested=True)

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

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(not _has_hoptorch, reason="requires hoptorch")
    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf and _has_gym),
        reason="requires hydra, omegaconf, and gym",
    )
    # This compares two independently compiled forward/backward paths and
    # CUDA capture. Even the previous loop backend took ~266s on CI runners.
    @pytest.mark.parametrize(
        "compile_train_step",
        [False, pytest.param(True, marks=pytest.mark.timeout(600))],
    )
    def test_dreamer_v3_full_learner_cuda_graph_matches_uncaptured(
        self, device, monkeypatch, compile_train_step
    ):
        device = torch.device(device)
        if device.type != "cuda":
            pytest.skip("CUDA graph test only runs for the CUDA parametrization")

        repo_root = Path(__file__).parents[2]
        example_dir = repo_root / "sota-implementations/dreamer_v3"
        monkeypatch.syspath_prepend(str(example_dir))
        example = runpy.run_path(
            example_dir / "train.py",
            run_name="dreamer_v3_learner_cuda_graph_test",
        )

        state_dim = 4
        torch.manual_seed(1)
        data = TensorDict(
            {
                "state": torch.zeros(2, 3, state_dim, device=device),
                "belief": torch.zeros(2, 3, 8, device=device),
                "action": torch.randn(2, 3, 1, device=device),
                "is_init": torch.zeros(2, 3, 1, dtype=torch.bool, device=device),
                "next": {
                    "observation": torch.randn(2, 3, 3, device=device),
                    "reward": torch.randn(2, 3, 1, device=device),
                    "done": torch.zeros(2, 3, 1, dtype=torch.bool, device=device),
                    "terminated": torch.zeros(2, 3, 1, dtype=torch.bool, device=device),
                },
            },
            [2, 3],
            device=device,
        )

        updates = []
        modules = []
        learners = []
        for cudagraph in (False, True):
            cfg = self._small_sota_config(
                example_dir,
                compile_train_step=compile_train_step,
                cudagraph_train_step=cudagraph,
                mixed_precision=True,
            )
            torch.manual_seed(0)
            learner = example["_build_learner"](cfg, device, 3, 1)
            learners.append(learner)
            modules.append(
                nn.ModuleList(
                    [learner.model_loss, learner.actor_loss, learner.value_loss]
                )
            )
            updates.append(
                example["_make_learner_update"](
                    cfg,
                    device,
                    learner,
                    cudagraph_warmup=5,
                )
            )

        initial_state = {
            key: value.detach().clone()
            for key, value in modules[0].state_dict().items()
        }
        torch.testing.assert_close(modules[1].state_dict(), initial_state)

        # Compile both paths and finish graph capture without applying updates.
        for update in updates:
            update.warmup(data)
        for module in modules:
            module.load_state_dict(initial_state)

        # Actual replay batches have squeezed boolean feature axes.
        for key in ("is_init", ("next", "done"), ("next", "terminated")):
            data.set(key, data.get(key).squeeze(-1))
        eager_data = data.clone()
        graph_data = data.clone()
        retained = []
        for seed in (10, 11):
            torch.manual_seed(seed)
            expected = updates[0].step(None, eager_data)
            torch.manual_seed(seed)
            actual = updates[1].step(None, graph_data)
            assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            assert_close(
                graph_data["replay_context"],
                eager_data["replay_context"],
                atol=2e-3,
                rtol=2e-3,
            )
            for previous, snapshot in retained:
                assert_close(previous, snapshot)
            retained.append((actual, actual.clone()))
            retained.append(
                (graph_data["replay_context"], graph_data["replay_context"].clone())
            )

        torch.testing.assert_close(
            modules[1].state_dict(),
            modules[0].state_dict(),
            atol=2e-4,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            learners[1].optimizer.state_dict(),
            learners[0].optimizer.state_dict(),
            atol=2e-3,
            rtol=2e-3,
        )
        for module in modules:
            assert any(
                not torch.equal(parameter, initial_state[name])
                for name, parameter in module.named_parameters()
            )
        assert learners[0].optimizer.state
        assert learners[1].optimizer.state
        assert learners[0].optimizer.param_groups[0]["step"] == 2
        assert learners[1].optimizer.param_groups[0]["step"] == 2
        target_prefix = "2.target_value_model_params"
        assert any(
            not torch.equal(value, initial_state[key])
            for key, value in modules[0].state_dict().items()
            if key.startswith(target_prefix)
        )

        benchmark = runpy.run_path(
            repo_root / "benchmarks/ad_hoc/bench_dreamer_v3_learner.py",
            run_name="dreamer_v3_native_replay_cuda_graph_test",
        )
        cfg = self._small_sota_config(
            example_dir,
            compile_train_step=compile_train_step,
            cudagraph_train_step=True,
            mixed_precision=True,
        )
        torch.manual_seed(0)
        replay_learner = example["_build_learner"](cfg, device, 3, 1)
        replay_update = example["_make_learner_update"](
            cfg,
            device,
            replay_learner,
            cudagraph_warmup=5,
        )
        example["_warm_up_learner"](cfg, device, replay_update, 3, 1)
        replay_step = benchmark["_ReplayLearnerStep"](
            example,
            cfg,
            replay_update,
            device=device,
            replay_device=torch.device("cpu"),
            obs_dim=3,
            action_dim=1,
        )
        try:
            for _ in range(6):
                replay_step()
            replay_step.synchronize()
        finally:
            replay_step.close()

    @pytest.mark.skipif(
        not (_has_hydra and _has_omegaconf and _has_gym),
        reason="requires hydra, omegaconf, and gym",
    )
    def test_dreamer_v3_full_learner_compile_avoids_nested_rssm(
        self, device, monkeypatch
    ):
        device = torch.device(device)
        if device.type != "cpu":
            pytest.skip("CPU compile regression")

        repo_root = Path(__file__).parents[2]
        example_dir = repo_root / "sota-implementations/dreamer_v3"
        monkeypatch.syspath_prepend(str(example_dir))
        example = runpy.run_path(
            example_dir / "train.py",
            run_name="dreamer_v3_learner_compile_test",
        )
        cfg = self._small_sota_config(
            example_dir,
            compile_train_step=True,
            cudagraph_train_step=False,
        )
        monkeypatch.setattr(
            torch, "compile", ft.partial(torch.compile, backend="eager")
        )

        learner = example["_build_learner"](cfg, device, 3, 1)
        if not getattr(torch._dynamo.config, "inline_inbuilt_nn_modules", False):
            with pytest.raises(RuntimeError, match="inline_inbuilt_nn_modules"):
                example["_make_learner_update"](cfg, device, learner)
            return
        update = example["_make_learner_update"](cfg, device, learner)
        sample = example["_fake_learner_sample"](cfg, device, 3, 1)
        reward = sample.get(("next", "reward")).clone()
        for key in ("is_init", ("next", "done"), ("next", "terminated")):
            sample.set(key, sample.get(key).squeeze(-1))
        update.loss_module.set_keys(replay_context=("learner", "refresh"))
        rng_before = torch.get_rng_state().clone()
        state_before = copy.deepcopy(update.loss_module.state_dict())
        with pytest.raises(RuntimeError, match="Call warmup"):
            update.step(None, sample)
        example["_warm_up_learner"](cfg, device, update, 3, 1)
        torch.testing.assert_close(torch.get_rng_state(), rng_before)
        torch.testing.assert_close(update.loss_module.state_dict(), state_before)
        parameters = list(learner.optimizer.param_groups[0]["params"])
        gradient_addresses = [
            parameter.grad.data_ptr() if parameter.grad is not None else None
            for parameter in parameters
        ]
        before = [parameter.detach().clone() for parameter in parameters]
        losses = update.step(None, sample)
        metrics = example["_learner_metrics"](losses)
        assert not sample["learner", "refresh", "state"].requires_grad
        assert gradient_addresses == [
            parameter.grad.data_ptr() if parameter.grad is not None else None
            for parameter in parameters
        ]

        assert any(
            not torch.equal(parameter, previous)
            for parameter, previous in zip(parameters, before)
        )
        assert torch.isfinite(metrics).all()
        logged = []

        def log_metrics(steps, metrics):
            logged.append((steps, metrics))

        trainer = Trainer(
            collector=[sample],
            total_frames=sample.numel(),
            frame_skip=1,
            optim_steps_per_batch=2,
            loss_module=update.loss_module,
            optimization_stepper=update,
            progress_bar=False,
        )
        trainer.register_op("post_optim_complete_log", log_metrics)
        trainer.optim_steps(sample)
        assert len(logged) == 1 and logged[0][0] == 2
        assert all(value.numel() == 1 for value in logged[0][1].values(True, True))
        update.target_updater = HardUpdate(
            learner.value_loss, value_network_update_interval=5
        )
        saved_stepper = copy.deepcopy(update.state_dict())
        update.step(None, sample)
        assert update.target_updater.counter == 1
        update.load_state_dict(saved_stepper)
        assert update.target_updater.counter == 0
        assert gradient_addresses == [
            parameter.grad.data_ptr() if parameter.grad is not None else None
            for parameter in parameters
        ]
        learner.optimizer.zero_grad(set_to_none=True)
        with pytest.raises(RuntimeError, match="no parameter gradients"):
            learner.optimizer.step()
        torch.testing.assert_close(sample.get(("next", "reward")), reward)
        update.loss_module.replay_value_loss_weight = 0.0
        update.loss_module.model_loss.detach_output = True
        short_sample = sample[:, :1]
        for key in ("is_init", ("next", "done"), ("next", "terminated")):
            short_sample.set(key, short_sample.get(key).unsqueeze(-1))
        losses = update.loss_module(short_sample)
        assert losses["loss_replay_value"] == 0
        assert all(
            torch.isfinite(value).all()
            for key, value in losses.items()
            if key.startswith("loss_")
        )

    @pytest.mark.skipif(not _has_omegaconf, reason="requires omegaconf")
    def test_dreamer_v3_dmc_benchmark_aggregation(self, device, tmp_path):
        from omegaconf import OmegaConf

        del device
        repo_root = Path(__file__).parents[2]
        benchmark = runpy.run_path(
            repo_root / "sota-implementations/dreamer_v3/benchmark.py",
            run_name="dreamer_v3_benchmark_test",
        )
        paths = []
        for seed, returns in enumerate(([1.0, 4.0], [3.0, 6.0], [2.0, 5.0])):
            path = tmp_path / f"seed_{seed}.jsonl"
            records = [
                {
                    "type": "train_episode",
                    "environment_steps": step,
                    "score": score,
                }
                for step, score in zip((100, 200), returns)
            ]
            records.append(
                {
                    "type": "summary",
                    "seed": seed,
                    "total_environment_steps": 200,
                }
            )
            path.write_text("\n".join(map(json.dumps, records)) + "\n")
            paths.append(path)

        summary = benchmark["aggregate_runs"](paths, window_size=100)
        assert summary["environment_steps"] == [100, 200]
        assert summary["median_return"] == [2.0, 5.0]
        assert summary["lower_quartile_return"] == [1.5, 4.5]
        assert summary["upper_quartile_return"] == [2.5, 5.5]

        config = OmegaConf.load(
            repo_root / "sota-implementations/dreamer_v3/config_dmc_walker.yaml"
        )
        assert config.env.name == "walker"
        assert config.env.task == "walk"
        assert config.collector.total_frames == 1_100_000
        assert config.optimization.train_ratio == 1024

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
        with (
            torch.no_grad(),
            loss_module.target_value_model_params.to_module(
                loss_module.value_model, preserve_module_state=False
            ),
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

        td_c = td_a.clone()
        td_d = td_a.clone()
        td_c["action"][:, 2].zero_()
        td_d["action"][:, 2].fill_(1.0)
        torch.manual_seed(0)
        out_c = rollout(td_c)
        torch.manual_seed(0)
        out_d = rollout(td_d)
        torch.testing.assert_close(
            out_c["next", "prior_logits"][:, 2],
            out_d["next", "prior_logits"][:, 2],
        )

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

    @pytest.mark.parametrize("heads", ["vector", "image", "both"])
    @pytest.mark.parametrize("compile_loss", [False, True])
    def test_dreamer_v3_reconstruction_heads(self, device, heads, compile_loss):
        class WorldModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vector = nn.Parameter(torch.ones(2, device=device))
                self.image = nn.Parameter(torch.tensor([0.25, 0.5], device=device))

            def forward(self, td):
                td["next", "decoded", "vector"] = self.vector.expand(2, 3, 2)
                td["next", "decoded", "image"] = self.image.expand(2, 3, 2)
                td["next", "reward"] = torch.zeros(2, 3, 1, device=device)
                return td

        model = WorldModel()
        sample = TensorDict(
            {
                ("next", "sensors", "vector"): torch.tensor(
                    [0.0, 3.0], device=device
                ).expand(2, 3, 2),
                ("next", "sensors", "image"): torch.tensor(
                    [0, 255], dtype=torch.uint8, device=device
                ).expand(2, 3, 2),
                ("next", "reward"): torch.zeros(2, 3, 1, device=device),
                ("next", "prior_logits"): torch.zeros(2, 3, 2, 2, device=device),
                ("next", "posterior_logits"): torch.zeros(2, 3, 2, 2, device=device),
            },
            [2, 3],
        )
        names = ["vector", "image"] if heads == "both" else [heads]
        symlog_flags = [name == "vector" for name in names]
        objective = DreamerV3ModelLoss(
            model,
            reward_two_hot=False,
            reco_symlog=symlog_flags if heads == "both" else symlog_flags[0],
        )
        objective.set_keys(
            pixels=[("sensors", name) for name in names],
            reco_pixels=[("decoded", name) for name in names],
        )
        call = torch.compile(objective, backend="eager") if compile_loss else objective
        losses, _ = call(sample)
        log_two = torch.tensor(2.0, device=device).log()
        expected = (2 * log_two.square() if "vector" in names else 0) + (
            0.3125 if "image" in names else 0
        )
        torch.testing.assert_close(
            losses["loss_model_reco"],
            torch.as_tensor(expected, device=device).reshape(1),
        )
        losses["loss_model_reco"].sum().backward()
        if "vector" in names:
            torch.testing.assert_close(
                model.vector.grad, torch.stack([log_two, -log_two])
            )
        if "image" in names:
            torch.testing.assert_close(
                model.image.grad, model.image.new_tensor([0.5, -1])
            )

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
        expected = -(log_prob * advantage / 10.0).mean()
        torch.testing.assert_close(loss_td["loss_actor"], expected)
        torch.testing.assert_close(
            loss_td["return_scale"], torch.tensor(10.0, device=device)
        )

        compiled_scale = torch.compile(
            loss_module._return_scale, backend=_compile_backend, fullgraph=True
        )
        torch.testing.assert_close(
            compiled_scale(fake_data["lambda_target"]),
            torch.tensor(10.0, device=device),
        )

    def test_dreamer_v3_reparam_return_normalization(self, device):
        """The reparameterization branch must divide the objective by the
        EMA return-percentile span, like the REINFORCE branch."""
        actor_model = self._create_actor_model().to(device)
        value_model = self._create_value_model().to(device)
        loss_module = DreamerV3ActorLoss(
            actor_model,
            value_model,
            self._create_mb_env().to(device),
            imagination_horizon=3,
            discount_loss=False,
            entropy_bonus=0.0,
            use_reinforce=False,
        ).to(device)
        loss_module.make_value_estimator(ValueEstimators.TDLambda)
        loss_module.return_low.fill_(-2.0)
        loss_module.return_high.fill_(8.0)
        loss_module.eval()

        loss_td, fake_data = loss_module(
            self._create_actor_data().to(device).reshape(-1)
        )
        expected = -(fake_data["lambda_target"] / 10.0).mean()
        torch.testing.assert_close(loss_td["loss_actor"], expected)
        torch.testing.assert_close(
            loss_td["return_scale"], torch.tensor(10.0, device=device)
        )

    def test_dreamer_v3_reparam_return_statistics_update(self, device):
        """Training-mode forward in the reparameterization branch must update
        the EMA return statistics."""
        loss_module = DreamerV3ActorLoss(
            self._create_actor_model().to(device),
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
            imagination_horizon=3,
            entropy_bonus=0.0,
            use_reinforce=False,
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
        torch.testing.assert_close(
            loss_td["return_scale"],
            (loss_module.return_high - loss_module.return_low).clamp_min(1.0),
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

    def test_dreamer_v3_legacy_retnorm_checkpoint_migrates(self, device):
        """Checkpoints written before the retnorm refactor stored 0-dim
        ``return_low`` / ``return_high`` buffers; loading them must fill the
        ``retnorm`` statistics without strict-mode key errors."""
        loss_module = DreamerV3ActorLoss(
            self._create_actor_model_with_log_prob().to(device),
            self._create_value_model().to(device),
            self._create_mb_env().to(device),
            imagination_horizon=3,
            use_reinforce=True,
        ).to(device)
        legacy_checkpoint = {
            key: value.detach().clone()
            for key, value in loss_module.state_dict().items()
            if key not in ("retnorm.low", "retnorm.high")
        }
        legacy_checkpoint["return_low"] = torch.tensor(-2.5, device=device)
        legacy_checkpoint["return_high"] = torch.tensor(7.5, device=device)
        loss_module.load_state_dict(legacy_checkpoint)
        torch.testing.assert_close(
            loss_module.retnorm.low, torch.tensor([-2.5], device=device)
        )
        torch.testing.assert_close(
            loss_module.retnorm.high, torch.tensor([7.5], device=device)
        )

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


@pytest.mark.skipif(
    not (_has_hydra and _has_omegaconf and _has_gym),
    reason="requires hydra, omegaconf, and gym",
)
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("count_reset_records", [False, True])
def test_dreamer_v3_native_stream_replay(monkeypatch, online, count_reset_records):
    from omegaconf import OmegaConf

    repo_root = Path(__file__).parents[2]
    example_dir = repo_root / "sota-implementations/dreamer_v3"
    monkeypatch.syspath_prepend(str(example_dir))
    example = runpy.run_path(
        example_dir / "train.py",
        run_name=f"dreamer_v3_native_replay_{online}",
    )
    cfg = OmegaConf.load(example_dir / "config.yaml")
    cfg.collector.num_envs = 2
    cfg.collector.count_reset_records = count_reset_records
    cfg.collector.total_frames = 16
    cfg.collector.frames_per_batch = 4
    cfg.env.max_episode_steps = 3
    expected_action_budget = 12 if count_reset_records else 16
    assert example["_validated_action_budget"](cfg) == expected_action_budget
    cfg.replay_buffer.buffer_size = 15
    cfg.replay_buffer.batch_size = 2
    cfg.replay_buffer.seq_len = 2
    cfg.replay_buffer.online = online
    rb = example["_build_replay"](cfg, 2, torch.device("cpu"), torch.device("cpu"))
    sampler_type = StreamingSliceSampler if online else SliceSampler
    assert [rb[index].storage.max_size for index in range(2)] == [8, 7]
    assert all(isinstance(rb[index].sampler, sampler_type) for index in range(2))

    data = TensorDict(
        {
            "action": torch.arange(12).reshape(2, 6, 1).float(),
            "is_init": torch.zeros(2, 6, 1, dtype=torch.bool),
            "state": torch.zeros(2, 6, 4),
            "belief": torch.zeros(2, 6, 5),
            "next": {
                "observation": torch.randn(2, 6, 3),
                "reward": torch.randn(2, 6, 1),
                "done": torch.zeros(2, 6, 1, dtype=torch.bool),
                "terminated": torch.zeros(2, 6, 1, dtype=torch.bool),
                "truncated": torch.zeros(2, 6, 1, dtype=torch.bool),
            },
        },
        [2, 6],
    )
    data["is_init"][:, 0] = True
    try:
        rb.extend(data)
        assert rb.stats()["write_count"] == 12
        assert rb.can_sample()
        sample = rb.sample().reshape(2, 3)
        assert ("collector", "context_valid") not in sample.keys(True, True)
        assert sample["is_init"].sum() <= 2
        sample_info = sample.select("index", "index_generation")
        index, generation, patch = example["replay_context_update"](
            sample_info,
            torch.ones(2, 2, 4),
            torch.ones(2, 2, 5),
        )
        result = rb.submit_update_if_present(
            index=index, generation=generation, patch=patch
        ).result()
        assert result.updated_count == index.numel()
        assert any(rb[index][:]["state"].any() for index in range(2))
        assert all(rb[index][:]["is_init"][0].all() for index in range(2))
    finally:
        rb.shutdown()


@pytest.mark.skipif(not _has_omegaconf, reason="requires omegaconf")
def test_dreamer_v3_replay_capacity_validation(monkeypatch):
    from omegaconf import OmegaConf

    repo_root = Path(__file__).parents[2]
    example_dir = repo_root / "sota-implementations/dreamer_v3"
    monkeypatch.syspath_prepend(str(example_dir))
    example = runpy.run_path(
        example_dir / "train.py", run_name="dreamer_v3_replay_capacity_test"
    )
    cfg = OmegaConf.load(example_dir / "config.yaml")
    cfg.collector.num_envs = 2
    cfg.replay_buffer.buffer_size = 7
    cfg.replay_buffer.seq_len = 3
    with pytest.raises(ValueError, match="cannot hold one 4-record sequence"):
        example["_build_replay"](cfg, 2, torch.device("cpu"), torch.device("cpu"))


@pytest.mark.skipif(
    not (_has_hydra and _has_omegaconf and _has_gym),
    reason="requires hydra, omegaconf, and gym",
)
def test_dreamer_v3_native_replay_benchmark_step_cpu(monkeypatch):
    from omegaconf import OmegaConf

    repo_root = Path(__file__).parents[2]
    example_dir = repo_root / "sota-implementations/dreamer_v3"
    monkeypatch.syspath_prepend(str(example_dir))
    example = runpy.run_path(
        example_dir / "train.py", run_name="dreamer_v3_replay_benchmark_cpu_test"
    )
    benchmark = runpy.run_path(
        repo_root / "benchmarks/ad_hoc/bench_dreamer_v3_learner.py",
        run_name="dreamer_v3_replay_benchmark_helpers_test",
    )
    cfg = OmegaConf.load(example_dir / "config.yaml")
    cfg.replay_buffer.batch_size = 2
    cfg.replay_buffer.seq_len = 3

    class LearnerUpdate:
        def __init__(self):
            self.calls = 0

        def step(self, trainer, sample):
            self.calls += 1
            assert sample.shape == (2, 3)
            sample.set(
                "replay_context",
                TensorDict(
                    state=sample["state"] + self.calls,
                    belief=sample["belief"] + self.calls,
                    batch_size=sample.batch_size,
                ),
            )
            return TensorDict({}, [])

    learner_update = LearnerUpdate()
    step = benchmark["_ReplayLearnerStep"](
        example,
        cfg,
        learner_update,
        device=torch.device("cpu"),
        replay_device=torch.device("cpu"),
        obs_dim=3,
        action_dim=1,
    )
    try:
        step()
        step()
        step.synchronize()
        assert learner_update.calls == 2
        assert any(step.replay_buffer[index][:]["state"].any() for index in range(4))
    finally:
        step.close()


@pytest.mark.skipif(
    not (_has_hydra and _has_omegaconf and _has_gym),
    reason="requires hydra, omegaconf, and gym",
)
@pytest.mark.parametrize(
    ("collector_backend", "custom", "pixels", "discrete", "budget"),
    [
        ("sync", False, False, False, "frames"),
        ("async", False, False, False, "frames"),
        ("sync", True, False, False, "frames"),
        ("async", True, True, False, "frames"),
        ("async", True, True, True, "frames"),
        ("sync", True, True, True, "frames"),
        ("async", True, False, False, "time"),
        ("async", True, False, False, "warmup"),
        ("sync", True, False, False, "reset_records"),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires CUDA"
                ),
            ],
        ),
    ],
)
def test_dreamer_v3_native_replay_collection_smoke(
    monkeypatch, tmp_path, collector_backend, custom, pixels, discrete, budget, device
):
    from omegaconf import OmegaConf

    repo_root = Path(__file__).parents[2]
    example_dir = repo_root / "sota-implementations/dreamer_v3"
    monkeypatch.syspath_prepend(str(example_dir))
    example = runpy.run_path(
        example_dir / "train.py",
        run_name=f"dreamer_v3_native_replay_collection_{collector_backend}",
    )
    cfg = OmegaConf.load(example_dir / "config.yaml")
    cfg.env.name = PENDULUM_VERSIONED()
    cfg.optimization.separate_policy_rng = collector_backend == "async" and not custom
    cfg.optimization.device = device
    # This test covers collection and replay; keep the learner eager apart
    # from the explicit capture below.
    cfg.optimization.compile = "off"
    cfg.optimization.cudagraph_train_step = device == "cuda"
    cfg.optimization.updates_per_batch = 1
    cfg.optimization.train_ratio = None
    cfg.collector.backend = collector_backend
    cfg.collector.num_envs = 2
    cfg.collector.frames_per_batch = 8
    cfg.collector.total_frames = 16
    cfg.replay_buffer.buffer_size = 64
    cfg.replay_buffer.batch_size = 2
    cfg.replay_buffer.seq_len = 2
    cfg.replay_buffer.warmup_factor = 1
    cfg.logger.eval_every = 0
    cfg.logger.train_every = 0
    cfg.logger.output_plot = None
    cfg.logger.metrics_jsonl = None
    if custom:
        cfg.env.backend = "custom"
        cfg.env.factory = f"{__name__}:_DreamerV3TestEnv"
        cfg.env.factory_kwargs = {"pixels": pixels, "discrete": discrete}
        cfg.env.vector_key = None if discrete else ["sensors", "vector"]
        cfg.env.pixels_key = ["sensors", "image"] if pixels else None
        cfg.env.milestone_key = ["episode", "milestones"]
        cfg.env.milestone_names = ["started", "completed"]
        cfg.networks.image_depth = 2
        cfg.networks.image_mults = [1, 1]
        cfg.networks.image_kernel_size = 3
        cfg.networks.image_decoder_blocks = 2
        cfg.logger.backend = "csv"
        cfg.logger.log_dir = str(tmp_path / "logs")
        cfg.logger.metrics_jsonl = str(tmp_path / "metrics.jsonl")
        cfg.logger.train_every = 8
    if budget == "time":
        cfg.collector.total_frames = -1
        cfg.optimization.max_time = 0.2
    elif budget == "warmup":
        cfg.optimization.collection_warmup_seconds = 60
    elif budget == "reset_records":
        cfg.collector.count_reset_records = True
        cfg.collector.total_frames = 18  # 16 actions under the configured horizon.
    example["main"].__wrapped__(cfg)
    if custom:
        records = [
            json.loads(line)
            for line in Path(cfg.logger.metrics_jsonl).read_text().splitlines()
        ]
        if budget == "time":
            assert records[-1]["total_action_steps"] > 0
            assert records[-1]["elapsed_seconds"] < 5
        else:
            assert records[-1]["total_action_steps"] == 16
            assert (records[-1]["updates"] > 0) == (budget != "warmup")
        episodes = [record for record in records if record["type"] == "train_episode"]
        assert episodes and all(
            record["milestones"] == [True, True] for record in episodes
        )
        assert list((tmp_path / "logs").rglob("*.csv"))
        if budget == "reset_records":
            # The custom environment terminates before the configured time limit.
            assert records[-1]["total_environment_steps"] == 22
            assert [episode["environment_steps"] for episode in episodes] == [
                7,
                8,
                15,
                16,
            ]


@pytest.mark.skipif(
    not (_has_hydra and _has_omegaconf and _has_gym),
    reason="requires hydra, omegaconf, and gym",
)
@pytest.mark.parametrize(
    ("collector_backend", "include_replay", "terminate"),
    [
        ("sync", False, False),
        ("async", False, False),
        ("sync", True, False),
        ("async", True, False),
        ("async", False, True),
        ("async", True, True),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires CUDA"
                ),
            ],
        ),
    ],
)
def test_dreamer_v3_checkpoint_resume_processes(
    monkeypatch, tmp_path, collector_backend, include_replay, terminate, device
):
    from omegaconf import OmegaConf

    repo_root = Path(__file__).parents[2]
    example_dir = repo_root / "sota-implementations/dreamer_v3"
    monkeypatch.syspath_prepend(str(example_dir))
    example = runpy.run_path(
        example_dir / "train.py", run_name="dreamer_v3_resume_test"
    )
    cfg = TestDreamerV3()._small_sota_config(
        example_dir, compile_train_step=False, cudagraph_train_step=device == "cuda"
    )
    cfg.optimization.device = device
    cfg.optimization.compile = "off"
    cfg.optimization.compile_rssm = None
    cfg.optimization.updates_per_batch = 1
    cfg.optimization.separate_policy_rng = True
    cfg.optimization.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.optimization.checkpoint_every = 1
    cfg.optimization.checkpoint_keep_last = 20
    cfg.optimization.train_ratio = 1.5
    cfg.optimization.checkpoint_include_replay = include_replay
    cfg.collector.backend = collector_backend
    cfg.collector.num_envs = 2
    cfg.collector.frames_per_batch = 8
    cfg.collector.total_frames = 100_000 if terminate else 16
    cfg.replay_buffer.online = collector_backend == "async"
    cfg.replay_buffer.buffer_size = 64
    cfg.replay_buffer.warmup_factor = 1
    cfg.logger.backend = "csv"
    cfg.logger.log_dir = str(tmp_path / "logs")
    cfg.logger.metrics_jsonl = str(tmp_path / "metrics.jsonl")
    cfg.logger.train_every = 8
    cfg.logger.eval_every = 0
    cfg.logger.output_plot = None
    config_path = tmp_path / "resume_test.yaml"
    process_env = {**os.environ, "PYTHONPATH": str(repo_root), "OMP_NUM_THREADS": "1"}
    # Exercise training in fresh processes without depending on Hydra's CLI
    # parser, whose lazy help strings are incompatible with Python 3.14.
    runner = tmp_path / "resume_process.py"
    runner.write_text(
        "from __future__ import annotations\n"
        "import runpy\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from omegaconf import OmegaConf\n"
        "if __name__ == '__main__':\n"
        "    sys.path.insert(0, str(Path(sys.argv[1]).parent))\n"
        "    example = runpy.run_path(sys.argv[1])\n"
        "    example['main'].__wrapped__(OmegaConf.load(sys.argv[2]))\n"
    )
    command = [
        sys.executable,
        str(runner),
        str(example_dir / "train.py"),
        str(config_path),
    ]
    OmegaConf.save(cfg, config_path)
    rotation = CheckpointRotation(cfg.optimization.checkpoint_dir, keep_last=20)
    if terminate:
        with open(tmp_path / "first.log", "w+") as log:
            process = subprocess.Popen(command, env=process_env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while rotation.latest() is None and process.poll() is None:
                    assert (
                        time.monotonic() < deadline
                    ), "No checkpoint before termination."
                    time.sleep(0.05)
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=60)
                log.seek(0)
                assert process.returncode == 0, log.read()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
    else:
        first = subprocess.run(
            command, env=process_env, capture_output=True, text=True, timeout=120
        )
        assert first.returncode == 0, first.stdout + first.stderr
    first_path = rotation.latest()
    first_state = {}
    Checkpoint(run_state=first_state).load(first_path)
    first_steps = first_state["action_steps"]
    assert first_steps >= 8 if terminate else first_steps == 16
    assert first_state["updates"] > 0
    assert ("replay" in Checkpoint.manifest(first_path)["components"]) == include_replay
    # Inspect CUDA checkpoints on CPU with an eager execution configuration.
    inspect_cfg = copy.deepcopy(cfg)
    inspect_cfg.optimization.cudagraph_train_step = False
    first_learner = example["_build_learner"](inspect_cfg, torch.device("cpu"), 3, 1)
    stepper = example["_make_learner_update"](
        inspect_cfg, torch.device("cpu"), first_learner
    )
    modules = stepper.loss_module
    policy = example["DreamerV3SeededPolicy"](first_learner.real_world_actor, seed=0)
    schedule = example["DreamerV3UpdateRatio"](0.0)
    Checkpoint(policy=policy, update_ratio=schedule).load(first_path)
    first_policy_counter = policy.get_extra_state()["counter"]
    Checkpoint(learner=modules).load(first_path)
    before = [parameter.detach().clone() for parameter in modules.parameters()]
    tails = []
    if include_replay:
        replay = example["_build_replay"](
            cfg, 2, torch.device("cpu"), torch.device("cpu")
        )
        try:
            Checkpoint(replay=replay).load(first_path)
            tails = [
                (replay[i].writer.state_dict()["_cursor"] - 1) % len(replay[i])
                for i in range(2)
            ]
        finally:
            replay.shutdown()

    cfg.optimization.resume_from = str(first_path)
    cfg.collector.total_frames = first_steps + 16
    OmegaConf.save(cfg, config_path)
    second = subprocess.run(
        command, env=process_env, capture_output=True, text=True, timeout=90
    )
    assert second.returncode == 0, second.stdout + second.stderr
    second_path = rotation.latest()
    second_state = {}
    Checkpoint(run_state=second_state, learner=modules).load(second_path)
    assert second_state["action_steps"] == first_steps + 16
    assert second_state["updates"] > first_state["updates"]
    assert second_state["elapsed_seconds"] > first_state["elapsed_seconds"]
    Checkpoint(policy=policy, update_ratio=schedule).load(second_path)
    assert policy.get_extra_state()["counter"] > first_policy_counter
    for key in ("exp_name", "log_dir"):
        assert second_state["logger"][key] == first_state["logger"][key]
    previous_logs = first_state["logger"]["local"]["scalars"]["train/updates"]
    assert (
        second_state["logger"]["local"]["scalars"]["train/updates"][
            : len(previous_logs)
        ]
        == previous_logs
    )
    assert any(
        not torch.equal(old, new) for old, new in zip(before, modules.parameters())
    )
    records = [
        json.loads(line)
        for line in Path(cfg.logger.metrics_jsonl).read_text().splitlines()
    ]
    assert [
        record["total_action_steps"]
        for record in records
        if record["type"] == "summary"
    ] == [first_steps, first_steps + 16]
    train_steps = [
        record["environment_steps"] for record in records if record["type"] == "train"
    ]
    assert train_steps == sorted(set(train_steps))
    csv_steps = [
        int(line.split(",")[0])
        for line in next((tmp_path / "logs").rglob("train/updates.csv"))
        .read_text()
        .splitlines()
    ]
    assert csv_steps == train_steps
    if include_replay:
        replay = example["_build_replay"](
            cfg, 2, torch.device("cpu"), torch.device("cpu")
        )
        try:
            Checkpoint(replay=replay).load(second_path)
            assert replay.stats()["size"] == min(first_steps + 16, 64)
            # A process restart closes the old stream without inventing a terminal.
            for stream in range(2) if not terminate else ():
                assert replay[stream][tails[stream]]["next", "done"].all()
                assert not replay[stream][tails[stream]]["next", "terminated"].any()
            for _ in range(20):
                sample = replay.sample().reshape(2, 4)
                assert not sample["is_init"][:, 1:].any()
        finally:
            replay.shutdown()


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
def test_dreamer_v3_dmc_reproduction_modes(tmp_path):
    repo_root = Path(__file__).parents[2]
    script = repo_root / "sota-implementations/dreamer_v3/reproduce_dmc_walker.sh"
    benchmark = repo_root / "sota-implementations/dreamer_v3/benchmark.py"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    fast = subprocess.run(
        ["bash", str(script), "--fast", "benchmark.seeds=[0]"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    ).stdout.splitlines()
    expected_benchmark = str(benchmark)
    if os.name == "nt":
        expected_benchmark = f"/{benchmark.drive[0].lower()}{benchmark.as_posix()[2:]}"
    assert fast == [
        expected_benchmark,
        "--output-dir",
        "dmc_walker_runs",
        "optimization.compile_rssm=scan",
        "optimization.rssm_scan_unroll=8",
        "optimization.cudagraph_train_step=true",
        "benchmark.seeds=[0]",
    ]

    smoke = subprocess.run(
        ["bash", str(script), "--smoke"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    ).stdout.splitlines()
    assert smoke[:3] == [expected_benchmark, "--output-dir", "dmc_walker_smoke"]
    assert "replay_buffer.buffer_size=400" in smoke
    assert "optimization.compile=off" in smoke
    assert "optimization.updates_per_batch=1" in smoke
    assert "optimization.train_ratio=null" in smoke

    incompatible = subprocess.run(
        ["bash", str(script), "--fast", "--smoke"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert incompatible.returncode == 2
    assert "mutually exclusive" in incompatible.stderr


@pytest.mark.parametrize("device", get_default_devices())
def test_dreamer_v3_optimizer_updates_and_resume(device):
    parameter = nn.Parameter(torch.tensor([3.0, 4.0], device=device))
    optimizer = DreamerV3Optimizer(
        [parameter], lr=0.05, agc=0.2, beta1=0.5, beta2=0.5, warmup_steps=2
    )
    parameter.grad = torch.tensor([6.0, 8.0], device=device)
    optimizer.step()
    # The first update builds moments but starts the warm-up at zero.
    torch.testing.assert_close(parameter, parameter.new_tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([0.0, 10.0], device=device)
    optimizer.step()
    # Clipped gradients are (0.6, 0.8), then (0, 1). The second
    # bias-corrected RMS for the second coordinate is sqrt(0.88).
    expected = parameter.new_tensor(
        [3.0 - 0.025 / 3, 4.0 - 0.025 * (1 + 2 / 0.88**0.5) / 3]
    )
    torch.testing.assert_close(parameter, expected)

    restored_parameter = nn.Parameter(parameter.detach().clone())
    restored = DreamerV3Optimizer([restored_parameter])
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for current, value in ((optimizer, parameter), (restored, restored_parameter)):
        current.zero_grad(set_to_none=True)
        with pytest.raises(RuntimeError, match="no parameter gradients"):
            current.step()
        value.grad = value.new_tensor([-2.0, 3.0])
        current.step()
    torch.testing.assert_close(restored_parameter, parameter)
    assert not torch.equal(parameter, expected)


@pytest.mark.skipif(
    not (_has_hydra and _has_omegaconf and _has_gym),
    reason="requires hydra, omegaconf, and gym",
)
class TestDreamerV3CompileStrategy:
    """`optimization.compile` resolves to concrete decisions, and the stepper follows them."""

    @staticmethod
    def _load(monkeypatch, run_name):
        example_dir = Path(__file__).parents[2] / "sota-implementations/dreamer_v3"
        monkeypatch.syspath_prepend(str(example_dir))
        return example_dir, runpy.run_path(example_dir / "train.py", run_name=run_name)

    def test_resolve_compile_settings(self, monkeypatch):
        from omegaconf import OmegaConf

        example_dir, example = self._load(monkeypatch, "dreamer_v3_compile_settings")
        resolve = example["resolve_compile_settings"]
        cpu, cuda = torch.device("cpu"), torch.device("cuda")
        cfg = OmegaConf.load(example_dir / "config.yaml")

        # auto: the whole fast path on CUDA, eager elsewhere.
        fast = resolve(cfg, cuda)
        assert (fast.train_step, fast.rssm, fast.cudagraph) == (True, "scan", True)
        assert fast.scan_unroll == cfg.optimization.rssm_scan_unroll
        assert fast.enabled
        eager = resolve(cfg, cpu)
        assert (eager.train_step, eager.rssm, eager.cudagraph) == (False, None, False)
        assert not eager.enabled

        # off: nothing, unless a switch is set explicitly.
        cfg.optimization.compile = "off"
        assert not resolve(cfg, cuda).enabled
        cfg.optimization.compile_rssm = "scan"
        explicit = resolve(cfg, cuda)
        assert (explicit.train_step, explicit.rssm, explicit.cudagraph) == (
            False,
            "scan",
            False,
        )

        # an explicit switch overrides one decision of auto.
        cfg.optimization.compile = "auto"
        cfg.optimization.compile_rssm = None
        cfg.optimization.compile_train_step = False
        partial = resolve(cfg, cuda)
        assert (partial.train_step, partial.rssm, partial.cudagraph) == (
            False,
            "scan",
            True,
        )

        cfg.optimization.compile = "sometimes"
        with pytest.raises(ValueError, match="'auto' or 'off'"):
            resolve(cfg, cuda)
        cfg.optimization.compile = "auto"
        cfg.optimization.cudagraph_train_step = True
        with pytest.raises(ValueError, match="requires a CUDA training device"):
            resolve(cfg, cpu)

    def test_stepper_defaults_follow_the_device(self, monkeypatch):
        from torchrl.trainers.algorithms import DreamerV3OptimizationStepper

        example_dir, example = self._load(monkeypatch, "dreamer_v3_stepper_defaults")
        cfg = TestDreamerV3()._small_sota_config(
            example_dir, compile_train_step=False, cudagraph_train_step=False
        )
        cfg.optimization.compile_rssm = None
        device = torch.device("cpu")
        learner = example["_build_learner"](cfg, device, 3, 1)
        loss_module = example["_make_learner_update"](cfg, device, learner).loss_module
        stepper = DreamerV3OptimizationStepper(
            loss_module, learner.optimizer, learner.value_target_updater
        )
        assert stepper.resolve(torch.device("cuda")) == (True, True)
        assert stepper.resolve(device) == (False, False)
        # On CPU the defaults run eagerly without a warm-up.
        sample = example["_fake_learner_sample"](cfg, device, 3, 1)
        before = [parameter.detach().clone() for parameter in loss_module.parameters()]
        stepper.step(None, sample)
        assert any(
            not torch.equal(parameter, previous)
            for parameter, previous in zip(loss_module.parameters(), before)
        )

    def test_compiled_step_selects_the_scan_for_untouched_rollouts(self, monkeypatch):
        example_dir, example = self._load(monkeypatch, "dreamer_v3_stepper_scan")
        cfg = TestDreamerV3()._small_sota_config(
            example_dir, compile_train_step=True, cudagraph_train_step=False
        )
        device = torch.device("cpu")
        learner = example["_build_learner"](cfg, device, 3, 1)
        update = example["_make_learner_update"](cfg, device, learner)
        rollouts = [
            module
            for module in update.loss_module.modules()
            if isinstance(module, RSSMRolloutV3)
        ]
        assert rollouts
        # The builder left the rollouts eager for the compiled step.
        assert all(
            rollout._scan_fn is None and rollout._step_fn is None
            for rollout in rollouts
        )
        selected = update._select_scan_backends()
        assert selected == rollouts
        assert all(
            isinstance(rollout._scan_fn, ft.partial)
            and rollout._scan_fn.keywords
            == {"unroll": cfg.optimization.rssm_scan_unroll}
            for rollout in rollouts
        )
        # A rollout with a backend keeps it, and the selection is idempotent.
        assert update._select_scan_backends() == []
        update.rssm_scan_unroll = None
        assert update._select_scan_backends() == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
