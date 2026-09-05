# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import copy
import functools as ft
import importlib.util
from unittest import mock

import pytest
import torch
from packaging import version
from pyvers import implement_for
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.nn import functional as F
from torchrl.data.tensor_specs import Bounded
from torchrl.modules import SafeModule
from torchrl.modules.models._dreamer_v3_block_gru_triton import (
    _has_triton as _has_dreamer_v3_triton,
)
from torchrl.modules.models.model_based import (
    _DreamerV3BlockLinear,
    _DreamerV3RMSNorm,
    _straight_through_categorical,
    DreamerActor,
    DreamerV3BlockGRU,
    DreamerV3BlockGRUCell,
    DreamerV3MLP,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPosteriorV3,
    RSSMPrior,
    RSSMPriorV3,
    RSSMRollout,
    RSSMRolloutV3,
)
from torchrl.testing import get_default_devices


_has_hoptorch = importlib.util.find_spec("hoptorch") is not None


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("batch_size", [[], [3], [5]])
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.11.0"),
    reason="""Dreamer works with batches of null to 2 dimensions. Torch < 1.11
requires one-dimensional batches (for RNN and Conv nets for instance). If you'd like
to see torch < 1.11 supported for dreamer, please submit an issue.""",
)
class TestDreamerComponents:
    @pytest.mark.parametrize("out_features", [3, 5])
    @pytest.mark.parametrize("temporal_size", [[], [2], [4]])
    def test_dreamer_actor(self, device, batch_size, temporal_size, out_features):
        actor = DreamerActor(
            out_features,
        ).to(device)
        emb = torch.randn(*batch_size, *temporal_size, 15, device=device)
        state = torch.randn(*batch_size, *temporal_size, 2, device=device)
        loc, scale = actor(emb, state)
        assert loc.shape == (*batch_size, *temporal_size, out_features)
        assert scale.shape == (*batch_size, *temporal_size, out_features)
        assert torch.all(scale > 0)

    @pytest.mark.parametrize("depth", [32, 64])
    @pytest.mark.parametrize("temporal_size", [[], [2], [4]])
    def test_dreamer_encoder(self, device, temporal_size, batch_size, depth):
        encoder = ObsEncoder(channels=depth).to(device)
        obs = torch.randn(*batch_size, *temporal_size, 3, 64, 64, device=device)
        emb = encoder(obs)
        assert emb.shape == (*batch_size, *temporal_size, depth * 8 * 4)

    @pytest.mark.parametrize("depth", [32, 64])
    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    @pytest.mark.parametrize("temporal_size", [[], [2], [4]])
    def test_dreamer_decoder(
        self, device, batch_size, temporal_size, depth, stoch_size, deter_size
    ):
        decoder = ObsDecoder(channels=depth).to(device)
        stoch_state = torch.randn(
            *batch_size, *temporal_size, stoch_size, device=device
        )
        det_state = torch.randn(*batch_size, *temporal_size, deter_size, device=device)
        obs = decoder(stoch_state, det_state)
        assert obs.shape == (*batch_size, *temporal_size, 3, 64, 64)

    @pytest.mark.parametrize("depth", [32, 64])
    @pytest.mark.parametrize("out_channels", [1, 3])
    @pytest.mark.parametrize("stoch_size", [10])
    @pytest.mark.parametrize("deter_size", [20])
    def test_dreamer_decoder_out_channels(
        self, device, batch_size, depth, out_channels, stoch_size, deter_size
    ):
        decoder = ObsDecoder(channels=depth, out_channels=out_channels).to(device)
        stoch_state = torch.randn(*batch_size, stoch_size, device=device)
        det_state = torch.randn(*batch_size, deter_size, device=device)
        obs = decoder(stoch_state, det_state)
        assert obs.shape == (*batch_size, out_channels, 64, 64)

    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    @pytest.mark.parametrize("action_size", [3, 6])
    def test_rssm_prior(self, device, batch_size, stoch_size, deter_size, action_size):
        action_spec = Bounded(shape=(action_size,), dtype=torch.float32, low=-1, high=1)
        rssm_prior = RSSMPrior(
            action_spec,
            hidden_dim=stoch_size,
            rnn_hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)
        state = torch.randn(*batch_size, deter_size, device=device)
        action = torch.randn(*batch_size, action_size, device=device)
        belief = torch.randn(*batch_size, stoch_size, device=device)
        prior_mean, prior_std, next_state, belief = rssm_prior(state, belief, action)
        assert prior_mean.shape == (*batch_size, deter_size)
        assert prior_std.shape == (*batch_size, deter_size)
        assert next_state.shape == (*batch_size, deter_size)
        assert belief.shape == (*batch_size, stoch_size)
        assert torch.all(prior_std > 0)

    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    def test_rssm_posterior(self, device, batch_size, stoch_size, deter_size):
        rssm_posterior = RSSMPosterior(
            hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)
        belief = torch.randn(*batch_size, stoch_size, device=device)
        obs_emb = torch.randn(*batch_size, 1024, device=device)
        # Init of lazy linears
        _ = rssm_posterior(belief.clone(), obs_emb.clone())

        torch.manual_seed(0)
        posterior_mean, posterior_std, next_state = rssm_posterior(
            belief.clone(), obs_emb.clone()
        )
        assert posterior_mean.shape == (*batch_size, deter_size)
        assert posterior_std.shape == (*batch_size, deter_size)
        assert next_state.shape == (*batch_size, deter_size)
        assert torch.all(posterior_std > 0)

        torch.manual_seed(0)
        posterior_mean_bis, posterior_std_bis, next_state_bis = rssm_posterior(
            belief.clone(), obs_emb.clone()
        )
        assert torch.allclose(posterior_mean, posterior_mean_bis)
        assert torch.allclose(posterior_std, posterior_std_bis)
        assert torch.allclose(next_state, next_state_bis)

    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    @pytest.mark.parametrize("temporal_size", [2, 4])
    @pytest.mark.parametrize("action_size", [3, 6])
    def test_rssm_rollout(
        self, device, batch_size, temporal_size, stoch_size, deter_size, action_size
    ):
        action_spec = Bounded(shape=(action_size,), dtype=torch.float32, low=-1, high=1)
        rssm_prior = RSSMPrior(
            action_spec,
            hidden_dim=stoch_size,
            rnn_hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)
        rssm_posterior = RSSMPosterior(
            hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)

        rssm_rollout = RSSMRollout(
            SafeModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_mean"),
                    ("next", "prior_std"),
                    "_",
                    ("next", "belief"),
                ],
            ),
            SafeModule(
                rssm_posterior,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[
                    ("next", "posterior_mean"),
                    ("next", "posterior_std"),
                    ("next", "state"),
                ],
            ),
        )

        state = torch.randn(*batch_size, temporal_size, deter_size, device=device)
        belief = torch.randn(*batch_size, temporal_size, stoch_size, device=device)
        action = torch.randn(*batch_size, temporal_size, action_size, device=device)
        obs_emb = torch.randn(*batch_size, temporal_size, 1024, device=device)

        tensordict = TensorDict(
            {
                "state": state.clone(),
                "action": action.clone(),
                "next": {
                    "encoded_latents": obs_emb.clone(),
                    "belief": belief.clone(),
                },
            },
            device=device,
            batch_size=torch.Size([*batch_size, temporal_size]),
        )
        ## Init of lazy linears
        _ = rssm_rollout(tensordict.clone())
        torch.manual_seed(0)
        rollout = rssm_rollout(tensordict)
        assert rollout["next", "prior_mean"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "prior_std"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "state"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "belief"].shape == (
            *batch_size,
            temporal_size,
            stoch_size,
        )
        assert rollout["next", "posterior_mean"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "posterior_std"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert torch.all(rollout["next", "prior_std"] > 0)
        assert torch.all(rollout["next", "posterior_std"] > 0)

        state[..., 1:, :] = 0
        belief[..., 1:, :] = 0
        # Only the first state is used for the prior. The rest are recomputed

        tensordict_bis = TensorDict(
            {
                "state": state.clone(),
                "action": action.clone(),
                "next": {"encoded_latents": obs_emb.clone(), "belief": belief.clone()},
            },
            device=device,
            batch_size=torch.Size([*batch_size, temporal_size]),
        )
        torch.manual_seed(0)
        rollout_bis = rssm_rollout(tensordict_bis)

        assert torch.allclose(
            rollout["next", "prior_mean"], rollout_bis["next", "prior_mean"]
        ), (rollout["next", "prior_mean"] - rollout_bis["next", "prior_mean"]).norm()
        assert torch.allclose(
            rollout["next", "prior_std"], rollout_bis["next", "prior_std"]
        )
        assert torch.allclose(rollout["next", "state"], rollout_bis["next", "state"])
        assert torch.allclose(rollout["next", "belief"], rollout_bis["next", "belief"])
        assert torch.allclose(
            rollout["next", "posterior_mean"], rollout_bis["next", "posterior_mean"]
        )
        assert torch.allclose(
            rollout["next", "posterior_std"], rollout_bis["next", "posterior_std"]
        )


class TestDreamerV3Components:
    def test_reference_normalization_and_block_fan_in(self):
        norm = _DreamerV3RMSNorm(8)
        assert set(dict(norm.named_parameters())) == {"weight"}
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
        value = torch.randn(3, 8, dtype=torch.bfloat16)
        expected = (
            value.float()
            * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + norm.eps)
            * norm.weight
        )
        torch.testing.assert_close(norm(value), expected.to(value.dtype))

        torch.manual_seed(0)
        one_block = _DreamerV3BlockLinear(1024, 1024, num_blocks=1)
        torch.manual_seed(1)
        eight_blocks = _DreamerV3BlockLinear(1024, 1024, num_blocks=8)
        ratio = eight_blocks.weight.std() / one_block.weight.std()
        assert ratio.item() == pytest.approx(1.0, rel=0.05)

    @staticmethod
    def _make_rollout(device):
        prior = RSSMPriorV3(
            action_shape=(2,),
            hidden_dim=8,
            rnn_hidden_dim=8,
            num_categoricals=2,
            num_classes=4,
            action_dim=2,
            recurrent_model="block_gru",
            num_blocks=2,
            device=device,
        )
        posterior = RSSMPosteriorV3(
            hidden_dim=8,
            num_categoricals=2,
            num_classes=4,
            rnn_hidden_dim=8,
            obs_embed_dim=6,
            device=device,
        )
        return RSSMRolloutV3(
            TensorDictModule(
                prior,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_logits"),
                    ("next", "state"),
                    ("next", "belief"),
                ],
            ),
            TensorDictModule(
                posterior,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[("next", "posterior_logits"), ("next", "state")],
            ),
        )

    @staticmethod
    def _make_rollout_data(device):
        return TensorDict(
            {
                "state": torch.zeros(2, 4, 8, device=device),
                "belief": torch.zeros(2, 4, 8, device=device),
                "action": torch.randn(2, 4, 2, device=device),
                "is_init": torch.tensor(
                    [[[True], [False], [True], [False]]], device=device
                ).expand(2, -1, -1),
                "next": {"encoded_latents": torch.randn(2, 4, 6, device=device)},
            },
            [2, 4],
        )

    def test_mlp_output_scale_and_multiple_inputs(self):
        module = DreamerV3MLP(
            6,
            4,
            depth=2,
            num_cells=8,
            outscale=0.0,
        )
        output = module(torch.randn(3, 2), torch.randn(3, 4))
        torch.testing.assert_close(output, torch.zeros_like(output))

    @pytest.mark.parametrize("device", get_default_devices())
    def test_block_gru_reference_fixture(self, device):
        prior = RSSMPriorV3(
            action_shape=(2,),
            hidden_dim=4,
            rnn_hidden_dim=4,
            num_categoricals=2,
            num_classes=2,
            action_dim=2,
            recurrent_model="block_gru",
            num_blocks=2,
            num_layers=1,
            prior_num_layers=1,
            device=device,
        )
        with torch.no_grad():
            for parameter in prior.parameters():
                parameter.copy_(
                    torch.linspace(
                        -0.2,
                        0.2,
                        parameter.numel(),
                        device=device,
                    ).reshape_as(parameter)
                )
        state = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=device)
        belief = torch.tensor([[0.1, -0.2, 0.3, -0.4]], device=device)
        action = torch.tensor([[2.0, -4.0]], device=device)

        torch.manual_seed(0)
        logits, sampled_state, next_belief = prior(state, belief, action)

        torch.testing.assert_close(
            logits,
            torch.tensor(
                [[[-0.2538432, -0.0850009], [0.0838414, 0.2526837]]],
                device=device,
            ),
            atol=5e-5,
            rtol=5e-5,
        )
        torch.testing.assert_close(
            next_belief,
            torch.tensor(
                [[0.0575409, -0.1607322, 0.2253285, -0.2480658]], device=device
            ),
            atol=5e-5,
            rtol=5e-5,
        )
        assert sampled_state.shape == (1, 4)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.7.0"),
        reason="hoptorch requires torch >= 2.7.0",
    )
    @pytest.mark.skipif(not _has_hoptorch, reason="hoptorch is not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(("num_blocks", "batch", "time"), [(1, 1, 1), (8, 3, 5)])
    def test_public_block_gru_sequence_forward_parity(
        self, device, dtype, num_blocks, batch, time
    ):
        torch.manual_seed(0)
        kwargs = {
            "input_size": 6,
            "hidden_size": 8,
            "projection_size": 4,
            "num_blocks": num_blocks,
            "num_layers": 2,
            "activation_class": torch.nn.Tanh if num_blocks == 1 else torch.nn.SiLU,
            "device": device,
        }
        reference = DreamerV3BlockGRU(**kwargs)
        scan = DreamerV3BlockGRU(**kwargs, recurrent_backend="scan")
        scan.load_state_dict(reference.state_dict())
        cell = DreamerV3BlockGRUCell(**kwargs)
        cell.load_state_dict(reference.cell.state_dict())
        value = torch.randn(batch, time, 6, device=device, dtype=dtype)
        initial = torch.randn(batch, 8, device=device, dtype=dtype)
        is_init = torch.zeros(batch, time, 1, device=device, dtype=torch.bool)
        if time > 1:
            is_init[0, 2] = True
            is_init[-1, 0] = True

        expected, expected_final = reference(value, initial, is_init)
        actual, actual_final = scan(value, initial, is_init)

        hidden = initial
        cell_outputs = []
        for value_t, init_t in zip(value.unbind(1), is_init.unbind(1)):
            hidden = torch.where(init_t, 0, hidden)
            hidden = cell(value_t, hidden)
            cell_outputs.append(hidden)
        cell_output = torch.stack(cell_outputs, 1)

        tolerance = {"atol": 2e-2, "rtol": 2e-2} if dtype is torch.bfloat16 else {}
        torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(actual_final, expected_final, **tolerance)
        torch.testing.assert_close(cell_output, expected, **tolerance)
        if time == 1:
            default_output, default_final = reference(value)
            zero_output, zero_final = reference(
                value, torch.zeros_like(initial), torch.zeros_like(is_init)
            )
            torch.testing.assert_close(default_output, zero_output, **tolerance)
            torch.testing.assert_close(default_final, zero_final, **tolerance)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.7.0"),
        reason="hoptorch requires torch >= 2.7.0",
    )
    @pytest.mark.skipif(not _has_hoptorch, reason="hoptorch is not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_blocks", [1, 8])
    def test_public_block_gru_sequence_gradient_parity(self, device, dtype, num_blocks):
        torch.manual_seed(1)
        kwargs = {
            "input_size": 6,
            "hidden_size": 8,
            "projection_size": 4,
            "num_blocks": num_blocks,
            "num_layers": 2,
            "activation_class": torch.nn.Tanh if num_blocks == 1 else torch.nn.SiLU,
            "device": device,
        }
        reference = DreamerV3BlockGRU(**kwargs)
        scan = DreamerV3BlockGRU(**kwargs, recurrent_backend="scan")
        scan.load_state_dict(reference.state_dict())
        value_source = torch.randn(3, 5, 6, device=device, dtype=dtype)
        hidden_source = torch.randn(3, 8, device=device, dtype=dtype)
        is_init = torch.tensor(
            [
                [False, False, True, False, False],
                [True, False, False, False, True],
                [False, False, False, False, False],
            ],
            device=device,
        )
        output_cotangent = torch.randn(3, 5, 8, device=device, dtype=dtype) / (
            3 * 5 * 8
        )
        hidden_cotangent = torch.randn(3, 8, device=device, dtype=dtype) / (3 * 8)

        def run(module):
            value = value_source.detach().clone().requires_grad_()
            hidden = hidden_source.detach().clone().requires_grad_()
            output, final_hidden = module(value, hidden, is_init)
            loss = (output * output_cotangent).float().sum() + (
                final_hidden * hidden_cotangent
            ).float().sum()
            loss.backward()
            return (
                output.detach(),
                final_hidden.detach(),
                value.grad,
                hidden.grad,
                {name: parameter.grad for name, parameter in module.named_parameters()},
            )

        expected = run(reference)
        actual = run(scan)
        tolerance = (
            {"atol": 3e-2, "rtol": 5e-2}
            if dtype is torch.bfloat16
            else {"atol": 2e-5, "rtol": 2e-5}
        )
        for expected_value, actual_value in zip(expected[:4], actual[:4]):
            torch.testing.assert_close(actual_value, expected_value, **tolerance)
        assert actual[4].keys() == expected[4].keys()
        for name in expected[4]:
            torch.testing.assert_close(actual[4][name], expected[4][name], **tolerance)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.7.0"),
        reason="hoptorch requires torch >= 2.7.0",
    )
    @pytest.mark.skipif(not _has_hoptorch, reason="hoptorch is not installed")
    def test_public_block_gru_scan_compile_recurrent_loss(self):
        module = DreamerV3BlockGRU(
            6,
            8,
            projection_size=4,
            num_blocks=2,
            num_layers=2,
            recurrent_backend="scan",
        )
        compiled = torch.compile(module, fullgraph=True)
        value = torch.randn(2, 5, 6, requires_grad=True)
        hidden = torch.randn(2, 8, requires_grad=True)
        is_init = torch.tensor(
            [[False, False, True, False, False], [True, False, False, False, True]]
        )

        output, final_hidden = compiled(value, hidden, is_init)
        prediction = output[:, :-1, :6]
        target = value.detach()[:, 1:]
        loss = (
            F.smooth_l1_loss(prediction, target) + 0.01 * final_hidden.square().mean()
        )
        loss.backward()

        assert value.grad is not None
        assert hidden.grad is not None
        assert all(parameter.grad is not None for parameter in module.parameters())

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.7.0"),
        reason="hoptorch requires torch >= 2.7.0",
    )
    @pytest.mark.skipif(not _has_hoptorch, reason="hoptorch is not installed")
    def test_public_block_gru_scan_double_backward_raises(self):
        torch.manual_seed(0)
        module = DreamerV3BlockGRU(
            6,
            8,
            projection_size=4,
            num_blocks=2,
            recurrent_backend="scan",
        )
        value = torch.randn(2, 5, 6, requires_grad=True)
        is_init = torch.zeros(2, 5, dtype=torch.bool)
        output, _ = module(value, torch.zeros(2, 8), is_init)
        cotangent = torch.randn_like(output).requires_grad_()
        (grad,) = torch.autograd.grad(
            output, value, grad_outputs=cotangent, create_graph=True
        )
        with pytest.raises(
            RuntimeError, match="differentiate twice|does not require grad"
        ):
            grad.sum().backward()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.7.0"),
        reason="hoptorch requires torch >= 2.7.0",
    )
    @pytest.mark.skipif(not _has_hoptorch, reason="hoptorch is not installed")
    def test_public_block_gru_scan_mixed_dtype_promotes(self):
        torch.manual_seed(0)
        kwargs = {
            "input_size": 6,
            "hidden_size": 8,
            "projection_size": 4,
            "num_blocks": 2,
        }
        reference = DreamerV3BlockGRU(**kwargs)
        scan_module = DreamerV3BlockGRU(**kwargs, recurrent_backend="scan")
        scan_module.load_state_dict(reference.state_dict())
        value = torch.randn(2, 5, 6, dtype=torch.bfloat16)
        hidden = torch.randn(2, 8)
        is_init = torch.zeros(2, 5, dtype=torch.bool)
        is_init[1, 2] = True
        expected_output, expected_hidden = reference(value, hidden, is_init)
        output, final_hidden = scan_module(value, hidden, is_init)
        assert output.dtype == expected_output.dtype
        assert final_hidden.dtype == expected_hidden.dtype
        torch.testing.assert_close(output, expected_output, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(final_hidden, expected_hidden, atol=2e-5, rtol=2e-5)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_block_gru_action_normalization_and_gradients(self, device):
        prior = RSSMPriorV3(
            action_shape=(2,),
            hidden_dim=8,
            rnn_hidden_dim=8,
            num_categoricals=2,
            num_classes=4,
            action_dim=2,
            recurrent_model="block_gru",
            num_blocks=2,
            device=device,
        )
        state = torch.randn(3, 8, device=device, requires_grad=True)
        belief = torch.randn(3, 8, device=device, requires_grad=True)
        action = torch.tensor([[2.0, -4.0], [0.5, -0.25], [-3.0, 2.0]], device=device)
        normalized_action = action / action.abs().clamp_min(1)

        torch.manual_seed(0)
        logits, _, next_belief = prior(state, belief, action)
        torch.manual_seed(0)
        normalized_logits, _, normalized_belief = prior(
            state, belief, normalized_action
        )

        torch.testing.assert_close(logits, normalized_logits)
        torch.testing.assert_close(next_belief, normalized_belief)
        (logits.square().mean() + next_belief.square().mean()).backward()
        assert state.grad is not None
        assert belief.grad is not None
        assert all(parameter.grad is not None for parameter in prior.parameters())

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.4.0"),
        reason="the native RMSNorm compile path requires Torch >= 2.4.0",
    )
    def test_block_gru_torch_compile(self):
        prior = RSSMPriorV3(
            action_shape=(2,),
            hidden_dim=8,
            rnn_hidden_dim=8,
            num_categoricals=2,
            num_classes=4,
            action_dim=2,
            recurrent_model="block_gru",
            num_blocks=2,
        )
        state = torch.randn(3, 8)
        belief = torch.randn(3, 8)
        action = torch.randn(3, 2)
        uniform = torch.rand(3, 2)
        compiled = torch.compile(prior, fullgraph=True)

        expected = prior(state, belief, action, _uniform=uniform)
        actual = compiled(state, belief, action, _uniform=uniform)
        for expected_item, actual_item in zip(expected, actual):
            torch.testing.assert_close(expected_item, actual_item)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_posterior_rms_norm(self, device):
        posterior = RSSMPosteriorV3(
            hidden_dim=8,
            num_categoricals=2,
            num_classes=4,
            rnn_hidden_dim=8,
            obs_embed_dim=6,
            use_rms_norm=True,
            num_layers=1,
            device=device,
        )
        belief = torch.randn(3, 8, device=device)
        embedding = torch.randn(3, 6, device=device)
        logits, state = posterior(belief, embedding)
        assert logits.shape == (3, 2, 4)
        assert state.shape == (3, 8)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_key", ["action", ("agent", "action")])
    def test_rssm_rollout_masks_action_on_reset(self, device, action_key):
        num_categoricals = num_classes = 2
        state_dim = num_categoricals * num_classes
        belief_dim = 4
        action_dim = 2
        embedding_dim = 3
        prior = RSSMPriorV3(
            action_shape=(action_dim,),
            hidden_dim=belief_dim,
            rnn_hidden_dim=belief_dim,
            num_categoricals=num_categoricals,
            num_classes=num_classes,
            action_dim=action_dim,
            recurrent_model="block_gru",
            num_blocks=2,
            device=device,
        )
        posterior = RSSMPosteriorV3(
            hidden_dim=belief_dim,
            num_categoricals=num_categoricals,
            num_classes=num_classes,
            rnn_hidden_dim=belief_dim,
            obs_embed_dim=embedding_dim,
            device=device,
        )
        rollout = RSSMRolloutV3(
            TensorDictModule(
                prior,
                in_keys=["state", "belief", action_key],
                out_keys=[
                    ("next", "prior_logits"),
                    ("next", "state"),
                    ("next", "belief"),
                ],
            ),
            TensorDictModule(
                posterior,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[("next", "posterior_logits"), ("next", "state")],
            ),
        )

        def run(action):
            tensordict = TensorDict(
                {
                    "state": torch.zeros(1, 2, state_dim, device=device),
                    "belief": torch.zeros(1, 2, belief_dim, device=device),
                    "is_init": torch.ones(1, 2, 1, dtype=torch.bool, device=device),
                    "next": {
                        "encoded_latents": torch.zeros(
                            1, 2, embedding_dim, device=device
                        )
                    },
                },
                [1, 2],
            )
            tensordict.set(action_key, action)
            torch.manual_seed(0)
            return rollout(tensordict)

        zero_action = torch.zeros(1, 2, action_dim, device=device)
        nonzero_action = torch.ones_like(zero_action)
        torch.testing.assert_close(
            run(zero_action)["next", "belief"],
            run(nonzero_action)["next", "belief"],
        )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_rssm_rollout_fast_path_matches_tensordict_path(self, device):
        fast = self._make_rollout(device)
        slow = copy.deepcopy(fast)
        slow._fast_path = False
        assert fast._fast_path
        data = self._make_rollout_data(device)

        def deterministic_sample(logits, unimix=0.0, uniform=None):
            uniform = logits.new_full(logits.shape[:-1], 0.5)
            return _straight_through_categorical(logits, unimix, uniform)

        with mock.patch(
            "torchrl.modules.models.model_based._straight_through_categorical",
            side_effect=deterministic_sample,
        ):
            fast_output = fast(data.clone())
            slow_output = slow(data.clone())
        for key in fast.out_keys:
            torch.testing.assert_close(fast_output[key], slow_output[key])

        fast_loss = sum(fast_output[key].square().mean() for key in fast.out_keys)
        slow_loss = sum(slow_output[key].square().mean() for key in slow.out_keys)
        fast_gradients = torch.autograd.grad(
            fast_loss, tuple(fast.parameters()), allow_unused=True
        )
        slow_gradients = torch.autograd.grad(
            slow_loss, tuple(slow.parameters()), allow_unused=True
        )
        for fast_gradient, slow_gradient in zip(fast_gradients, slow_gradients):
            if fast_gradient is None or slow_gradient is None:
                assert fast_gradient is slow_gradient
            else:
                torch.testing.assert_close(fast_gradient, slow_gradient)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("unroll", [1, 3, 8])
    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        reason="the higher-order scan backend requires Torch >= 2.6.0",
    )
    def test_rssm_rollout_higher_order_scan_matches_loop(self, device, unroll):
        scan_rollout = self._make_rollout(device)
        loop_rollout = copy.deepcopy(scan_rollout)
        scan_rollout._scan_fn = ft.partial(scan_rollout._scan, unroll=unroll)
        data = self._make_rollout_data(device)

        torch.manual_seed(0)
        scan_output = scan_rollout(data.clone())
        torch.manual_seed(0)
        loop_output = loop_rollout(data.clone())

        for key in scan_rollout.out_keys:
            torch.testing.assert_close(scan_output[key], loop_output[key])

        scan_loss = sum(
            scan_output[key].square().mean() for key in scan_rollout.out_keys
        )
        loop_loss = sum(
            loop_output[key].square().mean() for key in loop_rollout.out_keys
        )
        scan_gradients = torch.autograd.grad(
            scan_loss, tuple(scan_rollout.parameters())
        )
        loop_gradients = torch.autograd.grad(
            loop_loss, tuple(loop_rollout.parameters())
        )
        for scan_gradient, loop_gradient in zip(scan_gradients, loop_gradients):
            torch.testing.assert_close(
                scan_gradient, loop_gradient, atol=2e-4, rtol=5e-5
            )

    @implement_for("torch", None, "2.6.0", compilable=True)
    @pytest.mark.parametrize(("scope", "unroll"), [("step", 1)])
    def test_rssm_rollout_compile(self, scope, unroll):
        self._test_rssm_rollout_compile(scope, unroll)

    @implement_for("torch", "2.6.0", compilable=True)
    @pytest.mark.parametrize(
        ("scope", "unroll"), [("step", 1), ("scan", 1), ("scan", 3)]
    )
    def test_rssm_rollout_compile(self, scope, unroll):  # noqa: F811
        self._test_rssm_rollout_compile(scope, unroll)

    def _test_rssm_rollout_compile(self, scope, unroll):
        rollout = self._make_rollout(torch.device("cpu"))
        data = self._make_rollout_data(torch.device("cpu"))
        rollout.compile_rollout(scope, unroll=unroll)

        output = rollout(data)
        (
            output["next", "posterior_logits"].square().mean()
            + output["next", "prior_logits"].square().mean()
        ).backward()
        assert all(parameter.grad is not None for parameter in rollout.parameters())


def test_public_block_gru_triton_errors():
    with mock.patch("torchrl.modules.models.model_based._has_dreamer_v3_triton", False):
        with pytest.raises(RuntimeError, match="requires Triton"):
            DreamerV3BlockGRU(6, 8, recurrent_backend="triton")

    class CustomActivation(torch.nn.Module):
        def forward(self, value):
            return value.sigmoid()

    with mock.patch("torchrl.modules.models.model_based._has_dreamer_v3_triton", True):
        with pytest.raises(ValueError, match="supports nn.SiLU, nn.Tanh, and nn.ReLU"):
            DreamerV3BlockGRU(
                6,
                8,
                projection_size=4,
                num_blocks=2,
                activation_class=CustomActivation,
                recurrent_backend="triton",
            )

    with (
        mock.patch("torchrl.modules.models.model_based._has_dreamer_v3_triton", True),
        mock.patch(
            "torchrl.modules.models._dreamer_v3_block_gru_triton._has_triton", True
        ),
    ):
        module = DreamerV3BlockGRU(
            6,
            8,
            projection_size=4,
            num_blocks=2,
            recurrent_backend="triton",
        )
        with pytest.raises(RuntimeError, match="requires CUDA tensors"):
            module(torch.randn(2, 3, 6))
        with pytest.raises(
            ValueError, match="supports torch.float32 and torch.bfloat16"
        ):
            module(
                torch.randn(2, 3, 6, dtype=torch.float64),
                torch.zeros(2, 8, dtype=torch.float64),
            )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_dreamer_v3_triton,
    reason="DreamerV3 Triton backend requires CUDA and Triton 3.3+",
)
@pytest.mark.parametrize(
    (
        "activation_class",
        "num_blocks",
        "num_layers",
        "batch",
        "time",
        "dtype",
        "projection_size",
    ),
    [
        (torch.nn.SiLU, 1, 1, 1, 1, torch.float32, 32),
        (torch.nn.Tanh, 8, 2, 2, 3, torch.float32, 24),
        (torch.nn.ReLU, 8, 1, 3, 2, torch.bfloat16, 32),
        (torch.nn.SiLU, 8, 2, 2, 3, torch.bfloat16, 24),
        (torch.nn.SiLU, 4, 1, 2, 3, torch.float32, 48),
    ],
)
def test_public_block_gru_triton_gradient_parity(
    activation_class, num_blocks, num_layers, batch, time, dtype, projection_size
):
    torch.manual_seed(0)
    kwargs = {
        "input_size": 12,
        "hidden_size": 32,
        "projection_size": projection_size,
        "num_blocks": num_blocks,
        "num_layers": num_layers,
        "activation_class": activation_class,
        "device": "cuda",
    }
    reference = DreamerV3BlockGRU(**kwargs)
    triton_module = DreamerV3BlockGRU(**kwargs, recurrent_backend="triton")
    triton_module.load_state_dict(reference.state_dict())
    value_source = torch.randn(batch, time, 12, device="cuda", dtype=dtype)
    hidden_source = torch.randn(batch, 32, device="cuda", dtype=dtype)
    is_init = torch.zeros(batch, time, dtype=torch.bool, device="cuda")
    if time > 1:
        is_init[0, 1] = True
        is_init[-1, 0] = True
    output_cotangent = torch.randn(batch, time, 32, device="cuda", dtype=dtype) / (
        batch * time * 32
    )
    hidden_cotangent = torch.randn(batch, 32, device="cuda", dtype=dtype) / (batch * 32)

    def run(module):
        module.zero_grad(set_to_none=True)
        value = value_source.detach().clone().requires_grad_()
        hidden = hidden_source.detach().clone().requires_grad_()
        output, final_hidden = module(value, hidden, is_init)
        loss = (output * output_cotangent).float().sum() + (
            final_hidden * hidden_cotangent
        ).float().sum()
        loss.backward()
        return (
            output.detach(),
            final_hidden.detach(),
            value.grad,
            hidden.grad,
            {name: parameter.grad for name, parameter in module.named_parameters()},
        )

    expected = run(reference)
    tolerance = (
        {"atol": 4e-2, "rtol": 6e-2}
        if dtype is torch.bfloat16
        else {"atol": 3e-4, "rtol": 3e-4}
    )
    for _ in range(3):
        actual = run(triton_module)
        for expected_value, actual_value in zip(expected[:4], actual[:4]):
            torch.testing.assert_close(actual_value, expected_value, **tolerance)
        assert actual[4].keys() == expected[4].keys()
        for name in expected[4]:
            torch.testing.assert_close(actual[4][name], expected[4][name], **tolerance)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_dreamer_v3_triton,
    reason="DreamerV3 Triton backend requires CUDA and Triton 3.3+",
)
def test_public_block_gru_triton_frozen_parameter_gradients():
    torch.manual_seed(0)
    kwargs = {
        "input_size": 12,
        "hidden_size": 32,
        "projection_size": 24,
        "num_blocks": 8,
        "num_layers": 2,
        "activation_class": torch.nn.SiLU,
        "device": "cuda",
    }
    reference = DreamerV3BlockGRU(**kwargs)
    triton_module = DreamerV3BlockGRU(**kwargs, recurrent_backend="triton")
    triton_module.load_state_dict(reference.state_dict())
    value_source = torch.randn(2, 3, 12, device="cuda")
    hidden_source = torch.randn(2, 32, device="cuda")
    is_init = torch.zeros(2, 3, dtype=torch.bool, device="cuda")
    is_init[0, 1] = True
    tolerance = {"atol": 3e-4, "rtol": 3e-4}

    def run(module, inputs_require_grad):
        module.zero_grad(set_to_none=True)
        value = value_source.detach().clone().requires_grad_(inputs_require_grad)
        hidden = hidden_source.detach().clone().requires_grad_(inputs_require_grad)
        output, final_hidden = module(value, hidden, is_init)
        (output.square().mean() + final_hidden.square().mean()).backward()
        return value.grad, hidden.grad

    # Frozen world-model rollout: only the inputs receive gradients.
    for module in (reference, triton_module):
        module.requires_grad_(False)
    expected_value_grad, expected_hidden_grad = run(reference, True)
    value_grad, hidden_grad = run(triton_module, True)
    torch.testing.assert_close(value_grad, expected_value_grad, **tolerance)
    torch.testing.assert_close(hidden_grad, expected_hidden_grad, **tolerance)
    assert all(param.grad is None for param in triton_module.parameters())

    # Partial freeze: a scattered trainable subset checks that the backward
    # maps needs_input_grad entries onto the right gradient slots.
    trained = ("cell.dynamic_norms.1.weight", "cell.gates.bias")
    for module in (reference, triton_module):
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(name in trained)
    run(reference, False)
    run(triton_module, False)
    expected_parameters = dict(reference.named_parameters())
    for name, parameter in triton_module.named_parameters():
        if name in trained:
            torch.testing.assert_close(
                parameter.grad, expected_parameters[name].grad, **tolerance
            )
        else:
            assert parameter.grad is None


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_dreamer_v3_triton,
    reason="DreamerV3 Triton backend requires CUDA and Triton 3.3+",
)
def test_public_block_gru_triton_compile_recurrent_loss():
    torch.manual_seed(1)
    kwargs = {
        "input_size": 16,
        "hidden_size": 32,
        "projection_size": 32,
        "num_blocks": 8,
        "num_layers": 2,
        "device": "cuda",
    }
    reference = DreamerV3BlockGRU(**kwargs)
    module = DreamerV3BlockGRU(**kwargs, recurrent_backend="triton")
    module.load_state_dict(reference.state_dict())
    compiled = torch.compile(
        module, fullgraph=True, dynamic=False, mode="reduce-overhead"
    )
    value_source = torch.randn(2, 5, 16, device="cuda", requires_grad=True)
    hidden_source = torch.randn(2, 32, device="cuda", requires_grad=True)
    is_init = torch.tensor(
        [[False, False, True, False, False], [True, False, False, False, True]],
        device="cuda",
    )

    def run(candidate, value, hidden):
        output, final_hidden = candidate(value, hidden, is_init)
        loss = F.smooth_l1_loss(output[:, :-1, :16], value.detach()[:, 1:])
        loss = loss + 0.01 * final_hidden.square().mean()
        loss.backward()
        return output.detach(), final_hidden.detach()

    expected_value = value_source.detach().clone().requires_grad_()
    expected_hidden = hidden_source.detach().clone().requires_grad_()
    expected = run(reference, expected_value, expected_hidden)
    actual = run(compiled, value_source, hidden_source)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(
        value_source.grad, expected_value.grad, atol=3e-4, rtol=3e-4
    )
    torch.testing.assert_close(
        hidden_source.grad, expected_hidden.grad, atol=3e-4, rtol=3e-4
    )
    for parameter, expected_parameter in zip(
        module.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, expected_parameter.grad, atol=3e-4, rtol=3e-4
        )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_dreamer_v3_triton,
    reason="DreamerV3 Triton backend requires CUDA and Triton 3.3+",
)
def test_public_block_gru_triton_mixed_dtype_promotes():
    torch.manual_seed(2)
    kwargs = {
        "input_size": 8,
        "hidden_size": 16,
        "projection_size": 12,
        "num_blocks": 2,
        "device": "cuda",
    }
    reference = DreamerV3BlockGRU(**kwargs)
    triton_module = DreamerV3BlockGRU(**kwargs, recurrent_backend="triton")
    triton_module.load_state_dict(reference.state_dict())
    value = torch.randn(2, 3, 8, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(2, 16, device="cuda")
    is_init = torch.zeros(2, 3, dtype=torch.bool, device="cuda")
    expected_output, expected_hidden = reference(value, hidden, is_init)
    output, final_hidden = triton_module(value, hidden, is_init)
    assert output.dtype == expected_output.dtype
    assert final_hidden.dtype == expected_hidden.dtype
    torch.testing.assert_close(output, expected_output, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(final_hidden, expected_hidden, atol=3e-4, rtol=3e-4)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
