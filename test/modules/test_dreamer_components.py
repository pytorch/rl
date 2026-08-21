# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import copy

import pytest
import torch
from packaging import version
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data.tensor_specs import Bounded
from torchrl.modules import SafeModule
from torchrl.modules.models.model_based import (
    _DreamerV3BlockLinear,
    _DreamerV3RMSNorm,
    DreamerActor,
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
        compiled = torch.compile(prior, fullgraph=True)

        torch.manual_seed(0)
        expected = prior(state, belief, action)
        torch.manual_seed(0)
        actual = compiled(state, belief, action)
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

        torch.manual_seed(0)
        fast_output = fast(data.clone())
        torch.manual_seed(0)
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
    def test_rssm_rollout_higher_order_scan_matches_loop(self, device):
        scan_rollout = self._make_rollout(device)
        loop_rollout = copy.deepcopy(scan_rollout)
        scan_rollout._scan_fn = scan_rollout._scan
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

    @pytest.mark.parametrize("scope", ["step", "scan"])
    def test_rssm_rollout_compile(self, scope):
        rollout = self._make_rollout(torch.device("cpu"))
        data = self._make_rollout_data(torch.device("cpu"))
        rollout.compile_rollout(scope)

        output = rollout(data)
        (
            output["next", "posterior_logits"].square().mean()
            + output["next", "prior_logits"].square().mean()
        ).backward()
        assert all(parameter.grad is not None for parameter in rollout.parameters())


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
