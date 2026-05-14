# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
from packaging import version
from tensordict import TensorDict
from torchrl.data.tensor_specs import Bounded
from torchrl.modules import SafeModule
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
