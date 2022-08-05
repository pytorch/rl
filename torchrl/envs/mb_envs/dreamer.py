import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as d
from torch.distributions.kl import kl_divergence
from torchrl.modules.tensordict_module import TensorDictModule, TensorDictSequence
from torchrl.data import TensorDict
from ..model_based import ModelBasedEnv
from math import sqrt


class DreamerEnv(ModelBasedEnv):
    def __init__(
        self,
        obs_depth=32,
        rssm_hidden=200,
        rnn_hidden_dim=200,
        state_dim=20,
        device="cpu",
        dtype=None,
        batch_size=None,
    ):
        super().__init__(
            world_model=DreamerModel(
                obs_depth=obs_depth,
                rssm_hidden=rssm_hidden,
                rnn_hidden_dim=rnn_hidden_dim,
                state_dim=state_dim,
            ),
            reward_model=TensorDictModule(
                RewardModel(),
                in_keys=[
                    "posterior_states",
                    "beliefs",
                ],
                out_keys=["predicted_reward"],
            ),
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

        self.inference_world_model = self.word_model.select_subsequence(
            in_keys=["initial_state", "initial_rnn_hidden", "action"],
            out_keys=["prior_states", "prior_rnn_hiddens"],
        )
        self.inference_reward_model.in_keys = [
            "prior_states",
            "beliefs",
        ]

    def _set_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            list(self.word_model.parameters()) + list(self.reward_model.parameters()),
            lr=1e-3,
            eps=1e-4
        )

    def loss(self, tensordict: TensorDict) -> torch.Tensor:
        B, T, D = tensordict["prior_means"].shape
        flat_prior_means = tensordict["prior_means"].reshape(B * T, D)
        flat_prior_stds = tensordict["prior_stds"].reshape(B * T, D)
        flat_prior = d.Normal(flat_prior_means, flat_prior_stds)
        flat_posterior_means = tensordict["posterior_means"].reshape(B * T, D)
        flat_posterior_stds = tensordict["posterior_stds"].reshape(B * T, D)
        flat_posterior = d.Normal(flat_posterior_means, flat_posterior_stds)
        kl_loss = kl_divergence(flat_posterior, flat_prior).mean()
        reco_loss = (
            0.5
            * F.mse_loss(
                tensordict["observation"],
                tensordict["reco_observation"],
                reduction=None,
            )
            .mean(dim=[0, 1])
            .sum()
        )
        reward_loss = 0.5 * F.mse_loss(
            tensordict["reward"], tensordict["predicted_reward"]
        )
        return kl_loss + reco_loss + reward_loss


class DreamerModel(TensorDictSequence):
    def __init__(self, obs_depth=32, rssm_hidden=200, rnn_hidden_dim=200, state_dim=20):
        super().__init__(
            TensorDictModule(
                ObsEncoder(depth=obs_depth),
                in_keys=["observation"],
                out_keys=["observation_encoded"],
            ),
            TensorDictModule(
                RSSMPrior(
                    hidden_dim=rssm_hidden,
                    rnn_hidden_dim=rnn_hidden_dim,
                    state_dim=state_dim,
                ),
                in_keys=["initial_state", "initial_rnn_hidden", "action"],
                out_keys=["prior_means", "prior_stds", "prior_states", "beliefs"],
            ),
            TensorDictModule(
                RSSMPosterior(
                    hidden_dim=rssm_hidden,
                    state_dim=state_dim,
                ),
                in_keys=["action", "observation_encoded"],
                out_keys=["posterior_means", "posterior_stds", "posterior_states"],
            ),
            TensorDictModule(
                ObsDecoder(depth=obs_depth),
                in_keys=["posterior_states", "beliefs"],
                out_keys=["reco_observation"],
            ),
        )


class ObsEncoder(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyConv2d(depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth * 2, depth * 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth * 4, depth * 8, 4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, observation):
        *batch_sizes, C, H, W = observation.shape
        observation = observation.view(-1, C, H, W)
        obs_encoded = self.encoder(observation)
        latent = obs_encoded.view(*batch_sizes, -1)
        return latent


class ObsDecoder(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.state_to_latent = nn.Sequential(
            nn.LazyLinear(depth * 8 * 4 * 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(depth * 2, depth, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(depth, 3, 4, stride=2, padding=1),
        )
        self._depth = depth

    def forward(self, state, rnn_hidden):
        latent = self.state_to_latent(torch.cat([state, rnn_hidden], dim=-1))
        *batch_sizes, D = latent.shape
        latent = latent.view(-1, D)
        h = w = int(sqrt(D // (8 * self._depth)))
        latent_reshaped = latent.view(latent.shape[0], 8 * self._depth, h, w)
        obs_decoded = self.decoder(latent_reshaped)
        _, C, H, W = obs_decoded.shape
        obs_decoded = obs_decoded.view(*batch_sizes, C, H, W)
        return obs_decoded


class RSSMPrior(nn.Module):
    def __init__(self, hidden_dim=200, rnn_hidden_dim=200, state_dim=20):
        super().__init__()
        self.min_std = 0.1

        ### Prior
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.action_state_projector = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ELU())
        self.rnn_to_prior_projector = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim), nn.ELU()
        )
        self.prior_mean = nn.Linear(hidden_dim, state_dim)
        self.prior_std = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, rnn_hidden, action):
        prior_means = []
        prior_stds = []
        prior_states = []
        beliefs = []
        num_steps = action.size(1)
        for i in range(num_steps):
            prior_mean, prior_std, prior_state, belief = self.rssm_step(
                state, action[:, i], rnn_hidden
            )
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            prior_states.append(prior_state)
            beliefs.append(belief)
            state = prior_state
            rnn_hidden = belief
        prior_means = torch.stack(prior_means, dim=1)
        prior_stds = torch.stack(prior_stds, dim=1)
        prior_states = torch.stack(prior_states, dim=1)
        beliefs = torch.stack(beliefs, dim=1)
        return prior_means, prior_stds, prior_states, beliefs

    def rssm_step(self, state, action, rnn_hidden):
        action_state = self.action_state_projector(torch.cat([state, action], dim=-1))
        rnn_hidden = self.rnn(action_state, rnn_hidden)
        belief = rnn_hidden
        prior = self.rnn_to_prior_projector(belief)
        prior_mean = self.prior_mean(prior)
        prior_std = F.softplus(self.prior_std(prior)) + self.min_std
        prior_state = d.Normal(prior_mean, prior_std).rsample()
        return prior_mean, prior_std, prior_state, belief


class RSSMPosterior(nn.Module):
    def __init__(self, hidden_dim=200, state_dim=20):
        super().__init__()
        self.min_std = 0.1

        self.obs_rnn_to_post_projector = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.ELU()
        )
        self.post_mean = nn.Linear(hidden_dim, state_dim)
        self.post_std = nn.Linear(hidden_dim, state_dim)

    def forward(self, belief, obs_embedding):
        posterior = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        post_mean = self.post_mean(posterior)
        post_std = F.softplus(self.post_std(posterior)) + self.min_std
        post_state = d.Normal(post_mean, post_std).rsample()
        return post_mean, post_std, post_state


class RewardModel(nn.Module):
    def __init__(self, hidden_dim=300, num_layers=3):
        super().__init__()
        layers = (
            [
                nn.LazyLinear(hidden_dim),
                nn.ELU(),
            ]
            + [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
            ]
            * (num_layers - 1)
            + [nn.Linear(hidden_dim, 1)]
        )
        self.reward_model = nn.Sequential(*layers)

    def forward(self, state, belief):
        reward = self.reward_model(torch.cat([state, belief], dim=-1))
        return reward
