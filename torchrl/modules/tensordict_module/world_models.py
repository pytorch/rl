# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



from math import sqrt

import torch
import torch.distributions as d
import torch.nn as nn
import torch.nn.functional as F

from torchrl.modules.tensordict_module import TensorDictModule, TensorDictSequence

__all__ = [
    "DreamerWorldModeler",
    "WorldModelWrapper"
]

class WorldModelWrapper(TensorDictSequence):
    """
    World model wrapper.
    This module wraps together a world model and a reward model.
    The world model is used to predict an imaginary world state.
    The reward model is used to predict the reward of the imaginary world state.

    Args:
        world_model (TensorDictModule): a world model that generates a world state.
        reward_model (TensorDictModule): a reward model, that reads the world state and returns a reward

    """

    def __init__(
        self,
        world_modeler_operator: TensorDictModule,
        reward_operator: TensorDictModule,
    ):
        super().__init__(
            world_modeler_operator,
            reward_operator,
        )

    def get_world_modeler_operator(self) -> TensorDictSequence:
        """

        Returns a stand-alone policy operator that maps an observation to an action.

        """
        return self.module[0]

    def get_reward_operator(self) -> TensorDictSequence:
        """

        Returns a stand-alone value network operator that maps an observation to a value estimate.

        """
        return self.module[1]



class DreamerWorldModeler(TensorDictSequence):
    def __init__(self, obs_depth=32, rssm_hidden=200, rnn_hidden_dim=200, state_dim=20):
        super().__init__(
            TensorDictModule(
                ObsEncoder(depth=obs_depth),
                in_keys=["pixels"],
                out_keys=["observations_encoded"],
            ),
            TensorDictModule(
                RSSMPrior(
                    hidden_dim=rssm_hidden,
                    rnn_hidden_dim=rnn_hidden_dim,
                    state_dim=state_dim,
                ),
                in_keys=["prior_state", "belief", "action"],
                out_keys=["prior_means", "prior_stds", "next_prior_state", "next_belief"],
            ),
            TensorDictModule(
                RSSMPosterior(
                    hidden_dim=rssm_hidden,
                    state_dim=state_dim,
                ),
                in_keys=["next_belief", "observations_encoded"],
                out_keys=["posterior_means", "posterior_stds", "posterior_states"],
            ),
            TensorDictModule(
                ObsDecoder(depth=obs_depth),
                in_keys=["posterior_states", "next_belief"],
                out_keys=["reco_pixels"],
            ),
        )


class ObsEncoder(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyConv2d(depth, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(depth, depth * 2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(depth * 2, depth * 4, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(depth * 4, depth * 8, 4, stride=2),
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
            nn.LazyLinear(depth * 8 * 2 * 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.LazyConvTranspose2d(depth * 4, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth * 2, depth, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth, 3, 6, stride=2),
        )
        self._depth = depth

    def forward(self, state, rnn_hidden):
        latent = self.state_to_latent(torch.cat([state, rnn_hidden], dim=-1))
        *batch_sizes, D = latent.shape
        latent = latent.view(-1, D, 1, 1)
        obs_decoded = self.decoder(latent)
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

    def forward(self, prior_state, belief, action):
        prior_means = []
        prior_stds = []
        prior_states = []
        beliefs = []
        if prior_state.dim() == 3:
            if prior_state.shape[1] == 1:
                prior_state = prior_state.squeeze(1)
            else:
                raise ValueError("prior_state should be a single step")
        if belief.dim() == 3:
            if belief.shape[1] == 1:
                belief = belief.squeeze(1)
            else:
                raise ValueError("belief should be a single step")
        if len(action.shape) == 2:
            action = action.unsqueeze(1)
        elif len(action.shape) == 3:
            pass
        else:
            raise ValueError("Action must be a 3D tensor of shape BxTxD or 2D with shape BxD")
        num_steps = action.shape[1]

        for i in range(num_steps):
            prior_mean, prior_std, prior_state, belief = self.rssm_step(
                prior_state, action[:, i], belief
            )
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            prior_states.append(prior_state)
            beliefs.append(belief)
        if len(prior_states) == 1:
            prior_states = prior_states[0]
            beliefs = beliefs[0]
            prior_means = prior_means[0]
            prior_stds = prior_stds[0]
        else:
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