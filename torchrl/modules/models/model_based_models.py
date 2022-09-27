# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn

from torchrl.modules.distributions import NormalParamWrapper
from torchrl.modules.models import MLP

__all__ = [
    "DreamerActor",
    "ObsEncoder",
    "ObsDecoder",
    "RSSMPrior",
    "RSSMPosterior",
]


class DreamerActor(nn.Module):
    """Dreamer actor network.

    This network is used to predict the action distribution given the
    the stochastic state and the deterministic belief at the current
    time step.
    It output the mean and the scale of the action distribution.

    Reference: https://arxiv.org/abs/1912.016034

    Args:
        out_features (int): Number of output features.
        depth (int, optional): Number of hidden layers.
        num_cells (int, optional): Number of hidden units per layer.
        activation_class (nn.Module, optional): Activation class.
        rnn_hidden_size (int, optional): Size of the hidden state of the RNN in the RSSM module.
    """

    def __init__(
        self,
        out_features,
        depth=4,
        num_cells=200,
        activation_class=nn.ELU,
        rnn_hidden_dim=200,
    ):
        super().__init__()
        self.backbone = NormalParamWrapper(
            MLP(
                out_features=2 * out_features,
                depth=depth,
                num_cells=num_cells,
                activation_class=activation_class,
            ),
            scale_mapping="biased_softplus_5.0_1e-4",
        )
        self.rnn_hidden_dim = rnn_hidden_dim

    def forward(self, state, belief):
        if belief is None:
            *batch_size, _ = state.shape
            belief = torch.zeros(*batch_size, self.rnn_hidden_dim, device=state.device)
        loc, scale = self.backbone(state, belief)
        return loc, scale


class ObsEncoder(nn.Module):
    """Observation encoder network.

    Takes an pixel observation and encodes it into a latent space.

    Reference: https://arxiv.org/abs/1803.10122

    Args:
        depth (int, optional): Number of hidden units in the first layer.
    """

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
        if len(batch_sizes) == 0:
            end_dim = 0
        else:
            end_dim = len(batch_sizes) - 1
        observation = torch.flatten(observation, start_dim=0, end_dim=end_dim)
        obs_encoded = self.encoder(observation)
        latent = obs_encoded.reshape(*batch_sizes, -1)
        return latent


class ObsDecoder(nn.Module):
    """Observation decoder network.

    Takes the deterministic state and the stochastic belief and decodes it into a pixel observation.

    Reference: https://arxiv.org/abs/1803.10122

    Args:
        depth (int, optional): Number of hidden units in the last layer.
    """

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


class RSSMRollout(nn.Module):
    """Rollout the RSSM network.

    Given a set of encoded observations and actions, this module will rollout the RSSM network to compute all the intermediate
    states and believes.
    The previous posterior is used as the prior for the next time step. At the first time step, we use the an empty prior to start the rollout.
    The forward method returns a stack of all intermediate states and believes.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        rssm_prior (RSSMPrior): Prior network.
        rssm_posterior (RSSMPosterior): Posterior network.


    """

    def __init__(self, rssm_prior, rssm_posterior):
        super().__init__()
        self.rssm_prior = rssm_prior
        self.rssm_posterior = rssm_posterior
        self.rnn_hidden_dim = rssm_prior.rnn_hidden_dim

    def forward(self, posterior_state, belief, actions, obs_embedding):
        """Runs a rollout of simulated transitions in the latent space given
        an initial posterior state,  an initial belief, a defined sequence of actions, and a sequence of encoded observations.

        Args:
            posterior_state (torch.Tensor): a batch x state_size tensor containing the initial posterior_state state
            belief (torch.Tensor): a batch x belief_size tensor containing the initial belief state
            actions (torch.Tensor): a batch x time_steps x action_size tensor containing the sequence of actions
            obs_embedding (torch.Tensor): a batch x time_steps x latent_size tensor containing the sequence of encoded observations

        Returns:
            prior_means (torch.Tensor): a batch x time_steps x state_size containing the mean of the state distributions
            prior_stds (torch.Tensor): a batch x time_steps x state_size containing the standard deviation of the state distributions
            prior_states (torch.Tensor): a batch x time_steps x state_size containing the sampled states
            beliefs (torch.Tensor): a batch x time_steps x belief_size containing the sequence of beliefs
            posterior_means (torch.Tensor): a batch x time_steps x state_size containing the mean of the posterior state distributions
            posterior_stds (torch.Tensor): a batch x time_steps x state_size containing the standard deviation of the posterior state distributions
            posterior_states (torch.Tensor): a batch x time_steps x state_size containing the sampled posterior states
        """
        prior_means = []
        prior_stds = []
        prior_states = []
        beliefs = []
        posterior_means = []
        posterior_stds = []
        posterior_states = []

        for i in range(actions.shape[1]):
            prior_mean, prior_std, prior_state, belief = self.rssm_prior(
                posterior_state, belief, actions[:, i]
            )
            posterior_mean, posterior_std, posterior_state = self.rssm_posterior(
                belief, obs_embedding[:, i]
            )
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            prior_states.append(prior_state)
            beliefs.append(belief)
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
            posterior_states.append(posterior_state)
        prior_means = torch.stack(prior_means, dim=1)
        prior_stds = torch.stack(prior_stds, dim=1)
        prior_states = torch.stack(prior_states, dim=1)
        beliefs = torch.stack(beliefs, dim=1)
        posterior_means = torch.stack(posterior_means, dim=1)
        posterior_stds = torch.stack(posterior_stds, dim=1)
        posterior_states = torch.stack(posterior_states, dim=1)
        return (
            prior_means,
            prior_stds,
            prior_states,
            beliefs,
            posterior_means,
            posterior_stds,
            posterior_states,
        )


class RSSMPrior(nn.Module):
    """The prior network of the RSSM.

    This network takes as input the previous state and belief and the current action.
    It outputs the next prior state and belief, as well as the parameters of the prior state distribution.
    State is by construction stochastic and belief is deterministic. In the "Dream to control" paper, these are called "deterministic state " and "stochastic state", respectively.
    We distinguish states sampled according prior and posterior distribution for clarity.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        hidden_dim (int, optional): Number of hidden units in the linear network. Input size of the recurrent network.
        rnn_hidden_dim (int, optional): Number of hidden units in the recurrent network. Also size of the belief.
        state_dim (int, optional): Size of the state.
        action_spec (TensorSpec, optional): Action spec. If None an error will be raised when initializing.

    """

    def __init__(
        self, hidden_dim=200, rnn_hidden_dim=200, state_dim=30, action_spec=None
    ):
        super().__init__()

        # Prior
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.action_state_projector = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ELU())
        self.rnn_to_prior_projector = NormalParamWrapper(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 2 * state_dim),
            ),
            scale_lb=0,
            scale_mapping="softplus",
        )

        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        if action_spec is None:
            raise ValueError("action_spec must be provided")
        self.action_shape = action_spec.shape

    def forward(self, state, rnn_hidden, action):
        action_state = self.action_state_projector(torch.cat([state, action], dim=-1))
        rnn_hidden = self.rnn(action_state, rnn_hidden)
        belief = rnn_hidden
        prior_mean, prior_std = self.rnn_to_prior_projector(belief)
        prior_std = prior_std + 0.1
        prior_state = prior_mean + torch.randn_like(prior_std) * prior_std
        return prior_mean, prior_std, prior_state, belief


class RSSMPosterior(nn.Module):
    """The posterior network of the RSSM.

    This network takes as input the belief and the associated encoded observation.
    It outputs the next posterior state and belief, as well as the parameters of the posterior state distribution.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        hidden_dim (int, optional): Number of hidden units in the linear network.
        state_dim (int, optional): Size of the state.

    """

    def __init__(self, hidden_dim=200, state_dim=30):
        super().__init__()
        self.obs_rnn_to_post_projector = NormalParamWrapper(
            nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 2 * state_dim),
            ),
            scale_lb=0.1,
            scale_mapping="softplus",
        )
        self.hidden_dim = hidden_dim

    def forward(self, belief, obs_embedding):
        post_mean, post_std = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        # post_std = post_std + 0.1
        post_state = post_mean + torch.randn_like(post_std) * post_std
        return post_mean, post_std, post_state
