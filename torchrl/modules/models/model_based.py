# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn

from torchrl.envs.utils import step_mdp
from torchrl.modules.distributions import NormalParamWrapper
from torchrl.modules.models import MLP
from torchrl.modules.tensordict_module.common import TensorDictModule
from torchrl.modules.tensordict_module.sequence import TensorDictSequential

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

    def forward(self, state, belief):
        loc, scale = self.backbone(state, belief)
        return loc, scale


class ObsEncoder(nn.Module):
    """Observation encoder network.

    Takes an pixel observation and encodes it into a latent space.

    Reference: https://arxiv.org/abs/1803.10122

    Args:
        depth (int, optional): Number of hidden units in the first layer.
    """

    def __init__(
        self,
        conv_depth=32,
        state_obs_hidden_dim=32,
        use_pixels=True,
        use_r3m=False,
        use_states=False,
    ):
        super().__init__()
        self.use_pixels = use_pixels
        self.use_states = use_states
        self.use_r3m = use_r3m
        if self.use_pixels and self.use_r3m:
            self.pixel_encoder = nn.Sequential(
                nn.LazyLinear(state_obs_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_obs_hidden_dim, state_obs_hidden_dim),
                nn.ReLU(),
            )

        elif self.use_pixels:
            self.pixel_encoder = nn.Sequential(
                nn.LazyConv2d(conv_depth, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(conv_depth, conv_depth * 2, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(conv_depth * 2, conv_depth * 4, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(conv_depth * 4, conv_depth * 8, 4, stride=2),
                nn.ReLU(),
            )
        if self.use_states:
            self.states_encoder = nn.Sequential(
                nn.LazyLinear(state_obs_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_obs_hidden_dim, state_obs_hidden_dim),
                nn.ReLU(),
            )
        self.embedding_size = conv_depth * 8

        self.adapter = nn.LazyLinear(self.embedding_size)

    def forward(self, *observations):
        if self.use_pixels and self.use_states:
            pixel_obs, state_obs = observations
        elif self.use_pixels:
            pixel_obs = observations[0]
        elif self.use_states:
            state_obs = observations[0]
        if self.use_pixels and self.use_r3m:
            pixel_latent = self.pixel_encoder(pixel_obs)

        elif self.use_pixels:
            *batch_sizes, C, H, W = pixel_obs.shape
            if len(batch_sizes) == 0:
                end_dim = 0
            else:
                end_dim = len(batch_sizes) - 1
            pixel_obs = torch.flatten(pixel_obs, start_dim=0, end_dim=end_dim)
            obs_encoded = self.pixel_encoder(pixel_obs)
            pixel_latent = obs_encoded.reshape(*batch_sizes, -1)
        if self.use_states:
            state_latent = self.states_encoder(state_obs)
        if self.use_pixels and self.use_states:
            latent = torch.cat([pixel_latent, state_latent], dim=-1)
        elif self.use_pixels:
            latent = pixel_latent
        elif self.use_states:
            latent = state_latent
        else:
            raise ValueError("Must use either pixels or states")
        return self.adapter(latent)


class ObsDecoder(nn.Module):
    """Observation decoder network.

    Takes the deterministic state and the stochastic belief and decodes it into a pixel observation.

    Reference: https://arxiv.org/abs/1803.10122

    Args:
        depth (int, optional): Number of hidden units in the last layer.
    """

    def __init__(
        self,
        depth=32,
        state_obs_hidden_dim=32,
        state_spec=None,
        r3m_spec=None,
        use_pixels=True,
        use_r3m=False,
        use_states=False,
    ):
        super().__init__()
        self.use_pixels = use_pixels
        self.use_states = use_states
        self.use_r3m = use_r3m

        self.state_to_latent = nn.Sequential(
            nn.LazyLinear(depth * 8 * 2 * 2),
            nn.ReLU(),
        )
        if use_pixels and use_r3m:
            self.r3m_decoder = nn.Sequential(
                nn.LazyLinear(state_obs_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_obs_hidden_dim, state_obs_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_obs_hidden_dim, r3m_spec.shape[0]),
            )

        elif use_pixels:
            self.pixel_decoder = nn.Sequential(
                nn.LazyConvTranspose2d(depth * 4, 5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(depth * 4, depth * 2, 5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(depth * 2, depth, 6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(depth, 3, 6, stride=2),
            )
        if use_states:
            if state_spec is None:
                raise ValueError("Must specify state_spec if using states")
            self.states_decoder = nn.Sequential(
                nn.LazyLinear(state_obs_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_obs_hidden_dim, state_obs_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_obs_hidden_dim, state_spec.shape[0]),
            )

    def forward(self, state, rnn_hidden):
        latent = self.state_to_latent(torch.cat([state, rnn_hidden], dim=-1))
        *batch_sizes, D = latent.shape
        if self.use_pixels and self.use_r3m:
            obs_decoded = self.r3m_decoder(latent)
        elif self.use_pixels:
            pixel_latent = latent.view(-1, D, 1, 1)
            obs_decoded = self.pixel_decoder(pixel_latent)
            _, C, H, W = obs_decoded.shape
            obs_decoded = obs_decoded.view(*batch_sizes, C, H, W)
        if self.use_states:
            state_decoded = self.states_decoder(latent)
        if self.use_pixels and self.use_states:
            return obs_decoded, state_decoded
        elif self.use_pixels:
            return obs_decoded
        elif self.use_states:
            return state_decoded
        else:
            raise ValueError("Must use either pixels or states")


class RSSMRollout(nn.Module):
    """Rollout the RSSM network.

    Given a set of encoded observations and actions, this module will rollout the RSSM network to compute all the intermediate
    states and beliefs.
    The previous posterior is used as the prior for the next time step.
    The forward method returns a stack of all intermediate states and beliefs.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        rssm_prior (RSSMPrior): Prior network.
        rssm_posterior (RSSMPosterior): Posterior network.


    """

    def __init__(self, rssm_prior: TensorDictModule, rssm_posterior: TensorDictModule):
        super().__init__()
        _module = TensorDictSequential(rssm_prior, rssm_posterior)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.rssm_prior = rssm_prior
        self.rssm_posterior = rssm_posterior

    def forward(self, tensordict):
        """Runs a rollout of simulated transitions in the latent space given a sequence of actions and environment observations.

        The rollout requires a belief and posterior state primer.

        At each step, two probability distributions are built and sampled from:
        - A prior distribution p(s_{t+1} | s_t, a_t, b_t) where b_t is a
            deterministic transform of the form b_t(s_{t-1}, a_{t-1}). The
            previous state s_t is sampled according to the posterior
            distribution (see below), creating a chain of posterior-to-priors
            that accumulates evidence to compute a prior distribution over
            the current event distribution:
            p(s_{t+1} s_t | o_t, a_t, s_{t-1}, a_{t-1}) = p(s_{t+1} | s_t, a_t, b_t) q(s_t | b_t, o_t)

        - A posterior distribution of the form q(s_{t+1} | b_{t+1}, o_{t+1})
            which amends to q(s_{t+1} | s_t, a_t, o_{t+1})

        """
        tensordict_out = []
        *batch, time_steps = tensordict.shape
        _tensordict = tensordict[..., 0]

        update_values = tensordict.exclude(*self.out_keys)
        for t in range(time_steps):
            # samples according to p(s_{t+1} | s_t, a_t, b_t)
            # ["state", "belief", "action"] -> ["next_prior_mean", "next_prior_std", "_", "next_belief"]
            self.rssm_prior(_tensordict)

            # samples according to p(s_{t+1} | s_t, a_t, o_{t+1}) = p(s_t | b_t, o_t)
            # ["next_belief", "next_encoded_latents"] -> ["next_posterior_mean", "next_posterior_std", "next_state"]
            self.rssm_posterior(_tensordict)

            tensordict_out.append(_tensordict)
            if t < time_steps - 1:
                _tensordict = step_mdp(
                    _tensordict.select(*self.out_keys), keep_other=False
                )
                _tensordict = update_values[..., t + 1].update(_tensordict)

        return torch.stack(tensordict_out, tensordict.ndimension() - 1).contiguous()


class RSSMPrior(nn.Module):
    """The prior network of the RSSM.

    This network takes as input the previous state and belief and the current action.
    It returns the next prior state and belief, as well as the parameters of the prior state distribution.
    State is by construction stochastic and belief is deterministic. In "Dream to control", these are called "deterministic state " and "stochastic state", respectively.

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
            scale_lb=0.1,
            scale_mapping="softplus",
        )

        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        if action_spec is None:
            raise ValueError("action_spec must be provided")
        self.action_shape = action_spec.shape

    def forward(self, state, belief, action):
        projector_input = torch.cat([state, action], dim=-1)
        action_state = self.action_state_projector(projector_input)
        belief = self.rnn(action_state, belief)
        prior_mean, prior_std = self.rnn_to_prior_projector(belief)
        state = prior_mean + torch.randn_like(prior_std) * prior_std
        return prior_mean, prior_std, state, belief


class RSSMPosterior(nn.Module):
    """The posterior network of the RSSM.

    This network takes as input the belief and the associated encoded observation.
    It returns the parameters of the posterior as well as a state sampled according to this distribution.

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
        posterior_mean, posterior_std = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        # post_std = post_std + 0.1
        state = posterior_mean + torch.randn_like(posterior_std) * posterior_std
        return posterior_mean, posterior_std, state
