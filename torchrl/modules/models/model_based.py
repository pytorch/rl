# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from packaging import version
from torch import nn

from torchrl.envs.utils import step_mdp
from torchrl.modules.distributions import NormalParamWrapper
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.common import TensorDictModule
from torchrl.modules.tensordict_module.sequence import TensorDictSequential


class DreamerActor(nn.Module):
    """Dreamer actor network.

    This network is used to predict the action distribution given the
    the stochastic state and the deterministic belief at the current
    time step.
    It outputs the mean and the scale of the action distribution.

    Reference: https://arxiv.org/abs/1912.016034

    Args:
        out_features (int): Number of output features.
        depth (int, optional): Number of hidden layers.
            Defaults to 4.
        num_cells (int, optional): Number of hidden units per layer.
            Defaults to 200.
        activation_class (nn.Module, optional): Activation class.
            Defaults to nn.ELU.
        std_bias (float, optional): Bias of the softplus transform.
            Defaults to 5.0.
        std_min_val (float, optional): Minimum value of the standard deviation.
            Defaults to 1e-4.
    """

    def __init__(
        self,
        out_features,
        depth=4,
        num_cells=200,
        activation_class=nn.ELU,
        std_bias=5.0,
        std_min_val=1e-4,
    ):
        super().__init__()
        self.backbone = NormalParamWrapper(
            MLP(
                out_features=2 * out_features,
                depth=depth,
                num_cells=num_cells,
                activation_class=activation_class,
            ),
            scale_mapping=f"biased_softplus_{std_bias}_{std_min_val}",
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
            Defaults to 32.
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
            Defaults to 32.
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
    states and beliefs.
    The previous posterior is used as the prior for the next time step.
    The forward method returns a stack of all intermediate states and beliefs.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        rssm_prior (TensorDictModule): Prior network.
        rssm_posterior (TensorDictModule): Posterior network.


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
        action_spec (TensorSpec): Action spec.
        hidden_dim (int, optional): Number of hidden units in the linear network. Input size of the recurrent network.
            Defaults to 200.
        rnn_hidden_dim (int, optional): Number of hidden units in the recurrent network. Also size of the belief.
            Defaults to 200.
        state_dim (int, optional): Size of the state.
            Defaults to 30.
        scale_lb (float, optional): Lower bound of the scale of the state distribution.
            Defaults to 0.1.


    """

    def __init__(
        self,
        action_spec,
        hidden_dim=200,
        rnn_hidden_dim=200,
        state_dim=30,
        scale_lb=0.1,
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
            scale_lb=scale_lb,
            scale_mapping="softplus",
        )

        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.action_shape = action_spec.shape
        self._unsqueeze_rnn_input = version.parse(torch.__version__) < version.parse(
            "1.11"
        )

    def forward(self, state, belief, action):
        projector_input = torch.cat([state, action], dim=-1)
        action_state = self.action_state_projector(projector_input)
        unsqueeze = False
        if self._unsqueeze_rnn_input and action_state.ndimension() == 1:
            if belief is not None:
                belief = belief.unsqueeze(0)
            action_state = action_state.unsqueeze(0)
            unsqueeze = True
        belief = self.rnn(action_state, belief)
        if unsqueeze:
            belief = belief.squeeze(0)

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
            Defaults to 200.
        state_dim (int, optional): Size of the state.
            Defaults to 30.
        scale_lb (float, optional): Lower bound of the scale of the state distribution.
            Defaults to 0.1.

    """

    def __init__(self, hidden_dim=200, state_dim=30, scale_lb=0.1):
        super().__init__()
        self.obs_rnn_to_post_projector = NormalParamWrapper(
            nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 2 * state_dim),
            ),
            scale_lb=scale_lb,
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
