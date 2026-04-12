# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torch import nn
from torch.nn import GRUCell


class RSSMPriorV3(nn.Module):
    """DreamerV3 prior network with discrete categorical latent state.

    Implements the sequence model and dynamics predictor from DreamerV3.
    The GRU updates the deterministic hidden state:
        h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    Then the prior predicts a distribution over the stochastic latent:
        ẑ_t ~ Cat(MLP(h_t))

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        action_spec: Action spec (used only for shape validation).
        hidden_dim (int, optional): Hidden dimension of the linear projector.
            Defaults to 512.
        rnn_hidden_dim (int, optional): GRU hidden state dimension (belief size).
            Defaults to 512.
        num_categoricals (int, optional): Number of categorical variables in the
            discrete latent. Defaults to 32.
        num_classes (int, optional): Number of classes per categorical variable.
            Defaults to 32.
        action_dim (int, optional): Action dimension. If provided (along with
            ``num_categoricals * num_classes``), uses explicit ``nn.Linear`` instead
            of ``nn.LazyLinear``. Defaults to None.
        device (torch.device, optional): Device. Defaults to None.
    """

    def __init__(
        self,
        action_spec,
        hidden_dim: int = 512,
        rnn_hidden_dim: int = 512,
        num_categoricals: int = 32,
        num_classes: int = 32,
        action_dim: int | None = None,
        device=None,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.rnn_hidden_dim = rnn_hidden_dim
        state_dim = num_categoricals * num_classes

        self.rnn = GRUCell(hidden_dim, rnn_hidden_dim, device=device)

        if action_dim is not None:
            projector_in = state_dim + action_dim
            first_linear = nn.Linear(projector_in, hidden_dim, device=device)
        else:
            first_linear = nn.LazyLinear(hidden_dim, device=device)
        self.action_state_projector = nn.Sequential(first_linear, nn.SiLU())

        self.rnn_to_prior_projector = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_categoricals * num_classes, device=device),
        )

        self.action_shape = action_spec.shape

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prior distribution and update GRU belief.

        Args:
            state: Previous stochastic state, shape ``[..., num_categoricals * num_classes]``.
            belief: Previous GRU hidden state, shape ``[..., rnn_hidden_dim]``.
            action: Current action, shape ``[..., action_dim]``.

        Returns:
            prior_logits (torch.Tensor): Raw logits, shape
                ``[..., num_categoricals, num_classes]``.
            state (torch.Tensor): Sampled state (straight-through), shape
                ``[..., num_categoricals * num_classes]``.
            belief (torch.Tensor): Updated GRU hidden state, shape
                ``[..., rnn_hidden_dim]``.
        """
        projector_input = torch.cat([state, action], dim=-1)
        action_state = self.action_state_projector(projector_input)

        # Run GRU in full precision to avoid cuBLAS issues under autocast
        dtype = action_state.dtype
        device_type = action_state.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            belief = self.rnn(
                action_state.float(),
                belief.float() if belief is not None else None,
            )
        belief = belief.to(dtype)

        prior_logits_flat = self.rnn_to_prior_projector(belief)
        prior_logits = prior_logits_flat.view(
            *prior_logits_flat.shape[:-1], self.num_categoricals, self.num_classes
        )

        state = _straight_through_categorical(prior_logits)
        state = state.view(*state.shape[:-2], self.num_categoricals * self.num_classes)

        return prior_logits, state, belief


class RSSMPosteriorV3(nn.Module):
    """DreamerV3 posterior (representation model) with discrete categorical latent.

    Given the deterministic hidden state ``h_t`` and an observation embedding ``e_t``,
    produces the posterior distribution over the stochastic latent:
        z_t ~ Cat(MLP([h_t, e_t]))

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        hidden_dim (int, optional): Hidden dimension of the projector MLP.
            Defaults to 512.
        num_categoricals (int, optional): Number of categorical variables.
            Defaults to 32.
        num_classes (int, optional): Number of classes per categorical variable.
            Defaults to 32.
        rnn_hidden_dim (int, optional): Belief dimension. If provided along with
            ``obs_embed_dim``, uses explicit ``nn.Linear``. Defaults to None.
        obs_embed_dim (int, optional): Observation embedding dimension. If provided
            along with ``rnn_hidden_dim``, uses explicit ``nn.Linear``. Defaults to None.
        device (torch.device, optional): Device. Defaults to None.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_categoricals: int = 32,
        num_classes: int = 32,
        rnn_hidden_dim: int | None = None,
        obs_embed_dim: int | None = None,
        device=None,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes

        if rnn_hidden_dim is not None and obs_embed_dim is not None:
            projector_in = rnn_hidden_dim + obs_embed_dim
            first_linear = nn.Linear(projector_in, hidden_dim, device=device)
        else:
            first_linear = nn.LazyLinear(hidden_dim, device=device)

        self.obs_rnn_to_post_projector = nn.Sequential(
            first_linear,
            nn.SiLU(),
            nn.Linear(hidden_dim, num_categoricals * num_classes, device=device),
        )

    def forward(
        self,
        belief: torch.Tensor,
        obs_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution given belief and observation embedding.

        Args:
            belief: Deterministic GRU hidden state from prior, shape
                ``[..., rnn_hidden_dim]``.
            obs_embedding: Encoded observation, shape ``[..., obs_embed_dim]``.

        Returns:
            posterior_logits (torch.Tensor): Raw logits, shape
                ``[..., num_categoricals, num_classes]``.
            state (torch.Tensor): Sampled state (straight-through), shape
                ``[..., num_categoricals * num_classes]``.
        """
        post_logits_flat = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        posterior_logits = post_logits_flat.view(
            *post_logits_flat.shape[:-1], self.num_categoricals, self.num_classes
        )
        state = _straight_through_categorical(posterior_logits)
        state = state.view(*state.shape[:-2], self.num_categoricals * self.num_classes)
        return posterior_logits, state


class RSSMRolloutV3(TensorDictModuleBase):
    """Roll out the DreamerV3 RSSM over a sequence.

    Given encoded observations and actions for ``T`` time steps, this module
    runs the prior (GRU + categorical) then the posterior (categorical) at each
    step and returns a stacked TensorDict of all intermediate states.

    The previous posterior state ``z_t`` is used as the prior input for step
    ``t+1``, matching the recurrent structure of DreamerV3.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        rssm_prior (TensorDictModule): Prior module wrapping :class:`RSSMPriorV3`.
        rssm_posterior (TensorDictModule): Posterior module wrapping
            :class:`RSSMPosteriorV3`.
    """

    def __init__(
        self,
        rssm_prior: TensorDictModule,
        rssm_posterior: TensorDictModule,
    ):
        super().__init__()
        _module = TensorDictSequential(rssm_prior, rssm_posterior)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.rssm_prior = rssm_prior
        self.rssm_posterior = rssm_posterior

    def forward(self, tensordict):
        """Roll out the RSSM for one episode chunk.

        Args:
            tensordict (TensorDictBase): Input with shape ``[*batch, T]`` containing
                actions, encoded observations, and initial state/belief.

        Returns:
            TensorDictBase: Stacked outputs with shape ``[*batch, T]``.
        """
        tensordict_out = []
        *batch, time_steps = tensordict.shape

        update_values = tensordict.exclude(*self.out_keys).unbind(-1)
        _tensordict = update_values[0]

        output_keys = list(
            update_values[0].keys(include_nested=True, leaves_only=True)
        ) + list(self.out_keys)

        for t in range(time_steps):
            self.rssm_prior(_tensordict)
            self.rssm_posterior(_tensordict)

            tensordict_out.append(_tensordict.select(*output_keys, strict=False))
            if t < time_steps - 1:
                next_state = _tensordict.get(("next", "state"))
                next_belief = _tensordict.get(("next", "belief"))
                _tensordict = update_values[t + 1]
                _tensordict.set("state", next_state)
                _tensordict.set("belief", next_belief)

        return torch.stack(tensordict_out, tensordict.ndim - 1)


def _straight_through_categorical(logits: torch.Tensor) -> torch.Tensor:
    """Sample from categorical with straight-through gradient estimator.

    Forward: hard one-hot sample.
    Backward: gradients flow through the soft probabilities.

    Args:
        logits: ``[..., num_categoricals, num_classes]``

    Returns:
        one_hot tensor with same shape, gradients through softmax.
    """
    probs = torch.softmax(logits, dim=-1)
    indices = torch.distributions.Categorical(probs=probs).sample()
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    # straight-through: forward uses hard, backward uses soft
    return one_hot + (probs - probs.detach())
