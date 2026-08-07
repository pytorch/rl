# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 RSSM components: discrete categorical latent state.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

import torch
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torch import nn
from torch.nn import GRUCell


_DEFAULT_NUM_BINS = 255
_DEFAULT_BIN_RANGE = 20.0


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Apply the element-wise symmetric logarithm transform.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        A tensor with the same shape, dtype, and device as ``x``.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import symlog
        >>> symlog(torch.tensor([-100.0, 0.0, 100.0]))
        tensor([-4.6151,  0.0000,  4.6151])
    """
    return x.sign() * (x.abs() + 1).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Apply the inverse of :func:`symlog` element-wise.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        A tensor with the same shape, dtype, and device as ``x``.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import symexp, symlog
        >>> x = torch.tensor([-1000.0, 0.0, 1000.0])
        >>> torch.allclose(symexp(symlog(x)), x, atol=1e-4)
        True
    """
    return x.sign() * x.abs().expm1()


def _default_bins(
    num_bins: int = _DEFAULT_NUM_BINS,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build the symmetric raw-value support used by DreamerV3."""
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}.")
    dtype = dtype or torch.get_default_dtype()
    half_size = num_bins // 2
    if num_bins % 2:
        half = torch.linspace(
            -_DEFAULT_BIN_RANGE,
            0,
            half_size + 1,
            device=device,
            dtype=dtype,
        )
        half = symexp(half)
        return torch.cat((half, -half[:-1].flip(0)))
    step = 2 * _DEFAULT_BIN_RANGE / (num_bins - 1)
    half = -_DEFAULT_BIN_RANGE + step * torch.arange(
        half_size, device=device, dtype=dtype
    )
    half = symexp(half)
    return torch.cat((half, -half.flip(0)))


def two_hot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode raw scalar values on a sorted two-hot support.

    Values between adjacent support points are represented by linear
    interpolation in raw value space. Values outside the support saturate at
    its endpoints.

    Args:
        x (torch.Tensor): Raw scalar targets.
        bins (torch.Tensor): One-dimensional, ascending support.

    Returns:
        A tensor with shape ``(*x.shape, bins.numel())`` on the dtype and
        device of ``x``.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import two_hot_encode
        >>> bins = torch.tensor([-1.0, 0.0, 1.0])
        >>> two_hot_encode(torch.tensor([0.25]), bins)
        tensor([[0.0000, 0.7500, 0.2500]])
    """
    if bins.ndim != 1 or bins.numel() < 2:
        raise ValueError(
            "bins must be a one-dimensional tensor with at least 2 values."
        )
    bins = bins.to(device=x.device, dtype=x.dtype)
    x = x.clamp(bins[0], bins[-1])
    lower = (x.unsqueeze(-1) >= bins).sum(-1) - 1
    lower = lower.clamp(0, bins.numel() - 2)
    upper = lower + 1
    lower_value = bins[lower]
    upper_value = bins[upper]
    upper_weight = (x - lower_value) / (upper_value - lower_value)
    lower_weight = 1 - upper_weight
    target = torch.zeros((*x.shape, bins.numel()), device=x.device, dtype=x.dtype)
    target.scatter_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    target.scatter_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return target


def two_hot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode logits over a raw-value support to their scalar expectation.

    Args:
        logits (torch.Tensor): Categorical logits whose trailing dimension
            matches the support size.
        bins (torch.Tensor): One-dimensional support in raw value space.

    Returns:
        The softmax-weighted expectation with the trailing category dimension
        removed, preserving the dtype and device of ``logits``.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import two_hot_decode, two_hot_encode
        >>> bins = torch.tensor([-1.0, 0.0, 1.0])
        >>> encoded = two_hot_encode(torch.tensor([0.25]), bins)
        >>> two_hot_decode((encoded + 1e-8).log(), bins)
        tensor([0.2500])
    """
    if bins.ndim != 1 or logits.shape[-1] != bins.numel():
        raise ValueError(
            "The trailing logits dimension must match the one-dimensional support."
        )
    bins = bins.to(device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    size = logits.shape[-1]
    if size % 2:
        midpoint = (size - 1) // 2
        center = probs[..., midpoint] * bins[midpoint]
        paired = (
            (probs[..., :midpoint] * bins[:midpoint]).flip(-1)
            + probs[..., midpoint + 1 :] * bins[midpoint + 1 :]
        ).sum(-1)
        return center + paired
    midpoint = size // 2
    return (
        (probs[..., :midpoint] * bins[:midpoint]).flip(-1)
        + probs[..., midpoint:] * bins[midpoint:]
    ).sum(-1)


def two_hot_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """Return two-hot cross entropy for raw scalar targets.

    Args:
        logits (torch.Tensor): Categorical logits with bins in the trailing
            dimension.
        target (torch.Tensor): Raw scalar targets, optionally with a trailing
            singleton dimension.
        bins (torch.Tensor): One-dimensional support in raw value space.

    Returns:
        The unreduced cross entropy with the trailing category dimension
        removed.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import two_hot_cross_entropy
        >>> logits = torch.zeros(2, 3)
        >>> target = torch.tensor([-0.5, 0.5])
        >>> two_hot_cross_entropy(logits, target, torch.tensor([-1.0, 0.0, 1.0]))
        tensor([1.0986, 1.0986])
    """
    if target.shape == (*logits.shape[:-1], 1):
        target = target.squeeze(-1)
    if target.shape != logits.shape[:-1]:
        raise ValueError(
            f"target shape must be {logits.shape[:-1]} or "
            f"{(*logits.shape[:-1], 1)}, got {target.shape}."
        )
    encoded = two_hot_encode(target, bins)
    return -(encoded * torch.log_softmax(logits, dim=-1)).sum(-1)


class SymExpTwoHot(nn.Module):
    """DreamerV3 categorical scalar representation.

    The support contains ``num_bins`` raw scalar values obtained by applying
    ``symexp`` to an evenly spaced grid from -20 to 20. Targets are interpolated
    between adjacent raw support values, while predictions are decoded as the
    softmax-weighted raw-value expectation.

    Args:
        num_bins (int, optional): Number of categorical support values.
            Defaults to 255.

    Examples:
        >>> import torch
        >>> from torchrl.modules import SymExpTwoHot
        >>> two_hot = SymExpTwoHot(num_bins=5)
        >>> target = torch.tensor([-10.0, 0.0, 10.0])
        >>> encoded = two_hot.encode(target)
        >>> decoded = two_hot.decode(encoded.log())
        >>> torch.allclose(decoded, target, atol=1e-3)
        True
    """

    def __init__(self, num_bins: int = _DEFAULT_NUM_BINS):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("bins", _default_bins(num_bins))

    def encode(self, target: torch.Tensor) -> torch.Tensor:
        """Encode raw scalar targets as two-hot categorical targets."""
        return two_hot_encode(target, self.bins)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode categorical logits to raw scalar values."""
        return two_hot_decode(logits, self.bins)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute two-hot cross entropy against raw scalar targets."""
        return two_hot_cross_entropy(logits, target, self.bins)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits and retain a trailing scalar event dimension."""
        return self.decode(logits).unsqueeze(-1)


class RSSMPriorV3(nn.Module):
    """DreamerV3 prior network with discrete categorical latent state.

    Implements the sequence model and dynamics predictor from DreamerV3.
    The GRU updates the deterministic hidden state:

    .. code-block:: text

        h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])

    Then the prior predicts a distribution over the stochastic latent:

    .. code-block:: text

        z_hat_t ~ Cat(MLP(h_t))

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        action_spec (TensorSpec, optional): Action spec. Used only to read
            ``action_spec.shape``; mutually exclusive with ``action_shape``.
        action_shape (torch.Size or tuple of int, optional): Action tensor
            shape. Mutually exclusive with ``action_spec``.
        hidden_dim (int, optional): Hidden dimension of the linear projector.
            Defaults to 512.
        rnn_hidden_dim (int, optional): GRU hidden state dimension (belief size).
            Defaults to 512.
        num_categoricals (int, optional): Number of categorical variables in the
            discrete latent. Defaults to 32.
        num_classes (int, optional): Number of classes per categorical variable.
            Defaults to 32.
        action_dim (int, optional): Action dimension. If provided (along with
            ``num_categoricals * num_classes``), uses explicit ``nn.Linear``
            instead of ``nn.LazyLinear``. Defaults to None.
        device (torch.device, optional): Device. Defaults to None.

    Examples:
        >>> import torch
        >>> from torchrl.modules.models.model_based_v3 import RSSMPriorV3
        >>> prior = RSSMPriorV3(
        ...     action_shape=torch.Size([2]),
        ...     hidden_dim=16,
        ...     rnn_hidden_dim=8,
        ...     num_categoricals=4,
        ...     num_classes=4,
        ...     action_dim=2,
        ... )
        >>> state = torch.zeros(3, 16)
        >>> belief = torch.zeros(3, 8)
        >>> action = torch.randn(3, 2)
        >>> logits, next_state, next_belief = prior(state, belief, action)
        >>> logits.shape, next_state.shape, next_belief.shape
        (torch.Size([3, 4, 4]), torch.Size([3, 16]), torch.Size([3, 8]))
    """

    def __init__(
        self,
        action_spec=None,
        hidden_dim: int = 512,
        rnn_hidden_dim: int = 512,
        num_categoricals: int = 32,
        num_classes: int = 32,
        action_dim: int | None = None,
        device=None,
        *,
        action_shape: torch.Size | tuple[int, ...] | None = None,
    ):
        super().__init__()
        if action_spec is not None and action_shape is not None:
            raise ValueError(
                "Pass only one of `action_spec` or `action_shape`, not both."
            )
        if action_spec is not None:
            self.action_shape = torch.Size(action_spec.shape)
        elif action_shape is not None:
            self.action_shape = torch.Size(action_shape)
        else:
            self.action_shape = None

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

        # Run GRU in fp32 to avoid cuBLAS dispatch issues under autocast
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

    Given the deterministic hidden state ``h_t`` and an observation embedding
    ``e_t``, produces the posterior distribution over the stochastic latent:

    .. code-block:: text

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

    Examples:
        >>> import torch
        >>> from torchrl.modules.models.model_based_v3 import RSSMPosteriorV3
        >>> posterior = RSSMPosteriorV3(
        ...     hidden_dim=16,
        ...     num_categoricals=4,
        ...     num_classes=4,
        ...     rnn_hidden_dim=8,
        ...     obs_embed_dim=12,
        ... )
        >>> belief = torch.randn(3, 8)
        >>> obs_embed = torch.randn(3, 12)
        >>> logits, state = posterior(belief, obs_embed)
        >>> logits.shape, state.shape
        (torch.Size([3, 4, 4]), torch.Size([3, 16]))
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

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.model_based_v3 import (
        ...     RSSMPosteriorV3, RSSMPriorV3, RSSMRolloutV3,
        ... )
        >>> prior = TensorDictModule(
        ...     RSSMPriorV3(action_shape=torch.Size([2]), hidden_dim=8,
        ...                 rnn_hidden_dim=8, num_categoricals=4, num_classes=4,
        ...                 action_dim=2),
        ...     in_keys=["state", "belief", "action"],
        ...     out_keys=[("next", "prior_logits"), ("next", "state"), ("next", "belief")],
        ... )
        >>> posterior = TensorDictModule(
        ...     RSSMPosteriorV3(hidden_dim=8, num_categoricals=4, num_classes=4,
        ...                     rnn_hidden_dim=8, obs_embed_dim=6),
        ...     in_keys=[("next", "belief"), ("next", "encoded_latents")],
        ...     out_keys=[("next", "posterior_logits"), ("next", "state")],
        ... )
        >>> rollout = RSSMRolloutV3(prior, posterior)
        >>> td = TensorDict({
        ...     "state": torch.zeros(2, 4, 16),
        ...     "belief": torch.zeros(2, 4, 8),
        ...     "action": torch.randn(2, 4, 2),
        ...     "next": {"encoded_latents": torch.randn(2, 4, 6)},
        ... }, [2, 4])
        >>> out = rollout(td)
        >>> out.shape
        torch.Size([2, 4])
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

        # Cache the keys we want to keep; they're constant across timesteps.
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
    indices = torch.distributions.Categorical(logits=logits).sample()
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    # Straight-through: forward = one_hot, backward gradient = grad(probs).
    return probs + (one_hot - probs).detach()
