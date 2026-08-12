# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 RSSM components: discrete categorical latent state.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

from typing import Literal

import torch
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from tensordict.utils import NestedKey, unravel_key
from torch import nn
from torch.nn import GRUCell

from torchrl.modules.value_transforms import symexp, symlog  # noqa: F401


_DEFAULT_NUM_BINS = 255
_DEFAULT_BIN_RANGE = 20.0


def _dreamer_v3_init(module: nn.Module) -> None:
    """Initialize linear modules like the reference DreamerV3 implementation."""
    if isinstance(module, nn.Linear):
        std = 1.1368 / module.in_features**0.5
        nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _DreamerV3RMSNorm(nn.Module):
    """RMS normalization with learned scale and shift."""

    def __init__(self, features: int, eps: float = 1e-4, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features, device=device))
        self.bias = nn.Parameter(torch.zeros(features, device=device))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        value = value.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float() + self.bias.float()).to(dtype)


class _DreamerV3BlockLinear(nn.Module):
    """Independent linear projections over equally sized feature blocks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int,
        *,
        device=None,
    ):
        super().__init__()
        if in_features % num_blocks or out_features % num_blocks:
            raise ValueError(
                "in_features and out_features must be divisible by num_blocks, "
                f"got {in_features}, {out_features}, and {num_blocks}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        block_in = in_features // num_blocks
        block_out = out_features // num_blocks
        self.weight = nn.Parameter(
            torch.empty(num_blocks, block_in, block_out, device=device)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        std = 1.1368 / block_in**0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.reshape(
            *value.shape[:-1], self.num_blocks, self.in_features // self.num_blocks
        )
        value = torch.einsum("...bi,bio->...bo", value, self.weight)
        return value.flatten(-2) + self.bias


class _DreamerV3BlockGRU(nn.Module):
    """Grouped gated recurrent core used by the reference DreamerV3 RSSM."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        belief_dim: int,
        num_blocks: int,
        num_layers: int,
        norm_eps: float,
        device=None,
    ):
        super().__init__()
        if belief_dim % num_blocks:
            raise ValueError(
                "rnn_hidden_dim must be divisible by num_blocks, got "
                f"{belief_dim} and {num_blocks}."
            )
        self.belief_dim = belief_dim
        self.num_blocks = num_blocks
        self.belief_projection = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim, device=device),
            _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
            nn.SiLU(),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, device=device),
            _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
            nn.SiLU(),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, hidden_dim, device=device),
            _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
            nn.SiLU(),
        )
        layer_in = belief_dim + 3 * hidden_dim * num_blocks
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    _DreamerV3BlockLinear(
                        layer_in, belief_dim, num_blocks, device=device
                    ),
                    _DreamerV3RMSNorm(belief_dim, norm_eps, device=device),
                    nn.SiLU(),
                ]
            )
            layer_in = belief_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.gates = _DreamerV3BlockLinear(
            belief_dim, 3 * belief_dim, num_blocks, device=device
        )
        self.apply(_dreamer_v3_init)

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action = action / action.detach().abs().clamp_min(1)
        features = torch.cat(
            [
                self.belief_projection(belief),
                self.state_projection(state),
                self.action_projection(action),
            ],
            -1,
        )
        grouped_belief = belief.reshape(
            *belief.shape[:-1], self.num_blocks, self.belief_dim // self.num_blocks
        )
        repeated_features = features.unsqueeze(-2).expand(
            *features.shape[:-1], self.num_blocks, features.shape[-1]
        )
        hidden = torch.cat([grouped_belief, repeated_features], -1).flatten(-2)
        hidden = self.hidden_layers(hidden)
        gates = self.gates(hidden).reshape(
            *hidden.shape[:-1],
            self.num_blocks,
            3,
            self.belief_dim // self.num_blocks,
        )
        reset, candidate, update = gates.unbind(-2)
        reset = reset.flatten(-2).sigmoid()
        candidate = (reset * candidate.flatten(-2)).tanh()
        update = (update.flatten(-2) - 1).sigmoid()
        return update * candidate + (1 - update) * belief


class DreamerV3MLP(nn.Module):
    """RMS-normalized multilayer perceptron used by DreamerV3 heads.

    Args:
        in_features (int): Input feature count.
        out_features (int): Output feature count.
        depth (int, optional): Number of hidden layers. Defaults to 3.
        num_cells (int, optional): Hidden feature count. Defaults to 1024.
        outscale (float, optional): Multiplicative initialization scale for the
            output layer. Defaults to 1.0.
        norm_eps (float, optional): RMS normalization epsilon. Defaults to
            ``1e-4``.
        device (torch.device, optional): Device on which to create parameters.

    Examples:
        >>> import torch
        >>> from torchrl.modules import DreamerV3MLP
        >>> module = DreamerV3MLP(6, 4, depth=2, num_cells=8)
        >>> module(torch.randn(3, 2), torch.randn(3, 4)).shape
        torch.Size([3, 4])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 3,
        num_cells: int = 1024,
        outscale: float = 1.0,
        norm_eps: float = 1e-4,
        device=None,
    ):
        super().__init__()
        layers = []
        layer_in = in_features
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(layer_in, num_cells, device=device),
                    _DreamerV3RMSNorm(num_cells, norm_eps, device=device),
                    nn.SiLU(),
                ]
            )
            layer_in = num_cells
        output = nn.Linear(layer_in, out_features, device=device)
        layers.append(output)
        self.model = nn.Sequential(*layers)
        self.model.apply(_dreamer_v3_init)
        with torch.no_grad():
            output.weight.mul_(outscale)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        value = inputs[0] if len(inputs) == 1 else torch.cat(inputs, -1)
        return self.model(value)


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


def _unimix_probs(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """Return categorical probabilities mixed with a uniform distribution."""
    if not 0 <= unimix < 1:
        raise ValueError(f"unimix must be in [0, 1), got {unimix}.")
    probs = torch.softmax(logits, dim=-1)
    if unimix:
        probs = (1 - unimix) * probs + unimix / logits.shape[-1]
    return probs


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

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for the prior's
    role in observation-conditioned filtering and latent imagination.

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
        recurrent_model ("gru" or "block_gru", optional): Recurrent core.
            ``"gru"`` preserves the historical TorchRL implementation while
            ``"block_gru"`` selects the grouped DreamerV3 core. Defaults to
            ``"gru"``.
        num_blocks (int, optional): Number of groups in the block GRU.
            Defaults to 8.
        num_layers (int, optional): Number of block-linear dynamics layers.
            Defaults to 1.
        prior_num_layers (int, optional): Number of prior predictor layers in
            block-GRU mode. Defaults to 2.
        norm_eps (float, optional): RMS normalization epsilon. Defaults to
            ``1e-4``.
        unimix (float, optional): Fraction of uniform probability mixed into
            categorical samples. Defaults to ``0.0`` for compatibility.
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
        recurrent_model: Literal["gru", "block_gru"] = "gru",
        num_blocks: int = 8,
        num_layers: int = 1,
        prior_num_layers: int = 2,
        norm_eps: float = 1e-4,
        unimix: float = 0.0,
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
        if not 0 <= unimix < 1:
            raise ValueError(f"unimix must be in [0, 1), got {unimix}.")
        self.unimix = unimix
        state_dim = num_categoricals * num_classes

        if recurrent_model not in ("gru", "block_gru"):
            raise ValueError(
                "recurrent_model must be 'gru' or 'block_gru', got "
                f"{recurrent_model!r}."
            )
        if recurrent_model == "block_gru" and action_dim is None:
            raise ValueError("block_gru requires an explicit action_dim.")
        self.recurrent_model = recurrent_model

        if recurrent_model == "block_gru":
            self.rnn = _DreamerV3BlockGRU(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                belief_dim=rnn_hidden_dim,
                num_blocks=num_blocks,
                num_layers=num_layers,
                norm_eps=norm_eps,
                device=device,
            )
            prior_layers = []
            prior_in = rnn_hidden_dim
            for _ in range(prior_num_layers):
                prior_layers.extend(
                    [
                        nn.Linear(prior_in, hidden_dim, device=device),
                        _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
                        nn.SiLU(),
                    ]
                )
                prior_in = hidden_dim
            prior_layers.append(
                nn.Linear(
                    prior_in,
                    num_categoricals * num_classes,
                    device=device,
                )
            )
            self.rnn_to_prior_projector = nn.Sequential(*prior_layers)
            self.rnn_to_prior_projector.apply(_dreamer_v3_init)
            self.action_state_projector = None
        else:
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
        if self.recurrent_model == "block_gru":
            belief = self.rnn(state, belief, action)
        else:
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

        state = _straight_through_categorical(prior_logits, self.unimix)
        state = state.view(*state.shape[:-2], self.num_categoricals * self.num_classes)

        return prior_logits, state, belief


class RSSMPosteriorV3(nn.Module):
    """DreamerV3 posterior (representation model) with discrete categorical latent.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for the
    relationship between the posterior, prior, stochastic state, and belief.

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
        use_rms_norm (bool, optional): Build the observation predictor from
            RMS-normalized DreamerV3 layers. Defaults to ``False`` for checkpoint
            compatibility.
        num_layers (int, optional): Number of observation predictor layers when
            ``use_rms_norm=True``. Defaults to 1.
        norm_eps (float, optional): RMS normalization epsilon. Defaults to
            ``1e-4``.
        unimix (float, optional): Fraction of uniform probability mixed into
            categorical samples. Defaults to ``0.0`` for compatibility.
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
        *,
        use_rms_norm: bool = False,
        num_layers: int = 1,
        norm_eps: float = 1e-4,
        unimix: float = 0.0,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        if not 0 <= unimix < 1:
            raise ValueError(f"unimix must be in [0, 1), got {unimix}.")
        self.unimix = unimix

        if use_rms_norm and (rnn_hidden_dim is None or obs_embed_dim is None):
            raise ValueError(
                "use_rms_norm=True requires explicit rnn_hidden_dim and "
                "obs_embed_dim."
            )

        if rnn_hidden_dim is not None and obs_embed_dim is not None:
            projector_in = rnn_hidden_dim + obs_embed_dim
            first_linear = nn.Linear(projector_in, hidden_dim, device=device)
        else:
            first_linear = nn.LazyLinear(hidden_dim, device=device)

        if use_rms_norm:
            layers = []
            layer_in = projector_in
            for layer_index in range(num_layers):
                linear = (
                    first_linear
                    if layer_index == 0
                    else nn.Linear(layer_in, hidden_dim, device=device)
                )
                layers.extend(
                    [
                        linear,
                        _DreamerV3RMSNorm(hidden_dim, norm_eps, device=device),
                        nn.SiLU(),
                    ]
                )
                layer_in = hidden_dim
            layers.append(
                nn.Linear(
                    layer_in,
                    num_categoricals * num_classes,
                    device=device,
                )
            )
            self.obs_rnn_to_post_projector = nn.Sequential(*layers)
            self.obs_rnn_to_post_projector.apply(_dreamer_v3_init)
        else:
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
        state = _straight_through_categorical(posterior_logits, self.unimix)
        state = state.view(*state.shape[:-2], self.num_categoricals * self.num_classes)
        return posterior_logits, state


class RSSMRolloutV3(TensorDictModuleBase):
    """Roll out the DreamerV3 RSSM over a sequence.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for the RSSM
    data flow and terminology used by this rollout.

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
        reset_key (NestedKey or None, optional): Boolean key marking the first
            transition of an episode. State, belief, and action are zeroed at
            those positions. Defaults to ``"is_init"``.

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
        reset_key: NestedKey | None = "is_init",
    ):
        super().__init__()
        _module = TensorDictSequential(rssm_prior, rssm_posterior)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.rssm_prior = rssm_prior
        self.rssm_posterior = rssm_posterior
        self.reset_key = unravel_key(reset_key) if reset_key is not None else None

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
            reset = (
                _tensordict.get(self.reset_key, None)
                if self.reset_key is not None
                else None
            )
            if reset is not None:
                state = _tensordict.get("state")
                belief = _tensordict.get("belief")
                action = _tensordict.get("action")
                while reset.ndim < state.ndim:
                    reset = reset.unsqueeze(-1)
                _tensordict.set("state", torch.where(reset, 0, state))
                _tensordict.set("belief", torch.where(reset, 0, belief))
                _tensordict.set("action", torch.where(reset, 0, action))
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


def _straight_through_categorical(
    logits: torch.Tensor, unimix: float = 0.0
) -> torch.Tensor:
    """Sample from categorical with straight-through gradient estimator.

    Forward: hard one-hot sample.
    Backward: gradients flow through the soft probabilities.

    Args:
        logits: ``[..., num_categoricals, num_classes]``

    Returns:
        one_hot tensor with same shape, gradients through softmax.
    """
    probs = _unimix_probs(logits, unimix)
    indices = torch.distributions.Categorical(probs=probs).sample()
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    # Straight-through: forward = one_hot, backward gradient = grad(probs).
    return probs + (one_hot - probs).detach()
