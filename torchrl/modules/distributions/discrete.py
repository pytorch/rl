# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Sequence

import torch
import torch.distributions as D
import torch.nn.functional as F
from tensordict.utils import expand_as_right

from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits

__all__ = [
    "OneHotCategorical",
    "MaskedCategorical",
    "Ordinal",
    "OneHotOrdinal",
    "LLMMaskedCategorical",
]


def _treat_categorical_params(
    params: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if params is None:
        return None
    if params.shape[-1] == 1:
        params = params[..., 0]
    return params


def rand_one_hot(values: torch.Tensor, do_softmax: bool = True) -> torch.Tensor:
    if do_softmax:
        values = values.softmax(-1)
    out = values.cumsum(-1) > torch.rand_like(values[..., :1])
    out = (out.cumsum(-1) == 1).to(torch.long)
    return out


class _one_hot_wrapper:
    def __init__(self, parent_dist):
        self.parent_dist = parent_dist

    def __call__(self, func):
        @wraps(func)
        def wrapped(_self, *args, **kwargs):
            out = getattr(self.parent_dist, func.__name__)(_self, *args, **kwargs)
            n = _self.num_samples
            return torch.nn.functional.one_hot(out, n)

        return wrapped


class ReparamGradientStrategy(Enum):
    PassThrough = 1
    RelaxedOneHot = 2


class OneHotCategorical(D.Categorical):
    """One-hot categorical distribution.

    This class behaves exactly as torch.distributions.Categorical except that it reads and produces one-hot encodings
    of the discrete tensors.

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities
        grad_method (ReparamGradientStrategy, optional): strategy to gather
            reparameterized samples.
            ``ReparamGradientStrategy.PassThrough`` will compute the sample gradients
             by using the softmax valued log-probability as a proxy to the
             sample gradients.
            ``ReparamGradientStrategy.RelaxedOneHot`` will use
            :class:`torch.distributions.RelaxedOneHot` to sample from the distribution.

    Examples:
        >>> torch.manual_seed(0)
        >>> logits = torch.randn(4)
        >>> dist = OneHotCategorical(logits=logits)
        >>> print(dist.rsample((3,)))
        tensor([[1., 0., 0., 0.],
                [0., 0., 0., 1.],
                [1., 0., 0., 0.]])

    """

    num_params: int = 1

    # This is to make the compiler happy, see https://github.com/pytorch/pytorch/issues/140266
    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        grad_method: ReparamGradientStrategy = ReparamGradientStrategy.PassThrough,
        **kwargs,
    ) -> None:
        logits = _treat_categorical_params(logits)
        probs = _treat_categorical_params(probs)
        self.grad_method = grad_method
        super().__init__(probs=probs, logits=logits, **kwargs)
        # Get num_samples from logits or probs shape
        if logits is not None:
            self.num_samples = logits.shape[-1]
        else:
            self.num_samples = probs.shape[-1]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.argmax(dim=-1))

    @property
    def mode(self) -> torch.Tensor:
        if hasattr(self, "logits"):
            return (self.logits == self.logits.max(-1, True)[0]).to(torch.long)
        else:
            return (self.probs == self.probs.max(-1, True)[0]).to(torch.long)

    @property
    def deterministic_sample(self):
        return self.mode

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    @_one_hot_wrapper(D.Categorical)
    def sample(self, sample_shape: torch.Size | Sequence | None = None) -> torch.Tensor:
        ...

    def rsample(self, sample_shape: torch.Size | Sequence = None) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        if hasattr(self, "logits") and self.logits is not None:
            logits = self.logits
            probs = None
        else:
            logits = None
            probs = self.probs
        if self.grad_method == ReparamGradientStrategy.RelaxedOneHot:
            d = D.relaxed_categorical.RelaxedOneHotCategorical(
                1.0, probs=probs, logits=logits
            )
            out = d.rsample(sample_shape)
            out.data.copy_((out == out.max(-1)[0].unsqueeze(-1)).to(out.dtype))
            return out
        elif self.grad_method == ReparamGradientStrategy.PassThrough:
            if logits is not None:
                probs = self.probs
            else:
                probs = torch.softmax(self.logits, dim=-1)
            out = self.sample(sample_shape)
            out = out + probs - probs.detach()
            return out
        else:
            raise ValueError(
                f"Unknown reparameterization strategy {self.reparam_strategy}."
            )


class MaskedCategorical(D.Categorical):
    """MaskedCategorical distribution.

    Reference:
    https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities. If provided, the probabilities
            corresponding to masked items will be zeroed and the probability
            re-normalized along its last dimension.

    Keyword Args:
        mask (torch.Tensor): A boolean mask of the same shape as ``logits``/``probs``
            where ``False`` entries are the ones to be masked. Alternatively,
            if ``sparse_mask`` is True, it represents the list of valid indices
            in the distribution. Exclusive with ``indices``.
        indices (torch.Tensor): A dense index tensor representing which actions
            must be taken into account. Exclusive with ``mask``.
        neg_inf (:obj:`float`, optional): The log-probability value allocated to
            invalid (out-of-mask) indices. Defaults to -inf.
        padding_value: The padding value in the mask tensor. When
            sparse_mask == True, the padding_value will be ignored.
        use_cross_entropy (bool, optional): For faster computation of the log-probability,
            the cross_entropy loss functional can be used. Defaults to ``True``.
        padding_side (str, optional): The side of the padding. Defaults to ``"left"``.

    Examples:
        >>> torch.manual_seed(0)
        >>> logits = torch.randn(4) / 100  # almost equal probabilities
        >>> mask = torch.tensor([True, False, True, True])
        >>> dist = MaskedCategorical(logits=logits, mask=mask)
        >>> sample = dist.sample((10,))
        >>> print(sample)  # no `1` in the sample
        tensor([2, 3, 0, 2, 2, 0, 2, 0, 2, 2])
        >>> print(dist.log_prob(sample))
        tensor([-1.1203, -1.0928, -1.0831, -1.1203, -1.1203, -1.0831, -1.1203, -1.0831,
                -1.1203, -1.1203])
        >>> print(dist.log_prob(torch.ones_like(sample)))
        tensor([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
        >>> # with probabilities
        >>> prob = torch.ones(10)
        >>> prob = prob / prob.sum()
        >>> mask = torch.tensor([False] + 9 * [True])  # first outcome is masked
        >>> dist = MaskedCategorical(probs=prob, mask=mask)
        >>> print(dist.log_prob(torch.arange(10)))
        tensor([   -inf, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972,
                -2.1972, -2.1972])
    """

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        neg_inf: float = float("-inf"),
        padding_value: int | None = None,
        use_cross_entropy: bool = True,
        padding_side: str = "left",
    ) -> None:
        if not ((mask is None) ^ (indices is None)):
            raise ValueError(
                f"A ``mask`` or some ``indices`` must be provided for {type(self)}, but not both."
            )
        if mask is None:
            mask = indices
            sparse_mask = True
        else:
            sparse_mask = False

        if probs is not None:
            if logits is not None:
                raise ValueError(
                    "Either `probs` or `logits` must be specified, but not both."
                )
            # unnormalized logits
            probs = probs.clone()
            if mask.dtype == torch.bool:
                probs[~mask] = 0
            else:
                probs = torch.scatter(
                    torch.zeros_like(probs), -1, indices, probs.gather(-1, indices)
                )
            probs = probs / probs.sum(-1, keepdim=True)
            logits = probs.log()
        num_samples = logits.shape[-1]
        self.use_cross_entropy = use_cross_entropy
        logits = self._mask_logits(
            logits,
            mask,
            neg_inf=neg_inf,
            sparse_mask=sparse_mask,
            padding_value=padding_value,
        )
        self.neg_inf = neg_inf
        self._mask = mask
        self._sparse_mask = sparse_mask
        self._padding_value = padding_value
        self._padding_side = padding_side
        super().__init__(logits=logits)
        self.num_samples = num_samples

    @property
    def padding_value(self):
        """Padding value of the distribution mask.

        If the padding value is not set, it will be inferred from the logits.
        """
        return self._padding_value if self._padding_value is not None else 0

    @property
    def padding_side(self):
        return self._padding_side

    @property
    def mask(self):
        if self._sparse_mask:
            raise ValueError("MaskedCategorical.mask does not support sparse masks")
        return self._mask

    def entropy(self):
        """Compute the entropy of the distribution.

        For masked distributions, we only consider the entropy over the valid (unmasked) outcomes.
        Invalid outcomes have zero probability and don't contribute to entropy.
        """
        min_real = torch.finfo(self.logits.dtype).min

        # Clamp logits to avoid numerical issues
        logits = self.logits
        if self._mask.dtype is torch.bool:
            mask = expand_as_right(self._mask, logits)
            mask = (~mask) | (~logits.isfinite())
            logits = torch.masked_fill(logits, mask, min_real)
        else:
            # logits are already masked
            pass
        logits = logits - logits.logsumexp(-1, keepdim=True)

        # Get probabilities and mask them
        probs = logits.exp()

        # Compute entropy only for valid outcomes
        p_log_p = logits * probs
        return -p_log_p.sum(-1)

    def sample(
        self, sample_shape: torch.Size | Sequence[int] | None = None
    ) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        else:
            sample_shape = torch.Size(sample_shape)

        ret = super().sample(sample_shape)
        if not self._sparse_mask:
            return ret

        size = ret.size()
        outer_dim = sample_shape.numel()
        inner_dim = self._mask.shape[:-1].numel()
        idx_3d = self._mask.expand(outer_dim, inner_dim, -1)
        ret = idx_3d.gather(dim=-1, index=ret.view(outer_dim, inner_dim, 1))
        return ret.reshape(size)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self._sparse_mask:
            if self.use_cross_entropy:
                logits = self.logits
                if logits.ndim > 2:
                    # Bring channels in 2nd dim
                    logits = logits.transpose(-1, 1)
                original_value_shape = None
                if logits.ndim == 1 and value.ndim >= 1:
                    if value.ndim >= 2:
                        original_value_shape = value.shape
                        value = value.flatten()
                    logits = logits.unsqueeze(0).expand(value.shape + logits.shape)
                result = -torch.nn.functional.cross_entropy(logits, value, reduce=False)
                if original_value_shape is not None:
                    result = result.unflatten(0, original_value_shape)
            else:
                result = super().log_prob(value)
            result = torch.where(torch.isfinite(result), result, self.neg_inf)
            return result

        idx_3d = self._mask.view(1, -1, self._num_events)
        val_3d = value.view(-1, idx_3d.size(1), 1)
        mask = idx_3d == val_3d
        idx = mask.int().argmax(dim=-1, keepdim=True)
        idx = idx.view_as(value)
        if self.use_cross_entropy:
            logits = self.logits
            if logits.ndim > 2:
                # Bring channels in 2nd dim
                logits = logits.transpose(-1, 1)
            # possible shapes:
            # Don't work with cross_entropy (missing batch dimension)
            # logits.shape = (C,) and idx.shape = (B,)
            # logits.shape = (C,) and idx.shape = (B0, B1, ...) => requires flattening of idx, only one batch dimension
            # work with cross_entropy:
            # logits.shape = (B, C) and idx.shape = (B,)
            # logits.shape = (B, C, d1, d2, ...) and idx.shape = (B, d1, d2, ...)
            original_idx_shape = None
            if logits.ndim == 1 and idx.ndim >= 1:
                if idx.ndim >= 2:
                    original_idx_shape = idx.shape
                    idx = idx.flatten()
                logits = logits.unsqueeze(0).expand(idx.shape + logits.shape)
            ret = -torch.nn.functional.cross_entropy(logits, idx, reduce=False)
            if original_idx_shape is not None:
                ret = ret.unflatten(0, original_idx_shape)
        else:
            ret = super().log_prob(idx)
        # Fill masked values with neg_inf.
        ret = ret.view_as(val_3d)
        ret = ret.masked_fill(
            torch.logical_not(mask.any(dim=-1, keepdim=True)), self.neg_inf
        )
        return ret.view_as(value)

    @staticmethod
    def _mask_logits(
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        neg_inf: float = float("-inf"),
        sparse_mask: bool = False,
        padding_value: int | None = None,
    ) -> torch.Tensor:
        if mask is None:
            return logits

        if not sparse_mask:
            return logits.masked_fill(~mask, neg_inf)

        if padding_value is not None:
            padding_mask = mask == padding_value
            if padding_value != 0:
                # Avoid invalid indices in mask.
                mask = mask.masked_fill(padding_mask, 0)
        logits = logits.gather(dim=-1, index=mask)
        if padding_value is not None:
            logits.masked_fill_(padding_mask, neg_inf)
        return logits

    @property
    def deterministic_sample(self):
        return self.mode


class MaskedOneHotCategorical(MaskedCategorical):
    """MaskedCategorical distribution.

    Reference:
    https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities. If provided, the probabilities
            corresponding to masked items will be zeroed and the probability
            re-normalized along its last dimension.

    Keyword Args:
        mask (torch.Tensor): A boolean mask of the same shape as ``logits``/``probs``
            where ``False`` entries are the ones to be masked. Alternatively,
            if ``sparse_mask`` is True, it represents the list of valid indices
            in the distribution. Exclusive with ``indices``.
        indices (torch.Tensor): A dense index tensor representing which actions
            must be taken into account. Exclusive with ``mask``.
        neg_inf (:obj:`float`, optional): The log-probability value allocated to
            invalid (out-of-mask) indices. Defaults to -inf.
        padding_value: The padding value in then mask tensor when
            sparse_mask == True, the padding_value will be ignored.
        grad_method (ReparamGradientStrategy, optional): strategy to gather
            reparameterized samples.
            ``ReparamGradientStrategy.PassThrough`` will compute the sample gradients
             by using the softmax valued log-probability as a proxy to the
             samples gradients.
            ``ReparamGradientStrategy.RelaxedOneHot`` will use
            :class:`torch.distributions.RelaxedOneHot` to sample from the distribution.

    Examples:
        >>> torch.manual_seed(0)
        >>> logits = torch.randn(4) / 100  # almost equal probabilities
        >>> mask = torch.tensor([True, False, True, True])
        >>> dist = MaskedOneHotCategorical(logits=logits, mask=mask)
        >>> sample = dist.sample((10,))
        >>> print(sample)  # no `1` in the sample
        tensor([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0]])
        >>> print(dist.log_prob(sample))
        tensor([-1.1203, -1.0928, -1.0831, -1.1203, -1.1203, -1.0831, -1.1203, -1.0831,
                -1.1203, -1.1203])
        >>> sample_non_valid = torch.zeros_like(sample)
        >>> sample_non_valid[..., 1] = 1
        >>> print(dist.log_prob(sample_non_valid))
        tensor([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
        >>> # with probabilities
        >>> prob = torch.ones(10)
        >>> prob = prob / prob.sum()
        >>> mask = torch.tensor([False] + 9 * [True])  # first outcome is masked
        >>> dist = MaskedOneHotCategorical(probs=prob, mask=mask)
        >>> s = torch.arange(10)
        >>> s = torch.nn.functional.one_hot(s, 10)
        >>> print(dist.log_prob(s))
        tensor([   -inf, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972,
                -2.1972, -2.1972])
    """

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        mask: torch.Tensor = None,
        indices: torch.Tensor = None,
        neg_inf: float = float("-inf"),
        padding_value: int | None = None,
        grad_method: ReparamGradientStrategy = ReparamGradientStrategy.PassThrough,
    ) -> None:
        self.grad_method = grad_method
        super().__init__(
            logits=logits,
            probs=probs,
            mask=mask,
            indices=indices,
            neg_inf=neg_inf,
            padding_value=padding_value,
        )

    @_one_hot_wrapper(MaskedCategorical)
    def sample(
        self, sample_shape: torch.Size | Sequence[int] | None = None
    ) -> torch.Tensor:
        ...

    @property
    def deterministic_sample(self):
        return self.mode

    @property
    def mode(self) -> torch.Tensor:
        if hasattr(self, "logits"):
            return (self.logits == self.logits.max(-1, True)[0]).to(torch.long)
        else:
            return (self.probs == self.probs.max(-1, True)[0]).to(torch.long)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.argmax(dim=-1))

    def rsample(self, sample_shape: torch.Size | Sequence = None) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        if hasattr(self, "logits") and self.logits is not None:
            logits = self.logits
            probs = None
        else:
            logits = None
            probs = self.probs
        if self.grad_method == ReparamGradientStrategy.RelaxedOneHot:
            if self._sparse_mask:
                if probs is not None:
                    probs_extended = torch.full(
                        (*probs.shape[:-1], self.num_samples),
                        0,
                        device=probs.device,
                        dtype=probs.dtype,
                    )
                    probs_extended = torch.scatter(
                        probs_extended, -1, self._mask, probs
                    )
                    logits_extended = None
                else:
                    probs_extended = torch.full(
                        (*logits.shape[:-1], self.num_samples),
                        self.neg_inf,
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    logits_extended = torch.scatter(
                        probs_extended, -1, self._mask, logits
                    )
                    probs_extended = None
            else:
                probs_extended = probs
                logits_extended = logits

            d = D.relaxed_categorical.RelaxedOneHotCategorical(
                1.0, probs=probs_extended, logits=logits_extended
            )
            out = d.rsample(sample_shape)
            out.data.copy_((out == out.max(-1)[0].unsqueeze(-1)).to(out.dtype))
            return out
        elif self.grad_method == ReparamGradientStrategy.PassThrough:
            if logits is not None:
                probs = self.probs
            else:
                probs = torch.softmax(self.logits, dim=-1)
            if self._sparse_mask:
                probs_extended = torch.full(
                    (*probs.shape[:-1], self.num_samples),
                    0,
                    device=probs.device,
                    dtype=probs.dtype,
                )
                probs_extended = torch.scatter(probs_extended, -1, self._mask, probs)
            else:
                probs_extended = probs

            out = self.sample(sample_shape)
            out = out + probs_extended - probs_extended.detach()
            return out
        else:
            raise ValueError(
                f"Unknown reparameterization strategy {self.reparam_strategy}."
            )


class Ordinal(D.Categorical):
    """A discrete distribution for learning to sample from finite ordered sets.

    It is defined in contrast with the `Categorical` distribution, which does
    not impose any notion of proximity or ordering over its support's atoms.
    The `Ordinal` distribution explicitly encodes those concepts, which is
    useful for learning discrete sampling from continuous sets. See §5 of
    `Tang & Agrawal, 2020<https://arxiv.org/pdf/1901.10500.pdf>`_ for details.

    .. note::
        This class is mostly useful when you want to learn a distribution over
        a finite set which is obtained by discretising a continuous set.

    Args:
        scores (torch.Tensor): a tensor of shape [..., N] where N is the size of the set which supports the distributions.
            Typically, the output of a neural network parametrising the distribution.

    Examples:
        >>> num_atoms, num_samples = 5, 20
        >>> mean = (num_atoms - 1) / 2  # Target mean for samples, centered around the middle atom
        >>> torch.manual_seed(42)
        >>> logits = torch.ones((num_atoms), requires_grad=True)
        >>> optimizer = torch.optim.Adam([logits], lr=0.1)
        >>>
        >>> # Perform optimisation loop to minimise deviation from `mean`
        >>> for _ in range(20):
        >>>     sampler = Ordinal(scores=logits)
        >>>     samples = sampler.sample((num_samples,))
        >>>     # Define loss to encourage samples around the mean by penalising deviation from mean
        >>>     loss = torch.mean((samples - mean) ** 2 * sampler.log_prob(samples))
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()
        >>>
        >>> sampler.probs
        tensor([0.0308, 0.1586, 0.4727, 0.2260, 0.1120], ...)
        >>> # Print histogram to observe sample distribution frequency across 5 bins (0, 1, 2, 3, and 4)
        >>> torch.histogram(sampler.sample((1000,)).reshape(-1).float(), bins=num_atoms)
        torch.return_types.histogram(
            hist=tensor([ 24., 158., 478., 228., 112.]),
            bin_edges=tensor([0.0000, 0.8000, 1.6000, 2.4000, 3.2000, 4.0000]))
    """

    def __init__(self, scores: torch.Tensor):
        logits = _generate_ordinal_logits(scores)
        super().__init__(logits=logits)


class OneHotOrdinal(OneHotCategorical):
    """The one-hot version of the :class:`~tensordict.nn.distributions.Ordinal` distribution.

    Args:
        scores (torch.Tensor): a tensor of shape [..., N] where N is the size of the set which supports the distributions.
            Typically, the output of a neural network parametrising the distribution.
    """

    def __init__(self, scores: torch.Tensor):
        logits = _generate_ordinal_logits(scores)
        super().__init__(logits=logits)


def _generate_ordinal_logits(scores: torch.Tensor) -> torch.Tensor:
    """Implements Eq. 4 of `Tang & Agrawal, 2020<https://arxiv.org/pdf/1901.10500.pdf>`__."""
    # Assigns Bernoulli-like probabilities for each class in the set
    log_probs = F.logsigmoid(scores)
    complementary_log_probs = F.logsigmoid(-scores)

    # Total log-probability for being "larger than k"
    larger_than_log_probs = log_probs.cumsum(dim=-1)

    # Total log-probability for being "smaller than k"
    smaller_than_log_probs = (
        complementary_log_probs.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        - complementary_log_probs
    )

    return larger_than_log_probs + smaller_than_log_probs


class LLMMaskedCategorical(D.Distribution):
    """LLM-optimized masked categorical distribution.

    This class provides a more memory-efficient approach for LLM training by:
    1. Using ignore_index=-100 for log_prob computation (no masking overhead)
    2. Using traditional masking for sampling operations

    This is particularly beneficial for large vocabulary sizes where masking
    all logits can be memory-intensive.

    Args:
        logits (torch.Tensor): Event log probabilities (unnormalized), shape [B, T, C].
            - *B*: batch size (optional)
            - T: sequence length
            - C: vocabulary size (number of classes)
        mask (torch.Tensor): Boolean mask indicating valid positions/tokens.
            - If shape [*B, T]: position-level masking. True means the position is valid (all tokens allowed).
            - If shape [*B, T, C]: token-level masking. True means the token is valid at that position.

               .. warning:: Token-level masking is considerably more memory-intensive than position-level masking.
                   Only use this if you need to mask tokens.

        ignore_index (int, optional): Index to ignore in log_prob computation. Defaults to -100.

    Input shapes:
        - logits: [*B, T, C] (required)
        - mask: [*B, T] (position-level) or [*B, T, C] (token-level)
        - tokens (for log_prob): [*B, T] (token indices, with ignore_index for masked positions)

    Use cases:
        1. **Position-level masking**
            >>> logits = torch.randn(2, 10, 50000)  # [B=2, T=10, C=50000]
            >>> mask = torch.ones(2, 10, dtype=torch.bool)  # [B, T]
            >>> mask[0, :5] = False  # mask first 5 positions of first sequence
            >>> dist = LLMMaskedCategorical(logits=logits, mask=mask)
            >>> tokens = torch.randint(0, 50000, (2, 10))  # [B, T]
            >>> tokens[0, :5] = -100  # set masked positions to ignore_index
            >>> log_probs = dist.log_prob(tokens)
            >>> samples = dist.sample()  # [B, T]

        2. **Token-level masking**
            >>> logits = torch.randn(2, 10, 50000)
            >>> mask = torch.ones(2, 10, 50000, dtype=torch.bool)  # [B, T, C]
            >>> mask[0, :5, :1000] = False  # mask first 1000 tokens for first 5 positions
            >>> dist = LLMMaskedCategorical(logits=logits, mask=mask)
            >>> tokens = torch.randint(0, 50000, (2, 10))
            >>> # Optionally, set tokens at fully-masked positions to ignore_index
            >>> log_probs = dist.log_prob(tokens)
            >>> samples = dist.sample()  # [B, T]

    Notes:
        - For log_prob, tokens must be of shape [B, T] and contain valid token indices (0 <= token < C), or ignore_index for masked/ignored positions.
        - For token-level masking, if a token is masked at a given position, log_prob will return -inf for that entry.
        - For position-level masking, if a position is masked (ignore_index), log_prob will return 0.0 for that entry (correct for cross-entropy loss).
        - Sampling always respects the mask (masked tokens/positions are never sampled).

    All documented use cases are covered by tests in test_distributions.py.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        ignore_index: int = -100,
    ) -> None:
        # Validate shapes
        if logits.shape[:-1] != mask.shape and logits.shape != mask.shape:
            raise ValueError(
                f"Mask shape {mask.shape} must be either logits batch shape {logits.shape[:-1]} "
                f"(for position-level masking) or logits shape {logits.shape} "
                f"(for token-level masking)"
            )

        self._original_logits = logits
        self._mask = mask
        self.ignore_index = ignore_index
        self._position_level_masking = mask.shape == logits.shape[:-1]

        # Create masked logits for sampling (only when needed)
        self._masked_logits = None
        self._masked_dist = None

        # Set up distribution properties
        batch_shape = logits.shape[:-1]
        event_shape = logits.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def _sampling_logits(self):
        """Get masked logits for sampling operations."""
        if self._masked_logits is None:
            # Only create masked logits when needed for sampling
            large_neg = torch.finfo(self._original_logits.dtype).min

            if self._position_level_masking:
                # Position-level masking: expand mask to match logits shape
                mask_expanded = expand_as_right(self._mask, self._original_logits)
                self._masked_logits = self._original_logits.masked_fill(
                    ~mask_expanded, large_neg
                )
            else:
                # Token-level masking: direct masking
                self._masked_logits = self._original_logits.masked_fill(
                    ~self._mask, large_neg
                )
        return self._masked_logits

    @property
    def _sampling_dist(self):
        """Get masked distribution for sampling operations."""
        if self._masked_dist is None:
            self._masked_dist = D.Categorical(logits=self._sampling_logits)
        return self._masked_dist

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities using ignore_index approach.

        This is memory-efficient as it doesn't require masking the logits.
        The value tensor should use ignore_index for masked positions.
        """
        if not self._position_level_masking:
            logits = self.masked_logits
        else:
            # Use cross_entropy with ignore_index for efficiency

            # For position-level masking, keep the default behavior (0.0 for ignore_index)
            # This is correct for cross-entropy loss computation
            # For token-level masking, we need to check if specific tokens are masked

            logits = self._original_logits
            value = value.masked_fill(~self._mask, self.ignore_index)
        if value.ndim > 1:
            # Reshape for cross_entropy: (batch, seq_len, vocab) -> (batch*seq_len, vocab)
            logits_flat = logits.reshape(-1, logits.size(-1))
            value_flat = value.reshape(-1)

            # Compute cross_entropy with ignore_index
            log_probs_flat = -F.cross_entropy(
                logits_flat, value_flat, reduce=False, ignore_index=self.ignore_index
            )

            # Reshape back
            log_probs = log_probs_flat.reshape_as(value)
        else:
            log_probs = -F.cross_entropy(
                logits,
                value,
                reduce=False,
                ignore_index=self.ignore_index,
            )
        return log_probs

    def sample(
        self, sample_shape: torch.Size | Sequence[int] | None = None
    ) -> torch.Tensor:
        """Sample from the distribution using masked logits."""
        if sample_shape is None:
            sample_shape = torch.Size()
        return self._sampling_dist.sample(sample_shape)

    def rsample(
        self, sample_shape: torch.Size | Sequence[int] | None = None
    ) -> torch.Tensor:
        """Reparameterized sampling using masked logits."""
        # This would need to be implemented based on the specific reparameterization strategy
        # For now, fall back to regular sampling
        return self.sample(sample_shape)

    @property
    def mode(self) -> torch.Tensor:
        """Get the mode using masked logits."""
        masked_logits = self._sampling_logits
        return masked_logits.argmax(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Compute entropy using masked logits."""
        return self._sampling_dist.entropy()

    def clear_cache(self):
        """Clear cached masked tensors to free memory."""
        self._masked_logits = None
        self._masked_dist = None

    @property
    def mask(self) -> torch.Tensor:
        """Get the mask."""
        return self._mask

    @property
    def logits(self) -> torch.Tensor:
        """Get the original logits."""
        return self._original_logits

    @property
    def probs(self) -> torch.Tensor:
        """Get probabilities from original logits."""
        return torch.softmax(self._original_logits, dim=-1)

    @property
    def masked_logits(self) -> torch.Tensor:
        """Get the masked logits for sampling operations."""
        return self._sampling_logits

    @property
    def masked_dist(self) -> D.Categorical:
        """Get the masked distribution for sampling operations."""
        return self._sampling_dist

    @property
    def position_level_masking(self) -> bool:
        """Whether the mask is position-level (True) or token-level (False)."""
        return self._position_level_masking
