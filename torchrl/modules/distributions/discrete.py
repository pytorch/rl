# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from functools import wraps
from typing import Any, Optional, Sequence, Union

import torch
import torch.distributions as D

__all__ = [
    "OneHotCategorical",
    "MaskedCategorical",
]


def _treat_categorical_params(
    params: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
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
    PassThrough: Any = 1
    RelaxedOneHot: Any = 2


class OneHotCategorical(D.Categorical):
    """One-hot categorical distribution.

    This class behaves excacly as torch.distributions.Categorical except that it reads and produces one-hot encodings
    of the discrete tensors.

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities
        grad_method (ReparamGradientStrategy, optional): strategy to gather
            reparameterized samples.
            ``ReparamGradientStrategy.PassThrough`` will compute the sample gradients
             by using the softmax valued log-probability as a proxy to the
             samples gradients.
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

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        grad_method: ReparamGradientStrategy = ReparamGradientStrategy.PassThrough,
        **kwargs,
    ) -> None:
        logits = _treat_categorical_params(logits)
        probs = _treat_categorical_params(probs)
        self.grad_method = grad_method
        super().__init__(probs=probs, logits=logits, **kwargs)
        self.num_samples = self._param.shape[-1]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.argmax(dim=-1))

    @property
    def mode(self) -> torch.Tensor:
        if hasattr(self, "logits"):
            return (self.logits == self.logits.max(-1, True)[0]).to(torch.long)
        else:
            return (self.probs == self.probs.max(-1, True)[0]).to(torch.long)

    @_one_hot_wrapper(D.Categorical)
    def sample(
        self, sample_shape: Optional[Union[torch.Size, Sequence]] = None
    ) -> torch.Tensor:
        ...

    def rsample(self, sample_shape: Union[torch.Size, Sequence] = None) -> torch.Tensor:
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
                f"Unknown reparametrization strategy {self.reparam_strategy}."
            )


class MaskedCategorical(D.Categorical):
    """MaskedCategorical distribution.

    Reference:
    https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities. If provided, the probabilities
            corresponding to to masked items will be zeroed and the probability
            re-normalized along its last dimension.

    Keyword Args:
        mask (torch.Tensor): A boolean mask of the same shape as ``logits``/``probs``
            where ``False`` entries are the ones to be masked. Alternatively,
            if ``sparse_mask`` is True, it represents the list of valid indices
            in the distribution. Exclusive with ``indices``.
        indices (torch.Tensor): A dense index tensor representing which actions
            must be taken into account. Exclusive with ``mask``.
        neg_inf (float, optional): The log-probability value allocated to
            invalid (out-of-mask) indices. Defaults to -inf.
        padding_value: The padding value in the then mask tensor when
            sparse_mask == True, the padding_value will be ignored.

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

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        *,
        mask: torch.Tensor = None,
        indices: torch.Tensor = None,
        neg_inf: float = float("-inf"),
        padding_value: Optional[int] = None,
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
            probs[~mask] = 0
            probs = probs / probs.sum(-1, keepdim=True)
            logits = probs.log()
        num_samples = logits.shape[-1]
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
        super().__init__(logits=logits)
        self.num_samples = num_samples

    def sample(
        self, sample_shape: Optional[Union[torch.Size, Sequence[int]]] = None
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
            return super().log_prob(value)

        idx_3d = self._mask.view(1, -1, self._num_events)
        val_3d = value.view(-1, idx_3d.size(1), 1)
        mask = idx_3d == val_3d
        idx = mask.int().argmax(dim=-1, keepdim=True)
        ret = super().log_prob(idx.view_as(value))
        # Fill masked values with neg_inf.
        ret = ret.view_as(val_3d)
        ret = ret.masked_fill(
            torch.logical_not(mask.any(dim=-1, keepdim=True)), self.neg_inf
        )
        return ret.resize_as(value)

    @staticmethod
    def _mask_logits(
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        neg_inf: float = float("-inf"),
        sparse_mask: bool = False,
        padding_value: Optional[int] = None,
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


class MaskedOneHotCategorical(MaskedCategorical):
    """MaskedCategorical distribution.

    Reference:
    https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities. If provided, the probabilities
            corresponding to to masked items will be zeroed and the probability
            re-normalized along its last dimension.

    Keyword Args:
        mask (torch.Tensor): A boolean mask of the same shape as ``logits``/``probs``
            where ``False`` entries are the ones to be masked. Alternatively,
            if ``sparse_mask`` is True, it represents the list of valid indices
            in the distribution. Exclusive with ``indices``.
        indices (torch.Tensor): A dense index tensor representing which actions
            must be taken into account. Exclusive with ``mask``.
        neg_inf (float, optional): The log-probability value allocated to
            invalid (out-of-mask) indices. Defaults to -inf.
        padding_value: The padding value in the then mask tensor when
            sparse_mask == True, the padding_value will be ignored.
        grad_method (ReparamGradientStrategy, optional): strategy to gather
            reparameterized samples.
            ``ReparamGradientStrategy.PassThrough`` will compute the sample gradients
             by using the softmax valued log-probability as a proxy to the
             samples gradients.
            ``ReparamGradientStrategy.RelaxedOneHot`` will use
            :class:`torch.distributions.RelaxedOneHot` to sample from the distribution.

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

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        indices: torch.Tensor = None,
        neg_inf: float = float("-inf"),
        padding_value: Optional[int] = None,
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
        self, sample_shape: Optional[Union[torch.Size, Sequence[int]]] = None
    ) -> torch.Tensor:
        ...

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.argmax(dim=-1))

    def rsample(self, sample_shape: Union[torch.Size, Sequence] = None) -> torch.Tensor:
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
                f"Unknown reparametrization strategy {self.reparam_strategy}."
            )
