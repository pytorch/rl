# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import prod
from typing import Optional, Sequence, Union


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


class OneHotCategorical(D.Categorical):
    """One-hot categorical distribution.

    This class behaves excacly as torch.distributions.Categorical except that it reads and produces one-hot encodings
    of the discrete tensors.

    """

    num_params: int = 1

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        logits = _treat_categorical_params(logits)
        probs = _treat_categorical_params(probs)
        super().__init__(probs=probs, logits=logits, **kwargs)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.argmax(dim=-1))

    @property
    def mode(self) -> torch.Tensor:
        if hasattr(self, "logits"):
            return (self.logits == self.logits.max(-1, True)[0]).to(torch.long)
        else:
            return (self.probs == self.probs.max(-1, True)[0]).to(torch.long)

    def sample(
        self, sample_shape: Optional[Union[torch.Size, Sequence]] = None
    ) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        out = super().sample(sample_shape=sample_shape)
        out = torch.nn.functional.one_hot(out, self.logits.shape[-1]).to(torch.long)
        return out

    def rsample(self, sample_shape: Union[torch.Size, Sequence] = None) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        d = D.relaxed_categorical.RelaxedOneHotCategorical(
            1.0, probs=self.probs, logits=self.logits
        )
        out = d.rsample(sample_shape)
        out.data.copy_((out == out.max(-1)[0].unsqueeze(-1)).to(out.dtype))
        return out


class MaskedCategorical(D.Categorical):
    """MaskedCategorical distribution.
    Reference:
    https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical


    sparse_mask: True when we only pass indices of True values in the mask
        tensor.
    padding_value: The padding value in the then mask tensor when
        sparse_mask == True, the padding_value will be ignored.
    """

    # TODO: Design the APIs with probs, here we ony have logits.
    def __init__(self,
                 logits: torch.Tensor,
                 mask: torch.Tensor,
                 neg_inf: float = float("-inf"),
                 sparse_mask: bool = False,
                 padding_value: Optional[int] = None) -> None:
        logits = self._mask_logits(logits,
                                   mask,
                                   neg_inf=neg_inf,
                                   sparse_mask=sparse_mask,
                                   padding_value=padding_value)
        self._mask = mask
        self._sparse_mask = sparse_mask
        self._padding_value = padding_value
        super().__init__(logits=logits)

    def sample(
        self, sample_shape: Union[torch.Size, Sequence[int]] = torch.Size()
    ) -> torch.Tensor:
        ret = super().sample(sample_shape)
        if not self._sparse_mask:
            return ret

        ret_size = ret.size()
        outer_dim = prod(sample_shape)
        inner_dim = prod(self._mask.size()[:-1])
        mask_3d = self._mask.expand(outer_dim, inner_dim, -1)
        ret = mask_3d.gather(dim=-1, index=ret.view(outer_dim, inner_dim, 1))
        return ret.view(ret_size)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self._sparse_mask:
            return super().log_prob(value)
        value_size = value.size()
        mask_3d = self._mask.view(1, -1, self._num_events)
        value_3d = value.view(-1, mask_3d.size(1), 1)
        index = (mask_3d == value_3d).long().argmax(dim=-1, keepdim=True)
        return super().log_prob(index.view(value_size))

    @staticmethod
    def _mask_logits(logits: torch.Tensor,
                     mask: torch.Tensor,
                     neg_inf: float = float("-inf"),
                     sparse_mask: bool = False,
                     padding_value: Optional[int] = None) -> torch.Tensor:
        if not sparse_mask:
            return torch.where(mask, logits, neg_inf)

        if padding_value is not None:
            padding_mask = (mask == padding_value)
            if padding_value != 0:
                # Avoid invalid indices in mask.
                mask = mask.masked_fill(padding_mask, 0)
        logits = logits.gather(dim=-1, index=mask)
        if padding_value is not None:
            logits.masked_fill_(padding_mask, neg_inf)
        return logits
