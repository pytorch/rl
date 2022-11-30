# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Union

import torch
from torch import distributions as D

__all__ = [
    "OneHotCategorical",
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
