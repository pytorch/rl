from typing import Optional, Union, Iterable

import torch
from torch import distributions as D

__all__ = [
    "Categorical",
]


def _treat_categorical_params(params: Optional[torch.Tensor] = None) -> torch.Tensor:
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


class Categorical(D.Categorical):
    def __init__(self, logits: Optional[torch.Tensor] = None, probs: Optional[torch.Tensor] = None, *args, **kwargs):
        logits = _treat_categorical_params(logits)
        probs = _treat_categorical_params(probs)
        return super().__init__(probs=probs, logits=logits, *args, **kwargs)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.argmax(dim=-1))

    @property
    def mode(self) -> torch.Tensor:
        if hasattr(self, "logits"):
            return (self.logits == self.logits.max(-1, True)[0]).to(torch.long)
        else:
            return (self.probs == self.probs.max(-1, True)[0]).to(torch.long)

    def sample(self, sample_shape: Union[torch.Size, Iterable]=torch.Size([])) -> torch.Tensor:
        out = super().sample(sample_shape=sample_shape)
        out = torch.nn.functional.one_hot(out).to(torch.long)
        return out
