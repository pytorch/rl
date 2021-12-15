from os import stat
from torch import distributions as D
import torch
import numpy as np
from .utils import UNIFORM

__all__ = [
    "Categorical",
]


def _treat_categorical_params(params):
    if params is None:
        return None
    if params.shape[-1] == 1:
        params = params[..., 0]
    return params


def rand_one_hot(values, do_softmax=True):
    if do_softmax:
        values = values.softmax(-1)
    out = values.cumsum(-1) > torch.rand_like(values[..., :1])
    out = (out.cumsum(-1) == 1).to(torch.long)
    return out


class Categorical(D.Categorical):
    def __init__(self, logits=None, probs=None, *args, **kwargs):
        logits = _treat_categorical_params(logits)
        probs = _treat_categorical_params(probs)
        return super().__init__(probs=probs, logits=logits, *args, **kwargs)

    def log_prob(self, value):
        return super().log_prob(value.argmax(dim=-1))

    @property
    def mode(self):
        if hasattr(self, "logits"):
            return (self.logits == self.logits.max(-1, True)[0]).to(torch.long)
        else:
            return (self.probs == self.probs.max(-1, True)[0]).to(torch.long)

    def sample(self, sample_shape=torch.Size([])):
        out = super().sample(sample_shape=sample_shape)
        out = torch.nn.functional.one_hot(out).to(torch.long)
        return out

