from typing import Union

import torch
from torch import distributions as d


def _cast_device(elt: Union[torch.Tensor, float], device) -> Union[torch.Tensor, float]:
    if isinstance(elt, torch.Tensor):
        return elt.to(device)
    return elt


def _cast_transform_device(transform, device):
    if transform is None:
        return transform
    elif isinstance(transform, d.ComposeTransform):
        for i, t in enumerate(transform.parts):
            transform.parts[i] = _cast_transform_device(t, device)
    elif isinstance(transform, d.Transform):
        for attribute in dir(transform):
            value = getattr(transform, attribute)
            if isinstance(value, torch.Tensor):
                setattr(transform, attribute, value.to(device))
        return transform
    else:
        raise TypeError(
            f"Cannot perform device casting for transform of type {type(transform)}"
        )
