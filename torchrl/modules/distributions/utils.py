from typing import Union

import torch


def _cast_device(elt: Union[torch.Tensor, float], device) -> Union[torch.Tensor, float]:
    if isinstance(elt, torch.Tensor):
        return elt.to(device)
    return elt
