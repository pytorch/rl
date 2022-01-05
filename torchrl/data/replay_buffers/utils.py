import numpy as np
import torch
import tree
from torch import Tensor


def fields_pin_memory(input):
    return tree.map_structure(lambda x: pin_memory(x), input)


def pin_memory(data: Tensor) -> Tensor:
    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    else:
        return data


def to_numpy(data: Tensor) -> np.ndarray:
    return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else \
        data


def fast_map(func, *inputs):
    flat_inputs = (tree.flatten(x) for x in inputs)
    entries = zip(*flat_inputs)
    return tree.unflatten_as(inputs[-1], [func(*x) for x in entries])


def stack_tensors(input):
    if not len(input):
        raise RuntimeError("input length must be non-null")
    if isinstance(input[0], torch.Tensor):
        size = input[0].size()
        if len(size) == 0:
            return torch.stack(input)
        else:
            # torch.cat is much faster than torch.stack
            # https://github.com/pytorch/pytorch/issues/22462
            return torch.cat(input).view(-1, *size)
    else:
        return np.stack(input)


def stack_fields(input):
    if not len(input):
        raise RuntimeError("stack_fields requires non-empty list if tensors")
    return fast_map(lambda *x: stack_tensors(x), *input)


def first_field(data) -> Tensor:
    return next(iter(tree.flatten(data)))


def to_torch(data: Tensor, device,
             pin_memory: bool = False,
             non_blocking: bool = False) -> torch.Tensor:
    if isinstance(data, np.generic):
        return torch.tensor(data, device=device)

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    if pin_memory:
        data = data.pin_memory()
    if device is not None:
        data = data.to(device, non_blocking=non_blocking)

    return data


def cat_fields_to_device(input, device,
                         pin_memory: bool = False,
                         non_blocking: bool = False):
    input_on_device = fields_to_device(input, device, pin_memory, non_blocking)
    return cat_fields(input_on_device)


def cat_fields(input):
    if not input:
        raise RuntimeError("cat_fields requires a non-empty input collection.")
    return fast_map(lambda *x: torch.cat(x), *input)


def fields_to_device(input, device,
                     pin_memory: bool = False,
                     non_blocking: bool = False):
    if device is None or device == "cpu" or device == torch.device("cpu"):
        if pin_memory:
            return tree.map_structure(lambda x: x.cpu().pin_memory(), input)
        else:
            return tree.map_structure(lambda x: x.cpu(), input)
    else:
        if pin_memory:
            return tree.map_structure(lambda x: x.pin_memory().to(
                device, non_blocking=non_blocking), input)
        else:
            return tree.map_structure(lambda x: x.cuda(device), input)
