import torch


def expand_as_right(tensor: torch.Tensor, dest: torch.Tensor):
    if dest.ndimension() < tensor.ndimension():
        raise RuntimeError(
            "expand_as_right requires the destination tensor to have less dimensions than the input tensor, got"
            f"got tensor.ndimension()={tensor.ndimension()} and dest.ndimension()={dest.ndimension()}")
    if not (
            tensor.shape == dest.shape[:tensor.ndimension()]
    ):
        raise RuntimeError(
            f"tensor shape is incompatible with dest shape, got: tensor.shape={tensor.shape}, dest={dest.shape}")
    for _ in range(dest.ndimension() - tensor.ndimension()):
        tensor = tensor.unsqueeze(-1)
    return tensor.expand_as(dest)
