import torch


def expand_as_right(tensor: torch.Tensor, dest: torch.Tensor):
    assert dest.ndimension() >= tensor.ndimension()
    assert (
            tensor.shape == dest.shape[:tensor.ndimension()]
    ), f"tensor shape is incompatible with dest shape, got: tensor.shape={tensor.shape}, dest={dest.shape}"
    for _ in range(dest.ndimension() - tensor.ndimension()):
        tensor = tensor.unsqueeze(-1)
    return tensor.expand_as(dest)
