import torch
from torch.utils._pytree import tree_map
import contextlib

@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard

class FiniteTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        assert torch.isfinite(elem).all()
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

    def __repr__(self):
        return f"FiniteTensor({super().__repr__()})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # TODO: also explicitly recheck invariants on inplace/out mutation
        assert not kwargs
        with no_dispatch():
            rs = func(*args)
        return tree_map(lambda e: FiniteTensor(e) if isinstance(e, torch.Tensor) else e, rs)
