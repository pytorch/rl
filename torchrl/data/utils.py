from typing import Union, List, Callable, Any, Tuple

import numpy as np
import torch
from torch import Tensor

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}
torch_to_numpy_dtype_dict = {
    value: key for key, value in numpy_to_torch_dtype_dict.items()
}
DEVICE_TYPING = Union[torch.device, str]  # , int]

INDEX_TYPING = Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]


class CloudpickleWrapper(object):
    def __init__(self, fn: Callable):
        if fn.__class__.__name__ == "EnvCreator":
            raise RuntimeError(
                "CloudpickleWrapper usage with EnvCreator class is prohibited as it breaks the "
                "transmission of shared tensors."
            )
        self.fn = fn

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob: bytes):
        import pickle

        self.fn = pickle.loads(ob)

    def __call__(self, **kwargs) -> Any:
        return self.fn(**kwargs)
