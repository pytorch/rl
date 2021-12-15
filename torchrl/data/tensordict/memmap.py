from __future__ import annotations

import functools
import tempfile
from typing import List, Union, Optional

import numpy as np
import torch

from torchrl.data.utils import torch_to_numpy_dtype_dict

MEMMAP_HANDLED_FN = {}

__all__ = ["MemmapTensor", "set_transfer_ownership"]


def implements_for_memmap(torch_function):
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        MEMMAP_HANDLED_FN[torch_function] = func
        return func

    return decorator

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor

class MemmapTensor(object):
    def __init__(self, elem: Union[torch.Tensor, MemmapTensor], transfer_ownership: bool = False):
        """
        An torch.tensor interface with a np.memmap array. A temporary file is created and cleared once the object is
        out-of-scope. This class is aimed at being used for data transfer in between processes and remote workers,
        and as such it supports serialization and deserialization.
        As a consequence, one must choose if the ownership is transferred upon serialization / deserialization. If
        owenership is not transferred (transfer_ownership=False, default), then the process where the MemmapTensor was
        created will be responsible of clearing it once it gets out of scope in that process. Otherwise the process that
        deserialize the MemmapTensor will be responsible of clearing the files once the object is out of scope.
        Supports all tensor operations.
        Args:
            elem: Tensor or MemmapTensor. If MemmapTensor, a new MemmapTensor is created and the same data is stored in it.
            transfer_ownership: affects the ownership after serialization: if True, the current process looses ownership
                immediately after serialization. If False, the current process keeps the ownership of the temporary file.
        """
        if not isinstance(elem, (torch.Tensor, MemmapTensor)):
            raise TypeError("convert input to torch.Tensor before calling MemmapTensor() on it.")

        assert not elem.requires_grad, "MemmapTensor is incompatible with tensor.requires_grad. " \
                                       "Consider calling tensor.detach() first."

        self.idx = None
        self._memmap_array = None
        self.file = tempfile.NamedTemporaryFile()
        self.filename = self.file.name
        self._device = elem.device
        self._shape = elem.shape
        self.transfer_ownership = transfer_ownership
        self.np_shape = tuple(self._shape)
        self._dtype = elem.dtype
        self._tensor_dir = elem.__dir__()
        self._ndim = elem.ndimension()
        self._numel = elem.numel()
        self.mode = 'r+'
        self._has_ownership = True
        if isinstance(elem, MemmapTensor):
            prev_filename = elem.filename
            self._copy_item(prev_filename)
        else:
            self._save_item(elem)

    def _get_memmap_array(self):
        if self._memmap_array is None:
            self._memmap_array = np.memmap(
                self.filename,
                dtype=torch_to_numpy_dtype_dict[self.dtype],
                mode=self.mode,
                shape=self.np_shape)
        return self._memmap_array

    def _set_memmap_array(self, value):
        self._memmap_array = value

    memmap_array = property(_get_memmap_array, _set_memmap_array)

    def _save_item(self, value, idx=None):
        if isinstance(value, (torch.Tensor,)):
            np_array = value.cpu().numpy()
        else:
            np_array = value
        memmap_array = self.memmap_array
        if idx is None:
            memmap_array[:] = np_array
        else:
            memmap_array[idx] = np_array

    def _copy_item(self, filename):
        self.memmap_array[:] = np.memmap(
            filename,
            dtype=torch_to_numpy_dtype_dict[self.dtype],
            mode="r",
            shape=self.np_shape)
        # shutil.copyfile(filename, self.filename)

    def _load_item(self, idx=None, memmap_array=None):
        if memmap_array is None:
            memmap_array = self.memmap_array
        if idx is not None:
            memmap_array = memmap_array[idx]
        return self._np_to_tensor(memmap_array)

    def _np_to_tensor(self, memmap_array):
        return torch.from_numpy(memmap_array)#, device=self.device)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in MEMMAP_HANDLED_FN:
            args = [a._tensor if hasattr(a, '_tensor') else a for a in args]
            ret = func(*args, **kwargs)
            return ret

        return MEMMAP_HANDLED_FN[func](*args, **kwargs)

    @property
    def _tensor(self):
        return self._load_item()

    def ndimension(self):
        return self._ndim

    def numel(self):
        return self._numel

    def clone(self):
        return MemmapTensor(self)

    def contiguous(self):
        return self._tensor.clone()

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def cpu(self):
        return self._tensor.cpu()

    def numpy(self):
        return self._tensor.numpy()

    def copy_(self, other):
        self._save_item(other)
        return self

    def set_transfer_ownership(self, value: bool = True):
        assert isinstance(value,
                          bool), f"value provided to set_transfer_ownership should be a boolean, got {type(value)}"
        self.transfer_ownership = value
        return self

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()

    def __eq__(self, other):
        return self._tensor == other

    def __getattr__(self, attr):
        if attr in self.__dir__():
            print(f"loading {attr} has raised an exception")
            return self.__getattribute__(attr)  # make sure that appropriate exceptions are raised
        if not attr in self.__getattribute__("_tensor_dir"):
            raise AttributeError(f"{attr} not found")
        _tensor = self.__getattribute__("_tensor")
        return getattr(_tensor, attr)

        # if not hasattr(torch.Tensor, attr):
        #     raise AttributeError(attr)
        # return getattr(self._tensor, attr)

    def is_shared(self):
        return False

    def __add__(self, other):
        return torch.add(self, other)

    def __div__(self, other):
        return torch.div(self, other)

    def __neg__(self):
        return torch.neg(self)

    def __diff__(self, other):
        return torch.diff(self, other)

    def __matmul__(self, other):
        return torch.matmul(self, other)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __pow__(self, other):
        return torch.pow(self, other)

    def __repr__(self):
        return f"MemmapTensor(shape={self.shape}, device={self.device}, dtype={self.dtype})"

    def __getitem__(self, item):
        # return self._load_item(memmap_array=self.memmap_array[item])#[item]
        return self._load_item()[item]

    def __setitem__(self, idx, value):
        # self.memmap_array[idx] = to_numpy(value)
        self._load_item()[idx] = value

    def __setstate__(self, state):
        if state['file'] is None:
            delete = state['transfer_ownership'] and state['_has_ownership']
            state['_has_ownership'] = delete
            tmpfile = tempfile.NamedTemporaryFile(delete=delete)
            tmpfile.name = state['filename']
            tmpfile._closer.name = state['filename']
            state['file'] = tmpfile
        self.__dict__.update(state)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        state['_memmap_array'] = None
        self._has_ownership = self.file.delete
        return state

    def __reduce__(self, *args, **kwargs):
        if self.transfer_ownership:
            self.file.delete = False
            self.file._closer.delete = False
        return super(MemmapTensor, self).__reduce__(*args, **kwargs)

    def to(self, dest):
        if isinstance(dest, (int, str, torch.device)):
            dest = torch.device(dest)
            return self._tensor.to(dest)
        elif isinstance(dest, torch.dtype):
            return MemmapTensor(self._tensor.to(dest))
        else:
            raise NotImplementedError(f"argument dest={dest} to MemmapTensor.to(dest) is not handled. "
                                      f"Please provide a dtype or a device.")

    def unbind(self, dim):
        idx = [(tuple(slice(None) for _ in range(dim)) + (i,)) for i in range(self.shape[dim])]
        return tuple(self[_idx] for _idx in idx)


@implements_for_memmap(torch.stack)
def stack(list_of_memmap: List[MemmapTensor], dim: int, out: Optional[Union[torch.Tensor, MemmapTensor]] = None):
    list_of_tensors = [a._tensor if isinstance(a, MemmapTensor) else a for a in list_of_memmap]
    return torch.stack(list_of_tensors, dim, out=out)


@implements_for_memmap(torch.unbind)
def unbind(memmap: MemmapTensor, dim: int):
    return memmap.unbind(dim)


@implements_for_memmap(torch.cat)
def cat(list_of_memmap: List[MemmapTensor], dim: int, out: Optional[Union[torch.Tensor, MemmapTensor]] = None):
    list_of_tensors = [a._tensor if isinstance(a, MemmapTensor) else a for a in list_of_memmap]
    return torch.cat(list_of_tensors, dim, out=out)


def set_transfer_ownership(memmap: MemmapTensor, value: bool = True):
    if isinstance(memmap, MemmapTensor):
        memmap.set_transfer_ownership(value)
