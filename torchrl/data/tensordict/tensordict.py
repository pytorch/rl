from __future__ import annotations

import math
import tempfile
import textwrap
from collections import OrderedDict, Mapping
from collections.abc import Iterable
from copy import deepcopy, copy
from numbers import Number
from textwrap import indent
from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Callable,
    Generator,
    Type,
    KeysView,
    ItemsView,
    Iterator,
    Sequence,
)
from warnings import warn

import numpy as np
import torch

from torchrl.data.postprocs.utils import expand_as_right
from torchrl.data.tensordict.memmap import MemmapTensor
from torchrl.data.tensordict.metatensor import MetaTensor
from torchrl.data.tensordict.utils import _sub_index, _getitem_batch_size
from torchrl.data.utils import INDEX_TYPING, DEVICE_TYPING

__all__ = [
    "TensorDict",
    "SubTensorDict",
    "merge_tensor_dicts",
    "LazyStackedTensorDict",
    "SavedTensorDict",
]

TD_HANDLED_FUNCTIONS = dict()
COMPATIBLE_TYPES = Union[torch.Tensor, None]  # leaves space for _TensorDict
_accepted_classes = (torch.Tensor, MemmapTensor)


class _TensorDict(Mapping):
    """
    _TensorDict is an abstract parent class for TensorDicts, the torchrl data container.
    """

    _safe = False

    def __init__(self, source: Optional[dict] = None):
        raise NotImplementedError

    @property
    def shape(self) -> torch.Size:
        """
        See _TensorDict.batch_size
        """
        return self.batch_size

    @property
    def batch_size(self) -> torch.Size:
        """
        Shape of (or batch_size) of a TensorDict.
        The shape of a tensordict corresponds to the common N first dimensions of the tensors it contains, where N is
        an arbitrary number. The TensorDict shape is controlled by the user upon initialization (i.e. it is not
        inferred from the tensor shapes) and it should not be changed dynamically.

        Returns: a torch.Size object describing the TensorDict batch size.

        """
        raise NotImplementedError

    def size(self, dim: Optional[int] = None) -> torch.Size:
        if dim is None:
            return self.batch_size
        else:
            return self.batch_size[dim]

    @property
    def batch_dims(self) -> int:
        """
        Length of the tensordict batch size.

        Returns: int describing the number of dimensions of the tensordict.

        """
        return len(self.batch_size)

    def ndimension(self) -> int:
        return self.batch_dims

    def dim(self) -> int:
        return self.batch_dims

    @property
    def device(self) -> torch.device:
        """
        Device of a TensorDict. All tensors of a tensordict must live on the same device.

        Returns: torch.device object indicating the device where the tensors are placed.

        """
        raise NotImplementedError

    def is_shared(self, no_check=True) -> bool:
        """
        Checks if tensordict is in shared memory.

        This is always True for CUDA tensordicts, except when stored as MemmapTensors.

        Args:
            no_check (bool, optional): checks if all tensors are in shared memory or not

        """
        if not no_check:
            raise RuntimeError(
                f"no_check=False is not compatible with TensorDict of type {self.__class__.__name__}."
            )
        return all([item.is_shared() for key, item in self.items_meta()])

    def is_memmap(self) -> bool:
        """
        Checks if tensordict is stored with MemmapTensors.

        """

        return all([item.is_memmap() for key, item in self.items_meta()])

    def numel(self) -> int:
        """
        Total number of elements in the batch.

        """
        return max(1, math.prod(self.batch_size))

    def _check_batch_size(self) -> None:
        bs = [value.shape[: self.batch_dims] for key, value in self.items_meta()]
        if len(bs):
            if bs[0] != self.batch_size:
                raise RuntimeError(
                    "batch_size provided during initialization violates batch size of registered tensors, "
                    f"got self._batch_size={self.batch_size} and tensor.shape[:batch_dim]={bs[0]}"
                )
        if len(bs) > 1:
            for _bs in bs[1:]:
                if _bs != bs[0]:
                    raise RuntimeError(
                        f"batch_size are incongruent, got {_bs} and {bs[0]} -- "
                        f"expected {self.batch_size}"
                    )

    def _check_is_shared(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def _check_device(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def set(
        self, key: str, item: COMPATIBLE_TYPES, inplace=True, **kwargs
    ) -> _TensorDict:
        """
        Sets a new key-value pair.

        Args:
            key (str): name of the value
            item (torch.Tensor): value to be stored in the tensordict
            inplace (bool): if True and if a key matches an existing key in the tensordict, then the update will occur
                in-place for that key-value pair.

        Returns: self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def set_(self, key: str, item: COMPATIBLE_TYPES) -> _TensorDict:
        """
        Sets a value to an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            item (torch.Tensor): value to be stored in the tensordict

        Returns: self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def _default_get(
        self, key: str, default: Union[None, str, torch.Tensor] = "_no_default_"
    ) -> COMPATIBLE_TYPES:
        if not isinstance(default, str):
            return default
        if default == "_no_default_":
            raise KeyError(
                f"key {key} not found in {self.__class__.__name__} with keys {sorted(list(self.keys()))}"
            )
        else:
            raise ValueError(
                f"default should be None or a torch.Tensor instance, got {default}"
            )

    def get(
        self, key: str, default: Union[None, str, torch.Tensor] = "_no_default_"
    ) -> COMPATIBLE_TYPES:
        """
        Gets the value stored with the input key.

        Args:
            key (str): key to be queried.
            default: default value if the key is not found in the tensordict.

        Returns:

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def _get_meta(self, key, **kwargs) -> MetaTensor:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def apply_(self, fn: Callable) -> _TensorDict:
        """
        Applies a callable to all values stored in the tensordict and re-writes them in-place.
        Args:
            fn (Callable): function to be applied to the tensors in the tensordict.

        Returns: self

        """
        for key, item in self.items():
            item_trsf = fn(item)
            if item_trsf is not None:
                self.set(key, item_trsf, inplace=True)
        return self

    def apply(
        self, fn: Callable, batch_size: Optional[Iterable[int]] = None
    ) -> _TensorDict:
        """
        Applies a callable to all values stored in the tensordict and sets them in a new tensordict.
        Args:
            fn (Callable): function to be applied to the tensors in the tensordict.

        Returns: a new tensordict with transformed tensors.

        """
        if batch_size is None:
            td = TensorDict({}, batch_size=self.batch_size)
        else:
            td = TensorDict({}, batch_size=torch.Size(batch_size))
        for key, item in self.items():
            item_trsf = fn(item)
            td.set(key, item_trsf)
        return td

    def update(
        self,
        input_dict_or_td: Union[dict, _TensorDict],
        clone: bool = False,
        inplace: bool = True,
        **kwargs,
    ) -> _TensorDict:
        """
        Updates the TensorDict with values from either a dictionary or another TensorDict.

        Args:
            input_dict_or_td (_TensorDict or dict): Does not keyword arguments (unlike `dict.update()`).
            clone (bool): whether the tensors in the input (tensor)dict should be cloned before being set.
            inplace (bool): if True and if a key matches an existing key in the tensordict, then the update will occur
                in-place for that key-value pair.
            **kwargs: keyword arguments for the `TensorDict.set` method

        Returns: self

        """
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set(key, value, inplace=inplace, **kwargs)
        return self

    def update_(
        self, input_dict_or_td: Union[dict, _TensorDict], clone: bool = False
    ) -> _TensorDict:
        """
        Updates the TensorDict in-place with values from either a dictionary or another TensorDict. Unlike
        TensorDict.update, this function will throw an error if the key is unknown to the TensorDict

        Args:
            input_dict_or_td (_TensorDict or dict): Does not keyword arguments (unlike `dict.update()`).
            clone (bool): whether the tensors in the input (tensor)dict should be cloned before being set.

        Returns: self

        """
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_(key, value)
        return self

    def update_at_(
        self,
        input_dict_or_td: Union[dict, _TensorDict],
        idx: INDEX_TYPING,
        clone: bool = False,
    ) -> _TensorDict:
        """
        Updates the TensorDict in-place at the specified index with values from either a dictionary or another
        TensorDict. Unlike TensorDict.update, this function will throw an error if the key is unknown to the TensorDict.

        Args:
            input_dict_or_td (_TensorDict or dict): Does not keyword arguments (unlike `dict.update()`).
            idx (int, torch.Tensor, iterable, slice): index of the tensordict where the update should occur.
            clone (bool): whether the tensors in the input (tensor)dict should be cloned before being set.

        Returns: self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4, 5), 'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td.update_at_(
            >>>      TensorDict(source={'a': torch.ones(1, 4, 5), 'b': torch.ones(1, 4, 10)}, batch_size=[1, 4]),
            >>>      slice(1, 2))

        """

        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_at_(
                key,
                value,
                idx,
            )
        return self

    def _process_tensor(
        self,
        tensor: torch.Tensor,
        check_device: bool = True,
        check_tensor_shape: bool = True,
    ) -> torch.Tensor:
        # TODO: move to _TensorDict?
        if not isinstance(tensor, _accepted_classes):
            tensor = torch.tensor(tensor, device=self.device)
        if check_device and self.device and tensor.device is not self.device:
            tensor = tensor.to(self.device)
        if check_tensor_shape and tensor.shape[: self.batch_dims] != self.batch_size:
            raise RuntimeError(
                f"batch dimension mismatch, got self.batch_size={self.batch_size} "
                f"and tensor.shape[:self.batch_dims]={tensor.shape[: self.batch_dims]}"
            )
        if tensor.ndimension() == 0:
            tensor = tensor.view(1)
        return tensor

    def pin_memory(self) -> _TensorDict:
        """
        Calls pin_memory() on the stored tensors.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def is_pinned(self) -> bool:
        """
        CHecks if tensors are pinned.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def items(self) -> Generator:
        """
        Returns a generator of key-value pairs for the tensordict.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def items_meta(self) -> Generator:
        """
        Returns a generator of key-value pairs for the tensordict, where the values are MetaTensor instances
        corresponding to the stored tensors.

        """

        raise NotImplementedError(f"{self.__class__.__name__}")

    def keys(self) -> KeysView:
        """
        Returns a generator of tensordict keys.

        """

        raise NotImplementedError(f"{self.__class__.__name__}")

    def expand(self, *shape: Iterable) -> _TensorDict:
        """
        Expands each tensors of the tensordict according to
            tensor.expand(*shape, *tensor.shape)

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4, 5), 'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td_expand = td.expand(10)
            >>> assert td_expand.shape == torch.Size([10, 3, 4])
            >>> assert td_expand.get("a").shape == torch.Size([10, 3, 4, 5])
        """

        return TensorDict(
            source={
                key: value.expand(*shape, *value.shape) for key, value in self.items()
            },
            batch_size=[*shape, *self.batch_size],
        )

    def __ne__(self, other: _TensorDict) -> _TensorDict:
        """
        XOR operation over two tensordicts, for evey key. The two tensordicts must have the same key set.

        Returns: a new TensorDict instance with all tensors are boolean tensors of the same shape as the original
        tensors.

        """

        if not isinstance(other, _TensorDict):
            raise TypeError(
                f"TensorDict comparision requires both objects to be _TensorDict subclass, got {type(other)}"
            )
        keys1 = set(self.keys())
        keys2 = set(other.keys())
        if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
            raise KeyError(
                f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
            )
        d = dict()
        for (key, item1) in self.items():
            d[key] = item1 != other.get(key)
        return TensorDict(batch_size=self.batch_size, source=d)

    def __eq__(self, other: _TensorDict) -> _TensorDict:
        """
        Compares two tensordicts against each other, for evey key. The two tensordicts must have the same key set.

        Returns: a new TensorDict instance with all tensors are boolean tensors of the same shape as the original
        tensors.

        """

        if not isinstance(other, _TensorDict):
            raise TypeError(
                f"TensorDict comparision requires both objects to be _TensorDict subclass, got {type(other)}"
            )
        keys1 = set(self.keys())
        keys2 = set(other.keys())
        if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
            raise KeyError(
                f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
            )
        d = dict()
        for (key, item1) in self.items():
            d[key] = item1 == other.get(key)
        return TensorDict(batch_size=self.batch_size, source=d)

    def del_(self, key: str) -> None:
        """
        Deletes a key of the tensordict.

        Args:
            key (str): key to be deleted

        Returns: self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def select(self, *keys: str) -> _TensorDict:
        """
        Selects the keys of the tensordict and returns an new tensordict with only the selected keys. The values are
        not copied: in-place modifications a tensor of either of the original or new tensordict will result in a change
        in both tensordicts.

        Args:
            *keys (str): keys to select

        Returns: A new tensordict with the selected keys only.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> _TensorDict:
        """
        Sets the values in-place at the index indicated by `idx`.

        Args:
            key (str): key to be modified.
            value (torch.Tensor): value to be set at the index `idx`
            idx: index where to write the values.

        Returns: self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def copy_(self, tensor_dict: _TensorDict) -> _TensorDict:
        """
        See `_TensorDict.update_`.

        """
        return self.update_(tensor_dict)

    def copy_at_(self, tensor_dict: _TensorDict, idx: INDEX_TYPING) -> _TensorDict:
        """
        See `_TensorDict.update_at_`.

        """
        return self.update_at_(tensor_dict, idx)

    def get_at(
        self, key: str, idx: INDEX_TYPING, default: COMPATIBLE_TYPES = None
    ) -> COMPATIBLE_TYPES:
        """
        Get the value of a tensordict from the key `key` at the index `idx`.

        Args:
            key (str): key to be retrieved.
            idx (int, slice, torch.Tensor, iterable): index of the tensor.
            default (torch.Tensor): default value to return if the key is not present in the tensordict.

        Returns: indexed tensor.

        """
        try:
            return self.get(key)[idx]
        except KeyError:
            if default is not None:
                return default
            raise KeyError(
                f"key {key} not found in {self.__class__.__name__} with keys {sorted(list(self.keys()))}"
            )

    def share_memory_(self) -> _TensorDict:
        """
        Places all the tensors in shared memory.

        Returns: self.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def memmap_(self) -> _TensorDict:
        """
        Writes all tensors onto a MemmapTensor.

        Returns: self.

        """

        raise NotImplementedError(f"{self.__class__.__name__}")

    def detach_(self) -> _TensorDict:
        """
        Detach the tensors in the tensordict in-place.

        Returns: self.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def detach(self) -> _TensorDict:
        """
        Detach the tensors in the tensordict.

        Returns: a new tensordict with no tensor requiring gradient.

        """

        return self.clone().detach_()

    def to_tensordict(self):
        """
        Returns a regular TensorDict instance from the _TensorDict.

        Returns: a new TensorDict object containing the same values.

        """
        return self.to(TensorDict)

    def zero_(self) -> _TensorDict:
        """
        Zeros all tensors in the tensordict in-place.

        """
        for key in self.keys():
            self.fill_(key, 0)
        return self

    def unbind(self, dim: int) -> Tuple[_TensorDict, ...]:
        """
        Returns a tuple of indexed tensordicts unbound along the indicated dimension. Resulting tensordicts will share
        the storage of the initial tensordict.

        """
        idx = [
            (tuple(slice(None) for _ in range(dim)) + (i,))
            for i in range(self.shape[dim])
        ]
        return tuple(self[_idx] for _idx in idx)

    def chunk(self, chunks: int, dim: int = 0) -> Tuple[_TensorDict, ...]:
        """
        Attempts to split a tendordict into the specified number of chunks. Each chunk is a view of the input
        tensordict.

        Args:
            chunks (int): number of chunks to return
            dim (int): dimension along which to split the tensordict

        """
        if chunks < 1:
            raise ValueError(
                f"chunks must be a strictly positive integer, got {chunks}."
            )
        indices = []
        _idx_start = 0
        if chunks > 1:
            interval = _idx_end = self.batch_size[dim] // chunks
        else:
            interval = _idx_end = self.batch_size[dim]
        for c in range(chunks):
            indices.append(slice(_idx_start, _idx_end))
            _idx_start = _idx_end
            if c < chunks - 2:
                _idx_end = _idx_end + interval
            else:
                _idx_end = self.batch_size[dim]
        if dim < 0:
            dim = len(self.batch_size) + dim
        return tuple(self[(*[slice(None) for _ in range(dim)], idx)] for idx in indices)

    def clone(self, recursive: bool = True) -> _TensorDict:
        """
        Clones a _TensorDict subclass instance onto a new TensorDict.

        Args:
            recursive (bool): if True, each tensor contained in the TensorDict will be copied too.
        """
        return TensorDict(
            source={
                key: value.clone() if recursive else value
                for key, value in self.items()
            },
            batch_size=self.batch_size,
        )

    def __torch_function__(
        self, func: Callable, types, args: Tuple = (), kwargs: Optional[dict] = None
    ) -> Callable:
        if kwargs is None:
            kwargs = {}
        if func not in TD_HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, _TensorDict)) for t in types
        ):
            return NotImplemented
        return TD_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def to(self, dest: Union[DEVICE_TYPING, Type], **kwargs: dict) -> _TensorDict:
        """
        Maps a _TensorDict subclass either on a new device or to another _TensorDict subclass (if permitted).
        Casting tensors to a new dtype is not allowed, as tensordicts are not bound to contain a single tensor dtype.

        Args:
            dest (device or _TensorDict subclass): destination of the tensordict.

        Returns: a new tensordict. If device indicated by dest differs from the tensordict device, this is a no-op.

        """
        raise NotImplementedError

    def cpu(self) -> _TensorDict:
        """
        Casts a tensordict to cpu (if not already on cpu).

        """
        return self.to("cpu")

    def cuda(self, device: int = 0) -> _TensorDict:
        """
        Casts a tensordict to a cuda device (if not already on it).

        """
        return self.to(f"cuda:{device}")

    def masked_fill_(
        self, mask: torch.Tensor, value: Union[Number, bool]
    ) -> _TensorDict:
        """
        Fills the values corresponding to the mask with the desired value.

        Args:
            mask (boolean torch.Tensor): mask of values to be filled. Shape must match tensordict shape.
            value: value to used to fill the tensors.

        Returns: self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)}, batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td.masked_fill_(mask, 1.0)
            >>> td.get("a")

            result:
                tensor([[1., 1., 1., 1.],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.]])
        """
        raise NotImplementedError

    def masked_select(self, mask: torch.Tensor) -> _TensorDict:
        """
        Masks all tensors of the TensorDict and return a new TensorDict instance with similar keys pointing to masked
        values.

        Args:
            mask (torch.Tensor): boolean mask to be used for the tensors. Shape must match the TensorDict batch_size.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)}, batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td_mask = td.masked_select(mask)
            >>> td_mask.get("a")

            result:
                tensor([[0., 0., 0., 0.]])

        """
        d = dict()
        for key, value in self.items():
            mask_expand = mask.squeeze(-1)
            value_select = value[mask_expand]
            d[key] = value_select
        return TensorDict(
            device=self.device, source=d, batch_size=torch.Size([mask.sum()])
        )

    def is_contiguous(self) -> bool:
        """

        Returns: boolean indicating if all the tensors are contiguous.

        """
        raise NotImplementedError

    def contiguous(self) -> _TensorDict:
        """

        Returns: a new tensordict of the same type with contiguous values (or self if values are already contiguous).

        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """

        Returns: dictionary with key-value pairs matching those of the tensordict.

        """
        return {key: value for key, value in self.items()}

    def unsqueeze(self, dim: int) -> _TensorDict:
        """
        Unsqueeze all tensors for a dimension comprised in between -td.batch_dims and td.batch_dims and returns them
        in a new tensordict.

        Args:
            dim (int): dimension along which to unsqueeze

        """
        if dim < 0:
            dim = self.batch_dims + dim + 1

        if (dim > self.batch_dims) or (dim < 0):
            raise RuntimeError(
                f"unsqueezing is allowed for dims comprised between -td.batch_dims and td.batch_dims only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        return UnsqueezedTensorDict(
            source=self,
            custom_op="unsqueeze",
            inv_op="squeeze",
            custom_op_kwargs={"dim": dim},
            inv_op_kwargs={"dim": dim},
        )

    def squeeze(self, dim: int) -> _TensorDict:
        """
        Squeezes all tensors for a dimension comprised in between -td.batch_dims+1 and td.batch_dims-1 and returns them
        in a new tensordict.

        Args:
            dim (int): dimension along which to squeeze

        """
        if dim < 0:
            dim = self.batch_dims + dim

        if self.batch_dims and (dim >= self.batch_dims or dim < 0):
            raise RuntimeError(
                f"squeezing is allowed for dims comprised between 0 and td.batch_dims only. Got dim={dim}"
                f" and batch_size={self.batch_size}."
            )

        if dim >= self.batch_dims or self.batch_size[dim] != 1:
            return self
        return SqueezedTensorDict(
            source=self,
            custom_op="squeeze",
            inv_op="unsqueeze",
            custom_op_kwargs={"dim": dim},
            inv_op_kwargs={"dim": dim},
        )

    def reshape(
        self, *shape: int, size: Optional[Union[List, Tuple, torch.Size]] = None
    ) -> TensorDict:
        """
        Returns a contiguous, reshaped tensor of the desired shape.

        Args:
            *shape (int): new shape of the resulting tensordict.
            size: iterable

        Returns: A TensorDict with reshaped keys

        """
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        d = {}
        for key, item in self.items():
            d[key] = item.reshape(*shape, *item.shape[self.ndimension() :])
        if len(d):
            batch_size = d[key].shape[: len(shape)]
        else:
            if any(not isinstance(i, int) or i < 0 for i in shape):
                raise RuntimeError(
                    "Implicit reshaping is not permitted with empty tensordicts"
                )
            batch_size = shape
        return TensorDict(d, batch_size)

    def view(
        self, *shape: int, size: Optional[Union[List, Tuple, torch.Size]] = None
    ) -> _TensorDict:
        """
        Returns a tensordict with views of the tensors according to a new shape, compatible with the tensordict
        batch_size.

        Args:
            *shape (int): new shape of the resulting tensordict.
            size: iterable

        Returns: a new tensordict with the desired batch_size.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3,4,5), 'b': torch.zeros(3,4,10,1)}, batch_size=torch.Size([3, 4]))
            >>> td_view = td.view(12)
            >>> print(td_view.get("a").shape)  # torch.Size([12, 5])
            >>> print(td_view.get("b").shape)  # torch.Size([12, 10, 1])
            >>> td_view = td.view(-1, 4, 3)
            >>> print(td_view.get("a").shape)  # torch.Size([1, 4, 3, 5])
            >>> print(td_view.get("b").shape)  # torch.Size([1, 4, 3, 10, 1])

        """
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        return ViewedTensorDict(
            source=self,
            custom_op="view",
            inv_op="view",
            custom_op_kwargs={"size": shape},
            inv_op_kwargs={"size": self.batch_size},
        )

    def __repr__(self) -> str:
        fields = _td_fields(self)
        field_str = indent(f"fields={{{fields}}}", 4 * " ")
        batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
        device_str = indent(f"device={self.device}", 4 * " ")
        is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
        string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
        return f"{type(self).__name__}(\n{string})"

    def all(self, dim: int = None) -> Union[bool, _TensorDict]:
        """
        Checks if all values are True/non-null in the tensordict.

        Args:
            dim (int, optional): if None, returns a boolean indicating whether all tensors return tensor.all() == True
                If integer, all is called upon the dimension specified if and only if this dimension is compatible with
                the tensordict shape.

        """
        if dim is not None and (dim >= self.batch_dims or dim <= -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than -tensordict.batch_dims and smaller than tensordict.batchdims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.batch_dims + dim
            return TensorDict(
                source={key: value.all(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
            )
        return all([value.all() for key, value in self.items()])

    def any(self, dim: int = None) -> Union[bool, _TensorDict]:
        """
        Checks if any value is True/non-null in the tensordict.

        Args:
            dim (int, optional): if None, returns a boolean indicating whether all tensors return tensor.any() == True
                If integer, all is called upon the dimension specified if and only if this dimension is compatible with
                the tensordict shape.

        """
        if dim is not None and (dim >= self.batch_dims or dim <= -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than -tensordict.batch_dims and smaller than tensordict.batchdims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.batch_dims + dim
            return TensorDict(
                source={key: value.any(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
            )
        return any([value.any() for key, value in self.items()])

    def get_sub_tensor_dict(self, idx: INDEX_TYPING) -> _TensorDict:
        """
        Returns a SubTensorDict with the desired index.
        """
        sub_td = SubTensorDict(
            source=self,
            idx=idx,
        )
        return sub_td

    def __iter__(self) -> Generator:
        if not self.batch_dims:
            raise StopIteration
        length = self.batch_size[0]
        for i in range(length):
            yield self[i]

    def __len__(self) -> int:
        """

        Returns: Number of keys in _TensorDict instance.

        """
        return len(list(self.keys()))

    def __getitem__(self, idx: INDEX_TYPING) -> _TensorDict:
        """
        Indexes all tensors according to idx and returns a new tensordict where the values share the storage of the
        original tensors (even when the index is a torch.Tensor). Any in-place modification to the resulting
        tensordict will impact the parent tensordict too.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3,4,5)}, batch_size=torch.Size([3, 4]))
            >>> subtd = td[torch.zeros(1, dtype=torch.long)]
            >>> assert subtd.shape == torch.Size([1,4])
            >>> subtd.set("a", torch.ones(1,4,5))
            >>> print(td.get("a"))  # first row is full of 1
            >>> # Warning: this will not work as expected
            >>> subtd.get("a")[:] = 2.0
            >>> print(td.get("a"))  # values have not changed

        """
        if isinstance(idx, str):
            return self.get(idx)
        if isinstance(idx, Number):
            idx = (idx,)
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            return self.masked_select(idx)

        contiguous_input = (int, slice)
        return_simple_view = isinstance(idx, contiguous_input) or (
            isinstance(idx, tuple)
            and all(isinstance(_idx, contiguous_input) for _idx in idx)
        )
        if not self.batch_size:
            raise RuntimeError(
                "indexing a tensordict with td.batch_dims==0 is not permitted"
            )
        if return_simple_view and not self.is_memmap():
            return TensorDict(
                source={key: item[idx] for key, item in self.items()},
                batch_size=_getitem_batch_size(self.batch_size, idx),
            )
        # SubTensorDict keeps the same storage as TensorDict
        # in all cases not accounted for above
        return self.get_sub_tensor_dict(idx)

    def __setitem__(self, index: INDEX_TYPING, value: _TensorDict) -> None:
        indexed_bs = _getitem_batch_size(self.batch_size, index)
        if value.batch_size != indexed_bs:
            raise RuntimeError(
                f"indexed destination TensorDict batch size is {indexed_bs} "
                f"(batch_size = {self.batch_size}, index={index}), "
                f"which differs from the source batch size {value.batch_size}"
            )
        for key, item in value.items():
            self.set_at_(key, item, index)

    def rename_key(self, key1: str, key2: str, **kwargs: dict) -> _TensorDict:
        """
        Renames a key with a new string.

        Args:
            key1: key to be remaned
            key2: new name

        Returns: self

        """
        raise NotImplementedError

    def fill_(self, key: str, value: Union[Number, bool]) -> _TensorDict:
        """
        Fills a tensor pointed by the key with the a given value.

        Args:
            key (str): key to be remaned
            value (Number, bool): value to use for the filling

        Returns: self

        """

        meta_tensor = self._get_meta(key)
        shape = meta_tensor.shape
        device = meta_tensor.device
        dtype = meta_tensor.dtype
        value = torch.full(shape, value, device=device, dtype=dtype)
        self.set_(key, value)
        return self

    def empty(self) -> _TensorDict:
        """
        Returns a new, empty tensordict with the same device and batch size.

        """
        return self.select()


class TensorDict(_TensorDict):
    """
    A batched dictionary of tensors.
    TensorDict is a tensor container where all tensors are stored in a key-value pair fashion and where each element
    shares at least the following features:
        - device;
        - memory location (shared, memory-mapped array, ...);
        - batch size (i.e. n^th first dimensions).
    TensorDict instances support many regular tensor operations as long as they are dtype-independent (as a
    TensorDict instance can contain tensors of many different dtypes). Those operations include (but are not limited
    to):
         - operations on shape: when a shape operation is called (indexing, reshape, view, expand, transpose, permute,
            unsqueeze, squeeze, masking etc), the operations is done as if it was done on a tensor of the same shape as
            the batch size then expended to the right, e.g.:

            >>> td = TensorDict({'a': torch.zeros(3,4,5)}, batch_size=[3, 4])
            >>> td_unsqueeze = td.unsqueeze(-1)  # returns a TensorDict of batch size [3, 4, 1]
            >>> td_view = td.view(-1)  # returns a TensorDict of batch size [12]
            >>> a_view = td.view(-1).get("a")  # returns a tensor of batch size [12, 4]

        - casting operations: a TensorDict can be cast on a different device or another TensorDict type using

            >>> td_cpu = td.to("cpu")
            >>> td_savec = td.to(SavedTensorDict)  # TensorDict saved on disk
            >>> dictionary = td.to_dict()

            A call of the `.to()` method with a dtype will return an error.

        - Cloning, contiguous
        - Reading: `td.get(key)`, `td.get_at(key, index)`
        - Content modification: `td.set(key, value)`, `td.set_(key, value)`, `td.update(td_or_dict)`,
        `td.update_(td_or_dict)`, `td.fill_(key, value)`, `td.rename_key(old_name, new_name)`, etc.
        - Operations on multiple tensordicts: `torch.cat(tensordict_list, dim)`, `torch.stack(tensordict_list, dim)`,
        `td1 == td2` etc.

    Args:
        source (TensorDict or dictionary): a data source. If empty, the tensordict can be populated subsequently.
        batch_size (iterable of int, optional): a batch size for the tensordict. The batch size is immutable and can
            only be modified by calling operations that create a new TensorDict. Unless the source is another
            TensorDict, the batch_size argument must be provided as it won't be inferred from the data.
        device (torch.device or compatible type, optional): a device for the TensorDict. If the source is non-empty
            and the device is missing, it will be inferred from the input dictionary, assuming that all tensors are
            on the same device.

    Examples:
        >>> import torch
        >>> from torchrl.data import TensorDict
        >>> source = {'random': torch.randn(3, 4), 'zeros': torch.zeros(3, 4, 5)}
        >>> batch_size = [3]
        >>> td = TensorDict(source, batch_size)
        >>> print(td.shape)  # equivalent to td.batch_size
        torch.Size([3])
        >>> td_unqueeze = td.unsqueeze(-1)
        >>> print(td_unqueeze.get("zeros").shape)
        torch.Size([3, 1, 4, 5])
        >>> print(td_unqueeze[0].shape)
        torch.Size([1])
        >>> print(td_unqueeze.view(-1).shape)
        torch.Size([3])
        >>> print((td.clone()==td).all())
        True

    TODO: split, transpose, permute

    """

    _safe = True

    def __init__(
        self,
        source: Union[_TensorDict, dict],
        batch_size: Optional[Union[Iterable[int], torch.Size, int]] = None,
        device: Optional[DEVICE_TYPING] = None,
        _meta_source: Optional[dict] = None,
    ):
        self._tensor_dict = dict()
        self._tensor_dict_meta = OrderedDict()
        if not isinstance(source, (_TensorDict, dict)):
            raise ValueError(
                "A TensorDict source is expected to be a _TensorDict sub-type or a dictionary, "
                f"found type(source)={type(source)}."
            )
        if isinstance(
            batch_size,
            (
                Number,
                Iterable,
            ),
        ):
            if not isinstance(batch_size, torch.Size):
                if isinstance(batch_size, Number):
                    batch_size = torch.Size([batch_size])
                else:
                    batch_size = torch.Size(batch_size)
            self._batch_size = batch_size
            self._batch_dims = len(batch_size)
        elif isinstance(source, _TensorDict):
            self._batch_size = source.batch_size
        else:
            raise ValueError(
                "batch size was not specified when creating the TensorDict instance and it could not be "
                "retrieved from source."
            )

        if isinstance(device, (int, str)):
            device = torch.device(device)
        map_item_to_device = device is not None
        self._device = device
        if source is not None:
            for key, value in source.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Expected key to be a string but found {type(key)}"
                    )
                if not isinstance(value, _accepted_classes):
                    raise TypeError(
                        f"Expected value to be one of types {_accepted_classes} but got {type(value)}"
                    )
                if map_item_to_device:
                    value = value.to(device)
                _meta_val = None if _meta_source is None else _meta_source[key]
                self.set(key, value, _meta_val=_meta_val)
        self._check_batch_size()
        self._check_device()

    def _batch_dims_get(self) -> int:
        if self._safe and hasattr(self, "_batch_dims"):
            if len(self.batch_size) != self._batch_dims:
                raise RuntimeError("len(self.batch_size) and self._batch_dims mismatch")
        return len(self.batch_size)

    def _batch_dims_set(self, value: COMPATIBLE_TYPES) -> None:
        raise RuntimeError(
            f"Setting batch dims on {self.__class__.__name__} instances is not allowed."
        )

    batch_dims = property(_batch_dims_get, _batch_dims_set)

    def is_shared(self, no_check: bool = False) -> bool:
        if no_check:
            for key, item in self.items_meta():
                return item.is_shared()
        return self._check_is_shared()

    def is_memmap(self) -> bool:
        return self._check_is_memmap()

    def _device_get(self) -> DEVICE_TYPING:
        device = self._device
        if device is None and len(self):
            device = next(self.items_meta())[1].device
        if not isinstance(device, torch.device) and device is not None:
            device = torch.device(device)
        self._device = device
        return device

    def _device_set(self, value: DEVICE_TYPING) -> None:
        raise RuntimeError(
            f"Setting device on {self.__class__.__name__} instances is not allowed. "
            f"Please call {self.__class__.__name__}.to(device) instead."
        )

    device = property(_device_get, _device_set)

    def _batch_size_get(self) -> torch.Size:
        return self._batch_size

    def _batch_size_set(self, value: COMPATIBLE_TYPES) -> None:
        raise RuntimeError(
            f"Setting batch size on {self.__class__.__name__} instances is not allowed."
        )

    batch_size = property(_batch_size_get, _batch_size_set)

    # Checks
    def _check_is_shared(self) -> bool:
        share_list = [value.is_shared() for key, value in self.items_meta()]
        if any(share_list) and not all(share_list):
            shared_str = ", ".join(
                [f"{key}: {value.is_shared()}" for key, value in self.items_meta()]
            )
            raise RuntimeError(
                f"tensors must be either all shared or not, but mixed features is not allowed. "
                f"Found: {shared_str}"
            )
        return all(share_list) and len(share_list) > 0

    def _check_is_memmap(self) -> bool:
        memmap_list = [value.is_memmap() for key, value in self.items_meta()]
        if any(memmap_list) and not all(memmap_list):
            memmap_str = ", ".join(
                [f"{key}: {value.is_memmap()}" for key, value in self.items_meta()]
            )
            raise RuntimeError(
                f"tensors must be either all MemmapTensor or not, but mixed features is not allowed. "
                f"Found: {memmap_str}"
            )
        return all(memmap_list) and len(memmap_list) > 0

    def _check_device(self) -> None:
        devices = {key: value.device for key, value in self.items_meta()}
        if len(devices):
            if not (
                len(np.unique([str(device) for key, device in devices.items()])) == 1
            ):
                raise RuntimeError(
                    f"expected tensors to be on a single device, found {devices}"
                )
            device = devices[list(devices.keys())[0]]
            if torch.device(device) != self.device:
                raise RuntimeError(
                    f"expected {self.__class__.__name__}.device to be identical to tensors "
                    f"device, found {self.__class__.__name__}.device={self.device} and {device}"
                )

    def _process_tensor(
        self,
        tensor: Union[COMPATIBLE_TYPES, np.ndarray],
        check_device: bool = True,
        check_tensor_shape: bool = True,
        check_shared: bool = True,
    ) -> torch.Tensor:
        # TODO: move to _TensorDict?
        if not isinstance(tensor, _accepted_classes):
            tensor = torch.tensor(tensor, device=self.device)
        if check_device and self.device and tensor.device is not self.device:
            tensor = tensor.to(self.device)
        try:
            if check_shared:
                if self.is_shared():
                    tensor = tensor.share_memory_()
                elif self.is_memmap():
                    tensor = MemmapTensor(tensor)
                elif tensor.is_shared() and len(self):
                    tensor = tensor.clone()
        except:
            warn(
                f"check_shared for tensor {type(tensor)} with shape {tensor.shape} failed"
            )
        if check_tensor_shape and tensor.shape[: self.batch_dims] != self.batch_size:
            raise RuntimeError(
                f"batch dimension mismatch, got self.batch_size={self.batch_size} "
                f"and tensor.shape[:self.batch_dims]={tensor.shape[: self.batch_dims]}"
            )

        # minimum ndimension is 1
        if tensor.ndimension() - self.ndimension() == 0:
            tensor = tensor.unsqueeze(-1)
        return tensor

    def pin_memory(self) -> TensorDict:
        if self.device == torch.device("cpu"):
            for key, value in self.items():
                if value.dtype in (torch.half, torch.float, torch.double):
                    self.set(key, value.pin_memory(), inplace=False)
        return self

    def expand(self, *shape: int, inplace: bool = False) -> TensorDict:
        """expand every tensor with (*shape, *tensor.shape) and returns the same tensordict with new tensors with expanded shapes."""
        _batch_size = torch.Size([*shape, *self.batch_size])
        _batch_dims = len(_batch_size)
        d = {key: value.expand(*shape, *value.shape) for key, value in self.items()}
        if inplace:
            d_meta = {
                key: value.expand(*shape, *value.shape)
                for key, value in self.items_meta()
            }
            self._tensor_dict = d
            self._tensor_dict_meta = d_meta
            self._batch_size = _batch_size
            self._batch_dims = _batch_dims
        return TensorDict(source=d, batch_size=_batch_size)

    def set(
        self,
        key: str,
        value: COMPATIBLE_TYPES,
        inplace: bool = True,
        _run_checks: bool = True,
        _meta_val: Optional[MetaTensor] = None,
    ) -> TensorDict:
        """
        Sets a value in the TensorDict. If inplace=True (default), if the key already exists, set will call set_ (in place setting).
        """
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a string but found {type(key)}")

        if key in self._tensor_dict and value is self._tensor_dict[key]:
            return self

        value = self._process_tensor(
            value,
            check_tensor_shape=_run_checks,
            check_shared=_run_checks,
        )  # check_tensor_shape=_run_checks
        if key in self._tensor_dict and inplace:
            return self.set_(key, value)
        self._tensor_dict[key] = value
        self._tensor_dict_meta[key] = (
            MetaTensor(value) if _meta_val is None else _meta_val
        )
        return self

    def del_(self, key: str) -> TensorDict:
        del self._tensor_dict[key]
        del self._tensor_dict_meta[key]
        return self

    def rename_key(self, old_key: str, new_key: str, safe: bool = False) -> TensorDict:
        if not isinstance(old_key, str):
            raise TypeError(
                f"Expected old_name to be a string but found {type(old_key)}"
            )
        if not isinstance(new_key, str):
            raise TypeError(
                f"Expected new_name to be a string but found {type(new_key)}"
            )

        if safe and (new_key in self.keys()):
            raise KeyError(f"key {new_key} already present in TensorDict.")
        self.set(new_key, self.get(old_key))
        self.del_(old_key)
        return self

    def set_(self, key: str, value: COMPATIBLE_TYPES) -> TensorDict:
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a string but found {type(key)}")

        if key in self.keys():
            value = self._process_tensor(value, check_device=False, check_shared=False)
            target_shape = self._get_meta(key).shape
            if value.shape != target_shape:
                raise RuntimeError(
                    f'calling set_("{key}", tensor) with tensors of '
                    f'different shape: got tensor.shape={value.shape} and get("{key}").shape={target_shape}'
                )
            if value is not self._tensor_dict[key]:
                self._tensor_dict[key].copy_(value)
        else:
            raise AttributeError(
                f"key {key} not found in tensordict, "
                f"call td.set({key}, value) for populating tensordict with new key-value pair"
            )
        return self

    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> TensorDict:
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a string but found {type(key)}")

        # do we need this?
        # value = self._process_tensor(
        #     value, check_tensor_shape=False, check_device=False
        # )
        if key not in self.keys():
            raise KeyError(f"did not find key {key} in {self.__class__.__name__}")
        tensor_in = self._tensor_dict[key]
        if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
            warn(
                "Multiple indexing can lead to unexpected behaviours when setting items,"
                "for instance `td[idx1][idx2] = other` may not write to the desired location if idx1 is a list/tensor."
            )
            tensor_in = _sub_index(tensor_in, idx)
            tensor_in.copy_(value)
        else:
            tensor_in[idx] = value
        return self

    def get(
        self, key: str, default: Union[None, str, torch.Tensor] = "_no_default_"
    ) -> COMPATIBLE_TYPES:
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a string but found {type(key)}")

        try:
            return self._tensor_dict[key]
        except KeyError:
            return self._default_get(key, default)

    def _get_meta(self, key: str) -> MetaTensor:
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a string but found {type(key)}")

        try:
            return self._tensor_dict_meta[key]
        except KeyError:
            raise KeyError(
                f"key {key} not found in {self.__class__.__name__} with keys {sorted(list(self.keys()))}"
            )

    def share_memory_(self) -> TensorDict:
        if self.is_memmap():
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        if not len(self._tensor_dict):
            raise Exception(
                "share_memory_ must be called when the TensorDict is (partially) populated. Set a tensor first."
            )
        for key, value in self.items():
            value.share_memory_()
        for key, value in self.items_meta():
            value.share_memory_()
        return self

    def detach_(self) -> TensorDict:
        for key, value in self.items():
            value.detach_()
        return self

    def memmap_(self) -> TensorDict:
        if self.is_memmap():
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        if not len(self._tensor_dict):
            raise Exception(
                "memmap_() must be called when the TensorDict is (partially) populated. Set a tensor first."
            )
        for key, value in self.items():
            self._tensor_dict[key] = MemmapTensor(value)
        for key, value in self.items_meta():
            value.memmap_()
        return self

    def to(self, dest: Union[DEVICE_TYPING, Type], **kwargs: dict) -> _TensorDict:
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            td = dest(
                source=self,
                **kwargs,
            )
            return td
        elif isinstance(dest, (torch.device, str, int)):
            # must be device
            if not isinstance(dest, torch.device):
                dest = torch.device(dest)
            if dest == self.device:
                return self

            self_copy = TensorDict(
                source={key: value.to(dest) for key, value in self.items()},
                batch_size=self.batch_size,
            )
            if self._safe:
                # sanity check
                self_copy._check_device()
                self_copy._check_is_shared()
            return self_copy
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def masked_fill_(self, mask: torch.Tensor, val: Union[Number, bool]) -> TensorDict:
        for key, value in self.items():
            mask_expand = expand_as_right(mask, value)
            value.masked_fill_(mask_expand, val)
        return self

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def contiguous(self) -> TensorDict:
        if not self.is_contiguous():
            return self.clone()
        return self

    def select(self, *keys: str, inplace: bool = False) -> TensorDict:
        d = {key: value for (key, value) in self.items() if key in keys}
        if inplace:
            self._tensor_dict = d
            self._tensor_dict_meta = {
                key: value for (key, value) in self.items_meta() if key in keys
            }
            return self
        return TensorDict(device=self.device, batch_size=self.batch_size, source=d)

    def items(self) -> Iterator[Tuple[str, COMPATIBLE_TYPES]]:
        for k in self._tensor_dict:
            yield k, self.get(k)

    def items_meta(self) -> Iterator[Tuple[str, MetaTensor]]:
        for k in self._tensor_dict_meta:
            yield k, self._get_meta(k)

    def keys(self) -> KeysView:
        return self._tensor_dict.keys()


import functools


def implements_for_td(torch_function: Callable) -> Callable:
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        TD_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


# @implements_for_td(torch.testing.assert_allclose) TODO
def assert_allclose_td(
    actual: _TensorDict,
    expected: _TensorDict,
    rtol: Number = None,
    atol: Number = None,
    equal_nan: bool = True,
    msg: str = "",
) -> bool:
    if not isinstance(actual, _TensorDict) or not isinstance(expected, _TensorDict):
        raise TypeError("assert_allclose inputs must be of TensorDict type")
    set1 = set(actual.keys())
    set2 = set(expected.keys())
    if not (len(set1.difference(set2)) == 0 and len(set2) == len(set1)):
        raise KeyError(
            "actual and expected tensordict keys mismatch, "
            f"keys {(set1 - set2).union(set2 - set1)} appear in one but not the other."
        )
    keys = sorted(list(actual.keys()))
    for key in keys:
        input1 = actual.get(key)
        input2 = expected.get(key)
        mse = (
            (input1.to(torch.float) - input2.to(torch.float))
            .pow(2)
            .sum()
            .div(input1.numel())
            .sqrt()
            .item()
        )

        default_msg = f"key {key} does not match, got mse = {mse:4.4f}"
        if len(msg):
            msg = "\t".join([default_msg, msg])
        else:
            msg = default_msg
        torch.testing.assert_allclose(
            input1, input2, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=msg
        )
    return True


@implements_for_td(torch.unbind)
def unbind(td: _TensorDict, *args, **kwargs) -> Tuple[_TensorDict, ...]:
    return td.unbind(*args, **kwargs)


@implements_for_td(torch.clone)
def clone(td: _TensorDict, *args, **kwargs) -> _TensorDict:
    return td.clone(*args, **kwargs)


@implements_for_td(torch.cat)
def cat(
    list_of_tensor_dicts: Iterable[_TensorDict],
    dim: int = 0,
    device: DEVICE_TYPING = None,
    out: _TensorDict = None,
) -> _TensorDict:
    if not list_of_tensor_dicts:
        raise RuntimeError("list_of_tensor_dicts cannot be empty")
    if not device:
        device = list_of_tensor_dicts[0].device
    if dim < 0:
        raise RuntimeError(
            f"negative dim in torch.dim(list_of_tensor_dicts, dim=dim) not allowed, got dim={dim}"
        )

    batch_size = list(list_of_tensor_dicts[0].batch_size)
    if dim >= len(batch_size):
        raise RuntimeError(
            f"dim must be in the range 0 <= dim < len(batch_size), got dim={dim} "
            f"and batch_size={batch_size}"
        )
    batch_size[dim] = sum([td.batch_size[dim] for td in list_of_tensor_dicts])
    batch_size = torch.Size(batch_size)

    # check that all tensordict match
    keys = _check_keys(list_of_tensor_dicts, strict=True)
    if out is None:
        out_td = TensorDict({}, device=device, batch_size=batch_size)
        for key in keys:
            out_td.set(
                key, torch.cat([td.get(key) for td in list_of_tensor_dicts], dim)
            )
        return out_td
    else:
        out_td = out
        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and cat batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size={batch_size}"
            )

        for key in keys:
            out_td.set_(
                key, torch.cat([td.get(key) for td in list_of_tensor_dicts], dim)
            )
        return out_td


@implements_for_td(torch.stack)
def stack(
    list_of_tensor_dicts: Sequence[_TensorDict],
    dim: int = 0,
    out: _TensorDict = None,
    strict=False,
    contiguous=False,
) -> _TensorDict:
    if not list_of_tensor_dicts:
        raise RuntimeError("list_of_tensor_dicts cannot be empty")
    batch_size = list_of_tensor_dicts[0].batch_size
    if dim < 0:
        dim = len(batch_size) + dim + 1
    if len(list_of_tensor_dicts) > 1:
        for td in list_of_tensor_dicts[1:]:
            if td.batch_size != list_of_tensor_dicts[0].batch_size:
                raise RuntimeError(
                    "stacking tensor_dicts requires them to have congruent batch sizes, "
                    f"got td1.batch_size={td.batch_size} and "
                    f"td2.batch_size{list_of_tensor_dicts[0].batch_size}"
                )
    # check that all tensordict match
    keys = _check_keys(list_of_tensor_dicts)
    batch_size = list(batch_size)
    batch_size.insert(dim, len(list_of_tensor_dicts))
    batch_size = torch.Size(batch_size)

    if out is None:
        out_td = LazyStackedTensorDict(
            *list_of_tensor_dicts,
            stack_dim=dim,
        )
        if contiguous:
            out_td = out_td.contiguous()
        return out_td
    else:
        out_td = out
        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and stacked batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size={batch_size}"
            )
        if strict:
            out_keys = set(out_td.keys())
            in_keys = set(keys)
            if len(out_keys - in_keys) > 0:
                raise RuntimeError(
                    "The output tensordict has keys that are missing in the tensordict that has to be "
                    f"written: {out_keys - in_keys}. As per the call to `stack(..., strict=True)`, this "
                    f"is not permitted."
                )
            elif len(in_keys - out_keys) > 0:
                raise RuntimeError(
                    "The resulting tensordict has keys that are missing in its destination: "
                    f"{in_keys - out_keys}. As per the call to `stack(..., strict=True)`, this "
                    f"is not permitted."
                )

        for key in keys:
            out_td.set(
                key,
                torch.stack([td.get(key) for td in list_of_tensor_dicts], dim),
                inplace=True,
            )
    return out_td


# @implements_for_td(torch.nn.utils.rnn.pad_sequence)
def pad_sequence_td(
    list_of_tensor_dicts: Iterable[_TensorDict],
    batch_first: bool = True,
    padding_value: Number = 0.0,
    out: _TensorDict = None,
    device: Optional[DEVICE_TYPING] = None,
):
    if not list_of_tensor_dicts:
        raise RuntimeError("list_of_tensor_dicts cannot be empty")
    # check that all tensordict match
    keys = _check_keys(list_of_tensor_dicts)
    if out is None:
        out_td = TensorDict({}, [], device=device)
        for key in keys:
            out_td.set(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [td.get(key) for td in list_of_tensor_dicts],
                    batch_first=batch_first,
                    padding_value=padding_value,
                ),
            )
        return out_td
    else:
        out_td = out
        for key in keys:
            out_td.set_(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [td.get(key) for td in list_of_tensor_dicts],
                    batch_first=batch_first,
                    padding_value=padding_value,
                ),
            )
        return out_td


class SubTensorDict(_TensorDict):
    """
    A TensorDict that only sees an index of the stored tensors.

    By default, indexing a tensordict with an iterable will result in a SubTensorDict. This is done such that a
    TensorDict indexed with non-contiguous index (e.g. a Tensor) will still point to the original memory location (
    unlike regular indexing of tensors).

    Examples:
        >>> from torchrl.data import TensorDict, SubTensorDict
        >>> source = {'random': torch.randn(3, 4, 5, 6), 'zeros': torch.zeros(3, 4, 1, dtype=torch.bool)}
        >>> batch_size = torch.Size([3, 4])
        >>> td = TensorDict(source, batch_size)
        >>> td_index = td[:, 2]
        >>> print(type(td_index), td_index.shape)
        <class 'torchrl.data.tensordict.tensordict.TensorDict'> torch.Size([3])
        >>> td_index = td[:, slice(None)]
        >>> print(type(td_index), td_index.shape)
        <class 'torchrl.data.tensordict.tensordict.TensorDict'> torch.Size([3, 4])
        >>> td_index = td[:, torch.Tensor([0, 2]).to(torch.long)]
        >>> print(type(td_index), td_index.shape)
        <class 'torchrl.data.tensordict.tensordict.SubTensorDict'> torch.Size([3, 2])
        >>> td_index.fill_('zeros', 1)
        >>> print(td.get('zeros'))  # the indexed tensors are updated with Trues
        tensor([[[True],
                 [ False],
                 [True],
                 [ False]],
                [[True],
                 [False],
                 [True],
                 [False]],
                [[True],
                 [False],
                 [True],
                 [False]]])

    """

    _safe = False

    def __init__(
        self,
        source: _TensorDict,
        idx: INDEX_TYPING,
        batch_size: Optional[Iterable[int]] = None,
    ):
        if not isinstance(source, _TensorDict):
            raise TypeError(
                f"Expected source to be a subclass of _TensorDict, got {type(source)}"
            )
        self._source = source
        if not isinstance(idx, (tuple, list)):
            idx = (idx,)
        else:
            idx = tuple(idx)
        self.idx = idx
        self._batch_size = _getitem_batch_size(self._source.batch_size, self.idx)
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @property
    def device(self) -> DEVICE_TYPING:
        return self._source.device

    def _preallocate(self, key: str, value: COMPATIBLE_TYPES) -> _TensorDict:
        return self._source.set(key, value)

    def set(
        self,
        key: str,
        tensor: COMPATIBLE_TYPES,
        inplace: bool = True,
        _run_checks: bool = True,
    ) -> _TensorDict:
        if inplace and key in self.keys():
            return self.set_(key, tensor)

        tensor = self._process_tensor(
            tensor, check_device=False, check_tensor_shape=False
        )
        parent = self.get_parent_tensor_dict()
        tensor_expand = torch.zeros(
            *parent.batch_size,
            *tensor.shape[self.batch_dims :],
            dtype=tensor.dtype,
            device=self.device,
        )

        if self.is_shared():
            tensor_expand.share_memory_()
        elif self.is_memmap():
            tensor_expand = MemmapTensor(tensor_expand)

        parent.set(key, tensor_expand, _run_checks=_run_checks)
        self.set_(key, tensor)
        return self

    def keys(self) -> KeysView:
        return self._source.keys()

    def set_(self, key: str, tensor: COMPATIBLE_TYPES) -> SubTensorDict:
        if key not in self.keys():
            raise KeyError(f"key {key} not found in {self.keys()}")
        if tensor.shape[: self.batch_dims] != self.batch_size:
            raise RuntimeError(
                f"tensor.shape={tensor.shape[:self.batch_dims]} and "
                f"self.batch_size={self.batch_size} mismatch"
            )
        self._source.set_at_(key, tensor, self.idx)
        return self

    def to(self, dest: Union[DEVICE_TYPING, Type], **kwargs: dict) -> _TensorDict:
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            return dest(
                source=self.clone(),
            )
        elif isinstance(dest, (torch.device, str, int)):
            if not isinstance(dest, torch.device):
                dest = torch.device(dest)
            if dest == self.device:
                return self
            td = self.clone()
            # must be device
            return td.to(dest, **kwargs)
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def get(self, key: str, default: Optional[torch.Tensor] = None) -> COMPATIBLE_TYPES:
        return self._source.get_at(key, self.idx)

    def _get_meta(self, key: str) -> MetaTensor:
        return self._source._get_meta(key)[self.idx]

    def set_at_(
        self,
        key: str,
        value: COMPATIBLE_TYPES,
        idx: INDEX_TYPING,
        discard_idx_attr: bool = False,
    ) -> SubTensorDict:
        if not isinstance(idx, tuple):
            idx = (idx,)
        if discard_idx_attr:
            self._source.set_at_(key, value, idx)
        else:
            tensor = self._source.get_at(key, self.idx)
            tensor[idx] = value
            self._source.set_at_(key, tensor, self.idx)
            # self._source.set_at_(key, value, (self.idx, idx))
        return self

    def get_at(
        self, key: str, idx: INDEX_TYPING, discard_idx_attr: bool = False
    ) -> COMPATIBLE_TYPES:
        if not isinstance(idx, tuple):
            idx = (idx,)
        if discard_idx_attr:
            return self._source.get_at(key, idx)
        else:
            return self._source.get_at(key, self.idx)[idx]

    def update_(
        self, input_dict: Union[dict, _TensorDict], clone: bool = False
    ) -> SubTensorDict:
        return self.update_at_(
            input_dict, idx=self.idx, discard_idx_attr=True, clone=clone
        )

    def update_at_(
        self,
        input_dict: Union[dict, _TensorDict],
        idx: INDEX_TYPING,
        discard_idx_attr: bool = False,
        clone: bool = False,
    ) -> SubTensorDict:
        for key, value in input_dict.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_at_(
                key,
                value,
                idx,
                discard_idx_attr=discard_idx_attr,
            )
        return self

    def get_parent_tensor_dict(self) -> _TensorDict:
        if not isinstance(self._source, _TensorDict):
            raise TypeError(
                f"SubTensorDict was initialized with a source of type {self._source.__class__.__name__}, "
                "parent tensordict not accessible"
            )
        return self._source

    def del_(self, key: str) -> _TensorDict:
        self._source = self._source.del_(key)
        return self

    def clone(self, recursive: bool = True) -> SubTensorDict:
        if not recursive:
            return copy(self)
        return SubTensorDict(
            source=self._source,
            idx=self.idx,
        )

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def contiguous(self) -> TensorDict:
        if self.is_contiguous():
            return self
        return TensorDict(
            batch_size=self.batch_size,
            source={key: value for key, value in self.items()},
        )

    def items(self) -> Iterator[Tuple[str, COMPATIBLE_TYPES]]:
        for k in self.keys():
            yield k, self.get(k)

    def items_meta(self) -> Iterator[Tuple[str, MetaTensor]]:
        for key, value in self._source.items_meta():
            yield key, value[self.idx]

    def select(self, *keys: str, inplace: bool = False) -> _TensorDict:
        if inplace:
            self._source = self._source.select(*keys)
            return self
        return self._source.select(*keys)[self.idx]

    def expand(self, *shape: int, inplace: bool = False) -> _TensorDict:
        new_source = self._source.expand(*shape)
        idx = tuple(slice(None) for _ in shape) + tuple(self.idx)
        if inplace:
            self._source = new_source
            self.idx = idx
        return new_source[idx]

    def is_shared(self, no_check: bool = True) -> bool:
        return self._source.is_shared(no_check=no_check)

    def is_memmap(self) -> bool:
        return self._source.is_memmap()

    def rename_key(self, key1: str, key2: str, **kwargs) -> SubTensorDict:
        self._source.rename_key(key1, key2, **kwargs)
        return self

    def pin_memory(self) -> SubTensorDict:
        self._source.pin_memory()
        return self


def merge_tensor_dicts(*tensor_dicts: _TensorDict) -> TensorDict:
    if len(tensor_dicts) < 2:
        raise RuntimeError(
            f"at least 2 tensor_dicts must be provided, got {len(tensor_dicts)}"
        )
    d = tensor_dicts[0].to_dict()
    for td in tensor_dicts[1:]:
        d.update(td.to_dict())
    return TensorDict({}, [], device=td.device).update(d)


class LazyStackedTensorDict(_TensorDict):
    """
    A Lazy stack of TensorDicts.
    When stacking TensorDicts together, the default behaviour is to put them in a stack that is not instantiated.
    This allows to seamlessly work with stacks of tensordicts with operations that will affect the original
    tensordicts.

    Args:
         *tensor_dicts (TensorDict instances): a list of tensordict with same batch size.
         stack_dim (int): a dimension (between `-td.ndimension()` and `td.ndimension()-1` along which the stack should
             be performed.

    Examples:
        >>> from torchrl.data import TensorDict
        >>> import torch
        >>> tds = [TensorDict({'a': torch.randn(3, 4)}, batch_size=[3]) for _ in range(10)]
        >>> td_stack = torch.stack(tds, -1)
        >>> print(td_stack.shape)
        torch.Size([3, 10])
        >>> print(td_stack.get("a").shape)
        torch.Size([3, 10, 4])
        >>> print(td_stack[:, 0] is tds[0])
        True
    """

    _safe = False

    def __init__(
        self,
        *tensor_dicts: List[_TensorDict],
        stack_dim: int = 0,
        batch_size: Optional[Iterable[int]] = None,  # TODO: remove
    ):
        # sanity check
        N = len(tensor_dicts)
        if not isinstance(tensor_dicts[0], _TensorDict):
            raise TypeError(
                f"Expected input to be _TensorDict instance"
                f" but got {type(tensor_dicts[0])} instead."
            )
        if stack_dim < 0:
            raise RuntimeError(
                f"stack_dim must be non negative, got stack_dim={stack_dim}"
            )
        if not N:
            raise RuntimeError(
                "at least one tensordict must be provided to StackedTensorDict to be instantiated"
            )
        _batch_size = tensor_dicts[0].batch_size
        device = tensor_dicts[0].device

        for i, td in enumerate(tensor_dicts[1:]):
            if not isinstance(td, _TensorDict):
                raise TypeError(
                    f"Expected input to be _TensorDict instance"
                    f" but got {type(tensor_dicts[0])} instead."
                )
            _bs = td.batch_size
            _device = td.device
            if device != _device:
                raise RuntimeError(f"devices differ, got {device} and {_device}")
            _keys = set(td.keys())
            if _bs != _batch_size:
                raise RuntimeError(
                    f"batch sizes in tensor_dicts differs, StackedTensorDict cannot be created. Got "
                    f"td[0].batch_size={_batch_size} and td[i].batch_size={_bs} "
                )
        self.tensor_dicts = list(tensor_dicts)
        self.stack_dim = stack_dim
        self._batch_size = self._compute_batch_size(_batch_size, stack_dim, N)
        self._batch_dims = len(self._batch_size)
        self._update_valid_keys()
        self._meta_dict = dict()
        self._meta_dict.update({k: value for k, value in self.items_meta()})
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    @property
    def device(self) -> DEVICE_TYPING:
        device_set = {td.device for td in self.tensor_dicts}
        if len(device_set) != 1:
            raise RuntimeError(
                f"found multiple devices in {self.__class__.__name__}: {device_set}"
            )
        return self.tensor_dicts[0].device

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    def is_shared(self, no_check: bool = True) -> bool:
        return all(td.is_shared(no_check=no_check) for td in self.tensor_dicts)

    def is_memmap(self) -> bool:
        return all(td.is_memmap() for td in self.tensor_dicts)

    def get_valid_keys(self) -> Iterable[str]:
        self._update_valid_keys()
        return self._valid_keys

    def set_valid_keys(self, keys: Iterable[str]) -> None:
        raise RuntimeError(
            "setting valid keys is not permitted. valid keys are defined as the intersection of all the "
            "key sets from the TensorDicts in a stack and cannot be defined explicitely."
        )

    valid_keys = property(get_valid_keys, set_valid_keys)

    @staticmethod
    def _compute_batch_size(
        batch_size: torch.Size, stack_dim: int, N: int
    ) -> torch.Size:
        # s_list = list(reversed(list(batch_size)))
        # s = torch.Size([s_list.pop() if i != stack_dim else N for i in range(len(batch_size) + 1)])
        s = list(batch_size)
        s.insert(stack_dim, N)
        return torch.Size(s)

    def set(
        self, key: str, tensor: COMPATIBLE_TYPES, **kwargs
    ) -> LazyStackedTensorDict:
        if self.batch_size != tensor.shape[: self.batch_dims]:
            raise RuntimeError(
                "Setting tensor to tensordict failed because the shapes mismatch:"
                f"got tensor.shape = {tensor.shape} and tensordict.batch_size={self.batch_size}"
            )
        tensor = self._process_tensor(
            tensor, check_device=False, check_tensor_shape=False
        )
        tensor = tensor.unbind(self.stack_dim)
        for td, _item in zip(self.tensor_dicts, tensor):
            td.set(key, _item, **kwargs)
        return self

    def set_(
        self, key: str, tensor: COMPATIBLE_TYPES, **kwargs
    ) -> LazyStackedTensorDict:
        if self.batch_size != tensor.shape[: self.batch_dims]:
            raise RuntimeError(
                "Setting tensor to tensordict failed because the shapes mismatch:"
                f"got tensor.shape = {tensor.shape} and tensordict.batch_size={self.batch_size}"
            )
        if key not in self.valid_keys:
            raise KeyError(
                "setting a value in-place on a stack of TensorDict is only permitted if all "
                "members of the stack have this key in their register."
            )
        tensor = self._process_tensor(
            tensor, check_device=False, check_tensor_shape=False
        )
        tensor = tensor.unbind(self.stack_dim)
        for td, _item in zip(self.tensor_dicts, tensor):
            td.set_(key, _item, **kwargs)
        return self

    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> LazyStackedTensorDict:
        sub_td = self[idx]
        sub_td.set_(key, value)
        return self

    def get(
        self,
        key: str,
        default: Union[None, str, torch.Tensor] = "_no_default_",
        **kwargs,
    ) -> COMPATIBLE_TYPES:
        if not (key in self.valid_keys):
            return self._default_get(key, default)
        tensors = [td.get(key, default=default, **kwargs) for td in self.tensor_dicts]
        shapes = set(tensor.shape for tensor in tensors)
        if len(shapes) != 1:
            raise RuntimeError(
                f"found more than one unique shape in the tensors to be stacked ({shapes}). This is likely due to "
                "a modification of one of the stacked TensorDicts, where a key has been updated/created with "
                "an uncompatible shape."
            )
        return torch.stack(tensors, self.stack_dim)

    def _get_meta(self, key: str, **kwargs) -> MetaTensor:
        if key in self._meta_dict:
            return self._meta_dict[key]
        if key not in self.valid_keys:
            raise KeyError(f"key {key} not found in {list(self._valid_keys)}")
        return torch.stack(
            [td._get_meta(key, **kwargs) for td in self.tensor_dicts], self.stack_dim
        )

    def is_contiguous(self) -> bool:
        return False

    def contiguous(self) -> TensorDict:
        return TensorDict(
            source={key: value for key, value in self.items()},
            batch_size=self.batch_size,
            _meta_source={k: value for k, value in self.items_meta()},
        )

    def clone(self, recursive: bool = True) -> LazyStackedTensorDict:
        if recursive:
            return LazyStackedTensorDict(
                *[td.clone() for td in self.tensor_dicts], stack_dim=self.stack_dim
            )
        return LazyStackedTensorDict(
            *[td for td in self.tensor_dicts], stack_dim=self.stack_dim
        )

    def pin_memory(self) -> LazyStackedTensorDict:
        for td in self.tensor_dicts:
            td.pin_memory()
        return self

    def to(self, dest: Union[DEVICE_TYPING, Type], **kwargs) -> _TensorDict:
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            return dest(source=self, batch_size=self.batch_size)
        elif isinstance(dest, (torch.device, str, int)):
            if not isinstance(dest, torch.device):
                dest = torch.device(dest)
            if dest == self.device:
                return self
            tds = [td.to(dest) for td in self.tensor_dicts]
            self_copy = copy(self)
            self_copy.tensor_dicts = tds
            return self_copy
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def items(self) -> Iterator[Tuple[str, COMPATIBLE_TYPES]]:
        for key in self.keys():
            item = self.get(key)
            yield key, item

    def items_meta(self) -> Iterator[Tuple[str, MetaTensor]]:
        for key in self.keys():
            item = self._get_meta(key)
            yield key, item

    def keys(self) -> Iterator[str]:
        for key in self.valid_keys:
            yield key

    def _update_valid_keys(self) -> None:
        valid_keys = set(self.tensor_dicts[0].keys())
        for td in self.tensor_dicts[1:]:
            valid_keys = valid_keys.intersection(td.keys())
        self._valid_keys = valid_keys

    def select(self, *keys: str, inplace: bool = False) -> LazyStackedTensorDict:
        if len(self.valid_keys.intersection(keys)) != len(keys):
            raise KeyError(
                f"Selected and existing keys mismatch, got self.valid_keys={self.valid_keys} and keys={keys}"
            )
        tensor_dicts = [td.select(*keys, inplace=inplace) for td in self.tensor_dicts]
        if inplace:
            return self
        return LazyStackedTensorDict(
            *tensor_dicts,
            stack_dim=self.stack_dim,
        )

    def __getitem__(
        self, item: Union[torch.Tensor, slice, Number, Tuple]
    ) -> _TensorDict:

        if isinstance(item, torch.Tensor) and item.dtype == torch.bool:
            return self.masked_select(item)
        elif (
            isinstance(item, (Number,))
            or (isinstance(item, torch.Tensor) and item.ndimension() == 0)
        ) and self.stack_dim == 0:
            return self.tensor_dicts[item]
        elif isinstance(item, (torch.Tensor, list)) and self.stack_dim == 0:
            return LazyStackedTensorDict(
                *[self.tensor_dicts[_item] for _item in item], stack_dim=self.stack_dim
            )
        elif isinstance(item, slice) and self.stack_dim == 0:
            return LazyStackedTensorDict(
                *self.tensor_dicts[item], stack_dim=self.stack_dim
            )
        elif isinstance(item, (slice, Number)):
            new_stack_dim = (
                self.stack_dim - 1 if isinstance(item, Number) else self.stack_dim
            )
            return LazyStackedTensorDict(
                *[td[item] for td in self.tensor_dicts], stack_dim=new_stack_dim
            )
        elif isinstance(item, tuple):
            _sub_item = tuple(
                _item for i, _item in enumerate(item) if i == self.stack_dim
            )
            if len(_sub_item):
                tensor_dicts = self.tensor_dicts[_sub_item[0]]
                if isinstance(tensor_dicts, _TensorDict):
                    return tensor_dicts
            else:
                tensor_dicts = self.tensor_dicts
            # select sub tensor_dicts
            _sub_item = tuple(
                _item for i, _item in enumerate(item) if i != self.stack_dim
            )
            if len(_sub_item):
                tensor_dicts = [td[_sub_item] for td in tensor_dicts]
            new_stack_dim = self.stack_dim - sum(
                [isinstance(_item, Number) for _item in item[: self.stack_dim]]
            )
            return torch.stack(list(tensor_dicts), dim=new_stack_dim)
        else:
            raise NotImplementedError(
                f"selecting StackedTensorDicts with type {item.__class__.__name__} is not "
                f"supported yet"
            )

    def del_(self, *args, **kwargs) -> LazyStackedTensorDict:
        for td in self.tensor_dicts:
            td.del_(*args, **kwargs)
        return self

    def share_memory_(self) -> LazyStackedTensorDict:
        for td in self.tensor_dicts:
            td.share_memory_()
        return self

    def detach_(self) -> LazyStackedTensorDict:
        for td in self.tensor_dicts:
            td.detach_()
        return self

    def memmap_(self) -> LazyStackedTensorDict:
        for td in self.tensor_dicts:
            td.memmap_()
        return self

    def is_shared(self, no_check: bool = False) -> bool:
        are_shared = [td.is_shared(no_check=no_check) for td in self.tensor_dicts]
        if any(are_shared) and not all(are_shared):
            raise RuntimeError(
                f"tensor_dicts shared status mismatch, got {sum(are_shared)} "
                f"shared tensor_dicts and {len(are_shared) - sum(are_shared)} non "
                f"shared tensordict "
            )
        return all(are_shared)

    def is_memmap(self, no_check: bool = False) -> bool:
        are_memmap = [td.is_memmap() for td in self.tensor_dicts]
        if any(are_memmap) and not all(are_memmap):
            raise RuntimeError(
                f"tensor_dicts memmap status mismatch, got {sum(are_memmap)} "
                f"memmap tensor_dicts and {len(are_memmap) - sum(are_memmap)} non "
                f"memmap tensordict "
            )
        return all(are_memmap)

    def expand(self, *shape: int, inplace: bool = False) -> LazyStackedTensorDict:
        stack_dim = self.stack_dim + len(shape)
        tensor_dicts = [td.expand(*shape) for td in self.tensor_dicts]
        if inplace:
            self.tensor_dicts = tensor_dicts
            self.stack_dim = stack_dim
            return self
        return torch.stack(tensor_dicts, stack_dim)

    def update(
        self, input_dict_or_td: _TensorDict, clone: bool = False, **kwargs
    ) -> LazyStackedTensorDict:
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set(key, value, **kwargs)
        return self

    def update_(
        self, input_dict_or_td: _TensorDict, clone: bool = False, **kwargs
    ) -> LazyStackedTensorDict:
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_(key, value, **kwargs)
        return self

    def rename_key(self, key1: str, key2: str, **kwargs) -> LazyStackedTensorDict:
        for td in self.tensor_dicts:
            td.rename_key(key1, key2, **kwargs)
        return self


class SavedTensorDict(_TensorDict):
    _safe = False

    def __init__(
        self,
        source: _TensorDict,
        device=None,
        batch_size: Optional[Iterable[int]] = None,
    ):
        if not isinstance(source, _TensorDict):
            raise TypeError(
                f"Expected source to be a _TensorDict instance, "
                f"but got {type(source)} instead."
            )
        self.file = tempfile.NamedTemporaryFile()
        self.filename = self.file.name
        if source.is_memmap():
            source = source.clone()
        self._device = (
            device
            if device
            else source.device
            if hasattr(source, "device")
            else source[list(source.keys())[0]].device
            if len(source)
            else "cpu"
        )
        td = source
        self._save(td)
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    def _save(self, tensor_dict: _TensorDict) -> None:
        self._keys = list(tensor_dict.keys())
        self._device = tensor_dict.device
        self._batch_size = tensor_dict.batch_size
        self._td_fields = _td_fields(tensor_dict)
        self._tensor_dict_meta = {key: value for key, value in tensor_dict.items_meta()}
        torch.save(tensor_dict, self.filename)

    def _load(self) -> _TensorDict:
        return torch.load(self.filename, self._device)

    def _get_meta(self, key: str) -> COMPATIBLE_TYPES:
        return self._tensor_dict_meta.get(key)

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def keys(self) -> Iterable:
        for k in self._keys:
            yield k

    def get(
        self, key: str, default: Union[None, str, torch.Tensor] = "_no_default_"
    ) -> COMPATIBLE_TYPES:
        td = self._load()
        return td.get(key, default=default)

    def set(self, key: str, value: COMPATIBLE_TYPES, **kwargs) -> _TensorDict:
        td = self._load()
        td.set(key, value, **kwargs)
        self._save(td)
        return self

    def expand(self, *shape: int, inplace: bool = False) -> SavedTensorDict:
        td = self._load()
        td = td.expand(*shape)
        if inplace:
            self._save(td)
            return self
        return td.to(SavedTensorDict)

    def set_(self, key: str, value: COMPATIBLE_TYPES) -> SavedTensorDict:
        self.set(key, value)
        return self

    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> SavedTensorDict:
        td = self._load()
        td.set_at_(key, value, idx)
        self._save(td)
        return self

    def update(
        self, input_dict_or_td: Union[_TensorDict, dict], clone: bool = False, **kwargs
    ) -> SavedTensorDict:
        if input_dict_or_td is self:
            # no op
            return self
        td = self._load()
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} but got {type(value)}"
                )
            if clone:
                value = value.clone()
            td.set(key, value, **kwargs)
        self._save(td)
        return self

    def update_(
        self, input_dict_or_td: Union[_TensorDict, dict], clone: bool = False
    ) -> SavedTensorDict:
        if input_dict_or_td is self:
            return self
        return self.update(input_dict_or_td, clone=clone)

    def __del__(self) -> None:
        if hasattr(self, "file"):
            self.file.close()

    def is_shared(self, no_check: bool = False) -> bool:
        return False

    def is_memmap(self, no_check: bool = False) -> bool:
        return False

    def share_memory_(self) -> None:
        raise RuntimeError("SavedTensorDict cannot be put in shared memory.")

    def memmap_(self) -> None:
        raise RuntimeError(
            "SavedTensorDict and memmap are mutually exclusive features."
        )

    def detach_(self) -> None:
        raise RuntimeError("SavedTensorDict cannot be put detached.")

    def items(self) -> Generator:
        return self.contiguous().items()

    def items_meta(self) -> ItemsView:
        return self._tensor_dict_meta.items()

    def is_contiguous(self) -> bool:
        return False

    def contiguous(self) -> _TensorDict:
        return self._load().contiguous()

    def clone(self, recursive: bool = True) -> SavedTensorDict:
        return self._load().to(SavedTensorDict)

    def select(self, *keys: str, inplace: bool = False) -> SavedTensorDict:
        _source = self.contiguous().select(*keys)
        if inplace:
            self._save(_source)
            return self
        return SavedTensorDict(source=_source)

    def rename_key(self, key1: str, key2: str, **kwargs) -> SavedTensorDict:
        td = self._load()
        td.rename_key(key1, key2, **kwargs)
        self._save(td)
        return self

    def __repr__(self) -> str:
        return f"SavedTensorDict(\n\tfields={{{self._td_fields}}}, \n\tbatch_size={self.batch_size}, \n\tfile={self.filename})"

    def to(self, dest: Union[DEVICE_TYPING, Type], **kwargs):
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            td = dest(
                source=TensorDict(self.to_dict(), batch_size=self.batch_size),
                **kwargs,
            )
            return td
        elif isinstance(dest, (torch.device, str, int)):
            # must be device
            if not isinstance(dest, torch.device):
                dest = torch.device(dest)
            if dest == self.device:
                return self
            self_copy = copy(self)
            self_copy._device = dest
            return self_copy
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def del_(self, key: str) -> SavedTensorDict:
        td = self._load()
        td = td.del_(key)
        self._save(td)
        return self

    def pin_memory(self) -> None:
        raise RuntimeError("pin_memory requires tensordicts that live in memory.")

    def __reduce__(self, *args, **kwargs):
        if hasattr(self, "file"):
            file = self.file
            del self.file
            self_copy = copy(self)
            self.file = file
            return super(SavedTensorDict, self_copy).__reduce__(*args, **kwargs)
        return super().__reduce__(*args, **kwargs)

    def __getitem__(
        self, idx: Union[torch.Tensor, slice, Number, Tuple]
    ) -> SavedTensorDict:
        if isinstance(idx, Number):
            idx = (idx,)
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            return self.masked_select(idx)
        if not self.batch_size:
            raise IndexError(
                "indexing a tensordict with td.batch_dims==0 is not permitted"
            )
        return self.get_sub_tensor_dict(idx)


class _CustomOpTensorDict(_TensorDict):
    def __init__(
        self,
        source: _TensorDict,
        custom_op: str,
        inv_op: Optional[str] = None,
        custom_op_kwargs: Optional[dict] = None,
        inv_op_kwargs: Optional[dict] = None,
        batch_size: Optional[Iterable[int]] = None,
    ):
        """
        Encodes lazy operations on tensors contained in a TensorDict.

        Args:
            source:
            custom_op:
            inv_op:
            custom_op_kwargs:
            inv_op_kwargs:
        """
        if not isinstance(source, _TensorDict):
            raise TypeError(
                f"Expected source to be a _TensorDict isntance, "
                f"but got {type(source)} instead."
            )
        self._source = source
        self.custom_op = custom_op
        self.inv_op = inv_op
        self.custom_op_kwargs = custom_op_kwargs if custom_op_kwargs is not None else {}
        self.inv_op_kwargs = inv_op_kwargs if inv_op_kwargs is not None else {}
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    def _update_custom_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        """
        Allows for a transformation to be customized for a certain shape, device or dtype.
        By default, this is a no-op on self.custom_op_kwargs

        Args:
            source_meta_tensor: corresponding MetaTensor

        Returns: a dictionary with the kwargs of the operation to execute for the tensor

        """
        return self.custom_op_kwargs

    def _update_inv_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        """
        Allows for an inverse transformation to be customized for a certain shape, device or dtype.
        By default, this is a no-op on self.inv_op_kwargs

        Args:
            source_meta_tensor: corresponding MetaTensor

        Returns: a dictionary with the kwargs of the operation to execute for the tensor

        """
        return self.inv_op_kwargs

    @property
    def device(self) -> torch.device:
        return self._source.device

    def _get_meta(self, key: str, **kwargs) -> MetaTensor:
        item = self._source._get_meta(key, **kwargs)
        return getattr(item, self.custom_op)(**self._update_custom_op_kwargs(item))

    def items_meta(self) -> Iterator[Tuple[str, MetaTensor]]:
        for key, value in self._source.items_meta():
            yield key, self._get_meta(key)

    def items(self) -> Iterator[Tuple[str, COMPATIBLE_TYPES]]:
        for key in self._source.keys():
            yield key, self.get(key)

    @property
    def batch_size(self) -> torch.Size:
        return getattr(MetaTensor(*self._source.batch_size), self.custom_op)(
            **self.custom_op_kwargs
        ).shape

    def get(
        self,
        key: str,
        default: Union[None, str, torch.Tensor] = "_no_default_",
        _return_original_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, COMPATIBLE_TYPES], COMPATIBLE_TYPES]:
        try:
            source_meta_tensor = self._source._get_meta(key)
            item = self._source.get(key, default)
            transformed_tensor = getattr(item, self.custom_op)(
                **self._update_custom_op_kwargs(source_meta_tensor)
            )
            if not _return_original_tensor:
                return transformed_tensor
            return transformed_tensor, item
        except KeyError:
            return self._default_get(key, default)

    def set(self, key: str, value: COMPATIBLE_TYPES, **kwargs) -> _CustomOpTensorDict:
        if self.inv_op is None:
            raise Exception(
                f"{self.__class__.__name__} does not support setting values. "
                f"Consider calling .contiguous() before calling this method."
            )
        value = self._process_tensor(
            value, check_device=False, check_tensor_shape=False
        )
        if key in self.keys():
            source_meta_tensor = self._source._get_meta(key)
        else:
            source_meta_tensor = MetaTensor(
                *value.shape, device=value.device, dtype=value.dtype
            )
        value = getattr(value, self.inv_op)(
            **self._update_inv_op_kwargs(source_meta_tensor)
        )
        self._source.set(key, value, **kwargs)
        return self

    def set_(self, key: str, value: COMPATIBLE_TYPES, **kwargs) -> _CustomOpTensorDict:
        if self.inv_op is None:
            raise Exception(
                f"{self.__class__.__name__} does not support setting values. "
                f"Consider calling .contiguous() before calling this method."
            )
        meta_tensor = self._source._get_meta(key)
        value = getattr(value, self.inv_op)(**self._update_inv_op_kwargs(meta_tensor))
        self._source.set_(key, value, **kwargs)
        return self

    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> _CustomOpTensorDict:
        transformed_tensor, original_tensor = self.get(
            key, _return_original_tensor=True
        )
        if transformed_tensor.data_ptr() != original_tensor.data_ptr():
            raise RuntimeError(
                f"{self} original tensor and transformed do not point to the same storage. "
                f"Setting values in place is not currently supported in this setting, consider calling "
                f"`td.clone()` before `td.set_at_(...)`"
            )
        transformed_tensor[idx] = value
        return self

    def __repr__(self) -> str:
        custom_op_kwargs_str = ", ".join(
            [f"{key}={value}" for key, value in self.custom_op_kwargs.items()]
        )
        indented_source = textwrap.indent(f"source={self._source}", "\t")
        return f"{self.__class__.__name__}(\n{indented_source}, \n\top={self.custom_op}({custom_op_kwargs_str}))"

    def keys(self) -> KeysView:
        return self._source.keys()

    def select(self, *keys: str, inplace: bool = False) -> _CustomOpTensorDict:
        if inplace:
            self._source.select(*keys, inplace=inplace)
            return self
        try:
            return type(self)(
                source=self._source.select(*keys),
                custom_op=self.custom_op,
                inv_op=self.inv_op,
                custom_op_kwargs=self.custom_op_kwargs,
                inv_op_kwargs=self.inv_op_kwargs,
            )
        except TypeError:
            self_copy = deepcopy(self)
            self_copy._source = self._source.select(*keys)
            return self_copy

    def clone(self, recursive: bool = True) -> TensorDict:
        if not recursive:
            return copy(self)
        return TensorDict(
            source=self.to_dict(),
            batch_size=self.batch_size,
        )

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def contiguous(self) -> TensorDict:
        if self.is_contiguous():
            return self
        return self.to(TensorDict)

    def rename_key(self, key1: str, key2: str, **kwargs) -> _CustomOpTensorDict:
        self._source.rename_key(key1, key2, **kwargs)
        return self

    def del_(self, key: str) -> _CustomOpTensorDict:
        self._source = self._source.del_(key)
        return self

    def to(self, dest: Union[DEVICE_TYPING, Type], **kwargs) -> _TensorDict:
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            return dest(source=self.contiguous().clone())
        elif isinstance(dest, (torch.device, str, int)):
            if torch.device(dest) == self.device:
                return self
            td = self._source.to(dest)
            self_copy = copy(self)
            self_copy._source = td
            return self_copy
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def pin_memory(self) -> _CustomOpTensorDict:
        self._source.pin_memory()
        return self


class UnsqueezedTensorDict(_CustomOpTensorDict):
    """
    A lazy view on an unsqueezed TensorDict.
    When calling `tensordict.unsqueeze(dim)`, a lazy view of this operation is returned such that the following code
    snippet works without raising an exception:
        >>> assert tensordict.unsqueeze(dim).squeeze(dim) is tensordict

    Examples:
        >>> from torchrl.data import TensorDict
        >>> import torch
        >>> td = TensorDict({'a': torch.randn(3, 4)}, batch_size=[3])
        >>> td_unsqueeze = td.unsqueeze(-1)
        >>> print(td_unsqueeze.shape)
        torch.Size([3, 1])
        >>> print(td_unsqueeze.squeeze(-1) is td)
        True
    """

    def squeeze(self, dim: int) -> _TensorDict:
        if dim < 0:
            dim = self.batch_dims + dim
        if dim == self.custom_op_kwargs.get("dim"):
            return self._source
        return super().squeeze(dim)


class SqueezedTensorDict(_CustomOpTensorDict):
    """
    A lazy view on a squeezed TensorDict.
    See the `UnsqueezedTensorDict` class documentation for more information.
    """

    def unsqueeze(self, dim: int) -> _TensorDict:
        if dim < 0:
            dim = self.batch_dims + dim + 1
        if dim == self.inv_op_kwargs.get("dim"):
            return self._source
        return super().unsqueeze(dim)


class ViewedTensorDict(_CustomOpTensorDict):
    def _update_custom_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        new_dim = torch.Size(
            [
                *self.custom_op_kwargs.get("size"),
                *source_meta_tensor.shape[self._source.batch_dims :],
            ]
        )
        new_dict = deepcopy(self.custom_op_kwargs)
        new_dict.update({"size": new_dim})
        return new_dict

    def _update_inv_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        new_dim = torch.Size(
            [
                *self.inv_op_kwargs.get("size"),
                *source_meta_tensor.shape[self._source.batch_dims :],
            ]
        )
        new_dict = deepcopy(self.inv_op_kwargs)
        new_dict.update({"size": new_dim})
        return new_dict

    def view(
        self, *shape, size: Optional[Union[List, Tuple, torch.Size]] = None
    ) -> _TensorDict:
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        if shape == self._source.batch_size:
            return self._source
        return super().view(*shape)


def _td_fields(td: _TensorDict) -> str:
    return indent(
        "\n"
        + ",\n".join(
            [
                f"{key}: {item.class_name}({item.shape}, dtype={item.dtype})"
                for key, item in td.items_meta()
            ]
        ),
        4 * " ",
    )


def _check_keys(list_of_tensor_dicts: _TensorDict, strict: bool = False) -> set:
    keys = set()
    for td in list_of_tensor_dicts:
        if not len(keys):
            keys = set(td.keys())
        else:
            if not strict:
                keys = keys.intersection(set(td.keys()))
            else:
                if len(set(td.keys()).difference(keys)) or len(set(td.keys())) != len(
                    keys
                ):
                    raise KeyError(
                        f"got keys {keys} and {set(td.keys())} which are incompatible"
                    )

    return keys
