# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from functools import partial
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from tensordict import (
    is_non_tensor,
    MetaData,
    NestedKey,
    NonTensorData,
    NonTensorStack,
    TensorDict,
    TensorDictBase,
    UnbatchedTensor,
)
from torch.utils.data import DataLoader, Dataset

from torchrl.data.datasets._adapter_utils import _validate_export_dtypes
from torchrl.data.replay_buffers.dataloader import tensordict_collate

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
else:
    HFDataset = Any

__all__ = ["HuggingFaceDatasetAdapter"]


_has_datasets = importlib.util.find_spec("datasets") is not None
_has_pyarrow = importlib.util.find_spec("pyarrow") is not None


def _column_keys(batch: TensorDictBase) -> dict[str, Any]:
    if not isinstance(batch, TensorDict):
        raise ValueError(
            "Hugging Face export requires dense TensorDict batches with the same "
            "keys and tensor shapes for every item in a batch."
        )
    return {
        key: _column_keys(value)
        if isinstance(value, TensorDictBase) and not is_non_tensor(value)
        else None
        for key, value in batch.items()
    }


def _collate_for_export(batch: Any, *, dtypes: dict[NestedKey, torch.dtype]) -> Any:
    _validate_export_dtypes(batch, dtypes=dtypes)
    return tensordict_collate(batch)


def _batch_to_columns(
    batch: TensorDictBase, *, prefix: tuple[str, ...] = ()
) -> dict[str, Any]:
    if not isinstance(batch, TensorDict):
        raise ValueError(
            "Hugging Face export requires dense TensorDict batches with the same "
            "keys and tensor shapes for every item in a batch."
        )
    columns = {}
    for key, value in batch.items():
        path = (*prefix, key)
        display_key = path[0] if len(path) == 1 else path
        if isinstance(value, MetaData):
            columns[key] = [value.data] * len(batch)
        elif isinstance(value, NonTensorData):
            columns[key] = value.tolist()
        elif isinstance(value, NonTensorStack):
            values = [
                item.data if isinstance(item, MetaData) else item.tolist()
                for item in value.unbind(0)
            ]
            if len(values) != len(batch):
                raise ValueError(
                    f"NonTensorStack at key {display_key!r} has {len(values)} values for "
                    f"a batch of length {len(batch)}."
                )
            columns[key] = values
        elif isinstance(value, UnbatchedTensor):
            raise ValueError(
                f"Hugging Face export does not support UnbatchedTensor at key "
                f"{display_key!r}; convert it to a batched tensor first."
            )
        elif isinstance(value, TensorDictBase):
            if is_non_tensor(value):
                raise TypeError(
                    "Hugging Face export does not support non-tensor TensorDict "
                    f"entries of type {type(value).__name__} at key {display_key!r}."
                )
            if not value.keys():
                raise ValueError(
                    "Hugging Face export does not support empty nested "
                    f"TensorDicts at key {display_key!r}. Remove the empty entry "
                    "before export."
                )
            columns[key] = _batch_to_columns(value, prefix=path)
        elif isinstance(value, torch.Tensor):
            if (
                value.layout != torch.strided
                or value.is_quantized
                or value.is_complex()
            ):
                raise TypeError(
                    "Hugging Face export does not support tensor layout or dtype "
                    f"at key {display_key!r}: {value.layout}, {value.dtype}."
                )
            if value.ndim < batch.ndim or value.shape[: batch.ndim] != batch.batch_size:
                raise ValueError(
                    f"Tensor at key {display_key!r} does not start with batch size "
                    f"{list(batch.batch_size)}."
                )
            if 0 in value.shape[1:-1]:
                raise ValueError(
                    f"Tensor at key {display_key!r} has shape {list(value.shape[1:])} "
                    "with an empty inner dimension. Hugging Face export cannot "
                    "preserve dimensions following an empty dimension."
                )
            columns[key] = value
        else:
            raise TypeError(
                f"Hugging Face export does not support entries of type "
                f"{type(value).__name__} at key {display_key!r}."
            )
    return columns


def _feature_dtype(feature: Any) -> np.dtype | None:
    while feature is not None and not isinstance(feature, Mapping):
        dtype = getattr(feature, "dtype", None)
        if dtype is not None:
            try:
                return np.dtype(dtype)
            except (TypeError, ValueError):
                # Datasets 3.x Sequence declares dtype="list" on the wrapper.
                pass
        feature = getattr(feature, "feature", None)
    return None


def _contains_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.hasobject and any(
            _contains_null(item) for item in value.flat
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_null(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_null(item) for item in value.values())
    return False


def _as_numeric_array(value: Any, dtype: np.dtype | None) -> np.ndarray | None:
    """Convert numeric data without rounding uint64 or replacing nulls.

    Return None for missing data; unsupported or ragged values raise NumPy's
    conversion error. Dense arrays avoid a Python scan of their elements.
    """
    if value is None:
        return None
    array = None
    if isinstance(value, (list, tuple, np.ndarray)):
        # Integer conversion rejects None and avoids rounding large uint64 values.
        # Infer other dtypes first so None cannot become NaN or False.
        array_dtype = dtype if dtype is not None and dtype.kind in "iu" else None
        try:
            array = np.asarray(value, dtype=array_dtype)
        except (TypeError, ValueError):
            pass
        if (array is None or array.dtype.hasobject) and _contains_null(value):
            return None
    return np.asarray(array if array is not None else value, dtype=dtype)


def _array_shape(value: Any, feature: Any) -> tuple[int, ...] | None:
    dimensions = []
    while feature is not None:
        shape = getattr(feature, "shape", None)
        if shape is not None:
            dimensions.extend(
                len(value) if size is None and value is not None else size
                for size in shape
            )
            if None in dimensions:
                raise ValueError(
                    "Cannot reconstruct dynamic Hugging Face array dimensions "
                    "beneath an empty sequence."
                )
            return tuple(dimensions)
        child = getattr(feature, "feature", None)
        if child is not None:
            if value is None:
                length = getattr(feature, "length", -1)
                dimensions.append(length if length >= 0 else None)
            else:
                dimensions.append(len(value))
                value = value[0] if len(value) else None
        feature = child
    return None


def _decode_feature(value: Any, feature: Any) -> Any:
    if value is None:
        return None
    if isinstance(feature, Mapping) and isinstance(value, Mapping):
        return {
            key: _decode_feature(item, feature.get(key)) for key, item in value.items()
        }
    dtype = _feature_dtype(feature)
    try:
        array = _as_numeric_array(value, dtype)
        if array is None:
            # Dense tensors have no validity bitmap. Keep nulls and NaNs intact.
            return NonTensorData(value)
        if dtype is None:
            return value
        tensor = torch.as_tensor(array)
    except (TypeError, ValueError):
        return value
    shape = _array_shape(value, feature)
    return tensor.reshape(shape) if shape is not None else tensor


class _HuggingFaceDataset(Dataset[TensorDictBase]):
    def __init__(self, dataset: HFDataset) -> None:
        from pyarrow import ArrowInvalid

        # Read stored values without precision-changing formatters or transforms.
        # with_format returns a view and does not change the caller's dataset.
        self.dataset = dataset.with_format(None)
        self.features = self.dataset.features
        self._arrow_invalid = ArrowInvalid

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TensorDictBase:
        try:
            row = self.dataset[index]
        except self._arrow_invalid as error:
            raise ValueError(
                f"Hugging Face could not decode row {index}: {error}. "
                "For nullable ArrayXD data, use nested sequence features."
            ) from error
        return TensorDict.from_dict(
            {
                key: _decode_feature(value, self.features.get(key))
                for key, value in row.items()
            },
            batch_size=[],
        )


class HuggingFaceDatasetAdapter:
    """Convert between TorchRL map-style datasets and Hugging Face datasets.

    Each item in the TorchRL dataset maps to exactly one Hugging Face row.
    Items must be unbatched :class:`~tensordict.TensorDict` instances. Nested
    keys are preserved, and tensor leaves move to CPU for Arrow storage.
    Import reconstructs an unbatched TensorDict using the declared Hugging
    Face feature types for numeric tensor dtypes.

    Export reads the source synchronously in batches and converts tensor
    columns directly to Arrow arrays. Datasets with batched reads (including
    TorchRL storage datasets) avoid fetching items individually. Other datasets
    are stacked with :func:`~torchrl.data.tensordict_collate` before column
    conversion. Each batch must be dense, with matching keys and tensor shapes
    across its items. Tensor dtypes and column keys must match between all
    rows, including nested keys. :class:`~tensordict.NonTensorData` entries
    expand across their batch dimensions, including extra nested dimensions.
    :class:`~tensordict.MetaData` entries are repeated once per output row;
    :class:`~tensordict.NonTensorStack` entries provide one value per row.
    :class:`~tensordict.UnbatchedTensor` entries must be converted to batched
    tensors before export. Non-tensor payloads must be supported by Hugging Face
    Datasets and Apache Arrow feature inference. Empty nested TensorDicts and
    tensor shapes with dimensions following an empty dimension are rejected
    because inferred Arrow lists cannot preserve their structure reliably.
    Sparse, quantized, complex and float8 tensor leaves are not supported.
    Bfloat16 tensors are converted to float32 for Arrow storage.

    Import reads stored data without applying Hugging Face formatting or
    transforms, leaving the caller's dataset unchanged. Numeric sequences
    containing nulls remain :class:`~tensordict.NonTensorData` payloads, since
    tensors cannot distinguish missing values from valid values. Non-tensor
    wrapper types, nested batch sizes and dimension names are not reconstructed;
    other numeric values are normalized to tensors using their feature dtypes.
    Declared Hugging Face array shapes are restored, including empty dimensions;
    unresolved dynamic dimensions beneath empty sequences are rejected.
    Nullable ArrayXD features that Hugging Face cannot decode are rejected;
    use nested sequence features to preserve nulls in arrays.
    Export reads are not synchronized with replay writes; stop writers when a
    consistent snapshot is required. All converted Arrow batches stay in memory.

    Loading from and publishing to the Hub remain operations of the
    `Hugging Face datasets library <https://huggingface.co/docs/datasets>`_.

    Keyword Args:
        batch_size (int): maximum number of source items converted
            at once. Reduce this for large observations to limit temporary
            memory use. Defaults to 1024.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
        >>> from torchrl.data.datasets import HuggingFaceDatasetAdapter
        >>> replay_buffer = ReplayBuffer(storage=LazyTensorStorage(3))
        >>> _ = replay_buffer.extend(
        ...     TensorDict({"observation": torch.arange(6).reshape(3, 2)}, [3])
        ... )
        >>> adapter = HuggingFaceDatasetAdapter()
        >>> hf_dataset = adapter.to_external(
        ...     replay_buffer.storage.as_dataset()
        ... )
        >>> imported = adapter.from_external(hf_dataset)
        >>> imported[0]["observation"]
        tensor([0, 1])
        >>> hf_dataset.push_to_hub("organization/replay-data")  # doctest: +SKIP
    """

    def __init__(self, *, batch_size: int = 1024) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be a positive integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not _has_datasets:
            raise ImportError(
                "HuggingFaceDatasetAdapter requires the `datasets` package. "
                "Install it directly or install TorchRL with the "
                "`offline-data` extra."
            )
        if not _has_pyarrow:
            raise ImportError(
                "HuggingFaceDatasetAdapter requires the `pyarrow` package. "
                "Install it directly or install TorchRL with the "
                "`offline-data` extra."
            )
        import datasets
        import pyarrow as pa
        from datasets.features.features import (
            cast_to_python_objects,
            numpy_to_pyarrow_listarray,
        )

        self._datasets = datasets
        self._pa = pa
        self._to_numpy = cast_to_python_objects
        self._numpy_to_arrow = numpy_to_pyarrow_listarray
        self.batch_size = batch_size

    def _encode_columns(
        self, columns: Mapping[str, Any], *, prefix: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        arrays = {}
        for key, value in columns.items():
            path = (*prefix, key)
            if isinstance(value, Mapping):
                nested = self._encode_columns(value, prefix=path)
                arrays[key] = self._pa.StructArray.from_arrays(
                    list(nested.values()), names=list(nested)
                )
            elif isinstance(value, torch.Tensor):
                # NumPy can share CPU tensor memory. HF's Arrow converter takes
                # a copy so later replay writes cannot change exported values.
                try:
                    arrays[key] = self._numpy_to_arrow(self._to_numpy(value))
                except (TypeError, ValueError) as error:
                    display_key = path[0] if len(path) == 1 else path
                    raise TypeError(
                        f"Cannot convert tensor at key {display_key!r} with "
                        f"dtype {value.dtype} to Arrow."
                    ) from error
            else:
                arrays[key] = (
                    self._datasets.Dataset.from_dict({key: value})
                    .data[key]
                    .combine_chunks()
                )
        return arrays

    def to_external(self, dataset: Dataset[TensorDictBase], /) -> HFDataset:
        """Convert a TorchRL map-style dataset to a Hugging Face dataset.

        Args:
            dataset (Dataset[TensorDictBase]): finite map-style dataset with one
                unbatched TensorDict item per index. Items in each batch must
                have matching keys and tensor shapes. Tensor dtypes must be
                consistent across all rows.

        Returns:
            datasets.Dataset: materialized Hugging Face dataset. Call
                :meth:`datasets.Dataset.push_to_hub` to publish it.
        """
        tables = []
        column_keys = None
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=partial(_collate_for_export, dtypes={}),
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        offset = 0
        for batch in loader:
            if not isinstance(batch, TensorDictBase):
                raise TypeError(
                    "Hugging Face export expects TensorDict items, "
                    f"but got a {type(batch).__name__} batch."
                )
            if batch.ndim != 1:
                raise ValueError(
                    "Hugging Face export expects an unbatched TensorDict per item, "
                    f"but got a batch with {batch.ndim} batch dimensions."
                )
            keys = _column_keys(batch)
            if not keys:
                raise ValueError(
                    "Hugging Face export requires nonempty TensorDict items."
                )
            if column_keys is not None and keys != column_keys:
                raise ValueError(f"Source row {offset} has different column keys.")
            column_keys = keys
            columns = _batch_to_columns(batch.detach().cpu())
            table = self._pa.Table.from_pydict(self._encode_columns(columns))
            if table.num_rows != len(batch):
                raise ValueError(
                    f"Source batch at row {offset} contains {len(batch)} rows, "
                    f"but Arrow conversion produced {table.num_rows} rows."
                )
            tables.append(table)
            offset += len(batch)
        if not tables:
            return self._datasets.Dataset.from_dict({})
        schema = self._pa.unify_schemas(
            [table.schema for table in tables], promote_options="permissive"
        )
        # HF hashes each batch and composes their fingerprints on concatenation.
        # Hashing the entire dataset at once would serialize all its data in memory.
        chunks = [self._datasets.Dataset(table.cast(schema)) for table in tables]
        if len(chunks) == 1:
            return chunks[0]
        return self._datasets.concatenate_datasets(chunks)

    def from_external(self, dataset: HFDataset, /) -> Dataset[TensorDictBase]:
        """Expose a Hugging Face dataset as a TorchRL map-style dataset.

        Args:
            dataset (datasets.Dataset): finite Hugging Face dataset to adapt.
                Select one split before passing a ``DatasetDict`` returned by
                :func:`datasets.load_dataset`. Stored values are read without
                applying active formatters or transforms.

        Returns:
            Dataset[TensorDictBase]: lazy decoding view with one TensorDict item
                per Hugging Face row.
        """
        if not isinstance(dataset, self._datasets.Dataset):
            raise TypeError(
                "`dataset` must be a Hugging Face `datasets.Dataset`, but got "
                f"{type(dataset).__name__}. Select a split before adapting a "
                "DatasetDict."
            )
        return _HuggingFaceDataset(dataset)
