# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from tensordict import is_non_tensor, TensorDict, TensorDictBase
from torch.utils.data import DataLoader, Dataset

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


def _feature_dtype(feature: Any) -> np.dtype | None:
    while feature is not None and not isinstance(feature, Mapping):
        dtype = getattr(feature, "dtype", None)
        if dtype is not None:
            try:
                return np.dtype(dtype)
            except (TypeError, ValueError):
                return None
        feature = getattr(feature, "feature", None)
    return None


def _decode_feature(value: Any, feature: Any) -> Any:
    if value is None:
        return None
    if isinstance(feature, Mapping) and isinstance(value, Mapping):
        return {
            key: _decode_feature(item, feature.get(key)) for key, item in value.items()
        }
    dtype = _feature_dtype(feature)
    if dtype is None:
        return value
    try:
        return torch.as_tensor(np.asarray(value, dtype=dtype))
    except (TypeError, ValueError):
        return value


class _HuggingFaceDataset(Dataset[TensorDictBase]):
    def __init__(self, dataset: HFDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TensorDictBase:
        features = self.dataset.features
        return TensorDict.from_dict(
            {
                key: _decode_feature(value, features.get(key))
                for key, value in self.dataset[index].items()
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
    across its items. Column keys must also match between batches, including
    nested keys.

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

    def _encode_columns(self, columns: Mapping[str, Any]) -> dict[str, Any]:
        arrays = {}
        for key, value in columns.items():
            if isinstance(value, Mapping):
                nested = self._encode_columns(value)
                arrays[key] = self._pa.StructArray.from_arrays(
                    list(nested.values()), names=list(nested)
                )
            elif isinstance(value, torch.Tensor):
                # NumPy can share CPU tensor memory. HF's Arrow converter takes
                # a copy so later replay writes cannot change exported values.
                arrays[key] = self._numpy_to_arrow(self._to_numpy(value))
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
                have matching keys and tensor shapes.

        Returns:
            datasets.Dataset: materialized Hugging Face dataset. Call
                :meth:`datasets.Dataset.push_to_hub` to publish it.
        """
        tables = []
        column_keys = None
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=tensordict_collate,
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
            columns = batch.detach().cpu().to_dict()
            tables.append(self._pa.Table.from_pydict(self._encode_columns(columns)))
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
                :func:`datasets.load_dataset`.

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
