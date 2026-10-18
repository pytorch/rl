# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Protocol, TypeVar

from tensordict import TensorDictBase
from torch.utils.data import Dataset

__all__ = [
    "ReplayDatasetAdapter",
    "ReplayDatasetExporter",
    "ReplayDatasetImporter",
]


ExternalDatasetT = TypeVar("ExternalDatasetT")
_ExternalDatasetT_co = TypeVar("_ExternalDatasetT_co", covariant=True)
_ExternalDatasetT_contra = TypeVar("_ExternalDatasetT_contra", contravariant=True)


class ReplayDatasetExporter(Protocol[_ExternalDatasetT_co]):
    """Convert replay storage data to an external dataset format.

    Implement this protocol to export a finite, map-style dataset whose items
    are :class:`~tensordict.TensorDictBase` instances. The returned object is
    native to the external library and can, for example, be published or saved
    with that library. This converts stored data only; replay-buffer state such
    as the sampler and writer state is outside this protocol.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
        >>> from torchrl.data.datasets import ReplayDatasetExporter
        >>> class ListExporter:
        ...     def to_external(self, dataset):
        ...         return [dataset[index].to_dict() for index in range(len(dataset))]
        >>> replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=2))
        >>> _ = replay_buffer.extend(TensorDict({"observation": torch.arange(2)}, [2]))
        >>> exporter: ReplayDatasetExporter[list[dict]] = ListExporter()
        >>> rows = exporter.to_external(replay_buffer.storage.as_dataset())
        >>> rows[1]["observation"]
        tensor(1)

    .. seealso::
        :class:`~torchrl.data.datasets.ReplayDatasetImporter` for importing,
        :class:`~torchrl.data.datasets.ReplayDatasetAdapter` for bidirectional
        conversion, and :class:`~torchrl.data.StorageDataset` for a replay
        storage view accepted by :meth:`to_external`.
    """

    def to_external(self, dataset: Dataset[TensorDictBase], /) -> _ExternalDatasetT_co:
        """Convert a TorchRL map-style dataset to a native external dataset.

        Args:
            dataset (Dataset[TensorDictBase]): finite map-style dataset with one
                :class:`~tensordict.TensorDictBase` item per index.

        Returns:
            The dataset in the external library's native format.
        """
        ...


class ReplayDatasetImporter(Protocol[_ExternalDatasetT_contra]):
    """Expose an external dataset as a TorchRL map-style dataset.

    Implement this protocol to make an external library's native dataset
    available as a finite :class:`torch.utils.data.Dataset` whose items are
    :class:`~tensordict.TensorDictBase` instances. The result may be a lazy view
    or a materialized dataset.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import TensorStorage
        >>> from torchrl.data.datasets import ReplayDatasetImporter
        >>> class ColumnImporter:
        ...     def from_external(self, columns):
        ...         size = next(iter(columns.values())).shape[0]
        ...         return TensorStorage(TensorDict(columns, [size])).as_dataset()
        >>> importer: ReplayDatasetImporter[dict[str, torch.Tensor]] = ColumnImporter()
        >>> dataset = importer.from_external({"observation": torch.arange(2)})
        >>> dataset[1]["observation"]
        tensor(1)

    .. seealso::
        :class:`~torchrl.data.datasets.ReplayDatasetExporter` for exporting,
        :class:`~torchrl.data.datasets.ReplayDatasetAdapter` for bidirectional
        conversion, and :class:`~torchrl.data.StorageDataset` for the returned
        dataset interface.
    """

    def from_external(
        self, dataset: _ExternalDatasetT_contra, /
    ) -> Dataset[TensorDictBase]:
        """Create a TorchRL map-style dataset from a native dataset object.

        Args:
            dataset (ExternalDatasetT): dataset in the external library's native
                format.

        Returns:
            Dataset[TensorDictBase]: finite map-style dataset with one
                :class:`~tensordict.TensorDictBase` item per index.
        """
        ...


class ReplayDatasetAdapter(
    ReplayDatasetExporter[ExternalDatasetT],
    ReplayDatasetImporter[ExternalDatasetT],
    Protocol,
):
    """Import and export an external dataset format for use with replay buffers.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import TensorStorage
        >>> from torchrl.data.datasets import ReplayDatasetAdapter
        >>> class ColumnAdapter:
        ...     def from_external(self, columns):
        ...         size = next(iter(columns.values())).shape[0]
        ...         return TensorStorage(TensorDict(columns, [size])).as_dataset()
        ...     def to_external(self, dataset):
        ...         keys = dataset[0].keys()
        ...         return {
        ...             key: torch.stack([dataset[index][key] for index in range(len(dataset))])
        ...             for key in keys
        ...         }
        >>> adapter: ReplayDatasetAdapter[dict[str, torch.Tensor]] = ColumnAdapter()
        >>> external = {"observation": torch.arange(2)}
        >>> round_trip = adapter.to_external(adapter.from_external(external))
        >>> torch.equal(round_trip["observation"], external["observation"])
        True

    .. seealso::
        :class:`~torchrl.data.datasets.ReplayDatasetExporter` and
        :class:`~torchrl.data.datasets.ReplayDatasetImporter` for one-way
        integrations.
    """
