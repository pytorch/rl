# ReplayDatasetAdapter

*class*torchrl.data.datasets.ReplayDatasetAdapter(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/datasets/adapters.html#ReplayDatasetAdapter)

Import and export an external dataset format for use with replay buffers.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import TensorStorage
>>> from torchrl.data.datasets import ReplayDatasetAdapter
>>> class ColumnAdapter:
... def from_external(self, columns):
... size = next(iter(columns.values())).shape[0]
... return TensorStorage(TensorDict(columns, [size])).as_dataset()
... def to_external(self, dataset):
... keys = dataset[0].keys()
... return {
... key: torch.stack([dataset[index][key] for index in range(len(dataset))])
... for key in keys
... }
>>> adapter: ReplayDatasetAdapter[dict[str, torch.Tensor]] = ColumnAdapter()
>>> external = {"observation": torch.arange(2)}
>>> round_trip = adapter.to_external(adapter.from_external(external))
>>> torch.equal(round_trip["observation"], external["observation"])
True
```

See also

[`ReplayDatasetExporter`](torchrl.data.datasets.ReplayDatasetExporter.html#torchrl.data.datasets.ReplayDatasetExporter) and
[`ReplayDatasetImporter`](torchrl.data.datasets.ReplayDatasetImporter.html#torchrl.data.datasets.ReplayDatasetImporter) for one-way
integrations.

from_external(*dataset: _ExternalDatasetT_contra*, */*) → [Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]

Create a TorchRL map-style dataset from a native dataset object.

Parameters:

**dataset** (*ExternalDatasetT*) - dataset in the external library's native
format.

Returns:

finite map-style dataset with one

[`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) item per index.

Return type:

Dataset[TensorDictBase]

to_external(*dataset: [Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, */*) → _ExternalDatasetT_co

Convert a TorchRL map-style dataset to a native external dataset.

Parameters:

**dataset** (*Dataset**[**TensorDictBase**]*) - finite map-style dataset with one
[`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) item per index.

Returns:

The dataset in the external library's native format.