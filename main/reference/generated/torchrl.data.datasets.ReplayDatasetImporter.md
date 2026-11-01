# ReplayDatasetImporter

*class*torchrl.data.datasets.ReplayDatasetImporter(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/datasets/adapters.html#ReplayDatasetImporter)

Expose an external dataset as a TorchRL map-style dataset.

Implement this protocol to make an external library's native dataset
available as a finite [`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) whose items are
[`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) instances. The result may be a lazy view
or a materialized dataset.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import TensorStorage
>>> from torchrl.data.datasets import ReplayDatasetImporter
>>> class ColumnImporter:
... def from_external(self, columns):
... size = next(iter(columns.values())).shape[0]
... return TensorStorage(TensorDict(columns, [size])).as_dataset()
>>> importer: ReplayDatasetImporter[dict[str, torch.Tensor]] = ColumnImporter()
>>> dataset = importer.from_external({"observation": torch.arange(2)})
>>> dataset[1]["observation"]
tensor(1)
```

See also

[`ReplayDatasetExporter`](torchrl.data.datasets.ReplayDatasetExporter.html#torchrl.data.datasets.ReplayDatasetExporter) for exporting,
[`ReplayDatasetAdapter`](torchrl.data.datasets.ReplayDatasetAdapter.html#torchrl.data.datasets.ReplayDatasetAdapter) for bidirectional
conversion, and `StorageDataset` for the returned
dataset interface.

from_external(*dataset: _ExternalDatasetT_contra*, */*) → [Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/data/datasets/adapters.html#ReplayDatasetImporter.from_external)

Create a TorchRL map-style dataset from a native dataset object.

Parameters:

**dataset** (*ExternalDatasetT*) - dataset in the external library's native
format.

Returns:

finite map-style dataset with one

[`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) item per index.

Return type:

Dataset[TensorDictBase]