# ReplayDatasetExporter

*class*torchrl.data.datasets.ReplayDatasetExporter(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/datasets/adapters.html#ReplayDatasetExporter)

Convert replay storage data to an external dataset format.

Implement this protocol to export a finite, map-style dataset whose items
are [`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) instances. The returned object is
native to the external library and can, for example, be published or saved
with that library. Use exported replay data for offline learning,
archiving, or sharing with other tools.

Finish reading the input dataset before returning and return data that is
independent of subsequent changes to the input. This lets
[`export()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.export) resume replay-buffer writes after
conversion completes.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer
>>> from torchrl.data.datasets import ReplayDatasetExporter
>>> class ListExporter:
... def to_external(self, dataset):
... return [dataset[index].clone().to_dict() for index in range(len(dataset))]
>>> replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=2))
>>> _ = replay_buffer.extend(TensorDict({"observation": torch.arange(2)}, [2]))
>>> exporter: ReplayDatasetExporter[list[dict]] = ListExporter()
>>> rows = exporter.to_external(replay_buffer.storage.as_dataset())
>>> rows[1]["observation"]
tensor(1)
```

See also

[`ReplayDatasetImporter`](torchrl.data.datasets.ReplayDatasetImporter.html#torchrl.data.datasets.ReplayDatasetImporter) for importing,
[`ReplayDatasetAdapter`](torchrl.data.datasets.ReplayDatasetAdapter.html#torchrl.data.datasets.ReplayDatasetAdapter) for bidirectional
conversion, and `StorageDataset` for a replay
storage view accepted by `to_external()`.

to_external(*dataset: [Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, */*) → _ExternalDatasetT_co[[source]](../../_modules/torchrl/data/datasets/adapters.html#ReplayDatasetExporter.to_external)

Convert a TorchRL map-style dataset to a native external dataset.

Parameters:

**dataset** (*Dataset**[**TensorDictBase**]*) - finite map-style dataset with one
[`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) item per index.

Returns:

The dataset in the external library's native format.