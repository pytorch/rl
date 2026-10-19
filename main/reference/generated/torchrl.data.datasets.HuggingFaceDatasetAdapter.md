# HuggingFaceDatasetAdapter

*class*torchrl.data.datasets.HuggingFaceDatasetAdapter(***, *batch_size: int = 1024*)[[source]](../../_modules/torchrl/data/datasets/huggingface.html#HuggingFaceDatasetAdapter)

Convert between TorchRL map-style datasets and Hugging Face datasets.

Each item in the TorchRL dataset maps to exactly one Hugging Face row.
Items must be unbatched [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) instances. Nested
keys are preserved, and tensor leaves move to CPU for Arrow storage.
Import reconstructs an unbatched TensorDict using the declared Hugging
Face feature types for numeric tensor dtypes.

Export reads the source synchronously in batches and converts tensor
columns directly to Arrow arrays. Datasets with batched reads (including
TorchRL storage datasets) avoid fetching items individually. Other datasets
are stacked with [`tensordict_collate()`](torchrl.data.tensordict_collate.html#torchrl.data.tensordict_collate) before column
conversion. Each batch must be dense, with matching keys and tensor shapes
across its items. Tensor dtypes and column keys must match between all
rows, including nested keys. [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) entries
expand across their batch dimensions, including extra nested dimensions.
[`MetaData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.MetaData.html#tensordict.MetaData) entries are repeated once per output row;
[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) entries provide one value per row.
[`UnbatchedTensor`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor) entries must be converted to batched
tensors before export. Non-tensor payloads must be supported by Hugging Face
Datasets and Apache Arrow feature inference. Empty nested TensorDicts and
tensor shapes with dimensions following an empty dimension are rejected
because inferred Arrow lists cannot preserve their structure reliably.
Sparse, quantized, complex and float8 tensor leaves are not supported.
Bfloat16 tensors are converted to float32 for Arrow storage.

Import reads stored data without applying Hugging Face formatting or
transforms, leaving the caller's dataset unchanged. Numeric sequences
containing nulls remain [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) payloads, since
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
[Hugging Face datasets library](https://huggingface.co/docs/datasets).

Keyword Arguments:

**batch_size** (*int*) - maximum number of source items converted
at once. Reduce this for large observations to limit temporary
memory use. Defaults to 1024.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer
>>> from torchrl.data.datasets import HuggingFaceDatasetAdapter
>>> replay_buffer = ReplayBuffer(storage=LazyTensorStorage(3))
>>> _ = replay_buffer.extend(
... TensorDict({"observation": torch.arange(6).reshape(3, 2)}, [3])
... )
>>> adapter = HuggingFaceDatasetAdapter()
>>> hf_dataset = adapter.to_external(
... replay_buffer.storage.as_dataset()
... )
>>> imported = adapter.from_external(hf_dataset)
>>> imported[0]["observation"]
tensor([0, 1])
>>> hf_dataset.push_to_hub("organization/replay-data")
```

from_external(*dataset: Any*, */*) → [Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/data/datasets/huggingface.html#HuggingFaceDatasetAdapter.from_external)

Expose a Hugging Face dataset as a TorchRL map-style dataset.

Parameters:

**dataset** (*datasets.Dataset*) - finite Hugging Face dataset to adapt.
Select one split before passing a `DatasetDict` returned by
`datasets.load_dataset()`. Stored values are read without
applying active formatters or transforms.

Returns:

lazy decoding view with one TensorDict item

per Hugging Face row.

Return type:

Dataset[TensorDictBase]

to_external(*dataset: [Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, */*) → Any[[source]](../../_modules/torchrl/data/datasets/huggingface.html#HuggingFaceDatasetAdapter.to_external)

Convert a TorchRL map-style dataset to a Hugging Face dataset.

Parameters:

**dataset** (*Dataset**[**TensorDictBase**]*) - finite map-style dataset with one
unbatched TensorDict item per index. Items in each batch must
have matching keys and tensor shapes. Tensor dtypes must be
consistent across all rows.

Returns:

materialized Hugging Face dataset. Call

`datasets.Dataset.push_to_hub()` to publish it.

Return type:

datasets.Dataset