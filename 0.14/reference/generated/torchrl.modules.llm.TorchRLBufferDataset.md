# TorchRLBufferDataset

*class*torchrl.modules.llm.TorchRLBufferDataset(*replay_buffer: [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*, *batch_size: int*, ***, *keys: list[NestedKey] | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *num_batches: int | None = 1*)[[source]](../../_modules/torchrl/modules/llm/trl_interop.html#TorchRLBufferDataset)

An [`torch.utils.data.IterableDataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) backed by a TorchRL [`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer).

The PyTorch dataset can be consumed directly by
`transformers.Trainer`. Trainers such as `trl.GRPOTrainer`
that require a Hugging Face `datasets.IterableDataset` can consume
the object returned by `as_hf_dataset()`.

Each sampling call draws `batch_size` entries from the replay buffer and
yields them individually as flat `dict[str, Any]` objects. By default an
iterator samples one replay batch. Set `num_batches=None` for an
unbounded online stream; consumers of such a stream must impose their own
step limit.

Note

This class implements [`torch.utils.data.IterableDataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) (no
`__len__`), which is the safest choice for online / infinite replay
buffers. If you need a finite dataset with a known length, iterate
for a fixed number of steps yourself and collect the results.

Parameters:

- **replay_buffer** ([`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)) - the TorchRL
replay buffer to wrap.
- **batch_size** (*int*) - number of samples to draw from the buffer per
internal sampling call. Each yielded item is one *individual*
sample (no leading batch dimension).

Keyword Arguments:

- **keys** (list of `NestedKey`, optional) - if provided,
only these keys are included in the yielded dicts. Nested keys
are serialised as `"key0.key1"` strings so they remain
compatible with HuggingFace collators. Defaults to `None`
(all leaf keys, with nested keys flattened).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - if provided, all tensors are
moved to this device before yielding. Defaults to `None`
(tensors stay on their current device).
- **num_batches** (*int**or**None**,**optional*) - number of replay batches sampled
by each iterator. `None` produces an unbounded stream. Defaults
to `1`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import ReplayBuffer, ListStorage
>>> from torchrl.modules.llm.trl_interop import TorchRLBufferDataset
>>>
>>> rb = ReplayBuffer(storage=ListStorage(100), batch_size=4)
>>> for _ in range(10):
... _ = rb.add(TensorDict(
... {"input_ids": torch.randint(0, 100, (8,)),
... "attention_mask": torch.ones(8, dtype=torch.long)},
... batch_size=[],
... ))
>>>
>>> dataset = TorchRLBufferDataset(rb, batch_size=4)
>>> sample = next(iter(dataset))
>>> sample["input_ids"].shape
torch.Size([8])
```

See also

[`HFRewardModelWrapper`](torchrl.modules.llm.HFRewardModelWrapper.html#torchrl.modules.llm.HFRewardModelWrapper) for the reverse direction (TRL -> TorchRL).

as_hf_dataset() → Any[[source]](../../_modules/torchrl/modules/llm/trl_interop.html#TorchRLBufferDataset.as_hf_dataset)

Return a Hugging Face iterable dataset backed by this adapter.

The returned object is accepted by current `trl` trainers, which
require `datasets.Dataset` or
`datasets.IterableDataset` rather than a PyTorch iterable
dataset. The replay samples must still contain the schema required by
the selected trainer, such as a top-level `"prompt"` field for
`trl.GRPOTrainer`.

Returns:

A `datasets.IterableDataset` that yields the same samples
as this adapter.

Raises:

**ImportError** - if the optional `datasets` package is unavailable.