# Transition

*class*torchrl.data.Transition[[source]](../../_modules/torchrl/data/replay_buffers/sample_units.html#Transition)

The identity sample unit: every anchor is one transition.

This unit reproduces the classic replay-buffer behavior exactly and is
the implicit default when no `sample_unit` is passed to the buffer:
anchors selected by the sampler are the records of the batch, and the
info dictionary is returned untouched.

See also

`TransitionConfig`
for the Hydra configuration companion.

Examples

```
>>> import torch
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer
>>> from torchrl.data.replay_buffers import Transition
>>> rb = ReplayBuffer(
... storage=LazyTensorStorage(10),
... batch_size=4,
... sample_unit=Transition(),
... )
>>> rb.extend(torch.arange(10))
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> sample = rb.sample()
>>> sample.shape
torch.Size([4])
```

expand(*index: [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | tuple*, *info: dict[str, Any]*, *storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*) → tuple[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | tuple, dict[str, Any]][[source]](../../_modules/torchrl/data/replay_buffers/sample_units.html#Transition.expand)

Expands anchor indices into the final record indices of the batch.

Parameters:

- **index** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**tuple**of*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the anchor indices
selected by the sampler.
- **info** (*dict*) - the sampler's info dictionary.
- **storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)) - the storage the batch will be read from.

Returns:

A tuple `(index, info)` with the expanded indices and the
(possibly augmented) info dictionary.