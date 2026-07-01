# CompressedListStorage

*class*torchrl.data.replay_buffers.CompressedListStorage(*max_size: int*, ***, *compression_fn: Callable | None = None*, *decompression_fn: Callable | None = None*, *compression_level: int = 3*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) = 'cpu'*, *compilable: bool = False*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#CompressedListStorage)

A storage that compresses and decompresses data.

This storage compresses data when storing and decompresses when retrieving.
It's particularly useful for storing raw sensory observations like images
that can be compressed significantly to save memory.

Parameters:

- **max_size** (*int*) - size of the storage, i.e. maximum number of elements stored
in the buffer.
- **compression_fn** (*callable**,**optional*) - function to compress data. Should take
a tensor and return a compressed byte tensor. Defaults to zstd compression.
- **decompression_fn** (*callable**,**optional*) - function to decompress data. Should take
a compressed byte tensor and return the original tensor. Defaults to zstd decompression.
- **compression_level** (*int**,**optional*) - compression level (1-22 for zstd) when using the default compression function.
Defaults to 3.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device where the sampled tensors will be
stored and sent. Default is `torch.device("cpu")`.
- **compilable** (*bool**,**optional*) - whether the storage is compilable.
If `True`, the writer cannot be shared between multiple processes.
Defaults to `False`.

Examples

```
>>> import torch
>>> from torchrl.data import CompressedListStorage, ReplayBuffer
>>> from tensordict import TensorDict
>>>
>>> # Create a compressed storage for image data
>>> storage = CompressedListStorage(max_size=1000, compression_level=3)
>>> rb = ReplayBuffer(storage=storage, batch_size=5)
>>>
>>> # Add some image data
>>> images = torch.randn(10, 3, 84, 84) # Atari-like frames
>>> data = TensorDict({"obs": images}, batch_size=[10])
>>> rb.extend(data)
>>>
>>> # Sample and verify data is decompressed correctly
>>> sample = rb.sample(3)
>>> print(sample["obs"].shape) # torch.Size([3, 3, 84, 84])
```

attach(*buffer: Any*) → None

This function attaches a sampler to this storage.

Buffers that read from this storage must be included as an attached
entity by calling this method. This guarantees that when data
in the storage changes, components are made aware of changes even if the storage
is shared with other buffers (eg. Priority Samplers).

Parameters:

**buffer** - the object that reads from this storage.

bytes()[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#CompressedListStorage.bytes)

Return the number of bytes in the storage.

dump(**args*, ***kwargs*)

Alias for `dumps()`.

load(**args*, ***kwargs*)

Alias for `loads()`.

load_state_dict(*state_dict: dict[str, Any]*) → None[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#CompressedListStorage.load_state_dict)

Load the storage state.

register_load_hook(*hook*)

Register a load hook for this storage.

The hook is forwarded to the checkpointer.

register_save_hook(*hook*)

Register a save hook for this storage.

The hook is forwarded to the checkpointer.

save(**args*, ***kwargs*)

Alias for `dumps()`.

state_dict() → dict[str, Any][[source]](../../_modules/torchrl/data/replay_buffers/storages.html#CompressedListStorage.state_dict)

Save the storage state.

to_bytestream(*data_to_bytestream: [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | np.array | Any*) → bytes[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#CompressedListStorage.to_bytestream)

Convert data to a byte stream.