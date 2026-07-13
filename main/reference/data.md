# torchrl.data package

TorchRL provides a comprehensive data management system built around replay buffers, which are central to
off-policy RL algorithms. The library offers efficient implementations of various replay buffers with
composable components for storage, sampling, and data transformation.

## Key Features

- **Flexible storage backends**: Memory, memmap, and compressed storage options
- **Advanced sampling strategies**: Prioritized, slice-based, and custom samplers
- **Composable design**: Mix and match storage, samplers, and writers
- **Type flexibility**: Support for tensors, tensordicts, and arbitrary data types
- **Efficient transforms**: Apply preprocessing during sampling
- **Distributed support**: Ray-based and remote replay buffers

## Quick Example

```
import torch
from torchrl.data import ReplayBuffer, LazyMemmapStorage, PrioritizedSampler
from tensordict import TensorDict

# Create a replay buffer with memmap storage and prioritized sampling
buffer = ReplayBuffer(
 storage=LazyMemmapStorage(max_size=1000000),
 sampler=PrioritizedSampler(max_capacity=1000000, alpha=0.7, beta=0.5),
 batch_size=256,
)

# Add data
data = TensorDict({
 "observation": torch.randn(32, 4),
 "action": torch.randn(32, 2),
 "reward": torch.randn(32, 1),
}, batch_size=[32])
buffer.extend(data)

# Sample
sample = buffer.sample() # Returns batch_size=256
```

## CUDA prioritized replay buffers

Prioritized replay buffers can keep the priority trees on CPU or CUDA
independently from the data storage. By default, CUDA tensor storage selects a
CUDA sampler and CPU storage selects a CPU sampler. Use `sampler_device` when
the storage and priority sampler should live on different devices.

All-CUDA storage and sampling

```
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

rb = TensorDictPrioritizedReplayBuffer(
 alpha=0.7,
 beta=0.5,
 storage=LazyTensorStorage(1_000_000, device="cuda"),
 batch_size=65_536,
 priority_key="td_error",
)

data = TensorDict(
 {
 "obs": torch.randn(100_000, 32, device="cuda"),
 "td_error": torch.ones(100_000, device="cuda"),
 },
 batch_size=[100_000],
 device="cuda",
)
rb.extend(data)
sample = rb.sample()
sample["td_error"] = torch.rand(sample.shape, device="cuda")
rb.update_tensordict_priority(sample)
```

CPU memmap storage with CUDA priority sampling

```
import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer

rb = TensorDictPrioritizedReplayBuffer(
 alpha=0.7,
 beta=0.5,
 storage=LazyMemmapStorage(10_000_000, scratch_dir="/tmp/torchrl_rb"),
 sampler_device="cuda",
 batch_size=65_536,
 priority_key="td_error",
)

data = TensorDict(
 {
 "obs": torch.randn(100_000, 32),
 "td_error": torch.ones(100_000),
 },
 batch_size=[100_000],
)
rb.extend(data)
sample = rb.sample()
sample["td_error"] = torch.rand(sample.shape)
rb.update_tensordict_priority(sample)
```

CUDA storage with CPU priority sampling

```
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

rb = TensorDictPrioritizedReplayBuffer(
 alpha=0.7,
 beta=0.5,
 storage=LazyTensorStorage(1_000_000, device="cuda"),
 sampler_device="cpu",
 batch_size=65_536,
 priority_key="td_error",
)

data = TensorDict(
 {
 "obs": torch.randn(100_000, 32, device="cuda"),
 "td_error": torch.ones(100_000, device="cuda"),
 },
 batch_size=[100_000],
 device="cuda",
)
rb.extend(data)
sample = rb.sample()
sample["td_error"] = torch.rand(sample.shape, device="cuda")
rb.update_tensordict_priority(sample)
```

## Documentation Sections

- [Replay Buffers](data_replaybuffers.html)

- [Core Replay Buffer Classes](data_replaybuffers.html#core-replay-buffer-classes)
- [Offline-to-online helpers](data_replaybuffers.html#offline-to-online-helpers)
- [Trajectory queries](data_replaybuffers.html#trajectory-queries)
- [Composable Replay Buffers](data_replaybuffers.html#composable-replay-buffers)
- [Storage Backends](data_storage.html)

- [CompressedListStorage](generated/torchrl.data.replay_buffers.CompressedListStorage.html)
- [CompressedListStorageCheckpointer](generated/torchrl.data.replay_buffers.CompressedListStorageCheckpointer.html)
- [FlatStorageCheckpointer](generated/torchrl.data.replay_buffers.FlatStorageCheckpointer.html)
- [H5StorageCheckpointer](generated/torchrl.data.replay_buffers.H5StorageCheckpointer.html)
- [ImmutableDatasetWriter](generated/torchrl.data.replay_buffers.ImmutableDatasetWriter.html)
- [LazyMemmapStorage](generated/torchrl.data.replay_buffers.LazyMemmapStorage.html)
- [LazyTensorStorage](generated/torchrl.data.replay_buffers.LazyTensorStorage.html)
- [ListStorage](generated/torchrl.data.replay_buffers.ListStorage.html)
- [LazyStackStorage](generated/torchrl.data.replay_buffers.LazyStackStorage.html)
- [ListStorageCheckpointer](generated/torchrl.data.replay_buffers.ListStorageCheckpointer.html)
- [NestedStorageCheckpointer](generated/torchrl.data.replay_buffers.NestedStorageCheckpointer.html)
- [Storage](generated/torchrl.data.replay_buffers.Storage.html)
- [StorageCheckpointerBase](generated/torchrl.data.replay_buffers.StorageCheckpointerBase.html)
- [StorageEnsemble](generated/torchrl.data.replay_buffers.StorageEnsemble.html)
- [StorageEnsembleCheckpointer](generated/torchrl.data.replay_buffers.StorageEnsembleCheckpointer.html)
- [TensorStorage](generated/torchrl.data.replay_buffers.TensorStorage.html)
- [TensorStorageCheckpointer](generated/torchrl.data.replay_buffers.TensorStorageCheckpointer.html)
- [Storage Performance](data_storage.html#storage-performance)
- [Sampling Strategies](data_samplers.html)

- [PrioritizedSampler](generated/torchrl.data.replay_buffers.PrioritizedSampler.html)
- [PrioritizedSliceSampler](generated/torchrl.data.replay_buffers.PrioritizedSliceSampler.html)
- [PromptGroupSampler](generated/torchrl.data.replay_buffers.PromptGroupSampler.html)
- [ConsumingSampler](generated/torchrl.data.replay_buffers.ConsumingSampler.html)
- [RandomSampler](generated/torchrl.data.replay_buffers.RandomSampler.html)
- [Sampler](generated/torchrl.data.replay_buffers.Sampler.html)
- [SamplerEnsemble](generated/torchrl.data.replay_buffers.SamplerEnsemble.html)
- [SamplerWithoutReplacement](generated/torchrl.data.replay_buffers.SamplerWithoutReplacement.html)
- [SliceSampler](generated/torchrl.data.replay_buffers.SliceSampler.html)
- [SliceSamplerWithoutReplacement](generated/torchrl.data.replay_buffers.SliceSamplerWithoutReplacement.html)
- [StalenessAwareSampler](generated/torchrl.data.replay_buffers.StalenessAwareSampler.html)
- [Writers](data_samplers.html#writers)
- [Datasets](data_datasets.html)

- [TorchRL Episode Data (TED) Format](data_datasets.html#torchrl-episode-data-ted-format)
- [Dataset loading registry](data_datasets.html#dataset-loading-registry)
- [TensorSpec System](data_specs.html)

- [Binary](generated/torchrl.data.Binary.html)
- [Bounded](generated/torchrl.data.Bounded.html)
- [Categorical](generated/torchrl.data.Categorical.html)
- [Composite](generated/torchrl.data.Composite.html)
- [MultiCategorical](generated/torchrl.data.MultiCategorical.html)
- [MultiOneHot](generated/torchrl.data.MultiOneHot.html)
- [NonTensor](generated/torchrl.data.NonTensor.html)
- [OneHot](generated/torchrl.data.OneHot.html)
- [Stacked](generated/torchrl.data.Stacked.html)
- [StackedComposite](generated/torchrl.data.StackedComposite.html)
- [TensorSpec](generated/torchrl.data.TensorSpec.html)
- [Unbounded](generated/torchrl.data.Unbounded.html)
- [UnboundedContinuous](generated/torchrl.data.UnboundedContinuous.html)
- [UnboundedDiscrete](generated/torchrl.data.UnboundedDiscrete.html)
- [Supported PyTorch Operations](data_specs.html#supported-pytorch-operations)