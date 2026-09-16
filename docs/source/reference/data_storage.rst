.. currentmodule:: torchrl.data.replay_buffers

.. _checkpoint-rb:

Storage Backends
================

TorchRL provides various storage backends for replay buffers, each optimized for different use cases.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    CompressedListStorage
    CompressedListStorageCheckpointer
    FlatStorageCheckpointer
    H5StorageCheckpointer
    ImmutableDatasetWriter
    LazyMemmapStorage
    LazyTensorStorage
    ListStorage
    LazyStackStorage
    ListStorageCheckpointer
    NestedStorageCheckpointer
    Storage
    StorageCheckpointerBase
    StorageEnsemble
    StorageEnsembleCheckpointer
    TensorStorage
    TensorStorageCheckpointer

Storage Performance
-------------------

Storage choice is very influential on replay buffer sampling latency, especially
in distributed reinforcement learning settings with larger data volumes.
:class:`~torchrl.data.replay_buffers.LazyMemmapStorage` is highly
advised in distributed settings with shared storage due to the lower serialization
cost of MemoryMappedTensors as well as the ability to specify file storage locations
for improved node failure recovery.

Storages as torch datasets
--------------------------

Every storage is a :class:`torch.utils.data.Dataset`, so a
:class:`torch.utils.data.DataLoader` can read it with any torch sampler and
worker processes. Index batches are fetched with a single
:meth:`~torchrl.data.replay_buffers.Storage.get` call on the flattened storage
and collated as the buffer would collate them; pass
:func:`~torchrl.data.tensordict_collate` as ``collate_fn`` to keep the batch
intact. A :class:`~torchrl.data.replay_buffers.StorageEnsemble` is not a flat
dataset and rejects this:

    >>> import torch
    >>> from tensordict import TensorDict
    >>> from torch.utils.data import DataLoader
    >>> from torchrl.data import LazyMemmapStorage, ReplayBuffer, tensordict_collate
    >>> rb = ReplayBuffer(storage=LazyMemmapStorage(100))
    >>> rb.extend(TensorDict({"obs": torch.arange(100)}, [100]))
    >>> loader = DataLoader(rb.storage, batch_size=4, shuffle=True, num_workers=2, collate_fn=tensordict_collate)
    >>> next(iter(loader))["obs"].shape
    torch.Size([4])

See :meth:`~torchrl.data.ReplayBuffer.as_dataset` to keep the TorchRL sampler
and transforms while sampling in DataLoader workers.
