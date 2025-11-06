.. currentmodule:: torchrl.data.replay_buffers

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
    StorageEnsembleCheckpointer
    TensorStorage
    TensorStorageCheckpointer

Storage Performance
-------------------

Storage choice is very influential on replay buffer sampling latency, especially
in distributed reinforcement learning settings with larger data volumes.
:class:`~torchrl.data.replay_buffers.storages.LazyMemmapStorage` is highly
advised in distributed settings with shared storage due to the lower serialization
cost of MemoryMappedTensors as well as the ability to specify file storage locations
for improved node failure recovery.
