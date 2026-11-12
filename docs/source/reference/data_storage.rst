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
    StorageDataset
    StorageEnsemble
    StorageEnsembleCheckpointer
    TensorStorage
    TensorStorageCheckpointer

HDF5 checkpoint loading
-----------------------

:class:`H5StorageCheckpointer` follows the installed TensorDict version's
:meth:`~tensordict.PersistentTensorDict.from_h5` default when
``auto_batch_size=None`` (the default), including its metadata warnings.
TensorDict announces a change in 0.17 to restoring saved batch sizes and dimension
names. That change will apply when TorchRL's supported TensorDict range includes
0.17; this option does not bypass the dependency constraints.

To explicitly request batch-size inference when loading a checkpoint, configure
the destination buffer's checkpointer:

.. code-block:: python

    from torchrl.data import H5StorageCheckpointer, LazyTensorStorage, ReplayBuffer

    restored = ReplayBuffer(storage=LazyTensorStorage(100))
    restored.storage.checkpointer = H5StorageCheckpointer(auto_batch_size=True)
    restored.loads("replay-checkpoint")

Pass ``auto_batch_size=False`` to explicitly disable the additional inference
step in ``from_h5``. This is not a request to restore saved metadata before
TensorDict implements that behavior. The inference option affects loading only;
``h5_kwargs`` continues to configure HDF5 dataset creation when saving.

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

:meth:`~torchrl.data.replay_buffers.Storage.as_dataset` returns a map-style
:class:`torch.utils.data.Dataset`, a :class:`StorageDataset`, that a
:class:`torch.utils.data.DataLoader` reads with any torch sampler and
worker processes, fetching each index batch with a single
:meth:`~torchrl.data.replay_buffers.Storage.get` call. Pass
:func:`~torchrl.data.tensordict_collate` as ``collate_fn``, read
multi-dimensional storages through
:meth:`~torchrl.data.replay_buffers.Storage.flatten`, and see
:ref:`ref_buffers` for reading whole buffers through their sampler and
transforms.
