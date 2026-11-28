.. currentmodule:: torchrl.data

torchrl.data package
====================

.. _ref_data:

TorchRL provides a comprehensive data management system built around replay buffers, which are central to
off-policy RL algorithms. The library offers efficient implementations of various replay buffers with
composable components for storage, sampling, and data transformation.

Key Features
------------

- **Flexible storage backends**: Memory, memmap, and compressed storage options
- **Advanced sampling strategies**: Prioritized, slice-based, and custom samplers
- **Composable design**: Mix and match storage, samplers, and writers
- **Type flexibility**: Support for tensors, tensordicts, and arbitrary data types
- **Efficient transforms**: Apply preprocessing during sampling
- **Distributed support**: Ray-based and remote replay buffers

Quick Example
-------------

.. code-block:: python

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
    sample = buffer.sample()  # Returns batch_size=256

CUDA prioritized replay buffers
-------------------------------

Prioritized replay buffers can keep the priority trees on CPU or CUDA
independently from the data storage. By default, CUDA tensor storage selects a
CUDA sampler and CPU storage selects a CPU sampler. Use ``sampler_device`` when
the storage and priority sampler should live on different devices.

.. dropdown:: All-CUDA storage and sampling
   :icon: code

   .. code-block:: python

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

.. dropdown:: CPU memmap storage with CUDA priority sampling
   :icon: code

   .. code-block:: python

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

.. dropdown:: CUDA storage with CPU priority sampling
   :icon: code

   .. code-block:: python

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

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   data_replaybuffers
   data_storage
   data_samplers
   data_datasets
   data_specs


Mixed-source replay
-------------------

ReplayBufferEnsemble can serve RLPD, offline-to-online and
HIL-SERL style training from one buffer per data source:

- fractions sets the share of each batch taken from each member. Every batch
  gets the same counts, and the batch is flat, one row per record.
- Members may use different samplers. Rows from a uniform member next to a
  prioritized one get an importance weight of 1.
- routing_mode="mask" writes a record to several members at once (HIL-SERL
  stores intervened steps in both buffers) and returns the address of every
  copy, so one update_if_present call can fix all copies of a record.
- update_tensordict_priority sends TD errors back to the member each row came
  from.

.. code-block:: python

    demo = TensorDictReplayBuffer(
        storage=LazyTensorStorage(50_000),
        sampler=PrioritizedSampler(50_000, alpha=0.6, beta=0.4),
        writer=TensorDictRoundRobinWriter(track_generations=True),
    )
    online = TensorDictReplayBuffer(
        storage=LazyTensorStorage(1_000_000),
        writer=TensorDictRoundRobinWriter(track_generations=True),
    )
    rb = ReplayBufferEnsemble(
        demo, online, routing_key="route", routing_mode="mask",
        fractions=[0.5, 0.5], batch_size=256,
    )
    intervention = data["intervention"].squeeze(-1)
    data.set("route", torch.stack([intervention, torch.ones_like(intervention)], -1))
    addresses = rb.extend(data)
    batch = rb.sample()
    batch.set("td_error", loss(batch)["td_error"])
    rb.update_tensordict_priority(batch)

With fractions set, each member's share is fixed, so priorities only change
sampling inside a member. A member that receives only part of each trajectory
(the demo member above) has gaps in it. If it is sampled in slices, use a
SliceSampler with fragmented=True.

Ending replay streams on restart
--------------------------------

:meth:`~torchrl.data.ReplayBufferEnsemble.end_streams` closes each member's current
trajectory before restarted environments append new data. It drains pending
replay work, preserves row generations and terminal flags, and resets incomplete
fresh windows and cached uniform boundaries. Call it with collection paused and
before resuming producers. See its API documentation for supported member storage,
writer and sampler combinations and configurable nested boundary keys.
