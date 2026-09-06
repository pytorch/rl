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
