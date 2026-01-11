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

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   data_replaybuffers
   data_storage
   data_samplers
   data_datasets
   data_specs
