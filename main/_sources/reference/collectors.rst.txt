.. currentmodule:: torchrl.collectors

torchrl.collectors package
==========================

.. _ref_collectors:

Data collectors are the bridge between your environments and training loop, managing the process of gathering
experience data using your policy. They handle environment resets, policy execution, and data aggregation,
making it easy to collect high-quality training data efficiently.

TorchRL provides several collector implementations optimized for different scenarios:

- **SyncDataCollector**: Single-process collection on the training worker
- **MultiSyncDataCollector**: Parallel collection across multiple workers with synchronized delivery
- **MultiaSyncDataCollector**: Asynchronous collection with first-come-first-serve delivery
- **Distributed collectors**: For multi-node setups using Ray, RPC, or distributed backends

Key Features
------------

- **Flexible execution**: Choose between sync, async, and distributed collection
- **Device management**: Control where environments and policies execute
- **Weight synchronization**: Keep inference policies up-to-date with training weights
- **Replay buffer integration**: Seamless compatibility with TorchRL's replay buffers
- **Batching strategies**: Multiple ways to organize collected data

Quick Example
-------------

.. code-block:: python

    from torchrl.collectors import SyncDataCollector
    from torchrl.envs import GymEnv, ParallelEnv
    
    # Create a batched environment
    def make_env():
        return GymEnv("Pendulum-v1")
    
    env = ParallelEnv(4, make_env)
    
    # Create collector
    collector = SyncDataCollector(
        env,
        policy=my_policy,
        frames_per_batch=200,
        total_frames=10000,
    )
    
    # Collect data
    for data in collector:
        # data is a TensorDict with shape [4, 50] (4 envs, 50 steps each)
        # Use data for training...
        
        # Update policy weights periodically
        if should_update:
            collector.update_policy_weights_()
    
    collector.shutdown()

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   collectors_basics
   collectors_single
   collectors_distributed
   collectors_weightsync
   collectors_replay
