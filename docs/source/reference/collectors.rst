.. currentmodule:: torchrl.collectors

torchrl.collectors package
==========================

.. _ref_collectors:

Data collectors are the bridge between your environments and training loop, managing the process of gathering
experience data using your policy. They handle environment resets, policy execution, and data aggregation,
making it easy to collect high-quality training data efficiently.

TorchRL provides several collector implementations optimized for different scenarios:

- :class:`Collector`: Single-process collection on the training worker
- :class:`AsyncBatchedCollector`: Async environments + auto-batching inference server (see :class:`AsyncBatchedCollector`)
- :class:`MultiCollector`: Parallel collection across multiple workers (see below)
- :class:`Evaluator`: Sync or async evaluation during training (see :ref:`evaluation <collectors_eval>`)
- **Distributed collectors**: For multi-node setups using Ray, RPC, or distributed backends (see :class:`DistributedCollector` / :class:`RPCCollector`)

MultiCollector API
------------------

The :class:`MultiCollector` class provides a unified interface for multi-process data collection.
Use the ``sync`` parameter to choose between synchronous and asynchronous collection:

.. code-block:: python

    from torchrl.collectors import MultiCollector

    # Synchronous collection: all workers complete before delivering batch
    collector = MultiCollector(
        create_env_fn=[make_env] * 4,  # 4 parallel workers
        policy=my_policy,
        frames_per_batch=200,
        total_frames=10000,
        sync=True,  # synchronized delivery
    )

    # Asynchronous collection: first-come-first-serve delivery
    collector = MultiCollector(
        create_env_fn=[make_env] * 4,
        policy=my_policy,
        frames_per_batch=200,
        total_frames=10000,
        sync=False,  # async delivery (faster, but policy may lag)
    )

**When to use sync vs async:**

- ``sync=True``: Use for on-policy algorithms (PPO, A2C) where data must match current policy
- ``sync=False``: Use for off-policy algorithms (SAC, DQN) where slight policy lag is acceptable

Key Features
------------

- **Flexible execution**: Choose between sync, async, and distributed collection
- **Device management**: Control where environments and policies execute
- **Weight synchronization**: Keep inference policies up-to-date with training weights
- **Replay buffer integration**: Seamless compatibility with TorchRL's replay buffers
- **Trajectory assembly**: Collect complete trajectories with ``trajs_per_batch`` for
  clean :class:`~torchrl.data.SliceSampler` sampling — see :ref:`collectors_replay_trajs`
- **Batching strategies**: Multiple ways to organize collected data

Collection hooks
----------------

Collectors accept optional hooks for per-rollout side effects:
``pre_collect_hook`` is called before a rollout starts, and
``post_collect_hook`` is called with the :class:`~tensordict.TensorDictBase`
batch that will be yielded by iteration.  Hook return values are ignored, and
exceptions raised by hooks propagate to the caller and stop collection.

Hooks are intended for instrumentation and worker-local side effects, such as
stepping a profiler or recording rollout metrics.  Use ``postproc`` when the
collected data itself should be transformed before training.

For :class:`MultiCollector`, :class:`MultiSyncCollector`, and
:class:`MultiAsyncCollector`, hooks run in each worker process.  The helper
methods :meth:`~torchrl.collectors.BaseCollector.map_fn` and
:meth:`~torchrl.collectors.BaseCollector.get_distant_attr` broadcast to each
worker for multi-process collectors and to each actor for
:class:`~torchrl.collectors.distributed.RayCollector`.

Quick Example
-------------

.. code-block:: python

    from torchrl.collectors import Collector
    from torchrl.envs import GymEnv, ParallelEnv
    
    # Create a batched environment
    def make_env():
        return GymEnv("Pendulum-v1")
    
    env = ParallelEnv(4, make_env)
    
    # Create collector
    collector = Collector(
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

Legacy names
------------

The following names are kept for backward compatibility:

- ``SyncDataCollector`` → ``Collector``
- ``MultiSyncDataCollector`` → ``MultiCollector(sync=True)``
- ``MultiaSyncDataCollector`` → ``MultiCollector(sync=False)``
- ``DataCollectorBase`` → ``BaseCollector``

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   collectors_basics
   collectors_single
   collectors_eval
   collectors_distributed
   collectors_weightsync
   collectors_replay
