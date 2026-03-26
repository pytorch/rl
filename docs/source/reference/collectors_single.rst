.. currentmodule:: torchrl.collectors

Single Node Collectors
======================

TorchRL provides several collector classes for single-node data collection, each with different execution strategies.

Single node data collectors
---------------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    BaseCollector
    Collector
    AsyncCollector
    AsyncBatchedCollector
    MultiCollector
    MultiSyncCollector
    MultiAsyncCollector

.. note::
    The following legacy names are also available for backward compatibility:

    - ``DataCollectorBase`` → ``BaseCollector``
    - ``SyncDataCollector`` → ``Collector``
    - ``aSyncDataCollector`` → ``AsyncCollector``
    - ``_MultiDataCollector`` → ``MultiCollector``
    - ``MultiSyncDataCollector`` → ``MultiSyncCollector``
    - ``MultiaSyncDataCollector`` → ``MultiAsyncCollector``

Using AsyncBatchedCollector
---------------------------

The :class:`AsyncBatchedCollector` pairs an :class:`~torchrl.envs.AsyncEnvPool`
with an :class:`~torchrl.modules.InferenceServer` to pipeline environment
stepping and batched GPU inference.  You only need to supply **env factories**
and a **policy** -- all internal wiring is handled automatically:

.. code-block:: python

    from torchrl.collectors import AsyncBatchedCollector
    from torchrl.envs import GymEnv
    from tensordict.nn import TensorDictModule
    import torch.nn as nn

    policy = TensorDictModule(
        nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)),
        in_keys=["observation"],
        out_keys=["action"],
    )

    collector = AsyncBatchedCollector(
        create_env_fn=[lambda: GymEnv("CartPole-v1")] * 8,
        policy=policy,
        frames_per_batch=200,
        total_frames=10000,
        max_batch_size=8,
    )

    for data in collector:
        # data is a lazy-stacked TensorDict of collected transitions
        pass

    collector.shutdown()

**Key advantages over** :class:`Collector`:

- The inference server automatically **batches policy forward passes** from
  all environments, maximising GPU utilisation.
- Environment stepping and inference run in **overlapping fashion**, reducing
  idle time.
- Supports ``yield_completed_trajectories=True`` for episode-level yields.

Using MultiCollector
--------------------

The :class:`MultiCollector` class is the recommended way to run parallel data collection.
It uses a ``sync`` parameter to dispatch to either :class:`MultiSyncCollector` or :class:`MultiAsyncCollector`:

.. code-block:: python

    from torchrl.collectors import MultiCollector
    from torchrl.envs import GymEnv

    def make_env():
        return GymEnv("CartPole-v1")

    # Synchronous multi-worker collection (recommended for on-policy algorithms)
    sync_collector = MultiCollector(
        create_env_fn=[make_env] * 4,  # 4 parallel workers
        policy=my_policy,
        frames_per_batch=1000,
        total_frames=100000,
        sync=True,  # ← All workers complete before delivering batch
    )

    # Asynchronous multi-worker collection (recommended for off-policy algorithms)
    async_collector = MultiCollector(
        create_env_fn=[make_env] * 4,
        policy=my_policy,
        frames_per_batch=1000,
        total_frames=100000,
        sync=False,  # ← First-come-first-serve delivery
    )

    # Iterate over collected data
    for data in sync_collector:
        # Train on data...
        pass

    sync_collector.shutdown()

**Comparison:**

+------------------------+----------------------------------+----------------------------------+
| Feature                | ``sync=True``                    | ``sync=False``                   |
+========================+==================================+==================================+
| Batch delivery         | All workers complete first       | First available worker           |
+------------------------+----------------------------------+----------------------------------+
| Policy consistency     | All data from same policy version| Data may be from older policy    |
+------------------------+----------------------------------+----------------------------------+
| Best for               | On-policy (PPO, A2C)             | Off-policy (SAC, DQN)            |
+------------------------+----------------------------------+----------------------------------+
| Throughput             | Limited by slowest worker        | Higher throughput                |
+------------------------+----------------------------------+----------------------------------+

Running the Collector Asynchronously
------------------------------------

Passing replay buffers to a collector allows us to start the collection and get rid of the iterative nature of the
collector.
If you want to run a data collector in the background, simply run :meth:`~torchrl.collectors.BaseCollector.start`:

    >>> collector = Collector(..., replay_buffer=rb) # pass your replay buffer
    >>> collector.start()
    >>> # little pause
    >>> time.sleep(10)
    >>> # Start training
    >>> for i in range(optim_steps):
    ...     data = rb.sample()  # Sampling from the replay buffer
    ...     # rest of the training loop

Single-process collectors (:class:`~torchrl.collectors.Collector`) will run the process using multithreading,
so be mindful of Python's GIL and related multithreading restrictions.

Multiprocessed collectors will on the other hand let the child processes handle the filling of the buffer on their own,
which truly decouples the data collection and training.

Data collectors that have been started with `start()` should be shut down using
:meth:`~torchrl.collectors.BaseCollector.async_shutdown`.

.. warning:: Running a collector asynchronously decouples the collection from training, which means that the training
    performance may be drastically different depending on the hardware, load and other factors (although it is generally
    expected to provide significant speed-ups). Make sure you understand how this may affect your algorithm and if it
    is a legitimate thing to do! (For example, on-policy algorithms such as PPO should not be run asynchronously
    unless properly benchmarked).
