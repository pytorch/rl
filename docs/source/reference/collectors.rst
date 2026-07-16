.. currentmodule:: torchrl.collectors

torchrl.collectors package
==========================

.. _ref_collectors:

Data collectors are the bridge between your environments and training loop, managing the process of gathering
experience data using your policy. They handle environment resets, policy execution, and data aggregation,
making it easy to collect high-quality training data efficiently.

Use :class:`Collector` to construct collectors in new code. It is the stable
front door for local, multi-process, and distributed collection; changing the
execution topology does not require changing the imported class. TorchRL also
exposes the concrete implementations for type checks, subclassing, and
implementation-specific integrations:

- :class:`Collector`: Main construction API, with direct collection as its
  default
- :class:`AsyncBatchedCollector`: Async environments + auto-batching inference server (see :class:`AsyncBatchedCollector`)
- :class:`MultiCollector`: Parallel collection across multiple workers (see below)
- :class:`Evaluator`: Sync or async evaluation during training (see :ref:`evaluation <collectors_eval>`)
- **Distributed collectors**: For multi-node setups using Ray, RPC, or distributed backends (see :class:`~torchrl.collectors.distributed.DistributedCollector` / :class:`~torchrl.collectors.distributed.RPCCollector`)

Backend selection
-----------------

:class:`Collector` constructs the appropriate concrete collector without
changing the training code that consumes it. The default is direct collection
in the training process. Passing ``num_collectors`` selects a process
collector, while ``backend`` selects a backend explicitly:

.. code-block:: python

    from torchrl.collectors import Collector

    process_collector = Collector(
        create_env_fn=make_env,
        num_collectors=4,
        policy=my_policy,
        frames_per_batch=200,
        total_frames=10_000,
        sync=True,
    )

    ray_collector = Collector(
        create_env_fn=make_env,
        num_collectors=4,
        backend="ray",
        backend_options={
            "remote_configs": {"num_cpus": 1, "num_gpus": 0},
        },
        policy=my_policy,
        frames_per_batch=200,
        total_frames=10_000,
    )

The available selectors are ``"direct"``, ``"process"``, ``"ray"``,
``"rpc"``, ``"distributed"``, and ``"submitit"``. ``"submitit"`` is a
shortcut for a distributed collector with ``launcher="submitit"``. For every
non-direct backend, omitted ``sync`` defaults to ``False``.

.. important::

   Process and distributed selection is asynchronous when ``sync`` is omitted.
   On-policy algorithms should normally pass ``sync=True`` explicitly so every
   worker contributes to each synchronized batch.

Selection precedence is explicit ``backend``, then an enclosing
:func:`torchrl.service_backend`, then ``num_collectors`` implying
``"process"``, and finally ``"direct"``. An explicit ``backend="direct"``
accepts at most one collector.

.. list-table:: Collector construction
   :header-rows: 1

   * - User-facing selection
     - Concrete result
     - Typical use
   * - ``Collector(...)`` or ``backend="direct"``
     - :class:`Collector`
     - One collector in the training process
   * - ``num_collectors=N`` or ``backend="process"``
     - :class:`MultiCollector`
     - Multiple local worker processes
   * - ``backend="ray"``
     - :class:`~torchrl.collectors.distributed.RayCollector`
     - Ray-managed actors
   * - ``backend="rpc"``
     - :class:`~torchrl.collectors.distributed.RPCCollector`
     - PyTorch RPC workers
   * - ``backend="distributed"``
     - :class:`~torchrl.collectors.distributed.DistributedCollector`
     - Explicit distributed launcher and process-group configuration
   * - ``backend="submitit"``
     - :class:`~torchrl.collectors.distributed.DistributedCollector`
     - Submitit launcher shortcut

``backend_options`` forwards backend-specific options without mutating the
input mapping. This is where launcher options, Ray resources, and the inner
Gloo or NCCL ``backend`` are configured. Selector arguments cannot be repeated
inside ``backend_options``.

A callable environment factory is replicated to ``num_collectors`` workers.
When a sequence is provided, its length determines the number of collectors;
an explicitly supplied positive count must match. Empty sequences,
non-positive counts, and mismatches are rejected before construction. The
returned object keeps its concrete type, so a process selection returns
:class:`MultiCollector` rather than an instance of the direct
:class:`Collector`. Use :class:`BaseCollector` for an ``isinstance`` check that
must accept every collector returned by the unified constructor.

Selection can also be scoped with :func:`torchrl.service_backend`; see
:ref:`ref_services_workflow` for composing collector placement with replay and
inference transports.

Process collection
------------------

Pass ``num_collectors`` to the main :class:`Collector` entry point for local
multi-process collection. Use ``sync`` to choose synchronous or asynchronous
delivery:

.. code-block:: python

    from torchrl.collectors import Collector

    # Synchronous collection: all workers complete before delivering batch
    collector = Collector(
        create_env_fn=make_env,
        num_collectors=4,
        policy=my_policy,
        frames_per_batch=200,
        total_frames=10000,
        sync=True,  # synchronized delivery
    )

    # Asynchronous collection: first-come-first-serve delivery
    collector = Collector(
        create_env_fn=make_env,
        num_collectors=4,
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
- **Trajectory assembly**: Collect complete trajectories with ``trajs_per_batch`` —
  padded whole-episode batches for on-policy training, or flat unpadded writes into a
  replay buffer for clean :class:`~torchrl.data.replay_buffers.SliceSampler` sampling —
  see :ref:`collectors_replay_trajs`
- **Batching strategies**: Multiple ways to organize collected data
- **Profiler-ready**: Set ``TORCHRL_PROFILING=1`` to emit named ranges on the
  collector, env, and policy hot paths — see :ref:`ref_profiling`

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

When :class:`Collector` selects the process backend, hooks run in each worker
process. The concrete return types are :class:`MultiSyncCollector` and
:class:`MultiAsyncCollector`. The helper
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

Removed legacy names
--------------------

The deprecated collector aliases were removed in v0.13. Use ``Collector`` as
the construction entry point. ``MultiCollector``, ``MultiSyncCollector``,
``MultiAsyncCollector``, and ``BaseCollector`` remain available when code must
name a concrete implementation.

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   collectors_basics
   collectors_single
   collectors_internals
   collectors_eval
   collectors_distributed
   collectors_weightsync
   collectors_replay
