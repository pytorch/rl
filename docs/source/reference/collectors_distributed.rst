.. currentmodule:: torchrl.collectors.distributed

Distributed Collectors
======================

Construct distributed collectors through
:class:`torchrl.collectors.Collector`, the same entry point used for direct and
local process collection. Its ``backend`` selector covers Ray, PyTorch RPC,
and process-group-based collection; ``backend_options`` carries the concrete
backend's launcher, resources, and communication settings:

.. code-block:: python

    from torchrl.collectors import Collector

    ray_collector = Collector(
        make_env,
        policy,
        backend="ray",
        num_collectors=8,
        frames_per_batch=1024,
        backend_options={"remote_configs": {"num_cpus": 1}},
    )
    rpc_collector = Collector(
        make_env,
        policy,
        backend="rpc",
        num_collectors=8,
        frames_per_batch=1024,
        backend_options={"launcher": "submitit"},
    )
    distributed_collector = Collector(
        make_env,
        policy,
        backend="distributed",
        num_collectors=8,
        frames_per_batch=1024,
        backend_options={"launcher": "mp", "backend": "gloo"},
    )

The concrete implementations support Gloo, NCCL, MPI, or UCC with
:class:`~.DistributedCollector`, PyTorch RPC with :class:`~.RPCCollector`, and
Ray actor placement with :class:`~.RayCollector`. They can run synchronously
or asynchronously on a single node or across multiple nodes. Name the
concrete classes directly only when subclassing or working with an
implementation-specific API.

``Collector(backend="ray")`` uses Ray to place and control its worker actors,
but an
attached inference server or replay buffer selects its own payload transport.
With a replay buffer attached, workers write rollouts directly through the
replay endpoint instead of returning them to the driver. Without one, yielded
rollouts use Ray's object store. See :ref:`ref_service_transports` for the
backend/transport compatibility table and the recommended direct-to-replay
topology.

*Resources*: Find examples for these collectors in the
`dedicated folder <https://github.com/pytorch/rl/examples/distributed/collectors>`_.

.. note::
  *Choosing the sub-collector*: Distributed collector implementations support
  the various single-machine collector classes through ``backend_options``.
  One may wonder why using a process-backed :class:`torchrl.collectors.Collector`
  or a :class:`~torchrl.envs.ParallelEnv`
  instead. In general, multiprocessed collectors have a lower IO footprint than
  parallel environments which need to communicate at each step. Yet, the model specs
  play a role in the opposite direction, since using parallel environments will
  result in a faster execution of the policy (and/or transforms) since these
  operations will be vectorized.

.. note::
  *Choosing the device of a collector (or a parallel environment)*: Sharing data
  among processes is achieved via shared-memory buffers with parallel environment
  and multiprocessed environments executed on CPU. Depending on the capabilities
  of the machine being used, this may be prohibitively slow compared to sharing
  data on GPU which is natively supported by cuda drivers.
  In practice, this means that using the ``device="cpu"`` keyword argument when
  building a parallel environment or collector can result in a slower collection
  than using ``device="cuda"`` when available.

.. note::
  Given the library's many optional dependencies (eg, Gym, Gymnasium, and many others)
  warnings can quickly become quite annoying in multiprocessed / distributed settings.
  By default, TorchRL filters out these warnings in sub-processes. If one still wishes to
  see these warnings, they can be displayed by setting ``torchrl.filter_warnings_subprocess=False``.

.. tip::

  All distributed collectors support ``trajs_per_batch`` combined with
  ``replay_buffer``.  When set, each remote worker assembles **complete
  trajectories** and writes them to the shared buffer as flat 1-D sequences,
  which is directly compatible with :class:`~torchrl.data.replay_buffers.SliceSampler`.
  See :ref:`collectors_replay_trajs` for examples and best practices.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    DistributedCollector
    RPCCollector
    DistributedSyncCollector
    submitit_delayed_launcher
    RayCollector
    RayEvalWorker

Removed legacy names
--------------------

The deprecated distributed collector aliases were removed in v0.13. Use
:class:`torchrl.collectors.Collector` for construction. The canonical concrete
names ``DistributedCollector``, ``RPCCollector``, and
``DistributedSyncCollector`` remain available for implementation-specific
code.
