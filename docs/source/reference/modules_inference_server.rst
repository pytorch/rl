.. currentmodule:: torchrl.modules.inference_server

Inference Server
================

.. _ref_inference_server:

The inference server provides auto-batching model serving for RL actors.
Multiple actors submit individual TensorDicts; the server transparently
batches them, runs a single model forward pass, and routes results back.

Core API
--------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    InferenceServer
    InferenceServerConfig
    InferenceDeviceConfig
    ProcessInferenceServer
    InferenceClient
    PolicyClientModule
    InferenceTransport

Transport Backends
------------------

The transport can be selected behind the high-level
:class:`InferenceServer` constructor. Both process- and Ray-owned servers can
use ``transport="distributed"`` with Gloo/NCCL for fixed-layout TensorDict
payloads. A process-owned server requires explicit ``request_spec`` and
``response_spec`` values before its subprocess starts; a Ray-owned server can
bind those layouts on first use. Ray-owned inference can instead use
``transport="ray"`` for dynamic or non-tensor payloads. See
:ref:`ref_service_transports` for supported owner/transport combinations,
restrictions, and expected performance, and
:ref:`ref_distributed_transport_layouts` for the layout-discovery and buffer
lifecycle.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ThreadingTransport
    SlotTransport
    MPTransport
    SharedMemoryTransport
    ProcessSlotTransport
    RayTransport
    MonarchTransport

Usage
-----

The simplest setup uses :class:`ThreadingTransport` for actors that are
threads in the same process:

.. code-block:: python

    from tensordict.nn import TensorDictModule
    from torchrl.modules.inference_server import (
        InferenceServer,
        ThreadingTransport,
    )
    import torch.nn as nn
    import concurrent.futures

    policy = TensorDictModule(
        nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 4)),
        in_keys=["observation"],
        out_keys=["action"],
    )

    transport = ThreadingTransport()
    server = InferenceServer(policy, transport, max_batch_size=32)
    server.start()
    client = server.client()

    # actor threads call client(td) -- batched automatically
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        ...

    server.shutdown()

Static CUDA-graph batches
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA policies can remove Python dispatch and kernel-launch overhead by setting
``static_batch_size`` together with an explicit CUDA ``policy_device``. The
server clones the last real request into any pad rows, replays a
:class:`tensordict.nn.CudaGraphModule` at the fixed size, and discards padded
outputs before returning an owned copy to each actor. The static size must be
at least ``max_batch_size``:

.. code-block:: python

    import torch
    from tensordict import TensorDict

    request_spec = TensorDict(
        {
            "observation": torch.zeros(64),
            "state": torch.zeros(32 * 64),
            "belief": torch.zeros(512),
            "previous_action": torch.zeros(20),
            "is_init": torch.zeros(1, dtype=torch.bool),
        }
    )
    server = InferenceServer(
        policy,
        ThreadingTransport(),
        max_batch_size=64,
        static_batch_size=64,
        request_spec=request_spec,
        policy_device="cuda:0",
        output_device="cpu",
    )
    server.start()  # warm-up and capture finish before the worker starts

The representative ``request_spec`` is required when constructing a server
directly. It may be supplied without ``response_spec`` for transports whose
layout is dynamic. :class:`~torchrl.collectors.AsyncBatchedCollector` derives
the request from the environment specs and calls
:meth:`~torchrl.modules.inference_server.InferenceServer.prepare_cudagraph`
before starting its inference and coordinator threads. Initial weight
synchronization also finishes before capture. Recurrent tensor inputs such as
``state``, ``belief``, and ``is_init`` remain inside the graph. The first real
request is checked for every captured policy input key so an incomplete
``request_spec`` cannot silently leave stale input values in the graph.

In-place parameter copies preserve the captured graph; for example, use
``TensorDict.from_module(learner).to_module(behavior, inplace=True)``. A
storage-replacing update raises an error and disables the graph, leaving the
server on the safe eager path until it is stopped and prepared again. This
avoids capturing while collector or environment threads are live. The
interaction type is frozen to its effective value at capture time, including an ambient
``set_interaction_type`` context; later requests using another type are
rejected.

CUDA operations using PyTorch's default generator advance its graph-safe state
across replays. Custom generators must manage CUDA graph state explicitly, and
policies that reseed with ``manual_seed`` during each forward cannot be
captured. Consequently, DreamerV3's ``separate_policy_rng`` mode and
``static_batch_size`` cannot be enabled together.

Shared-memory transport
^^^^^^^^^^^^^^^^^^^^^^^

For cross-process actors with large request payloads (e.g. image
observations), :class:`SharedMemoryTransport` preallocates request and
response slot banks in CPU shared memory and passes only slot indices
through the multiprocessing queues, removing per-request pickling from the
hot path. The caller provides representative request and response
TensorDicts that fix the slot layout (keys, shapes, dtypes); only the
declared keys are transmitted:

.. code-block:: python

    import torch
    from tensordict import TensorDict
    from torchrl.modules.inference_server import (
        InferenceServer,
        SharedMemoryTransport,
    )

    transport = SharedMemoryTransport(
        request_spec=TensorDict({"pixels": torch.zeros(3, 224, 224)}),
        response_spec=TensorDict(
            {
                "action": torch.zeros(7),
                "policy_version": torch.zeros((), dtype=torch.long),
            }
        ),
        num_slots=64,
    )
    # Create clients before spawning env workers
    clients = [transport.client() for _ in range(n_workers)]

    server = InferenceServer(policy, transport, policy_device="cuda:0")
    server.start()

Slots are CPU-only: clients must submit CPU tensors, and the server owns
all device transfers (batches are moved to ``policy_device`` before the
forward pass, results copied back into the CPU response slots).
``num_slots`` bounds the number of concurrently in-flight requests and
provides natural backpressure.

Direct process-slot transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For one synchronous acting loop per environment worker,
:class:`ProcessSlotTransport` assigns one fixed request/response slot to each
worker. Workers notify the dedicated inference process through a shared
semaphore, and the server sweeps ready slots in round-robin order. Observation
and action tensors therefore never cross the driver process:

.. code-block:: python

    import torch
    from tensordict import TensorDict
    from torchrl.collectors import AsyncBatchedCollector
    from torchrl.modules.inference_server import (
        InferenceServerConfig,
        ProcessSlotTransport,
    )

    num_envs = 64
    transport = ProcessSlotTransport(
        request_spec=TensorDict({"pixels": torch.zeros(3, 84, 84, dtype=torch.uint8)}),
        response_spec=TensorDict(
            {
                "action": torch.zeros(6),
                "policy_version": torch.zeros((), dtype=torch.long),
            }
        ),
        num_slots=num_envs,
    )
    collector = AsyncBatchedCollector(
        create_env_fn=[make_env] * num_envs,
        policy_factory=make_policy,
        transport=transport,
        env_backend="multiprocessing",
        server_config=InferenceServerConfig(
            service_backend="process", max_batch_size=num_envs
        ),
        frames_per_batch=1024,
    )

The driver continues to receive completed transitions from the workers. The
transport requires fixed-shape CPU tensor request and response specs; policy
execution and device transfers remain owned by the inference process.

Structured Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Server execution, batching, and device placement are grouped into two
dataclasses instead of loose keyword arguments: :class:`InferenceServerConfig`
collects the execution ``service_backend`` (``"thread"`` or ``"process"``) and the
batching/instrumentation knobs (``max_batch_size``, ``static_batch_size``,
``min_batch_size``, ``timeout``, ``collect_stats``, ``stats_window_size``), and
:class:`InferenceDeviceConfig` describes device placement across the
collection pipeline (``policy_device``, ``output_device``, ``env_device``,
``storing_device``). Both :class:`InferenceServer` and
:class:`~torchrl.collectors.AsyncBatchedCollector` accept them through the
``server_config`` and ``device_config`` keyword arguments; a config object is
mutually exclusive with the individual keyword arguments it replaces, and the
config objects are the only way to set the per-role devices and the server
backend on the collector. Servers consume only the
``policy_device``/``output_device`` fields (``env_device`` doubles as an
``output_device`` fallback), while ``env_device`` and ``storing_device``
drive the collector-side transfers:

.. code-block:: python

    from torchrl.collectors import AsyncBatchedCollector
    from torchrl.modules.inference_server import (
        InferenceDeviceConfig,
        InferenceServerConfig,
    )

    collector = AsyncBatchedCollector(
        create_env_fn=[make_env] * 8,
        policy=my_policy,
        frames_per_batch=200,
        server_config=InferenceServerConfig(max_batch_size=8, timeout=0.005),
        device_config=InferenceDeviceConfig(
            policy_device="cuda:0",
            env_device="cpu",
            storing_device="cpu",
        ),
    )

Remote policy module
^^^^^^^^^^^^^^^^^^^^

Use :class:`PolicyClientModule` when an actor or collector expects a regular
TensorDict policy but inference should be served by the policy server:

.. code-block:: python

    remote_policy = PolicyClientModule(
        server,
        in_keys=["observation"],
        out_keys=["action", "policy_version"],
    )

``PolicyClientModule`` accepts a server owner, transport, or existing callable
client. Owners and transports are automatically reduced to their restricted
client before the module is sent to a worker.

    data = remote_policy(data)

The server writes ``policy_version`` by default so asynchronous collectors can
track behavior-policy lag. This is the general *service-stamped metadata*
pattern: any service may stamp its responses with metadata about the state it
served them from, and the data pipeline may enforce freshness constraints on
it. Bounded staleness is enforced by the replay buffer through
:class:`~torchrl.envs.transforms.PolicyAgeFilter`, which drops elements whose
stamped version lags the live version by more than ``max_policy_lag`` --
either at extension time or dynamically at sampling time.

Weight Synchronisation
^^^^^^^^^^^^^^^^^^^^^^

The server integrates with :class:`~torchrl.weight_update.WeightSyncScheme`
to receive updated model weights from a trainer between inference batches:

.. code-block:: python

    from torchrl.weight_update import SharedMemWeightSyncScheme

    weight_sync = SharedMemWeightSyncScheme()
    # Initialise on the trainer (sender) side first
    weight_sync.init_on_sender(model=training_model, ...)

    server = InferenceServer(
        model=inference_model,
        transport=ThreadingTransport(),
        weight_sync=weight_sync,
    )
    server.start()

    # Training loop
    for batch in dataloader:
        loss = loss_fn(training_model(batch))
        loss.backward()
        optimizer.step()
        weight_sync.send(model=training_model)  # pushed to server

Integration with Collectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to use the inference server with RL data collection is
through :class:`~torchrl.collectors.AsyncBatchedCollector`, which
creates the server, transport, and env pool automatically:

.. code-block:: python

    from torchrl.collectors import AsyncBatchedCollector
    from torchrl.envs import GymEnv

    collector = AsyncBatchedCollector(
        create_env_fn=[lambda: GymEnv("CartPole-v1")] * 8,
        policy=my_policy,
        frames_per_batch=200,
        total_frames=10_000,
        max_batch_size=8,
    )

    for data in collector:
        # train on data ...
        pass

    collector.shutdown()
