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

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ThreadingTransport
    SlotTransport
    MPTransport
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
    client = transport.client()

    # actor threads call client(td) -- batched automatically
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        ...

    server.shutdown()

Structured Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Server execution, batching, and device placement are grouped into two
dataclasses instead of loose keyword arguments: :class:`InferenceServerConfig`
collects the execution ``backend`` (``"thread"`` or ``"process"``) and the
batching/instrumentation knobs (``max_batch_size``, ``min_batch_size``,
``timeout``, ``collect_stats``, ``stats_window_size``), and
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
        transport,
        in_keys=["observation"],
        out_keys=["action", "policy_version"],
    )

    data = remote_policy(data)

The server writes ``policy_version`` by default so asynchronous collectors can
track behavior-policy lag. This is the general *service-stamped metadata*
pattern: any service may stamp its responses with metadata about the state it
served them from, and clients may enforce freshness constraints on it (for the
policy server: ``target_policy_version`` and ``max_policy_lag`` on
:class:`PolicyClientModule`).

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
