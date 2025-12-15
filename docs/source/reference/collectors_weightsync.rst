.. currentmodule:: torchrl.weight_update

Weight Synchronization
======================

RL pipelines are typically split in two big computational buckets: training, and inference.
While the inference pipeline sends data to the training one, the training pipeline needs to occasionally
synchronize its weights with the inference one.
In the most basic setting (fully synchronized data collection with traditional neural networks), the same weights are
used in both instances. From there, anything can happen:

- In multiprocessed or distributed settings, several copies of the policy can be held by the inference workers (named
  `DataCollectors` in TorchRL). When synchronizing the weights, each worker needs to receive a new copy of the weights
  for their instance of the policy.
- In some cases, the environment or the postprocessing hooks can rely on the usage of a model which itself needs
  synchronization. This means that there can be multiple ends in the data transfer API and one needs to think beyond
  policy-to-policy weight synchronization strategies.
- In the LLM world, the inference engine and the training one are very different: they will use different libraries,
  kernels and calling APIs (e.g., `generate` vs. `forward`). The weight format can also be drastically different (quantized
  vs non-quantized).
  This makes the weight synchronization much more complex, as one cannot simply dump and load a state dict on both ends.
- One typically also has to choose who instantiates a transfer: should this come from the inference engine who actively
  asks for new weights, or must it only be the trainer who pushes its weights to the workers? An intermediate approach
  is to store the weights on some intermediary server and let the workers fetch them when necessary.

TorchRL tries to account for each of these problems in a flexible manner. We identify three basic components in a weight
transfer:

- A **Scheme** class that orchestrates the entire weight synchronization lifecycle, including initialization,
  connection setup, and weight transfer coordination.
- A **Transport** class that handles the actual transfer of weights (through shared memory, queues, torch.distributed,
  Ray, etc.). Each scheme creates one or more transports for communication with workers.
- A **Strategy** class that determines the weight format (TensorDict or state_dict) and how weights are
  extracted from and applied to models.

Each of these classes is detailed below.

.. note::
    **For most users, weight synchronization happens automatically.** When using TorchRL collectors
    with the ``weight_sync_schemes`` argument, the collector handles all initialization, connection,
    and synchronization calls internally. You simply call ``collector.update_policy_weights_()`` and
    the weights are propagated to all workers.

    The ``update_policy_weights_`` method supports multiple calling conventions::

        # No arguments - uses registered policy
        collector.update_policy_weights_()

        # Positional argument - policy module or TensorDict
        collector.update_policy_weights_(policy_module)
        collector.update_policy_weights_(weights_tensordict)

        # Keyword arguments for clarity
        collector.update_policy_weights_(policy=actor_module)
        collector.update_policy_weights_(weights=weights_td, model_id="actor")

        # Multiple models atomically
        collector.update_policy_weights_(weights_dict={"actor": actor_td, "critic": critic_td})

    The detailed lifecycle documentation below is primarily intended for developers who want to:

    - Understand the internals of weight synchronization
    - Implement custom weight sync schemes for specialized use cases (e.g., new distributed backends, custom serialization)
    - Debug synchronization issues in complex distributed setups
    - Use weight sync schemes outside of collectors for custom multiprocessing scenarios

Lifecycle of Weight Synchronization
-----------------------------------

Weight synchronization follows a **two-phase initialization pattern** with a clear separation between
local setup and inter-process communication.

For **queue / store-based schemes** (e.g. multiprocessing, TCPStore), the receiver starts a small
**background loop** that waits for “update” instructions and runs the actual receive/apply logic.

For **RPC / Ray schemes**, the sender triggers the receiver via a **remote call** to
``_receive_weights_scheme()``, which runs ``scheme.receive()`` on the receiver side (no dedicated
background thread is required).

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        SENDER (Main Process)                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  1. scheme.init_on_sender(model_id, context, ...)                       │
    │     └─ Sets up local state, creates transports, NO communication        │
    │                                                                         │
    │  2. Make scheme available on receiver (scheme-dependent)                │
    │     └─ e.g. via multiprocessing pickle/serialization, RPC, Ray actor init │
    │                                                                         │
    │  3. scheme.connect()  ◄──── BLOCKING RENDEZ-VOUS ────►                  │
    │     └─ Sets up connection / rendez-vous                                  │
    │     └─ May send initial weights (scheme-dependent)                        │
    │                                                                         │
    │  4. scheme.send(weights)  [ready for ongoing updates]                   │
    │     └─ Triggers receiver to run ``scheme.receive()``                     │
    │        (instruction queue / TCPStore / remote call, scheme-dependent)    │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       RECEIVER (Worker Process)                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  1. scheme.init_on_receiver(model_id, context, ...)                     │
    │     └─ Sets up local state, resolves model, NO communication             │
    │                                                                         │
    │  2. scheme.connect()  ◄──── BLOCKING RENDEZ-VOUS ────►                  │
    │     └─ Receives initial weights (scheme-dependent)                        │
    │     └─ If needed: starts a background loop for update instructions        │
    │                                                                         │
    │  3. Receiver-side handler (scheme-dependent)                             │
    │     └─ Background thread for queue/store schemes                           │
    │     └─ RPC/Ray remote call handler for RPC/Ray schemes                     │
    └─────────────────────────────────────────────────────────────────────────┘

Phase 1: Initialization (No Communication)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``init_on_sender()`` and ``init_on_receiver()`` methods prepare local state without any
inter-process communication:

- Set up local attributes and references (model, context, worker indices)
- Create transport objects and register them
- Prepare queues, buffers, or other communication primitives
- **Do NOT perform any inter-worker communication**

This separation allows the scheme to be pickled and sent to worker processes after sender
initialization but before any actual communication occurs.

.. code-block:: python

    # === SENDER (main process) ===
    scheme = SharedMemWeightSyncScheme()
    scheme.init_on_sender(
        model_id="policy",
        context=collector,  # or explicit params like weights, devices, num_workers
    )

    # === Scheme is passed to workers via multiprocessing ===
    # (The scheme object is pickled and sent to worker processes)

    # === RECEIVER (worker process) ===
    scheme.init_on_receiver(
        model_id="policy",
        context=inner_collector,  # or explicit params like model, worker_idx
    )

Phase 2: Connection and Initial Weights (Rendez-vous)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``connect()`` method performs the actual inter-process communication. **In most schemes, both
sender and receiver call this method** (simultaneously or in the expected order for the scheme).
Some specialized schemes can be sender-driven (e.g. ``RayModuleTransformScheme`` triggers receiver setup
via a Ray call).

1. **Connection rendez-vous**: Sender and receiver synchronize (e.g., torch.distributed process group
   initialization, shared memory buffer exchange via queues)
2. **Initial weight transfer** (scheme-dependent): Some schemes send initial weights during ``connect()``
   (e.g. ``SharedMemWeightSyncScheme``, ``MultiProcessWeightSyncScheme``, ``DistributedWeightSyncScheme``,
   ``RayWeightSyncScheme``). Others (notably ``RPCWeightSyncScheme``) typically start synchronizing on the
   first ``send()`` call.
3. **Receiver readiness**: For queue/store-based schemes, ``connect()`` starts a background loop on the
   receiver that waits for update instructions.

.. code-block:: python

    # === Called simultaneously on both ends ===

    # Sender side (main process):
    scheme.connect()  # Blocks until rendez-vous completes (scheme-dependent)

    # Receiver side (worker process):
    scheme.connect(worker_idx=0)  # Blocks until rendez-vous completes (scheme-dependent)

.. note::
    The ``connect()`` method is a **blocking rendez-vous** for most schemes. The exact behavior
    depends on the scheme:

    - **Queue-based schemes** (SharedMem, MultiProcess): Sender puts to queue, receiver blocks reading
    - **Distributed schemes** (Distributed, Ray): Both sides block on ``torch.distributed.send/recv``
    - **RPC/Ray with remote calls**: Receiver's ``connect()`` may be a no-op if the sender triggers
      the receiver via a remote call (e.g., ``RayModuleTransformScheme``)

Phase 3: Ongoing Weight Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After ``connect()`` completes, the scheme is ready for ongoing weight synchronization. The sender
calls ``send()`` / ``send_async()`` to push weights and trigger the receiver to run ``scheme.receive()``.

.. code-block:: python

    # Training loop
    for batch in dataloader:
        loss = train_step(batch)
        scheme.send(new_weights)

Scheme-Specific Behavior
------------------------

SharedMemWeightSyncScheme
~~~~~~~~~~~~~~~~~~~~~~~~~

Uses shared memory for zero-copy weight updates. After initial setup, weight updates are instantaneous
since all processes share the same memory buffers.

.. list-table::
   :header-rows: 1

   * - Phase
     - Sender
     - Receiver
     - Communication
   * - ``init``
     - Creates shared buffers + instruction queues
     - Stores model reference
     - None
   * - ``connect``
     - Sends buffer references + initial weights
     - Receives buffers, applies weights, starts background thread
     - mp.Queue (blocking)
   * - ``send``
     - Updates shared memory, sends instruction
     - Background thread applies shared memory weights
     - Zero-copy shared memory + mp.Queue

MultiProcessWeightSyncScheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sends weight copies through multiprocessing queues. More flexible than shared memory but requires
explicit data transfer for each update. Supports timeout for non-blocking receives.

.. list-table::
   :header-rows: 1

   * - Phase
     - Sender
     - Receiver
     - Communication
   * - ``init``
     - Creates weight + instruction queues
     - Gets queue references
     - None
   * - ``connect``
     - Sends initial weights
     - Receives weights, applies via strategy, starts background thread
     - mp.Queue (blocking)
   * - ``send``
     - Puts weights + instruction
     - Background thread receives and applies weights
     - mp.Queue (supports timeout)

DistributedWeightSyncScheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses ``torch.distributed`` primitives with a TCPStore for signaling. Suitable for distributed
training scenarios where processes are already part of a process group. Supports timeout via
``irecv(return_premature=True)`` for non-blocking receives.

.. list-table::
   :header-rows: 1

   * - Phase
     - Sender
     - Receiver
     - Communication
   * - ``init``
     - Creates transports with TCPStore + rank
     - Creates transport with store + rank
     - None
   * - ``connect``
     - Sends initial weights via ``torch.distributed.send()``
     - Receives weights, applies via strategy, starts background thread
     - torch.distributed send/recv
   * - ``send``
     - Sets TCPStore flag + ``torch.distributed.send()``
     - Background thread polls TCPStore and receives weights
     - TCPStore + torch.distributed (supports timeout)

RPCWeightSyncScheme
~~~~~~~~~~~~~~~~~~~

Uses ``torch.distributed.rpc`` for signaling with ``torch.distributed`` for data transfer.
The sender's transport signals the remote collector via an RPC call to ``_receive_weights_scheme()``,
and then transfers weights via ``torch.distributed`` send/recv. Supports timeout via
``irecv(return_premature=True)`` for non-blocking receives.

.. list-table::
   :header-rows: 1

   * - Phase
     - Sender
     - Receiver
     - Communication
   * - ``init``
     - Creates transports with RPC refs
     - Stores model reference, creates transport
     - None
   * - ``connect``
     - No-op for RPC transport (no initial weight transfer)
     - No-op
     - None
   * - ``send``
     - RPC call to ``_receive_weights_scheme()`` + ``torch.distributed.send()``
     - Receiver runs ``scheme.receive()`` in the RPC call context and applies weights
     - RPC + torch.distributed (supports timeout)

RayWeightSyncScheme
~~~~~~~~~~~~~~~~~~~

Uses Ray actors for coordination with ``torch.distributed`` for efficient weight transfer.
Suitable for Ray-based distributed RL setups. Supports timeout via ``irecv(return_premature=True)``
for non-blocking receives.

.. list-table::
   :header-rows: 1

   * - Phase
     - Sender
     - Receiver
     - Communication
   * - ``init``
     - Creates transports with Ray actor handles
     - Creates transport, stores model
     - None
   * - ``connect``
     - Creates ConnectionInfo, ``init_process_group(rank=0)``, sends initial weights
     - Waits for ConnectionInfo, ``init_process_group(rank=N)``, receives weights
     - Ray actor + torch.distributed
   * - ``send``
     - Ray remote call to ``_receive_weights_scheme()`` + ``torch.distributed.isend()``
     - Receiver runs ``scheme.receive()`` in the Ray call context and applies weights
     - Ray + torch.distributed (supports timeout)

RayModuleTransformScheme
~~~~~~~~~~~~~~~~~~~~~~~~

Specialized scheme for synchronizing weights to a module running inside a ``RayModuleTransform``.
The sender triggers all receiver operations via Ray remote calls.

.. list-table::
   :header-rows: 1

   * - Phase
     - Sender
     - Receiver
     - Communication
   * - ``init``
     - Creates transport for transform actor
     - Creates transport, stores module
     - None
   * - ``connect``
     - Ray call triggers receiver init + weight send
     - Triggered by Ray: joins process group, receives weights
     - Ray + torch.distributed
   * - ``send``
     - Ray remote call to ``_receive_weights_scheme()`` + ``torch.distributed.isend()``
     - Receiver runs ``scheme.receive()`` in the Ray call context and applies weights
     - Ray + torch.distributed

.. note::
    ``RayModuleTransformScheme`` is unique in that even ``connect`` on the sender
    triggers the receiver initialization via a Ray remote call. The user only needs to call
    ``connect()`` on the sender side.

Background Thread Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some schemes use a **background receiver thread** on the receiver side. This is used when the sender
cannot directly invoke receiver logic (e.g. multiprocessing queues or TCPStore-based signaling).
The thread is started during ``connect()`` and runs ``scheme.receive()`` when instructed by the sender.

**Instruction mechanisms** (scheme-specific):
- **SharedMem/MultiProcess**: Queue-based (``queue.put("receive")``)
- **Distributed**: TCPStore-based (``store.set("receive")``)
- **RPC/Ray**: Remote calls to ``_receive_weights_scheme()`` (no dedicated background thread)

**Benefits**: non-blocking main process for queue/store-based schemes, sender-triggered updates,
automatic cascading to sub-collectors, and graceful timeout handling.

Usage Examples
--------------

.. note::
    **Runnable versions** of these examples are available in the repository:
    
    - `examples/collectors/weight_sync_standalone.py <https://github.com/pytorch/rl/blob/main/examples/collectors/weight_sync_standalone.py>`_: Standalone weight synchronization
    - `examples/collectors/weight_sync_collectors.py <https://github.com/pytorch/rl/blob/main/examples/collectors/weight_sync_collectors.py>`_: Collector integration

Using Weight Sync Schemes with Collectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weight sync schemes integrate seamlessly with TorchRL collectors. The collector handles calling
``init_on_sender()``, ``init_on_receiver()``, and ``connect()`` automatically:

.. code-block:: python

    import torch.nn as nn
    from tensordict.nn import TensorDictModule
    from torchrl.collectors import MultiCollector
    from torchrl.envs import GymEnv
    from torchrl.weight_update import SharedMemWeightSyncScheme

    # Create environment and policy
    env = GymEnv("CartPole-v1")
    policy = TensorDictModule(
        nn.Linear(env.observation_spec["observation"].shape[-1],
                  env.action_spec.shape[-1]),
        in_keys=["observation"],
        out_keys=["action"],
    )

    # Create scheme - collector handles initialization
    scheme = SharedMemWeightSyncScheme(strategy="tensordict")

    collector = MultiCollector(
        sync=True,
        create_env_fn=[lambda: GymEnv("CartPole-v1")] * 3,
        policy=policy,
        frames_per_batch=192,
        total_frames=10000,
        weight_sync_schemes={"policy": scheme},
    )

    # Collect data and update weights
    for i, data in enumerate(collector):
        # ... training step ...
        
        # Update weights - multiple calling conventions supported:
        if i % 10 == 0:
            # Option 1: No arguments (uses registered policy)
            collector.update_policy_weights_()
            
            # Option 2: Pass policy module (positional)
            collector.update_policy_weights_(policy)
            
            # Option 3: Pass weights TensorDict (positional)
            # collector.update_policy_weights_(weights_tensordict)
            
            # Option 4: Use keyword arguments for clarity
            # collector.update_policy_weights_(policy=policy)
            # collector.update_policy_weights_(weights=weights_td, model_id="policy")

    collector.shutdown()

Using Weight Sync Schemes Standalone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom multiprocessing scenarios, you can use schemes directly. The key is to follow the
two-phase pattern: initialize first (no communication), then connect (blocking rendez-vous):

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch import multiprocessing as mp
    from tensordict import TensorDict
    from torchrl.weight_update import SharedMemWeightSyncScheme

    def worker_fn(scheme, worker_idx):
        """Worker process - receives scheme via pickle."""
        # Create local model (weights will be overwritten by sender's weights)
        model = nn.Linear(4, 2)

        # PHASE 1: Initialize on receiver (no communication yet)
        scheme.init_on_receiver(model_id="policy", model=model, worker_idx=worker_idx)

        # PHASE 2: Blocking rendez-vous - receive initial weights from sender
        scheme.connect(worker_idx=worker_idx)
        # model now has the sender's weights; background thread started

        # Ready to work - background thread handles weight updates automatically
        while True:
            # ... use model for inference ...

    # === MAIN PROCESS (Sender) ===
    policy = nn.Linear(4, 2)
    scheme = SharedMemWeightSyncScheme()

    # PHASE 1: Initialize on sender (no communication yet)
    scheme.init_on_sender(
        model_id="policy",
        weights=TensorDict.from_module(policy),
        devices=[torch.device("cpu")] * 2,
        num_workers=2,
    )

    # Spawn workers - scheme is pickled and sent to each worker
    workers = [mp.Process(target=worker_fn, args=(scheme, i)) for i in range(2)]
    for w in workers:
        w.start()

    # PHASE 2: Blocking rendez-vous - send initial weights to workers
    scheme.connect()
    # Workers now have copies of policy's weights!

    # PHASE 3: Ongoing updates (zero-copy for shared memory)
    for epoch in range(10):
        # ... training step updates policy weights ...
        scheme.send()  # Background threads automatically apply weights

    scheme.shutdown()  # Stop background threads
    for w in workers:
        w.join()

.. note::
    With ``SharedMemWeightSyncScheme``, weight updates are zero-copy since all processes share the same
    memory buffers. Background threads automatically apply updates when instructed by the sender.

.. note::
    The ``strategy`` parameter determines the weight format: ``"state_dict"`` uses PyTorch's native state
    dictionaries, while ``"tensordict"`` (default) uses TensorDict format which is more efficient for
    structured models and supports features like device mapping.

Transports
----------

Transports handle the low-level communication between sender and receiver. Each scheme creates
appropriate transport instances for its workers.

Transport Interface
~~~~~~~~~~~~~~~~~~~

All transports implement the ``TransportBackend`` protocol with a stateless design. The key methods
accept ``weights``, ``model``, and ``strategy`` as keyword arguments rather than storing them as
instance attributes:

.. code-block:: python

    # Transport methods accept model/weights/strategy as kwargs
    transport.receive_weights(
        timeout=None,      # Optional timeout in seconds (None = blocking)
        weights=buffer,    # Pre-allocated weight buffer
        model=policy,      # Model to apply weights to
        strategy=strategy, # WeightStrategy for weight application
    )

    transport.setup_connection_and_weights_on_receiver(
        worker_idx=0,
        weights=buffer,
        model=policy,
        strategy=strategy,
    )

Timeout Support
~~~~~~~~~~~~~~~

Transports support timeout for non-blocking weight reception:

.. list-table::
   :header-rows: 1

   * - Transport
     - Timeout Support
     - Notes
   * - ``MPTransport``
     - ✅ Yes
     - Uses ``queue.get(timeout=...)``
   * - ``RPCTransport``
     - ✅ Yes
     - Uses ``irecv(return_premature=True)`` with polling
   * - ``RayTransport``
     - ✅ Yes
     - Uses ``irecv(return_premature=True)`` with polling
   * - ``DistributedTransport``
     - ✅ Yes
     - Uses ``irecv(return_premature=True)`` with polling
   * - ``SharedMemTransport``
     - N/A
     - Shared memory is instant (no waiting)

When ``timeout=None`` (default), the receive operation blocks until weights arrive.
When a timeout is specified, the method returns ``None`` if the timeout expires before
weights are received.

Available Transports
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TransportBackend
    MPTransport
    SharedMemTransport
    RayTransport
    RPCTransport
    DistributedTransport

Schemes
-------

Schemes orchestrate the weight synchronization lifecycle, managing initialization, connection setup,
and ongoing weight transfers.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    WeightSyncScheme
    WeightStrategy
    MultiProcessWeightSyncScheme
    SharedMemWeightSyncScheme
    NoWeightSyncScheme
    RayWeightSyncScheme
    RayModuleTransformScheme
    RPCWeightSyncScheme
    DistributedWeightSyncScheme

Legacy: Weight Updaters
-----------------------

.. warning::
    The `WeightUpdater` API is deprecated as of the 0.11 release.
    The Weight Sync Schemes API provides more flexibility and better compatibility with heavy
    weight transfers (e.g., LLMs) and should be preferred for all new code.

.. currentmodule:: torchrl.collectors

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    WeightUpdaterBase
    VanillaWeightUpdater
    MultiProcessedWeightUpdater
    RayWeightUpdater

.. currentmodule:: torchrl.collectors.distributed

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RPCWeightUpdater
    DistributedWeightUpdater
