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
  for his instance of the policy.
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

TorchRL tries to account for each of these problems in a flexible manner. We individuate three basic components in a weight
transfer:

- A **Scheme** class that orchestrates the entire weight synchronization lifecycle, including initialization,
  connection setup, and weight transfer coordination.
- A **Transport** class that handles the actual transfer of weights (through shared memory, queues, torch.distributed,
  Ray, etc.). Each scheme creates one or more transports for communication with workers.
- A **Strategy** class that determines the weight format (TensorDict or state_dict) and how weights are
  extracted from and applied to models.

Each of these classes is detailed below.

Lifecycle of Weight Synchronization
-----------------------------------

Weight synchronization follows a **two-phase initialization pattern**:

Phase 1: Initialization (No Communication)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first phase uses ``init_on_sender()`` and ``init_on_receiver()`` methods. These methods:

- Set up local attributes and references (model, context, worker indices)
- Create transport objects and register them
- Prepare queues, buffers, or other communication primitives
- **Do NOT perform any inter-worker communication**

This phase can happen independently on sender and receiver sides, in any order.

.. code-block:: python

    # On sender (main process)
    scheme = SharedMemWeightSyncScheme()
    scheme.init_on_sender(
        model_id="policy",
        context=collector,  # or explicit params
    )

    # On receiver (worker process) - can happen before or after sender init
    scheme.init_on_receiver(
        model_id="policy",
        context=inner_collector,
    )

Phase 2: Connection and Initial Weights (Rendez-vous)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second phase uses ``connect()`` which dispatches to:

- ``_setup_connection_and_weights_on_sender_impl()`` on the sender side
- ``_setup_connection_and_weights_on_receiver_impl()`` on the receiver side

This phase performs the actual inter-worker communication:

1. **Connection rendez-vous**: Sender and receiver synchronize (e.g., torch.distributed process group initialization,
   shared memory buffer exchange via queues)
2. **Initial weight transfer** (optional): If the model has weights, they are sent from sender to receivers

.. code-block:: python

    # Both sides must call this - order depends on the scheme
    # Sender side:
    scheme.connect()

    # Receiver side (in worker process):
    scheme.connect(worker_idx=0)

.. note::
    The ``connect()`` method is a **blocking rendez-vous** for most schemes. Both sender
    and receiver must call it for the synchronization to complete. The exact blocking behavior depends on the
    scheme:
    
    - **Queue-based schemes** (SharedMem, MultiProcess): Sender puts to queue, receiver blocks reading from queue
    - **Distributed schemes** (Ray, RPC, Distributed): Both sides block on ``init_process_group`` or similar collective operations

Ongoing Weight Updates
~~~~~~~~~~~~~~~~~~~~~~

After initialization, weight updates use:

- ``send()`` / ``send_async()`` on the sender side
- ``receive()`` on the receiver side (or automatic for shared memory)

For some schemes (Ray, RPC), the sender's ``send()`` makes a remote call that triggers the receiver
automatically, so the user doesn't need to explicitly poll ``receive()``.

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
     - Creates shared buffers + per-worker queues
     - Stores model reference
     - None
   * - ``connect``
     - Puts buffer references into queues
     - Reads from queue, applies to model
     - mp.Queue (blocking)
   * - ``send``
     - Updates shared memory in-place
     - N/A (sees updates automatically)
     - Zero-copy shared memory

MultiProcessWeightSyncScheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sends weight copies through multiprocessing queues. More flexible than shared memory but requires
explicit data transfer for each update.

.. list-table::
   :header-rows: 1

   * - Phase
     - Sender
     - Receiver
     - Communication
   * - ``init``
     - Creates per-worker queues
     - Gets queue reference
     - None
   * - ``connect``
     - Sends weights via queue
     - Reads from queue, applies to model
     - mp.Queue (blocking)
   * - ``send``
     - Puts weights into queues
     - Must call ``receive()``
     - mp.Queue

DistributedWeightSyncScheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses ``torch.distributed`` primitives with a TCPStore for signaling. Suitable for distributed
training scenarios where processes are already part of a process group.

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
     - No-op (process group already exists)
     - No-op
     - None
   * - ``send``
     - Sets TCPStore flag + ``torch.distributed.send()``
     - Must poll TCPStore, then call ``receive()``
     - TCPStore + torch.distributed

RPCWeightSyncScheme
~~~~~~~~~~~~~~~~~~~

Uses ``torch.distributed.rpc`` for signaling with ``torch.distributed`` for data transfer.
The sender's ``send()`` triggers the receiver via RPC, so no explicit receiver polling is needed.

.. list-table::
   :header-rows: 1

   * - Phase
     - Sender
     - Receiver
     - Communication
   * - ``init``
     - Creates transports with RPC refs
     - Stores model reference
     - None
   * - ``connect``
     - No-op
     - No-op
     - None
   * - ``send``
     - **RPC call** triggers receiver + ``send()``
     - Triggered by RPC, does ``recv()``
     - RPC + torch.distributed

RayWeightSyncScheme
~~~~~~~~~~~~~~~~~~~

Uses Ray actors for coordination with ``torch.distributed`` for efficient weight transfer.
Suitable for Ray-based distributed RL setups.

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
     - Creates ConnectionInfo Ray actor, ``init_process_group(rank=0)``, sends initial weights
     - Waits for ConnectionInfo, ``init_process_group(rank=N)``, receives weights
     - **Rendez-vous**: Ray actor + torch.distributed
   * - ``send``
     - **Ray remote call** triggers receiver + ``isend()``
     - Triggered by Ray, does ``irecv()``
     - Ray + torch.distributed

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
     - **Ray call** triggers receiver init, then rendez-vous + weight send
     - **Triggered by Ray**: joins process group, receives weights
     - Ray + torch.distributed
   * - ``send``
     - **Ray remote call** triggers receiver + ``isend()``
     - Triggered by Ray, does ``irecv()``
     - Ray + torch.distributed

.. note::
    ``RayModuleTransformScheme`` is unique in that even ``connect`` on the sender
    triggers the receiver initialization via a Ray remote call. The user only needs to call
    ``connect()`` on the sender side.

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
    from torchrl.collectors import MultiSyncDataCollector
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

    collector = MultiSyncDataCollector(
        create_env_fn=[lambda: GymEnv("CartPole-v1")] * 3,
        policy=policy,
        frames_per_batch=192,
        total_frames=10000,
        weight_sync_schemes={"policy": scheme},
    )

    # Collect data and update weights
    for i, data in enumerate(collector):
        # ... training step ...
        
        # Update weights - workers see updates via shared memory
        if i % 10 == 0:
            collector.update_policy_weights_()

    collector.shutdown()

Using Weight Sync Schemes Standalone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom multiprocessing scenarios, you can use schemes directly:

.. code-block:: python

    import torch.nn as nn
    from torch import multiprocessing as mp
    from tensordict import TensorDict
    from torchrl.weight_update import SharedMemWeightSyncScheme

    def worker_fn(scheme, worker_idx):
        # Phase 1: Initialize on receiver (no communication)
        model = nn.Linear(4, 2)
        scheme.init_on_receiver(model_id="policy", model=model, worker_idx=worker_idx)
        
        # Phase 2: Rendez-vous - receive initial weights
        scheme.connect(worker_idx=worker_idx)
        
        # Now model has the weights from sender
        # For SharedMem, subsequent updates are automatic (shared memory)

    # Main process
    policy = nn.Linear(4, 2)
    scheme = SharedMemWeightSyncScheme()

    # Phase 1: Initialize on sender
    scheme.init_on_sender(
        model_id="policy",
        weights=TensorDict.from_module(policy),
        devices=[torch.device("cpu")] * 2,
        num_workers=2,
    )

    # Start workers
    workers = [mp.Process(target=worker_fn, args=(scheme, i)) for i in range(2)]
    for w in workers:
        w.start()

    # Phase 2: Rendez-vous - send initial weights
    scheme.connect()

    # Ongoing updates (zero-copy for shared memory)
    for _ in range(10):
        # ... training ...
        scheme.send()  # Updates shared memory in-place

    for w in workers:
        w.join()

.. note::
    When using ``SharedMemWeightSyncScheme``, weight updates after initialization are zero-copy and extremely
    fast since all processes share the same memory buffers. Workers don't need to call ``receive()`` - they
    automatically see updates.

.. note::
    The ``strategy`` parameter determines the weight format: ``"state_dict"`` uses PyTorch's native state
    dictionaries, while ``"tensordict"`` (default) uses TensorDict format which is more efficient for
    structured models and supports features like device mapping.

Transports
----------

Transports handle the low-level communication between sender and receiver. Each scheme creates
appropriate transport instances for its workers.

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
