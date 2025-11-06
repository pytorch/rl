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

TorchRL tries to account for each of these problems in a flexible manner. We individuate four basic components in a weight
transfer:

- A `Sender` class that somehow gets the weights (or a reference to them) and initializes the transfer;
- A `Receiver` class that casts the weights to the destination module (policy or other utility module);
- A `Transport` class that codes up the actual transfer of the weights (through shared memory, nccl or anything else).
- A Scheme that defines what sender, receiver and transport have to be used and how to initialize them.

Each of these classes is detailed below.

Usage Examples
--------------

.. note::
    **Runnable versions** of these examples are available in the repository:
    
    - `examples/collectors/weight_sync_standalone.py <https://github.com/pytorch/rl/blob/main/examples/collectors/weight_sync_standalone.py>`_: Standalone weight synchronization
    - `examples/collectors/weight_sync_collectors.py <https://github.com/pytorch/rl/blob/main/examples/collectors/weight_sync_collectors.py>`_: Collector integration

Using Weight Update Schemes Independently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weight update schemes can be used outside of collectors for custom synchronization scenarios.
The new simplified API provides four core methods for weight synchronization:

- ``init_on_sender(model_id, **kwargs)`` - Initialize on the main process (trainer) side
- ``init_on_worker(model_id, **kwargs)`` - Initialize on worker process side
- ``get_sender()`` - Get the configured sender instance
- ``get_receiver()`` - Get the configured receiver instance

Here's a basic example:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch import multiprocessing as mp
    from tensordict import TensorDict
    from torchrl.weight_update import (
        MultiProcessWeightSyncScheme,
        SharedMemWeightSyncScheme,
    )

    # Create a simple policy
    policy = nn.Linear(4, 2)

    # Example 1: Multiprocess weight synchronization with state_dict
    # --------------------------------------------------------------
    # On the main process side (trainer):
    scheme = MultiProcessWeightSyncScheme(strategy="state_dict")
    
    # Initialize scheme with pipes
    parent_pipe, child_pipe = mp.Pipe()
    scheme.init_on_sender(model_id="policy", pipes=[parent_pipe])
    
    # Get the sender and send weights
    sender = scheme.get_sender()
    weights = policy.state_dict()
    sender.send(weights)  # Synchronous send
    # or sender.send_async(weights); sender.wait_async()  # Asynchronous send

    # On the worker process side:
    # scheme.init_on_worker(model_id="policy", pipe=child_pipe, model=policy)
    # receiver = scheme.get_receiver()
    # # Non-blocking check for new weights
    # if receiver.receive(timeout=0.001):
    #     # Weights were received and applied

    # Example 2: Shared memory weight synchronization
    # ------------------------------------------------
    # Create shared memory scheme with auto-registration
    shared_scheme = SharedMemWeightSyncScheme(strategy="tensordict", auto_register=True)
    
    # Initialize with pipes for lazy registration
    parent_pipe2, child_pipe2 = mp.Pipe()
    shared_scheme.init_on_sender(model_id="policy", pipes=[parent_pipe2])
    
    # Get sender and send weights (automatically creates shared buffer on first send)
    shared_sender = shared_scheme.get_sender()
    weights_td = TensorDict.from_module(policy)
    shared_sender.send(weights_td)

    # Workers automatically see updates via shared memory!

Using Weight Update Schemes with Collectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weight update schemes integrate seamlessly with TorchRL collectors, enabling efficient weight synchronization
across multiple inference workers:

.. code-block:: python

    import torch.nn as nn
    from tensordict.nn import TensorDictModule
    from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
    from torchrl.envs import GymEnv
    from torchrl.weight_update import (
        MultiProcessWeightSyncScheme,
        SharedMemWeightSyncScheme,
    )

    # Create environment and policy
    env = GymEnv("CartPole-v1")
    policy = TensorDictModule(
        nn.Linear(env.observation_spec["observation"].shape[-1],
                  env.action_spec.shape[-1]),
        in_keys=["observation"],
        out_keys=["action"],
    )

    # Example 1: Single collector with multiprocess scheme
    # -----------------------------------------------------
    scheme = MultiProcessWeightSyncScheme(strategy="state_dict")

    collector = SyncDataCollector(
        create_env_fn=lambda: GymEnv("CartPole-v1"),
        policy=policy,
        frames_per_batch=64,
        total_frames=1000,
        weight_sync_schemes={"policy": scheme},
    )

    # Collect data and update weights periodically
    for i, data in enumerate(collector):
        # ... training step with data ...

        # Update policy weights every N iterations
        if i % 10 == 0:
            new_weights = policy.state_dict()
            collector.update_policy_weights_(new_weights)

    collector.shutdown()

    # Example 2: Multiple collectors with shared memory
    # --------------------------------------------------
    # Shared memory is more efficient for frequent updates
    shared_scheme = SharedMemWeightSyncScheme(strategy="tensordict", auto_register=True)

    collector = MultiSyncDataCollector(
        create_env_fn=[
            lambda: GymEnv("CartPole-v1"),
            lambda: GymEnv("CartPole-v1"),
            lambda: GymEnv("CartPole-v1"),
        ],
        policy=policy,
        frames_per_batch=192,
        total_frames=10000,
        weight_sync_schemes={"policy": shared_scheme},
    )

    # Workers automatically see weight updates via shared memory
    for data in collector:
        # ... training ...
        collector.update_policy_weights_(TensorDict.from_module(policy))

    collector.shutdown()

.. note::
    When using ``SharedMemWeightSyncScheme``, weight updates are zero-copy and extremely fast since all
    processes share the same memory buffers. This is ideal for frequent weight updates but requires all
    processes to be on the same machine.

.. note::
    The ``strategy`` parameter determines the weight format: ``"state_dict"`` uses PyTorch's native state
    dictionaries, while ``"tensordict"`` uses TensorDict format which can be more efficient for structured
    models and supports advanced features like lazy initialization.

Weight Senders
--------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    WeightSender
    RayModuleTransformSender

Weight Receivers
----------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    WeightReceiver
    RayModuleTransformReceiver

Transports
----------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TransportBackend
    MPTransport
    SharedMemTransport
    RayTransport
    RayActorTransport
    RPCTransport
    DistributedTransport

Schemes
-------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    WeightSyncScheme
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
    The `WeightUpdater` is considered legacy as per the 0.11 release and will be deprecated soon.
    The Weight update schemes, which provides more flexibility and a better compatibility with heavy
    weight transfers (e.g., LLMs) is to be preferred.

In distributed and multiprocessed environments, ensuring that all instances of a policy are synchronized with the
latest trained weights is crucial for consistent performance. The API introduces a flexible and extensible
mechanism for updating policy weights across different devices and processes, accommodating various deployment scenarios.

Sending and receiving model weights with WeightUpdaters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The weight synchronization process is facilitated by one dedicated extension point:
:class:`~torchrl.collectors.WeightUpdaterBase`. These base class provides a structured interface for
implementing custom weight update logic, allowing users to tailor the synchronization process to their specific needs.

:class:`~torchrl.collectors.WeightUpdaterBase` handles the distribution of policy weights to
the policy or to remote inference workers, as well as formatting / gathering the weights from a server if necessary.
Every collector -- server or worker -- should have a `WeightUpdaterBase` instance to handle the
weight synchronization with the policy.
Even the simplest collectors use a :class:`~torchrl.collectors.VanillaWeightUpdater` instance to update the policy
state-dict (assuming it is a :class:`~torch.nn.Module` instance).

Extending the Updater Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To accommodate diverse use cases, the API allows users to extend the updater classes with custom implementations.
The goal is to be able to customize the weight sync strategy while leaving the collector and policy implementation
untouched.
This flexibility is particularly beneficial in scenarios involving complex network architectures or specialized hardware
setups.
By implementing the abstract methods in these base classes, users can define how weights are retrieved,
transformed, and applied, ensuring seamless integration with their existing infrastructure.

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
