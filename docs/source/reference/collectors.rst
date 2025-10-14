.. currentmodule:: torchrl.collectors

torchrl.collectors package
==========================

.. _ref_collectors:

Data collectors are somewhat equivalent to pytorch dataloaders, except that (1) they
collect data over non-static data sources and (2) the data is collected using a model
(likely a version of the model that is being trained).

TorchRL's data collectors accept two main arguments: an environment (or a list of
environment constructors) and a policy. They will iteratively execute an environment
step and a policy query over a defined number of steps before delivering a stack of
the data collected to the user. Environments will be reset whenever they reach a done
state, and/or after a predefined number of steps.

Because data collection is a potentially compute heavy process, it is crucial to
configure the execution hyperparameters appropriately.
The first parameter to take into consideration is whether the data collection should
occur serially with the optimization step or in parallel. The :obj:`SyncDataCollector`
class will execute the data collection on the training worker. The :obj:`MultiSyncDataCollector`
will split the workload across an number of workers and aggregate the results that
will be delivered to the training worker. Finally, the :obj:`MultiaSyncDataCollector` will
execute the data collection on several workers and deliver the first batch of results
that it can gather. This execution will occur continuously and concomitantly with
the training of the networks: this implies that the weights of the policy that
is used for the data collection may slightly lag the configuration of the policy
on the training worker. Therefore, although this class may be the fastest to collect
data, it comes at the price of being suitable only in settings where it is acceptable
to gather data asynchronously (e.g. off-policy RL or curriculum RL).
For remotely executed rollouts (:obj:`MultiSyncDataCollector` or :obj:`MultiaSyncDataCollector`)
it is necessary to synchronise the weights of the remote policy with the weights
from the training worker using either the `collector.update_policy_weights_()` or
by setting `update_at_each_batch=True` in the constructor.

The second parameter to consider (in the remote settings) is the device where the
data will be collected and the device where the environment and policy operations
will be executed. For instance, a policy executed on CPU may be slower than one
executed on CUDA. When multiple inference workers run concomitantly, dispatching
the compute workload across the available devices may speed up the collection or
avoid OOM errors. Finally, the choice of the batch size and passing device (ie the
device where the data will be stored while waiting to be passed to the collection
worker) may also impact the memory management. The key parameters to control are
:obj:`devices` which controls the execution devices (ie the device of the policy)
and :obj:`storing_device` which will control the device where the environment and
data are stored during a rollout. A good heuristic is usually to use the same device
for storage and compute, which is the default behavior when only the `devices` argument
is being passed.

Besides those compute parameters, users may choose to configure the following parameters:

- max_frames_per_traj: the number of frames after which a :obj:`env.reset()` is called
- frames_per_batch: the number of frames delivered at each iteration over the collector
- init_random_frames: the number of random steps (steps where :obj:`env.rand_step()` is being called)
- reset_at_each_iter: if :obj:`True`, the environment(s) will be reset after each batch collection
- split_trajs: if :obj:`True`, the trajectories will be split and delivered in a padded tensordict
  along with a :obj:`"mask"` key that will point to a boolean mask representing the valid values.
- exploration_type: the exploration strategy to be used with the policy.
- reset_when_done: whether environments should be reset when reaching a done state.

Collectors and batch size
-------------------------

Because each collector has its own way of organizing the environments that are
run within, the data will come with different batch-size depending on how
the specificities of the collector. The following table summarizes what is to
be expected when collecting data:


+--------------------+---------------------+--------------------------------------------+------------------------------+
|                    | SyncDataCollector   |       MultiSyncDataCollector (n=B)         |MultiaSyncDataCollector (n=B) |
+====================+=====================+=============+==============+===============+==============================+
|   `cat_results`    |          NA         |  `"stack"`  |      `0`     |      `-1`     |             NA               |
+--------------------+---------------------+-------------+--------------+---------------+------------------------------+
|     Single env     |         [T]         |   `[B, T]`  |  `[B*(T//B)` |  `[B*(T//B)]` |              [T]             |
+--------------------+---------------------+-------------+--------------+---------------+------------------------------+
| Batched env (n=P)  |       [P, T]        | `[B, P, T]` |  `[B * P, T]`|  `[P, T * B]` |            [P, T]            |
+--------------------+---------------------+-------------+--------------+---------------+------------------------------+

In each of these cases, the last dimension (``T`` for ``time``) is adapted such
that the batch size equals the ``frames_per_batch`` argument passed to the
collector.

.. warning:: :class:`~torchrl.collectors.MultiSyncDataCollector` should not be
  used with ``cat_results=0``, as the data will be stacked along the batch
  dimension with batched environment, or the time dimension for single environments,
  which can introduce some confusion when swapping one with the other.
  ``cat_results="stack"`` is a better and more consistent way of interacting
  with the environments as it will keep each dimension separate, and provide
  better interchangeability between configurations, collector classes and other
  components.

Whereas :class:`~torchrl.collectors.MultiSyncDataCollector`
has a dimension corresponding to the number of sub-collectors being run (``B``),
:class:`~torchrl.collectors.MultiaSyncDataCollector` doesn't. This
is easily understood when considering that :class:`~torchrl.collectors.MultiaSyncDataCollector`
delivers batches of data on a first-come, first-serve basis, whereas
:class:`~torchrl.collectors.MultiSyncDataCollector` gathers data from
each sub-collector before delivering it.

Collectors and policy copies
----------------------------

When passing a policy to a collector, we can choose the device on which this policy will be run. This can be used to
keep the training version of the policy on a device and the inference version on another. For example, if you have two
CUDA devices, it may be wise to train on one device and execute the policy for inference on the other. If that is the
case, a :meth:`~torchrl.collectors.DataCollector.update_policy_weights_` can be used to copy the parameters from one
device to the other (if no copy is required, this method is a no-op).

Since the goal is to avoid calling `policy.to(policy_device)` explicitly, the collector will do a deepcopy of the
policy structure and copy the parameters placed on the new device during instantiation if necessary.
Since not all policies support deepcopies (e.g., policies using CUDA graphs or relying on third-party libraries), we
try to limit the cases where a deepcopy will be executed. The following chart shows when this will occur.

.. figure:: /_static/img/collector-copy.png

   Policy copy decision tree in Collectors.

Weight Synchronization in Distributed Environments
--------------------------------------------------

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

Weight Synchronization API
~~~~~~~~~~~~~~~~~~~~~~~~~~

The weight synchronization API provides a simple, modular approach to updating model weights across
distributed collectors. This system is designed to handle the complexities of modern RL setups where multiple
models may need to be synchronized independently.

Overview
^^^^^^^^

In reinforcement learning, particularly with multi-process data collection, it's essential to keep the inference
policies synchronized with the latest trained weights. The API addresses this challenge through a clean
separation of concerns, where four classes are involved:

- **Configuration**: :class:`~torchrl.weight_update.weight_sync_schemes.WeightSyncScheme` objects define *what* to synchronize and *how*. For DataCollectors, this is
  your main entrypoint to configure the weight synchronization.
- **Sending**: :class:`~torchrl.weight_update.weight_sync_schemes.WeightSender` handles distributing weights from the main process to workers.
- **Receiving**: :class:`~torchrl.weight_update.weight_sync_schemes.WeightReceiver` handles applying weights in worker processes.
- **Transport**: Backend-specific communication mechanisms (pipes, shared memory, Ray, RPC)

The following diagram shows the different classes involved in the weight synchronization process:

.. aafig::
    :aspect: 60
    :scale: 130
    :proportional:

    INITIALIZATION PHASE
    ====================

                        WeightSyncScheme
                        +------------------+
                        |                  |
                        | Configuration:   |
                        | - strategy       |
                        | - transport_type |
                        |                  |
                        +--------+---------+
                                 |
                    +------------+-------------+
                    |                          |
                creates                    creates
                    |                          |
                    v                          v
            Main Process                 Worker Process
            +--------------+             +---------------+
            | WeightSender |             | WeightReceiver|
            |              |             |               |
            | - strategy   |             | - strategy    |
            | - transports |             | - transport   |
            | - model_ref  |             | - model_ref   |
            |              |             |               |
            | Registers:   |             | Registers:    |
            | - model      |             | - model       |
            | - workers    |             | - transport   |
            +--------------+             +---------------+
                    |                            |
                    |   Transport Layer          |
                    |   +----------------+       |
                    +-->+ MPTransport    |<------+
                    |   | (pipes)        |       |
                    |   +----------------+       |
                    |   +----------------+       |
                    +-->+ SharedMemTrans |<------+
                    |   | (shared mem)   |       |
                    |   +----------------+       |
                    |   +----------------+       |
                    +-->+ RayTransport   |<------+
                        | (Ray store)    |
                        +----------------+


    SYNCHRONIZATION PHASE
    =====================

        Main Process                                    Worker Process
        
    +-------------------+                           +-------------------+
    | WeightSender      |                           | WeightReceiver    |
    |                   |                           |                   |
    | 1. Extract        |                           | 4. Poll transport |
    |    weights from   |                           |    for weights    |
    |    model using    |                           |                   |
    |    strategy       |                           |                   |
    |                   |    2. Send via            |                   |
    | +-------------+   |       Transport           | +--------------+  |
    | | Strategy    |   |    +------------+         | | Strategy     |  |
    | | extract()   |   |    |            |         | | apply()      |  |
    | +-------------+   +----+ Transport  +-------->+ +--------------+  |
    |        |          |    |            |         |        |          |
    |        v          |    +------------+         |        v          |
    | +-------------+   |                           | +--------------+  |
    | | Model       |   |                           | | Model        |  |
    | | (source)    |   |  3. Ack (optional)        | | (dest)       |  |
    | +-------------+   | <-----------------------+ | +--------------+  |
    |                   |                           |                   |
    +-------------------+                           | 5. Apply weights  |
                                                    |    to model using |
                                                    |    strategy       |
                                                    +-------------------+

Key Challenges Addressed
^^^^^^^^^^^^^^^^^^^^^^^^^

Modern RL training often involves multiple models that need independent synchronization:

1. **Multiple Models Per Collector**: A collector might need to update:
   
   - The main policy network
   - A value network in a Ray actor within the replay buffer
   - Models embedded in the environment itself
   - Separate world models or auxiliary networks

2. **Different Update Strategies**: Each model may require different synchronization approaches:
   
   - Full state_dict transfer vs. TensorDict-based updates
   - Different transport mechanisms (multiprocessing pipes, shared memory, Ray object store, collective communication, RDMA, etc.)
   - Varied update frequencies

3. **Worker-Agnostic Updates**: Some models (like those in shared Ray actors) shouldn't be tied to
   specific worker indices, requiring a more flexible update mechanism.

Architecture
^^^^^^^^^^^^

The API follows a scheme-based design where users specify synchronization requirements upfront,
and the collector handles the orchestration transparently:

.. aafig::
    :aspect: 60
    :scale: 130
    :proportional:

      Main Process                 Worker Process 1         Worker Process 2
      
    +-----------------+            +---------------+        +---------------+
    | Collector       |            | Collector     |        | Collector     |
    |                 |            |               |        |               |
    | Models:         |            | Models:       |        | Models:       |
    |  +----------+   |            |  +--------+   |        |  +--------+   |
    |  | Policy A |   |            |  |Policy A|   |        |  |Policy A|   |
    |  +----------+   |            |  +--------+   |        |  +--------+   |
    |  +----------+   |            |  +--------+   |        |  +--------+   |
    |  | Model  B |   |            |  |Model  B|   |        |  |Model  B|   |
    |  +----------+   |            |  +--------+   |        |  +--------+   |
    |                 |            |               |        |               |
    | Weight Senders: |            | Weight        |        | Weight        |
    |  +----------+   |            | Receivers:    |        | Receivers:    |
    |  | Sender A +---+------------+->Receiver A   |        |  Receiver A   |
    |  +----------+   |            |               |        |               |
    |  +----------+   |            |  +--------+   |        |  +--------+   |
    |  | Sender B +---+------------+->Receiver B   |        |  Receiver B   |
    |  +----------+   |  Pipes     |               |  Pipes |               |
    +-----------------+            +-------+-------+        +-------+-------+
           ^                               ^                        ^
           |                               |                        |
           | update_policy_weights_()      |   Apply weights        |
           |                               |                        |
    +------+-------+                       |                        |
    | User Code    |                       |                        |
    | (Training)   |                       |                        |
    +--------------+                       +------------------------+

The weight synchronization flow:

1. **Initialization**: User creates ``weight_sync_schemes`` dict mapping model IDs to schemes
2. **Registration**: Collector creates ``WeightSender`` for each model in the main process
3. **Worker Setup**: Each worker creates corresponding ``WeightReceiver`` instances  
4. **Synchronization**: Calling ``update_policy_weights_()`` triggers all senders to push weights
5. **Application**: Receivers automatically apply weights to their registered models

Available Classes
^^^^^^^^^^^^^^^^^

**Synchronization Schemes** (User-Facing Configuration):

- :class:`~torchrl.weight_update.weight_sync_schemes.WeightSyncScheme`: Base class for schemes
- :class:`~torchrl.weight_update.weight_sync_schemes.MultiProcessWeightSyncScheme`: For multiprocessing with pipes
- :class:`~torchrl.weight_update.weight_sync_schemes.SharedMemWeightSyncScheme`: For shared memory synchronization
- :class:`~torchrl.weight_update.weight_sync_schemes.RayWeightSyncScheme`: For Ray-based distribution
- :class:`~torchrl.weight_update.weight_sync_schemes.NoWeightSyncScheme`: Dummy scheme for no synchronization

**Internal Classes** (Automatically Managed):

- :class:`~torchrl.weight_update.weight_sync_schemes.WeightSender`: Sends weights to all workers for one model
- :class:`~torchrl.weight_update.weight_sync_schemes.WeightReceiver`: Receives and applies weights in worker
- :class:`~torchrl.weight_update.weight_sync_schemes.TransportBackend`: Communication layer abstraction

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

    from torchrl.collectors import MultiSyncDataCollector
    from torchrl.weight_update.weight_sync_schemes import MultiProcessWeightSyncScheme

    # Define synchronization for multiple models
    weight_sync_schemes = {
        "policy": MultiProcessWeightSyncScheme(strategy="tensordict"),
        "value_net": MultiProcessWeightSyncScheme(strategy="state_dict"),
    }

    collector = MultiSyncDataCollector(
        create_env_fn=[make_env] * 4,
        policy=policy,
        frames_per_batch=1000,
        weight_sync_schemes=weight_sync_schemes,  # Pass schemes dict
    )

    # Single call updates all registered models across all workers
    for i, batch in enumerate(collector):
        # Training step
        loss = train(batch)
        
        # Sync all models with one call
        collector.update_policy_weights_(policy)

The collector automatically:

- Creates ``WeightSender`` instances in the main process for each model
- Creates ``WeightReceiver`` instances in each worker process
- Resolves models by ID (e.g., ``"policy"`` → ``collector.policy``)
- Handles transport setup and communication
- Applies weights using the appropriate strategy (state_dict vs tensordict)

API Reference
^^^^^^^^^^^^^

.. currentmodule:: torchrl.weight_update.weight_sync_schemes

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    WeightSyncScheme
    MultiProcessWeightSyncScheme
    SharedMemWeightSyncScheme
    RayWeightSyncScheme
    NoWeightSyncScheme
    WeightSender
    WeightReceiver

Collectors and replay buffers interoperability
----------------------------------------------

In the simplest scenario where single transitions have to be sampled
from the replay buffer, little attention has to be given to the way
the collector is built. Flattening the data after collection will
be a sufficient preprocessing step before populating the storage:

    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N),
    ...     transform=lambda data: data.reshape(-1))
    >>> for data in collector:
    ...     memory.extend(data)

If trajectory slices have to be collected, the recommended way to achieve this is to create
a multidimensional buffer and sample using the :class:`~torchrl.data.replay_buffers.SliceSampler`
sampler class. One must ensure that the data passed to the buffer is properly shaped, with the
``time`` and ``batch`` dimensions clearly separated. In practice, the following configurations
will work:

    >>> # Single environment: no need for a multi-dimensional buffer
    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N),
    ...     sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
    ... )
    >>> collector = SyncDataCollector(env, policy, frames_per_batch=N, total_frames=-1)
    >>> for data in collector:
    ...     memory.extend(data)
    >>> # Batched environments: a multi-dim buffer is required
    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N, ndim=2),
    ...     sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
    ... )
    >>> env = ParallelEnv(4, make_env)
    >>> collector = SyncDataCollector(env, policy, frames_per_batch=N, total_frames=-1)
    >>> for data in collector:
    ...     memory.extend(data)
    >>> # MultiSyncDataCollector + regular env: behaves like a ParallelEnv if cat_results="stack"
    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N, ndim=2),
    ...     sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
    ... )
    >>> collector = MultiSyncDataCollector([make_env] * 4,
    ...     policy,
    ...     frames_per_batch=N,
    ...     total_frames=-1,
    ...     cat_results="stack")
    >>> for data in collector:
    ...     memory.extend(data)
    >>> # MultiSyncDataCollector + parallel env: the ndim must be adapted accordingly
    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N, ndim=3),
    ...     sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
    ... )
    >>> collector = MultiSyncDataCollector([ParallelEnv(2, make_env)] * 4,
    ...     policy,
    ...     frames_per_batch=N,
    ...     total_frames=-1,
    ...     cat_results="stack")
    >>> for data in collector:
    ...     memory.extend(data)

Using replay buffers that sample trajectories with :class:`~torchrl.collectors.MultiSyncDataCollector`
isn't currently fully supported as the data batches can come from any worker and in most cases consecutive
batches written in the buffer won't come from the same source (thereby interrupting the trajectories).

Running the Collector Asynchronously
------------------------------------

Passing replay buffers to a collector allows us to start the collection and get rid of the iterative nature of the
collector.
If you want to run a data collector in the background, simply run :meth:`~torchrl.DataCollectorBase.start`:

    >>> collector = SyncDataCollector(..., replay_buffer=rb) # pass your replay buffer
    >>> collector.start()
    >>> # little pause
    >>> time.sleep(10)
    >>> # Start training
    >>> for i in range(optim_steps):
    ...     data = rb.sample()  # Sampling from the replay buffer
    ...     # rest of the training loop

Single-process collectors (:class:`~torchrl.collectors.SyncDataCollector`) will run the process using multithreading,
so be mindful of Python's GIL and related multithreading restrictions.

Multiprocessed collectors will on the other hand let the child processes handle the filling of the buffer on their own,
which truly decouples the data collection and training.

Data collectors that have been started with `start()` should be shut down using
:meth:`~torchrl.DataCollectorBase.async_shutdown`.

.. warning:: Running a collector asynchronously decouples the collection from training, which means that the training
    performance may be drastically different depending on the hardware, load and other factors (although it is generally
    expected to provide significant speed-ups). Make sure you understand how this may affect your algorithm and if it
    is a legitimate thing to do! (For example, on-policy algorithms such as PPO should not be run asynchronously
    unless properly benchmarked).

Single node data collectors
---------------------------
.. currentmodule:: torchrl.collectors

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    DataCollectorBase
    SyncDataCollector
    MultiSyncDataCollector
    MultiaSyncDataCollector
    aSyncDataCollector


Distributed data collectors
---------------------------
TorchRL provides a set of distributed data collectors. These tools support
multiple backends (``'gloo'``, ``'nccl'``, ``'mpi'`` with the :class:`~.DistributedDataCollector`
or PyTorch RPC with :class:`~.RPCDataCollector`) and launchers (``'ray'``,
``submitit`` or ``torch.multiprocessing``).
They can be efficiently used in synchronous or asynchronous mode, on a single
node or across multiple nodes.

*Resources*: Find examples for these collectors in the
`dedicated folder <https://github.com/pytorch/rl/examples/distributed/collectors>`_.

.. note::
  *Choosing the sub-collector*: All distributed collectors support the various single machine collectors.
  One may wonder why using a :class:`MultiSyncDataCollector` or a :class:`~torchrl.envs.ParallelEnv`
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

.. currentmodule:: torchrl.collectors.distributed

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    DistributedDataCollector
    RPCDataCollector
    DistributedSyncDataCollector
    submitit_delayed_launcher
    RayCollector

Helper functions
----------------

.. currentmodule:: torchrl.collectors.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    split_trajectories
