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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In distributed and multiprocessed environments, ensuring that all instances of a policy are synchronized with the
latest trained weights is crucial for consistent performance. The API introduces a flexible and extensible
mechanism for updating policy weights across different devices and processes, accommodating various deployment scenarios.

Local and Remote Weight Updaters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The weight synchronization process is facilitated by two main components: :class:`~torchrl.collectors.WeightUpdateReceiverBase`
and :class:`~torchrl.collectors.WeightUpdateSenderBase`. These base classes provide a structured interface for
implementing custom weight update logic, allowing users to tailor the synchronization process to their specific needs.

- :class:`~torchrl.collectors.WeightUpdateReceiverBase`: This component is responsible for updating the policy weights on
  the local inference worker. It is particularly useful when the training and inference occur on the same machine but on
  different devices. Users can extend this class to define how weights are fetched from a server and applied locally.
  It is also the extension point for collectors where the workers need to ask for weight updates (in contrast with
  situations where the server decides when to update the worker policies).
- :class:`~torchrl.collectors.WeightUpdateSenderBase`: This component handles the distribution of policy weights to
  remote inference workers. It is essential in distributed systems where multiple workers need to be kept in sync with
  the central policy. Users can extend this class to implement custom logic for synchronizing weights across a network of
  devices or processes.

Extending the Updater Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To accommodate diverse use cases, the API allows users to extend the updater classes with custom implementations.
This flexibility is particularly beneficial in scenarios involving complex network architectures or specialized hardware
setups. By implementing the abstract methods in these base classes, users can define how weights are retrieved,
transformed, and applied, ensuring seamless integration with their existing infrastructure.

Default Implementations
~~~~~~~~~~~~~~~~~~~~~~~

For common scenarios, the API provides default implementations of these updaters, such as
:class:`~torchrl.collectors.VanillaLocalWeightUpdater`, :class:`~torchrl.collectors.MultiProcessedRemoteWeightUpdate`,
:class:`~torchrl.collectors.RayWeightUpdateSender`, :class:`~torchrl.collectors.RPCWeightUpdateSender`, and
:class:`~torchrl.collectors.DistributedWeightUpdateSender`.
These implementations cover a range of typical deployment configurations, from single-device setups to large-scale
distributed systems.

Practical Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

When designing a system that leverages this API, consider the following:

- Network Latency: In distributed environments, network latency can impact the speed of weight updates. Ensure that your
  implementation accounts for potential delays and optimizes data transfer where possible.
- Consistency: Ensure that all workers receive the updated weights in a timely manner to maintain consistency across
  the system. This is particularly important in reinforcement learning scenarios where stale weights can lead to
  suboptimal policy performance.
- Scalability: As your system grows, the weight synchronization mechanism should scale efficiently. Consider the
  overhead of broadcasting weights to a large number of workers and optimize the process to minimize bottlenecks.

By leveraging the API, users can achieve robust and efficient weight synchronization across a variety of deployment
scenarios, ensuring that their policies remain up-to-date and performant.

.. currentmodule:: torchrl.collectors

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    WeightUpdateReceiverBase
    WeightUpdateSenderBase
    VanillaLocalWeightUpdater
    MultiProcessedRemoteWeightUpdate
    RayWeightUpdateSender
    DistributedWeightUpdateSender
    RPCWeightUpdateSender

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

LLM Collectors
---------------------------
TorchRL also provides a data collectors for large language models. These collectors
are meant to include a subset of the functionality of other data collectors, targeted
at supporting researchers in fine-tuning large language models.  These classes 
currently derive from the :class:`~torchrl.collectors.SyncDataCollector` class.
These classes are experimental and subject to change.

.. currentmodule:: torchrl.collectors.llm_collectors

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    LLMCollector

Helper functions
----------------

.. currentmodule:: torchrl.collectors.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    split_trajectories
