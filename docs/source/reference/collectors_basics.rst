.. currentmodule:: torchrl.collectors

.. _ref_collectors_basics:

Collector Basics
================

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
occur serially with the optimization step or in parallel. Construct all of these
topologies through :class:`Collector`. ``Collector(...)`` executes collection on
the training worker. ``Collector(num_collectors=N, sync=True)`` splits the
workload across local processes and aggregates their results (its concrete return
type is :class:`MultiSyncCollector`). ``Collector(num_collectors=N, sync=False)``
delivers the first worker result it can gather (its concrete return type is
:class:`MultiAsyncCollector`). This asynchronous execution occurs continuously and concomitantly with
the training of the networks: this implies that the weights of the policy that
is used for the data collection may slightly lag the configuration of the policy
on the training worker. Therefore, although this class may be the fastest to collect
data, it comes at the price of being suitable only in settings where it is acceptable
to gather data asynchronously (e.g. off-policy RL or curriculum RL).
For worker-executed rollouts (``Collector(num_collectors=N)`` with either
``sync=True`` or ``sync=False``)
it is necessary to synchronise the weights of the remote policy with the weights
from the training worker using either the :meth:`collector.update_policy_weights_` or
by setting ``update_at_each_batch=True`` in the constructor.

The second parameter to consider (in the remote settings) is the device where the
data will be collected and the device where the environment and policy operations
will be executed. For instance, a policy executed on CPU may be slower than one
executed on CUDA. When multiple inference workers run concomitantly, dispatching
the compute workload across the available devices may speed up the collection or
avoid OOM errors. Finally, the choice of the batch size and passing device (ie the
device where the data will be stored while waiting to be passed to the collection
worker) may also impact the memory management. The key parameters to control are
``devices`` which controls the execution devices (ie the device of the policy)
and ``storing_device`` which will control the device where the environment and
data are stored during a rollout. A good heuristic is usually to use the same device
for storage and compute, which is the default behavior when only the ``devices`` argument
is being passed.

Besides those compute parameters, users may choose to configure the following parameters:

- max_frames_per_traj: the number of frames after which a :meth:`env.reset` is called
- frames_per_batch: the number of frames delivered at each iteration over the collector
- init_random_frames: the number of random steps (steps where :meth:`env.rand_step` is being called)
- reset_at_each_iter: if ``True``, the environment(s) will be reset after each batch collection
- split_trajs: if ``True``, the trajectories will be split and delivered in a padded tensordict
  along with a ``"mask"`` key that will point to a boolean mask representing the valid values.
- exploration_type: the exploration strategy to be used with the policy.
- reset_when_done: whether environments should be reset when reaching a done state.
- track_traj_ids: if ``True`` (default), the collector writes a unique
  ``("collector", "traj_ids")`` integer per frame and updates it on every
  env reset.  Setting it to ``False`` skips the per-step bookkeeping —
  useful when neither :func:`~torchrl.collectors.utils.split_trajectories`
  nor :class:`~torchrl.data.replay_buffers.SliceSampler` are used downstream.  Note that
  ``split_trajs=True`` requires ``track_traj_ids=True``.

Trajectory IDs
--------------

When ``track_traj_ids=True``, every frame yielded by a collector carries a
``("collector", "traj_ids")`` integer that uniquely identifies the trajectory
it belongs to.  IDs are drawn from a process-local pool so they remain
unique across resets within a single worker.

This is what makes trajectory-aware downstream consumers possible:

- :class:`~torchrl.data.replay_buffers.SliceSampler` draws whole trajectories (or slices of
  them) from a replay buffer by grouping frames by ``traj_ids``.
- :func:`~torchrl.collectors.utils.split_trajectories` reshapes a flat batch
  into a padded ``[n_trajs, max_len]`` tensordict using ``traj_ids`` to find
  the boundaries.

Set ``track_traj_ids=False`` only when the downstream consumer does not need
trajectory identity — e.g. step-level off-policy training that samples
i.i.d. transitions from a uniform buffer.

See :ref:`ref_collectors_internals` for the per-step bookkeeping that
maintains these IDs.

Collectors and batch size
-------------------------

Because each collector has its own way of organizing the environments that are
run within, the data will come with different batch-size depending on how
the specificities of the collector. The following table summarizes what is to
be expected when collecting data:


+--------------------+---------------------+--------------------------------------------+------------------------------+
|                    | Collector (direct)  | Collector(num_collectors=B, sync=True)     |Collector(..., sync=False)    |
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

.. warning:: ``Collector(num_collectors=N, sync=True)`` (a
  :class:`~torchrl.collectors.MultiSyncCollector`) should not be
  used with ``cat_results=0``, as the data will be stacked along the batch
  dimension with batched environment, or the time dimension for single environments,
  which can introduce some confusion when swapping one with the other.
  ``cat_results="stack"`` is a better and more consistent way of interacting
  with the environments as it will keep each dimension separate, and provide
  better interchangeability between configurations, collector classes and other
  components.

Whereas ``Collector(num_collectors=B, sync=True)`` has a dimension corresponding
to the number of sub-collectors being run (``B``), the asynchronous
``Collector(num_collectors=B, sync=False)`` form does not. The concrete
implementations are :class:`~torchrl.collectors.MultiSyncCollector` and
:class:`~torchrl.collectors.MultiAsyncCollector`, respectively. This
is easily understood when considering that :class:`~torchrl.collectors.MultiAsyncCollector`
delivers batches of data on a first-come, first-serve basis, whereas
:class:`~torchrl.collectors.MultiSyncCollector` gathers data from
each sub-collector before delivering it.

Collectors and policy copies
----------------------------

When passing a policy to a collector, we can choose the device on which this policy will be run. This can be used to
keep the training version of the policy on a device and the inference version on another. For example, if you have two
CUDA devices, it may be wise to train on one device and execute the policy for inference on the other. If that is the
case, a :meth:`~torchrl.collectors.Collector.update_policy_weights_` can be used to copy the parameters from one
device to the other (if no copy is required, this method is a no-op).

Since the goal is to avoid calling `policy.to(policy_device)` explicitly, the collector will do a deepcopy of the
policy structure and copy the parameters placed on the new device during instantiation if necessary.
Since not all policies support deepcopies (e.g., policies using CUDA graphs or relying on third-party libraries), we
try to limit the cases where a deepcopy will be executed. The following chart shows when this will occur.

.. figure:: /_static/img/collector-copy.png

   Policy copy decision tree in Collectors.
