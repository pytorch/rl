.. currentmodule:: torchrl.collectors

torchrl.collectors package
==========================

Data collectors are somewhat equivalent to pytorch dataloaders, except that (1) they
collect data over non-static data sources and (2) the data is collected using a model
(likely a version of the model that is being trained).

TorchRL's data collectors accept two main arguments: an environment (or a list of
environment constructors) and a policy. They will iteratively execute an environment
step and a policy query over a defined number of steps before delivering a stack of
the data collected to the user. Environments will be reset whenever they reach a done
state, and/or after a predifined number of steps.

Because data collection is a potentially compute heavy process, it is crucial to
configure the execution hyperparameters appropriately.
The first parameter to take into consideration is whether the data collection should
occur serially with the optimization step or in parallel. The :obj:`SyncDataCollector`
class will execute the data collection on the training worker. The :obj:`MultiSyncDataCollector`
will split the workload across an number of workers and aggregate the results that
will be delivered to the training worker. Finally, the :obj:`MultiaSyncDataCollector` will
execute the data collection on several workers and deliver the first batch of results
that it can gather. This execution will occur continuously and concomittantly with
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
executed on CUDA. When multiple inference workers run concomittantly, dispatching
the compute workload across the available devices may speed up the collection or
avoid OOM errors. Finally, the choice of the batch size and passing device (ie the
device where the data will be stored while waiting to be passed to the collection
worker) may also impact the memory management. The key parameters to control are
:obj:`devices` which controls the execution devices (ie the device of the policy)
and :obj:`passing_devices` which will control the device where the environment and
data are stored during a rollout. A good heuristic is usually to use the same device
for storage and compute, which is the default behaviour when only the `devices` argument
is being passed.

Besides those compute parameters, users may choose to configure the following parameters:

- max_frames_per_traj: the number of frames after which a :obj:`env.reset()` is called
- frames_per_batch: the number of frames delivered at each iteration over the collector
- init_random_frames: the number of random steps (steps where :obj:`env.rand_step()` is being called)
- reset_at_each_iter: if :obj:`True`, the environment(s) will be reset after each batch collection
- split_trajs: if :obj:`True`, the trajectories will be split and delivered in a padded tensordict
  along with a :obj:`"mask"` key that will point to a boolean mask representing the valid values.
- exploration_mode: the exploration strategy to be used with the policy.
- reset_when_done: whether environments should be reset when reaching a done state.


Data collectors
---------------


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    MultiSyncDataCollector
    MultiaSyncDataCollector
    SyncDataCollector
    aSyncDataCollector


Helper functions
----------------

.. currentmodule:: torchrl.collectors.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    split_trajectories
