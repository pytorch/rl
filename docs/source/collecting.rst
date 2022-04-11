.. collecting:

Collecting data
===============

TorchRL is packed with a set of serial and parallel data collectors that can work in a synchronous or asynchronous way.
In essence, a data collector is a modules that executes iteratively a policy based on an observation on a simulated
environments.
Besides this, several environments can be executed in parallel, providing further flexibility in the way those data are
collected.

Three collector classes are provided by torchrl:

- ``torchrl.collectors.SyncDataCollector`` is the base data collector class, that simply executes the policy in a
  single environment.
- ``torchrl.collectors.MultiaSyncDataCollector`` will instantiate a series of ``SyncDataCollector`` and execute them
  asynchroneously. At each iteration of the data collector, the resulting `TensorDict` will come from a single worker,
  leaving the other workers busy collecting their own batch of data. The asynchronous nature of this collector makes it
  well suited for offline RL algorithms, as the policy used to collect data could be lagged from the policy being
  trained on the main process.
- ``torchrl.collectors.MultiSyncDataCollector`` will collect an equal share of frames from each worker and yield a
  single `TensorDict` containing this stack of data. This collector is well suited for online RL algorithms.
- ``torchrl.collectors.aSyncDataCollector``

To build a data collector, one needs to instantiate a policy module, design a (list of) environment building functions
and decide upon a series of hyperparameters:

- the number of frames that must be collected at each iteration of the collector;
- the maximum number of frames collected by the collector;

The following hyperparameters are only optional:
- the device on which the collected data will be stored;
- the device where the policy will be placed;
