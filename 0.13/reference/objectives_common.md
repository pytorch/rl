# Common Components

Base classes and common utilities for all loss modules.

| [`LossModule`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)(*args, **kwargs) | A parent class for RL losses. |
| --- | --- |
| [`add_random_module`](generated/torchrl.objectives.add_random_module.html#torchrl.objectives.add_random_module)(module) | Adds a random module to the list of modules that will be detected by [`vmap_randomness()`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule.vmap_randomness) as random. |

## Value Estimators

| [`ValueEstimatorBase`](generated/torchrl.objectives.value.ValueEstimatorBase.html#torchrl.objectives.value.ValueEstimatorBase)(*args, **kwargs) | An abstract parent class for value function modules. |
| --- | --- |
| [`TD0Estimator`](generated/torchrl.objectives.value.TD0Estimator.html#torchrl.objectives.value.TD0Estimator)(*args, **kwargs) | Temporal Difference (TD(0)) estimate of advantage function. |
| [`TD1Estimator`](generated/torchrl.objectives.value.TD1Estimator.html#torchrl.objectives.value.TD1Estimator)(*args, **kwargs) | \(\infty\)-Temporal Difference (TD(1)) estimate of advantage function. |
| [`TDLambdaEstimator`](generated/torchrl.objectives.value.TDLambdaEstimator.html#torchrl.objectives.value.TDLambdaEstimator)(*args, **kwargs) | TD(\(\lambda\)) estimate of advantage function. |
| [`GAE`](generated/torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE)(*args, **kwargs) | A class wrapper around the generalized advantage estimate functional. |
| [`VTrace`](generated/torchrl.objectives.value.VTrace.html#torchrl.objectives.value.VTrace)(*args, **kwargs) | A class wrapper around V-Trace estimate functional. |
| [`MultiAgentGAE`](generated/torchrl.objectives.value.MultiAgentGAE.html#torchrl.objectives.value.MultiAgentGAE)(*args, **kwargs) | Multi-agent Generalized Advantage Estimator. |

| [`ValueEstimators`](generated/torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators)(value[, names, module, ...]) | Value function enumerator for custom-built estimators. |
| --- | --- |