# Common Components

Base classes and common utilities for all loss modules.

| [`LossModule`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)(*args, **kwargs) | A parent class for RL losses. |
| --- | --- |
| [`add_random_module`](generated/torchrl.objectives.add_random_module.html#torchrl.objectives.add_random_module)(module) | Adds a random module to the list of modules that will be detected by [`vmap_randomness()`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule.vmap_randomness) as random. |

## Masked reduction

Batches of padded sequences carry a per-position validity mask, and positions
marked invalid must not contribute to the loss. Every loss reduces through
`LossModule._reduce_loss()`, which reads that mask from the input according
to [`LossModule.loss_mask_key`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule.loss_mask_key):

- `"auto"` (the default) looks for each entry of
`AUTO_LOSS_MASK_KEYS` and ANDs the ones it
finds, so a batch from `SliceSampler` with
`pad_output=True` is handled without any configuration. On data carrying
none of those entries the reduction is unchanged.
- a `NestedKey` restricts masking to that single entry.
- `None` disables masking.

```
loss = PPOLoss(actor, critic)
loss.loss_mask_key = ("my_masks", "valid") # use this entry only
loss.loss_mask_key = None # reduce over every position
```

Masked positions are selected out rather than multiplied by zero, so a
non-finite value at a masked position affects neither the loss nor the
gradients.

| [`AUTO_LOSS_MASK_KEYS`](generated/torchrl.objectives.AUTO_LOSS_MASK_KEYS.html#torchrl.objectives.AUTO_LOSS_MASK_KEYS) | Built-in immutable sequence. |
| --- | --- |

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