# ValueEstimatorHook

*class*torchrl.trainers.ValueEstimatorHook(*value_estimator: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*)[[source]](../../_modules/torchrl/trainers/trainers.html#ValueEstimatorHook)

A hook that computes value estimates over a collected batch.

Wraps a value estimator module such as [`GAE`](torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE)
and applies it to the whole collected batch at the `pre_epoch` stage, so
that advantage and value-target entries are available when the loss module
consumes sub-batches during optimization.

In async-collection mode the training loop has no batch to hand to the
`pre_epoch` stage (`batch` is `None`); the hook then passes the batch
through untouched, and value estimates are expected to be computed
elsewhere (e.g. by a replay-buffer transform).

Parameters:

**value_estimator** (*Callable**[**[**TensorDictBase**]**,**TensorDictBase**]*) - the value
estimator to apply to the collected batch, e.g. an instance of
[`GAE`](torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE).

Examples

```
>>> gae = GAE(gamma=0.99, lmbda=0.95, value_network=critic, average_gae=True)
>>> value_estimator_hook = ValueEstimatorHook(gae)
>>> value_estimator_hook.register(trainer)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'value_estimator'*)[[source]](../../_modules/torchrl/trainers/trainers.html#ValueEstimatorHook.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.