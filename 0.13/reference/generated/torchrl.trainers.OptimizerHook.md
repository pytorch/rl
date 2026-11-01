# OptimizerHook

*class*torchrl.trainers.OptimizerHook(*optimizer: [Optimizer](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)*, *loss_components: Sequence[str] | None = None*)[[source]](../../_modules/torchrl/trainers/trainers.html#OptimizerHook)

Add an optimizer for one or more loss components.

Deprecated since version ``OptimizerHook``: will be replaced by
`DefaultOptimizationStepper` in a future release.

Parameters:

- **optimizer** (*optim.Optimizer*) - An optimizer to apply to the loss_components.
- **loss_components** (*Sequence**[**str**]**,**optional*) - The keys in the loss TensorDict
for which the optimizer should be appled to the respective values.
If omitted, the optimizer is applied to all components with the
names starting with loss_.

Examples

```
>>> optimizer_hook = OptimizerHook(optimizer, ["loss_actor"])
>>> trainer.register_op("optimizer", optimizer_hook)
```

register(*trainer*, *name='optimizer'*) → None[[source]](../../_modules/torchrl/trainers/trainers.html#OptimizerHook.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.