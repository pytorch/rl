# ClearCudaCache

*class*torchrl.trainers.ClearCudaCache(*interval: int*)[[source]](../../_modules/torchrl/trainers/trainers.html#ClearCudaCache)

Clears cuda cache at a given interval.

Examples

```
>>> clear_cuda = ClearCudaCache(100)
>>> trainer.register_op("pre_optim_steps", clear_cuda)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'clear_cuda_cache'*)[[source]](../../_modules/torchrl/trainers/trainers.html#ClearCudaCache.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.