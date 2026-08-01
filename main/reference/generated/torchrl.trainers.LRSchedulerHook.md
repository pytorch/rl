# LRSchedulerHook

*class*torchrl.trainers.LRSchedulerHook(*scheduler: [LRScheduler](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html#torch.optim.lr_scheduler.LRScheduler)*, *interval: Literal['batch', 'optim'] = 'batch'*)[[source]](../../_modules/torchrl/trainers/trainers.html#LRSchedulerHook)

A hook that steps a learning-rate scheduler during training.

Parameters:

- **scheduler** ([*torch.optim.lr_scheduler.LRScheduler*](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html#torch.optim.lr_scheduler.LRScheduler)) - the scheduler to step.
- **interval** (*Literal**[**"batch"**,**"optim"**]**,**optional*) - `"batch"` to step the
scheduler once per collected batch, or `"optim"` to step it after
every optimization step. With `"optim"`, the number of scheduler
steps per collected batch scales with `num_epochs` and the number
of sub-batches per batch. Defaults to `"batch"`.

Once registered with a trainer, the hook only steps the scheduler when at
least one optimization step has run since its last call, so the learning
rate is not decayed during warmup phases (e.g. while
`collector.init_random_frames` has not been reached).

Examples

```
>>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
>>> lr_scheduler_hook = LRSchedulerHook(scheduler)
>>> lr_scheduler_hook.register(trainer)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'lr_scheduler'*)[[source]](../../_modules/torchrl/trainers/trainers.html#LRSchedulerHook.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.