# BatchSubSampler

*class*torchrl.trainers.BatchSubSampler(*batch_size: int*, *sub_traj_len: int = 0*, *min_sub_traj_len: int = 0*)[[source]](../../_modules/torchrl/trainers/trainers.html#BatchSubSampler)

Data subsampler for online RL sota-implementations.

This class subsamples a part of a whole batch of data just collected from the
environment.

Parameters:

- **batch_size** (*int*) - sub-batch size to collect. The provided batch size
must be equal to the total number of items in the output tensordict,
which will have size [batch_size // sub_traj_len, sub_traj_len].
- **sub_traj_len** (*int**,**optional*) - length of the trajectories that
sub-samples must have in online settings. Default is -1 (i.e.
takes the full length of the trajectory)
- **min_sub_traj_len** (*int**,**optional*) - minimum value of `sub_traj_len`, in
case some elements of the batch contain few steps.
Default is -1 (i.e. no minimum value)

Examples

```
>>> td = TensorDict(
... {
... key1: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
... key2: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
... },
... [2, 10],
... )
>>> trainer.register_op(
... "process_optim_batch",
... BatchSubSampler(batch_size=batch_size, sub_traj_len=sub_traj_len),
... )
>>> td_out = trainer._process_optim_batch_hook(td)
>>> assert td_out.shape == torch.Size([batch_size // sub_traj_len, sub_traj_len])
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'batch_subsampler'*)[[source]](../../_modules/torchrl/trainers/trainers.html#BatchSubSampler.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.