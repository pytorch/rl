# ReplayBufferTrainer

*class*torchrl.trainers.ReplayBufferTrainer(*replay_buffer: [TensorDictReplayBuffer](torchrl.data.TensorDictReplayBuffer.html#torchrl.data.TensorDictReplayBuffer)*, *batch_size: int | None = None*, *memmap: bool = False*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *flatten_tensordicts: bool = False*, *max_dims: Sequence[int] | None = None*, *iterate: bool = False*)[[source]](../../_modules/torchrl/trainers/trainers.html#ReplayBufferTrainer)

Replay buffer hook provider.

Parameters:

- **replay_buffer** ([*TensorDictReplayBuffer*](torchrl.data.TensorDictReplayBuffer.html#torchrl.data.TensorDictReplayBuffer)) - replay buffer to be used.
- **batch_size** (*int**,**optional*) - batch size when sampling data from the
latest collection or from the replay buffer. If none is provided,
the replay buffer batch-size will be used (preferred option for
unchanged batch-sizes).
- **memmap** (*bool**,**optional*) - if `True`, a memmap tensordict is created.
Default is `False`.
- **device** (*device**,**optional*) - device where the samples must be placed.
Default to `None`.
- **flatten_tensordicts** (*bool**,**optional*) - if `True`, the tensordicts will be
flattened (or equivalently masked with the valid mask obtained from
the collector) before being passed to the replay buffer. Otherwise,
no transform will be achieved other than padding (see `max_dims` arg below).
Defaults to `False`.
- **max_dims** (*sequence**of**int**,**optional*) - if `flatten_tensordicts` is set to False,
this will be a list of the length of the batch_size of the provided
tensordicts that represent the maximum size of each. If provided,
this list of sizes will be used to pad the tensordict and make their shape
match before they are passed to the replay buffer. If there is no
maximum value, a -1 value should be provided.
- **iterate** (*bool**,**optional*) - if `True`, the replay buffer will be iterated over
in a loop. Defaults to `False` (call to [`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample) will be used).

Examples

```
>>> rb_trainer = ReplayBufferTrainer(replay_buffer=replay_buffer, batch_size=N)
>>> trainer.register_op("batch_process", rb_trainer.extend)
>>> trainer.register_op("process_optim_batch", rb_trainer.sample)
>>> trainer.register_op("post_loss", rb_trainer.update_priority)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'replay_buffer'*)[[source]](../../_modules/torchrl/trainers/trainers.html#ReplayBufferTrainer.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.