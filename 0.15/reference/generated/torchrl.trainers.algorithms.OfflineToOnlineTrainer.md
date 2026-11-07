# OfflineToOnlineTrainer

*class*torchrl.trainers.algorithms.OfflineToOnlineTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/offline_to_online.html#OfflineToOnlineTrainer)

Train from offline replay before optional online fine-tuning.

See also `OfflineToOnlineTrainerConfig`
for the Hydra configuration counterpart.

Builds on [`SACTrainer`](torchrl.trainers.algorithms.SACTrainer.html#torchrl.trainers.algorithms.SACTrainer) target updates,
collector weight synchronization and logging. Offline updates precede online
collection within the standard Trainer lifecycle. With a mixed buffer,
online collection anneals the offline sampling fraction over `anneal_frames`.

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)*or**callable**,**optional*) - online collector or a
factory called after offline training, unless loss initialization
requires environment metadata first. Defaults to `None`.
- **total_frames** (*int*) - online frames to collect. Defaults to zero.
- **frame_skip** (*int*) - frames skipped between policy updates. Defaults to one.
- **optim_steps_per_batch** (*int*) - updates per collected batch. Defaults to one.
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)) - actor-critic objective with an `actor_network`.
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*or*[*OfflineToOnlineReplayBuffer*](torchrl.data.OfflineToOnlineReplayBuffer.html#torchrl.data.OfflineToOnlineReplayBuffer)) - regular
replay storage or independently sampled offline and online buffers.

Keyword Arguments:

- **anneal_frames** (*int**,**optional*) - frames over which `offline_fraction`
decays to 0. Defaults to `total_frames`; pass `<= 0` to keep the
fraction fixed.
- **batch_size** (*int**,**optional*) - replay-buffer sampling batch size.
- **offline_steps** (*int*) - gradient updates before online collection (default zero).
- **device** ([*device*](torchrl.data.Binary.html#torchrl.data.Binary.device)*,**optional*) - device for sampled training batches.
- **compile_loss** (*bool*) - compile the loss module, retaining its checkpoint keys.

A regular replay buffer retains offline and online transitions together. A
mixed offline-to-online buffer instead controls their sampling fractions.
The collector may be omitted for offline-only training or supplied as a
factory, created when collection or environment metadata requires it.
Losses are logged by update count
during pretraining and by collected frames during online training.

See [`SACTrainer`](torchrl.trainers.algorithms.SACTrainer.html#torchrl.trainers.algorithms.SACTrainer) for the remaining
keyword arguments.

Note

Experimental/prototype feature; the API may change.

*property*checkpoint_step*: int*

Use update counts across both phases when pretraining is configured.

*property*completed_steps*: int*

Number of completed offline and online updates.

compute_loss(*sub_batch: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *method: str | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | tuple[Any, ...]

Evaluate the configured loss through the active execution boundary.

load_from_file(*file: str | Path*, ***kwargs*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function for
legacy torch checkpoints and unified components explicitly saved with
the torch state-dict payload format. Unified checkpoints additionally
accept `strict` to control missing or incompatible components.
Arguments are ignored when `CKPT_BACKEND=memmap`.

Note

Unified state-dict components use TensorDict storage by default and
do not invoke the pickle loader. For explicit torch payloads and
`CKPT_BACKEND=torch` checkpoints, `weights_only=True` is the
default for safer deserialization. Pass `weights_only=False`
explicitly only if the state dict contains custom objects. On
torch < 2.4 the default is `weights_only=False` because the
weights-only unpickler of those versions cannot deserialize the
`torch.device` instances contained in TensorDict state-dicts.

Note

Explicit torch payloads and `CKPT_BACKEND=torch` checkpoints use
`mmap=True` by default. Pass `mmap=False` for legacy pre-zipfile
`torch.save` files or file-like objects. On Windows the default
is `mmap=False` because a mapped checkpoint keeps the file locked,
preventing deletion or re-save.

Note

Unified checkpoint tensors are mapped to CPU by default. Pass an
explicit `map_location` to select another device mapping.

Note

After restoring an independently registered policy component, the
trainer synchronizes the collector once so local policy copies and
remote workers observe the restored learner weights.

Note

`file` may also be a [`CheckpointRotation`](torchrl.checkpoint.CheckpointRotation.html#torchrl.checkpoint.CheckpointRotation)
directory, in which case its newest checkpoint is restored.

optim_steps(*batch: ~tensordict.base.TensorDictBase*, ***, *optim_steps_per_batch: int | None | object = <object object>*, *num_epochs: int | object = <object object>*) → None

Run the configured optimization loop for one collected batch.

Keyword overrides are applied only to this call and do not change the
trainer configuration. They are useful for algorithms that need a
one-time optimization schedule while retaining the standard Trainer
hooks and logging behavior.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.

shutdown() → None[[source]](../../_modules/torchrl/trainers/algorithms/offline_to_online.html#OfflineToOnlineTrainer.shutdown)

Release the online collector, if it was initialized.

stop_on_signal(*signals: Collection[int] = (Signals.SIGINT, Signals.SIGTERM)*)

Stop training cleanly when the process receives a termination signal.

Wrap `train()` in this context. The first signal calls
`request_stop()`, so the loop finishes the current batch, writes a
final checkpoint when a save destination is configured, shuts the
collector down and returns. A second signal raises
`KeyboardInterrupt`. Previous handlers are restored on exit.

Parameters:

**signals** (*Collection**[**int**]**,**optional*) - signal numbers to handle.
Defaults to `SIGINT` and `SIGTERM`.

Examples

```
>>> with trainer.stop_on_signal(): 
... trainer.train()
```

update() → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/trainers/algorithms/offline_to_online.html#OfflineToOnlineTrainer.update)

Run one replay update through the standard optimization hooks.