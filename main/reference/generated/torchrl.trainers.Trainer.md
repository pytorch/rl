# Trainer

*class*torchrl.trainers.Trainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/trainers.html#Trainer)

A generic Trainer class.

A trainer is responsible for collecting data and training the model.
To keep the class as versatile as possible, Trainer does not construct any
of its specific operations: they all must be hooked at specific points in
the training loop.

To build a Trainer, one needs an iterable data source (a `collector`), a
loss module and an optimizer.

Parameters:

- **collector** (*Sequence**[**TensorDictBase**]*) - An iterable returning batches of
data in a TensorDict form of shape [batch x time steps].
- **total_frames** (*int*) - Total number of frames to be collected during
training.
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)) - A module that reads TensorDict batches
(possibly sampled from a replay buffer) and return a loss
TensorDict where every key points to a different loss component.
- **optimizer** (*optim.Optimizer*) - An optimizer that trains the parameters
of the model.
- **logger** (*Logger**,**optional*) - a Logger that will handle the logging.
- **optim_steps_per_batch** (*int**,**optional*) - number of optimization steps
per collection of data. An trainer works as follows: a main loop
collects batches of data (epoch loop), and a sub-loop (training
loop) performs model updates in between two collections of data.
If None, the trainer will use the number of workers as the number of optimization steps.
- **clip_grad_norm** (*bool**,**optional*) - If True, the gradients will be clipped
based on the total norm of the model parameters. If False,
all the partial derivatives will be clamped to
(-clip_norm, clip_norm). Default is `True`.
- **clip_norm** (*Number**,**optional*) - value to be used for clipping gradients.
Default is None (no clip norm).
- **progress_bar** (*bool**,**optional*) - If True, a progress bar will be
displayed using tqdm. If tqdm is not installed, this option
won't have any effect. Default is `True`
- **seed** (*int**,**optional*) - Seed to be used for the collector, pytorch and
numpy. Default is `None`.
- **save_trainer_interval** (*int**,**optional*) - How often the trainer should be
saved to disk, in frame count. Default is 10000.
- **log_interval** (*int**,**optional*) - How often the values should be logged,
in frame count. Default is 10000.
- **save_trainer_file** (*path**,**optional*) - path where to save the trainer.
Default is None (no saving)
- **checkpoint** ([*Checkpoint*](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint)*,**optional*) - unified checkpoint object used for
scheduled saves and restores. The trainer registers any missing
standard components on this object. When omitted, the legacy
`CKPT_BACKEND` path is retained during the compatibility window.
- **async_collection** (*bool**,**optional*) - Whether to collect data asynchronously.
This will only work if the replay buffer is registered within the data collector.
If using this, the UTD ratio (Update to Data) will be logged under the key "utd_ratio".
Default is False.
- **log_timings** (*bool**,**optional*) - If True, automatically register a LogTiming hook to log
timing information for all hooks to the logger (e.g., wandb, tensorboard).
Timing metrics will be logged with prefix "time/" (e.g., "time/hook/UpdateWeights").
Default is False.
- **auto_log_optim_steps** (*bool**,**optional*) - If True, automatically log `optim_steps` and the
keys of the averaged loss TensorDict at the end of every optimization loop, in addition
to anything `post_optim_complete_log` hooks return. Set to False to fully delegate
this logging to user-registered hooks. Default is True.
- **replay_buffer** (*optional*) - Replay owner used by a remote learner backend.
- **target_net_updater** (*TargetNetUpdater**,**optional*) - Target updater serialized
with the learner object graph.
- **batch_size** (*int**,**optional*) - Global learner batch size. Defaults to the
replay buffer batch size.
- **learner_backend** (*str*) - Optimization placement, `"local"` or `"ray"`.
Defaults to `"local"`.
- **learner_backend_options** (*dict**,**optional*) - Backend-specific options.
- **learner_poll_interval** (*float*) - Remote replay polling interval. Defaults
to `0.05` seconds.

compute_loss(*sub_batch: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *method: str | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | tuple[Any, ...][[source]](../../_modules/torchrl/trainers/trainers.html#Trainer.compute_loss)

Evaluate the configured loss through the active execution boundary.

load_from_file(*file: str | Path*, ***kwargs*) → Trainer[[source]](../../_modules/torchrl/trainers/trainers.html#Trainer.load_from_file)

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

request_stop(*reason: str | None = None*) → None[[source]](../../_modules/torchrl/trainers/trainers.html#Trainer.request_stop)

Signal that training should stop at the next loop boundary.