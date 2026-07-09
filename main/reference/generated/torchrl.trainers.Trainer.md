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

load_from_file(*file: str | Path*, ***kwargs*) → Trainer[[source]](../../_modules/torchrl/trainers/trainers.html#Trainer.load_from_file)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function.
They are ignored when `CKPT_BACKEND=memmap`.

Note

When `CKPT_BACKEND=torch`, `weights_only=True` is set by
default for safer deserialization. Pass `weights_only=False`
explicitly only if you have custom (non-stdlib) objects in your
state dict.

Note

When `CKPT_BACKEND=torch`, `mmap=True` is set by default so
the checkpoint is memory-mapped rather than materialized in RAM
at load time. Pass `mmap=False` if the checkpoint was saved
with the legacy (pre-zipfile) `torch.save` format or if
`file` is a file-like object rather than a path.

request_stop(*reason: str | None = None*) → None[[source]](../../_modules/torchrl/trainers/trainers.html#Trainer.request_stop)

Signal that training should stop at the next loop boundary.