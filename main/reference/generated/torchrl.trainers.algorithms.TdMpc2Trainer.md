# TdMpc2Trainer

*class*torchrl.trainers.algorithms.TdMpc2Trainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/tdmpc2.html#TdMpc2Trainer)

A trainer class for the TD-MPC2 algorithm.

See also `TdMpc2TrainerConfig` for the
Hydra configuration counterpart.

This trainer implements TD-MPC2, a scalable and robust model-based reinforcement
learning algorithm for continuous control. TD-MPC2 learns a latent world model,
including transition, reward, value, and policy components, and uses model
predictive control with MPPI-style planning to select actions.

The trainer handles:
- Sequence-based replay buffer sampling for model learning
- Joint optimization of the latent world model and policy prior
- Model-predictive action planning through a TD-MPC2 planner
- Synchronization of planner weights with the data collector
- Logging and checkpointing of training metrics

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The data collector used to gather environment interactions.
- **total_frames** (*int*) - Total number of frames to collect during training.
- **loss_module** ([*TdMpc2Loss*](torchrl.objectives.TdMpc2Loss.html#torchrl.objectives.TdMpc2Loss)) - The TD-MPC2 loss module.
- **planner** ([*TdMpc2Planner*](torchrl.modules.TdMpc2Planner.html#torchrl.modules.TdMpc2Planner)) - The planner used for model-predictive action selection.
- **optimizer_model** (*optim.Optimizer*) - Optimizer for the world model and value functions.
- **optimizer_actor** (*optim.Optimizer*) - Optimizer for the policy prior.
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)) - Replay buffer containing training trajectories.
- **batch_size** (*int**,**optional*) - Number of trajectory sequences sampled per update.
- **optim_steps_per_batch** (*int**,**optional*) - Number of optimization steps per collected batch.
- **logger** (*Logger**,**optional*) - Logger for recording training metrics.
- **progress_bar** (*bool**,**optional*) - Whether to show a progress bar during training.
- **seed** (*int**,**optional*) - Random seed for reproducibility.

*property*checkpoint_step*: int*

Progress used for checkpoint intervals and rotation filenames.

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

optim_steps(*batch: ~tensordict.base.TensorDictBase*, ***, *optim_steps_per_batch: int | None | object = <object object>*, *num_epochs: int | object = <object object>*) → None[[source]](../../_modules/torchrl/trainers/algorithms/tdmpc2.html#TdMpc2Trainer.optim_steps)

Run seed-data pretraining once, then use the normal cadence.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.

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