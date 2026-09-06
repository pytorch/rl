# GRPOTrainer

*class*torchrl.trainers.algorithms.GRPOTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/grpo.html#GRPOTrainer)

A trainer for LLM alignment using GRPO (or compatible) objectives.

See also `GRPOTrainerConfig`
for the Hydra configuration counterpart.

Warning

This is an experimental/prototype feature. The API may change in future
versions. Please report any issues or feedback to help improve this
implementation.

This trainer integrates the full GRPO training loop --
mixed-precision, gradient accumulation, inference-weight synchronization,
and LLM-specific logging -- into the standard
[`Trainer`](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer) hook system. Scalar diagnostics emitted
by the loss (e.g. `ESS`, `clip_fraction`, `kl_approx` for
[`GRPOLoss`](torchrl.objectives.llm.GRPOLoss.html#torchrl.objectives.llm.GRPOLoss)) are logged automatically after
each optimization loop.

It is designed to work with:

- [`GRPOLoss`](torchrl.objectives.llm.GRPOLoss.html#torchrl.objectives.llm.GRPOLoss) (or any `LossModule` whose
outputs start with `"loss_"`)
- [`RayLLMCollector`](torchrl.collectors.llm.RayLLMCollector.html#torchrl.collectors.llm.RayLLMCollector)
- `RayReplayBuffer` (or any
[`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer))

The weight-sync sender is intentionally decoupled from the trainer so that
neither `vllm` nor `sglang` need to be imported by the core library.

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The data collector (typically a
[`RayLLMCollector`](torchrl.collectors.llm.RayLLMCollector.html#torchrl.collectors.llm.RayLLMCollector)).
- **total_frames** (*int*) - Total number of frames / dialog turns.
- **frame_skip** (*int*) - Frame skip value (set to 1 for LLM tasks).
- **optim_steps_per_batch** (*int**,**optional*) - Number of micro-batches drawn
from the replay buffer per collected batch and epoch. `None`
(default) iterates over the whole replay buffer once per epoch.
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)) - The GRPO loss module.
- **optimizer** (*optim.Optimizer**,**optional*) - Optimizer. Required when
`optimization_stepper` is not provided.
- **optimization_stepper** (*OptimizationStepper**,**optional*) - Custom
stepper. If omitted, a
[`MixedPrecisionOptimizationStepper`](torchrl.trainers.MixedPrecisionOptimizationStepper.html#torchrl.trainers.MixedPrecisionOptimizationStepper) is
constructed automatically from `optimizer` and the
mixed-precision arguments below.
- **weight_sync_sender** (*optional*) - Object with an `update_weights()`
method used to push training weights to the inference engine.
Pass `None` to disable weight synchronization (useful for
offline testing).
- **weight_update_frequency** (*int**,**optional*) - Optimizer steps between
weight pushes to the inference engine when `async_collection=True`
(registered at the `post_optim` stage through
[`UpdateWeights`](torchrl.trainers.UpdateWeights.html#torchrl.trainers.UpdateWeights)). In sync mode weights
are pushed once per collected batch and this value is unused.
Default: `1`.
- **empty_replay_buffer_on_weight_update** (*bool**,**optional*) - If `True`,
the replay buffer is emptied after each weight push (sync GRPO).
Default: `False`.
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - The replay buffer used for
sampling.
- **batch_size** (*int**,**optional*) - Override the replay buffer's batch size.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device on which sampled batches are
placed before the loss forward pass (typically the training
device). `None` leaves samples on their storage device.
- **mixed_precision** (*bool**,**optional*) - Enable autocast + GradScaler.
Default: `False`.
- **autocast_dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype for `autocast`.
Default: `torch.bfloat16`.
- **gradient_accumulation_steps** (*int**,**optional*) - Gradient accumulation.
Default: `1`.
- **logger** (*Logger**,**optional*) - Logger (e.g. `WandbLogger`).
- **clip_norm** (*float**,**optional*) - Gradient clip norm, applied by the
stepper. Default: `1.0`.
- **progress_bar** (*bool**,**optional*) - Show a `tqdm` progress bar.
- **seed** (*int**,**optional*) - Random seed.
- **save_trainer_interval** (*int**,**optional*) - Frame interval between saves.
- **log_interval** (*int**,**optional*) - Frame interval between logs.
- **save_trainer_file** (*str**|**Path**,**optional*) - Path for legacy saves.
- **checkpoint** ([*Checkpoint*](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint)*,**optional*) - Unified checkpoint object.
- **checkpoint_rotation** ([*CheckpointRotation*](torchrl.checkpoint.CheckpointRotation.html#torchrl.checkpoint.CheckpointRotation)*,**optional*) - Rotation policy.
- **checkpoint_metadata** (*Callable**,**optional*) - Extra metadata callback.
- **num_epochs** (*int**,**optional*) - Epochs per collected batch. Default: `1`.
- **async_collection** (*bool**,**optional*) - Whether data is collected
asynchronously (`grpo-async` mode). Default: `False`.
- **log_timings** (*bool**,**optional*) - Log timing of each hook. Default: `False`.
- **auto_log_optim_steps** (*bool**,**optional*) - Log `optim_steps` after each
optimization loop. Default: `True`.
- **log_rewards** (*bool**,**optional*) - Log reward / return statistics.
Default: `True`.
- **log_kl** (*bool**,**optional*) - Log KL-divergence keys from the loss output.
Default: `True`.

Examples

```
>>> from torchrl.trainers.algorithms.grpo import GRPOTrainer
>>> # Assuming you have a collector, loss_fn, optimizer, replay_buffer,
>>> # and weight_sync_sender already constructed (see SOTA scripts):
>>> trainer = GRPOTrainer(
... collector=collector,
... total_frames=cfg.train.total_dialog_turns,
... frame_skip=1,
... optim_steps_per_batch=cfg.train.epochs,
... loss_module=loss_fn,
... optimizer=optimizer,
... weight_sync_sender=sender,
... weight_update_frequency=1,
... empty_replay_buffer_on_weight_update=cfg.train.empty_replay_buffer,
... replay_buffer=replay_buffer,
... mixed_precision=cfg.train.mixed_precision,
... gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
... clip_norm=cfg.optimizer.clip_grad_norm,
... logger=wandb_logger,
... )
>>> trainer.train()
```

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

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.