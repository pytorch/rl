# MixedPrecisionOptimizationStepper

*class*torchrl.trainers.MixedPrecisionOptimizationStepper(*optimizer: [Optimizer](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)*, ***, *mixed_precision: bool = False*, *autocast_dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) = torch.bfloat16*, *gradient_accumulation_steps: int = 1*, *clip_norm: float | None = 1.0*, *device_type: str | None = None*)[[source]](../../_modules/torchrl/trainers/trainers.html#MixedPrecisionOptimizationStepper)

Optimization step with mixed precision and gradient accumulation.

This stepper wraps each forward/backward pass in `torch.amp.autocast`
and optionally scales gradients with `torch.amp.GradScaler` (for fp16).
It also implements *gradient accumulation*: gradients are accumulated for
`gradient_accumulation_steps` micro-batches before the optimizer is
stepped and zeroed.

It can be used with any [`Trainer`](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer); LLM trainers
such as [`GRPOTrainer`](torchrl.trainers.algorithms.GRPOTrainer.html#torchrl.trainers.algorithms.GRPOTrainer) construct it by
default.

Parameters:

**optimizer** (*optim.Optimizer*) - The optimizer to use.

Keyword Arguments:

- **mixed_precision** (*bool**,**optional*) - Whether to enable mixed-precision
training. Default: `False`.
- **autocast_dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - The dtype to use inside
`autocast`. Default: `torch.bfloat16`.
- **gradient_accumulation_steps** (*int**,**optional*) - Number of micro-batches
over which gradients are accumulated before a step. Default: `1`.
- **clip_norm** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Maximum gradient norm for clipping.
Default: `1.0`.
- **device_type** (*str**,**optional*) - Device type passed to `autocast` and
`GradScaler` (e.g. `"cuda"` or `"cpu"`). Defaults to the
device type of the optimizer's first parameter.

Note

`GradScaler` is only enabled when `mixed_precision=True` *and*
`autocast_dtype=torch.float16`. With bfloat16 (the recommended
dtype for modern GPUs) the scaler is a no-op and is not created.

*property*optimizer_step_count*: int*

Number of completed optimizer steps.

Discounts gradient-accumulation micro-steps and steps skipped by the
GradScaler on overflow or by the non-finite guards. Read by hooks that
act on an optimizer-step cadence (e.g.
[`UpdateWeights`](torchrl.trainers.UpdateWeights.html#torchrl.trainers.UpdateWeights) with
`interval_unit="optim_steps"`).

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'optimization_stepper'*) → None

Register the stepper with a Trainer for checkpointing.

step(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *sub_batch: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/trainers/trainers.html#MixedPrecisionOptimizationStepper.step)

Perform one forward pass and scaled backward pass.

The optimizer is only stepped and zeroed every
`gradient_accumulation_steps` calls.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - The owning [`Trainer`](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer).
- **sub_batch** (*TensorDictBase*) - Mini-batch used for this step.

Returns:

A [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with scalar metrics (losses,
grad_norm) suitable for logging.