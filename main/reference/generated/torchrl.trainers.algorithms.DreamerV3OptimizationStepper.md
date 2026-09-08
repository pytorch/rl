# DreamerV3OptimizationStepper

*class*torchrl.trainers.algorithms.DreamerV3OptimizationStepper(*loss_module: [DreamerV3Loss](torchrl.objectives.DreamerV3Loss.html#torchrl.objectives.DreamerV3Loss)*, *optimizer: [Optimizer](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)*, *target_updater: TargetNetUpdater | None = None*, ***, *compile_train_step: bool | None = None*, *compile_mode: Literal['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'] = 'default'*, *cudagraph: bool | None = None*, *rssm_scan_unroll: int | None = 8*, *warmup_steps: int = 5*, *mixed_precision: bool = False*)[[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3OptimizationStepper)

Execute a complete DreamerV3 forward/backward and optimizer update.

One optional compile scope owns all shared loss modules. CUDA graph capture
covers forward/backward only; optimizer and target updates run afterwards.
Call `warmup()` with a representative replay sample before starting
collection when compilation or capture is enabled. Warm-up preserves model
buffers and global RNG state and never advances the optimizer or targets.
Returned scalar metrics and posterior features written to the input's
`replay_context` key retain their values after later captured updates.

Parameters:

- **loss_module** ([*DreamerV3Loss*](torchrl.objectives.DreamerV3Loss.html#torchrl.objectives.DreamerV3Loss)) - Complete learner objective.
- **optimizer** ([*torch.optim.Optimizer*](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)) - Optimizer owning the shared learner
parameters once each.
- **target_updater** (*TargetNetUpdater**,**optional*) - Target update performed
after each optimizer step. Default: `None`.

Keyword Arguments:

- **compile_train_step** (*bool**or**None**,**optional*) - Compile the complete
forward/backward pass with TorchInductor. Requires PyTorch's
`torch._dynamo.config.inline_inbuilt_nn_modules` support to be
enabled for functional parameter contexts. `None` picks the
fastest supported path for the sample device: compiled on CUDA,
eager elsewhere. Default: `None`.
- **compile_mode** (*str**,**optional*) - PyTorch compile mode. Default: `"default"`.
- **cudagraph** (*bool**or**None**,**optional*) - Capture forward/backward on CUDA.
`None` captures on CUDA and stays eager elsewhere. Default: `None`.
- **rssm_scan_unroll** (*int**or**None**,**optional*) - When the step is compiled,
every [`RSSMRolloutV3`](torchrl.modules.RSSMRolloutV3.html#torchrl.modules.RSSMRolloutV3) of the loss without a
selected backend switches to the higher-order scan, unrolled by this
many steps, so the compile traces one scan instead of the explicit
loop over the whole sequence. `None` leaves the rollouts as they
are. Default: `8`.
- **warmup_steps** (*int**,**optional*) - Representative forward/backward calls
before training. Must be positive. Default: `5`.
- **mixed_precision** (*bool**,**optional*) - Use bfloat16 autocast for CUDA
forward/backward. Default: `False`.

Note

Shared modules must not also have an independently compiled execution
scope. Pause collection and synchronize pending replay operations before
warm-up or checkpointing. Distributed execution is outside this stepper's
supported modes.

Note

The device decides the defaults, so a stepper built with the default
arguments must call `warmup()` before its first update on CUDA;
on CPU the first update runs eagerly without it.

Examples

Continue from the runnable [`DreamerV3Loss`](torchrl.objectives.DreamerV3Loss.html#torchrl.objectives.DreamerV3Loss)
example, which constructs `loss_module`, `target_updater` and
`sample` from public components:

```
>>> from torchrl.trainers.algorithms import (
... DreamerV3OptimizationStepper, DreamerV3Optimizer,
... )
>>> optimizer = DreamerV3Optimizer(loss_module.parameters(), warmup_steps=0)
>>> stepper = DreamerV3OptimizationStepper(
... loss_module, optimizer, target_updater, warmup_steps=1,
... )
>>> stepper.warmup(sample)
>>> before = [parameter.detach().clone() for parameter in loss_module.parameters()]
>>> metrics = stepper.step(None, sample)
>>> assert any(
... not torch.equal(parameter, previous)
... for parameter, previous in zip(loss_module.parameters(), before)
... )
>>> assert not sample["replay_context", "state"].requires_grad
```

See also [`DreamerV3OptimizationStepperConfig`](torchrl.trainers.algorithms.configs.DreamerV3OptimizationStepperConfig.html#torchrl.trainers.algorithms.configs.DreamerV3OptimizationStepperConfig).

load_state_dict(*state_dict: dict[str, Any]*) → None[[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3OptimizationStepper.load_state_dict)

Restore optimizer state without replacing captured gradient buffers.

resolve(*device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*) → tuple[bool, bool][[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3OptimizationStepper.resolve)

Return the `(compile_train_step, cudagraph)` pair used on `device`.

`None` requests resolve to `True` on CUDA and `False` elsewhere;
explicit requests are returned unchanged.

state_dict() → dict[str, Any][[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3OptimizationStepper.state_dict)

Return optimizer and target-update progress; checkpoint the loss separately.

step(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer) | None*, *sub_batch: TensorDictBase*) → TensorDictBase[[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3OptimizationStepper.step)

Update learner parameters and targets, returning detached metrics.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*or**None*) - Owning trainer, or `None` for a custom
training loop. An owning trainer must use this stepper's loss.
- **sub_batch** (*TensorDictBase*) - Real transition sequences with the
schema supplied to `warmup()` when capture is enabled.
Detached posterior features are written under the loss's
configured `replay_context` key for subsequent replay updates.

Returns:

Detached scalar metrics suitable for Trainer logging.

warmup(*sample: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3OptimizationStepper.warmup)

Prepare execution using representative data without training updates.

Parameters:

**sample** (*TensorDictBase*) - Sample with the shape, keys, dtype and
device used for subsequent updates. Collection and replay
operations must be quiescent for CUDA capture.