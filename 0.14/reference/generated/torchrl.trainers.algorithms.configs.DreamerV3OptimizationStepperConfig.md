# DreamerV3OptimizationStepperConfig

*class*torchrl.trainers.algorithms.configs.DreamerV3OptimizationStepperConfig(*loss_module: Any = None*, *optimizer: Any = None*, *target_updater: Any = None*, *compile_train_step: bool | None = None*, *compile_mode: str = 'default'*, *cudagraph: bool | None = None*, *rssm_scan_unroll: int | None = 8*, *warmup_steps: int = 5*, *mixed_precision: bool = False*, *_target_: str = 'torchrl.trainers.algorithms.DreamerV3OptimizationStepper'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#DreamerV3OptimizationStepperConfig)

Hydra configuration for [`DreamerV3OptimizationStepper`](torchrl.trainers.algorithms.DreamerV3OptimizationStepper.html#torchrl.trainers.algorithms.DreamerV3OptimizationStepper).

Examples

With the learner and optimizer from the public stepper example:

```
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import DreamerV3OptimizationStepperConfig
>>> configured_stepper = instantiate(
... DreamerV3OptimizationStepperConfig(),
... loss_module=loss_module, optimizer=optimizer,
... target_updater=target_updater,
... )
>>> metrics = configured_stepper.step(None, sample)
>>> assert not sample["replay_context", "state"].requires_grad
```