# torchrl.trainers.algorithms.configs.hooks.TdMpc2OptimizationStepperConfig

*class*torchrl.trainers.algorithms.configs.hooks.TdMpc2OptimizationStepperConfig(*loss_module: Any = None*, *optimizer_model: Any = None*, *optimizer_actor: Any = None*, *target_tau: float = 0.01*, *zero_grad_set_to_none: bool = True*, *_target_: str = 'torchrl.trainers.algorithms.tdmpc2.TdMpc2OptimizationStepper'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/hooks.html#TdMpc2OptimizationStepperConfig)

Configuration for the TD-MPC2 optimization stepper.

The stepper owns the model and actor optimizers and performs the staged
model, policy, and target-Q updates.

Examples

```
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import (
... TdMpc2OptimizationStepperConfig,
... )
>>> configured_stepper = instantiate( 
... TdMpc2OptimizationStepperConfig(
... loss_module=loss_module,
... optimizer_model=optimizer_model,
... optimizer_actor=optimizer_actor,
... )
... )
```

See also [`TdMpc2OptimizationStepper`](torchrl.trainers.algorithms.TdMpc2OptimizationStepper.html#torchrl.trainers.algorithms.TdMpc2OptimizationStepper).