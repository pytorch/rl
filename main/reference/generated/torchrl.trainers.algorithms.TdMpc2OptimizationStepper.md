# TdMpc2OptimizationStepper

*class*torchrl.trainers.algorithms.TdMpc2OptimizationStepper(*loss_module: [TdMpc2Loss](torchrl.objectives.TdMpc2Loss.html#torchrl.objectives.TdMpc2Loss)*, *optimizer_model: [Optimizer](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)*, *optimizer_actor: [Optimizer](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)*, ***, *target_tau: float = 0.01*, *zero_grad_set_to_none: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/tdmpc2.html#TdMpc2OptimizationStepper)

Execute the two-phase TD-MPC2 learner update.

The model optimizer updates the world model and online Q-functions first;
the actor optimizer then updates the policy objective on the detached
imagined latent sequence captured by that model update. The policy update
uses the model/Q parameters after the first optimizer step. The target
Q-functions are soft-updated last.

Parameters:

- **loss_module** - TD-MPC2 loss providing model and actor objectives.
- **optimizer_model** - Optimizer for the world model and online Q-functions.
- **optimizer_actor** - Optimizer for the policy prior.
- **target_tau** - Target-Q Polyak averaging factor. Defaults to `0.01`.
- **zero_grad_set_to_none** - Whether optimizer `zero_grad` calls set
gradients to `None`. Defaults to `True`.

See also

`TdMpc2OptimizationStepperConfig`,
[TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828).

load_state_dict(*state_dict: dict*) → None[[source]](../../_modules/torchrl/trainers/algorithms/tdmpc2.html#TdMpc2OptimizationStepper.load_state_dict)

Restore optimizer state and completed-update count.

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'optimization_stepper'*) → None[[source]](../../_modules/torchrl/trainers/algorithms/tdmpc2.html#TdMpc2OptimizationStepper.register)

Register the stepper and validate exclusive trainer ownership.

state_dict() → dict[[source]](../../_modules/torchrl/trainers/algorithms/tdmpc2.html#TdMpc2OptimizationStepper.state_dict)

Return optimizer state and completed-update count.

step(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *sub_batch: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/trainers/algorithms/tdmpc2.html#TdMpc2OptimizationStepper.step)

Run model, actor, and target-Q updates and return detached metrics.

*property*update_count*: int*

Number of completed TD-MPC2 updates.