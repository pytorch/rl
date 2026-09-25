# Offline RL Methods

Loss modules for offline reinforcement learning.

| [`CQLLoss`](generated/torchrl.objectives.CQLLoss.html#torchrl.objectives.CQLLoss)(*args, **kwargs) | TorchRL implementation of the continuous CQL loss. |
| --- | --- |
| [`DiscreteCQLLoss`](generated/torchrl.objectives.DiscreteCQLLoss.html#torchrl.objectives.DiscreteCQLLoss)(*args, **kwargs) | TorchRL implementation of the discrete CQL loss. |
| [`IQLLoss`](generated/torchrl.objectives.IQLLoss.html#torchrl.objectives.IQLLoss)(*args, **kwargs) | TorchRL implementation of the IQL loss. |
| [`DiscreteIQLLoss`](generated/torchrl.objectives.DiscreteIQLLoss.html#torchrl.objectives.DiscreteIQLLoss)(*args, **kwargs) | TorchRL implementation of the discrete IQL loss. |
| [`TD3BCLoss`](generated/torchrl.objectives.TD3BCLoss.html#torchrl.objectives.TD3BCLoss)(*args, **kwargs) | TD3+BC Loss Module. |
| [`FQLLoss`](generated/torchrl.objectives.FQLLoss.html#torchrl.objectives.FQLLoss)(*args, **kwargs) | Flow Q-learning for normalized continuous actions. |

## Flow Q-learning

[`FQLLoss`](generated/torchrl.objectives.FQLLoss.html#torchrl.objectives.FQLLoss) trains a [`FlowMatchingPolicy`](generated/torchrl.modules.FlowMatchingPolicy.html#torchrl.modules.FlowMatchingPolicy), a
[`OneStepPolicy`](generated/torchrl.modules.OneStepPolicy.html#torchrl.modules.OneStepPolicy), and an ensemble of TensorDict critics.
Use normalized actions in `[-1, 1]` for both replay data and policies.

`loss_flow` fits the behavior velocity along straight paths from Gaussian
noise to dataset actions. `loss_actor` combines raw student-to-teacher
distillation with the negative mean Q value of clipped student actions.
`loss_qvalue` fits the Bellman target using the current student and target
critics. Each objective updates only its own network parameters.

The default TD0 estimator bootstraps truncations and masks true terminations.
Set the discount with `loss.make_value_estimator(gamma=0.99)` and configure
the critic target update with `SoftUpdate(loss, tau=0.005)`. No target actor
is used. Sum the three loss entries, backpropagate once, step the optimizer,
then call the updater, following the standard TorchRL training sequence.

Observation inputs follow the policies' and critics' `in_keys`, including
nested keys. Wrap either policy in a [`TensorDictSequential`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)
to prepend an encoder that reads multiple observation keys. Each policy's
encoder receives gradients only from its own objective.

Inference passes the one-step policy directly to TorchRL collectors.
Exploration remains stochastic at evaluation, as in the reference implementation.