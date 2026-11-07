# Value-Based Methods

Loss modules for value-based RL algorithms.

| [`DQNLoss`](generated/torchrl.objectives.DQNLoss.html#torchrl.objectives.DQNLoss)(*args, **kwargs) | The DQN Loss class. |
| --- | --- |
| [`DistributionalDQNLoss`](generated/torchrl.objectives.DistributionalDQNLoss.html#torchrl.objectives.DistributionalDQNLoss)(*args, **kwargs) | A distributional DQN loss class. |
| [`IQLLoss`](generated/torchrl.objectives.IQLLoss.html#torchrl.objectives.IQLLoss)(*args, **kwargs) | TorchRL implementation of the IQL loss. |
| [`DiscreteIQLLoss`](generated/torchrl.objectives.DiscreteIQLLoss.html#torchrl.objectives.DiscreteIQLLoss)(*args, **kwargs) | TorchRL implementation of the discrete IQL loss. |
| [`CQLLoss`](generated/torchrl.objectives.CQLLoss.html#torchrl.objectives.CQLLoss)(*args, **kwargs) | TorchRL implementation of the continuous CQL loss. |
| [`DiscreteCQLLoss`](generated/torchrl.objectives.DiscreteCQLLoss.html#torchrl.objectives.DiscreteCQLLoss)(*args, **kwargs) | TorchRL implementation of the discrete CQL loss. |

## Parallel Q-Network lambda returns

[`DQNLoss`](generated/torchrl.objectives.DQNLoss.html#torchrl.objectives.DQNLoss) supports the lambda-return target used by [Parallel Q-Networks
(PQN)](https://arxiv.org/abs/2407.04811) through its existing value-estimator
interface:

```
loss = DQNLoss(value_network, action_space=action_spec)
loss.make_value_estimator(
 ValueEstimators.TDLambda,
 gamma=0.99,
 lmbda=0.95,
)
```

The value network's greedy action selection writes `chosen_action_value`, so
the lambda return bootstraps from \(\max_a Q(s_{t+1}, a)\). It does not use
the behavior action at the next step, and no `("next", "action")` entry is
required.