# Exploration Strategies

Exploration modules add noise to actions to enable exploration during training.

| [`AdditiveGaussianModule`](generated/torchrl.modules.AdditiveGaussianModule.html#torchrl.modules.AdditiveGaussianModule)(*args, **kwargs) | Additive Gaussian PO module. |
| --- | --- |
| [`ConsistentDropoutModule`](generated/torchrl.modules.ConsistentDropoutModule.html#torchrl.modules.ConsistentDropoutModule)(*args, **kwargs) | A TensorDictModule wrapper for `ConsistentDropout`. |
| [`EGreedyModule`](generated/torchrl.modules.EGreedyModule.html#torchrl.modules.EGreedyModule)(*args, **kwargs) | Epsilon-Greedy exploration module. |
| [`OrnsteinUhlenbeckProcessModule`](generated/torchrl.modules.OrnsteinUhlenbeckProcessModule.html#torchrl.modules.OrnsteinUhlenbeckProcessModule)(*args, **kwargs) | Ornstein-Uhlenbeck exploration policy module. |

## Helpers

| [`set_exploration_modules_spec_from_env`](generated/torchrl.modules.set_exploration_modules_spec_from_env.html#torchrl.modules.set_exploration_modules_spec_from_env)(...) | Sets exploration module specs from an environment action spec. |
| --- | --- |