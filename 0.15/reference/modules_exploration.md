# Exploration Strategies

Exploration modules add noise to actions to enable exploration during training.
[`NoisyLinear`](generated/torchrl.modules.NoisyLinear.html#torchrl.modules.NoisyLinear) instead injects learnable noise in
parameter space. The forward pass does not resample; callers must call
[`reset_noise()`](generated/torchrl.modules.NoisyLinear.html#torchrl.modules.NoisyLinear.reset_noise) or
`module.apply(reset_noise)`.
[`make_trainer()`](generated/torchrl.trainers.helpers.make_trainer.html#torchrl.trainers.helpers.make_trainer) registers such a resample on
the `pre_optim_steps` hook when `cfg.noisy` is set.

| [`AdditiveGaussianModule`](generated/torchrl.modules.AdditiveGaussianModule.html#torchrl.modules.AdditiveGaussianModule)(*args, **kwargs) | Additive Gaussian PO module. |
| --- | --- |
| [`ConsistentDropoutModule`](generated/torchrl.modules.ConsistentDropoutModule.html#torchrl.modules.ConsistentDropoutModule)(*args, **kwargs) | A TensorDictModule wrapper for `ConsistentDropout`. |
| [`EGreedyModule`](generated/torchrl.modules.EGreedyModule.html#torchrl.modules.EGreedyModule)(*args, **kwargs) | Epsilon-Greedy exploration module. |
| [`NoisyLazyLinear`](generated/torchrl.modules.NoisyLazyLinear.html#torchrl.modules.NoisyLazyLinear)(out_features[, bias, ...]) | Noisy Lazy Linear Layer. |
| [`NoisyLinear`](generated/torchrl.modules.NoisyLinear.html#torchrl.modules.NoisyLinear)(in_features, out_features[, ...]) | Noisy Linear Layer. |
| [`OrnsteinUhlenbeckProcessModule`](generated/torchrl.modules.OrnsteinUhlenbeckProcessModule.html#torchrl.modules.OrnsteinUhlenbeckProcessModule)(*args, **kwargs) | Ornstein-Uhlenbeck exploration policy module. |

## Helpers

| [`reset_noise`](generated/torchrl.modules.reset_noise.html#torchrl.modules.reset_noise)(layer) | Resets the noise of noisy layers. |
| --- | --- |
| [`set_exploration_modules_spec_from_env`](generated/torchrl.modules.set_exploration_modules_spec_from_env.html#torchrl.modules.set_exploration_modules_spec_from_env)(...) | Sets exploration module specs from an environment action spec. |