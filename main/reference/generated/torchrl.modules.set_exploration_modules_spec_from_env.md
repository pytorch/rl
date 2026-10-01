# set_exploration_modules_spec_from_env

*class*torchrl.modules.set_exploration_modules_spec_from_env(*policy: nn.Module*, *env: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*)[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#set_exploration_modules_spec_from_env)

Sets exploration module specs from an environment action spec.

This is intended for cases where exploration modules (e.g. AdditiveGaussianModule)
are instantiated with `spec=None` and must be configured once the environment
is known (e.g. inside a collector).