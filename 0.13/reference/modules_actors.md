# Actor Modules

Actor modules represent policies in RL. They map observations to actions, either deterministically
or stochastically.

## TensorDictModules and SafeModules

| [`Actor`](generated/torchrl.modules.tensordict_module.Actor.html#torchrl.modules.tensordict_module.Actor)(*args, **kwargs) | General class for deterministic actors in RL. |
| --- | --- |
| [`DiffusionActor`](generated/torchrl.modules.tensordict_module.DiffusionActor.html#torchrl.modules.tensordict_module.DiffusionActor)(*args, **kwargs) | Diffusion-based actor for RL. |
| [`MultiStepActorWrapper`](generated/torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper)(*args, **kwargs) | A wrapper around a multi-action actor. |
| [`SafeModule`](generated/torchrl.modules.tensordict_module.SafeModule.html#torchrl.modules.tensordict_module.SafeModule)(*args, **kwargs) | [`tensordict.nn.TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) subclass that accepts a [`TensorSpec`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec) as argument to control the output domain. |
| [`SafeSequential`](generated/torchrl.modules.tensordict_module.SafeSequential.html#torchrl.modules.tensordict_module.SafeSequential)(*args, **kwargs) | A safe sequence of TensorDictModules. |
| [`TanhModule`](generated/torchrl.modules.tensordict_module.TanhModule.html#torchrl.modules.tensordict_module.TanhModule)(*args, **kwargs) | A Tanh module for deterministic policies with bounded action space. |
| [`RandomPolicy`](generated/torchrl.modules.tensordict_module.RandomPolicy.html#torchrl.modules.tensordict_module.RandomPolicy)([action_spec, action_key]) | A random policy for data collectors. |

## Probabilistic actors

| [`ProbabilisticActor`](generated/torchrl.modules.tensordict_module.ProbabilisticActor.html#torchrl.modules.tensordict_module.ProbabilisticActor)(*args, **kwargs) | General class for probabilistic actors in RL. |
| --- | --- |
| [`SafeProbabilisticModule`](generated/torchrl.modules.tensordict_module.SafeProbabilisticModule.html#torchrl.modules.tensordict_module.SafeProbabilisticModule)(*args, **kwargs) | [`tensordict.nn.ProbabilisticTensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) subclass that accepts a `TensorSpec` as an argument to control the output domain. |
| [`SafeProbabilisticTensorDictSequential`](generated/torchrl.modules.tensordict_module.SafeProbabilisticTensorDictSequential.html#torchrl.modules.tensordict_module.SafeProbabilisticTensorDictSequential)(*args, ...) | [`tensordict.nn.ProbabilisticTensorDictSequential`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential) subclass that accepts a `TensorSpec` as argument to control the output domain. |

## Q-Value actors

| [`QValueActor`](generated/torchrl.modules.QValueActor.html#torchrl.modules.QValueActor)(*args, **kwargs) | A Q-Value actor class. |
| --- | --- |
| [`DistributionalQValueActor`](generated/torchrl.modules.DistributionalQValueActor.html#torchrl.modules.DistributionalQValueActor)(*args, **kwargs) | A Distributional DQN actor class. |
| [`QValueModule`](generated/torchrl.modules.QValueModule.html#torchrl.modules.QValueModule)(*args, **kwargs) | Q-Value TensorDictModule for Q-value policies. |
| [`DistributionalQValueModule`](generated/torchrl.modules.DistributionalQValueModule.html#torchrl.modules.DistributionalQValueModule)(*args, **kwargs) | Distributional Q-Value hook for Q-value policies. |