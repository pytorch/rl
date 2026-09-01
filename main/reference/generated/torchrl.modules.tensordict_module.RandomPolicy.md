# RandomPolicy

*class*torchrl.modules.tensordict_module.RandomPolicy(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec) | None = None*, *action_key: NestedKey = 'action'*)[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#RandomPolicy)

A random policy for data collectors.

This is a wrapper around the action_spec.rand method.

Parameters:

- **action_spec** - TensorSpec object describing the action specs. If `None`,
the spec is initialized lazily (e.g. by a collector from
`env.full_action_spec`). A `RandomPolicy` with no spec will
raise if called before the spec is set.
- **action_key** - key at which the action is written. Defaults to `"action"`.

Examples

```
>>> from tensordict import TensorDict
>>> from torchrl.data.tensor_specs import Bounded
>>> action_spec = Bounded(-torch.ones(3), torch.ones(3))
>>> actor = RandomPolicy(action_spec=action_spec)
>>> td = actor(TensorDict()) # selects a random action in the cube [-1; 1]
```

Lazy initialization -- let the collector fill in the spec from the env:

```
>>> from torchrl.collectors import Collector
>>> collector = Collector(env, RandomPolicy(), ...)
```

set_action_spec_from_env(*env: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*) → None[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#RandomPolicy.set_action_spec_from_env)

Initialize `action_spec` from `env.full_action_spec`.

No-op if the spec is already set. Intended for lazy initialization by
data collectors.