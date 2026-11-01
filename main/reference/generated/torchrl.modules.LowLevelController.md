# LowLevelController

*class*torchrl.modules.LowLevelController(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/controllers.html#LowLevelController)

Deploy a TensorDict policy on independent controller instances.

The adapter and policy run on a private, flattened TensorDict. Only the
low-level action and the namespaced next state are written back. Policy
weights are shared; recurrent and adapter state remain independent for
every environment and every member of the selected group.

Parameters:

- **policy** (*TensorDictModuleBase*) - pretrained policy with explicit TensorDict
inputs, outputs, and primers for persistent state.
- **decision_spec** ([*Composite*](torchrl.data.Composite.html#torchrl.data.Composite)) - unbatched spec for one controller's high-level
decisions. All leaves are held fixed during a closed-loop macro step.

Keyword Arguments:

- **adapter** (*TensorDictModuleBase**or*[*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*,**optional*) - module applied
before the policy, using keys relative to the group. Defaults to
None (pass inputs directly to the policy).
- **group_key** (*NestedKey**,**optional*) - group whose batch dimensions enumerate
controller instances. Defaults to None (the root TensorDict).
- **state_key** (*NestedKey**,**optional*) - state namespace within the group.
Defaults to "controller". The non-private namespace is retained by
collectors and replay buffers.
- **policy_action_key** (*NestedKey**,**optional*) - action output of the wrapped
policy. Defaults to "action".
- **action_key** (*NestedKey**,**optional*) - destination action within the group.
Defaults to "action".
- **reset_key** (*NestedKey**,**optional*) - additional per-instance reset signal in
the group's next observation, such as "fallen". Defaults to None
(ordinary environment resets only).

The module does not disable gradients or change the policy's training mode.
[`ClosedLoopMultiAction`](torchrl.envs.transforms.ClosedLoopMultiAction.html#torchrl.envs.transforms.ClosedLoopMultiAction) controls inference
when the controller is deployed in an environment.

Examples

```
>>> import torch
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.data import Bounded, Composite
>>> from torchrl.envs.transforms import ClosedLoopMultiAction
>>> from torchrl.testing.mocking_classes import CountingEnv
>>> policy = TensorDictModule(
... torch.nn.Identity(), in_keys=["command"], out_keys=["action"])
>>> controller = LowLevelController(
... policy, Composite(command=Bounded(0, 1, shape=(1,))))
>>> env = ClosedLoopMultiAction.from_env(CountingEnv(), controller, steps=2)
>>> td = env.reset().set("command", torch.ones(1))
>>> env.step(td)["next", "observation"]
tensor([2], dtype=torch.int32)
>>> env.close()
```

See also

[`ClosedLoopMultiAction`](torchrl.envs.transforms.ClosedLoopMultiAction.html#torchrl.envs.transforms.ClosedLoopMultiAction) repeatedly
executes the controller against fresh observations;
[`MicroDuckSkillController`](torchrl.envs.MicroDuckSkillController.html#torchrl.envs.MicroDuckSkillController) specializes the
controller for a task-conditioned MicroDuck policy; and
`LowLevelControllerConfig`
exposes this class through Hydra configuration.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/controllers.html#LowLevelController.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

make_tensordict_primer() → [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)[[source]](../../_modules/torchrl/modules/tensordict_module/controllers.html#LowLevelController.make_tensordict_primer)

Return the group's state initialization and partial-reset transforms.

Returns:

primers that infer group shapes from the parent env and
retain the original policy and adapter initialization values.

Return type:

[Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)