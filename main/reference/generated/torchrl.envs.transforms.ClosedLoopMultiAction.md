# ClosedLoopMultiAction

*class*torchrl.envs.transforms.ClosedLoopMultiAction(*controller: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*, ***, *steps: int*, *decision_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite) | None = None*, *reward_aggregation: Literal['last', 'stack', 'sum', 'mean'] = 'sum'*, *exploration_type: [InteractionType](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) = InteractionType.DETERMINISTIC*, *no_grad: bool = True*, *dim: int = 1*, *stack_observations: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ClosedLoopMultiAction)

Execute a controller against fresh observations for one high-level decision.

Unlike an action chunk, the low-level action is recomputed at every physical
step. High-level decisions and their log probabilities remain on the outer
transition. Finished environments stop executing the controller.

Parameters:

**controller** (*TensorDictModuleBase*) - low-level policy, typically
[`LowLevelController`](torchrl.modules.LowLevelController.html#torchrl.modules.LowLevelController).

Keyword Arguments:

- **steps** (*int*) - positive number of physical steps per decision.
- **decision_spec** ([*Composite*](torchrl.data.Composite.html#torchrl.data.Composite)*,**optional*) - complete policy-facing action spec,
including environment batch dimensions. Defaults to None, inferring
the spec from LowLevelController and preserving unrelated actions.
- **reward_aggregation** (*str**,**optional*) - "sum", "mean", "last", or "stack".
Defaults to "sum". Mean counts only executed steps; last returns
the last executed reward. Stack uses MultiAction's ragged convention.
- **exploration_type** (*ExplorationType**,**optional*) - controller sampling mode.
Defaults to DETERMINISTIC; the caller's exploration mode is restored.
- **no_grad** (*bool**,**optional*) - disable gradients during controller inference.
Defaults to True. This does not freeze the policy's parameters.
- **dim** (*int**,**optional*) - stack dimension relative to each leaf's containing
TensorDict batch dimensions. Defaults to 1, keeping agent dimensions
before the stack dimension.
- **stack_observations** (*bool**,**optional*) - return stacked inner observations.
Defaults to False (the final observation). Persistent state remains
unstacked. The controller uses the latest observation on its next call.

Use `from_env()` to install controller primers before this transform.
The base environment must honor partial-step masks, as for MultiAction.
Discount factors on the resulting environment count high-level decisions.
See also `ClosedLoopMultiActionConfig`.

Examples

```
>>> import torch
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.data import Bounded, Composite
>>> from torchrl.modules import LowLevelController
>>> from torchrl.testing.mocking_classes import CountingEnv
>>> policy = TensorDictModule(
... torch.nn.Identity(), in_keys=["command"], out_keys=["action"])
>>> controller = LowLevelController(
... policy, Composite(command=Bounded(0, 1, shape=(1,))))
>>> env = ClosedLoopMultiAction.from_env(CountingEnv(), controller, steps=3)
>>> td = env.reset().set("command", torch.ones(1))
>>> env.step(td)["next", "observation"]
tensor([3], dtype=torch.int32)
>>> env.close()
```

*classmethod*from_env(*env: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*, *controller: TensorDictModuleBase*, ***, *steps: int*, *init_key: str = 'is_init'*, ***kwargs: Any*) → [TransformedEnv](torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ClosedLoopMultiAction.from_env)

Wrap an environment, automatically installing controller state.

Parameters:

- **env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - physical environment.
- **controller** (*TensorDictModuleBase*) - controller to execute.

Keyword Arguments:

- **steps** (*int*) - positive number of controller steps per decision.
- **init_key** (*str**,**optional*) - episode-start marker. Defaults to "is_init".
- ****kwargs** - additional ClosedLoopMultiAction constructor arguments.

Returns:

environment exposing high-level actions.

Return type:

[TransformedEnv](torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv)

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ClosedLoopMultiAction.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform