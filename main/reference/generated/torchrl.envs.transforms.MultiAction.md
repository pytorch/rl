# MultiAction

*class*torchrl.envs.transforms.MultiAction(***, *dim: int = 1*, *stack_rewards: bool = True*, *stack_observations: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#MultiAction)

A transform to execute multiple actions in the parent environment.

This transform unbinds the actions along a specific dimension and passes each action independently.
The returned transform can be either a stack of the observations gathered during the steps or only the
last observation (and similarly for the rewards, see args below).

By default, the actions must be stacked along the first dimension after the root tensordict batch-dims, i.e.

```
>>> td = policy(td)
>>> actions = td.select(*env.action_keys)
>>> # Adapt the batch-size
>>> actions = actions.auto_batch_size_(td.ndim + 1)
>>> # Step-wise actions
>>> actions = actions.unbind(-1)
```

If a "done" entry is encountered, the next steps are skipped for the env that has reached that state.

Note

If a transform is appended before the MultiAction, it will be called multiple times. If it is appended
after, it will be called once per macro-step.

Keyword Arguments:

- **dim** (*int**,**optional*) - the stack dimension with respect to the tensordict `ndim` attribute.
Must be greater than 0. Defaults to `1` (the first dimension after the batch-dims).
- **stack_rewards** (*bool**,**optional*) - if `True`, each step's reward will be stack in the output tensordict.
If `False`, only the last reward will be returned. The reward spec is adapted accordingly. The
stack dimension is the same as the action stack dimension. Defaults to `True`.
- **stack_observations** (*bool**,**optional*) - if `True`, each step's observation will be stack in the output tensordict.
If `False`, only the last observation will be returned. The observation spec is adapted accordingly. The
stack dimension is the same as the action stack dimension. Defaults to `False`.

See also

[`ActionChunkTransform`](torchrl.envs.transforms.ActionChunkTransform.html#torchrl.envs.transforms.ActionChunkTransform) - when
the stacked actions are a chunk policy's *prediction* (overlapping
per-step training targets) rather than a macro action to replay
verbatim. The chunk transform builds the training targets on the data
path and, attached to an env, executes only the first action of each
predicted chunk (re-planning at every step) instead of stepping the
base env once per action.

transform_input_spec(*input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_action.html#MultiAction.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_action.html#MultiAction.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform