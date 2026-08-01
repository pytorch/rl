# MultiAction

*class*torchrl.envs.transforms.MultiAction(***, *dim: int = 1*, *stack_rewards: bool = True*, *stack_observations: bool = False*, *action_key: NestedKey | None = None*, *chunk_key: NestedKey | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#MultiAction)

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

Note

Extra entries written by the policy alongside the actions (e.g. the action tokens and
log-probabilities of a token-head policy) are left untouched on the root tensordict and therefore
ride along on the outer (macro-step) transition: each outer step of a rollout carries the policy
outputs of the chunk decided at that step.

Note

When a done state fires inside the chunk (with `stack_rewards=True`), the reward stack of
that outer step holds the executed steps' rewards followed by a single zero-filled slot for the
skipped remainder of the chunk. Its length therefore differs from a full chunk's, and stacking
such outer steps in a rollout yields a lazy stack with ragged reward entries. If the per-chunk
reward is computed from the outer transition anyway (e.g. with
[`SuccessReward`](torchrl.envs.transforms.SuccessReward.html#torchrl.envs.transforms.SuccessReward) appended after this transform), pass
`stack_rewards=False` to keep the outer transition dense and uniform.

Note

Skipping the remaining steps after a done state relies on the `"_step"` partial-step
entry. Single (unbatched) environments and batched environments
([`SerialEnv`](torchrl.envs.SerialEnv.html#torchrl.envs.SerialEnv) / [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv)) handle it natively; for a
batch-locked vectorized environment, the base environment's `_step` is trusted to honor the
mask itself (see [`step()`](torchrl.envs.EnvBase.html#id4)) and environments that ignore it will keep
stepping every sub-environment until the end of the chunk.

Keyword Arguments:

- **dim** (*int**,**optional*) - the stack dimension with respect to the tensordict `ndim` attribute.
Must be greater than 0. Defaults to `1` (the first dimension after the batch-dims).
- **stack_rewards** (*bool**,**optional*) - if `True`, each step's reward will be stack in the output tensordict.
If `False`, only the last reward will be returned. The reward spec is adapted accordingly. The
stack dimension is the same as the action stack dimension. Defaults to `True`.
- **stack_observations** (*bool**,**optional*) - if `True`, each step's observation will be stack in the output tensordict.
If `False`, only the last observation will be returned. The observation spec is adapted accordingly. The
stack dimension is the same as the action stack dimension. Defaults to `False`.
- **action_key** (*NestedKey**,**optional*) - the one-step action key consumed by
the base environment. Defaults to the parent environment action key.
- **chunk_key** (*NestedKey**,**optional*) - the policy-facing key that holds the
stacked actions. Defaults to `action_key` for backward
compatibility. Set this to values such as
`("vla_action", "chunk")` when a chunk policy should act through
`MultiAction` without re-keying its output. See also
`from_vla()`.

See also

[`ActionChunkTransform`](torchrl.envs.transforms.ActionChunkTransform.html#torchrl.envs.transforms.ActionChunkTransform) - when
the stacked actions are a chunk policy's *prediction* (overlapping
per-step training targets) rather than a macro action to replay
verbatim. The chunk transform builds the training targets on the data
path and, attached to an env, executes only the first action of each
predicted chunk (re-planning at every step) instead of stepping the
base env once per action.

*classmethod*from_vla(***, *action_key: NestedKey = 'action'*, ***kwargs*) → MultiAction[[source]](../../_modules/torchrl/envs/transforms/_action.html#MultiAction.from_vla)

Build a `MultiAction` that consumes the default VLA chunk key.

Parameters:

**action_key** (*NestedKey*) - the one-step action key consumed by the base
environment. Defaults to `"action"`.

:keyword Additional `MultiAction` keyword arguments.:

Examples

```
>>> from torchrl.envs.transforms import MultiAction
>>> transform = MultiAction.from_vla(stack_rewards=False)
>>> transform.out_keys_inv
[('vla_action', 'chunk')]
```

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