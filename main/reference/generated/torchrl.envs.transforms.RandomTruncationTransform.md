# RandomTruncationTransform

*class*torchrl.envs.transforms.RandomTruncationTransform(*min_horizon: int*, *max_horizon: int*, *prob: float = 0.0*, *first_episode_prob: float | None = None*, ***, *step_count_key: NestedKey = 'step_count'*, *truncated_key: NestedKey = 'truncated'*, *done_key: NestedKey = 'done'*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#RandomTruncationTransform)

Randomly truncate episodes to decorrelate synchronized batched envs.

When many batched environments share the same `max_episode_steps`, all
environments hit truncation at nearly the same step, creating correlated
waves of start-of-episode data in the replay buffer. This transform
breaks that synchronisation by assigning each environment a random horizon.

On the **first reset** every environment receives a horizon drawn from
`Uniform(1, max_horizon)` so they immediately spread across different
phases of the episode. On **subsequent resets**, with probability
`prob` a new horizon is sampled from `Uniform(min_horizon, max_horizon)`;
otherwise the full `max_horizon` is used.

`first_episode_prob` controls the truncation probability for each
environment's first episode after the initial spread. By default it matches
`prob` so that `prob=0.0` disables all subsequent random truncation
after the initial spread. Setting it higher than `prob` can accelerate
decorrelation when batch sizes are large relative to `max_horizon`.

Note

This transform must be placed **after** `StepCounter`
in the transform chain. It reads the same step-count
`NestedKey` that the validator accepts - a key
whose last component equals `step_count_key` (default
`"step_count"`), not only a root-level `"step_count"`. Truncation
flags are written next to that counter (e.g. `("agent", "step_count")`
produces `("agent", "truncated")` / `("agent", "done")`), matching
`StepCounter`. Pass the same `step_count_key`
used by `StepCounter` when that key is customized.

Parameters:

- **min_horizon** (*int*) - minimum horizon for random truncation
(inclusive).
- **max_horizon** (*int*) - maximum horizon for random truncation
(inclusive). Also used as the full-length horizon when no random
truncation is applied. This should typically match the
environment's `max_episode_steps`, which unfortunately cannot
be retrieved automatically in general.
- **prob** (*float**,**optional*) - probability of sampling a random horizon on
each subsequent reset. Defaults to `0.0` (only the initial
spread is applied). When nonzero, a low value (e.g. `0.01`) is
recommended - frequent truncation can negatively impact training.
- **first_episode_prob** (*float**,**optional*) - truncation probability for each
environment's first episode after the initial spread. Defaults to
`prob` when omitted.

Keyword Arguments:

- **step_count_key** (*NestedKey**,**optional*) - key of the step counter written
by `StepCounter`. A string is matched against
the last component of any nested key (so `"step_count"` finds
`("agent", "step_count")`). A tuple is used as an exact
`NestedKey`. Must match
`StepCounter`'s `step_count_key` when that
is customized. Defaults to `"step_count"`.
- **truncated_key** (*NestedKey**,**optional*) - key where the truncation flag is
written. A string is placed next to the resolved step-count key
via last-component replacement, matching
`StepCounter`. A tuple is used as an exact
key. Defaults to `"truncated"`.
- **done_key** (*NestedKey**,**optional*) - key of the done flag that is OR-ed
with the truncation signal, resolved like `truncated_key`.
Defaults to `"done"`.

Examples

```
>>> from torchrl.envs import GymEnv, TransformedEnv, StepCounter
>>> base_env = GymEnv("Pendulum-v1")
>>> env = TransformedEnv(
... base_env,
... Compose(
... StepCounter(),
... RandomTruncationTransform(
... prob=0.1, min_horizon=50, max_horizon=200
... ),
... ),
... )
>>> rollout = env.rollout(300)
>>> # Episode length will be at most 200 steps
>>> print(rollout.shape)
torch.Size([...])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_env.html#RandomTruncationTransform.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

By default, this method:

- calls directly `_apply_transform()`.
- does not call `_step()` or `_call()`.

This method is not called within env.step at any point. However, is is called within
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).

Note

`forward` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Examples

```
>>> class TransformThatMeasuresBytes(Transform):
... '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
... def __init__(self):
... super().__init__(in_keys=[], out_keys=["bytes"])
...
... def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
... bytes_in_td = tensordict.bytes()
... tensordict["bytes"] = bytes
... return tensordict
>>> t = TransformThatMeasuresBytes()
>>> env = env.append_transform(t) # works within envs
>>> t(TensorDict(a=0)) # Works offline too.
```

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_env.html#RandomTruncationTransform.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform