# TD1Estimator

*class*torchrl.objectives.value.TD1Estimator(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/value/advantages.html#TD1Estimator)

\(\infty\)-Temporal Difference (TD(1)) estimate of advantage function.

Keyword Arguments:

- **gamma** (*scalar*) - exponential mean discount.
- **value_network** (*TensorDictModule*) - value operator used to retrieve the value estimates.
- **average_rewards** (*bool**,**optional*) - if `True`, rewards will be standardized
before the TD is computed.
- **differentiable** (*bool**,**optional*) -

if `True`, gradients are propagated through
the computation of the value function. Default is `False`.

Note

The proper way to make the function call non-differentiable is to
decorate it in a torch.no_grad() context manager/decorator or
pass detached parameters for functional modules.
- **skip_existing** (*bool**,**optional*) - if `True`, the value network will skip
modules which outputs are already present in the tensordict.
Defaults to `None`, i.e., the value of [`tensordict.nn.skip_existing()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.skip_existing.html#tensordict.nn.skip_existing)
is not affected.
- **advantage_key** (*str**or**tuple**of**str**,**optional*) - [Deprecated] the key of
the advantage entry. Defaults to `"advantage"`.
- **value_target_key** (*str**or**tuple**of**str**,**optional*) - [Deprecated] the key
of the advantage entry. Defaults to `"value_target"`.
- **value_key** (*str**or**tuple**of**str**,**optional*) - [Deprecated] the value key to
read from the input tensordict. Defaults to `"state_value"`.
- **shifted** (*bool**,**optional*) -

controls how value and next-value
are obtained from the value network. `False` (default) calls
the value network twice (once on the root tensordict, once on
`"next"`), which is correct whenever `"next"` may differ
non-trivially from `obs[t+1]`. Truthy values request a single
call:

- `True`: fixed-budget single-call path. Inserts the true
`("next", <in_key>)` entry after every internal truncation
(`done & ~terminated`), shifts subsequent samples to the
right inside a sequence of length `T + shifted_budget` and
masks the displaced suffix via `"shifted_valid"`. Terminal
steps (`done & terminated`) do not consume budget. Retained
samples use exact next observations.

Note

**Single-step rollout assumption.** `shifted=True` relies
on the standard one-step rollout layout produced by
`env.step` + auto-reset: at every position where
`done[t] = False`, the value-net inputs in
`("next", <in_key>)[t]` are expected to equal
`<in_key>[t+1]`. The backend uses this invariant to
evaluate `V` once over a fused
`[T + shifted_budget]` sequence instead of twice over
`[T]` streams.

The canonical pipeline that breaks the invariant is
**multi-step return processing** (`MultiStep` / n-step
bootstrapping), which rewrites `("next", obs)[t]` to
`obs[t+n]` with `n > 1`. `shifted=True` is unsupported
with multi-step returns -- use `shifted=False` instead.

Single-call paths also require that the parameters at time
`t` and `t+1` are identical (i.e. `target_params` is
not used).

Defaults to `False`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device where the buffers will be instantiated.
Defaults to `torch.get_default_device()`.
- **time_dim** (*int**,**optional*) - the dimension corresponding to the time
in the input tensordict. If not provided, defaults to the dimension
marked with the `"time"` name if any, and to the last dimension
otherwise. Can be overridden during a call to
`value_estimate()`.
Negative dimensions are considered with respect to the input
tensordict.
- **deactivate_vmap** (*bool**,**optional*) - whether to deactivate vmap calls and replace them with a plain for loop.
Defaults to `False`.
- **value_chunk_size** (*int**,**optional*) - if set, splits value-network calls
into chunks of this many elements along `value_chunk_dim`.
Defaults to `None`.
- **num_chunks** (*int**,**optional*) - if set, splits value-network calls into
this many chunks along `value_chunk_dim`. Mutually exclusive
with `value_chunk_size`. `num_chunk` is accepted as an alias.
Defaults to `None`.
- **num_chunk** (*int**,**optional*) - alias for `num_chunks`. Cannot be set
together with a different `num_chunks` value. Defaults to `None`.
- **value_chunk_dim** (*int**,**optional*) - dimension used for chunked value-network
calls. Defaults to `0`.
- **shifted_budget** (*int**,**optional*) - number of extra value-network time slots
used when `shifted=True`. `1` uses a `T+1`
budget, `2` can represent one internal reset plus the rollout
boundary without dropping samples, and so on. Defaults to `1`.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, ***, *params: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *target_params: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/value/advantages.html#TD1Estimator.forward)

Computes the TD(1) advantage given the data in tensordict.

If a functional module is provided, a nested TensorDict containing the parameters
(and if relevant the target parameters) can be passed to the module.

Parameters:

**tensordict** (*TensorDictBase*) - A TensorDict containing the data
(an observation key, `"action"`, `("next", "reward")`,
`("next", "done")`, `("next", "terminated")`,
and `"next"` tensordict state as returned by the environment)
necessary to compute the value estimates and the TDEstimate.
The data passed to this module should be structured as `[*B, T, *F]` where `B` are
the batch size, `T` the time dimension and `F` the feature dimension(s).
The tensordict must have shape `[*B, T]`.

Keyword Arguments:

- **params** (*TensorDictBase**,**optional*) - A nested TensorDict containing the params
to be passed to the functional value network module.
- **target_params** (*TensorDictBase**,**optional*) - A nested TensorDict containing the
target params to be passed to the functional value network module.

Returns:

An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

Examples

```
>>> from tensordict import TensorDict
>>> value_net = TensorDictModule(
... nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
... )
>>> module = TDEstimate(
... gamma=0.98,
... value_network=value_net,
... )
>>> obs, next_obs = torch.randn(2, 1, 10, 3)
>>> reward = torch.randn(1, 10, 1)
>>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
>>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
>>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs, "done": done, "reward": reward, "terminated": terminated}}, [1, 10])
>>> _ = module(tensordict)
>>> assert "advantage" in tensordict.keys()
```

The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

Examples

```
>>> value_net = TensorDictModule(
... nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
... )
>>> module = TDEstimate(
... gamma=0.98,
... value_network=value_net,
... )
>>> obs, next_obs = torch.randn(2, 1, 10, 3)
>>> reward = torch.randn(1, 10, 1)
>>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
>>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
>>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)
```

value_estimate(*tensordict*, *target_params: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *next_value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *time_dim: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/value/advantages.html#TD1Estimator.value_estimate)

Gets a value estimate, usually used as a target value for the value network.

If the state value key is present under `tensordict.get(("next", self.tensor_keys.value))`
then this value will be used without recurring to the value network.

Parameters:

- **tensordict** (*TensorDictBase*) - the tensordict containing the data to
read.
- **target_params** (*TensorDictBase**,**optional*) - A nested TensorDict containing the
target params to be passed to the functional value network module.
- **next_value** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - the value of the next state
or state-action pair. Exclusive with `target_params`.
- ****kwargs** - the keyword arguments to be passed to the value network.

Returns: a tensor corresponding to the state value.