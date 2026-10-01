# GAE

*class*torchrl.objectives.value.GAE(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/value/advantages.html#GAE)

A class wrapper around the generalized advantage estimate functional.

Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
[https://arxiv.org/pdf/1506.02438.pdf](https://arxiv.org/pdf/1506.02438.pdf) for more context.

Parameters:

- **gamma** (*scalar*) - exponential mean discount.
- **lmbda** (*scalar*) - trajectory discount.
- **value_network** (*TensorDictModule**,**optional*) - value operator used to retrieve the value estimates.
If `None`, this module will expect the `"state_value"` keys to be already filled, and
will not call the value network to produce it.
- **average_gae** (*bool*) - if `True`, the resulting GAE values will be standardized.
Default is `False`.
- **differentiable** (*bool**,**optional*) -

if `True`, gradients are propagated through
the computation of the value function. Default is `False`.

Note

The proper way to make the function call non-differentiable is to
decorate it in a torch.no_grad() context manager/decorator or
pass detached parameters for functional modules.
- **vectorized** (*bool**,**optional*) - whether to use the vectorized version of the
lambda return. Default is True if not compiling.
- **skip_existing** (*bool**,**optional*) - if `True`, the value network will skip
modules which outputs are already present in the tensordict.
Defaults to `None`, i.e., the value of [`tensordict.nn.skip_existing()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.skip_existing.html#tensordict.nn.skip_existing)
is not affected.
Defaults to "state_value".
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
- **auto_reset_env** (*bool**,**optional*) - if `True`, the last `"next"` state
of the episode isn't valid, so the GAE calculation will use the `value`
instead of `next_value` to bootstrap truncated episodes.
- **deactivate_vmap** (*bool**,**optional*) - if `True`, no vmap call will be used, and
vectorized maps will be replaced with simple for loops. Defaults to `False`.
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

GAE will return an `"advantage"` entry containing the advantage value. It will also
return a `"value_target"` entry with the return value that is to be used
to train the value network. Finally, if `gradient_mode` is `True`,
an additional and differentiable `"value_error"` entry will be returned,
which simply represents the difference between the return and the value network
output (i.e. an additional distance loss should be applied to that signed value).

Note

As other advantage functions do, if the `value_key` is already present
in the input tensordict, the GAE module will ignore the calls to the value
network (if any) and use the provided value instead.

Note

GAE can be used with value networks that rely on recurrent neural networks, provided that the
init markers ("is_init") and terminated / truncated markers are properly set.
With `shifted=True`, reset next-observations are inserted into a
fixed-shape value-network call according to `shifted_budget`. If `shifted=False`,
the root and `"next"` trajectories are stacked and the value network is called with `vmap` over the
stack of trajectories. Because RNNs require a fair amount of control flow, they are currently not
compatible with `torch.vmap` and, as such, the `deactivate_vmap` option must be turned on in these
cases. Similarly, if `shifted=False`, the `"is_init"` entry of the root tensordict will be copied
onto the `"is_init"` of the `"next"` entry, such that trajectories are well separated both for root
and `"next"` data.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, ***, *params: list[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | None = None*, *target_params: list[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | None = None*, *time_dim: int | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/value/advantages.html#GAE.forward)

Computes the GAE given the data in tensordict.

If a functional module is provided, a nested TensorDict containing the parameters
(and if relevant the target parameters) can be passed to the module.

Parameters:

**tensordict** (*TensorDictBase*) - A TensorDict containing the data
(an observation key, `"action"`, `("next", "reward")`,
`("next", "done")`, `("next", "terminated")`,
and `"next"` tensordict state as returned by the environment)
necessary to compute the value estimates and the GAE.
The data passed to this module should be structured as `[*B, T, *F]` where `B` are
the batch size, `T` the time dimension and `F` the feature dimension(s).
The tensordict must have shape `[*B, T]`.

Keyword Arguments:

- **params** (*TensorDictBase**,**optional*) - A nested TensorDict containing the params
to be passed to the functional value network module.
- **target_params** (*TensorDictBase**,**optional*) - A nested TensorDict containing the
target params to be passed to the functional value network module.
- **time_dim** (*int**,**optional*) - the dimension corresponding to the time
in the input tensordict. If not provided, defaults to the dimension
marked with the `"time"` name if any, and to the last dimension
otherwise.
Negative dimensions are considered with respect to the input
tensordict.

Returns:

An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

Examples

```
>>> from tensordict import TensorDict
>>> value_net = TensorDictModule(
... nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
... )
>>> module = GAE(
... gamma=0.98,
... lmbda=0.94,
... value_network=value_net,
... differentiable=False,
... )
>>> obs, next_obs = torch.randn(2, 1, 10, 3)
>>> reward = torch.randn(1, 10, 1)
>>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
>>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
>>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs}, "done": done, "reward": reward, "terminated": terminated}, [1, 10])
>>> _ = module(tensordict)
>>> assert "advantage" in tensordict.keys()
```

The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

Examples

```
>>> value_net = TensorDictModule(
... nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
... )
>>> module = GAE(
... gamma=0.98,
... lmbda=0.94,
... value_network=value_net,
... differentiable=False,
... )
>>> obs, next_obs = torch.randn(2, 1, 10, 3)
>>> reward = torch.randn(1, 10, 1)
>>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
>>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
>>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)
```

value_estimate(*tensordict*, *params: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *target_params: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *time_dim: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/value/advantages.html#GAE.value_estimate)

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