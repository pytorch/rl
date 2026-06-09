# ValueEstimatorBase

*class*torchrl.objectives.value.ValueEstimatorBase(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/value/advantages.html#ValueEstimatorBase)

An abstract parent class for value function modules.

Its `ValueFunctionBase.forward()` method will compute the value (given
by the value network) and the value estimate (given by the value estimator)
as well as the advantage and write these values in the output tensordict.

If only the value estimate is needed, the `ValueFunctionBase.value_estimate()`
should be used instead.

Keyword Arguments:

- **value_chunk_size** (*int**,**optional*) - if set, splits value-network calls
into chunks of this many elements along the leading dimension.
Defaults to `None`.
- **num_chunks** (*int**,**optional*) - if set, splits value-network calls into
this many chunks along the leading dimension. Mutually exclusive
with `value_chunk_size`. `num_chunk` is accepted as an alias.
Defaults to `None`.
- **num_chunk** (*int**,**optional*) - alias for `num_chunks`. Cannot be set
together with a different `num_chunks` value. Defaults to `None`.
- **shifted_budget** (*int**,**optional*) - number of extra value-network time slots
used when `shifted=True`. `1` uses a `T+1`
budget, `2` can represent one internal reset plus the rollout
boundary without dropping samples, and so on. Defaults to `1`.

default_keys

alias of `_AcceptedKeys`

*classmethod*for_loss(*loss_module*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/value/advantages.html#ValueEstimatorBase.for_loss)

Construct an instance configured against `loss_module`.

Used by the value-estimator registry
(`build_value_estimator()`) to keep
per-estimator wiring quirks out of every loss class. The default
implementation picks up `loss_module.critic_network` if present,
falling back to `loss_module.value_network`, and forwards the
remaining `hyperparams` to the constructor.

A loss that owns a value module under a non-standard name can pass
`value_network=<the module>` through
`dispatch_value_estimator()` -- it
wins over the auto-detected one. Estimator subclasses with
additional dependencies (e.g. [`VTrace`](torchrl.objectives.value.VTrace.html#torchrl.objectives.value.VTrace) needing the actor)
override this method.

*abstract*forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *params: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *target_params: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/value/advantages.html#ValueEstimatorBase.forward)

Computes the advantage estimate given the data in tensordict.

If a functional module is provided, a nested TensorDict containing the parameters
(and if relevant the target parameters) can be passed to the module.

Parameters:

**tensordict** (*TensorDictBase*) - A TensorDict containing the data
(an observation key, `"action"`, `("next", "reward")`,
`("next", "done")`, `("next", "terminated")`,
and `"next"` tensordict state as returned by the environment)
necessary to compute the value estimates and the TDEstimate.
The data passed to this module should be structured as
`[*B, T, *F]` where `B` are
the batch size, `T` the time dimension and `F` the
feature dimension(s). The tensordict must have shape `[*B, T]`.

Keyword Arguments:

- **params** (*TensorDictBase**,**optional*) - A nested TensorDict containing the params
to be passed to the functional value network module.
- **target_params** (*TensorDictBase**,**optional*) - A nested TensorDict containing the
target params to be passed to the functional value network module.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device where the buffers will be instantiated.
Defaults to `torch.get_default_device()`.

Returns:

An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

set_keys(***kwargs*) → None[[source]](../../_modules/torchrl/objectives/value/advantages.html#ValueEstimatorBase.set_keys)

Set tensordict key names.

value_estimate(*tensordict*, *target_params: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *next_value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/value/advantages.html#ValueEstimatorBase.value_estimate)

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