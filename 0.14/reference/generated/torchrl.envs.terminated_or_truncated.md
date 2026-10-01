# terminated_or_truncated

torchrl.envs.terminated_or_truncated(*data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *full_done_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec) | None = None*, *key: str = '_reset'*, *write_full_false: bool = False*) → bool[[source]](../../_modules/torchrl/envs/utils.html#terminated_or_truncated)

Reads the done / terminated / truncated keys within a tensordict, and writes a new tensor where the values of both signals are aggregated.

The modification occurs in-place within the TensorDict instance provided.
This function can be used to compute the "_reset" signals in batched
or multiagent settings, hence the default name of the output key.

Parameters:

- **data** (*TensorDictBase*) - the input data, generally resulting from a call
to [`step()`](torchrl.envs.EnvBase.html#id4).
- **full_done_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - the done_spec from the env,
indicating where the done leaves have to be found.
If not provided, the default
`"done"`, `"terminated"` and `"truncated"` entries will be
searched for in the data.
- **key** (*NestedKey**,**optional*) -

where the aggregated result should be written.
If `None`, then the function will not write any key but just output
whether any of the done values was true.

Note

if a value is already present for the `key` entry,
the previous value will prevail and no update will be achieved.
- **write_full_false** (*bool**,**optional*) - if `True`, the reset keys will be
written even if the output is `False` (ie, no done is `True`
in the provided data structure).
Defaults to `False`.

Returns: a boolean value indicating whether any of the done states found in the data

contained a `True`.

Examples

```
>>> from torchrl.data.tensor_specs import Categorical
>>> from tensordict import TensorDict
>>> spec = Composite(
... done=Categorical(2, dtype=torch.bool),
... truncated=Categorical(2, dtype=torch.bool),
... nested=Composite(
... done=Categorical(2, dtype=torch.bool),
... truncated=Categorical(2, dtype=torch.bool),
... )
... )
>>> data = TensorDict({
... "done": True, "truncated": False,
... "nested": {"done": False, "truncated": True}},
... batch_size=[]
... )
>>> data = _terminated_or_truncated(data, spec)
>>> print(data["_reset"])
tensor(True)
>>> print(data["nested", "_reset"])
tensor(True)
```