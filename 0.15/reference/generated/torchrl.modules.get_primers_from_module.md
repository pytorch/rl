# get_primers_from_module

*class*torchrl.modules.get_primers_from_module(*module*, *warn=True*, *strict=True*)[[source]](../../_modules/torchrl/modules/utils/utils.html#get_primers_from_module)

Get all tensordict primers from all submodules of a module.

This method is useful for retrieving primers from modules that are contained within a
parent module.

Parameters:

- **module** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - The parent module.
- **warn** (*bool**,**optional*) - if `True`, a warning is raised when no primers
are found. Defaults to `True`.
- **strict** (*bool**,**optional*) - if `True` (default), exceptions raised by
`make_tensordict_primer()` propagate. If `False`, failures are
caught per-submodule and a `UserWarning` lists the offending
module types; primers from sibling submodules are still returned.
Set to `False` from the collector dry-run path so that a single
conditionally-built primer (e.g.
[`ConsistentDropoutModule`](torchrl.modules.ConsistentDropoutModule.html#torchrl.modules.ConsistentDropoutModule) without
`input_shape`) doesn't drop primers from other submodules
(e.g. a sibling [`LSTMModule`](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule)).

Returns:

A TensorDictPrimer Transform.

Return type:

[TensorDictPrimer](torchrl.envs.transforms.TensorDictPrimer.html#torchrl.envs.transforms.TensorDictPrimer)

Example

```
>>> from torchrl.modules.utils import get_primers_from_module
>>> from torchrl.modules import GRUModule, MLP
>>> from tensordict.nn import TensorDictModule, TensorDictSequential
>>> # Define a GRU module
>>> gru_module = GRUModule(
... input_size=10,
... hidden_size=10,
... num_layers=1,
... in_keys=["input", "recurrent_state", "is_init"],
... out_keys=["features", ("next", "recurrent_state")],
... )
>>> # Define a head module
>>> head = TensorDictModule(
... MLP(
... in_features=10,
... out_features=10,
... num_cells=[],
... ),
... in_keys=["features"],
... out_keys=["output"],
... )
>>> # Create a sequential model
>>> model = TensorDictSequential(gru_module, head)
>>> # Retrieve primers from the model
>>> primers = get_primers_from_module(model)
>>> print(primers)
```

TensorDictPrimer(primers=Composite(
recurrent_state: UnboundedContinuous(

shape=torch.Size([1, 10]),
space=None,
device=cpu,
dtype=torch.float32,
domain=continuous), device=None, shape=torch.Size([])), default_value={'recurrent_state': 0.0}, random=None)