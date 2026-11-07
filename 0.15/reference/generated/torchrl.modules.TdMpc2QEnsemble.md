# TdMpc2QEnsemble

*class*torchrl.modules.TdMpc2QEnsemble(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/models/tdmpc2.html#TdMpc2QEnsemble)

Vectorized TD-MPC2 ensemble of distributional Q-functions.

Each Q-function receives the concatenation of a latent state and an action
and returns logits for a scalar categorical representation. The ensemble
dimension is exposed immediately before the category dimension.

`forward()` returns the logits of every Q-function. `reduce()`
follows TD-MPC2's value-estimation rule by selecting two Q-functions at
random, decoding their logits, and taking either their minimum or average.
Online, detached, and target parameter sources are available for both
operations.

Parameters:

- **q_networks** - Sequence of identically shaped single-network Q-functions.
- **num_bins** - Number of categorical bins. Must be greater than 1.
- **vmin** - Minimum value of the symlog-space categorical support.
- **vmax** - Maximum value of the symlog-space categorical support.
- **in_keys** - Two TensorDict keys for the latent state and action.
Defaults to `["latent", "action"]`.
- **out_keys** - One TensorDict key for the ensemble logits. Defaults to
`["q_logits"]`.
- **q_value_key** - TensorDict key written by `reduce()`. Defaults to
`"q_value"`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import MLP
>>> from torchrl.modules.models.tdmpc2 import TdMpc2QEnsemble
>>> q_networks = [
... MLP(in_features=6, out_features=5, depth=2, num_cells=8)
... for _ in range(5)
... ]
>>> q_ensemble = TdMpc2QEnsemble(
... q_networks, num_bins=5, vmin=-10.0, vmax=10.0
... )
>>> data = TensorDict(
... {"latent": torch.randn(4, 4), "action": torch.randn(4, 2)},
... batch_size=[4],
... )
>>> data = q_ensemble(data)
>>> data["q_logits"].shape
torch.Size([4, 5, 5])
>>> q_ensemble.reduce(data, reduction="min")["q_value"].shape
torch.Size([4, 1])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *source: Literal['online', 'detached', 'target'] = 'online'*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/models/tdmpc2.html#TdMpc2QEnsemble.forward)

Write the logits of all Q-functions to the input TensorDict.

Parameters:

- **tensordict** - TensorDict containing the latent state and action.
- **source** - Parameter source used for the Q-functions. Defaults to
`"online"`.

reduce(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *reduction: Literal['min', 'avg']*, *source: Literal['online', 'detached', 'target'] = 'online'*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/models/tdmpc2.html#TdMpc2QEnsemble.reduce)

Write a TD-MPC2 two-Q value reduction to the input TensorDict.

Parameters:

- **tensordict** - TensorDict containing the latent state and action.
- **reduction** - Either `"min"` or `"avg"` for the two sampled
Q-functions.
- **source** - Parameter source used for the Q-functions. Defaults to
`"online"`.

soft_update_target(*tau: float*) → None[[source]](../../_modules/torchrl/modules/models/tdmpc2.html#TdMpc2QEnsemble.soft_update_target)

Update target Q-function buffers by Polyak averaging.

Parameters:

**tau** - Interpolation factor in `[0, 1]`. A value of `0` keeps
the target unchanged and a value of `1` copies the online
parameters.

train(*mode: bool = True*)[[source]](../../_modules/torchrl/modules/models/tdmpc2.html#TdMpc2QEnsemble.train)

Set training mode while keeping the target Q-functions in eval mode.