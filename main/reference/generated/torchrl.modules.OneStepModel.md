# OneStepModel

*class*torchrl.modules.OneStepModel(*network: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *action_dim: int*, ***, *low: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) = -1.0*, *high: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) = 1.0*)[[source]](../../_modules/torchrl/modules/models/flow.html#OneStepModel)

Tensor-only network for one-step flow distillation.

Parameters:

- **network** (*nn.Module*) - maps concatenated observation and Gaussian noise
directly to an action, without an output activation.
- **action_dim** (*int*) - number of action coordinates.

Keyword Arguments:

- **low** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or**Tensor**,**optional*) - lower action bound, broadcast over
actions. Must be strictly less than `high`. Defaults to -1.
- **high** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or**Tensor**,**optional*) - upper action bound, broadcast over
actions. Defaults to 1.

[`OneStepPolicy`](torchrl.modules.OneStepPolicy.html#torchrl.modules.OneStepPolicy) provides the TensorDict interface.
Outputs are clipped to `[low, high]`. Both models sample Gaussian noise
even under deterministic exploration; pass explicit noise for repeatability.

forward(*observation: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *noise: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, ***, *clamp: bool = True*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/flow.html#OneStepModel.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.