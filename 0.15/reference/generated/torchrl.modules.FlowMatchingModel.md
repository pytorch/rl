# FlowMatchingModel

*class*torchrl.modules.FlowMatchingModel(*velocity_network: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *action_dim: int*, *num_steps: int = 10*, ***, *low: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) = -1.0*, *high: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) = 1.0*)[[source]](../../_modules/torchrl/modules/models/flow.html#FlowMatchingModel)

Tensor-only Euler sampler for bounded continuous actions.

Parameters:

- **velocity_network** (*nn.Module*) - maps concatenated observation, action and
scalar time to an action-sized velocity.
- **action_dim** (*int*) - number of action coordinates.
- **num_steps** (*int**,**optional*) - Euler integration steps. Defaults to 10.

Keyword Arguments:

- **low** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or**Tensor**,**optional*) - lower action bound, broadcast over
actions. Must be strictly less than `high`. Defaults to -1.
- **high** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or**Tensor**,**optional*) - upper action bound, broadcast over
actions. Defaults to 1.

Outputs are clipped to `[low, high]` after integration.
[`FlowMatchingPolicy`](torchrl.modules.FlowMatchingPolicy.html#torchrl.modules.FlowMatchingPolicy) provides the TensorDict interface.
On PyTorch 2.14+, eager execution (including autograd) and compiled
inference use scan. Older versions and compiled calls with gradients enabled
use an explicit Euler loop: PyTorch 2.14 Inductor can produce incorrect
gradients through a peeled scan followed by clipping.

forward(*observation: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *noise: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/flow.html#FlowMatchingModel.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.