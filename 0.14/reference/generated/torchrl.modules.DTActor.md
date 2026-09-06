# DTActor

*class*torchrl.modules.DTActor(*state_dim: int*, *action_dim: int*, *transformer_config: dict | [DTConfig](torchrl.modules.DecisionTransformer.html#torchrl.modules.DecisionTransformer.DTConfig) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/models.html#DTActor)

Decision Transformer Actor class.

Actor class for the Decision Transformer to output deterministic action as
presented in "Decision Transformer" <https://arxiv.org/abs/2202.05607.pdf>.
Returns the deterministic actions.

Parameters:

- **state_dim** (*int*) - state dimension.
- **action_dim** (*int*) - action dimension.
- **transformer_config** (Dict or [`DecisionTransformer.DTConfig`](torchrl.modules.DecisionTransformer.html#torchrl.modules.DecisionTransformer.DTConfig), optional) - config for the GPT2 transformer.
Defaults to `default_config()`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to use. Defaults to None.

Examples

```
>>> model = DTActor(state_dim=4, action_dim=2,
... transformer_config=DTActor.default_config())
>>> observation = torch.randn(32, 10, 4)
>>> action = torch.randn(32, 10, 2)
>>> return_to_go = torch.randn(32, 10, 1)
>>> output = model(observation, action, return_to_go)
>>> output.shape
torch.Size([32, 10, 2])
```

*classmethod*default_config()[[source]](../../_modules/torchrl/modules/models/models.html#DTActor.default_config)

Default configuration for `DTActor`.

forward(*observation: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *action: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *return_to_go: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/models.html#DTActor.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.