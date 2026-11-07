# DecisionTransformer

*class*torchrl.modules.DecisionTransformer(*state_dim*, *action_dim*, *config: dict | DTConfig = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*)[[source]](../../_modules/torchrl/modules/models/decision_transformer.html#DecisionTransformer)

Online Decision Transformer.

Desdescribed in [https://arxiv.org/abs/2202.05607](https://arxiv.org/abs/2202.05607) .

The transformer utilizes a default config to create the GPT2 model if the user does not provide a specific config.
default_config = {
... "n_embd": 256,
... "n_layer": 4,
... "n_head": 4,
... "n_inner": 1024,
... "activation": "relu",
... "n_positions": 1024,
... "resid_pdrop": 0.1,
... "attn_pdrop": 0.1,
}

Parameters:

- **state_dim** (*int*) - dimension of the state space
- **action_dim** (*int*) - dimension of the action space
- **config** (`DTConfig` or dict, optional) - transformer architecture configuration,
used to create the GPT2Config from transformers.
Defaults to `default_config`.

Example

```
>>> config = DecisionTransformer.default_config()
>>> config.n_embd = 128
>>> print(config)
DTConfig(n_embd: 128, n_layer: 4, n_head: 4, n_inner: 1024, activation: relu, n_positions: 1024, resid_pdrop: 0.1, attn_pdrop: 0.1)
>>> # alternatively
>>> config = DecisionTransformer.DTConfig(n_embd=128)
>>> model = DecisionTransformer(state_dim=4, action_dim=2, config=config)
>>> batch_size = [3, 32]
>>> length = 10
>>> observation = torch.randn(*batch_size, length, 4)
>>> action = torch.randn(*batch_size, length, 2)
>>> return_to_go = torch.randn(*batch_size, length, 1)
>>> output = model(observation, action, return_to_go)
>>> output.shape
torch.Size([3, 32, 10, 128])
```

*class*DTConfig(*n_embd: Any = 256*, *n_layer: Any = 4*, *n_head: Any = 4*, *n_inner: Any = 1024*, *activation: Any = 'relu'*, *n_positions: Any = 1024*, *resid_pdrop: Any = 0.1*, *attn_pdrop: Any = 0.1*)[[source]](../../_modules/torchrl/modules/models/decision_transformer.html#DecisionTransformer.DTConfig)

Default configuration for DecisionTransformer.

forward(*observation: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *action: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *return_to_go: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*)[[source]](../../_modules/torchrl/modules/models/decision_transformer.html#DecisionTransformer.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.