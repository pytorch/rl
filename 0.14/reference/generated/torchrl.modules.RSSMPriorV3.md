# RSSMPriorV3

*class*torchrl.modules.RSSMPriorV3(*action_spec=None*, *hidden_dim: int = 512*, *rnn_hidden_dim: int = 512*, *num_categoricals: int = 32*, *num_classes: int = 32*, *action_dim: int | None = None*, *device=None*, ***, *action_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | tuple[int, ...] | None = None*, *recurrent_model: Literal['gru', 'block_gru'] = 'gru'*, *num_blocks: int = 8*, *num_layers: int = 1*, *prior_num_layers: int = 2*, *norm_eps: float = 0.0001*, *unimix: float = 0.0*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMPriorV3)

DreamerV3 prior network with discrete categorical latent state.

See [DreamerV3 in a nutshell](../dreamer_v3.html) for the prior's
role in observation-conditioned filtering and latent imagination.

Implements the sequence model and dynamics predictor from DreamerV3.
The GRU updates the deterministic hidden state:

```
h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
```

Then the prior predicts a distribution over the stochastic latent:

```
z_hat_t ~ Cat(MLP(h_t))
```

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - Action spec. Used only to read
`action_spec.shape`; mutually exclusive with `action_shape`.
- **action_shape** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**tuple**of**int**,**optional*) - Action tensor
shape. Mutually exclusive with `action_spec`.
- **hidden_dim** (*int**,**optional*) - Hidden dimension of the linear projector.
Defaults to 512.
- **rnn_hidden_dim** (*int**,**optional*) - GRU hidden state dimension (belief size).
Defaults to 512.
- **num_categoricals** (*int**,**optional*) - Number of categorical variables in the
discrete latent. Defaults to 32.
- **num_classes** (*int**,**optional*) - Number of classes per categorical variable.
Defaults to 32.
- **action_dim** (*int**,**optional*) - Action dimension. If provided (along with
`num_categoricals * num_classes`), uses explicit `nn.Linear`
instead of `nn.LazyLinear`. Defaults to None.
- **recurrent_model** (*"gru"**or**"block_gru"**,**optional*) - Recurrent core.
`"gru"` preserves the historical TorchRL implementation while
`"block_gru"` selects the grouped DreamerV3 core. Defaults to
`"gru"`.
- **num_blocks** (*int**,**optional*) - Number of groups in the block GRU.
Defaults to 8.
- **num_layers** (*int**,**optional*) - Number of block-linear dynamics layers.
Defaults to 1.
- **prior_num_layers** (*int**,**optional*) - Number of prior predictor layers in
block-GRU mode. Defaults to 2.
- **norm_eps** (*float**,**optional*) - RMS normalization epsilon. Defaults to
`1e-4`.
- **unimix** (*float**,**optional*) - Fraction of uniform probability mixed into
categorical samples. Defaults to `0.0` for compatibility.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device. Defaults to None.

Examples

```
>>> import torch
>>> from torchrl.modules.models.model_based import RSSMPriorV3
>>> prior = RSSMPriorV3(
... action_shape=torch.Size([2]),
... hidden_dim=16,
... rnn_hidden_dim=8,
... num_categoricals=4,
... num_classes=4,
... action_dim=2,
... )
>>> state = torch.zeros(3, 16)
>>> belief = torch.zeros(3, 8)
>>> action = torch.randn(3, 2)
>>> logits, next_state, next_belief = prior(state, belief, action)
>>> logits.shape, next_state.shape, next_belief.shape
(torch.Size([3, 4, 4]), torch.Size([3, 16]), torch.Size([3, 8]))
```

forward(*state: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *belief: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *action: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *_uniform: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMPriorV3.forward)

Compute prior distribution and update GRU belief.

Parameters:

- **state** - Previous stochastic state, shape `[..., num_categoricals * num_classes]`.
- **belief** - Previous GRU hidden state, shape `[..., rnn_hidden_dim]`.
- **action** - Current action, shape `[..., action_dim]`.
- **_uniform** - Optional pre-sampled uniforms used by the scan backend.

Returns:

Raw logits, shape

`[..., num_categoricals, num_classes]`.

state (torch.Tensor): Sampled state (straight-through), shape

`[..., num_categoricals * num_classes]`.

belief (torch.Tensor): Updated GRU hidden state, shape

`[..., rnn_hidden_dim]`.

Return type:

prior_logits ([torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor))