# RSSMPosteriorV3

*class*torchrl.modules.RSSMPosteriorV3(*hidden_dim: int = 512*, *num_categoricals: int = 32*, *num_classes: int = 32*, *rnn_hidden_dim: int | None = None*, *obs_embed_dim: int | None = None*, *device=None*, ***, *use_rms_norm: bool = False*, *num_layers: int = 1*, *norm_eps: float = 0.0001*, *unimix: float = 0.0*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMPosteriorV3)

DreamerV3 posterior (representation model) with discrete categorical latent.

See [DreamerV3 in a nutshell](../dreamer_v3.html) for the
relationship between the posterior, prior, stochastic state, and belief.

Given the deterministic hidden state `h_t` and an observation embedding
`e_t`, produces the posterior distribution over the stochastic latent:

```
z_t ~ Cat(MLP([h_t, e_t]))
```

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **hidden_dim** (*int**,**optional*) - Hidden dimension of the projector MLP.
Defaults to 512.
- **num_categoricals** (*int**,**optional*) - Number of categorical variables.
Defaults to 32.
- **num_classes** (*int**,**optional*) - Number of classes per categorical variable.
Defaults to 32.
- **rnn_hidden_dim** (*int**,**optional*) - Belief dimension. If provided along with
`obs_embed_dim`, uses explicit `nn.Linear`. Defaults to None.
- **obs_embed_dim** (*int**,**optional*) - Observation embedding dimension. If provided
along with `rnn_hidden_dim`, uses explicit `nn.Linear`. Defaults to None.
- **use_rms_norm** (*bool**,**optional*) - Build the observation predictor from
RMS-normalized DreamerV3 layers. Defaults to `False` for checkpoint
compatibility.
- **num_layers** (*int**,**optional*) - Number of observation predictor layers when
`use_rms_norm=True`. Defaults to 1.
- **norm_eps** (*float**,**optional*) - RMS normalization epsilon. Defaults to
`1e-4`.
- **unimix** (*float**,**optional*) - Fraction of uniform probability mixed into
categorical samples. Defaults to `0.0` for compatibility.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device. Defaults to None.

Examples

```
>>> import torch
>>> from torchrl.modules.models.model_based import RSSMPosteriorV3
>>> posterior = RSSMPosteriorV3(
... hidden_dim=16,
... num_categoricals=4,
... num_classes=4,
... rnn_hidden_dim=8,
... obs_embed_dim=12,
... )
>>> belief = torch.randn(3, 8)
>>> obs_embed = torch.randn(3, 12)
>>> logits, state = posterior(belief, obs_embed)
>>> logits.shape, state.shape
(torch.Size([3, 4, 4]), torch.Size([3, 16]))
```

forward(*belief: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *obs_embedding: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *_uniform: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMPosteriorV3.forward)

Compute posterior distribution given belief and observation embedding.

Parameters:

- **belief** - Deterministic GRU hidden state from prior, shape
`[..., rnn_hidden_dim]`.
- **obs_embedding** - Encoded observation, shape `[..., obs_embed_dim]`.
- **_uniform** - Optional pre-sampled uniforms used by the scan backend.

Returns:

Raw logits, shape

`[..., num_categoricals, num_classes]`.

state (torch.Tensor): Sampled state (straight-through), shape

`[..., num_categoricals * num_classes]`.

Return type:

posterior_logits ([torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor))