# ACTLoss

*class*torchrl.objectives.ACTLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/act.html#ACTLoss)

Loss module for Action Chunking with Transformers (ACT).

Implements the training objective from *Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware* ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)), pairing an L1
chunk-reconstruction term with a KL-divergence penalty on the CVAE
latent:

\[\mathcal{L} = \underbrace{\|a_{\text{pred}} -
a_{\text{chunk}}\|_1}_{\text{reconstruction}}
+ \beta \cdot
\underbrace{D_{\mathrm{KL}}\!\left(q(z|o,a)\,\|\,
\mathcal{N}(0,I)\right)}_{\text{KL}}\]

The input tensordict stores expert chunks under
`("vla_action", "chunk")` by default. The `actor_network` itself must
read `"observation"` and `"action_chunk"` and write
`"action_pred"`, `"mu"`, and `"log_var"`. This matches the contract
of `ACTModel` when wrapped with a
[`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule).

Three values are returned in the output TensorDict:

- `"loss_act"` -- the full (differentiable) training loss.
- `"loss_reconstruction"` -- detached L1 reconstruction term (for
logging).
- `"loss_kl"` -- detached KL term (for logging).

Parameters:

**actor_network** (*TensorDictModule*) - ACT policy. Must expose `in_keys`
containing `"observation"` and `"action_chunk"` and write
`"action_pred"`, `"mu"`, `"log_var"`.

Keyword Arguments:

- **kl_weight** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - β -- weight on the KL divergence term.
Defaults to `10.0` (as in the original paper).
- **reduction** (*str**,**optional*) - `"none"` | `"mean"` | `"sum"`.
Defaults to `"mean"`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules.models import ACTModel
>>> from torchrl.objectives import ACTLoss
>>> model = ACTModel(obs_dim=14, action_dim=7, chunk_size=10)
>>> actor = TensorDictModule(
... model,
... in_keys=["observation", "action_chunk"],
... out_keys=["action_pred", "mu", "log_var"],
... )
>>> loss_fn = ACTLoss(actor, kl_weight=10.0)
>>> td = TensorDict(
... {
... "observation": torch.randn(4, 14),
... ("vla_action", "chunk"): torch.randn(4, 10, 7),
... },
... batch_size=[4],
... )
>>> loss_td = loss_fn(td)
>>> loss_td["loss_act"].backward()
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)[[source]](../../_modules/torchrl/objectives/act.html#ACTLoss.forward)

Compute the ACT loss.

Parameters:

**tensordict** (*TensorDictBase*) - Input data containing
`"observation"` and `("vla_action", "chunk")` by default.

Returns:

TensorDict with keys `"loss_act"`, `"loss_reconstruction"`,
and `"loss_kl"`.