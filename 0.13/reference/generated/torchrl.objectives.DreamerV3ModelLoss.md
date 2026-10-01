# DreamerV3ModelLoss

*class*torchrl.objectives.DreamerV3ModelLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3ModelLoss)

DreamerV3 World Model Loss.

Computes three terms:

1. **KL loss** -- balanced KL between prior and posterior categorical
distributions (see `categorical_kl_balanced()`).
2. **Reconstruction loss** -- symlog MSE between predicted and true
observations.
3. **Reward loss** -- two-hot cross-entropy or symlog MSE for the predicted
reward.

Optionally a **continue loss** (binary cross-entropy) can be enabled
when the world model outputs a continue predictor.

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **world_model** (*TensorDictModule*) - World model that takes a tensordict with
observations/actions and writes predicted observations, rewards, and
RSSM prior/posterior logits.
- **lambda_kl** (*float**,**optional*) - KL loss weight. Default: 1.0.
- **lambda_reco** (*float**,**optional*) - Reconstruction loss weight. Default: 1.0.
- **lambda_reward** (*float**,**optional*) - Reward prediction loss weight. Default: 1.0.
- **lambda_continue** (*float**,**optional*) - Continue prediction loss weight.
Default: 0.0 (disabled).
- **kl_alpha** (*float**,**optional*) - KL balancing factor (alpha in the paper).
Default: 0.8.
- **free_bits** (*float**,**optional*) - Minimum KL per categorical in nats.
Default: 1.0.
- **reco_loss** (*str**,**optional*) - Reconstruction loss type (`"l2"` or
`"l1"`). Default: `"l2"`.
- **reward_two_hot** (*bool**,**optional*) - If `True`, the reward head is
expected to output **logits over** `num_reward_bins` and the loss
is two-hot cross-entropy. If `False`, the reward head outputs a
**scalar** prediction and the loss is symlog MSE. Default: `True`.
- **num_reward_bins** (*int**,**optional*) - Number of bins for the two-hot reward
distribution. Default: 255.
- **global_average** (*bool**,**optional*) - If `True`, averages losses over all
dimensions. Otherwise sums over non-batch/time dims first. Default:
`False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torch import nn
>>> from torchrl.objectives import DreamerV3ModelLoss
>>> class StubWorldModel(nn.Module):
... def __init__(self):
... super().__init__()
... self.head = nn.LazyLinear(4 * 4)
... self.reward_head = nn.LazyLinear(16)
... self.decoder = nn.LazyLinear(3 * 8 * 8)
... def forward(self, td):
... B, T = td.shape
... x = torch.cat([td["state"], td["belief"]], dim=-1)
... logits = self.head(x).view(B, T, 4, 4)
... reco = self.decoder(x).view(B, T, 3, 8, 8)
... reward = self.reward_head(x)
... td.set(("next", "prior_logits"), logits)
... td.set(("next", "posterior_logits"), logits)
... td.set(("next", "reco_pixels"), reco)
... td.set(("next", "reward"), reward)
... return td
>>> wm = StubWorldModel()
>>> td = TensorDict({
... "state": torch.zeros(2, 3, 16),
... "belief": torch.zeros(2, 3, 8),
... "action": torch.randn(2, 3, 2),
... "next": {
... "pixels": torch.rand(2, 3, 3, 8, 8),
... "reward": torch.randn(2, 3, 1),
... "done": torch.zeros(2, 3, dtype=torch.bool),
... },
... }, [2, 3])
>>> with torch.no_grad():
... wm(td.clone())
TensorDict(...)
>>> loss = DreamerV3ModelLoss(wm, num_reward_bins=16, free_bits=0.1)
>>> loss_td, _ = loss(td)
>>> sorted(loss_td.keys())
['loss_model_kl', 'loss_model_reco', 'loss_model_reward']
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)*) → tuple[[TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)][[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3ModelLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.