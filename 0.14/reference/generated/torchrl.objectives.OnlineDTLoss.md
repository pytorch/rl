# OnlineDTLoss

*class*torchrl.objectives.OnlineDTLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/decision_transformer.html#OnlineDTLoss)

TorchRL implementation of the Online Decision Transformer loss.

Presented in "Online Decision Transformer" <https://arxiv.org/abs/2202.05607>

Parameters:

**actor_network** (*ProbabilisticTensorDictSequential*) - stochastic actor

Keyword Arguments:

- **alpha_init** (`float`, optional) - initial entropy multiplier.
Default is 1.0.
- **min_alpha** (`float`, optional) - min value of alpha.
Default is None (no minimum value).
- **max_alpha** (`float`, optional) - max value of alpha.
Default is None (no maximum value).
- **fixed_alpha** (*bool**,**optional*) - if `True`, alpha will be fixed to its
initial value. Otherwise, alpha will be optimized to
match the 'target_entropy' value.
Default is `False`.
- **target_entropy** (`float` or str, optional) - Target entropy for the
stochastic policy. Default is "auto", where target entropy is
computed as `-prod(n_actions)`.
- **samples_mc_entropy** (*int*) - number of samples to estimate the entropy
- **reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
`"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied,
`"mean"`: the sum of the output will be divided by the number of
elements in the output, `"sum"`: the output will be summed. Default: `"mean"`.

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/decision_transformer.html#OnlineDTLoss.forward)

Compute the loss for the Online Decision Transformer.