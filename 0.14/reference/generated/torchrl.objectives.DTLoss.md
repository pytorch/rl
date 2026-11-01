# DTLoss

*class*torchrl.objectives.DTLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/decision_transformer.html#DTLoss)

TorchRL implementation of the Online Decision Transformer loss.

Presented in "Decision Transformer: Reinforcement Learning via Sequence Modeling" <https://arxiv.org/abs/2106.01345>

Parameters:

**actor_network** (*ProbabilisticTensorDictSequential*) - stochastic actor

Keyword Arguments:

- **loss_function** (*str*) - loss function to use. Defaults to `"l2"`.
- **reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
`"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied,
`"mean"`: the sum of the output will be divided by the number of
elements in the output, `"sum"`: the output will be summed. Default: `"mean"`.

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/decision_transformer.html#DTLoss.forward)

Compute the loss for the Online Decision Transformer.