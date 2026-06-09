# GAILLoss

*class*torchrl.objectives.GAILLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/gail.html#GAILLoss)

TorchRL implementation of the Generative Adversarial Imitation Learning (GAIL) loss.

Presented in "Generative Adversarial Imitation Learning" <https://arxiv.org/pdf/1606.03476>

Parameters:

**discriminator_network** (*TensorDictModule*) - stochastic actor

Keyword Arguments:

- **use_grad_penalty** (*bool**,**optional*) - Whether to use gradient penalty. Default: `False`.
- **gp_lambda** (`float`, optional) - Gradient penalty lambda. Default: `10`.
- **reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
`"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied,
`"mean"`: the sum of the output will be divided by the number of
elements in the output, `"sum"`: the output will be summed. Default: `"mean"`.

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/gail.html#GAILLoss.forward)

The forward method.

Computes the discriminator loss and gradient penalty if use_grad_penalty is set to True. If use_grad_penalty is set to True, the detached gradient penalty loss is also returned for logging purposes.
To see what keys are expected in the input tensordict and what keys are expected as output, check the
class's "in_keys" and "out_keys" attributes.