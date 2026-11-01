# ExponentialQuadraticCost

*class*torchrl.objectives.ExponentialQuadraticCost(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/pilco.html#ExponentialQuadraticCost)

Computes the expected saturating cost for a Gaussian-distributed state.

This serves as a smooth, unimodal approximation of a 0-1 cost over a target area,
allowing for analytic gradient computation during policy search (e.g., PILCO).
Calculates E_{x_t}[c(x_t)] over N(m, s) as defined in Eq. (24) and (25) of
Deisenroth & Rasmussen (2011).

Parameters:

- **target** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - The target state vector. Defaults to the origin.
- **weights** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - The precision matrix mapping state dimensions
to the cost distance metric. Defaults to the identity matrix.
- **reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
'mean' | 'sum' | 'none'. Defaults to 'mean'.

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/pilco.html#ExponentialQuadraticCost.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.