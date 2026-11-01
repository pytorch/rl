# DreamerValueLoss

*class*torchrl.objectives.DreamerValueLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/dreamer.html#DreamerValueLoss)

Dreamer Value Loss.

Computes the loss of the dreamer value model. The value loss is computed
between the predicted value and the lambda target.

Reference: [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

Parameters:

- **value_model** (*TensorDictModule*) - the value model.
- **value_loss** (*str**,**optional*) - the loss to use for the value loss.
Default: `"l2"`.
- **discount_loss** (*bool**,**optional*) - if `True`, the loss is discounted with a
gamma discount factor. Default: False.
- **gamma** (`float`, optional) - the gamma discount factor. Default: `0.99`.

default_keys

alias of `_AcceptedKeys`

forward(*fake_data*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/objectives/dreamer.html#DreamerValueLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.