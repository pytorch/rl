# DreamerModelLoss

*class*torchrl.objectives.DreamerModelLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/dreamer.html#DreamerModelLoss)

Dreamer Model Loss.

Computes the loss of the dreamer world model. The loss is composed of the
kl divergence between the prior and posterior of the RSSM,
the reconstruction loss over the reconstructed observation and the reward
loss over the predicted reward.

Reference: [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

Parameters:

- **world_model** (*TensorDictModule*) - the world model.
- **lambda_kl** (`float`, optional) - the weight of the kl divergence loss. Default: 1.0.
- **lambda_reco** (`float`, optional) - the weight of the reconstruction loss. Default: 1.0.
- **lambda_reward** (`float`, optional) - the weight of the reward loss. Default: 1.0.
- **reco_loss** (*str**,**optional*) - the reconstruction loss. Default: "l2".
- **reward_loss** (*str**,**optional*) - the reward loss. Default: "l2".
- **free_nats** (*int**,**optional*) - the free nats. Default: 3.
- **delayed_clamp** (*bool**,**optional*) - if `True`, the KL clamping occurs after
averaging. If False (default), the kl divergence is clamped to the
free nats value first and then averaged.
- **global_average** (*bool**,**optional*) - if `True`, the losses will be averaged
over all dimensions. Otherwise, a sum will be performed over all
non-batch/time dimensions and an average over batch and time.
Default: False.

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/objectives/dreamer.html#DreamerModelLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.