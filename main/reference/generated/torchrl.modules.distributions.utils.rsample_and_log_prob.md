# rsample_and_log_prob

torchrl.modules.distributions.utils.rsample_and_log_prob(*distribution: [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)*, *sample_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | tuple[int, ...] = ()*) → tuple[Any, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/modules/distributions/utils.html#rsample_and_log_prob)

Reparameterize once and score the same draw atomically when supported.

Parameters:

- **distribution** (*Distribution*) - distribution to sample and score.
- **sample_shape** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**tuple**of**int**,**optional*) - leading sample
dimensions. Defaults to an empty shape.

Returns:

A tuple containing the reparameterized sample and its log probability.