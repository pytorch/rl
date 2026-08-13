# sample_and_log_prob

torchrl.modules.distributions.utils.sample_and_log_prob(*distribution: [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)*, *sample_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | tuple[int, ...] = ()*, ***, *reparameterize: bool = False*) → tuple[Any, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/modules/distributions/utils.html#sample_and_log_prob)

Sample once and score the same draw atomically when supported.

If the distribution implements `sample_and_log_prob` or
`rsample_and_log_prob`, the matching method is used so that the score is
computed from the same latent draw as the sample. Otherwise, this function
falls back to separate sampling and scoring. Composite distributions are
handled component by component and respect
`composite_lp_aggregate()`.

Parameters:

- **distribution** (*Distribution*) - distribution to sample and score.
- **sample_shape** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**tuple**of**int**,**optional*) - leading sample
dimensions. Defaults to an empty shape.
- **reparameterize** (*bool**,**optional*) - if `True`, use reparameterized
sampling. Defaults to `False`.

Returns:

A tuple containing the sample and its log probability.