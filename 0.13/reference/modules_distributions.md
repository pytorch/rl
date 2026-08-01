# Distribution Classes

Custom distribution classes for RL, extending PyTorch distributions.

| [`Delta`](generated/torchrl.modules.Delta.html#torchrl.modules.Delta)(param[, atol, rtol, batch_shape, ...]) | Delta distribution. |
| --- | --- |
| [`IndependentNormal`](generated/torchrl.modules.IndependentNormal.html#torchrl.modules.IndependentNormal)(loc, scale[, upscale, ...]) | Implements a Normal distribution with location scaling. |
| [`MaskedCategorical`](generated/torchrl.modules.MaskedCategorical.html#torchrl.modules.MaskedCategorical)([logits, probs, mask, ...]) | MaskedCategorical distribution. |
| [`NormalParamExtractor`](generated/torchrl.modules.NormalParamExtractor.html#torchrl.modules.NormalParamExtractor)([scale_mapping, scale_lb]) | A non-parametric nn.Module that splits its input into loc and scale parameters. |
| [`OneHotCategorical`](generated/torchrl.modules.OneHotCategorical.html#torchrl.modules.OneHotCategorical)([logits, probs, grad_method]) | One-hot categorical distribution. |
| [`ReparamGradientStrategy`](generated/torchrl.modules.ReparamGradientStrategy.html#torchrl.modules.ReparamGradientStrategy)(value[, names, ...]) | |
| [`TanhDelta`](generated/torchrl.modules.TanhDelta.html#torchrl.modules.TanhDelta)(param[, low, high, event_dims, ...]) | Implements a Tanh transformed_in Delta distribution. |
| [`TanhNormal`](generated/torchrl.modules.TanhNormal.html#torchrl.modules.TanhNormal)(loc, scale[, upscale, low, high, ...]) | Implements a TanhNormal distribution with location scaling. |
| [`TruncatedNormal`](generated/torchrl.modules.TruncatedNormal.html#torchrl.modules.TruncatedNormal)(loc, scale[, upscale, low, ...]) | Implements a Truncated Normal distribution with location scaling. |