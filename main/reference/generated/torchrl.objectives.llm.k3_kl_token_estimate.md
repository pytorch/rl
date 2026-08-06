# k3_kl_token_estimate

*class*torchrl.objectives.llm.k3_kl_token_estimate(*target_log_prob: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *log_prob: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*)[[source]](../../_modules/torchrl/objectives/llm/distillation.html#k3_kl_token_estimate)

Per-token k3 estimate of the KL divergence to a target distribution.

Uses the k3 estimator `exp(d) - 1 - d` with
`d = target_log_prob - log_prob`, the same approximation used by
[`GRPOLoss`](torchrl.objectives.llm.GRPOLoss.html#torchrl.objectives.llm.GRPOLoss) and
[`SFTLoss`](torchrl.objectives.llm.SFTLoss.html#torchrl.objectives.llm.SFTLoss) for their KL regularizers, but
returned per token instead of reduced, so callers can apply their own
masking and reduction.

The estimator is direction-neutral: it estimates
`KL(sampling distribution || target distribution)`, and the direction
of the resulting KL is determined by which distribution's
log-probabilities are passed as which argument. The estimate is unbiased
only when the scored tokens are actual samples from the distribution
behind `log_prob`; on tokens drawn from any other source, the (masked)
sum or mean of the result is still a nonnegative divergence between the
two distributions, but it no longer estimates either KL divergence.

Parameters:

- **target_log_prob** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - per-token log-probabilities of the
target (non-sampling) distribution, evaluated on the same tokens
as `log_prob`.
- **log_prob** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - per-token log-probabilities of the
distribution the tokens were sampled from.

Returns:

A tensor of nonnegative per-token KL contributions with the same
shape as the inputs.

References

- John Schulman, 2020. ["Approximating KL Divergence"](http://joschu.net/blog/kl-approx.html)

Examples

```
>>> import torch
>>> from torchrl.objectives.llm import k3_kl_token_estimate
>>> log_prob = torch.full((4,), -1.0)
>>> k3_kl_token_estimate(log_prob, log_prob)
tensor([0., 0., 0., 0.])
```