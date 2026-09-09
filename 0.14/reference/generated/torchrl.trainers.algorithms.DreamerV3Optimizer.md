# DreamerV3Optimizer

*class*torchrl.trainers.algorithms.DreamerV3Optimizer(*parameters: Iterable[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | Iterable[dict[str, Any]]*, ***, *lr: float = 4e-05*, *agc: float = 0.3*, *parameter_norm_min: float = 0.001*, *beta1: float = 0.9*, *beta2: float = 0.999*, *eps: float = 1e-20*, *warmup_steps: int = 1000*)[[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3Optimizer)

DreamerV3 adaptive gradient clipping, RMS scaling and momentum.

Clips each parameter's gradient by its parameter norm, normalizes it by a
bias-corrected moving RMS, and applies bias-corrected momentum. A linear
learning-rate warm-up starts at zero on the first step when enabled.
Moment estimates are accumulated in float32. Parameters with no gradient
are skipped; an update with no parameter gradients raises `RuntimeError`.

Reference: Hafner et al., "Mastering Diverse Domains through World Models"
(2023), [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).

See also [`DreamerV3OptimizerConfig`](torchrl.trainers.algorithms.configs.DreamerV3OptimizerConfig.html#torchrl.trainers.algorithms.configs.DreamerV3OptimizerConfig).

Parameters:

**parameters** (*iterable**of**Tensor**or**dict*) - Parameters to optimize, or
parameter-group dictionaries. Group options override the defaults
below; each group maintains its own update counter.

Keyword Arguments:

- **lr** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Learning rate after warm-up. Default: `4e-5`.
- **agc** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Maximum gradient norm as a fraction of the
clamped parameter norm. Zero disables clipping. Default: `0.3`.
- **parameter_norm_min** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Lower bound on parameter norms
used for clipping. Default: `1e-3`.
- **beta1** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Decay of normalized-gradient momentum.
Default: `0.9`.
- **beta2** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Decay of the squared-gradient average.
Default: `0.999`.
- **eps** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Added to the RMS denominator. Default: `1e-20`.
- **warmup_steps** (*int**,**optional*) - Number of updates before the full learning
rate is reached. Zero disables warm-up. Default: `1000`.

Examples

```
>>> import torch
>>> from torchrl.trainers.algorithms import DreamerV3Optimizer
>>> parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
>>> optimizer = DreamerV3Optimizer([parameter], lr=0.01, warmup_steps=0)
>>> parameter.square().sum().backward()
>>> optimizer.step()
>>> bool((parameter.abs() < 1).all())
True
>>> optimizer.zero_grad(set_to_none=False)
>>> checkpoint = optimizer.state_dict()
>>> optimizer.load_state_dict(checkpoint)
```

step(*closure: Callable[[], [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None[[source]](../../_modules/torchrl/trainers/algorithms/dreamer_v3.html#DreamerV3Optimizer.step)

Update parameters with gradients and return the optional closure loss.

Parameters:

**closure** (*callable**,**optional*) - Re-evaluates the model, computes
gradients, and returns its loss. Default: `None`.

Returns:

The closure's loss, or `None` when no closure is supplied.