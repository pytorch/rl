# DreamerV3OptimizerConfig

*class*torchrl.trainers.algorithms.configs.DreamerV3OptimizerConfig(*lr: float = 4e-05*, *agc: float = 0.3*, *parameter_norm_min: float = 0.001*, *beta1: float = 0.9*, *beta2: float = 0.999*, *eps: float = 1e-20*, *warmup_steps: int = 1000*, *_target_: str = 'torchrl.trainers.algorithms.DreamerV3Optimizer'*, *_partial_: bool = True*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/utils.html#DreamerV3OptimizerConfig)

Hydra configuration for [`DreamerV3Optimizer`](torchrl.trainers.algorithms.DreamerV3Optimizer.html#torchrl.trainers.algorithms.DreamerV3Optimizer).

Instantiation returns a partial optimizer constructor; supply its parameters
after constructing the learner modules.

Examples

```
>>> import torch
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import DreamerV3OptimizerConfig
>>> make_optimizer = instantiate(DreamerV3OptimizerConfig(warmup_steps=0))
>>> parameter = torch.nn.Parameter(torch.ones(2))
>>> optimizer = make_optimizer([parameter])
>>> parameter.sum().backward()
>>> optimizer.step()
>>> bool((parameter < 1).all())
True
```