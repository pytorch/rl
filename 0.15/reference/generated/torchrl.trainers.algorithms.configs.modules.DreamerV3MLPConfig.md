# torchrl.trainers.algorithms.configs.modules.DreamerV3MLPConfig

*class*torchrl.trainers.algorithms.configs.modules.DreamerV3MLPConfig(*_partial_: bool = False*, *in_features: int = '???'*, *out_features: int | None = '???'*, *depth: int = 3*, *num_cells: int = 1024*, *outscale: float = 1.0*, *norm_eps: float = 0.0001*, *device: Any = None*, *_target_: str = 'torchrl.modules.DreamerV3MLP'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#DreamerV3MLPConfig)

A class to configure a DreamerV3 multilayer perceptron.

Example

```
>>> import torch
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import DreamerV3MLPConfig
>>> cfg = DreamerV3MLPConfig(
... in_features=6, out_features=4, depth=2, num_cells=8
... )
>>> net = instantiate(cfg)
>>> y = net(torch.randn(3, 2), torch.randn(3, 4))
>>> assert y.shape == (3, 4)
```

See also

[`DreamerV3MLP`](torchrl.modules.DreamerV3MLP.html#torchrl.modules.DreamerV3MLP)