# torchrl.trainers.algorithms.configs.modules.TdMpc2MLPConfig

*class*torchrl.trainers.algorithms.configs.modules.TdMpc2MLPConfig(*_partial_: bool = False*, *in_features: int = '???'*, *out_features: int = '???'*, *depth: int | None = None*, *num_cells: Any = '???'*, *output_activation: Any = None*, *dropout: float = 0.0*, *device: Any = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_tdmpc2_mlp'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#TdMpc2MLPConfig)

A class to configure a TDMPC2 multilayer perceptron.

Example

```
>>> import torch
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import TdMpc2MLPConfig
>>> cfg = TdMpc2MLPConfig(
... in_features=6, out_features=4, depth=2, num_cells=8
... )
>>> net = instantiate(cfg)
>>> y = net(torch.randn(3, 6))
>>> assert y.shape == (3, 4)
```

See also

[`MLP`](torchrl.modules.MLP.html#torchrl.modules.MLP)