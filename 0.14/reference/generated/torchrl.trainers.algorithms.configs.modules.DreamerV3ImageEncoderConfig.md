# torchrl.trainers.algorithms.configs.modules.DreamerV3ImageEncoderConfig

*class*torchrl.trainers.algorithms.configs.modules.DreamerV3ImageEncoderConfig(*_partial_: bool = False*, *in_channels: int = 3*, *depth: int = 64*, *mults: list[int] = <factory>*, *kernel_size: int = 5*, *norm_eps: float = 0.0001*, *device: ~typing.Any = None*, *_target_: str = 'torchrl.modules.DreamerV3ImageEncoder'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#DreamerV3ImageEncoderConfig)

Hydra configuration for [`DreamerV3ImageEncoder`](torchrl.modules.DreamerV3ImageEncoder.html#torchrl.modules.DreamerV3ImageEncoder).

Example

```
>>> import torch
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import DreamerV3ImageEncoderConfig
>>> cfg = DreamerV3ImageEncoderConfig(depth=8, mults=[1, 2])
>>> net = instantiate(cfg)
>>> image = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8)
>>> assert net(image).shape == (4, 256)
```

See also

[`DreamerV3ImageEncoder`](torchrl.modules.DreamerV3ImageEncoder.html#torchrl.modules.DreamerV3ImageEncoder)