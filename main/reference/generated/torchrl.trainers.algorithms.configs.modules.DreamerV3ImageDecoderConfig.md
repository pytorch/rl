# torchrl.trainers.algorithms.configs.modules.DreamerV3ImageDecoderConfig

*class*torchrl.trainers.algorithms.configs.modules.DreamerV3ImageDecoderConfig(*_partial_: bool = False*, *in_features: int = '???'*, *image_shape: list[int] = <factory>*, *depth: int = 64*, *mults: list[int] = <factory>*, *kernel_size: int = 5*, *num_blocks: int = 8*, *norm_eps: float = 0.0001*, *device: ~typing.Any = None*, *_target_: str = 'torchrl.modules.DreamerV3ImageDecoder'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#DreamerV3ImageDecoderConfig)

Hydra configuration for [`DreamerV3ImageDecoder`](torchrl.modules.DreamerV3ImageDecoder.html#torchrl.modules.DreamerV3ImageDecoder).

Example

```
>>> import torch
>>> from hydra.utils import instantiate
>>> from torchrl.trainers.algorithms.configs import DreamerV3ImageDecoderConfig
>>> cfg = DreamerV3ImageDecoderConfig(
... in_features=12, image_shape=[3, 16, 16], depth=8, mults=[1, 2], num_blocks=2
... )
>>> net = instantiate(cfg)
>>> assert net(torch.randn(4, 12)).shape == (4, 3, 16, 16)
```

See also

[`DreamerV3ImageDecoder`](torchrl.modules.DreamerV3ImageDecoder.html#torchrl.modules.DreamerV3ImageDecoder)