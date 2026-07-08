# TensorDictPixelsBackend

*class*torchrl.render.backends.TensorDictPixelsBackend[[source]](../../_modules/torchrl/render/backends/pixels.html#TensorDictPixelsBackend)

Captures frames from TensorDict pixel entries.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.render import RenderConfig
>>> from torchrl.render.backends import TensorDictPixelsBackend
>>> td = TensorDict({"pixels": torch.zeros(2, 2, 3, dtype=torch.uint8)}, [])
>>> cfg = RenderConfig("policy.pt", "p:make", "e:make", max_steps=1)
>>> TensorDictPixelsBackend().capture(None, td, cfg, step=0, trajectory_index=0).step
0
```