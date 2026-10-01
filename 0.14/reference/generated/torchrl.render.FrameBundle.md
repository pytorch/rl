# FrameBundle

*class*torchrl.render.FrameBundle(*frames: dict[str, ~typing.Any], step: int, trajectory_index: int, timestamp: float | None = None, metadata: dict[str, ~typing.Any] = <factory>*)[[source]](../../_modules/torchrl/render/config.html#FrameBundle)

One rendered step containing one or more named camera frames.

Parameters:

- **frames** - Mapping from camera name to `uint8` RGB image arrays.
- **step** - Step index within the trajectory.
- **trajectory_index** - Trajectory index.
- **timestamp** - Optional external timestamp.
- **metadata** - Backend-specific frame metadata.

Examples

```
>>> import numpy as np
>>> from torchrl.render import FrameBundle
>>> bundle = FrameBundle({"default": np.zeros((2, 2, 3), dtype=np.uint8)}, 0, 0)
>>> sorted(bundle.frames)
['default']
```