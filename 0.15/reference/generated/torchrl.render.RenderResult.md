# RenderResult

*class*torchrl.render.RenderResult(*artifact_path: ~pathlib.Path | None, trajectories: list[~tensordict.base.TensorDictBase], frame_paths: list[~pathlib.Path], metadata: dict[str, ~typing.Any], warnings: list[str], frames: list[list[~torchrl.render.config.FrameBundle]] = <factory>*)[[source]](../../_modules/torchrl/render/config.html#RenderResult)

Result returned by [`torchrl.render.render_policy()`](torchrl.render.render_policy.html#torchrl.render.render_policy).

Parameters:

- **artifact_path** - Main artifact path, if one was written.
- **trajectories** - Collected rollout TensorDicts.
- **frame_paths** - Frame or video paths written by artifact writers.
- **metadata** - JSON-serializable metadata dictionary.
- **warnings** - Non-fatal warnings collected during rendering.
- **frames** - In-memory frame bundles used by artifact writers.

Examples

```
>>> from torchrl.render import RenderResult
>>> result = RenderResult(None, [], [], {"num_trajs": 0}, [])
>>> result.metadata["num_trajs"]
0
```