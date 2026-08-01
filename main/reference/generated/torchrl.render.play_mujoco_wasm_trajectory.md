# play_mujoco_wasm_trajectory

torchrl.render.play_mujoco_wasm_trajectory(*qpos: Any*, ***, *dt: float | None = None*, *fps: float | None = None*, *port: int = 5178*, *viewer_origin: str | None = None*, *degrees: bool = False*, *pause: bool = True*, *loop: bool = False*, *interpolate: bool = True*, *wait: bool = False*, *timeout: float | None = None*, *viewer_dir: str | Path | None = None*) → dict[str, Any] | None[[source]](../../_modules/torchrl/render/mujoco_wasm.html#play_mujoco_wasm_trajectory)

Plays a qpos trajectory in a live MuJoCo WASM notebook viewer.

Parameters:

- **qpos** - Iterable of waypoints, each containing qpos values.
- **dt** - Seconds between waypoints. Defaults to `0.1` when `fps` is not
supplied.
- **fps** - Optional playback rate in frames per second.
- **port** - Viewer port when `viewer_origin` is not supplied.
- **viewer_origin** - Browser origin of the viewer iframe.
- **degrees** - Whether the provided values are degrees.
- **pause** - Whether physics should remain paused after playback.
- **loop** - Whether the browser should loop the trajectory.
- **interpolate** - Whether to linearly interpolate between waypoints.
- **wait** - Whether to block until a non-looping trajectory completes.
- **timeout** - Seconds to wait before raising `TimeoutError`.
- **viewer_dir** - Optional generated viewer directory. When provided, the
trajectory is written under the viewer's `public/` directory and a
viewer iframe is displayed with an autoplay URL. This avoids relying on
notebook JavaScript outputs in JupyterLab.

Returns:

The acknowledgement payload when `wait=True`. When `viewer_dir` is
provided, returns a small metadata dictionary for the generated autoplay
trajectory.

Examples

```
>>> from torchrl.render.mujoco_wasm import play_mujoco_wasm_trajectory
>>> callable(play_mujoco_wasm_trajectory)
True
```