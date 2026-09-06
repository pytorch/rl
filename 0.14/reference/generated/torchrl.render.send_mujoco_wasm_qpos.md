# send_mujoco_wasm_qpos

torchrl.render.send_mujoco_wasm_qpos(*qpos: Any*, ***, *port: int = 5178*, *viewer_origin: str | None = None*, *degrees: bool = False*, *pause: bool = True*, *wait: bool = False*, *timeout: float = 5.0*) → dict[str, Any] | None[[source]](../../_modules/torchrl/render/mujoco_wasm.html#send_mujoco_wasm_qpos)

Sends one qpos vector to a live MuJoCo WASM notebook viewer.

Parameters:

- **qpos** - Iterable of joint position values. A full MuJoCo `nq` vector is
preferred; the browser also accepts vectors matching the exposed
hinge/slide joint controls.
- **port** - Viewer port when `viewer_origin` is not supplied.
- **viewer_origin** - Browser origin of the viewer iframe.
- **degrees** - Whether the provided values are degrees.
- **pause** - Whether the viewer should remain in kinematic pause mode.
- **wait** - Whether to block until the browser acknowledges the update.
- **timeout** - Seconds to wait before raising `TimeoutError`.

Returns:

The acknowledgement payload when `wait=True`, otherwise `None`.

Examples

```
>>> from torchrl.render.mujoco_wasm import send_mujoco_wasm_qpos
>>> callable(send_mujoco_wasm_qpos)
True
```