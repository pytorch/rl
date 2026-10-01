# display_mujoco_wasm_viewer

torchrl.render.display_mujoco_wasm_viewer(*viewer_dir: str | Path*, ***, *port: int = 5178*, *host: str = '127.0.0.1'*, *install: bool = True*, *width: str = '100%'*, *height: int = 720*, *package_manager: str | None = None*, *startup_timeout: float = 20.0*) → Any[[source]](../../_modules/torchrl/render/mujoco_wasm.html#display_mujoco_wasm_viewer)

Starts and displays a generated MuJoCo WASM viewer in a notebook.

Parameters:

- **viewer_dir** - Directory produced by [`write_mujoco_wasm_viewer()`](torchrl.render.write_mujoco_wasm_viewer.html#torchrl.render.write_mujoco_wasm_viewer).
- **port** - Preferred local port for the Vite development server. A free port
is selected automatically when the preferred port is unavailable.
- **host** - Bind host. Use `127.0.0.1` for local notebooks.
- **install** - Whether to install viewer JavaScript dependencies if needed.
- **width** - IFrame display width.
- **height** - IFrame display height in pixels.
- **package_manager** - Optional package manager command, such as `"pnpm"`.
- **startup_timeout** - Seconds to wait for the Vite server to accept
connections before displaying the iframe.

Returns:

The `subprocess.Popen` object for the Vite server. The selected port
and origin are also attached as `torchrl_mujoco_wasm_port` and
`torchrl_mujoco_wasm_origin` attributes.

Examples

```
>>> from torchrl.render.mujoco_wasm import display_mujoco_wasm_viewer
>>> callable(display_mujoco_wasm_viewer)
True
```