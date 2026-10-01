# write_mujoco_wasm_viewer

torchrl.render.write_mujoco_wasm_viewer(*output_dir: str | Path*, *model_path: str | Path*, ***, *asset_paths: list[str | Path] | None = None*, *package_manager: str | None = None*) → Path[[source]](../../_modules/torchrl/render/mujoco_wasm.html#write_mujoco_wasm_viewer)

Writes a local Vite viewer for MuJoCo WASM notebook rendering.

Parameters:

- **output_dir** - Directory that will contain the generated viewer project.
- **model_path** - MJCF/XML file to load in the browser.
- **asset_paths** - Optional extra files or directories referenced by the MJCF.
- **package_manager** - Optional package manager recorded in `package.json`.

Returns:

The generated viewer directory.

Examples

```
>>> import tempfile
>>> from pathlib import Path
>>> from torchrl.render.mujoco_wasm import write_mujoco_wasm_viewer
>>> with tempfile.TemporaryDirectory() as tmpdir:
... model = Path(tmpdir) / "scene.xml"
... _ = model.write_text("<mujoco/>")
... viewer = write_mujoco_wasm_viewer(Path(tmpdir) / "viewer", model)
... (viewer / "package.json").exists()
True
```