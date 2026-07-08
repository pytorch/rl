# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import atexit
import importlib
import importlib.util
import json
import math
import shutil
import socket
import subprocess
import threading
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from tensordict import NestedKey, TensorDictBase

from torchrl.render.config import parse_nested_key

_has_ipython = importlib.util.find_spec("IPython") is not None

_IPYTHON_ERROR = (
    "MuJoCo WASM notebook display helpers require IPython. Install the notebook "
    "extra or call write_mujoco_wasm_viewer() and serve the generated viewer "
    "manually."
)

_DEFAULT_VIEWER_PORT = 5178
_CALLBACKS: dict[str, dict[str, Any]] = {}
_CALLBACKS_LOCK = threading.Lock()
_CALLBACK_SERVER: ThreadingHTTPServer | None = None
_CALLBACK_URL: str | None = None

__all__ = [
    "display_mujoco_wasm_viewer",
    "extract_qpos_trajectory",
    "play_mujoco_wasm_trajectory",
    "send_mujoco_wasm_qpos",
    "write_mujoco_wasm_viewer",
]


def write_mujoco_wasm_viewer(
    output_dir: str | Path,
    model_path: str | Path,
    *,
    asset_paths: list[str | Path] | None = None,
    package_manager: str | None = None,
) -> Path:
    """Writes a local Vite viewer for MuJoCo WASM notebook rendering.

    Args:
        output_dir: Directory that will contain the generated viewer project.
        model_path: MJCF/XML file to load in the browser.
        asset_paths: Optional extra files or directories referenced by the MJCF.
        package_manager: Optional package manager recorded in ``package.json``.

    Returns:
        The generated viewer directory.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from torchrl.render.mujoco_wasm import write_mujoco_wasm_viewer
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     model = Path(tmpdir) / "scene.xml"
        ...     _ = model.write_text("<mujoco/>")
        ...     viewer = write_mujoco_wasm_viewer(Path(tmpdir) / "viewer", model)
        ...     (viewer / "package.json").exists()
        True
    """
    output_dir = Path(output_dir)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model file does not exist: {model_path!s}")
    if not model_path.is_file():
        raise ValueError(f"MuJoCo model path must be a file, got {model_path!s}.")

    scene_dir = output_dir / "public" / "scene"
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    copied = _copy_default_mujoco_scene(model_path, scene_dir)
    for asset_path in asset_paths or []:
        copied.extend(_copy_scene_item(Path(asset_path), scene_dir, model_path.parent))

    manifest = {
        "scene_file": model_path.name,
        "files": sorted({str(path.relative_to(scene_dir)) for path in copied}),
    }
    (scene_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    (output_dir / "src").mkdir(parents=True, exist_ok=True)
    (output_dir / "src" / "main.js").write_text(_VIEWER_JS, encoding="utf-8")
    (output_dir / "src" / "style.css").write_text(_VIEWER_CSS, encoding="utf-8")
    (output_dir / "index.html").write_text(_VIEWER_HTML, encoding="utf-8")
    (output_dir / "package.json").write_text(
        json.dumps(_package_json(package_manager), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def display_mujoco_wasm_viewer(
    viewer_dir: str | Path,
    *,
    port: int = _DEFAULT_VIEWER_PORT,
    host: str = "127.0.0.1",
    install: bool = True,
    width: str = "100%",
    height: int = 720,
    package_manager: str | None = None,
    startup_timeout: float = 20.0,
) -> Any:
    """Starts and displays a generated MuJoCo WASM viewer in a notebook.

    Args:
        viewer_dir: Directory produced by :func:`write_mujoco_wasm_viewer`.
        port: Preferred local port for the Vite development server. A free port
            is selected automatically when the preferred port is unavailable.
        host: Bind host. Use ``127.0.0.1`` for local notebooks.
        install: Whether to install viewer JavaScript dependencies if needed.
        width: IFrame display width.
        height: IFrame display height in pixels.
        package_manager: Optional package manager command, such as ``"pnpm"``.
        startup_timeout: Seconds to wait for the Vite server to accept
            connections before displaying the iframe.

    Returns:
        The ``subprocess.Popen`` object for the Vite server. The selected port
        and origin are also attached as ``torchrl_mujoco_wasm_port`` and
        ``torchrl_mujoco_wasm_origin`` attributes.

    Examples:
        >>> from torchrl.render.mujoco_wasm import display_mujoco_wasm_viewer
        >>> callable(display_mujoco_wasm_viewer)
        True
    """
    display_module = _ipython_display_module()
    viewer_dir = Path(viewer_dir)
    manager = package_manager or _find_package_manager()
    if install and not (viewer_dir / "node_modules").exists():
        _run_package_manager_install(manager, viewer_dir)
    actual_port = _available_viewer_port(host, port)
    process = _start_vite(manager, viewer_dir, host, actual_port)
    process.torchrl_mujoco_wasm_port = actual_port
    process.torchrl_mujoco_wasm_origin = f"http://{host}:{actual_port}"
    atexit.register(process.terminate)
    try:
        _wait_for_viewer_server(process, host, actual_port, startup_timeout)
    except Exception:
        process.terminate()
        raise
    display_module.display(
        display_module.IFrame(
            process.torchrl_mujoco_wasm_origin,
            width=width,
            height=height,
        )
    )
    return process


def send_mujoco_wasm_qpos(
    qpos: Any,
    *,
    port: int = _DEFAULT_VIEWER_PORT,
    viewer_origin: str | None = None,
    degrees: bool = False,
    pause: bool = True,
    wait: bool = False,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """Sends one qpos vector to a live MuJoCo WASM notebook viewer.

    Args:
        qpos: Iterable of joint position values. A full MuJoCo ``nq`` vector is
            preferred; the browser also accepts vectors matching the exposed
            hinge/slide joint controls.
        port: Viewer port when ``viewer_origin`` is not supplied.
        viewer_origin: Browser origin of the viewer iframe.
        degrees: Whether the provided values are degrees.
        pause: Whether the viewer should remain in kinematic pause mode.
        wait: Whether to block until the browser acknowledges the update.
        timeout: Seconds to wait before raising ``TimeoutError``.

    Returns:
        The acknowledgement payload when ``wait=True``, otherwise ``None``.

    Examples:
        >>> from torchrl.render.mujoco_wasm import send_mujoco_wasm_qpos
        >>> callable(send_mujoco_wasm_qpos)
        True
    """
    payload = {
        "type": "torchrl:mujoco_wasm:setQpos",
        "qpos": _as_float_list(qpos),
        "degrees": bool(degrees),
        "pause": bool(pause),
    }
    return _send_viewer_message(
        payload,
        viewer_origin=_viewer_origin(port, viewer_origin),
        wait=wait,
        wait_for="torchrl:mujoco_wasm:qposApplied",
        timeout=timeout,
    )


def play_mujoco_wasm_trajectory(
    qpos: Any,
    *,
    dt: float | None = None,
    fps: float | None = None,
    port: int = _DEFAULT_VIEWER_PORT,
    viewer_origin: str | None = None,
    degrees: bool = False,
    pause: bool = True,
    loop: bool = False,
    interpolate: bool = True,
    wait: bool = False,
    timeout: float | None = None,
    viewer_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Plays a qpos trajectory in a live MuJoCo WASM notebook viewer.

    Args:
        qpos: Iterable of waypoints, each containing qpos values.
        dt: Seconds between waypoints. Defaults to ``0.1`` when ``fps`` is not
            supplied.
        fps: Optional playback rate in frames per second.
        port: Viewer port when ``viewer_origin`` is not supplied.
        viewer_origin: Browser origin of the viewer iframe.
        degrees: Whether the provided values are degrees.
        pause: Whether physics should remain paused after playback.
        loop: Whether the browser should loop the trajectory.
        interpolate: Whether to linearly interpolate between waypoints.
        wait: Whether to block until a non-looping trajectory completes.
        timeout: Seconds to wait before raising ``TimeoutError``.
        viewer_dir: Optional generated viewer directory. When provided, the
            trajectory is written under the viewer's ``public/`` directory and a
            viewer iframe is displayed with an autoplay URL. This avoids relying on
            notebook JavaScript outputs in JupyterLab.

    Returns:
        The acknowledgement payload when ``wait=True``. When ``viewer_dir`` is
        provided, returns a small metadata dictionary for the generated autoplay
        trajectory.

    Examples:
        >>> from torchrl.render.mujoco_wasm import play_mujoco_wasm_trajectory
        >>> callable(play_mujoco_wasm_trajectory)
        True
    """
    if dt is not None and fps is not None:
        raise ValueError("Pass either dt or fps, not both.")
    if dt is not None and dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if fps is not None and fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if wait and loop:
        raise ValueError("wait=True is only supported for non-looping trajectories.")
    waypoints = [_as_float_list(waypoint) for waypoint in qpos]
    if not waypoints:
        raise ValueError("qpos trajectory must contain at least one waypoint.")
    waypoint_size = len(waypoints[0])
    if any(len(waypoint) != waypoint_size for waypoint in waypoints[1:]):
        raise ValueError("All qpos trajectory waypoints must have the same length.")
    payload = {
        "type": "torchrl:mujoco_wasm:playTrajectory",
        "qpos": waypoints,
        "degrees": bool(degrees),
        "pause": bool(pause),
        "loop": bool(loop),
        "interpolate": bool(interpolate),
    }
    if fps is None:
        payload["dt"] = float(0.1 if dt is None else dt)
    else:
        payload["fps"] = float(fps)
    if viewer_dir is not None:
        if wait:
            raise ValueError(
                "wait=True is not supported with viewer_dir autoplay playback."
            )
        return _display_autoplay_trajectory(
            payload, Path(viewer_dir), port, viewer_origin
        )
    return _send_viewer_message(
        payload,
        viewer_origin=_viewer_origin(port, viewer_origin),
        wait=wait,
        wait_for="torchrl:mujoco_wasm:trajectoryComplete",
        timeout=_default_trajectory_timeout(payload, timeout),
    )


def extract_qpos_trajectory(
    rollout: TensorDictBase,
    qpos_key: NestedKey | str = "qpos",
) -> list[list[float]]:
    """Extracts a qpos trajectory from a rollout TensorDict.

    Args:
        rollout: Rollout TensorDict saved by ``rlrender``.
        qpos_key: TensorDict key containing a ``T x nq`` qpos tensor.

    Returns:
        A Python list of qpos waypoints suitable for
        :func:`play_mujoco_wasm_trajectory`.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.render.mujoco_wasm import extract_qpos_trajectory
        >>> rollout = TensorDict({"qpos": torch.zeros(2, 3)}, batch_size=[2])
        >>> extract_qpos_trajectory(rollout, "qpos")
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    """
    # Saved rlrender trajectories store the step's "next" contents at the
    # root, so a "next"-prefixed key also matches its root-level spelling
    # (and vice versa).
    key = parse_nested_key(qpos_key)
    candidates = [key]
    if isinstance(key, tuple) and len(key) > 1 and key[0] == "next":
        candidates.append(key[1] if len(key) == 2 else key[1:])
    elif isinstance(key, tuple):
        candidates.append(("next", *key))
    else:
        candidates.append(("next", key))
    value = None
    for candidate in candidates:
        value = rollout.get(candidate, None)
        if value is not None:
            break
    if value is None:
        raise KeyError(
            f"qpos key {qpos_key!r} was not found in the rollout; tried {candidates}."
        )
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.detach().cpu()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim < 2:
        raise ValueError(
            f"Expected qpos trajectory with at least 2 dimensions, got {tuple(value.shape)}."
        )
    value = value.reshape(-1, value.shape[-1])
    return [[float(item) for item in row.tolist()] for row in value]


def _display_autoplay_trajectory(
    payload: dict[str, Any],
    viewer_dir: Path,
    port: int,
    viewer_origin: str | None,
) -> dict[str, Any]:
    display_module = _ipython_display_module()
    trajectory_dir = viewer_dir / "public" / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_name = f"trajectory_{uuid.uuid4().hex}.json"
    trajectory_path = trajectory_dir / trajectory_name
    trajectory_path.write_text(
        json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8"
    )
    origin = _viewer_origin(port, viewer_origin)
    query = urllib.parse.urlencode(
        {"trajectory": f"/trajectories/{trajectory_name}", "autoplay": "1"}
    )
    display_module.display(
        display_module.IFrame(f"{origin}/?{query}", width="100%", height=720)
    )
    return {
        "ok": True,
        "trajectory_path": str(trajectory_path),
        "viewer_url": f"{origin}/?{query}",
    }


def _copy_default_mujoco_scene(model_path: Path, scene_dir: Path) -> list[Path]:
    copied = _copy_scene_item(model_path, scene_dir, model_path.parent)
    for xml_path in sorted(model_path.parent.glob("*.xml")):
        if xml_path != model_path:
            copied.extend(_copy_scene_item(xml_path, scene_dir, model_path.parent))
    default_assets = model_path.parent / "assets"
    if default_assets.exists():
        copied.extend(_copy_scene_item(default_assets, scene_dir, model_path.parent))
    return copied


def _copy_scene_item(path: Path, scene_dir: Path, base_dir: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo asset path does not exist: {path!s}")
    try:
        relative = path.relative_to(base_dir)
    except ValueError:
        relative = Path(path.name)
    destination = scene_dir / relative
    if path.is_dir():
        shutil.copytree(path, destination, dirs_exist_ok=True)
        return [item for item in destination.rglob("*") if item.is_file()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return [destination]


def _package_json(package_manager: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "torchrl-mujoco-wasm-viewer",
        "version": "0.0.0",
        "private": True,
        "type": "module",
        "scripts": {"dev": "vite --host 127.0.0.1 --port 5178"},
        "dependencies": {"mujoco-js": "0.0.7", "three": "0.184.0"},
        "devDependencies": {"vite": "8.0.16"},
    }
    if package_manager is not None:
        payload["packageManager"] = package_manager
    return payload


def _find_package_manager() -> str:
    for candidate in ("pnpm", "npm"):
        if shutil.which(candidate) is not None:
            return candidate
    raise RuntimeError(
        "MuJoCo WASM viewer startup requires pnpm or npm on PATH. "
        "Call write_mujoco_wasm_viewer() and serve the viewer manually if needed."
    )


def _run_package_manager_install(manager: str, viewer_dir: Path) -> None:
    subprocess.run([manager, "install"], cwd=viewer_dir, check=True)


def _start_vite(
    manager: str, viewer_dir: Path, host: str, port: int
) -> subprocess.Popen:
    if manager == "npm":
        command = [
            "npm",
            "exec",
            "vite",
            "--",
            "--host",
            host,
            "--port",
            str(port),
            "--strictPort",
        ]
    else:
        command = [
            manager,
            "exec",
            "vite",
            "--host",
            host,
            "--port",
            str(port),
            "--strictPort",
        ]
    return subprocess.Popen(command, cwd=viewer_dir)


def _wait_for_viewer_server(
    process: subprocess.Popen,
    host: str,
    port: int,
    timeout: float,
    poll_interval: float = 0.05,
) -> None:
    if timeout <= 0:
        return
    attempts = max(1, math.ceil(timeout / poll_interval))
    sleeper = threading.Event()
    for _ in range(attempts):
        if _can_connect(host, port):
            return
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "MuJoCo WASM Vite server exited before it was ready "
                f"(return code {return_code}). Command: {_format_command(process.args)}"
            )
        sleeper.wait(poll_interval)
    raise TimeoutError(
        "Timed out waiting for MuJoCo WASM Vite server to start at "
        f"http://{host}:{port}. Command: {_format_command(process.args)}"
    )


def _can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _format_command(command: Any) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command)


def _available_viewer_port(host: str, preferred_port: int) -> int:
    if preferred_port == 0:
        return _reserve_free_port(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex((host, preferred_port)) != 0:
            return preferred_port
    return _reserve_free_port(host)


def _reserve_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _ipython_display_module() -> Any:
    if not _has_ipython:
        raise ModuleNotFoundError(_IPYTHON_ERROR)
    return importlib.import_module("IPython.display")


def _viewer_origin(port: int, viewer_origin: str | None) -> str:
    if viewer_origin is not None:
        return viewer_origin.rstrip("/")
    return f"http://127.0.0.1:{port}"


def _send_viewer_message(
    payload: dict[str, Any],
    *,
    viewer_origin: str,
    wait: bool,
    wait_for: str,
    timeout: float,
) -> dict[str, Any] | None:
    display_module = _ipython_display_module()
    request_id = None
    callback_url = None
    slot = None
    if wait:
        request_id = uuid.uuid4().hex
        payload = dict(payload)
        payload["requestId"] = request_id
        callback_url = _ensure_callback_server()
        slot = {"event": threading.Event(), "payload": None}
        with _CALLBACKS_LOCK:
            _CALLBACKS[request_id] = slot
    display_module.display(
        display_module.Javascript(
            _post_message_javascript(payload, viewer_origin, wait_for, callback_url)
        )
    )
    if not wait:
        return None
    try:
        if not slot["event"].wait(timeout):
            raise TimeoutError(f"timed out after {timeout:.1f}s waiting for {wait_for}")
        response = slot["payload"]
        if response is None:
            raise RuntimeError("MuJoCo WASM viewer did not return an acknowledgement.")
        if response.get("ok") is False:
            raise RuntimeError(response.get("error", str(response)))
        return response
    finally:
        with _CALLBACKS_LOCK:
            _CALLBACKS.pop(request_id, None)


def _post_message_javascript(
    payload: dict[str, Any],
    viewer_origin: str,
    wait_for: str,
    callback_url: str | None,
) -> str:
    return f"""
(() => {{
  const payload = {json.dumps(payload)};
  const viewerOrigin = {json.dumps(viewer_origin)};
  const waitFor = {json.dumps(wait_for)};
  const callbackUrl = {json.dumps(callback_url)};
  const requestId = payload.requestId;
  const notifyPython = (body) => {{
    if (!callbackUrl) return;
    fetch(callbackUrl, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{requestId, ...body}}),
    }}).catch((error) => console.error('MuJoCo WASM callback failed', error));
  }};
  try {{
    const frame = [...document.querySelectorAll('iframe')]
      .find((candidate) => candidate.src.startsWith(viewerOrigin));
    if (!frame || !frame.contentWindow) {{
      throw new Error('MuJoCo WASM viewer iframe not found; display the viewer before sending data.');
    }}
    if (requestId && callbackUrl) {{
      const onMessage = (event) => {{
        if (event.origin !== viewerOrigin) return;
        const message = event.data || {{}};
        if (message.requestId !== requestId) return;
        const done = message.type === waitFor;
        const cancelled = message.type === 'torchrl:mujoco_wasm:trajectoryCancelled';
        if (message.ok === false || done || cancelled) {{
          window.removeEventListener('message', onMessage);
          notifyPython(message);
        }}
      }};
      window.addEventListener('message', onMessage);
    }}
    frame.contentWindow.postMessage(payload, viewerOrigin);
  }} catch (error) {{
    notifyPython({{
      type: 'torchrl:mujoco_wasm:dispatchError',
      ok: false,
      error: String(error?.message ?? error),
    }});
    if (!callbackUrl) throw error;
  }}
}})();
"""


def _ensure_callback_server() -> str:
    global _CALLBACK_SERVER, _CALLBACK_URL
    if _CALLBACK_URL is not None:
        return _CALLBACK_URL

    class CallbackHandler(BaseHTTPRequestHandler):
        def _finish(self, status: int = 204) -> None:
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_OPTIONS(self) -> None:
            self._finish()

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw_body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw_body or "{}")
            except json.JSONDecodeError:
                self._finish(400)
                return
            request_id = payload.get("requestId")
            with _CALLBACKS_LOCK:
                slot = _CALLBACKS.get(request_id)
            if slot is None:
                self._finish(404)
                return
            slot["payload"] = payload
            slot["event"].set()
            self._finish()

        def log_message(self, *_args: Any) -> None:
            return None

    _CALLBACK_SERVER = ThreadingHTTPServer(("127.0.0.1", 0), CallbackHandler)
    thread = threading.Thread(
        target=_CALLBACK_SERVER.serve_forever,
        name="torchrl-mujoco-wasm-callback",
        daemon=True,
    )
    thread.start()
    port = _CALLBACK_SERVER.server_address[1]
    _CALLBACK_URL = f"http://127.0.0.1:{port}/ack"
    return _CALLBACK_URL


def _default_trajectory_timeout(
    payload: dict[str, Any], timeout: float | None
) -> float:
    if timeout is not None:
        return float(timeout)
    if "fps" in payload:
        dt = 1.0 / float(payload["fps"])
    else:
        dt = float(payload.get("dt", 0.1))
    return max(30.0, len(payload["qpos"]) * dt + 5.0)


def _as_float_list(values: Any) -> list[float]:
    if torch.is_tensor(values):
        values = values.detach().cpu().reshape(-1).tolist()
    result = [float(value) for value in values]
    if not result:
        raise ValueError("qpos vectors must contain at least one value.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("qpos values must be finite.")
    return result


_VIEWER_HTML = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>TorchRL MuJoCo WASM viewer</title>
  </head>
  <body>
    <main>
      <section id=\"viewer\"></section>
      <aside>
        <h1>TorchRL MuJoCo WASM</h1>
        <p id=\"status\">starting</p>
        <div class=\"buttons\">
          <button id=\"reset\">Reset</button>
          <button id=\"pause\">Run physics</button>
          <button id=\"step\">Step</button>
        </div>
        <label class=\"toggle\"><input id=\"auto-sweep\" type=\"checkbox\" /> demo sweep</label>
        <h2>Joint controls</h2>
        <div id=\"joint-controls\"></div>
      </aside>
    </main>
    <script type=\"module\" src=\"/src/main.js\"></script>
  </body>
</html>
"""

_VIEWER_CSS = """html, body, main {
  width: 100%;
  height: 100%;
  margin: 0;
}
body {
  background: #0f172a;
  color: #dbeafe;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
}
#viewer {
  min-width: 0;
  min-height: 420px;
}
aside {
  box-sizing: border-box;
  overflow: auto;
  padding: 18px;
  border-left: 1px solid #334155;
  background: #111827;
}
h1, h2 {
  margin: 0 0 12px;
}
#status {
  color: #93c5fd;
  min-height: 3em;
}
.buttons {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}
button {
  border: 1px solid #60a5fa;
  border-radius: 6px;
  background: #1d4ed8;
  color: white;
  cursor: pointer;
  padding: 6px 10px;
}
.toggle {
  display: block;
  margin-bottom: 16px;
}
.joint {
  margin-bottom: 12px;
}
.joint label {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 12px;
}
.joint input {
  width: 100%;
}
output {
  color: #bfdbfe;
}
"""

_VIEWER_JS = r"""import "./style.css";

import loadMujoco from "mujoco-js";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const WORKING_DIR = "/working";
const MANIFEST_URL = "/scene/manifest.json";
const PLANE_GEOM_TYPE = 0;
const SPHERE_GEOM_TYPE = 2;
const CAPSULE_GEOM_TYPE = 3;
const CYLINDER_GEOM_TYPE = 5;
const BOX_GEOM_TYPE = 6;
const MESH_GEOM_TYPE = 7;
const HINGE_JOINT_TYPE = 3;
const SLIDE_JOINT_TYPE = 2;
const HIDDEN_COLLISION_GROUP = 3;

const statusEl = element("status");
const viewerEl = element("viewer");
const resetButton = element("reset");
const pauseButton = element("pause");
const stepButton = element("step");
const autoSweepInput = element("auto-sweep");
const jointControlsEl = element("joint-controls");

let paused = true;
let model = null;
let data = null;
let mujoco = null;
let jointControls = [];
let homeQpos = [];
let threeScene = null;
let renderer = null;
let camera = null;
let controls = null;
let geomVisuals = [];
let lastFrameMs = performance.now();
let trajectoryPlayback = null;

resetButton.addEventListener("click", () => {
  cancelActiveTrajectory("trajectory cancelled by reset");
  resetHomePose();
  updateJointControlValues();
  renderOnce();
});

pauseButton.addEventListener("click", () => {
  paused = !paused;
  updatePauseButton();
});

stepButton.addEventListener("click", () => {
  if (!mujoco || !model || !data) return;
  mujoco.mj_step(model, data);
  updateJointControlValues();
  renderOnce();
});

window.addEventListener("message", (event) => {
  const message = parseViewerMessage(event.data);
  if (!message) return;
  const responseTarget = {
    source: event.source,
    targetOrigin: event.origin || "*",
    requestId: message.requestId,
  };
  if (message.type === "torchrl:mujoco_wasm:setQpos") {
    postViewerResponse(responseTarget, "torchrl:mujoco_wasm:qposApplied", applyQposMessage(message));
  } else {
    postViewerResponse(responseTarget, "torchrl:mujoco_wasm:trajectoryStarted", applyTrajectoryMessage(message, responseTarget));
  }
});

main().catch((error) => {
  console.error(error);
  statusEl.textContent = error instanceof Error ? error.message : String(error);
  statusEl.classList.add("error");
});

async function main() {
  setStatus("fetching scene manifest");
  const manifest = await fetchManifest();
  setStatus("loading MuJoCo WASM");
  mujoco = await loadMujoco();
  setStatus("copying MJCF assets into WASM FS");
  await populateWorkingDirectory(mujoco, manifest);
  setStatus("compiling MJCF in WASM");
  model = mujoco.MjModel.loadFromXML(`${WORKING_DIR}/${manifest.scene_file}`);
  data = new mujoco.MjData(model);
  resetHomePose();
  setStatus(`compiled MuJoCo scene: ${model.nbody} bodies, ${model.nmesh} meshes`);
  buildViewer(model, data);
  buildJointControls(model, data);
  updatePauseButton();
  await playTrajectoryFromUrl();
  requestAnimationFrame(animate);
}

async function fetchManifest() {
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) throw new Error(`Missing ${MANIFEST_URL}. Regenerate the notebook artifact.`);
  return await response.json();
}

async function populateWorkingDirectory(activeMujoco, manifest) {
  if (!activeMujoco.FS.analyzePath(WORKING_DIR, false).exists) {
    activeMujoco.FS.mkdir(WORKING_DIR);
    activeMujoco.FS.mount(activeMujoco.MEMFS, {root: "."}, WORKING_DIR);
  }
  for (const relPath of manifest.files) {
    ensureVirtualDirectories(activeMujoco, relPath);
    const response = await fetch(`/scene/${relPath}`);
    if (!response.ok) throw new Error(`Could not fetch /scene/${relPath}: ${response.status}`);
    activeMujoco.FS.writeFile(`${WORKING_DIR}/${relPath}`, new Uint8Array(await response.arrayBuffer()));
  }
}

function ensureVirtualDirectories(activeMujoco, relPath) {
  const parts = relPath.split("/");
  let current = WORKING_DIR;
  for (const part of parts.slice(0, -1)) {
    current = `${current}/${part}`;
    if (!activeMujoco.FS.analyzePath(current, false).exists) activeMujoco.FS.mkdir(current);
  }
}

function buildViewer(activeModel, activeData) {
  THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
  const scene = new THREE.Scene();
  threeScene = scene;
  scene.background = new THREE.Color(0x142033);
  scene.fog = new THREE.Fog(scene.background, 4.0, 12.0);
  camera = new THREE.PerspectiveCamera(42, viewerEl.clientWidth / viewerEl.clientHeight, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.8, -2.2, 1.2);
  scene.add(camera);
  scene.add(new THREE.HemisphereLight(0xdce9ff, 0x1a2638, 1.2));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
  keyLight.position.set(2.0, -2.2, 3.2);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0x9bbdff, 0.8);
  fillLight.position.set(-2.0, 1.2, 1.4);
  scene.add(fillLight);
  addFloor(scene);
  geomVisuals = addCompiledMuJoCoGeoms(scene, activeModel);
  renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewerEl.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.35);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();
  new ResizeObserver(() => {
    if (!camera || !renderer) return;
    const width = Math.max(viewerEl.clientWidth, 1);
    const height = Math.max(viewerEl.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    renderOnce();
  }).observe(viewerEl);
  updateGeomVisuals(activeData);
  renderOnce(scene);
}

function addFloor(scene) {
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.MeshStandardMaterial({color: 0x1d3046, roughness: 0.8, side: THREE.DoubleSide}));
  scene.add(floor);
  const grid = new THREE.GridHelper(8, 32, 0x8bb2df, 0x35516d);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.002;
  scene.add(grid);
}

function addCompiledMuJoCoGeoms(scene, activeModel) {
  const visuals = [];
  const geometryCache = new Map();
  for (let geomId = 0; geomId < activeModel.ngeom; geomId++) {
    if (Number(activeModel.geom_group[geomId]) === HIDDEN_COLLISION_GROUP) continue;
    const geomType = Number(activeModel.geom_type[geomId]);
    if (geomType === PLANE_GEOM_TYPE) continue;
    const geometry = geometryForGeom(activeModel, geomId, geomType, geometryCache);
    if (!geometry) continue;
    const mesh = new THREE.Mesh(geometry, materialForGeom(activeModel, geomId));
    mesh.matrixAutoUpdate = false;
    scene.add(mesh);
    visuals.push({geomId, object: mesh});
  }
  return visuals;
}

function geometryForGeom(activeModel, geomId, geomType, cache) {
  if (geomType === MESH_GEOM_TYPE) return getMeshGeometry(activeModel, Number(activeModel.geom_dataid[geomId]), cache);
  const size = activeModel.geom_size;
  const s = geomId * 3;
  if (geomType === SPHERE_GEOM_TYPE) return new THREE.SphereGeometry(Number(size[s]), 32, 16);
  if (geomType === BOX_GEOM_TYPE) return new THREE.BoxGeometry(2 * Number(size[s]), 2 * Number(size[s + 1]), 2 * Number(size[s + 2]));
  if (geomType === CYLINDER_GEOM_TYPE) {
    const geometry = new THREE.CylinderGeometry(Number(size[s]), Number(size[s]), 2 * Number(size[s + 1]), 32);
    geometry.rotateX(Math.PI / 2);
    return geometry;
  }
  if (geomType === CAPSULE_GEOM_TYPE) {
    const geometry = new THREE.CapsuleGeometry(Number(size[s]), 2 * Number(size[s + 1]), 8, 32);
    geometry.rotateX(Math.PI / 2);
    return geometry;
  }
  return null;
}

function getMeshGeometry(activeModel, meshId, cache) {
  const cached = cache.get(meshId);
  if (cached) return cached;
  const vertexCount = Number(activeModel.mesh_vertnum[meshId]);
  const vertexAddress = Number(activeModel.mesh_vertadr[meshId]);
  const vertices = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount * 3; i++) vertices[i] = Number(activeModel.mesh_vert[vertexAddress * 3 + i]);
  const faceCount = Number(activeModel.mesh_facenum[meshId]);
  const faceAddress = Number(activeModel.mesh_faceadr[meshId]);
  const indices = vertexCount > 65535 ? new Uint32Array(faceCount * 3) : new Uint16Array(faceCount * 3);
  for (let i = 0; i < faceCount * 3; i++) indices[i] = Number(activeModel.mesh_face[faceAddress * 3 + i]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  cache.set(meshId, geometry);
  return geometry;
}

function materialForGeom(activeModel, geomId) {
  const matId = Number(activeModel.geom_matid[geomId]);
  const rgba = new Float32Array(4);
  if (matId >= 0) {
    for (let i = 0; i < 4; i++) rgba[i] = Number(activeModel.mat_rgba[matId * 4 + i]);
  } else {
    for (let i = 0; i < 4; i++) rgba[i] = Number(activeModel.geom_rgba[geomId * 4 + i]);
  }
  const alpha = rgba[3] || 1;
  return new THREE.MeshStandardMaterial({color: new THREE.Color(rgba[0], rgba[1], rgba[2]), roughness: 0.48, metalness: 0.02, transparent: alpha < 1, opacity: alpha, side: THREE.DoubleSide});
}

function buildJointControls(activeModel, activeData) {
  jointControlsEl.replaceChildren();
  jointControls = [];
  for (let jointId = 0; jointId < activeModel.njnt; jointId++) {
    const jointType = Number(activeModel.jnt_type[jointId]);
    if (jointType !== HINGE_JOINT_TYPE && jointType !== SLIDE_JOINT_TYPE) continue;
    const qposAddress = Number(activeModel.jnt_qposadr[jointId]);
    const name = decodeName(activeModel.names, Number(activeModel.name_jntadr[jointId]));
    const limited = Boolean(Number(activeModel.jnt_limited[jointId]));
    const min = limited ? Number(activeModel.jnt_range[jointId * 2]) : -2 * Math.PI;
    const max = limited ? Number(activeModel.jnt_range[jointId * 2 + 1]) : 2 * Math.PI;
    const wrapper = document.createElement("div");
    wrapper.className = "joint";
    const label = document.createElement("label");
    label.htmlFor = `joint-${jointId}`;
    label.append(document.createTextNode(name));
    const output = document.createElement("output");
    output.value = formatRadians(Number(activeData.qpos[qposAddress]));
    label.append(output);
    const slider = document.createElement("input");
    slider.id = `joint-${jointId}`;
    slider.type = "range";
    slider.min = String(min);
    slider.max = String(max);
    slider.step = "0.001";
    slider.value = String(activeData.qpos[qposAddress]);
    slider.addEventListener("input", () => {
      if (!mujoco || !model || !data) return;
      cancelActiveTrajectory("trajectory interrupted by slider");
      autoSweepInput.checked = false;
      const value = Number(slider.value);
      data.qpos[qposAddress] = value;
      setMatchingActuatorCtrl(data, jointId, value);
      mujoco.mj_forward(model, data);
      output.value = formatRadians(value);
      renderOnce();
    });
    wrapper.append(label, slider);
    jointControlsEl.append(wrapper);
    jointControls.push({name, jointId, qposAddress, slider, output});
  }
}

async function playTrajectoryFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const trajectoryUrl = params.get("trajectory");
  if (!trajectoryUrl || params.get("autoplay") === "0") return;
  setStatus(`fetching trajectory ${trajectoryUrl}`);
  const response = await fetch(trajectoryUrl);
  if (!response.ok) throw new Error(`Could not fetch trajectory ${trajectoryUrl}: ${response.status}`);
  const payload = await response.json();
  const result = applyTrajectoryMessage(payload, null);
  if (result?.ok === false) throw new Error(result.error || "Could not start trajectory playback");
}

function parseViewerMessage(value) {
  if (!value || typeof value !== "object") return null;
  if (value.type !== "torchrl:mujoco_wasm:setQpos" && value.type !== "torchrl:mujoco_wasm:playTrajectory") return null;
  return value;
}

function applyQposMessage(message) {
  if (!mujoco || !model || !data) return {ok: false, error: "MuJoCo model is unavailable"};
  try {
    const qpos = normalizeQposVector(message.qpos, message.degrees);
    cancelActiveTrajectory("trajectory interrupted by qpos command");
    paused = message.pause !== false;
    updatePauseButton();
    autoSweepInput.checked = false;
    applyQposVector(qpos);
    renderOnce();
    setStatus(`applied qpos (${qpos.length} value(s))`);
    return {ok: true, nq: qpos.length};
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    setStatus(text);
    return {ok: false, error: text};
  }
}

function applyTrajectoryMessage(message, responseTarget) {
  if (!mujoco || !model || !data) return {ok: false, error: "MuJoCo model is unavailable"};
  try {
    if (!Array.isArray(message.qpos) || message.qpos.length === 0) throw new Error("qpos trajectory must be a non-empty array");
    const waypoints = message.qpos.map((waypoint) => normalizeQposVector(waypoint, message.degrees));
    const dtMs = trajectoryDtMs(message);
    cancelActiveTrajectory("trajectory replaced by another trajectory");
    paused = true;
    updatePauseButton();
    autoSweepInput.checked = false;
    trajectoryPlayback = {waypoints, startMs: performance.now(), dtMs, loop: message.loop === true, interpolate: message.interpolate !== false, pauseAfter: message.pause !== false, lastWaypointIndex: -1, responseTarget};
    applyQposVector(waypoints[0]);
    renderOnce();
    setStatus(`playing trajectory: ${waypoints.length} frame(s), ${(dtMs / 1000).toFixed(3)} s/frame`);
    return {ok: true, frames: waypoints.length, dt: dtMs / 1000};
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    setStatus(text);
    return {ok: false, error: text};
  }
}

function normalizeQposVector(value, degrees) {
  if (!Array.isArray(value)) throw new Error("qpos must be an array");
  const scale = degrees ? Math.PI / 180 : 1;
  const out = value.map((item) => {
    const number = Number(item);
    if (!Number.isFinite(number)) throw new Error(`qpos value must be finite, got ${item}`);
    return number * scale;
  });
  if (out.length !== model.nq && out.length !== jointControls.length) throw new Error(`qpos must have model.nq (${model.nq}) values or ${jointControls.length} joint-control values`);
  return out;
}

function applyQposVector(values) {
  if (values.length === model.nq) {
    for (let i = 0; i < values.length; i++) data.qpos[i] = values[i];
  } else {
    for (let i = 0; i < values.length; i++) {
      const control = jointControls[i];
      data.qpos[control.qposAddress] = values[i];
      setMatchingActuatorCtrl(data, control.jointId, values[i]);
    }
  }
  mujoco.mj_forward(model, data);
  updateJointControlValues();
}

function trajectoryDtMs(message) {
  if (message.dt !== undefined) {
    const dt = Number(message.dt);
    if (!Number.isFinite(dt) || dt <= 0) throw new Error(`dt must be positive, got ${message.dt}`);
    return dt * 1000;
  }
  if (message.fps !== undefined) {
    const fps = Number(message.fps);
    if (!Number.isFinite(fps) || fps <= 0) throw new Error(`fps must be positive, got ${message.fps}`);
    return 1000 / fps;
  }
  return 100;
}

function animate(nowMs) {
  requestAnimationFrame(animate);
  if (!mujoco || !model || !data) return;
  const elapsedMs = Math.min(nowMs - lastFrameMs, 80);
  lastFrameMs = nowMs;
  if (trajectoryPlayback) updateTrajectoryPlayback(nowMs);
  else if (paused) {
    if (autoSweepInput.checked) {
      applyKinematicSweep(nowMs / 1000);
      mujoco.mj_forward(model, data);
      updateJointControlValues();
    }
  } else {
    const timestepMs = Math.max(Number(model.opt.timestep) * 1000, 1);
    const steps = Math.max(1, Math.floor(elapsedMs / timestepMs));
    for (let i = 0; i < steps; i++) mujoco.mj_step(model, data);
    updateJointControlValues();
  }
  controls?.update();
  updateGeomVisuals(data);
  if (renderer && camera) renderer.render(rendererScene(), camera);
}

function updateTrajectoryPlayback(nowMs) {
  if (!trajectoryPlayback) return;
  const playback = trajectoryPlayback;
  const framePosition = (nowMs - playback.startMs) / playback.dtMs;
  const finalIndex = playback.waypoints.length - 1;
  if (!playback.loop && framePosition >= finalIndex) {
    applyQposVector(playback.waypoints[finalIndex]);
    trajectoryPlayback = null;
    paused = playback.pauseAfter;
    updatePauseButton();
    setStatus("trajectory complete");
    postViewerResponse(playback.responseTarget, "torchrl:mujoco_wasm:trajectoryComplete", {ok: true, frames: playback.waypoints.length});
    return;
  }
  if (!playback.interpolate) {
    const waypointIndex = playback.loop ? positiveModulo(Math.floor(framePosition), playback.waypoints.length) : Math.max(0, Math.min(finalIndex, Math.floor(framePosition)));
    if (waypointIndex !== playback.lastWaypointIndex) {
      applyQposVector(playback.waypoints[waypointIndex]);
      playback.lastWaypointIndex = waypointIndex;
    }
    return;
  }
  const lowerPosition = Math.floor(framePosition);
  const lowerIndex = playback.loop ? positiveModulo(lowerPosition, playback.waypoints.length) : Math.max(0, Math.min(finalIndex, lowerPosition));
  const upperIndex = playback.loop ? positiveModulo(lowerPosition + 1, playback.waypoints.length) : Math.max(0, Math.min(finalIndex, lowerPosition + 1));
  const alpha = framePosition - Math.floor(framePosition);
  applyQposVector(playback.waypoints[lowerIndex].map((lowerValue, jointIndex) => lowerValue + alpha * (playback.waypoints[upperIndex][jointIndex] - lowerValue)));
}

function positiveModulo(value, divisor) {
  return ((value % divisor) + divisor) % divisor;
}

function cancelActiveTrajectory(reason) {
  if (!trajectoryPlayback) return;
  const playback = trajectoryPlayback;
  trajectoryPlayback = null;
  postViewerResponse(playback.responseTarget, "torchrl:mujoco_wasm:trajectoryCancelled", {ok: false, error: reason});
}

function postViewerResponse(target, type, body) {
  if (!target) return;
  target.source?.postMessage({type, requestId: target.requestId, ...body}, {targetOrigin: target.targetOrigin});
}

function applyKinematicSweep(timeSeconds) {
  if (!data || homeQpos.length < 3) return;
  for (let i = 0; i < Math.min(3, model.nq); i++) data.qpos[i] = homeQpos[i] + 0.25 * Math.sin(timeSeconds * (0.5 + i * 0.2) + i);
}

function resetHomePose() {
  if (!mujoco || !model || !data) return;
  mujoco.mj_resetData(model, data);
  mujoco.mj_forward(model, data);
  homeQpos = Array.from(data.qpos).slice(0, model.nq);
}

function updateGeomVisuals(activeData) {
  const matrix = new THREE.Matrix4();
  const xpos = activeData.geom_xpos;
  const xmat = activeData.geom_xmat;
  for (const {geomId, object} of geomVisuals) {
    const p = geomId * 3;
    const r = geomId * 9;
    matrix.set(Number(xmat[r]), Number(xmat[r + 1]), Number(xmat[r + 2]), Number(xpos[p]), Number(xmat[r + 3]), Number(xmat[r + 4]), Number(xmat[r + 5]), Number(xpos[p + 1]), Number(xmat[r + 6]), Number(xmat[r + 7]), Number(xmat[r + 8]), Number(xpos[p + 2]), 0, 0, 0, 1);
    object.matrix.copy(matrix);
    object.matrixWorldNeedsUpdate = true;
  }
}

function updateJointControlValues() {
  if (!data) return;
  for (const control of jointControls) {
    const value = Number(data.qpos[control.qposAddress]);
    control.slider.value = String(value);
    control.output.value = formatRadians(value);
  }
}

function setMatchingActuatorCtrl(activeData, jointId, value) {
  if (!activeData.ctrl || !model) return;
  const actuatorCount = Number(model.nu ?? 0);
  for (let actuatorId = 0; actuatorId < actuatorCount; actuatorId++) {
    const trnType = model.actuator_trntype ? Number(model.actuator_trntype[actuatorId]) : 0;
    if (trnType !== 0) continue;
    if (Number(model.actuator_trnid[actuatorId * 2]) !== jointId) continue;
    if (actuatorId < activeData.ctrl.length) activeData.ctrl[actuatorId] = value;
  }
}

function renderOnce(sceneOverride) {
  if (!renderer || !camera) return;
  if (data) updateGeomVisuals(data);
  controls?.update();
  renderer.render(sceneOverride ?? rendererScene(), camera);
}

function rendererScene() {
  if (!threeScene) throw new Error("Three scene is not initialized");
  return threeScene;
}

function updatePauseButton() {
  pauseButton.textContent = paused ? "Run physics" : "Pause";
}

function setStatus(message) {
  statusEl.textContent = message;
}

function element(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id}`);
  return el;
}

function decodeName(names, address) {
  const bytes = names;
  let end = address;
  while (end < bytes.length && bytes[end] !== 0) end++;
  return new TextDecoder().decode(bytes.slice(address, end));
}

function formatRadians(value) {
  return `${value.toFixed(3)} rad`;
}
"""
