# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import json
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torchrl.render as render_module
import torchrl.render.artifacts as artifacts_module
import torchrl.render.mujoco_wasm as mujoco_wasm_module
from tensordict import TensorDict

from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.record.loggers.common import _has_torchcodec
from torchrl.render import (
    call_with_supported_kwargs,
    checkpoint_hash,
    collect_render_rollouts,
    display_mujoco_wasm_viewer,
    import_from_string,
    infer_state_dict,
    load_checkpoint,
    load_render_policy,
    make_render_env,
    parse_nested_key,
    play_mujoco_wasm_trajectory,
    render_policy,
    RenderConfig,
    RenderPolicySpec,
    TensorDictPolicyAdapter,
    write_mujoco_wasm_viewer,
    write_render_artifact,
)
from torchrl.render.cli import build_parser, config_from_args, main as cli_main
from torchrl.render.mujoco_wasm import extract_qpos_trajectory
from torchrl.render.video import compose_frame_grid, normalize_frame_output

_has_pil = importlib.util.find_spec("PIL") is not None


class FakeIFrame:
    def __init__(self, src, width=None, height=None):
        self.src = src
        self.width = width
        self.height = height


class FakeDisplayModule:
    IFrame = FakeIFrame

    def __init__(self):
        self.objects = []

    def display(self, obj):
        self.objects.append(obj)


class FakeProcess:
    def __init__(self, return_code=None):
        self.args = ["fake-vite"]
        self.return_code = return_code
        self.terminated = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True


class TinyRenderEnv(EnvBase):
    def __init__(self, device="cpu"):
        device = torch.device(device)
        super().__init__(device=device, batch_size=[])
        self.observation_spec = Composite(
            count=Unbounded(shape=(), dtype=torch.float32, device=device),
            pixels=Unbounded(shape=(4, 4, 3), dtype=torch.uint8, device=device),
            device=device,
        )
        self.action_spec = Composite(
            action=Unbounded(shape=(1,), dtype=torch.float32, device=device),
            device=device,
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=device)
        self.done_spec = Composite(
            done=Unbounded(shape=(1,), dtype=torch.bool, device=device),
            terminated=Unbounded(shape=(1,), dtype=torch.bool, device=device),
            truncated=Unbounded(shape=(1,), dtype=torch.bool, device=device),
            device=device,
        )
        self._count = 0
        self._seed_value = None

    def _reset(self, tensordict=None, **kwargs):
        self._count = 0
        return TensorDict(
            {
                "count": torch.zeros((), device=self.device),
                "pixels": self._pixels(),
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )

    def _step(self, tensordict):
        self._count += 1
        done = torch.tensor([self._count >= 2], dtype=torch.bool, device=self.device)
        return TensorDict(
            {
                "count": torch.tensor(float(self._count), device=self.device),
                "pixels": self._pixels(),
                "reward": torch.ones(1, device=self.device),
                "done": done,
                "terminated": done,
                "truncated": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        self._seed_value = seed

    def render(self):
        return np.full((4, 4, 3), self._count, dtype=np.uint8)

    def _pixels(self):
        return torch.full((4, 4, 3), self._count, dtype=torch.uint8, device=self.device)


class ZeroPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tensordict):
        tensordict.set("action", self.bias.detach().clone())
        return tensordict


def tiny_env_factory(spec):
    return TinyRenderEnv(device=spec.device)


def tiny_policy_factory(spec):
    return ZeroPolicy().to(spec.device)


def tensor_only_policy(obs):
    if not torch.is_tensor(obs):
        raise TypeError("expected a tensor observation")
    return obs + 1


class TestRenderConfig:
    def test_parse_nested_key_and_import_from_string(self, tmp_path, monkeypatch):
        module = tmp_path / "render_factories.py"
        module.write_text("VALUE = 3\n", encoding="utf-8")
        file_module = tmp_path / "file_factories.py"
        file_module.write_text("VALUE = 4\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        assert parse_nested_key("agent.action") == ("agent", "action")
        assert import_from_string("render_factories:VALUE") == 3
        assert import_from_string(f"{file_module}:VALUE") == 4
        with pytest.raises(ImportError, match="attribute"):
            import_from_string("render_factories:MISSING")

        def factory_with_default(task="default"):
            return task

        assert (
            call_with_supported_kwargs(factory_with_default, object(), {}) == "default"
        )
        assert (
            call_with_supported_kwargs(
                factory_with_default, object(), {"task": "configured"}
            )
            == "configured"
        )


class TestRenderCLI:
    def test_config_from_args_infers_format_and_kwargs(self, tmp_path):
        args = build_parser().parse_args(
            [
                "--ckpt",
                str(tmp_path / "policy.pt"),
                "--policy",
                "project.policy:make_policy",
                "--env",
                "project.env:make_env",
                "--out",
                str(tmp_path / "report.ipynb"),
                "--env-kwargs",
                '{"task": "Tiny"}',
                "--action-key",
                "agent.action",
            ]
        )
        config = config_from_args(args)
        assert config.format == "ipynb"
        assert config.save_rollout is True
        assert config.env_kwargs == {"task": "Tiny"}
        assert config.action_key == ("agent", "action")

    def test_config_and_cli_validation(self, tmp_path, capsys):
        with pytest.raises(ValueError, match="format must be one of"):
            RenderConfig(
                tmp_path / "policy.pt",
                "project.policy:make_policy",
                "project.env:make_env",
                format="invalid",
            )
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--ckpt", str(tmp_path / "policy.pt")])
        assert exc_info.value.code == 2
        assert "Missing required rlrender option --policy" in capsys.readouterr().err

    def test_config_from_args_accepts_mujoco_wasm_notebook_options(self, tmp_path):
        model = tmp_path / "scene.xml"
        asset_dir = tmp_path / "assets"
        args = build_parser().parse_args(
            [
                "--ckpt",
                str(tmp_path / "policy.pt"),
                "--policy",
                "project.policy:make_policy",
                "--env",
                "project.env:make_env",
                "--format",
                "ipynb",
                "--notebook-render-backend",
                "mujoco-wasm",
                "--mujoco-model-path",
                str(model),
                "--mujoco-asset-paths",
                str(asset_dir),
                "--mujoco-qpos-key",
                "next.qpos",
                "--notebook-viewer-port",
                "5180",
            ]
        )
        config = config_from_args(args)
        assert config.notebook_render_backend == "mujoco_wasm"
        assert config.mujoco_model_path == model
        assert config.mujoco_asset_paths == [asset_dir]
        assert config.mujoco_qpos_key == ("next", "qpos")
        assert config.notebook_viewer_port == 5180


class TestRenderCheckpoint:
    def test_checkpoint_helpers(self, tmp_path):
        path = tmp_path / "policy.pt"
        payload = {"model_state_dict": {"bias": torch.ones(1)}}
        torch.save(payload, path)
        assert load_checkpoint(path)["model_state_dict"]["bias"].item() == 1
        assert len(checkpoint_hash(path)) == 64
        assert infer_state_dict(payload)["bias"].item() == 1
        with pytest.raises(ValueError, match="Only local checkpoint"):
            load_checkpoint("https://example.com/policy.pt")


class TestRenderPolicy:
    def test_tensordict_policy_adapter_nested_action_key(self):
        adapter = TensorDictPolicyAdapter(
            tensor_only_policy,
            obs_key=("agent", "obs"),
            action_key=("agent", "action"),
        )
        td = TensorDict(
            {"agent": TensorDict({"obs": torch.zeros(1)}, batch_size=[])}, batch_size=[]
        )
        out = adapter(td)
        assert torch.equal(out.get(("agent", "action")), torch.ones(1))


class TestRenderRollouts:
    def test_make_env_policy_and_collect_rollouts(self, tmp_path):
        ckpt = tmp_path / "policy.pt"
        policy = ZeroPolicy()
        policy.bias.data.fill_(0.5)
        torch.save({"model_state_dict": policy.state_dict()}, ckpt)
        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=3,
            format="jsonl",
        )
        env = make_render_env(config)
        policy = load_render_policy(config, env)
        result = collect_render_rollouts(env, policy, config)
        assert len(result.trajectories) == 1
        assert result.trajectories[0].shape[0] == 2
        assert result.frames[0][0].frames["default"].shape == (4, 4, 3)
        assert result.metadata["trajectories"][0]["return"] == 2.0
        assert torch.equal(
            result.trajectories[0].get("action"), torch.full((2, 1), 0.5)
        )
        assert "next" in result.trajectories[0].keys()
        assert "count" in result.trajectories[0].keys()
        assert "pixels" not in result.trajectories[0].keys(True)
        assert ("next", "pixels") not in result.trajectories[0].keys(True)


class TestRenderArtifacts:
    def test_render_policy_writes_jsonl(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "policy.pt"
        torch.save({}, ckpt)
        out = tmp_path / "events.jsonl"
        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=3,
            format="jsonl",
            out=out,
            auto_load_policy=False,
            overwrite=True,
        )
        hash_calls = 0
        original_checkpoint_hash = checkpoint_hash

        def counted_checkpoint_hash(path):
            nonlocal hash_calls
            hash_calls += 1
            return original_checkpoint_hash(path)

        monkeypatch.setattr(render_module, "checkpoint_hash", counted_checkpoint_hash)
        monkeypatch.setattr(
            artifacts_module, "checkpoint_hash", counted_checkpoint_hash
        )
        result = render_policy(config)
        assert result.artifact_path == out
        lines = out.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["type"] == "metadata"
        metadata = json.loads(out.with_suffix(".jsonl.metadata.json").read_text())
        assert metadata["checkpoint"]["sha256"] == checkpoint_hash(ckpt)
        assert hash_calls == 1

    def test_write_npz_and_notebook_artifacts(self, tmp_path):
        ckpt = tmp_path / "policy.pt"
        torch.save({}, ckpt)
        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=2,
            format="npz",
            out=tmp_path / "rollouts.npz",
            auto_load_policy=False,
            overwrite=True,
        )
        env = make_render_env(config)
        policy = load_render_policy(config, env)
        result = collect_render_rollouts(env, policy, config)
        written = write_render_artifact(result, config)
        assert written.artifact_path.exists()
        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=2,
            format="ipynb",
            out=tmp_path / "report.ipynb",
            auto_load_policy=False,
            overwrite=True,
        )
        env = make_render_env(config)
        policy = load_render_policy(config, env)
        result = collect_render_rollouts(env, policy, config)
        written = write_render_artifact(result, config)
        notebook = json.loads(written.artifact_path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        namespace = {}
        exec("".join(notebook["cells"][2]["source"]), namespace)
        assert namespace["render_config"]["artifact_dir"] is None
        assert (tmp_path / "report" / "metadata.json").exists()
        saved_rollout = torch.load(
            tmp_path / "report" / "rollouts" / "traj_000.pt",
            weights_only=False,
        )
        assert "pixels" not in saved_rollout.keys(True)
        assert ("next", "pixels") not in saved_rollout.keys(True)

    @pytest.mark.skipif(
        not _has_pil, reason="Pillow is required for PNG frame artifacts"
    )
    def test_frames_artifact_optional_dependency(self, tmp_path):
        ckpt = tmp_path / "policy.pt"
        torch.save({}, ckpt)
        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=2,
            format="frames",
            out=tmp_path / "frames",
            auto_load_policy=False,
            overwrite=True,
        )
        result = render_policy(config)
        assert result.frame_paths
        assert result.frame_paths[0].suffix == ".png"

    @pytest.mark.skipif(not _has_pil, reason="Pillow is required for GIF artifacts")
    def test_gif_artifact_optional_dependency(self, tmp_path):
        ckpt = tmp_path / "policy.pt"
        torch.save({}, ckpt)
        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=2,
            format="gif",
            out=tmp_path / "render.gif",
            auto_load_policy=False,
            overwrite=True,
        )
        result = render_policy(config)
        assert result.artifact_path.exists()

    @pytest.mark.skipif(
        not _has_torchcodec, reason="torchcodec is required for MP4 artifacts"
    )
    def test_mp4_artifact_optional_dependency(self, tmp_path):
        ckpt = tmp_path / "policy.pt"
        torch.save({}, ckpt)
        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=2,
            format="mp4",
            out=tmp_path / "render.mp4",
            auto_load_policy=False,
            overwrite=True,
        )
        result = render_policy(config)
        assert result.artifact_path.exists()


class TestRenderVideo:
    def test_frame_normalization_and_grid(self):
        frames = normalize_frame_output(torch.zeros(3, 4, 4, dtype=torch.uint8))
        assert frames["default"].shape == (4, 4, 3)
        grid = compose_frame_grid(
            [frames["default"], np.ones((4, 4, 3), dtype=np.uint8)]
        )
        assert grid.shape == (4, 8, 3)


class TestMujocoWasm:
    def test_mujoco_wasm_viewer_assets_and_notebook_cells(self, tmp_path):
        ckpt = tmp_path / "policy.pt"
        torch.save({}, ckpt)
        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        model_path = scene_dir / "scene.xml"
        model_path.write_text("<mujoco/>", encoding="utf-8")
        robot_path = scene_dir / "robot.xml"
        robot_path.write_text("<mujoco/>", encoding="utf-8")
        asset_dir = scene_dir / "assets"
        asset_dir.mkdir()
        mesh_path = asset_dir / "mesh.obj"
        mesh_path.write_text("# mesh\n", encoding="utf-8")

        viewer_dir = write_mujoco_wasm_viewer(tmp_path / "viewer", model_path)
        manifest = json.loads(
            (viewer_dir / "public" / "scene" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["scene_file"] == "scene.xml"
        assert "robot.xml" in manifest["files"]
        assert "assets/mesh.obj" in manifest["files"]
        assert (viewer_dir / "src" / "main.js").exists()
        package_json = json.loads((viewer_dir / "package.json").read_text())
        assert "strictPort" not in package_json["scripts"]["dev"]
        viewer_js = (viewer_dir / "src" / "main.js").read_text()
        assert "playTrajectoryFromUrl" in viewer_js

        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=2,
            format="ipynb",
            out=tmp_path / "wasm_report.ipynb",
            auto_load_policy=False,
            overwrite=True,
            notebook_render_backend="mujoco_wasm",
            mujoco_model_path=model_path,
            mujoco_qpos_key="qpos",
        )
        env = make_render_env(config)
        policy = load_render_policy(config, env)
        result = collect_render_rollouts(env, policy, config)
        written = write_render_artifact(result, config)
        notebook = json.loads(written.artifact_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert "torchrl.render.mujoco_wasm" in source
        assert "display_mujoco_wasm_viewer" in source
        assert "torchrl_mujoco_wasm_port" in source
        assert "viewer_dir=viewer_dir" not in source
        assert "    wait=True,\n" in source
        metadata = json.loads((tmp_path / "wasm_report" / "metadata.json").read_text())
        assert metadata["mujoco_wasm"]["viewer_dir"] == "mujoco_wasm"

    def test_display_mujoco_wasm_viewer_falls_back_from_busy_port(
        self, monkeypatch, tmp_path
    ):
        display_module = FakeDisplayModule()
        process = FakeProcess()
        starts = []
        monkeypatch.setattr(
            mujoco_wasm_module, "_ipython_display_module", lambda: display_module
        )
        monkeypatch.setattr(mujoco_wasm_module, "_find_package_manager", lambda: "npm")
        waits = []
        monkeypatch.setattr(
            mujoco_wasm_module,
            "_wait_for_viewer_server",
            lambda process, host, port, timeout: waits.append(
                (process, host, port, timeout)
            ),
        )
        monkeypatch.setattr(
            mujoco_wasm_module,
            "_start_vite",
            lambda manager, viewer_dir, host, port: starts.append(
                (manager, viewer_dir, host, port)
            )
            or process,
        )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            busy_port = int(server.getsockname()[1])
            returned = display_mujoco_wasm_viewer(
                tmp_path, port=busy_port, install=False
            )
        assert returned is process
        assert starts[0][3] != busy_port
        assert waits == [(process, "127.0.0.1", starts[0][3], 20.0)]
        assert process.torchrl_mujoco_wasm_port == starts[0][3]
        assert display_module.objects[0].src == process.torchrl_mujoco_wasm_origin

    def test_mujoco_wasm_viewer_startup_failure_raises(self, monkeypatch, tmp_path):
        display_module = FakeDisplayModule()
        process = FakeProcess(return_code=1)
        monkeypatch.setattr(
            mujoco_wasm_module, "_ipython_display_module", lambda: display_module
        )
        monkeypatch.setattr(mujoco_wasm_module, "_find_package_manager", lambda: "npm")
        monkeypatch.setattr(
            mujoco_wasm_module,
            "_start_vite",
            lambda manager, viewer_dir, host, port: process,
        )
        with pytest.raises(RuntimeError, match="exited before it was ready"):
            display_mujoco_wasm_viewer(
                tmp_path,
                port=0,
                install=False,
                startup_timeout=0.1,
            )
        assert process.terminated is True
        assert display_module.objects == []

    def test_play_mujoco_wasm_trajectory_autoplay_writes_public_asset(
        self, monkeypatch, tmp_path
    ):
        display_module = FakeDisplayModule()
        monkeypatch.setattr(
            mujoco_wasm_module, "_ipython_display_module", lambda: display_module
        )
        info = play_mujoco_wasm_trajectory(
            [[0.0], [1.0]],
            fps=12.0,
            port=5181,
            viewer_dir=tmp_path / "viewer",
        )
        trajectory_files = list(
            (tmp_path / "viewer" / "public" / "trajectories").glob("trajectory_*.json")
        )
        assert len(trajectory_files) == 1
        payload = json.loads(trajectory_files[0].read_text())
        assert payload["qpos"] == [[0.0], [1.0]]
        assert payload["fps"] == 12.0
        assert info["ok"] is True
        assert info["trajectory_path"] == str(trajectory_files[0])
        assert info["viewer_url"].startswith("http://127.0.0.1:5181/?")
        assert display_module.objects[0].src == info["viewer_url"]

        with pytest.raises(ValueError, match="fps must be positive"):
            play_mujoco_wasm_trajectory([[0.0]], fps=0)
        with pytest.raises(ValueError, match="same length"):
            play_mujoco_wasm_trajectory([[0.0], [0.0, 1.0]])

    def test_send_mujoco_wasm_qpos(self, monkeypatch):
        calls = []
        acknowledgement = {"ok": True}

        def send(payload, **kwargs):
            calls.append((payload, kwargs))
            return acknowledgement

        monkeypatch.setattr(mujoco_wasm_module, "_send_viewer_message", send)
        result = mujoco_wasm_module.send_mujoco_wasm_qpos(
            [1.0, 2.0],
            viewer_origin="http://127.0.0.1:5178/",
            degrees=True,
            pause=False,
            wait=True,
            timeout=3.0,
        )
        assert result is acknowledgement
        assert calls == [
            (
                {
                    "type": "torchrl:mujoco_wasm:setQpos",
                    "qpos": [1.0, 2.0],
                    "degrees": True,
                    "pause": False,
                },
                {
                    "viewer_origin": "http://127.0.0.1:5178",
                    "wait": True,
                    "wait_for": "torchrl:mujoco_wasm:qposApplied",
                    "timeout": 3.0,
                },
            )
        ]

    def test_extract_qpos_trajectory(self):
        rollout = TensorDict({"qpos": torch.arange(6).reshape(2, 3)}, batch_size=[2])
        assert extract_qpos_trajectory(rollout, "qpos") == [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ]


class TestSotaCheckpointFactories:
    def test_dqn_cartpole_checkpoint_render_factories(self, tmp_path, monkeypatch):
        OmegaConf = pytest.importorskip("omegaconf").OmegaConf
        dqn_dir = Path("sota-implementations/dqn").resolve()
        utils_path = dqn_dir / "utils_cartpole.py"
        make_dqn_model = import_from_string(f"{utils_path}:make_dqn_model")
        checkpoint_path = tmp_path / "dqn_cartpole.pt"
        model = make_dqn_model("CartPole-v1", device="cpu")
        torch.save(
            {"model_state_dict": model.state_dict(), "env_name": "CartPole-v1"},
            checkpoint_path,
        )
        config = RenderConfig(
            ckpt=checkpoint_path,
            policy=f"{utils_path}:make_render_policy",
            env=f"{utils_path}:make_render_env",
            from_pixels=True,
            max_steps=3,
            num_trajs=1,
            format="npz",
            out=tmp_path / "dqn_cartpole.npz",
            overwrite=True,
        )
        env = make_render_env(config)
        try:
            policy = load_render_policy(config, env)
            result = collect_render_rollouts(env, policy, config)
        finally:
            env.close()
        assert result.frames[0][0].frames["default"].shape == (240, 320, 3)
        assert result.metadata["trajectories"][0]["num_frames"] > 0

        monkeypatch.syspath_prepend(str(dqn_dir))
        save_checkpoint = import_from_string(
            f"{dqn_dir / 'dqn_cartpole.py'}:_save_checkpoint"
        )
        saved_path = tmp_path / "saved_dqn.pt"
        save_checkpoint(
            saved_path,
            cfg=OmegaConf.create({"env": {"env_name": "CartPole-v1"}}),
            model=model,
            collected_frames=12,
            metrics={"eval/reward": 10.0},
        )
        saved = torch.load(saved_path, weights_only=False)
        assert saved["frames"] == 12
        assert saved["env_name"] == "CartPole-v1"
        assert "model_state_dict" in saved

        make_render_policy = import_from_string(f"{utils_path}:make_render_policy")
        calls = []

        def make_model(env_name, device):
            calls.append((env_name, device))
            return model

        monkeypatch.setitem(
            make_render_policy.__globals__, "make_dqn_model", make_model
        )
        policy = make_render_policy(
            RenderPolicySpec(
                checkpoint_path,
                {"env_name": "CartPole-v1"},
                None,
                torch.device("cpu"),
                None,
                {"env_name": "Acrobot-v1"},
                config,
            )
        )
        assert policy is model
        assert calls == [("Acrobot-v1", torch.device("cpu"))]


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
