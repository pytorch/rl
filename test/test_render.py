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
import torchrl.render.mujoco_wasm as mujoco_wasm_module
from omegaconf import OmegaConf
from tensordict import TensorDict

from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, ObservationNorm, StepCounter, VecNorm
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
    RenderEnvSpec,
    save_render_checkpoint,
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


def raising_env_factory(spec):
    raise AssertionError("env factory should not run")


def raising_policy_factory(spec):
    raise AssertionError("policy factory should not run")


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
                "--notebook-rollout-mode",
                "live",
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
        assert config.notebook_rollout_mode == "live"
        assert config.save_rollout is False
        assert config.mujoco_model_path == model
        assert config.mujoco_asset_paths == [asset_dir]
        assert config.mujoco_qpos_key == ("next", "qpos")
        assert config.notebook_viewer_port == 5180

    def test_cli_dry_run_from_config_file(self, tmp_path, capsys):
        config_path = tmp_path / "render_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "ckpt": str(tmp_path / "missing_policy.pt"),
                    "policy": "missing_module:make_policy",
                    "env": "missing_module:make_env",
                    "max_steps": 1,
                    "dry_run": True,
                }
            ),
            encoding="utf-8",
        )
        # A clobbered dry_run would run the full render and fail on the missing
        # checkpoint and factories with exit code 2.
        assert cli_main(["--config", str(config_path)]) == 0
        assert '"dry_run": true' in capsys.readouterr().out


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

    def test_save_render_checkpoint(self, tmp_path):
        module = torch.nn.Linear(2, 2)
        assert save_render_checkpoint(None, module) is None
        assert save_render_checkpoint("", module) is None
        path = save_render_checkpoint(
            tmp_path / "checkpoints" / "policy.pt",
            module,
            env_metadata={"env_name": "CartPole-v1", "vecnorm": None},
            frames=7,
            metrics={"eval/reward": 1.0},
            config={"env": {"env_name": "CartPole-v1"}},
        )
        payload = load_checkpoint(path)
        assert payload["env_name"] == "CartPole-v1"
        assert payload["vecnorm"] is None
        assert payload["frames"] == 7
        assert payload["metrics"] == {"eval/reward": 1.0}
        assert infer_state_dict(payload)["weight"].shape == (2, 2)


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

    def test_collect_rollouts_frame_alignment(self, tmp_path):
        ckpt = tmp_path / "policy.pt"
        torch.save({}, ckpt)
        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=5,
            format="jsonl",
            auto_load_policy=False,
        )
        env = make_render_env(config)
        policy = load_render_policy(config, env)
        result = collect_render_rollouts(env, policy, config)
        frames = result.frames[0]
        # TinyRenderEnv terminates after two steps; frames span the initial state
        # through the terminal state.
        assert result.trajectories[0].shape[-1] == 2
        assert len(frames) == 3
        assert int(frames[0].frames["default"][0, 0, 0]) == 0
        assert int(frames[1].frames["default"][0, 0, 0]) == 1
        assert int(frames[-1].frames["default"][0, 0, 0]) == 2


class TestRenderArtifacts:
    def test_render_policy_writes_jsonl(self, tmp_path):
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
        result = render_policy(config)
        assert result.artifact_path == out
        lines = out.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["type"] == "metadata"
        metadata = json.loads(out.with_suffix(".jsonl.metadata.json").read_text())
        assert metadata["checkpoint"]["sha256"] == checkpoint_hash(ckpt)

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

    @pytest.mark.skipif(not _has_pil, reason="Pillow is required for GIF artifacts")
    def test_gif_camera_layout_separate(self, tmp_path):
        ckpt = tmp_path / "policy.pt"
        torch.save({}, ckpt)
        config = RenderConfig(
            ckpt=ckpt,
            policy=tiny_policy_factory,
            env=tiny_env_factory,
            max_steps=2,
            num_trajs=2,
            format="gif",
            out=tmp_path / "render.gif",
            camera_layout="separate",
            auto_load_policy=False,
            overwrite=True,
        )
        result = render_policy(config)
        names = sorted(path.name for path in result.frame_paths)
        assert names == ["render_traj_000_default.gif", "render_traj_001_default.gif"]
        assert all((tmp_path / name).exists() for name in names)


class TestRenderVideo:
    def test_frame_normalization_and_grid(self):
        frames = normalize_frame_output(torch.zeros(3, 4, 4, dtype=torch.uint8))
        assert frames["default"].shape == (4, 4, 3)
        grid = compose_frame_grid(
            [frames["default"], np.ones((4, 4, 3), dtype=np.uint8)]
        )
        assert grid.shape == (4, 8, 3)


class TestRenderNotebook:
    def test_live_notebook_defers_rollout_collection(self, tmp_path):
        ckpt = tmp_path / "policy.pt"
        torch.save({}, ckpt)
        config = RenderConfig(
            ckpt=ckpt,
            policy=raising_policy_factory,
            env=raising_env_factory,
            max_steps=2,
            format="ipynb",
            out=tmp_path / "live_report.ipynb",
            notebook_rollout_mode="live",
            auto_load_policy=False,
            overwrite=True,
        )
        result = render_policy(config)
        notebook = json.loads(result.artifact_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert "collect_rollouts_in_notebook" in source
        assert "live_result = collect_rollouts_in_notebook()" in source
        assert not (tmp_path / "live_report" / "rollouts").exists()
        metadata = json.loads((tmp_path / "live_report" / "metadata.json").read_text())
        assert metadata["num_trajs"] == 0
        assert metadata["requested_num_trajs"] == 1
        assert metadata["config"]["notebook_rollout_mode"] == "live"


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

    def test_extract_qpos_trajectory(self):
        expected = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        rollout = TensorDict({"qpos": torch.arange(6).reshape(2, 3)}, batch_size=[2])
        assert extract_qpos_trajectory(rollout, "qpos") == expected
        # Saved trajectories hold the "next" contents at the root, so a
        # "next"-prefixed key falls back to the root spelling and vice versa.
        assert extract_qpos_trajectory(rollout, "next.qpos") == expected
        nested = TensorDict(
            {"next": {"qpos": torch.arange(6).reshape(2, 3)}}, batch_size=[2]
        )
        assert extract_qpos_trajectory(nested, "qpos") == expected
        with pytest.raises(KeyError, match="not found"):
            extract_qpos_trajectory(rollout, "missing_qpos")


class TestSotaCheckpointFactories:
    def test_dqn_cartpole_checkpoint_render_factories(self, tmp_path, monkeypatch):
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

    @pytest.mark.skipif(
        importlib.util.find_spec("mujoco") is None,
        reason="MuJoCo is required for the PPO render factory integration test",
    )
    def test_ppo_inverted_pendulum_checkpoint_render_factories(
        self, tmp_path, monkeypatch
    ):
        ppo_dir = Path("sota-implementations/ppo").resolve()
        utils_path = ppo_dir / "utils_mujoco.py"
        make_ppo_models = import_from_string(f"{utils_path}:make_ppo_models")
        checkpoint_path = tmp_path / "ppo_inverted_pendulum.pt"
        actor, _ = make_ppo_models(
            "InvertedPendulum-v4",
            device="cpu",
            normalize_observation=False,
        )
        torch.save(
            {
                "model_state_dict": actor.state_dict(),
                "env_name": "InvertedPendulum-v4",
                "normalize_observation": False,
            },
            checkpoint_path,
        )
        config = RenderConfig(
            ckpt=checkpoint_path,
            policy=f"{utils_path}:make_render_policy",
            env=f"{utils_path}:make_render_env",
            env_kwargs={"env_name": "InvertedPendulum-v4"},
            max_steps=3,
            num_trajs=1,
            format="npz",
            render_backend="null",
            out=tmp_path / "ppo_inverted_pendulum.npz",
            mujoco_qpos_key="qpos",
            overwrite=True,
        )
        env = make_render_env(config)
        try:
            policy = load_render_policy(config, env)
            result = collect_render_rollouts(env, policy, config)
        finally:
            env.close()
        qpos = result.trajectories[0].get("qpos")
        assert qpos.shape[-1] == 2
        assert result.metadata["trajectories"][0]["num_steps"] > 0

        monkeypatch.syspath_prepend(str(ppo_dir))
        save_checkpoint = import_from_string(
            f"{ppo_dir / 'ppo_mujoco.py'}:_save_checkpoint"
        )
        saved_path = tmp_path / "saved_ppo.pt"
        save_checkpoint(
            saved_path,
            cfg=OmegaConf.create(
                {
                    "env": {
                        "env_name": "InvertedPendulum-v4",
                        "backend": "gym",
                        "config_overrides": {},
                        "num_envs": 1,
                        "batch_mode": "parallel",
                        "normalize_observation": False,
                    }
                }
            ),
            model=actor,
            collected_frames=24,
            metrics={"eval/reward": 20.0},
        )
        saved = torch.load(saved_path, weights_only=False)
        assert saved["frames"] == 24
        assert saved["env_backend"] == "gym"
        assert saved["normalize_observation"] is False

    def test_mujoco_playground_ppo_factory(self, monkeypatch):
        utils_path = Path("sota-implementations/ppo/utils_mujoco.py").resolve()
        make_env = import_from_string(f"{utils_path}:make_env")
        module_globals = make_env.__globals__

        class FakeMujocoPlaygroundEnv:
            def __init__(self, env_name, **kwargs):
                self.env_name = env_name
                self.kwargs = kwargs

        class FakeTransformedEnv:
            def __init__(self, env):
                self.base_env = env
                self.transforms = []

            def append_transform(self, transform):
                self.transforms.append(transform)

        monkeypatch.setitem(module_globals, "_has_mujoco_playground", True)
        monkeypatch.setitem(
            module_globals, "_MujocoPlaygroundEnv", FakeMujocoPlaygroundEnv
        )
        monkeypatch.setitem(module_globals, "TransformedEnv", FakeTransformedEnv)
        monkeypatch.setitem(
            module_globals, "_observation_dtype", lambda env: torch.float32
        )

        env = make_env(
            "PandaPickCube",
            backend="mujoco_playground",
            config_overrides={"impl": "jax"},
            normalize_observation=False,
        )
        assert env.base_env.env_name == "PandaPickCube"
        assert env.base_env.kwargs["config_overrides"] == {"impl": "jax"}
        assert len(env.transforms) == 2

    def test_mujoco_playground_ppo_batch_modes(self, monkeypatch):
        utils_path = Path("sota-implementations/ppo/utils_mujoco.py").resolve()
        make_env = import_from_string(f"{utils_path}:make_env")
        module_globals = make_env.__globals__

        class FakeMujocoPlaygroundEnv:
            def __init__(self, env_name, **kwargs):
                self.env_name = env_name
                self.kwargs = kwargs

        class FakeParallelEnv:
            def __init__(self, num_workers, create_env_fn, **kwargs):
                self.num_workers = num_workers
                self.create_env_fn = create_env_fn
                self.kwargs = kwargs

        monkeypatch.setitem(module_globals, "_has_mujoco_playground", True)
        monkeypatch.setitem(
            module_globals, "_MujocoPlaygroundEnv", FakeMujocoPlaygroundEnv
        )
        monkeypatch.setitem(module_globals, "ParallelEnv", FakeParallelEnv)
        monkeypatch.setitem(
            module_globals, "_add_ppo_transforms", lambda env, **kwargs: env
        )

        vmapped = make_env(
            "PandaPickCube",
            backend="mujoco_playground",
            num_envs=4,
            batch_mode="vmap",
        )
        assert vmapped.env_name == "PandaPickCube"
        assert vmapped.kwargs["batch_size"] == [4]

        parallel = make_env(
            "PandaPickCube",
            backend="mujoco_playground",
            num_envs=3,
            batch_mode="parallel",
        )
        assert parallel.num_workers == 3
        assert parallel.kwargs["serial_for_single"] is True

    def test_mujoco_playground_ppo_uses_scalar_proof_and_eval_envs(self, monkeypatch):
        ppo_dir = Path("sota-implementations/ppo").resolve()
        utils_path = ppo_dir / "utils_mujoco.py"
        make_ppo_models = import_from_string(f"{utils_path}:make_ppo_models")
        module_globals = make_ppo_models.__globals__
        calls = []

        class FakeProofEnv:
            is_closed = False

            def close(self):
                self.is_closed = True

        proof_env = FakeProofEnv()

        def make_fake_env(*args, **kwargs):
            calls.append((args, kwargs))
            return proof_env

        monkeypatch.setitem(module_globals, "make_env", make_fake_env)
        monkeypatch.setitem(
            module_globals,
            "make_ppo_models_state",
            lambda env, device: ("actor", "critic"),
        )
        actor, critic = make_ppo_models(
            "PandaPickCube",
            "cpu",
            backend="mujoco_playground",
            num_envs=32,
            batch_mode="vmap",
        )
        assert (actor, critic) == ("actor", "critic")
        assert calls[0][1]["num_envs"] == 1
        assert calls[0][1]["batch_mode"] == "parallel"
        assert proof_env.is_closed is True

        monkeypatch.syspath_prepend(str(ppo_dir))
        make_eval_env_kwargs = import_from_string(
            f"{ppo_dir / 'ppo_mujoco.py'}:_make_eval_env_kwargs"
        )
        cfg = OmegaConf.create(
            {
                "env": {
                    "backend": "mujoco_playground",
                    "config_overrides": {},
                    "num_envs": 32,
                    "batch_mode": "vmap",
                    "normalize_observation": False,
                }
            }
        )
        assert make_eval_env_kwargs(cfg)["num_envs"] == 1
        assert make_eval_env_kwargs(cfg)["batch_mode"] == "parallel"

    def test_ppo_vecnorm_checkpoint_roundtrip(self, tmp_path):
        utils_path = Path("sota-implementations/ppo/utils_mujoco.py").resolve()
        ppo_make_env = import_from_string(f"{utils_path}:make_env")
        get_vecnorm_state = import_from_string(f"{utils_path}:get_vecnorm_state")
        train_env = ppo_make_env("CartPole-v1", normalize_observation=True)
        try:
            train_env.rollout(4)
            stats = get_vecnorm_state(train_env)
        finally:
            train_env.close()
        assert set(stats) == {"loc", "scale"}
        frozen_env = ppo_make_env(
            "CartPole-v1",
            normalize_observation=True,
            vecnorm_stats=stats,
            max_episode_steps=7,
        )
        try:
            transforms = list(frozen_env.transform)
            assert any(isinstance(item, ObservationNorm) for item in transforms)
            assert not any(isinstance(item, VecNorm) for item in transforms)
            step_counters = [
                item for item in transforms if isinstance(item, StepCounter)
            ]
            assert step_counters and step_counters[0].max_steps == 7
        finally:
            frozen_env.close()

        make_render_env_fn = import_from_string(f"{utils_path}:make_render_env")
        config = RenderConfig(
            ckpt=tmp_path / "policy.pt",
            policy="missing_module:make_policy",
            env=f"{utils_path}:make_render_env",
            max_steps=2,
            env_kwargs={"env_name": "CartPole-v1"},
        )
        spec = RenderEnvSpec.from_config(
            config,
            checkpoint={
                "env_name": "CartPole-v1",
                "normalize_observation": True,
                "vecnorm": None,
            },
        )
        with pytest.warns(UserWarning, match="VecNorm statistics"):
            env = make_render_env_fn(spec)
        env.close()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
