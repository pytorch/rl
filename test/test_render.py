# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np
import pytest
import torch
import torchrl.render as render_module
import torchrl.render.artifacts as artifacts_module
from tensordict import TensorDict

from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.record.loggers.common import _has_torchcodec
from torchrl.render import (
    call_with_supported_kwargs,
    checkpoint_hash,
    collect_render_rollouts,
    import_from_string,
    infer_state_dict,
    load_checkpoint,
    load_render_policy,
    make_render_env,
    parse_nested_key,
    render_policy,
    RenderConfig,
    TensorDictPolicyAdapter,
    write_render_artifact,
)
from torchrl.render.cli import build_parser, config_from_args, main as cli_main
from torchrl.render.video import compose_frame_grid, normalize_frame_output

_has_pil = importlib.util.find_spec("PIL") is not None


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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
