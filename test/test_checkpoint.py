# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import json
import os
import random
import zipfile

import numpy as np
import pytest
import tensordict
import torch
from tensordict import TensorDict
from tensordict.memmap import MemoryMappedTensor
from torchrl.checkpoint import (
    Checkpoint,
    CheckpointAdapter,
    CheckpointError,
    CheckpointOptions,
    CheckpointRotation,
    GlobalRNGState,
    StateDictCheckpointAdapter,
)
from torchrl.data import CompressedListStorage, ReplayBuffer


class DumpObject:
    def __init__(self, value=0):
        self.value = value
        self.calls = []

    def dump(self, path, label, *, enabled):
        self.calls.append(("dump", label, enabled))
        path.mkdir(parents=True, exist_ok=True)
        (path / "value.json").write_text(json.dumps(self.value))

    def load(self, path, label, *, enabled):
        self.calls.append(("load", label, enabled))
        self.value = json.loads((path / "value.json").read_text())


class NeverLoadDump(DumpObject):
    def load(self, path, label="unused", *, enabled=False):
        raise AssertionError("unrequested payload was loaded")


class Box:
    def __init__(self, value=0):
        self.value = value


class BoxAdapter(CheckpointAdapter):
    adapter_id = "test.box"

    def save(self, component, path, *, args, kwargs):
        del args, kwargs
        path.mkdir(parents=True, exist_ok=True)
        (path / "box.json").write_text(json.dumps(component.value))

    def load(
        self,
        component,
        path,
        *,
        map_location,
        tensor_load_kwargs,
        args,
        kwargs,
    ):
        del map_location, tensor_load_kwargs, args, kwargs
        component.value = json.loads((path / "box.json").read_text())


class FailingAdapter(BoxAdapter):
    adapter_id = "test.failing"

    def save(self, component, path, *, args, kwargs):
        super().save(component, path, args=args, kwargs=kwargs)
        raise RuntimeError("interrupted")


class VersionedBoxAdapter(BoxAdapter):
    format_version = 2


class OptionReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_calls = []

    def dump(self, path, *args, **kwargs):
        self.checkpoint_calls.append(("dump", args, kwargs))
        return super().dump(path)

    def load(self, path, *args, **kwargs):
        self.checkpoint_calls.append(("load", args, kwargs))
        return super().load(path)


def migrate_v0_manifest(manifest):
    manifest["format_version"] = 1
    manifest["metadata"]["migrated"] = True
    return manifest


def _record_checkpoint_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "torchrl.checkpoint._checkpoint.torchrl_logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    return warnings


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_state_dict_json_and_manifest_roundtrip(tmp_path, format):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(source.parameters(), lr=0.03)
    path = tmp_path / "checkpoint"
    Checkpoint(
        format=format,
        policy=source,
        optimizer=optimizer,
        metadata={"step": 12, "tags": ["unit"], "score": torch.tensor(1.5)},
    ).save(path, metadata={"run": "test"})

    target_metadata = {}
    result = Checkpoint(policy=target, metadata=target_metadata).load(
        path, components={"policy", "metadata"}, map_location="cpu"
    )

    assert result.loaded == {"policy", "metadata"}
    assert result.unrequested == {"optimizer"}
    assert target_metadata == {"step": 12, "tags": ["unit"], "score": 1.5}
    assert result.manifest["format"] == "torchrl.checkpoint"
    assert result.manifest["format_version"] == 1
    assert result.manifest["container"] == format
    assert result.manifest["versions"]["tensordict"] == tensordict.__version__
    assert result.manifest["metadata"] == {"run": "test"}
    policy_files = result.manifest["components"]["policy"]["files"]
    assert "state.json" in policy_files
    assert "state/meta.json" in policy_files
    assert any(name.endswith(".memmap") for name in policy_files)
    assert not any(name.endswith((".pt", ".pickle")) for name in policy_files)
    torch.testing.assert_close(source.weight, target.weight)


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_component_save_selection_and_restored_value(tmp_path, format):
    path = tmp_path / "checkpoint"
    Checkpoint(
        format=format,
        policy=torch.nn.Linear(2, 2),
        value=3,
        omitted={"large": True},
    ).save(path, components={"policy", "value"})
    assert set(Checkpoint.manifest(path)["components"]) == {"policy", "value"}

    result = Checkpoint(policy=torch.nn.Linear(2, 2), value=0).load(path)
    assert result.loaded == {"policy", "value"}
    assert result.values["value"] == 3


def test_default_save_component_selection(tmp_path):
    path = tmp_path / "checkpoint"
    Checkpoint(
        save_components={"policy"},
        policy=torch.nn.Linear(2, 2),
        replay_buffer={"large": True},
    ).save(path)
    assert set(Checkpoint.manifest(path)["components"]) == {"policy"}


def test_deflate_archive(tmp_path):
    path = tmp_path / "checkpoint.torchrl"
    Checkpoint(
        format="archive",
        archive_compression="deflate",
        value={"payload": "x" * 1024},
    ).save(path)
    with zipfile.ZipFile(path) as archive:
        assert all(
            member.compress_type == zipfile.ZIP_DEFLATED
            for member in archive.infolist()
        )


def test_default_state_dict_does_not_use_torch_load(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    source = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    Checkpoint(policy=source).save(path)

    def load(*args, **kwargs):
        raise AssertionError("torch.load must not be used for the default payload")

    monkeypatch.setattr(torch, "load", load)
    Checkpoint(policy=target).load(path, map_location="cpu")
    torch.testing.assert_close(source.weight, target.weight)


def test_torch_state_dict_tensor_load_options(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    source = torch.nn.Linear(2, 2)
    Checkpoint().register(
        "policy",
        source,
        adapter=StateDictCheckpointAdapter(payload_format="torch"),
    ).save(path)
    torch_load = torch.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs.copy())
        return torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    Checkpoint(policy=torch.nn.Linear(2, 2)).load(
        path,
        map_location="cpu",
        tensor_load_kwargs={"weights_only": True, "mmap": False},
    )
    assert calls == [{"weights_only": True, "mmap": False, "map_location": "cpu"}]


@pytest.mark.parametrize(
    ("payload_format", "payload_file"),
    [
        ("directory", "state/meta.json"),
        ("archive", "state.tdz"),
        ("consolidated", "state.tdc"),
        ("torch", "state.pt"),
    ],
)
def test_state_dict_payload_formats(tmp_path, payload_format, payload_file):
    path = tmp_path / "checkpoint"
    source = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    Checkpoint().register(
        "policy",
        source,
        adapter=StateDictCheckpointAdapter(payload_format=payload_format),
    ).save(path)

    files = Checkpoint.manifest(path)["components"]["policy"]["files"]
    assert payload_file in files
    Checkpoint(policy=target).load(path, map_location="cpu")
    torch.testing.assert_close(source.weight, target.weight)


def test_tensordict_optimizer_state_roundtrip(tmp_path):
    path = tmp_path / "checkpoint"
    source = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.Adam(source.parameters(), lr=0.03)
    source(torch.randn(4, 3)).sum().backward()
    source_optimizer.step()
    target = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.Adam(target.parameters(), lr=0.1)
    Checkpoint(policy=source, optimizer=source_optimizer).save(path)

    Checkpoint(policy=target, optimizer=target_optimizer).load(path, map_location="cpu")

    assert target_optimizer.param_groups[0]["lr"] == 0.03
    source_state = source_optimizer.state_dict()["state"]
    target_state = target_optimizer.state_dict()["state"]
    assert source_state.keys() == target_state.keys()
    for parameter_id in source_state:
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                source_state[parameter_id][key], target_state[parameter_id][key]
            )


def test_archive_accepts_windows_manifest_separators(tmp_path):
    path = tmp_path / "checkpoint.torchrl"
    source = torch.nn.Linear(2, 2)
    Checkpoint(format="archive", policy=source).save(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    for record in manifest["components"].values():
        record["path"] = record["path"].replace("/", "\\")
        record["files"] = [name.replace("/", "\\") for name in record["files"]]
    members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    target = torch.nn.Linear(2, 2)
    Checkpoint(policy=target).load(path)
    torch.testing.assert_close(source.weight, target.weight)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are required")
def test_directory_uses_normal_directory_permissions(tmp_path):
    path = tmp_path / "checkpoint"
    Checkpoint(value={"ok": True}).save(path)
    assert path.stat().st_mode & 0o050 == 0o050


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_dump_load_options_and_partial_restore(tmp_path, format):
    source = DumpObject(17)
    target = DumpObject()
    path = tmp_path / "checkpoint"
    saver = (
        Checkpoint(format=format, metadata={"ok": True})
        .register(
            "payload",
            source,
            options=CheckpointOptions(
                save_args=("registered",), save_kwargs={"enabled": False}
            ),
        )
        .register(
            "never",
            NeverLoadDump(99),
            options=CheckpointOptions(
                save_args=("unused",), save_kwargs={"enabled": False}
            ),
        )
    )
    saver.save(
        path,
        component_options={
            "payload": CheckpointOptions(
                save_args=("operation",), save_kwargs={"enabled": True}
            )
        },
    )
    assert source.calls == [("dump", "operation", True)]

    loader = Checkpoint().register(
        "payload",
        target,
        options=CheckpointOptions(
            load_args=("registered",), load_kwargs={"enabled": False}
        ),
    )
    result = loader.load(
        path,
        components={"payload"},
        component_options={
            "payload": CheckpointOptions(
                load_args=("operation",), load_kwargs={"enabled": True}
            )
        },
    )
    assert target.value == 17
    assert target.calls == [("load", "operation", True)]
    assert result.unrequested == {"metadata", "never"}


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_scoped_adapter(tmp_path, format):
    path = tmp_path / "checkpoint"
    source = Box(5)
    saver = Checkpoint(format=format)
    saver.register_adapter(Box, BoxAdapter()).register("box", source).save(path)

    target = Box()
    loader = Checkpoint()
    loader.register_adapter(Box, BoxAdapter()).register("box", target).load(path)
    assert target.value == 5


def test_strict_modes(tmp_path, caplog):
    path = tmp_path / "checkpoint"
    Checkpoint(policy=torch.nn.Linear(2, 2)).save(path)
    checkpoint = Checkpoint(policy=torch.nn.Linear(2, 2), optimizer={"json": "target"})
    with pytest.raises(CheckpointError, match="missing"):
        checkpoint.load(path)
    result = checkpoint.load(path, strict="ignore")
    assert result.loaded == {"policy"}
    assert result.missing == {"optimizer"}
    result = checkpoint.load(path, strict="warn")
    assert result.missing == {"optimizer"}
    assert "Checkpoint restore issues" in caplog.text


def test_matching_manifest_versions_are_reported(tmp_path, monkeypatch):
    warnings = _record_checkpoint_warnings(monkeypatch)
    path = tmp_path / "checkpoint"
    Checkpoint(value={"version": 1}).save(path)

    result = Checkpoint(value={}).load(path)

    assert result.loaded == {"value"}
    assert result.comparison["version_skew"] is False
    assert result.comparison["versions"] == {
        name: {
            "checkpoint": version,
            "runtime": version,
            "matches": True,
        }
        for name, version in result.manifest["versions"].items()
    }
    assert not warnings


@pytest.mark.parametrize("package", ["torchrl", "tensordict", "torch"])
def test_manifest_version_skew_warns_without_blocking_load(
    tmp_path, monkeypatch, package
):
    warnings = _record_checkpoint_warnings(monkeypatch)
    path = tmp_path / "checkpoint"
    Checkpoint(value={"version": 1}).save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    runtime_version = manifest["versions"][package]
    manifest["versions"][package] = "checkpoint-version"
    manifest_path.write_text(json.dumps(manifest))
    target = {}

    result = Checkpoint(value=target).load(path)

    assert result.loaded == {"value"}
    assert target == {"version": 1}
    assert result.comparison["version_skew"] is True
    assert result.comparison["versions"][package] == {
        "checkpoint": "checkpoint-version",
        "runtime": runtime_version,
        "matches": False,
    }
    assert all(
        comparison["matches"] is (name != package)
        for name, comparison in result.comparison["versions"].items()
    )
    assert len(warnings) == 1
    assert "Checkpoint dependency version skew detected" in warnings[0]
    assert package in warnings[0]
    assert "The mismatch is non-fatal" in warnings[0]


def test_manifest_without_versions_loads_silently(tmp_path, monkeypatch):
    warnings = _record_checkpoint_warnings(monkeypatch)
    path = tmp_path / "checkpoint"
    Checkpoint(value={"version": 1}).save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["versions"]
    manifest_path.write_text(json.dumps(manifest))

    result = Checkpoint(value={}).load(path)

    assert result.loaded == {"value"}
    assert result.comparison == {}
    assert not warnings


def test_incompatible_adapter(tmp_path):
    path = tmp_path / "checkpoint"
    Checkpoint(value=torch.nn.Linear(2, 2)).save(path)
    result = Checkpoint(strict="ignore", value={}).load(path)
    assert "value" in result.incompatible
    assert not result.loaded

    version_path = tmp_path / "versioned"
    Checkpoint().register("box", Box(1), adapter=BoxAdapter()).save(version_path)
    result = (
        Checkpoint(strict="ignore")
        .register("box", Box(), adapter=VersionedBoxAdapter())
        .load(version_path)
    )
    assert "adapter version" in result.incompatible["box"]


def test_manifest_migration(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    Checkpoint(value={"version": 1}).save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CheckpointError, match="newer"):
        Checkpoint.manifest(path)
    manifest["format_version"] = 0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CheckpointError, match="no migration"):
        Checkpoint.manifest(path)
    monkeypatch.setattr(Checkpoint, "_format_migrations", {})
    Checkpoint.register_migration(0, migrate_v0_manifest)
    assert Checkpoint.manifest(path)["metadata"]["migrated"] is True


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_interrupted_write_preserves_checkpoint(tmp_path, format):
    path = tmp_path / "checkpoint"
    Checkpoint(format=format, value={"version": 1}).save(path)
    checkpoint_id = Checkpoint.manifest(path)["checkpoint_id"]
    with pytest.raises(RuntimeError, match="interrupted"):
        Checkpoint(format=format).register(
            "value", Box(2), adapter=FailingAdapter()
        ).save(path)
    assert Checkpoint.manifest(path)["checkpoint_id"] == checkpoint_id
    value = {}
    Checkpoint(value=value).load(path)
    assert value == {"version": 1}


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_checkpoint_rotation_keeps_latest_and_loads_newest(tmp_path, format):
    rotation = CheckpointRotation(tmp_path / "checkpoints", keep_last=3)
    for step in range(1, 6):
        rotation.save(
            Checkpoint(format=format, value={"step": step}),
            step=step,
        )

    paths = rotation.checkpoints()
    assert [path.stem for path in paths] == [
        "checkpoint-000000000003",
        "checkpoint-000000000004",
        "checkpoint-000000000005",
    ]
    assert rotation.latest() == paths[-1]

    value = {}
    result = rotation.load_latest(Checkpoint(value=value))
    assert result.loaded == {"value"}
    assert value == {"step": 5}


@pytest.mark.parametrize("mode", ["min", "max"])
def test_checkpoint_rotation_keeps_best_outside_latest(tmp_path, mode):
    scores = [5.0, 1.0, 3.0, 4.0] if mode == "min" else [1.0, 5.0, 3.0, 2.0]
    rotation = CheckpointRotation(
        tmp_path / "checkpoints",
        keep_last=2,
        keep_best=("score", mode),
    )
    for step, score in enumerate(scores, 1):
        rotation.save(
            Checkpoint(value={"step": step}),
            step=step,
            metadata={"score": score},
        )

    assert [path.name for path in rotation.checkpoints()] == [
        "checkpoint-000000000002",
        "checkpoint-000000000003",
        "checkpoint-000000000004",
    ]
    assert rotation.best().name == "checkpoint-000000000002"
    assert rotation.latest().name == "checkpoint-000000000004"


@pytest.mark.parametrize("mode", ["min", "max"])
def test_checkpoint_rotation_best_tie_prefers_latest(tmp_path, mode):
    rotation = CheckpointRotation(
        tmp_path,
        keep_last=1,
        keep_best=("score", mode),
    )
    rotation.save(Checkpoint(value=1), step=1, metadata={"score": 2.0})
    rotation.save(Checkpoint(value=2), step=2, metadata={"score": 2.0})

    assert rotation.checkpoints() == (rotation.latest(),)
    assert rotation.best() == rotation.latest()


@pytest.mark.parametrize(
    "metadata,error",
    [
        ({}, KeyError),
        ({"score": True}, TypeError),
        ({"score": "high"}, TypeError),
        ({"score": float("nan")}, ValueError),
        ({"score": float("inf")}, ValueError),
    ],
)
def test_checkpoint_rotation_rejects_invalid_best_metric(tmp_path, metadata, error):
    rotation = CheckpointRotation(
        tmp_path / "checkpoints",
        keep_last=1,
        keep_best=("score", "max"),
    )

    with pytest.raises(error):
        rotation.save(Checkpoint(value=1), step=1, metadata=metadata)

    assert not rotation.directory.exists()


def test_checkpoint_rotation_ignores_unrecognized_entries(tmp_path):
    rotation = CheckpointRotation(tmp_path, keep_last=1)
    (tmp_path / "notes.txt").write_text("unrelated")
    (tmp_path / "checkpoint-invalid").mkdir()
    (tmp_path / "checkpoint-000000000001").mkdir()
    (tmp_path / ".checkpoint-000000000002.delete-deadbeef").mkdir()

    assert rotation.checkpoints() == ()
    assert rotation.latest() is None
    assert rotation.best() is None


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_checkpoint_rotation_failed_save_does_not_prune(tmp_path, format):
    rotation = CheckpointRotation(tmp_path, keep_last=1)
    original = rotation.save(Checkpoint(format=format, value=1), step=1)
    failing = Checkpoint(format=format).register(
        "value", Box(2), adapter=FailingAdapter()
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        rotation.save(failing, step=2)

    assert rotation.checkpoints() == (original,)
    assert Checkpoint.is_checkpoint(original)


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_checkpoint_rotation_replaces_same_step(tmp_path, format):
    rotation = CheckpointRotation(tmp_path, keep_last=2)
    path = rotation.save(Checkpoint(format=format, value={"version": 1}), step=4)
    replacement = rotation.save(Checkpoint(format=format, value={"version": 2}), step=4)

    assert replacement == path
    assert rotation.checkpoints() == (path,)
    value = {}
    rotation.load_latest(Checkpoint(value=value))
    assert value == {"version": 2}


def test_checkpoint_rotation_replaces_same_step_across_formats(tmp_path):
    rotation = CheckpointRotation(tmp_path, keep_last=2)
    directory = rotation.save(Checkpoint(value={"format": "directory"}), step=4)
    archive = rotation.save(
        Checkpoint(format="archive", value={"format": "archive"}), step=4
    )

    assert not directory.exists()
    assert rotation.checkpoints() == (archive,)
    value = {}
    rotation.load_latest(Checkpoint(value=value))
    assert value == {"format": "archive"}


@pytest.mark.parametrize(
    ("old_format", "new_format"),
    [("archive", "directory"), ("directory", "archive")],
)
def test_checkpoint_rotation_recovers_interrupted_cross_format_replacement(
    tmp_path, monkeypatch, old_format, new_format
):
    rotation = CheckpointRotation(tmp_path, keep_last=1)
    old_path = rotation.save(
        Checkpoint(format=old_format, value={"version": 1}), step=4
    )

    remove_atomically = rotation._remove_atomically

    def interrupted_remove(path):
        del path
        raise RuntimeError("interrupted")

    monkeypatch.setattr(rotation, "_remove_atomically", interrupted_remove)
    with pytest.raises(RuntimeError, match="interrupted"):
        rotation.save(
            Checkpoint(format=new_format, value={"version": 2}), step=4
        )
    monkeypatch.setattr(rotation, "_remove_atomically", remove_atomically)

    new_path = rotation._checkpoint_path(4, new_format)
    assert rotation.checkpoints() == (new_path,)
    assert rotation.latest() == new_path
    value = {}
    rotation.load_latest(Checkpoint(value=value))
    assert value == {"version": 2}
    assert rotation.prune() == (old_path,)
    assert rotation.checkpoints() == (new_path,)


def test_checkpoint_rotation_prune_returns_removed_paths(tmp_path):
    rotation = CheckpointRotation(tmp_path, keep_last=3)
    paths = [rotation.save(Checkpoint(value=step), step=step) for step in range(1, 4)]
    rotation.keep_last = 1

    assert rotation.prune() == tuple(paths[:2])
    assert rotation.checkpoints() == (paths[-1],)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"keep_last": 0}, ValueError),
        ({"keep_last": True}, TypeError),
        ({"keep_last": 1, "prefix": "bad/name"}, ValueError),
        ({"keep_last": 1, "keep_best": ("score", "median")}, ValueError),
    ],
)
def test_checkpoint_rotation_validates_configuration(tmp_path, kwargs, error):
    with pytest.raises(error):
        CheckpointRotation(tmp_path, **kwargs)


def test_checkpoint_rotation_validates_step(tmp_path):
    rotation = CheckpointRotation(tmp_path, keep_last=1)
    with pytest.raises(TypeError):
        rotation.save(Checkpoint(value=1), step=True)
    with pytest.raises(ValueError):
        rotation.save(Checkpoint(value=1), step=-1)


def test_checkpoint_rotation_load_latest_requires_checkpoint(tmp_path):
    rotation = CheckpointRotation(tmp_path, keep_last=1)
    with pytest.raises(FileNotFoundError, match="No checkpoints"):
        rotation.load_latest(Checkpoint(value={}))


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_global_rng_roundtrip(tmp_path, format):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    path = tmp_path / "checkpoint"
    Checkpoint(format=format, rng=GlobalRNGState()).save(path)
    expected = (random.random(), np.random.rand(), torch.rand(()))
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    Checkpoint(rng=GlobalRNGState()).load(path)
    actual = (random.random(), np.random.rand(), torch.rand(()))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_compressed_replay_buffer_roundtrip(tmp_path, format):
    path = tmp_path / "checkpoint"
    source = OptionReplayBuffer(storage=CompressedListStorage(10), batch_size=1)
    data = TensorDict({"observation": torch.randn(4, 3)}, [4])
    source.extend(data)
    Checkpoint(format=format).register(
        "replay_buffer",
        source,
        options=CheckpointOptions(
            save_args=("baseline",),
            save_kwargs={"enabled": False, "preserved": 1},
        ),
    ).save(
        path,
        component_options={
            "replay_buffer": CheckpointOptions(
                save_args=("operation",), save_kwargs={"enabled": True}
            )
        },
    )
    assert source.checkpoint_calls == [
        ("dump", ("operation",), {"enabled": True, "preserved": 1})
    ]

    target = OptionReplayBuffer(storage=CompressedListStorage(10), batch_size=1)
    Checkpoint().register(
        "replay_buffer",
        target,
        options=CheckpointOptions(
            load_args=("baseline",),
            load_kwargs={"enabled": False, "preserved": 2},
        ),
    ).load(
        path,
        component_options={
            "replay_buffer": CheckpointOptions(
                load_args=("operation",), load_kwargs={"enabled": True}
            )
        },
    )
    assert target.checkpoint_calls == [
        ("load", ("operation",), {"enabled": True, "preserved": 2})
    ]
    assert len(target) == len(source)
    torch.testing.assert_close(target.storage.get(0), source.storage.get(0))
    loaded_observation = target.storage._storage[0]["observation"]
    assert isinstance(loaded_observation, MemoryMappedTensor) is (format == "directory")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_map_location_and_rng(tmp_path):
    path = tmp_path / "checkpoint"
    source = torch.nn.Linear(2, 2, device="cuda")
    torch.cuda.manual_seed_all(0)
    Checkpoint(format="archive", policy=source, rng=GlobalRNGState()).save(path)
    target = torch.nn.Linear(2, 2)
    Checkpoint(policy=target).load(path, components={"policy"}, map_location="cpu")
    assert target.weight.device.type == "cpu"


def test_reject_remote_path():
    with pytest.raises(ValueError, match="Only local"):
        Checkpoint(value={}).save("s3://bucket/checkpoint")


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
