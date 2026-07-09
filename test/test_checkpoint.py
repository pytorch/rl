# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import json
import random
import zipfile

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.checkpoint import (
    Checkpoint,
    CheckpointAdapter,
    CheckpointError,
    CheckpointOptions,
    GlobalRNGState,
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

    def load(self, component, path, *, map_location, args, kwargs):
        del map_location, args, kwargs
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
    assert result.manifest["metadata"] == {"run": "test"}
    assert result.manifest["components"]["policy"]["files"] == ["state.pt"]
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
    monkeypatch.setattr(Checkpoint, "_format_migrations", {0: migrate_v0_manifest})
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
