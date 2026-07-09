"""
Unified checkpointing in TorchRL
===============================

**What you will learn**

This tutorial shows how to save a standalone training state, restore selected
components, choose a directory or archive container, integrate the same object
with a Trainer, and extend checkpointing for a custom component.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from torchrl.checkpoint import (
    Checkpoint,
    CheckpointAdapter,
    CheckpointOptions,
    GlobalRNGState,
)


# Standalone training state
# -------------------------
# Components are independent. Omitting the replay buffer, optimizer, or RNG is
# valid, and restoration can select a smaller subset than was saved.

temporary_directory = tempfile.TemporaryDirectory()
root = Path(temporary_directory.name)
policy = torch.nn.Linear(4, 2)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
progress = {"frames": 128}

checkpoint = Checkpoint(
    policy=policy,
    optimizer=optimizer,
    trainer_state=progress,
    rng=GlobalRNGState(),
    config={"learning_rate": 3e-4},
)
directory_path = checkpoint.save(root / "step-128")


# Partial restoration and device remapping
# ----------------------------------------
# Only the policy file is read. Large unrequested components, such as replay
# buffers, are neither opened nor materialized.

restored_policy = torch.nn.Linear(4, 2)
load_result = Checkpoint(policy=restored_policy).load(
    directory_path,
    components={"policy"},
    map_location="cpu",
)
assert load_result.loaded == {"policy"}
assert load_result.unrequested == {
    "config",
    "optimizer",
    "rng",
    "trainer_state",
}


# Single-file archives
# --------------------
# Archives contain the same manifest and component tree. Stored ZIP entries are
# the default so a replay buffer's own compression is not repeated.

archive_path = checkpoint.save(root / "step-128.torchrl", format="archive")
archive_manifest = Checkpoint.manifest(archive_path)
assert archive_manifest["container"] == "archive"


# Trainer integration
# -------------------
# Every TorchRL algorithm trainer accepts ``checkpoint=``. The trainer adds its
# known policy, loss, optimizer, collector, replay buffer, progress, and hooks to
# registrations that the caller has not already supplied. Scheduled saves use
# this same manifest and adapter implementation.

trainer_checkpoint = Checkpoint(format="directory", rng=GlobalRNGState())
# trainer = PPOTrainer(..., checkpoint=trainer_checkpoint,
#                      save_trainer_file="run/checkpoint")


# Custom serialization
# --------------------
# ``dump`` and ``load`` objects work automatically. An adapter is useful when a
# third-party class cannot be changed or needs a different on-disk layout.


class Cursor:
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset


class CursorAdapter(CheckpointAdapter):
    adapter_id = "tutorial.cursor"

    def save(self, component, path, *, args, kwargs):
        del args, kwargs
        path.mkdir(parents=True, exist_ok=True)
        (path / "cursor.json").write_text(json.dumps(component.offset))

    def load(self, component, path, *, map_location, args, kwargs):
        del map_location, args, kwargs
        component.offset = json.loads((path / "cursor.json").read_text())


cursor_path = root / "cursor-checkpoint"
Checkpoint().register(
    "cursor",
    Cursor(7),
    adapter=CursorAdapter(),
    options=CheckpointOptions(),
).save(cursor_path)
restored_cursor = Cursor()
Checkpoint().register("cursor", restored_cursor, adapter=CursorAdapter()).load(
    cursor_path
)
assert restored_cursor.offset == 7


# Compatibility
# -------------
# Unified loads validate the manifest version and each component adapter
# version before touching payload files. Trainer checkpoints written through
# the legacy torch, torchsnapshot, and memmap backends remain readable during
# the compatibility window; pass ``checkpoint=Checkpoint(...)`` to opt into
# unified scheduled saves.


# Conclusion
# ----------
# A single Checkpoint object now covers standalone scripts and trainers while
# keeping every component optional. The manifest makes partial loading,
# compatibility reporting, and user-defined adapters explicit.


# Further reading
# ---------------
# See :class:`torchrl.checkpoint.Checkpoint`,
# :class:`torchrl.checkpoint.CheckpointAdapter`, and the replay-buffer tutorial
# for storage-specific checkpoint configuration.

temporary_directory.cleanup()
