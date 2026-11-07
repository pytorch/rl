# LeRobotDatasetAdapter

*class*torchrl.data.datasets.LeRobotDatasetAdapter(***, *repo_id: str | None = None*, *root: str | Path | None = None*, *fps: int | None = None*, *features: Mapping[str, dict[str, Any]] | None = None*, *key_map: Mapping[str, [NestedKey](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NestedKey.html#tensordict.NestedKey)] | None = None*, *task: str | None = None*, *task_key: [NestedKey](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NestedKey.html#tensordict.NestedKey) | None = None*, *robot_type: str | None = None*, *batch_size: int = 256*, *video_codec: str = 'h264'*)[[source]](../../_modules/torchrl/data/datasets/lerobot_adapter.html#LeRobotDatasetAdapter)

Exchange trajectory data between TorchRL and LeRobot.

Use this adapter to save demonstrations or replay data as LeRobot datasets,
or to read existing LeRobot recordings as TensorDict frames.

Export writes the fields selected by `features`, including optional
images or videos, to a new local dataset. The returned native
`LeRobotDataset` can be uploaded with its `push_to_hub` method.

Import returns a lazy dataset that reads frames through LeRobot, preserving
its formatting and transforms. Use it with a PyTorch DataLoader and
[`tensordict_collate()`](torchrl.data.tensordict_collate.html#torchrl.data.tensordict_collate) to read TensorDict batches.

For export, provide `repo_id`, `fps` and `features`. Frames
must be grouped by episode, with consecutive steps within each episode.
[`export()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.export) can order them using
`trajectory_key` and `step_key`.

Each export requires a new destination directory. Retained episode fragments
are renumbered from frame zero, with timestamps generated from `fps`.
Next observations and terminal flags must be supplied explicitly.

For import only, construct the adapter without export settings and call
`from_external()` with a finalized `LeRobotDataset`. Temporal windows
configured through `delta_timestamps` are not supported.

Keyword Arguments:

- **repo_id** (*str**or**None**,**optional*) - Hub repository identifier, e.g.
`"user/robot-demos"`. Defaults to `None`; required for export.
Used as metadata and does not trigger an upload.
- **root** (*str**,**Path**or**None**,**optional*) - destination directory passed to
LeRobot. Defaults to `None`, using LeRobot's default location.
- **fps** (*int**or**None**,**optional*) - positive recording frame rate. Defaults
to `None`; required for export.
- **features** (*Mapping**[**str**,**dict**] or**None**,**optional*) - LeRobot schema mapping
column names to their feature definitions. Only these columns are
exported; LeRobot adds its generated metadata. Defaults to `None`;
required for export. An empty mapping exports only generated metadata.
Camera tensors may use CHW or HWC layout. CHW tensors are permuted
when the declared feature shape is HWC.
- **key_map** (*Mapping**[**str**,**NestedKey**] or**None**,**optional*) - fixed overrides from
LeRobot column names to source and imported TensorDict keys. Other
columns use the canonical mapping of
[`lerobot_columns_to_tensordict()`](torchrl.data.datasets.lerobot_columns_to_tensordict.html#torchrl.data.datasets.lerobot_columns_to_tensordict).
Defaults to `None`. Duplicate or overlapping target keys are refused.
The source episode ID and step number use the mappings for
`episode_index` and `frame_index`, respectively (by default
`"episode"` and `"frame"`). These are not exported as features.
- **task** (*str**or**None**,**optional*) - fixed language instruction for every
exported frame. Defaults to `None`; export requires either
`task` or `task_key`.
- **task_key** (*NestedKey**or**None**,**optional*) - source string instruction key,
overriding `task` per frame when set. Defaults to `None`.
- **robot_type** (*str**or**None**,**optional*) - robot name stored in metadata.
Defaults to `None`.
- **batch_size** (*int**,**optional*) - maximum frames fetched at once. Defaults
to 256. Reduce for large cameras to limit host staging memory.
- **video_codec** (*str**,**optional*) - native LeRobot video encoder name.
Defaults to `"h264"` for browser playback.
Choose another supported codec if your PyAV build lacks H.264.

Examples

```
>>> import tempfile
>>> import torch
>>> from pathlib import Path
>>> from tensordict import TensorDict
>>> from torchrl.data import ReplayBuffer, LazyTensorStorage
>>> from torchrl.data.datasets import LeRobotDatasetAdapter
>>> replay = ReplayBuffer(storage=LazyTensorStorage(4))
>>> _ = replay.extend(TensorDict({
... "action": torch.arange(4, dtype=torch.float32).unsqueeze(-1),
... "episode": torch.tensor([1, 0, 1, 0]),
... "frame": torch.tensor([1, 1, 0, 0]),
... }, [4]))
>>> with tempfile.TemporaryDirectory() as directory:
... adapter = LeRobotDatasetAdapter(
... repo_id="local/example", root=Path(directory) / "dataset", fps=30,
... features={"action": {"dtype": "float32", "shape": (1,)}},
... task="Move forward.",
... )
... native = replay.export(adapter, trajectory_key="episode", step_key="frame")
... actions = [adapter.from_external(native)[i]["action"].item() for i in range(4)]
>>> actions
[3.0, 1.0, 2.0, 0.0]
>>> native.push_to_hub(private=True)
```

from_external(*dataset: Any*, */*) → [Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/data/datasets/lerobot_adapter.html#LeRobotDatasetAdapter.from_external)

Expose native LeRobot frames through a lazy TensorDict dataset.

Parameters:

**dataset** (*lerobot.datasets.lerobot_dataset.LeRobotDataset*) - finalized
native dataset with no `delta_timestamps` windows. Its numeric
formatting, camera decoding options and image transforms apply
on access. Keep the dataset and its backing files fixed.

Returns:

one unbatched TensorDict per native frame,
with canonical or explicitly mapped keys and decoded camera tensors.

Return type:

Dataset[TensorDictBase]

to_external(*dataset: [Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, */*) → Any[[source]](../../_modules/torchrl/data/datasets/lerobot_adapter.html#LeRobotDatasetAdapter.to_external)

Write ordered TensorDict frames to a new finalized LeRobot dataset.

Parameters:

**dataset** (*Dataset**[**TensorDictBase**]*) - finite map-style source with
unbatched frames, ordered by episode ID and consecutive step.

Returns:

finalized native
dataset backed by `root`. No source tensors are retained.

Return type:

lerobot.datasets.lerobot_dataset.LeRobotDataset