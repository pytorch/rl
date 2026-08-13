# CheckpointRotation

*class*torchrl.checkpoint.CheckpointRotation(*directory: str | Path*, ***, *keep_last: int*, *keep_best: tuple[str, Literal['min', 'max']] | None = None*, *prefix: str = 'checkpoint'*)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointRotation)

Manage a directory of retained TorchRL checkpoints.

Parameters:

- **directory** - Directory containing the rotated checkpoints.
- **keep_last** - Number of newest checkpoints to retain.
- **keep_best** - Optional `(metadata_key, mode)` pair. The best checkpoint
is retained in addition to the newest checkpoints.
- **prefix** - Filename prefix for checkpoint entries.

Examples

```
>>> import tempfile
>>> from torchrl.checkpoint import Checkpoint, CheckpointRotation
>>> with tempfile.TemporaryDirectory() as tmpdir:
... checkpoint = Checkpoint(value={"step": 1})
... rotation = CheckpointRotation(tmpdir, keep_last=2)
... path = rotation.save(checkpoint, step=1)
... rotation.latest() == path
True
```

best() → Path | None[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointRotation.best)

Return the best recognized checkpoint, if configured and available.

checkpoints() → tuple[Path, ...][[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointRotation.checkpoints)

Return recognized checkpoints ordered by step.

latest() → Path | None[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointRotation.latest)

Return the newest recognized checkpoint, if any.

load_latest(*checkpoint: [Checkpoint](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint)*, ***, *components: Collection[str] | None = None*, *component_options: Mapping[str, [CheckpointOptions](torchrl.checkpoint.CheckpointOptions.html#torchrl.checkpoint.CheckpointOptions)] | None = None*, *map_location: Any = None*, *tensor_load_kwargs: Mapping[str, Any] | None = None*, *strict: Literal['error', 'warn', 'ignore'] | None = None*) → [CheckpointLoadResult](torchrl.checkpoint.CheckpointLoadResult.html#torchrl.checkpoint.CheckpointLoadResult)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointRotation.load_latest)

Restore the newest recognized checkpoint.

prune() → tuple[Path, ...][[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointRotation.prune)

Apply the retention policy and return the removed paths.

save(*checkpoint: [Checkpoint](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint)*, ***, *step: int*, *metadata: Mapping[str, Any] | None = None*, *components: Collection[str] | None = None*, *component_options: Mapping[str, [CheckpointOptions](torchrl.checkpoint.CheckpointOptions.html#torchrl.checkpoint.CheckpointOptions)] | None = None*) → Path[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointRotation.save)

Save a checkpoint at `step` and apply the retention policy.