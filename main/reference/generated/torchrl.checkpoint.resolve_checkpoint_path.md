# resolve_checkpoint_path

torchrl.checkpoint.resolve_checkpoint_path(*path: str | Path*, ***, *prefix: str = 'checkpoint'*) → Path[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#resolve_checkpoint_path)

Return the checkpoint at `path` or the newest checkpoint of a rotation directory.

Parameters:

- **path** - A checkpoint directory or archive, or a directory managed by
[`CheckpointRotation`](torchrl.checkpoint.CheckpointRotation.html#torchrl.checkpoint.CheckpointRotation).
- **prefix** - Filename prefix of the rotated checkpoints. Defaults to
`"checkpoint"`.

Returns:

The resolved checkpoint path.

Raises:

**FileNotFoundError** - If `path` is neither a checkpoint nor a directory
 containing rotated checkpoints.

Examples

```
>>> import tempfile
>>> from torchrl.checkpoint import Checkpoint, CheckpointRotation
>>> from torchrl.checkpoint import resolve_checkpoint_path
>>> with tempfile.TemporaryDirectory() as tmpdir:
... rotation = CheckpointRotation(tmpdir, keep_last=1)
... path = rotation.save(Checkpoint(value={"step": 1}), step=1)
... resolve_checkpoint_path(tmpdir) == path
True
```