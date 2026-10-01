# RunCheckpointer

*class*torchrl.checkpoint.RunCheckpointer(*checkpoint: [Checkpoint](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint)*, ***, *directory: str | Path | None*, *interval: int*, *keep_last: int = 2*, *exclude: Collection[str] = ()*, *optional: Collection[str] = ('logger', 'replay_buffer')*, *resume_path: str | Path | None = None*)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#RunCheckpointer)

Save a training script's [`Checkpoint`](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint) at loop boundaries and restore it on resume.

`save(step)` writes a rotated checkpoint to `directory` every
`interval` steps and `save(step, force=True)` writes a final one when
the loop ends. `restore()` loads every component but `config` and
`rng`, then `rng` last, so random numbers drawn while the script built
its objects do not change the restored RNG state.

Parameters:

- **checkpoint** ([*Checkpoint*](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint)) - the run's components.
- **directory** (*str**or**Path**or**None*) - rotation directory for scheduled
saves. `None` disables saving. When resuming, checkpoints keep
accumulating next to the resumed checkpoint instead.
- **interval** (*int*) - steps between scheduled saves.
- **keep_last** (*int**,**optional*) - checkpoints retained by the rotation.
Defaults to `2`.
- **exclude** (*Collection**[**str**]**,**optional*) - components left out of every save,
for example `("replay_buffer",)`. Defaults to none.
- **optional** (*Collection**[**str**]**,**optional*) - components restored only when the
checkpoint holds them. Defaults to `("logger", "replay_buffer")`.
- **resume_path** (*str**or**Path**,**optional*) - checkpoint or rotation directory to
restore from, see [`resolve_checkpoint_path()`](torchrl.checkpoint.resolve_checkpoint_path.html#torchrl.checkpoint.resolve_checkpoint_path).

Examples

```
>>> import tempfile
>>> import torch
>>> from torchrl.checkpoint import Checkpoint, RunCheckpointer
>>> checkpoint = Checkpoint(policy=torch.nn.Linear(2, 1), run_state={"step": 0})
>>> with tempfile.TemporaryDirectory() as tmpdir:
... run = RunCheckpointer(checkpoint, directory=tmpdir, interval=10)
... run.save(5) is None, run.save(10) is not None
(True, True)
```

restore(***, *map_location: Any = None*) → bool[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#RunCheckpointer.restore)

Restore the run from `resume_path`; return whether anything was restored.

save(*step: int*, ***, *force: bool = False*) → Path | None[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#RunCheckpointer.save)

Save when `interval` steps have passed since the last save.

Parameters:

- **step** (*int*) - the current step, used as the rotation step.
- **force** (*bool**,**optional*) - save regardless of the interval, unless a
checkpoint was already written for this exact step. Use it
when the loop ends. Defaults to `False`.

Returns:

The written checkpoint path, or `None` when nothing was saved.