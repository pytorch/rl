# Checkpointing

TorchRL checkpoints use one manifest-driven format for standalone scripts,
trainers, and policy-only consumers. Components are registered independently,
so a checkpoint may contain only a policy or a complete training state.

The directory and archive containers share the same logical layout. Directory
checkpoints are the default and are best suited to large replay buffers;
archives are convenient single-file artifacts. Loading either container is
automatic.

TorchRL checkpoints target local filesystems. URI paths and coordinated
distributed rank checkpoints are rejected rather than importing an optional
remote-storage stack implicitly.

## Basic usage

```
from torchrl.checkpoint import Checkpoint, GlobalRNGState

checkpoint = Checkpoint(
 policy=policy,
 optimizer=optimizer,
 replay_buffer=replay_buffer,
 rng=GlobalRNGState(),
)
checkpoint.save("run/checkpoint")
checkpoint.load(
 "run/checkpoint",
 components={"policy", "optimizer", "rng"},
 map_location="cpu",
)
```

Replay buffers use their `dump` and `load` implementations, including the
configured storage checkpointer and compression. Other TorchRL and PyTorch
objects normally use `state_dict` and `load_state_dict`. Their tensor state
is stored with `tensordict.save()` by default, while a JSON schema preserves
the state-dict structure without pickle. JSON-compatible configuration,
metrics, and metadata are also stored without pickle.

Set `save_components={"policy", "optimizer", "trainer_state"}` on a
[`Checkpoint`](generated/torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint) to keep large components such as replay buffers out of
scheduled Trainer saves. An explicit `components=` argument to
[`Checkpoint.save()`](generated/torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint.save) overrides this default selection.

## State-dict payload formats

The inferred [`StateDictCheckpointAdapter`](generated/torchrl.checkpoint.StateDictCheckpointAdapter.html#torchrl.checkpoint.StateDictCheckpointAdapter) writes a TensorDict directory.
The same adapter can write a TensorDict ZIP archive or consolidated file, and
loads auto-detect all of these payloads. This component payload choice is
independent of the outer [`Checkpoint`](generated/torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint) directory or archive container.

```
from torchrl.checkpoint import Checkpoint, StateDictCheckpointAdapter

checkpoint = Checkpoint().register(
 "policy",
 policy,
 adapter=StateDictCheckpointAdapter(payload_format="archive"),
)
```

Use `payload_format="consolidated"` for consolidated TensorDict storage.
Pickle-based [`torch.save()`](https://docs.pytorch.org/docs/stable/generated/torch.save.html#torch.save) remains available explicitly with
`payload_format="torch"`. TensorDict payloads reject unsupported Python
objects with an error that points to this opt-in rather than silently falling
back to pickle.

## Custom components

Objects exposing `dump(path, ...)` and `load(path, ...)` are detected before
objects exposing `state_dict` and `load_state_dict`. A custom
[`CheckpointAdapter`](generated/torchrl.checkpoint.CheckpointAdapter.html#torchrl.checkpoint.CheckpointAdapter) can instead be supplied to
[`Checkpoint.register()`](generated/torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint.register), or registered by type on one checkpoint with
[`Checkpoint.register_adapter()`](generated/torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint.register_adapter).

Use [`CheckpointOptions`](generated/torchrl.checkpoint.CheckpointOptions.html#torchrl.checkpoint.CheckpointOptions) to preserve component-specific arguments. Options
registered with a component are the baseline; operation-level keyword arguments
override matching entries and explicitly supplied positional arguments replace
the baseline tuple.

## Compatibility

The manifest records the checkpoint format version, adapter versions, component
files, and TorchRL, TensorDict, and PyTorch versions. Newer unsupported formats
and incompatible adapters fail clearly. Partial restoration reports loaded,
missing, incompatible, and unrequested components through
[`CheckpointLoadResult`](generated/torchrl.checkpoint.CheckpointLoadResult.html#torchrl.checkpoint.CheckpointLoadResult).

Trainer's legacy `CKPT_BACKEND` path remains available during the migration
window. Passing `checkpoint=Checkpoint(...)` to a trainer opts into the
unified format. Existing torch, torchsnapshot, and memmap trainer checkpoints
remain readable.

The [`torchrl.render.save_render_checkpoint()`](generated/torchrl.render.save_render_checkpoint.html#torchrl.render.save_render_checkpoint) helper also keeps its legacy
`torch.save` payload by default during the compatibility window. Pass
`format="archive"` or `format="directory"` to opt into the unified format;
the default changes in v0.15.

## API

| [`Checkpoint`](generated/torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint)(*[, format, strict, ...]) | Standard TorchRL checkpoint coordinator. |
| --- | --- |
| [`CheckpointAdapter`](generated/torchrl.checkpoint.CheckpointAdapter.html#torchrl.checkpoint.CheckpointAdapter)() | Interface used to save and restore one checkpoint component. |
| [`CheckpointError`](generated/torchrl.checkpoint.CheckpointError.html#torchrl.checkpoint.CheckpointError)(message[, result]) | Error raised when a checkpoint cannot be saved or restored. |
| [`CheckpointLoadResult`](generated/torchrl.checkpoint.CheckpointLoadResult.html#torchrl.checkpoint.CheckpointLoadResult)(loaded, missing, ...) | Structured result returned by [`Checkpoint.load()`](generated/torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint.load). |
| [`CheckpointOptions`](generated/torchrl.checkpoint.CheckpointOptions.html#torchrl.checkpoint.CheckpointOptions)([save_args, save_kwargs, ...]) | Arguments forwarded to a component's serialization methods. |
| [`CheckpointFormat`](generated/torchrl.checkpoint.CheckpointFormat.html#torchrl.checkpoint.CheckpointFormat) | alias of `Literal`['directory', 'archive'] |
| [`CheckpointStrictness`](generated/torchrl.checkpoint.CheckpointStrictness.html#torchrl.checkpoint.CheckpointStrictness) | alias of `Literal`['error', 'warn', 'ignore'] |
| [`DumpLoadCheckpointAdapter`](generated/torchrl.checkpoint.DumpLoadCheckpointAdapter.html#torchrl.checkpoint.DumpLoadCheckpointAdapter)() | Adapter for objects exposing `dump(path)` and `load(path)`. |
| [`GlobalRNGState`](generated/torchrl.checkpoint.GlobalRNGState.html#torchrl.checkpoint.GlobalRNGState)() | Checkpointable process-global random-number-generator state. |
| [`JSONCheckpointAdapter`](generated/torchrl.checkpoint.JSONCheckpointAdapter.html#torchrl.checkpoint.JSONCheckpointAdapter)() | Adapter for JSON-compatible configuration, metrics, and metadata. |
| [`StateDictCheckpointAdapter`](generated/torchrl.checkpoint.StateDictCheckpointAdapter.html#torchrl.checkpoint.StateDictCheckpointAdapter)([payload_format, ...]) | Adapter for `state_dict` / `load_state_dict` objects. |
| [`StateDictFormat`](generated/torchrl.checkpoint.StateDictFormat.html#torchrl.checkpoint.StateDictFormat) | alias of `Literal`['directory', 'archive', 'consolidated', 'torch'] |