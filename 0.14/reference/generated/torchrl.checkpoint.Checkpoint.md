# Checkpoint

*class*torchrl.checkpoint.Checkpoint(***, *format: Literal['directory', 'archive'] = 'directory'*, *strict: Literal['error', 'warn', 'ignore'] = 'error'*, *archive_compression: Literal['stored', 'deflate'] = 'stored'*, *save_components: Collection[str] | None = None*, ***components: Any*)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#Checkpoint)

Standard TorchRL checkpoint coordinator.

Components are bound to the checkpoint and may be independently selected
for each save or restore. The default directory format supports direct,
lazy access to component payloads; `format="archive"` stores the same
manifest and component tree in one ZIP file.

Parameters:

- **format** - Default output format, `"directory"` or `"archive"`.
- **strict** - Default restoration behavior for missing or incompatible
requested components.
- **archive_compression** - `"stored"` avoids recompressing payloads;
`"deflate"` enables ZIP deflate compression.
- **save_components** - Optional default component selection for saves. This
is useful for excluding large components from scheduled saves.
- ****components** - Named objects or JSON-compatible values to register.

Examples

```
>>> import tempfile
>>> import torch
>>> from torchrl.checkpoint import Checkpoint
>>> source = torch.nn.Linear(2, 1)
>>> target = torch.nn.Linear(2, 1)
>>> with tempfile.TemporaryDirectory() as tmpdir:
... Checkpoint(policy=source).save(f"{tmpdir}/checkpoint")
... result = Checkpoint(policy=target).load(f"{tmpdir}/checkpoint")
>>> result.loaded
{'policy'}
```

*property*components*: Mapping[str, Any]*

Read-only mapping view of registered component values.

*classmethod*is_checkpoint(*path: str | Path*) → bool[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#Checkpoint.is_checkpoint)

Return whether `path` contains a TorchRL checkpoint manifest.

load(*path: str | Path*, ***, *components: Collection[str] | None = None*, *component_options: Mapping[str, [CheckpointOptions](torchrl.checkpoint.CheckpointOptions.html#torchrl.checkpoint.CheckpointOptions)] | None = None*, *map_location: Any = None*, *tensor_load_kwargs: Mapping[str, Any] | None = None*, *strict: Literal['error', 'warn', 'ignore'] | None = None*) → [CheckpointLoadResult](torchrl.checkpoint.CheckpointLoadResult.html#torchrl.checkpoint.CheckpointLoadResult)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#Checkpoint.load)

Restore selected components from a local checkpoint.

Parameters:

- **path** - Directory or archive checkpoint.
- **components** - Registered names to restore. `None` selects all
registered components.
- **component_options** - Per-operation option overrides.
- **map_location** - Device mapping used while reading tensor payloads.
- **tensor_load_kwargs** - Additional keyword arguments passed to
[`torch.load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) for state-dict components explicitly saved
with the torch payload format. `weights_only` defaults to
`True`.
- **strict** - Per-operation strictness override.

Returns:

A structured component load report.

*classmethod*manifest(*path: str | Path*) → dict[str, Any][[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#Checkpoint.manifest)

Read and validate a checkpoint manifest without loading payloads.

register(*name: str*, *component: Any*, ***, *adapter: [CheckpointAdapter](torchrl.checkpoint.CheckpointAdapter.html#torchrl.checkpoint.CheckpointAdapter) | None = None*, *options: [CheckpointOptions](torchrl.checkpoint.CheckpointOptions.html#torchrl.checkpoint.CheckpointOptions) | None = None*) → Checkpoint[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#Checkpoint.register)

Register a named component and return `self`.

Parameters:

- **name** - Stable manifest component name.
- **component** - Live object or JSON-compatible value.
- **adapter** - Optional explicit serialization adapter.
- **options** - Persistent component method arguments.

Returns:

This checkpoint.

register_adapter(*component_type: type*, *adapter: [CheckpointAdapter](torchrl.checkpoint.CheckpointAdapter.html#torchrl.checkpoint.CheckpointAdapter)*) → Checkpoint[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#Checkpoint.register_adapter)

Register an adapter for a type on this checkpoint instance.

*classmethod*register_migration(*from_version: int*, *migration: Callable[[dict[str, Any]], dict[str, Any]]*) → None[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#Checkpoint.register_migration)

Register one manifest migration from `from_version` to the next.

Migrations must return a new manifest whose `format_version` is
exactly `from_version + 1`. Payload migrations remain the
responsibility of the component adapter identified by the migrated
manifest.

save(*path: str | Path*, ***, *components: Collection[str] | None = None*, *component_options: Mapping[str, [CheckpointOptions](torchrl.checkpoint.CheckpointOptions.html#torchrl.checkpoint.CheckpointOptions)] | None = None*, *format: Literal['directory', 'archive'] | None = None*, *metadata: Mapping[str, Any] | None = None*) → Path[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#Checkpoint.save)

Save selected registered components atomically.

Parameters:

- **path** - Local checkpoint destination.
- **components** - Component names to save. `None` uses
`save_components` from construction, or saves all when no
default selection was configured.
- **component_options** - Per-operation option overrides.
- **format** - Per-operation container override.
- **metadata** - JSON-compatible manifest metadata.

Returns:

Expanded destination path.