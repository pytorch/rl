# StateDictCheckpointAdapter

*class*torchrl.checkpoint.StateDictCheckpointAdapter(*payload_format: Literal['directory', 'archive', 'consolidated', 'torch'] = 'directory'*, ***, *archive_compression: str | int | None = None*)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#StateDictCheckpointAdapter)

Adapter for `state_dict` / `load_state_dict` objects.

TensorDict directory payloads are used by default. `payload_format` can
select a TensorDict archive, a consolidated TensorDict, or the pickle-based
[`torch.save()`](https://docs.pytorch.org/docs/stable/generated/torch.save.html#torch.save) format. Loading auto-detects all four payload formats.

Parameters:

- **payload_format** - Format used for new payloads. One of `"directory"`,
`"archive"`, `"consolidated"`, or `"torch"`.
- **archive_compression** - Compression passed to TensorDict archive saves.

Examples

```
>>> from torchrl.checkpoint import StateDictCheckpointAdapter
>>> StateDictCheckpointAdapter().payload_format
'directory'
>>> StateDictCheckpointAdapter(payload_format="torch").payload_format
'torch'
```

load(*component: Any*, *path: Path*, ***, *map_location: Any*, *tensor_load_kwargs: Mapping[str, Any]*, *args: tuple[Any, ...]*, *kwargs: Mapping[str, Any]*) → Any[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#StateDictCheckpointAdapter.load)

Restore `component` from `path` and optionally return a value.

save(*component: Any*, *path: Path*, ***, *args: tuple[Any, ...]*, *kwargs: Mapping[str, Any]*) → Any[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#StateDictCheckpointAdapter.save)

Save `component` below `path`.