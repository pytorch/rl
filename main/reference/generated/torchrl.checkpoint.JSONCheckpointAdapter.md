# JSONCheckpointAdapter

*class*torchrl.checkpoint.JSONCheckpointAdapter[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#JSONCheckpointAdapter)

Adapter for JSON-compatible configuration, metrics, and metadata.

Mutable mappings and lists are updated in place during restoration. Other
values are returned through `CheckpointLoadResult.values`.

Examples

```
>>> from torchrl.checkpoint import JSONCheckpointAdapter
>>> JSONCheckpointAdapter().adapter_id
'torchrl.json'
```

load(*component: Any*, *path: Path*, ***, *map_location: Any*, *tensor_load_kwargs: Mapping[str, Any]*, *args: tuple[Any, ...]*, *kwargs: Mapping[str, Any]*) → Any[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#JSONCheckpointAdapter.load)

Restore `component` from `path` and optionally return a value.

save(*component: Any*, *path: Path*, ***, *args: tuple[Any, ...]*, *kwargs: Mapping[str, Any]*) → Any[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#JSONCheckpointAdapter.save)

Save `component` below `path`.