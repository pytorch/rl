# DumpLoadCheckpointAdapter

*class*torchrl.checkpoint.DumpLoadCheckpointAdapter[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#DumpLoadCheckpointAdapter)

Adapter for objects exposing `dump(path)` and `load(path)`.

Examples

```
>>> from torchrl.checkpoint import DumpLoadCheckpointAdapter
>>> DumpLoadCheckpointAdapter().adapter_id
'torchrl.dump_load'
```

load(*component: Any*, *path: Path*, ***, *map_location: Any*, *tensor_load_kwargs: Mapping[str, Any]*, *args: tuple[Any, ...]*, *kwargs: Mapping[str, Any]*) → Any[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#DumpLoadCheckpointAdapter.load)

Restore `component` from `path` and optionally return a value.

save(*component: Any*, *path: Path*, ***, *args: tuple[Any, ...]*, *kwargs: Mapping[str, Any]*) → Any[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#DumpLoadCheckpointAdapter.save)

Save `component` below `path`.