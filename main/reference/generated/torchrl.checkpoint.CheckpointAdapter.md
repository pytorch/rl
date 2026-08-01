# CheckpointAdapter

*class*torchrl.checkpoint.CheckpointAdapter[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointAdapter)

Interface used to save and restore one checkpoint component.

Adapters receive a real local directory even when the outer checkpoint is
an archive. Custom adapters can therefore write any number of files without
depending on the container implementation.

Examples

```
>>> from torchrl.checkpoint import CheckpointAdapter
>>> issubclass(CheckpointAdapter, object)
True
```

*abstract*load(*component: Any*, *path: Path*, ***, *map_location: Any*, *tensor_load_kwargs: Mapping[str, Any]*, *args: tuple[Any, ...]*, *kwargs: Mapping[str, Any]*) → Any[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointAdapter.load)

Restore `component` from `path` and optionally return a value.

*abstract*save(*component: Any*, *path: Path*, ***, *args: tuple[Any, ...]*, *kwargs: Mapping[str, Any]*) → Any[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointAdapter.save)

Save `component` below `path`.