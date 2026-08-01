# CheckpointLoadResult

*class*torchrl.checkpoint.CheckpointLoadResult(*loaded: set[str] = <factory>*, *missing: set[str] = <factory>*, *incompatible: dict[str*, *str] = <factory>*, *unrequested: set[str] = <factory>*, *values: dict[str*, *~typing.Any] = <factory>*, *manifest: dict[str*, *~typing.Any] = <factory>*)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointLoadResult)

Structured result returned by [`Checkpoint.load()`](torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint.load).

Parameters:

- **loaded** - Components restored successfully.
- **missing** - Requested components absent from the checkpoint.
- **incompatible** - Components that could not be restored, mapped to reasons.
- **unrequested** - Manifest components intentionally not requested.
- **values** - Values returned by adapters, including immutable JSON components.
- **manifest** - Parsed checkpoint manifest.

Examples

```
>>> from torchrl.checkpoint import CheckpointLoadResult
>>> result = CheckpointLoadResult()
>>> result.loaded
set()
```