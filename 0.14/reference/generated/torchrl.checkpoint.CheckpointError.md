# CheckpointError

*class*torchrl.checkpoint.CheckpointError(*message: str*, *result: [CheckpointLoadResult](torchrl.checkpoint.CheckpointLoadResult.html#torchrl.checkpoint.CheckpointLoadResult) | None = None*)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointError)

Error raised when a checkpoint cannot be saved or restored.

Parameters:

- **message** - Description of the checkpoint failure.
- **result** - Optional partial load result associated with the failure.

Examples

```
>>> from torchrl.checkpoint import CheckpointError
>>> error = CheckpointError("invalid checkpoint")
>>> str(error)
'invalid checkpoint'
```

add_note()

Exception.add_note(note) -
add a note to the exception

with_traceback()

Exception.with_traceback(tb) -
set self.__traceback__ to tb and return self.