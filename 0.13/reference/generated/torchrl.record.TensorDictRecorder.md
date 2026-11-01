# TensorDictRecorder

torchrl.record.TensorDictRecorder(*out_file_base: str*, *skip_reset: bool = True*, *skip: int = 4*, *in_keys: Sequence[str] | None = None*) → None[[source]](../../_modules/torchrl/record/recorder.html#TensorDictRecorder)

TensorDict recorder.

When the 'dump' method is called, this class will save a stack of the tensordict resulting from `env.step(td)` in a
file with a prefix defined by the out_file_base argument.

Parameters:

- **out_file_base** (*str*) - a string defining the prefix of the file where the tensordict will be written.
- **skip_reset** (*bool*) - if `True`, the first TensorDict of the list will be discarded (usually the tensordict
resulting from the call to `env.reset()`)
default: True
- **skip** (*int*) - frame interval for the saved tensordict.
default: 4