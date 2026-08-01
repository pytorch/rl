# GlobalRNGState

*class*torchrl.checkpoint.GlobalRNGState[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#GlobalRNGState)

Checkpointable process-global random-number-generator state.

The object captures Python, NumPy, Torch CPU, and initialized accelerator
RNGs when `state_dict()` is called.

Examples

```
>>> import torch
>>> from torchrl.checkpoint import GlobalRNGState
>>> state = GlobalRNGState().state_dict()
>>> "torch_cpu" in state
True
```

load_state_dict(*state_dict: Mapping[str, Any]*) → None[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#GlobalRNGState.load_state_dict)

Restore process-global RNG state.

state_dict() → dict[str, Any][[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#GlobalRNGState.state_dict)

Capture all supported process-global RNG state.