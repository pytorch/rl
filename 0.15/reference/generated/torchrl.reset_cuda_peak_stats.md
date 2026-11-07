# reset_cuda_peak_stats

*class*torchrl.reset_cuda_peak_stats(*device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | int | str | None = None*)[[source]](../../_modules/torchrl/_utils.html#reset_cuda_peak_stats)

Reset the peak-memory counters for `device`.

Thin wrapper around `torch.cuda.reset_peak_memory_stats()`. No-op when
CUDA is unavailable or `device` is non-CUDA.

Parameters:

**device** - CUDA device whose peaks should be cleared. `None` (default)
targets the current CUDA device.

Examples

```
>>> from torchrl import reset_cuda_peak_stats
>>> reset_cuda_peak_stats() # safe even without CUDA
```

See also

[`cuda_memory_stats()`](torchrl.cuda_memory_stats.html#torchrl.cuda_memory_stats), [`cuda_memory_profile`](torchrl.cuda_memory_profile.html#torchrl.cuda_memory_profile).