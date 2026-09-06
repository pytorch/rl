# cuda_memory_stats

*class*torchrl.cuda_memory_stats(*device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | int | str | None = None*)[[source]](../../_modules/torchrl/_utils.html#cuda_memory_stats)

Return current CUDA memory statistics for `device` in gigabytes.

Wraps `torch.cuda.memory_allocated()`, `torch.cuda.memory_reserved()`,
`torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`
into a single dict suitable for logging or comparing phases of a training loop.

Parameters:

**device** - CUDA device to query. `None` (default) targets the current
CUDA device. CPU/MPS/unset devices return zeros (no warning) so the
helper can be called unconditionally from device-agnostic code.

Returns:

Mapping with keys `"allocated_gb"`, `"reserved_gb"`,
`"max_allocated_gb"`, `"max_reserved_gb"`. Values are floats in
gigabytes. When CUDA is not available, all values are `0.0`.

Examples

```
>>> from torchrl import cuda_memory_stats
>>> stats = cuda_memory_stats()
>>> sorted(stats)
['allocated_gb', 'max_allocated_gb', 'max_reserved_gb', 'reserved_gb']
```

See also

[`reset_cuda_peak_stats()`](torchrl.reset_cuda_peak_stats.html#torchrl.reset_cuda_peak_stats), [`cuda_memory_profile`](torchrl.cuda_memory_profile.html#torchrl.cuda_memory_profile).