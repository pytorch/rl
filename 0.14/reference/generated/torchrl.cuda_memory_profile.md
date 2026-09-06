# cuda_memory_profile

*class*torchrl.cuda_memory_profile(*label: str*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | int | str | None = None*, *log: bool = True*, *reset_peaks: bool = True*)[[source]](../../_modules/torchrl/_utils.html#cuda_memory_profile)

Context manager / decorator that reports CUDA memory deltas for a code block.

On `__enter__` (optionally) clears the peak-memory counters; on
`__exit__` reads [`cuda_memory_stats()`](torchrl.cuda_memory_stats.html#torchrl.cuda_memory_stats) and logs the delta (current -
pre-block) plus the new peaks. The collected stats are stored on the
`stats` attribute for programmatic access after the block exits.

Parameters:

- **label** - Short identifier prepended to the log line and stored on the
instance for downstream metric routing.
- **device** - CUDA device to profile. `None` (default) targets the current
CUDA device. On non-CUDA devices the manager is a no-op.
- **log** - When `True` (default), emit a single `INFO` line via
`torchrl.torchrl_logger` at exit. When `False` only the
`stats` attribute is populated.
- **reset_peaks** - When `True` (default), reset peak counters on enter so
the reported `max_*` values reflect the block only.

Examples

```
>>> import torch
>>> from torchrl import cuda_memory_profile
>>> with cuda_memory_profile("warmup", log=False) as prof:
... if torch.cuda.is_available():
... _ = torch.zeros(1, device="cuda")
>>> sorted(prof.stats)
['allocated_gb', 'max_allocated_gb', 'max_reserved_gb', 'reserved_gb']
```

See also

[`cuda_memory_stats()`](torchrl.cuda_memory_stats.html#torchrl.cuda_memory_stats), [`reset_cuda_peak_stats()`](torchrl.reset_cuda_peak_stats.html#torchrl.reset_cuda_peak_stats), `timeit`.