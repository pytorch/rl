# torchrl._utils package

Set of utility methods that are used internally by the library.

| [`implement_for`](generated/torchrl.implement_for.html#torchrl.implement_for)(module_name[, from_version, ...]) | A version decorator that checks version compatibility and implements functions. |
| --- | --- |
| [`set_auto_unwrap_transformed_env`](generated/torchrl.set_auto_unwrap_transformed_env.html#torchrl.set_auto_unwrap_transformed_env)(mode) | A context manager or decorator to control whether TransformedEnv should automatically unwrap nested TransformedEnv instances. |
| [`auto_unwrap_transformed_env`](generated/torchrl.auto_unwrap_transformed_env.html#torchrl.auto_unwrap_transformed_env)([allow_none]) | Get the current setting for automatically unwrapping TransformedEnv instances. |

## Memory profiling

CUDA memory helpers that pair well with `timeit` for scoping
per-phase allocation peaks in training loops. They are safe to call on
CPU-only / MPS systems (they return zeros and no-op respectively), so the
calls can live unconditionally in device-agnostic code paths.

| [`cuda_memory_stats`](generated/torchrl.cuda_memory_stats.html#torchrl.cuda_memory_stats)([device]) | Return current CUDA memory statistics for `device` in gigabytes. |
| --- | --- |
| [`reset_cuda_peak_stats`](generated/torchrl.reset_cuda_peak_stats.html#torchrl.reset_cuda_peak_stats)([device]) | Reset the peak-memory counters for `device`. |
| [`cuda_memory_profile`](generated/torchrl.cuda_memory_profile.html#torchrl.cuda_memory_profile)(label, *[, device, log, ...]) | Context manager / decorator that reports CUDA memory deltas for a code block. |