.. currentmodule:: torchrl

torchrl._utils package
======================

Set of utility methods that are used internally by the library.


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    implement_for
    set_auto_unwrap_transformed_env
    auto_unwrap_transformed_env

Memory profiling
----------------

CUDA memory helpers that pair well with :class:`timeit` for scoping
per-phase allocation peaks in training loops. They are safe to call on
CPU-only / MPS systems (they return zeros and no-op respectively), so the
calls can live unconditionally in device-agnostic code paths.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    cuda_memory_stats
    reset_cuda_peak_stats
    cuda_memory_profile
