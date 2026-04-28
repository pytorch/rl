.. _ref_profiling:

Profiling collectors and envs
=============================

TorchRL ships with a lightweight, opt-in profiling layer built on top of
``torch.profiler.record_function``. When enabled, the collector pipeline,
environment ``step`` / ``reset`` / ``rollout``, vectorised environments,
``TransformedEnv``, and the policy call inside the collector all emit named
ranges that show up directly in a Chrome trace or TensorBoard timeline.

When disabled — which is the default — every instrumentation site is a true
no-op: the decorator becomes the identity function (``lambda f: f``) and the
inline context managers return a shared ``nullcontext``. This makes the
instrumentation safe to keep enabled in production code.

Enabling profiling
------------------

Profiling is gated by the ``TORCHRL_PROFILING`` environment variable, which is
read **once at import time**.

.. code-block:: bash

    TORCHRL_PROFILING=1 python my_inference_loop.py

The variable must be set **before** ``torchrl`` is imported, because the
decorator's behaviour is decided at module-import time. Setting it later in
the same Python process has no effect on already-decorated functions.

Once armed, the instrumentation can still be toggled at runtime:

.. code-block:: python

    from torchrl import set_profiling_enabled

    set_profiling_enabled(False)  # silence the timeline temporarily
    ...
    set_profiling_enabled(True)   # re-enable around a region of interest

Calling ``set_profiling_enabled(True)`` without ``TORCHRL_PROFILING=1`` set
at import time emits a warning and is a no-op.

What gets instrumented
----------------------

Setting ``TORCHRL_PROFILING=1`` arms named ranges on the hot paths of:

- **Collectors** — ``Collector.rollout``, ``Collector.update_policy_weights_``,
  ``Collector.policy`` (the inline policy call inside the rollout loop),
  ``MultiSyncCollector.update_policy_weights_``,
  ``MultiAsyncCollector.update_policy_weights_``,
  ``AsyncBatchedCollector._rollout_frames`` and
  ``AsyncBatchedCollector._rollout_yield_trajs``.
- **Environments** — ``EnvBase.step``, ``EnvBase.reset``, ``EnvBase.rollout``,
  ``EnvBase.step_and_maybe_reset``.
- **Vectorised environments** — ``SerialEnv._step`` / ``_reset``,
  ``ParallelEnv._step`` / ``_reset`` / ``step_and_maybe_reset``.
- **Transforms** — ``TransformedEnv._step`` and ``TransformedEnv._reset``.

A typical inference rollout produces a timeline along the lines of::

    Collector.rollout
        Collector.policy
        EnvBase.step_and_maybe_reset
            TransformedEnv._step
                ParallelEnv._step

Capturing a trace
-----------------

The instrumentation works with any standard ``torch.profiler`` consumer.
A minimal example::

    import os
    os.environ["TORCHRL_PROFILING"] = "1"  # set BEFORE importing torchrl

    import torch
    from torchrl.collectors import Collector
    from torchrl.envs import GymEnv

    env = GymEnv("Pendulum-v1")
    collector = Collector(env, policy=my_policy, frames_per_batch=200, total_frames=1000)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        for _ in collector:
            pass

    prof.export_chrome_trace("collector_trace.json")

Open ``collector_trace.json`` in Chrome's ``chrome://tracing`` (or Perfetto)
to see the named ranges emitted by the collector and env hot paths.

Multi-process and Ray
---------------------

The decorator's import-time gate runs in **every process** that imports
``torchrl``. This means:

- **Multi-process collectors / ParallelEnv subprocesses** — ``os.environ`` is
  inherited by spawned children, so setting ``TORCHRL_PROFILING=1`` in the
  parent before launching is sufficient.
- **Ray actors** — TorchRL's :class:`~torchrl.collectors.distributed.RayCollector`
  and ``as_remote(...)`` automatically inject ``TORCHRL_PROFILING`` into each
  actor's ``runtime_env.env_vars`` when it is set on the driver, so remote
  workers are armed identically to the driver.

If you launch Ray actors yourself outside of TorchRL helpers, propagate the
variable explicitly::

    ray.remote(
        runtime_env={"env_vars": {"TORCHRL_PROFILING": "1"}},
    )(MyActor).remote(...)

Performance impact
------------------

When ``TORCHRL_PROFILING`` is unset (the default):

- Decorated methods are unwrapped — the decorator returns the original
  function reference, so calling ``env.step(...)`` has zero added cost.
- Inline ``with _maybe_record_function(...)`` blocks enter and exit a shared
  ``nullcontext`` singleton; the cost is negligible (~hundreds of nanoseconds
  per call).

When set, a per-call ``record_function`` push/pop is added on each
instrumented method. This is suitable for ad-hoc profiling sessions but is
not intended to be left on in long-running production jobs.
