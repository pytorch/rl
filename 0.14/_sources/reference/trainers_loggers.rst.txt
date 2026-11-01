.. currentmodule:: torchrl.record.loggers

.. _ref_loggers:

Loggers
=======

Logger classes for experiment tracking and visualization.

Loggers support an owner/client deployment model. The direct backend returns
the logger itself from ``client()``. Process and Ray backends return a
picklable client that only exposes logging operations; lifecycle calls remain
on the owner. Remote calls preserve direct-logger semantics: they return after
the concrete logger method has run, propagate service-side errors immediately,
and preserve custom ``log_*`` return values. Bounded transport queues provide
backpressure when several clients log concurrently.

.. code-block:: python

    from torchrl.record import CSVLogger

    logger = CSVLogger(
        exp_name="run",
        log_dir="logs",
        service_backend="process",
        service_backend_options={"max_queue_size": 256},
    )
    worker_logger = logger.client()
    worker_logger.log_scalar("loss", 1.0, step=0)
    logger.flush()  # Flush buffers owned by the concrete logging SDK.
    logger.shutdown()

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    Logger
    ProcessLogger
    RayLogger
    csv.CSVLogger
    mlflow.MLFlowLogger
    tensorboard.TensorboardLogger
    trackio.TrackioLogger
    wandb.WandbLogger
    get_logger
    generate_exp_name

Monitoring collectors and replay buffers
----------------------------------------

Any collector or replay buffer exposes a cheap
:meth:`~torchrl.collectors.BaseCollector.stats` /
:meth:`~torchrl.data.ReplayBuffer.stats` snapshot of operational counters
(frames collected, buffer size, write count, worker liveness, ...). A
:class:`~torchrl.record.loggers.monitoring.LoggerMonitor` periodically pulls
those snapshots on a wall-clock or counter
(:class:`~torchrl.record.loggers.monitoring.Every`) schedule, derives rates
such as frames per second, and forwards namespaced metrics to any logger,
without adding work to collection or sampling hot paths. This works with
local, multiprocessing and Ray implementations alike.

.. code-block:: python

    from torchrl.record import WandbLogger
    from torchrl.record.loggers.monitoring import Every, LoggerMonitor

    logger = WandbLogger(exp_name="experiment", project="torchrl")

    with LoggerMonitor(logger, poll_interval=1.0) as monitor:
        monitor.watch(collector, name="collector", schedule=Every.counter("frames", 10_000))
        monitor.watch(replay_buffer, name="replay_buffer", schedule=Every.seconds(5), step="write_count")
        collector.start()
        run_training()

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    monitoring.LoggerMonitor
    monitoring.Every

Recording utils
---------------

.. currentmodule:: torchrl.record

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    VideoRecorder
    TensorDictRecorder
    PixelRenderTransform
