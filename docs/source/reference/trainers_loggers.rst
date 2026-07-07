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

Recording utils
---------------

.. currentmodule:: torchrl.record

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    VideoRecorder
    TensorDictRecorder
    PixelRenderTransform
