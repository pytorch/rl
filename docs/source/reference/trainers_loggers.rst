.. currentmodule:: torchrl.record.loggers

.. _ref_loggers:

Loggers
=======

Logger classes for experiment tracking and visualization.

Loggers support an owner/client deployment model. The direct backend returns
the logger itself from ``client()``. Process and Ray backends return a
picklable client that only exposes logging operations; lifecycle calls remain
on the owner. Ordinary records are submitted asynchronously, while
``log_video`` waits for encoding or upload to finish.

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
    logger.flush()
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
