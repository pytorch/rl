.. currentmodule:: torchrl.record.loggers

.. _ref_loggers:

Loggers
=======

Logger classes for experiment tracking and visualization.

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    Logger
    csv.CSVLogger
    mlflow.MLFlowLogger
    tensorboard.TensorboardLogger
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
