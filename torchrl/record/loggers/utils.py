# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pathlib
import uuid
from datetime import datetime

from torchrl.record.loggers.common import Logger


def generate_exp_name(model_name: str, experiment_name: str) -> str:
    """Generates an ID (str) for the described experiment using UUID and current date."""
    exp_name = "_".join(
        (
            model_name,
            experiment_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        )
    )
    return exp_name


def get_logger(
    logger_type: str, logger_name: str, experiment_name: str, **kwargs
) -> Logger:
    """Get a logger instance of the provided `logger_type`.

    Args:
        logger_type (str): One of tensorboard / csv / wandb / mlflow.
            If empty, ``None`` is returned.
        logger_name (str): Name to be used as a log_dir
        experiment_name (str): Name of the experiment
        kwargs (dict[str]): might contain either `wandb_kwargs` or `mlflow_kwargs`
    """
    if logger_type == "tensorboard":
        from torchrl.record.loggers.tensorboard import TensorboardLogger

        logger = TensorboardLogger(log_dir=logger_name, exp_name=experiment_name)
    elif logger_type == "csv":
        from torchrl.record.loggers.csv import CSVLogger

        logger = CSVLogger(log_dir=logger_name, exp_name=experiment_name)
    elif logger_type == "wandb":
        from torchrl.record.loggers.wandb import WandbLogger

        wandb_kwargs = kwargs.get("wandb_kwargs", {})
        logger = WandbLogger(
            log_dir=logger_name, exp_name=experiment_name, **wandb_kwargs
        )
    elif logger_type == "mlflow":
        from torchrl.record.loggers.mlflow import MLFlowLogger

        mlflow_kwargs = kwargs.get("mlflow_kwargs", {})
        logger = MLFlowLogger(
            tracking_uri=pathlib.Path(os.path.abspath(logger_name)).as_uri(),
            exp_name=experiment_name,
            **mlflow_kwargs,
        )
    elif logger_type in ("", None):
        return None
    else:
        raise NotImplementedError(f"Unsupported logger_type: '{logger_type}'")
    return logger
