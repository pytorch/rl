# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
import pathlib
import uuid
from datetime import datetime
from typing import Any, Literal

from torchrl.record.loggers.common import Logger
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record.loggers.mlflow import MLFlowLogger
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.record.loggers.trackio import TrackioLogger
from torchrl.record.loggers.wandb import WandbLogger


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
    logger_type: Literal["tensorboard", "csv", "wandb", "mlflow", "trackio", ""] | None,
    logger_name: str,
    experiment_name: str,
    *,
    service_backend: Literal["direct", "process", "ray"] = "direct",
    service_backend_options: dict[str, Any] | None = None,
    use_ray_service: bool = False,
    ray_actor_options: dict[str, Any] | None = None,
    **kwargs,
) -> Logger | None:
    """Get a logger instance of the provided `logger_type`.

    Args:
        logger_type (str): One of tensorboard / csv / wandb / mlflow / trackio.
            If empty, ``None`` is returned.
        logger_name (str): Name to be used as a log_dir
        experiment_name (str): Name of the experiment
        service_backend: One of ``"direct"``, ``"process"``, or ``"ray"``.
        service_backend_options: Process or Ray initialization options.
        use_ray_service: Deprecated compatibility flag for the Ray backend.
        ray_actor_options: Deprecated spelling for Ray actor options.
        **kwargs: May contain ``wandb_kwargs``, ``mlflow_kwargs``, or
            ``trackio_kwargs``.
    """
    service_kwargs = {
        "service_backend_options": dict(service_backend_options or {}),
    }
    if use_ray_service:
        service_kwargs["use_ray_service"] = True
    else:
        service_kwargs["service_backend"] = service_backend
    if ray_actor_options is not None:
        if service_kwargs["service_backend_options"]:
            raise ValueError(
                "ray_actor_options and service_backend_options are mutually exclusive."
            )
        # Keep the legacy argument on the metaclass path so it retains its
        # exact behavior while use_ray_service emits the single warning.
        if use_ray_service:
            service_kwargs["ray_actor_options"] = ray_actor_options
        else:
            service_kwargs["service_backend_options"] = {
                "actor_options": ray_actor_options
            }

    if logger_type == "tensorboard":
        logger = TensorboardLogger(
            log_dir=logger_name, exp_name=experiment_name, **service_kwargs
        )
    elif logger_type == "csv":
        logger = CSVLogger(
            log_dir=logger_name,
            exp_name=experiment_name,
            video_format="mp4",
            **service_kwargs,
        )
    elif logger_type == "wandb":
        wandb_kwargs = kwargs.get("wandb_kwargs", {})
        logger = WandbLogger(
            log_dir=logger_name,
            exp_name=experiment_name,
            **wandb_kwargs,
            **service_kwargs,
        )
    elif logger_type == "mlflow":
        mlflow_kwargs = kwargs.get("mlflow_kwargs", {})
        logger = MLFlowLogger(
            tracking_uri=pathlib.Path(os.path.abspath(logger_name)).as_uri(),
            exp_name=experiment_name,
            **mlflow_kwargs,
            **service_kwargs,
        )
    elif logger_type == "trackio":
        trackio_kwargs = kwargs.get("trackio_kwargs", {})
        project = trackio_kwargs.pop("project", "torchrl")
        logger = TrackioLogger(
            project=project,
            exp_name=experiment_name,
            **trackio_kwargs,
            **service_kwargs,
        )
    elif logger_type in ("", None):
        return None
    else:
        raise NotImplementedError(f"Unsupported logger_type: '{logger_type}'")
    return logger
