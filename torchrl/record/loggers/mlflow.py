# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import os
from tempfile import TemporaryDirectory
from typing import Any, Sequence

from torch import Tensor

from torchrl.record.loggers.common import Logger

_has_tv = importlib.util.find_spec("torchvision") is not None

_has_mlflow = importlib.util.find_spec("mlflow") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None


class MLFlowLogger(Logger):
    """Wrapper for the mlflow logger.

    Args:
        exp_name (str): The name of the experiment.
        tracking_uri (str): A tracking URI to a datastore that supports MLFlow or a local directory.

    Keyword Args:
        fps (int, optional): Number of frames per second when recording videos. Defaults to ``30``.

    """

    def __init__(
        self,
        exp_name: str,
        tracking_uri: str,
        tags: dict[str, Any] | None = None,
        *,
        video_fps: int = 30,
        **kwargs,
    ) -> None:
        import mlflow

        self._mlflow_kwargs = {
            "name": exp_name,
            "artifact_location": tracking_uri,
            "tags": tags,
        }
        mlflow.set_tracking_uri(tracking_uri)
        super().__init__(exp_name=exp_name, log_dir=tracking_uri)
        self.video_log_counter = 0
        self.video_fps = video_fps

    def _create_experiment(self) -> mlflow.ActiveRun:  # noqa
        import mlflow

        """Creates an mlflow experiment.

        Returns:
            mlflow.ActiveRun: The mlflow experiment object.
        """
        if not _has_mlflow:
            raise ImportError("MLFlow is not installed")

        # Only create experiment if it doesnt exist
        experiment = mlflow.get_experiment_by_name(self._mlflow_kwargs["name"])
        if experiment is None:
            self.id = mlflow.create_experiment(**self._mlflow_kwargs)
        else:
            self.id = experiment.experiment_id
        return mlflow.start_run(experiment_id=self.id)

    def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
        """Logs a scalar value to mlflow.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
        """
        import mlflow

        mlflow.set_experiment(experiment_id=self.id)
        mlflow.log_metric(key=name, value=value, step=step)

    def log_video(self, name: str, video: Tensor, **kwargs) -> None:
        """Log video inputs to mlflow.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged, expected to be in (T, C, H, W) format
                for consistency with other loggers.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'step' (integer indicating the step index) and 'fps' (defaults to ``self.video_fps``).
        """
        import mlflow
        import torchvision

        if not _has_tv:
            raise ImportError(
                "Logging a video with MLFlow requires torchvision to be installed."
            )
        mlflow.set_experiment(experiment_id=self.id)
        if video.ndim == 5:
            video = video[-1]  # N T C H W -> T C H W
        video = video.permute(0, 2, 3, 1)  # T C H W -> T H W C
        if video.size(dim=-1) != 3:
            raise ValueError(
                "The MLFlow logger only supports videos with 3 color channels."
            )
        self.video_log_counter += 1
        fps = kwargs.pop("fps", self.video_fps)
        step = kwargs.pop("step", None)
        with TemporaryDirectory() as temp_dir:
            video_name = f"{name}_step_{step:04}.mp4" if step else f"{name}.mp4"
            with open(os.path.join(temp_dir, video_name), "wb") as f:
                torchvision.io.write_video(filename=f.name, video_array=video, fps=fps)
                mlflow.log_artifact(f.name, "videos")

    def log_hparams(self, cfg: DictConfig | dict) -> None:  # noqa: F821
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig or dict): The configuration of the experiment.
        """
        import mlflow
        from omegaconf import OmegaConf

        mlflow.set_experiment(experiment_id=self.id)
        if type(cfg) is not dict and _has_omegaconf:
            cfg = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(cfg)

    def __repr__(self) -> str:
        return f"MLFlowLogger(experiment={self.experiment.__repr__()})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        raise NotImplementedError("Logging histograms in cvs is not permitted.")
