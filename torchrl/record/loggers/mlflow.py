# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

try:
    import torchvision

    _has_tv = True
except ImportError:
    _has_tv = False
from torch import Tensor

from .common import Logger

MLFLOW_ERR = None
try:
    import mlflow

    _has_mlflow = True
except ImportError as err:
    _has_mlflow = False
    MLFLOW_ERR = err

try:
    from omegaconf import OmegaConf

    _has_omgaconf = True
except ImportError:
    _has_omgaconf = False


class MLFlowLogger(Logger):
    """Wrapper for the mlflow logger.

    Args:
        exp_name (str): The name of the experiment.
        tracking_uri (str): A tracking URI to a datastore that supports MLFlow or a local directory.
    """

    def __init__(
        self,
        exp_name: str,
        tracking_uri: str,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self._mlflow_kwargs = {
            "name": exp_name,
            "artifact_location": tracking_uri,
            "tags": tags,
        }
        mlflow.set_tracking_uri(tracking_uri)
        super().__init__(exp_name=exp_name, log_dir=tracking_uri)
        self.video_log_counter = 0

    def _create_experiment(self) -> "mlflow.ActiveRun":
        """Creates an mlflow experiment.

        Returns:
            mlflow.ActiveRun: The mlflow experiment object.
        """
        if not _has_mlflow:
            raise ImportError("MLFlow is not installed") from MLFLOW_ERR
        self.id = mlflow.create_experiment(**self._mlflow_kwargs)
        return mlflow.start_run(experiment_id=self.id)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Logs a scalar value to mlflow.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
        """
        mlflow.set_experiment(experiment_id=self.id)
        mlflow.log_metric(key=name, value=value, step=step)

    def log_video(self, name: str, video: Tensor, **kwargs) -> None:
        """Log video inputs to mlflow.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged, expected to be in (T, C, H, W) format
                for consistency with other loggers.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'step' (integer indicating the step index) and 'fps' (default: 6).
        """
        if not _has_tv:
            raise ImportError(
                "Loggin a video with MLFlow requires torchvision to be installed."
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
        fps = kwargs.pop("fps", 6)
        step = kwargs.pop("step", None)
        with TemporaryDirectory() as temp_dir:
            video_name = f"{name}_step_{step:04}.mp4" if step else f"{name}.mp4"
            with open(os.path.join(temp_dir, video_name), "wb") as f:
                torchvision.io.write_video(filename=f.name, video_array=video, fps=fps)
                mlflow.log_artifact(f.name, "videos")

    def log_hparams(self, cfg: "DictConfig") -> None:  # noqa: F821
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig): The configuration of the experiment.
        """
        mlflow.set_experiment(experiment_id=self.id)
        if type(cfg) is not dict and _has_omgaconf:
            cfg = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(cfg)

    def __repr__(self) -> str:
        return f"MLFlowLogger(experiment={self.experiment.__repr__()})"
