# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

from collections.abc import Sequence

import numpy as np

from torch import Tensor

from .common import Logger

_has_trackio = importlib.util.find_spec("trackio") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None


class TrackioLogger(Logger):
    """Wrapper for the trackio logger.

    Args:
        exp_name (str): The name of the experiment.
        project (str): The name of the project.

    Keyword Args:
        fps (int, optional): Number of frames per second when recording videos. Defaults to ``30``.
        **kwargs: Extra keyword arguments for ``trackio.init``.

    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(
        self,
        exp_name: str,
        project: str,
        *,
        video_fps: int = 32,
        **kwargs,
    ) -> None:
        if not _has_trackio:
            raise ImportError("trackio could not be imported")

        self.video_fps = video_fps
        self._trackio_kwargs = {
            "name": exp_name,
            "project": project,
            "resume": "allow",
            **kwargs,
        }

        super().__init__(exp_name=exp_name, log_dir=project)

    def _create_experiment(self):
        """Creates a trackio experiment.

        Args:
            exp_name (str): The name of the experiment.

        Returns:
            A trackio.Experiment object.
        """
        if not _has_trackio:
            raise ImportError("Trackio is not installed")
        import trackio

        return trackio.init(**self._trackio_kwargs)

    def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
        """Logs a scalar value to trackio.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
        """
        self.experiment.log({name: value}, step=step)

    def log_video(self, name: str, video: Tensor, **kwargs) -> None:
        """Log videos inputs to trackio.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'step' (integer indicating the step index), 'format'
                (default is 'mp4') and 'fps' (defaults to ``self.video_fps``). Other kwargs are
                passed as-is to the :obj:`experiment.log` method.
        """
        import trackio

        fps = kwargs.pop("fps", self.video_fps)
        format = kwargs.pop("format", "mp4")
        self.experiment.log(
            {
                name: trackio.Video(
                    video.numpy().astype(np.uint8), fps=fps, format=format
                )
            },
            **kwargs,
        )

    def log_hparams(self, cfg: DictConfig | dict) -> None:  # noqa: F821
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig or dict): The configuration of the experiment.

        """
        if type(cfg) is not dict and _has_omegaconf:
            if not _has_omegaconf:
                raise ImportError(
                    "OmegaConf could not be imported. "
                    "Cannot log hydra configs without OmegaConf."
                )
            from omegaconf import OmegaConf

            cfg = OmegaConf.to_container(cfg, resolve=True)
        self.experiment.config.update(cfg)

    def __repr__(self) -> str:
        return f"TrackioLogger(experiment={self.experiment.__repr__()})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        """Add histogram to log.

        Args:
            name (str): Data identifier
            data (torch.Tensor, numpy.ndarray): Values to build histogram

        Keyword Args:
            step (int): Global step value to record
            bins (int): Number of bins to use for the histogram

        """
        import trackio

        num_bins = kwargs.pop("bins", None)
        step = kwargs.pop("step", None)
        self.experiment.log(
            {name: trackio.Histogram(data, num_bins=num_bins)}, step=step
        )

    def log_str(self, name: str, value: str, step: int | None = None) -> None:
        """Logs a string value to trackio using a table format for better visualization.

        Args:
            name (str): The name of the string data.
            value (str): The string value to log.
            step (int, optional): The step at which the string is logged.
                Defaults to None.
        """
        import trackio

        # Create a table with a single row
        table = trackio.Table(columns=["text"], data=[[value]])
        self.experiment.log({name: table}, step=step)
