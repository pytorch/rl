# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import os
from collections.abc import Sequence

from torch import Tensor

from .common import Logger

_has_wandb = importlib.util.find_spec("wandb") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None


class WandbLogger(Logger):
    """Wrapper for the wandb logger.

    The keyword arguments are mainly based on the :func:`wandb.init` kwargs.
    See the doc `here <https://docs.wandb.ai/ref/python/init>`__.

    Args:
        exp_name (str): The name of the experiment.
        offline (bool, optional): if ``True``, the logs will be stored locally
            only. Defaults to ``False``.
        save_dir (path, optional): the directory where to save data. Exclusive with
            ``log_dir``.
        log_dir (path, optional): the directory where to save data. Exclusive with
            ``save_dir``.
        id (str, optional): A unique ID for this run, used for resuming.
            It must be unique in the project, and if you delete a run you can't reuse the ID.
        project (str, optional): The name of the project where you're sending
            the new run. If the project is not specified, the run is put in
            an ``"Uncategorized"`` project.

    Keyword Args:
        fps (int, optional): Number of frames per second when recording videos. Defaults to ``30``.
        **kwargs: Extra keyword arguments for ``wandb.init``. See relevant page for
            more info.

    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(
        self,
        exp_name: str,
        offline: bool = False,
        save_dir: str = None,
        id: str = None,
        project: str = None,
        *,
        video_fps: int = 32,
        **kwargs,
    ) -> None:
        if not _has_wandb:
            raise ImportError("wandb could not be imported")

        log_dir = kwargs.pop("log_dir", None)
        self.offline = offline
        if save_dir and log_dir:
            raise ValueError(
                "log_dir and save_dir point to the same value in "
                "WandbLogger. Both cannot be specified."
            )
        save_dir = save_dir if save_dir and not log_dir else log_dir
        self.save_dir = save_dir
        self.id = id
        self.project = project
        self.video_fps = video_fps
        self._wandb_kwargs = {
            "name": exp_name,
            "dir": save_dir,
            "id": id,
            "project": project,
            "resume": "allow",
            **kwargs,
        }

        super().__init__(exp_name=exp_name, log_dir=save_dir)
        if self.offline:
            os.environ["WANDB_MODE"] = "dryrun"

    def _create_experiment(self):
        """Creates a wandb experiment.

        Args:
            exp_name (str): The name of the experiment.

        Returns:
            A wandb.Experiment object.
        """
        if not _has_wandb:
            raise ImportError("Wandb is not installed")
        import wandb

        if self.offline:
            os.environ["WANDB_MODE"] = "dryrun"

        return wandb.init(**self._wandb_kwargs)

    def log_scalar(
        self, name: str, value: float, step: int | None = None, commit: bool = False
    ) -> None:
        """Logs a scalar value to wandb.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
            commit: If true, data for current step is assumed to be final (and
                no further data for this step should be logged).
        """
        self.experiment.log({name: value}, step=step, commit=commit)

    def log_video(self, name: str, video: Tensor, **kwargs) -> None:
        """Log videos inputs to wandb.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'step' (integer indicating the step index), 'format'
                (default is 'mp4') and 'fps' (defaults to ``self.video_fps``). Other kwargs are
                passed as-is to the :obj:`experiment.log` method.
        """
        import wandb

        fps = kwargs.pop("fps", self.video_fps)
        format = kwargs.pop("format", "mp4")
        self.experiment.log(
            {name: wandb.Video(video, fps=fps, format=format)},
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
        self.experiment.config.update(cfg, allow_val_change=True)

    def __repr__(self) -> str:
        return f"WandbLogger(experiment={self.experiment.__repr__()})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        """Add histogram to log.

        Args:
            name (str): Data identifier
            data (torch.Tensor, numpy.ndarray, or string/blobname): Values to build histogram

        Keyword Args:
            step (int): Global step value to record
            bins (str): One of {'tensorflow','auto', 'fd', …}. This determines how the bins are made. You can find other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

        """
        import wandb

        num_bins = kwargs.pop("bins", None)
        step = kwargs.pop("step", None)
        extra_kwargs = {}
        if step is not None:
            extra_kwargs["trainer/step"] = step
        self.experiment.log(
            {name: wandb.Histogram(data, num_bins=num_bins), **extra_kwargs}
        )

    def log_str(self, name: str, value: str, step: int | None = None) -> None:
        """Logs a string value to wandb using a table format for better visualization.

        Args:
            name (str): The name of the string data.
            value (str): The string value to log.
            step (int, optional): The step at which the string is logged.
                Defaults to None.
        """
        import wandb

        # Create a table with a single row
        table = wandb.Table(columns=["text"], data=[[value]])

        if step is not None:
            self.experiment.log({name: value}, step=step)
        else:
            self.experiment.log({name: table})
