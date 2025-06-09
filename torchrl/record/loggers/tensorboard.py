# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import os
from typing import Sequence

from torch import Tensor

from .common import Logger

_has_tb = importlib.util.find_spec("tensorboard") is not None
_has_omgaconf = importlib.util.find_spec("omegaconf") is not None


class TensorboardLogger(Logger):
    """Wrapper for the Tensoarboard logger.

    Args:
        exp_name (str): The name of the experiment.
        log_dir (str): the tensorboard log_dir. Defaults to ``td_logs``.

    """

    def __init__(self, exp_name: str, log_dir: str = "tb_logs") -> None:
        super().__init__(exp_name=exp_name, log_dir=log_dir)
        # re-write log_dir
        self.log_dir = self.experiment.log_dir

        self._has_imported_moviepy = False

    def _create_experiment(self) -> SummaryWriter:  # noqa
        """Creates a tensorboard experiment.

        Args:
            exp_name (str): The name of the experiment.

        Returns:
            SummaryWriter: The tensorboard experiment.

        """
        if not _has_tb:
            raise ImportError("torch.utils.tensorboard could not be imported")

        from torch.utils.tensorboard import SummaryWriter

        log_dir = str(os.path.join(self.log_dir, self.exp_name))
        return SummaryWriter(log_dir=log_dir)

    def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
        """Logs a scalar value to the tensorboard.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged. Defaults to None.

        """
        self.experiment.add_scalar(name, value, global_step=step)

    def log_video(
        self, name: str, video: Tensor, step: int | None = None, **kwargs
    ) -> None:
        """Log videos inputs to the tensorboard.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            step (int, optional): The step at which the video is logged. Defaults to None.

        """
        # check for correct format of the video tensor ((N), T, C, H, W)
        # check that the color channel (C) is either 1 or 3
        if video.dim() != 5 or video.size(dim=2) not in {1, 3}:
            raise Exception(
                "Wrong format of the video tensor. Should be ((N), T, C, H, W)"
            )
        if not self._has_imported_moviepy:
            try:
                import moviepy  # noqa

                self._has_imported_moviepy = True
            except ImportError:
                raise Exception(
                    "moviepy not found, videos cannot be logged with TensorboardLogger"
                )
        self.experiment.add_video(
            tag=name,
            vid_tensor=video,
            global_step=step,
            **kwargs,
        )

    def log_hparams(self, cfg: DictConfig | dict) -> None:  # noqa: F821
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig or dict): The configuration of the experiment.

        """
        if type(cfg) is not dict and _has_omgaconf:
            if not _has_omgaconf:
                raise ImportError(
                    "OmegaConf could not be imported. "
                    "Cannot log hydra configs without OmegaConf."
                )
            from omegaconf import OmegaConf

            cfg = OmegaConf.to_container(cfg, resolve=True)
        self.experiment.add_hparams(cfg, metric_dict={})

    def __repr__(self) -> str:
        return f"TensorboardLogger(experiment={self.experiment.__repr__()})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        """Add histogram to summary.

        Args:
            name (str): Data identifier
            data (torch.Tensor, numpy.ndarray, or string/blobname): Values to build histogram

        Keyword Args:
            step (int): Global step value to record
            bins (str): One of {‘tensorflow’,’auto’, ‘fd’, …}. This determines how the bins are made. You can find other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            walltime (:obj:`float`): Optional override default walltime (time.time()) seconds after epoch of event

        """
        global_step = kwargs.pop("step", None)
        bins = kwargs.pop("bins")
        walltime = kwargs.pop("walltime", None)
        if len(kwargs):
            raise TypeError(f"Unrecognised arguments {kwargs}.")
        self.experiment.add_histogram(
            tag=name, values=data, global_step=global_step, bins=bins, walltime=walltime
        )
