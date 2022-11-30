# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from .common import Logger


class CSVExperiment:
    """A CSV logger experiment class."""

    def __init__(self, log_dir: str):
        self.scalars = defaultdict(lambda: [])
        self.videos_counter = defaultdict(lambda: 0)
        self.text_counter = defaultdict(lambda: 0)
        self.log_dir = log_dir
        os.makedirs(self.log_dir)
        os.makedirs(os.path.join(self.log_dir, "scalars"))
        os.makedirs(os.path.join(self.log_dir, "videos"))
        os.makedirs(os.path.join(self.log_dir, "texts"))

    def add_scalar(self, name: str, value: float, global_step: Optional[int] = None):
        if global_step is None:
            global_step = len(self.scalars[name])
        value = float(value)
        self.scalars[name].append((global_step, value))
        filepath = os.path.join(self.log_dir, "scalars", "".join([name, ".csv"]))
        with open(filepath, "a") as fd:
            fd.write(",".join([str(global_step), str(value)]) + "\n")

    def add_video(self, tag, vid_tensor, global_step: Optional[int] = None, **kwargs):
        if global_step is None:
            global_step = self.videos_counter[tag]
            self.videos_counter[tag] += 1
        filepath = os.path.join(
            self.log_dir, "videos", "_".join([tag, str(global_step)]) + ".pt"
        )
        path_to_create = Path(str(filepath)).parent
        os.makedirs(path_to_create, exist_ok=True)
        torch.save(vid_tensor, filepath)

    def add_text(self, tag, text, global_step: Optional[int] = None):
        if global_step is None:
            global_step = self.videos_counter[tag]
            self.videos_counter[tag] += 1
        filepath = os.path.join(
            self.log_dir, "texts", "".join([tag, str(global_step)]) + ".txt"
        )
        with open(filepath, "w+") as f:
            f.writelines(text)

    def __repr__(self) -> str:
        return f"CSVExperiment(log_dir={self.log_dir})"


class CSVLogger(Logger):
    """A minimal-dependecy CSV-logger.

    Args:
        exp_name (str): The name of the experiment.

    """

    def __init__(self, exp_name: str, log_dir: Optional[str] = None) -> None:
        if log_dir is None:
            log_dir = "csv_logs"
        super().__init__(exp_name=exp_name, log_dir=log_dir)

        self._has_imported_moviepy = False

    def _create_experiment(self) -> "CSVExperiment":
        """Creates a CSV experiment."""
        log_dir = str(os.path.join(self.log_dir, self.exp_name))
        return CSVExperiment(log_dir)

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        """Logs a scalar value to the tensorboard.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged. Defaults to None.
        """
        self.experiment.add_scalar(name, value, global_step=step)

    def log_video(self, name: str, video: Tensor, step: int = None, **kwargs) -> None:
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
        self.experiment.add_video(
            tag=name,
            vid_tensor=video,
            global_step=step,
            **kwargs,
        )

    def log_hparams(self, cfg: "DictConfig") -> None:  # noqa: F821
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig): The configuration of the experiment.
        """
        txt = "\n\t".join([f"{k}: {val}" for k, val in sorted(vars(cfg).items())])
        self.experiment.add_text("hparams", txt)

    def __repr__(self) -> str:
        return f"CSVLogger(exp_name={self.exp_name}, experiment={self.experiment.__repr__()})"
