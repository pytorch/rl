# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import tensordict.utils
import torch
from tensordict import MemoryMappedTensor
from torch import Tensor

from .common import Logger


class CSVExperiment:
    """A CSV logger experiment class."""

    def __init__(self, log_dir: str, *, video_format="pt", video_fps: int = 30):
        self.scalars = defaultdict(lambda: [])
        self.videos_counter = defaultdict(lambda: 0)
        self.text_counter = defaultdict(lambda: 0)
        self.log_dir = log_dir
        self.video_format = video_format
        self.video_fps = video_fps
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "scalars"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "texts"), exist_ok=True)

        self.files = {}

    def add_scalar(self, name: str, value: float, global_step: int | None = None):
        if global_step is None:
            global_step = len(self.scalars[name])
        value = float(value)
        self.scalars[name].append((global_step, value))
        filepath = os.path.join(self.log_dir, "scalars", "".join([name, ".csv"]))
        if not os.path.isfile(filepath):
            os.makedirs(Path(filepath).parent, exist_ok=True)
        if filepath not in self.files:
            os.makedirs(Path(filepath).parent, exist_ok=True)
            self.files[filepath] = open(filepath, "a+")
        fd = self.files[filepath]
        fd.write(",".join([str(global_step), str(value)]) + "\n")
        fd.flush()

    def add_video(self, tag, vid_tensor, global_step: int | None = None, **kwargs):
        """Writes a video on a file on disk.

        The video format can be one of

        - `"pt"`: uses :func:`~torch.save` to save the video tensor);
        - `"memmap"`: saved the file as memory-mapped array (reading this file will require
          the dtype and shape to be known at read time);
        - `"mp4"`: saves the file as an `.mp4` file using torchvision :func:`~torchvision.io.write_video`
          API. Any ``kwargs`` passed to ``add_video`` will be transmitted to ``write_video``.
          These include ``preset``, ``crf`` and others.
          See ffmpeg's doc (https://trac.ffmpeg.org/wiki/Encode/H.264) for some more information of the video format options.

        """
        if global_step is None:
            global_step = self.videos_counter[tag]
            self.videos_counter[tag] += 1
        if self.video_format == "pt":
            extension = ".pt"
        elif self.video_format == "memmap":
            extension = ".memmap"
        elif self.video_format == "mp4":
            extension = ".mp4"
        else:
            raise ValueError(
                f"Unknown video format {self.video_format}. Must be one of 'pt', 'memmap' or 'mp4'."
            )

        filepath = os.path.join(
            self.log_dir, "videos", "_".join([tag, str(global_step)]) + extension
        )
        path_to_create = Path(str(filepath)).parent
        os.makedirs(path_to_create, exist_ok=True)
        if self.video_format == "pt":
            torch.save(vid_tensor, filepath)
        elif self.video_format == "memmap":
            MemoryMappedTensor.from_tensor(vid_tensor, filename=filepath)
        elif self.video_format == "mp4":
            import torchvision

            if vid_tensor.shape[-3] not in (3, 1):
                raise RuntimeError(
                    "expected the video tensor to be of format [T, C, H, W] but the third channel "
                    f"starting from the end isn't in (1, 3) but is {vid_tensor.shape[-3]}."
                )
            if vid_tensor.ndim > 4:
                vid_tensor = vid_tensor.flatten(0, vid_tensor.ndim - 4)
            vid_tensor = vid_tensor.permute((0, 2, 3, 1))
            vid_tensor = vid_tensor.expand(*vid_tensor.shape[:-1], 3)
            kwargs.setdefault("fps", self.video_fps)
            torchvision.io.write_video(filepath, vid_tensor, **kwargs)
        else:
            raise ValueError(
                f"Unknown video format {self.video_format}. Must be one of 'pt', 'memmap' or 'mp4'."
            )

    def add_text(self, tag, text, global_step: int | None = None):
        if global_step is None:
            global_step = self.videos_counter[tag]
            self.videos_counter[tag] += 1
        filepath = os.path.join(
            self.log_dir, "texts", "".join([tag, str(global_step)]) + ".txt"
        )
        if not os.path.isfile(filepath):
            os.makedirs(Path(filepath).parent, exist_ok=True)
        if filepath not in self.files:
            self.files[filepath] = open(filepath, "w+")
        fd = self.files[filepath]
        fd.writelines(text)
        fd.flush()

    def __repr__(self) -> str:
        return f"CSVExperiment(log_dir={self.log_dir})"

    def __del__(self):
        for val in getattr(self, "files", {}).values():
            val.close()


class CSVLogger(Logger):
    """A minimal-dependency CSV logger.

    Args:
        exp_name (str): The name of the experiment.
        log_dir (str or Path, optional): where the experiment should be saved.
            Defaults to ``<cur_dir>/csv_logs``.
        video_format (str, optional): how videos should be saved when calling :meth:`~torchrl.record.loggers.csv.CSVExperiment.add_video`. Must be one of
            ``"pt"`` (video saved as a `video_<tag>_<step>.pt` file with torch.save),
            ``"memmap"`` (video saved as a `video_<tag>_<step>.memmap` file with :class:`~tensordict.MemoryMappedTensor`),
            ``"mp4"`` (video saved as a `video_<tag>_<step>.mp4` file, requires torchvision to be installed).
            Defaults to ``"pt"``.
        video_fps (int, optional): the video frames-per-seconds if `video_format="mp4"`. Defaults to 30.

    """

    experiment: CSVExperiment

    def __init__(
        self,
        exp_name: str,
        log_dir: str | None = None,
        video_format: str = "pt",
        video_fps: int = 30,
    ) -> None:
        if log_dir is None:
            log_dir = "csv_logs"
        self.video_format = video_format
        self.video_fps = video_fps
        super().__init__(exp_name=exp_name, log_dir=log_dir)
        self._has_imported_moviepy = False

    def _create_experiment(self) -> CSVExperiment:
        """Creates a CSV experiment."""
        log_dir = str(os.path.join(self.log_dir, self.exp_name))
        return CSVExperiment(
            log_dir, video_format=self.video_format, video_fps=self.video_fps
        )

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        """Logs a scalar value to the tensorboard.

        Args:
            name (str): The name of the scalar.
            value (:obj:`float`): The value of the scalar.
            step (int, optional): The step at which the scalar is logged. Defaults to None.
        """
        self.experiment.add_scalar(name, value, global_step=step)

    def log_video(self, name: str, video: Tensor, step: int = None, **kwargs) -> None:
        """Log videos inputs to a .pt (or other format) file.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            step (int, optional): The step at which the video is logged. Defaults to None.
            **kwargs: other kwargs passed to the underlying video logger.

        .. note:: If the video format is `mp4`, many more arguments can be passed to the :meth:`~torchvision.io.write_video`
            function.
            For more information on video logging with :class:`~torchrl.record.loggers.csv.CSVLogger`,
            see the :meth:`~torchrl.record.loggers.csv.CSVExperiment.add_video` documentation.
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

    def log_hparams(self, cfg: DictConfig | dict) -> None:  # noqa: F821
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig or dict): The configuration of the experiment.
        """
        txt = "\n".join([f"{k}: {val}" for k, val in sorted(cfg.items())])
        self.experiment.add_text("hparams", txt)

    def __repr__(self) -> str:
        return f"CSVLogger(exp_name={self.exp_name}, experiment={self.experiment.__repr__()})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        raise NotImplementedError("Logging histograms in cvs is not permitted.")

    def print_log_dir(self):
        """Prints the log directory content."""
        tensordict.utils.print_directory_tree(self.log_dir)
