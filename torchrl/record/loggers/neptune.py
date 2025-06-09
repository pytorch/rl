# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor

from .common import Logger

_has_neptune = importlib.util.find_spec("neptune") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None
_has_moviepy = importlib.util.find_spec("moviepy") is not None


class NeptuneLogger(Logger):
    """Wrapper for the Neptune logger.

    Args:
        exp_name (str): The name of the experiment.
        project (str, optional): Name of a project in the form workspace-name/project-name.
            If None, the value of the NEPTUNE_PROJECT environment variable is used.
        api_token (str, optional): Your Neptune API token. If None, the value of the
            NEPTUNE_API_TOKEN environment variable is used.
        log_dir (str, optional): The directory where to save data.
        offline (bool, optional): If True, the logs will be stored locally only.
            Defaults to False.

    Keyword Args:
        **kwargs: Extra keyword arguments for neptune.init_run(). See relevant page for
            more info.
    """

    def __init__(
        self,
        exp_name: str,
        project: Optional[str] = None,
        api_token: Optional[str] = None,
        log_dir: Optional[str] = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        if not _has_neptune:
            raise ImportError("neptune could not be imported")

        self.offline = offline
        if self.offline:
            os.environ["NEPTUNE_MODE"] = "offline"

        self._neptune_kwargs = {
            "name": exp_name,
            "project": project,
            "api_token": api_token,
            **kwargs,
        }
        super().__init__(exp_name=exp_name, log_dir=log_dir or "neptune_logs")

    def _create_experiment(self) -> Any:  # noqa
        """Creates a Neptune run.

        Returns:
            neptune.Run: The Neptune run object.
        """
        if not _has_neptune:
            raise ImportError("neptune could not be imported")
        import neptune

        return neptune.init_run(**self._neptune_kwargs)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Logs a scalar value to Neptune.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
        """
        if step is not None:
            self.experiment[name].append(value, step=step)
        else:
            self.experiment[name].append(value)

    def log_video(
        self, name: str, video: Tensor, step: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Log videos inputs to Neptune.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            step (int, optional): The step at which the video is logged.
                Defaults to None.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'format' (default is 'mp4') and 'fps' (defaults to 30).
        """
        if not _has_moviepy:
            raise ImportError("moviepy could not be imported")
        import moviepy.editor as mpy

        # check for correct format of the video tensor ((N), T, C, H, W)
        # check that the color channel (C) is either 1 or 3
        if video.dim() != 5 or video.size(dim=2) not in {1, 3}:
            raise Exception(
                "Wrong format of the video tensor. Should be ((N), T, C, H, W)"
            )

        # Convert tensor to numpy array and scale to [0, 255]
        video_np = (video.cpu().numpy() * 255).astype(np.uint8)
        # Transpose to (T, H, W, C) format for moviepy
        video_np = video_np.transpose(0, 3, 4, 2)

        # Create a temporary file to save the video
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            # Create a moviepy clip and write to file
            clip = mpy.ImageSequenceClip(list(video_np), fps=kwargs.get("fps", 30))
            clip.write_videofile(temp_file.name, codec="libx264", audio=False)
            
            # Log the video file to Neptune
            if step is not None:
                self.experiment[name].upload(temp_file.name, step=step)
            else:
                self.experiment[name].upload(temp_file.name)
            
            # Clean up
            os.unlink(temp_file.name)

    def log_hparams(self, cfg: Union[DictConfig, Dict[str, Any]]) -> None:
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig or dict): The configuration of the experiment.
        """
        if not isinstance(cfg, dict) and _has_omegaconf:
            if not _has_omegaconf:
                raise ImportError(
                    "OmegaConf could not be imported. "
                    "Cannot log hydra configs without OmegaConf."
                )
            from omegaconf import OmegaConf

            cfg = OmegaConf.to_container(cfg, resolve=True)
        self.experiment["parameters"] = cfg

    def __repr__(self) -> str:
        return f"NeptuneLogger(experiment={self.experiment.__repr__()})"

    def log_histogram(self, name: str, data: Union[Tensor, NDArray, Sequence], **kwargs: Any):
        """Log histogram data to Neptune.

        Args:
            name (str): The name of the histogram.
            data (Sequence): Values to build histogram from.
            **kwargs: Additional arguments for histogram creation.
                Supports 'step' (int) and 'bins' (int).
        """
        if not _has_neptune:
            raise ImportError("neptune could not be imported")
        import numpy as np
        from neptune.types import Histogram

        step = kwargs.get("step", None)
        bins = kwargs.get("bins", None)

        # Convert data to numpy array
        if isinstance(data, Tensor):
            data = data.cpu().detach().numpy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)

        # Create histogram data
        hist_values, bin_edges = np.histogram(data, bins=bins)
        
        # Create Neptune Histogram object
        histogram = Histogram(
            bin_edges=bin_edges.tolist(),
            counts=hist_values.tolist()
        )
        
        # Log histogram data
        if step is not None:
            self.experiment[name].append(histogram, step=step)
        else:
            self.experiment[name].append(histogram) 