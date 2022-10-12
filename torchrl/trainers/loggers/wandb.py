# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from typing import Optional

from torch import Tensor

from .common import Logger


try:
    import wandb

    _has_wandb = True
except ImportError:
    _has_wandb = False


try:
    from omegaconf import OmegaConf

    _has_omgaconf = True
except ImportError:
    _has_omgaconf = False


class WandbLogger(Logger):
    """
    Wrapper for the wandb logger.

    Args:
        exp_name (str): The name of the experiment.

    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._prev_video_step = -1
        return super().__new__(cls)

    def __init__(
        self,
        exp_name: str,
        offline: bool = False,
        save_dir: str = None,
        id: str = None,
        project: str = None,
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
        self._wandb_kwargs = {
            "name": exp_name,
            "dir": save_dir,
            "id": id,
            "project": "torchrl-private",
            "entity": "vmoens",
            "resume": "allow",
            **kwargs,
        }
        self._has_imported_wandb = False
        super().__init__(exp_name=exp_name, log_dir=save_dir)
        if self.offline:
            os.environ["WANDB_MODE"] = "dryrun"

        self._has_imported_moviepy = False

        self._has_imported_omgaconf = False

        self.video_log_counter = 0

    def _create_experiment(self) -> "WandbLogger":
        """
        Creates a wandb experiment.

        Args:
            exp_name (str): The name of the experiment.
        Returns:
            WandbLogger: The wandb experiment logger.
        """

        if self.offline:
            os.environ["WANDB_MODE"] = "dryrun"

        if not _has_wandb:
            raise ImportError("Wandb is not installed")
        return wandb.init(**self._wandb_kwargs)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Logs a scalar value to wandb.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
        """
        if step is not None:
            self.experiment.log({name: value, "trainer/step": step})
        else:
            self.experiment.log({name: value})

    def log_video(self, name: str, video: Tensor, **kwargs) -> None:
        """
        Log videos inputs to wandb.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'step' (integer indicating the step index), 'format'
                (default is 'mp4') and 'fps' (default: 6). Other kwargs are
                passed as-is to the `experiment.log` method.
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
        self.video_log_counter += 1
        fps = kwargs.pop("fps", 6)
        step = kwargs.pop("step", None)
        format = kwargs.pop("format", "mp4")
        if step not in (None, self._prev_video_step, self._prev_video_step + 1):
            warnings.warn(
                "when using step with wandb_logger.log_video, it is expected "
                "that the step is equal to the previous step or that value incremented "
                f"by one. Got step={step} but previous value was {self._prev_video_step}. "
                f"The step value will be set to {self._prev_video_step+1}. This warning will "
                f"be silenced from now on but the values will keep being incremented."
            )
            step = self._prev_video_step + 1
        self._prev_video_step = step if step is not None else self._prev_video_step + 1
        self.experiment.log(
            {name: wandb.Video(video, fps=fps, format=format)},
            # step=step,
            **kwargs,
        )

    def log_hparams(self, cfg: "DictConfig") -> None:  # noqa: F821
        """
        Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig): The configuration of the experiment.
        """

        if type(cfg) is not dict and _has_omgaconf:
            if not _has_omgaconf:
                raise ImportError(
                    "OmegaConf could not be imported. "
                    "Cannot log hydra configs without OmegaConf."
                )
            cfg = OmegaConf.to_container(cfg, resolve=True)
        self.experiment.config.update(cfg, allow_val_change=True)

    def __repr__(self) -> str:
        return f"WandbLogger(experiment={self.experiment.__repr__()})"
