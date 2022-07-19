# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from torch import Tensor

from .common import Logger

try:
    import wandb
except ImportError:
    raise ImportError("wandb could not be imported")
try:
    from omegaconf import OmegaConf

    _has_imported_omgaconf = True
except ImportError:
    print("OmegaConf could not be imported. Cannot log hydra configs without OmegaConf")


class WandbLogger(Logger):
    """
    Wrapper for the wandb logger.

    Args:
        exp_name (str): The name of the experiment.

    """

    def __init__(
        self,
        exp_name: str,
        offline: bool = False,
        save_dir: str = None,
        id: str = None,
        project: str = None,
        **kwargs,
    ) -> None:
        self.offline = offline
        self.save_dir = save_dir
        self.id = id
        self.project = project
        self._wandb_kwargs = {
            "name": exp_name,
            "dir": save_dir,
            "id": id,
            "project": project,
            "resume": "allow",
            **kwargs,
        }
        self._has_imported_wandb = False
        super().__init__(exp_name=exp_name)
        if self.offline:
            os.environ["WANDB_MODE"] = "dryrun"
        self.log_dir = save_dir

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

        return wandb.init(**self._wandb_kwargs)

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        """
        Logs a scalar value to wandb.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged. Defaults to None.
        """
        if step is not None:
            self.experiment.log({name: value, "trainer/step": step})
        else:
            self.experiment.log({name: value})

    def log_video(self, name: str, video: Tensor, step: int = None, **kwargs) -> None:
        """
        Log videos inputs to wandb.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            step (int, optional): The step at which the video is logged. Defaults to None.
        """
        if not self._has_imported_moviepy:
            try:
                import moviepy  # noqa

                self._has_imported_moviepy = True
            except ImportError:
                raise Exception(
                    "moviepy not found, videos cannot be logged with TensorboardLogger"
                )
        self.video_log_counter += 1
        if step is not None:
            self.experiment.log(
                {name: wandb.Video(video, fps=6, format="mp4")}, step=step
            )
        else:
            self.experiment.log({name: wandb.Video(video, fps=6, format="mp4")})

    def log_hparams(self, cfg: "DictConfig") -> None:
        """
        Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig): The configuration of the experiment.
        """

        if type(cfg) is not dict and _has_imported_omgaconf:
            cfg = OmegaConf.to_container(cfg, resolve=True)
        self.experiment.config.update(cfg, allow_val_change=True)

    def __repr__(self) -> str:
        return self.experiment.__repr__()
