# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from torch import Tensor

from .common import Logger


try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tb = True
except ImportError:
    _has_tb = False


class TensorboardLogger(Logger):
    """Wrapper for the Tensoarboard logger.

    Args:
        exp_name (str): The name of the experiment.
        log_dir (str): the tensorboard log_dir.

    """

    def __init__(self, exp_name: str, log_dir: str = "tb_logs") -> None:
        super().__init__(exp_name=exp_name, log_dir=log_dir)
        # re-write log_dir
        self.log_dir = self.experiment.log_dir

        self._has_imported_moviepy = False

    def _create_experiment(self) -> "SummaryWriter":
        """Creates a tensorboard experiment.

        Args:
            exp_name (str): The name of the experiment.

        Returns:
            SummaryWriter: The tensorboard experiment.

        """
        if not _has_tb:
            raise ImportError("torch.utils.tensorboard could not be imported")

        log_dir = str(os.path.join(self.log_dir, self.exp_name))
        return SummaryWriter(log_dir=log_dir)

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

    def log_hparams(self, cfg: "DictConfig") -> None:  # noqa: F821
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig): The configuration of the experiment.

        """
        txt = "\n\t".join([f"{k}: {val}" for k, val in sorted(vars(cfg).items())])
        self.experiment.add_text("hparams", txt)

    def __repr__(self) -> str:
        return f"TensorboardLogger(experiment={self.experiment.__repr__()})"
