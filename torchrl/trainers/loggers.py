# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os

from torch import Tensor


__all__ = ["TensorboardLogger"]


class Logger:
    """
    A template for loggers

    """

    def __init__(self, exp_name: str) -> None:
        self.exp_name = exp_name
        self.experiment = self._create_experiment()

    @abc.abstractmethod
    def _create_experiment(self) -> "Experiment":
        raise NotImplementedError

    @abc.abstractmethod
    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_video(self, name: str, video: Tensor, step: int = None, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_hparams(self, cfg: "DictConfig") -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError


class TensorboardLogger(Logger):
    """
    Wrapper for the Tensoarboard logger.

    Args:
        exp_name (str): The name of the experiment.

    """

    def __init__(self, exp_name: str) -> None:
        super().__init__(exp_name=exp_name)
        self.log_dir = self.experiment.log_dir

        self._has_imported_moviepy = False

    def _create_experiment(self) -> "SummaryWriter":
        """
        Creates a tensorboard experiment.

        Args:
            exp_name (str): The name of the experiment.

        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("torch.utils.tensorboard could not be imported")

        return SummaryWriter(log_dir=self.exp_name)

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        """
        Logs a scalar value to the tensorboard.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged. Defaults to None.
        """
        self.experiment.add_scalar(name, value, global_step=step)

    def log_video(self, name: str, video: Tensor, step: int = None, **kwargs) -> None:
        """
        Log videos inputs to the tensorboard.

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
        self.experiment.add_video(
            tag=name,
            vid_tensor=video,
            global_step=step,
            **kwargs,
        )

    def log_hparams(self, cfg: "DictConfig") -> None:
        """
        Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig): The configuration of the experiment.
        """
        txt = "\n\t".join([f"{k}: {val}" for k, val in sorted(vars(cfg).items())])
        self.experiment.add_text("hparams", txt)

    def __repr__(self) -> str:
        return self.experiment.__repr__()


class WandbLogger(Logger):
    """
    Wrapper for the wandb logger.

    Args:
        exp_name (str): The name of the experiment.

    """

    def __init__(
        self,
        exp_name: str = None,
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
            "offline": offline,
            "save_dir": save_dir,
            "id": id,
            "project": project,
            "resume": "allow",
            **kwargs,
        }
        super().__init__(exp_name=exp_name)
        self.log_dir = self.experiment.save_dir

        self._has_imported_moviepy = False

    def _create_experiment(self) -> "WandbLogger":
        """
        Creates a wandb experiment.

        Args:
            exp_name (str): The name of the experiment.

        """
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb could not be imported")

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
        self.experiment.log({"name": wandb.Video(video, fps=6, format="mp4")})

    def log_hparams(self, cfg: "DictConfig") -> None:
        """
        Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig): The configuration of the experiment.
        """
        self.experiment.config.update(cfg, allow_val_change=True)
