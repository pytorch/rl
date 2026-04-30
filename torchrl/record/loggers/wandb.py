# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import os
from collections.abc import Sequence
from typing import Any

from tensordict import TensorDictBase
from torch import Tensor

from .common import _make_metrics_safe, Logger

_has_wandb = importlib.util.find_spec("wandb") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None
_has_moviepy = importlib.util.find_spec("moviepy") is not None


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
        save_dir: str | None = None,
        id: str | None = None,
        project: str | None = None,
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
        self._step_registry: dict[str, int] = {}
        self._defined_step_metrics: set[str] = set()
        self._defined_metrics: set[str] = set()
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
        self,
        name: str,
        value: float,
        step: int | None = None,
        commit: bool = False,
        *,
        override_global_step: bool = False,
    ) -> None:
        """Logs a scalar value to wandb.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
            commit: If true, data for current step is assumed to be final (and
                no further data for this step should be logged).
            override_global_step (bool, optional): If ``True``, bypasses
                per-group step injection and forwards ``step`` to wandb's
                global ``step`` argument. Defaults to ``False``.
        """
        self._log_payload(
            {name: value},
            step=step,
            commit=commit,
            override_global_step=override_global_step,
        )

    def log_video(self, name: str, video: Tensor, **kwargs) -> None:
        """Log videos inputs to wandb.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'step' (integer indicating the step index), 'format'
                (default is 'mp4') and 'fps' (defaults to ``self.video_fps``). Other kwargs are
                passed as-is to the :obj:`experiment.log` method.

        Raises:
            ImportError: If moviepy is not installed (required by wandb for video encoding).
        """
        if not _has_moviepy:
            raise ImportError(
                "Video logging with wandb requires moviepy. "
                "Install with: pip install moviepy\n"
                "Or install wandb with media support: pip install 'wandb[media]'"
            )
        import wandb

        fps = kwargs.pop("fps", self.video_fps)
        format = kwargs.pop("format", "mp4")
        self._log_payload(
            {name: wandb.Video(video, fps=fps, format=format)},
            step=kwargs.pop("step", None),
            override_global_step=kwargs.pop("override_global_step", False),
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
        self._log_payload(
            {name: wandb.Histogram(data, num_bins=num_bins)},
            step=step,
            override_global_step=kwargs.pop("override_global_step", False),
            **kwargs,
        )

    def log_str(
        self,
        name: str,
        value: str,
        step: int | None = None,
        *,
        override_global_step: bool = False,
    ) -> None:
        """Logs a string value to wandb using a table format for better visualization.

        Args:
            name (str): The name of the string data.
            value (str): The string value to log.
            step (int, optional): The step at which the string is logged.
                Defaults to None.
            override_global_step (bool, optional): If ``True``, bypasses
                per-group step injection and forwards ``step`` to wandb's
                global ``step`` argument. Defaults to ``False``.
        """
        import wandb

        # Create a table with a single row
        table = wandb.Table(columns=["text"], data=[[value]])
        self._log_payload(
            {name: value if step is not None else table},
            step=step,
            override_global_step=override_global_step,
        )

    def log_metrics(
        self,
        metrics: dict[str, Any] | TensorDictBase,
        step: int | None = None,
        *,
        keys_sep: str = "/",
        override_global_step: bool = False,
    ) -> dict[str, Any]:
        """Log multiple scalar metrics at once to wandb.

        This method efficiently handles tensor values by batching CUDA->CPU
        transfers and performing a single synchronization, then logs all
        metrics in a single wandb API call.

        Args:
            metrics: Dictionary or TensorDict mapping metric names to values.
                Tensor values are automatically converted to Python scalars/lists.
                For TensorDict inputs, nested keys are flattened using ``keys_sep``.
            step: Optional step value for all metrics.
            keys_sep: Separator used to flatten nested TensorDict keys into strings.
                Defaults to "/". Only used for TensorDict inputs.

        Returns:
            The converted metrics dictionary (with tensors converted to Python types).
        """
        safe_metrics = _make_metrics_safe(metrics, keys_sep=keys_sep)
        self._log_payload(
            safe_metrics, step=step, override_global_step=override_global_step
        )
        return safe_metrics

    @staticmethod
    def _is_step_key(name: str) -> bool:
        return name == "step" or name.endswith("/step")

    @staticmethod
    def _step_key(name: str) -> str:
        if WandbLogger._is_step_key(name):
            return name
        prefix, sep, _ = name.rpartition("/")
        return f"{prefix}{sep}step" if sep else "step"

    def _consume_step(self, step_key: str, step: int | None) -> int:
        last_step = self._step_registry.get(step_key, -1)
        if step is None:
            step = last_step + 1
        self._step_registry[step_key] = max(last_step, step)
        return step

    def _define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        if step_metric is None:
            if name in self._defined_step_metrics:
                return
            self._defined_step_metrics.add(name)
            self.experiment.define_metric(name)
            return

        if name in self._defined_metrics:
            return
        self._defined_metrics.add(name)
        self.experiment.define_metric(name, step_metric=step_metric)

    def _prepare_payload(
        self, payload: dict[str, Any], step: int | None
    ) -> dict[str, Any]:
        prepared = dict(payload)

        for key, value in list(prepared.items()):
            if self._is_step_key(key):
                self._consume_step(key, value)

        for key in list(prepared):
            if self._is_step_key(key):
                continue
            step_key = self._step_key(key)
            if step_key not in prepared:
                prepared[step_key] = self._consume_step(step_key, step)

        return prepared

    def _register_metrics(self, payload: dict[str, Any]) -> None:
        for key in payload:
            if self._is_step_key(key):
                self._define_metric(key)
        for key in payload:
            if self._is_step_key(key):
                continue
            self._define_metric(key, step_metric=self._step_key(key))

    def _log_payload(
        self,
        payload: dict[str, Any],
        *,
        step: int | None = None,
        override_global_step: bool = False,
        **kwargs,
    ) -> None:
        if override_global_step:
            self.experiment.log(payload, step=step, **kwargs)
            return

        payload = self._prepare_payload(payload, step)
        self._register_metrics(payload)
        self.experiment.log(payload, **kwargs)
