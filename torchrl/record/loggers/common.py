# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from typing import Sequence

from torch import Tensor


__all__ = ["Logger"]


class Logger:
    """A template for loggers."""

    def __init__(self, exp_name: str, log_dir: str) -> None:
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.experiment = self._create_experiment()

    @abc.abstractmethod
    def _create_experiment(self) -> Experiment:  # noqa: F821
        ...

    @abc.abstractmethod
    def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
        ...

    @abc.abstractmethod
    def log_video(
        self, name: str, video: Tensor, step: int | None = None, **kwargs
    ) -> None:
        ...

    @abc.abstractmethod
    def log_hparams(self, cfg: DictConfig | dict) -> None:  # noqa: F821
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        ...

    @abc.abstractmethod
    def log_histogram(self, name: str, data: Sequence, **kwargs):
        ...
