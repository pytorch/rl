# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

from torch import Tensor


__all__ = ["Logger"]


class Logger:
    """A template for loggers."""

    def __init__(self, exp_name: str, log_dir: str) -> None:
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.experiment = self._create_experiment()

    @abc.abstractmethod
    def _create_experiment(self) -> "Experiment":  # noqa: F821
        raise NotImplementedError

    @abc.abstractmethod
    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_video(self, name: str, video: Tensor, step: int = None, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def log_hparams(self, cfg: "DictConfig") -> None:  # noqa: F821
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
