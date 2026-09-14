# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class CheckpointConfig(ConfigBase):
    """Hydra configuration for :class:`~torchrl.checkpoint.Checkpoint`.

    Every kwarg accepted by ``Checkpoint.__init__`` is exposed as a field here.
    Components are registered by the trainer that receives the checkpoint.

    .. seealso::
        :class:`~torchrl.checkpoint.Checkpoint`
    """

    format: str = "directory"
    strict: str = "error"
    archive_compression: str = "stored"
    save_components: list[str] | None = None

    _target_: str = "torchrl.checkpoint.Checkpoint"

    def __post_init__(self) -> None:
        pass


@dataclass
class CheckpointRotationConfig(ConfigBase):
    """Hydra configuration for :class:`~torchrl.checkpoint.CheckpointRotation`.

    ``keep_best`` is a two-item list ``[metadata_key, mode]`` with ``mode`` one
    of ``"min"`` or ``"max"``.

    .. seealso::
        :class:`~torchrl.checkpoint.CheckpointRotation`
    """

    directory: str
    keep_last: int
    keep_best: list[str] | None = None
    prefix: str = "checkpoint"

    _target_: str = "torchrl.checkpoint.CheckpointRotation"

    def __post_init__(self) -> None:
        pass
