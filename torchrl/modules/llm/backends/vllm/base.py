# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Base classes for TorchRL vLLM backends."""

from __future__ import annotations

import abc
from collections.abc import Iterator

import torch


class RLvLLMEngine(abc.ABC):
    """Abstract base class for TorchRL vLLM engines that support weight updates.

    All TorchRL vLLM engines (AsyncVLLM, Ray workers, etc.) should inherit from this
    class and implement the required methods for weight synchronization.
    """

    @abc.abstractmethod
    def get_tp_size(self) -> int:
        """Get the tensor parallel size for this engine.

        Returns:
            int: Tensor parallel size
        """

    @abc.abstractmethod
    def get_model_metadata(self) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Get model parameter metadata.

        Returns:
            dict: Mapping of parameter names to (dtype, shape) tuples
        """

    @abc.abstractmethod
    def get_master_address(self) -> str:
        """Get the master address for weight synchronization.

        Returns:
            str: Master address (e.g., "localhost")
        """

    @abc.abstractmethod
    def get_master_port(self) -> int:
        """Get the master port for weight synchronization.

        Returns:
            int: Master port number
        """

    @abc.abstractmethod
    def init_weight_update_group(self) -> None:
        """Initialize the weight update communication group.

        This should set up NCCL or other communication mechanisms needed
        for weight broadcasting.
        """

    @abc.abstractmethod
    def update_weights(self, weights: Iterator[tuple[str, torch.Tensor]]) -> None:
        """Update model weights from an iterator.

        This method should handle the actual weight broadcasting/updating
        using the engine's internal communication mechanisms.

        Args:
            weights: Iterator yielding (parameter_name, tensor) tuples
        """
