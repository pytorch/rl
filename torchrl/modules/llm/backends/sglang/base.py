# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Base classes for TorchRL SGLang backends."""

from __future__ import annotations

import abc
from collections.abc import Iterator

import torch


class RLSGLangEngine(abc.ABC):
    """Abstract base class for TorchRL SGLang engines that support weight updates.

    All TorchRL SGLang engines should inherit from this class and implement
    the required methods for weight synchronization.

    The SGLang backend uses HTTP-based communication with the SGLang server
    for generation, and NCCL for weight synchronization in RL training workflows.

    Example:
        >>> # All SGLang engines implement the same interface
        >>> class MySGLangEngine(RLSGLangEngine):
        ...     def get_tp_size(self) -> int:
        ...         return self._tp_size
        ...
        ...     def get_model_metadata(self) -> dict[str, tuple[torch.dtype, torch.Size]]:
        ...         return self._model_metadata
        ...
        ...     # ... implement other abstract methods
    """

    @abc.abstractmethod
    def get_tp_size(self) -> int:
        """Get the tensor parallel size for this engine.

        Returns:
            int: Tensor parallel size
        """

    @abc.abstractmethod
    def get_dp_size(self) -> int:
        """Get the data parallel size for this engine.

        Returns:
            int: Data parallel size
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
    def init_weight_update_group(
        self,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        """Initialize the weight update communication group.

        This should set up NCCL communication for weight broadcasting
        via the SGLang server's /init_weights_update_group API.

        Args:
            master_address: Override for master address. If None, uses default.
            master_port: Override for master port. If None, uses default.
        """

    @abc.abstractmethod
    def update_weights_from_distributed(
        self,
        name: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> None:
        """Signal the server to receive a weight update via NCCL broadcast.

        This coordinates with the SGLang server's /update_weights_from_distributed API
        to receive a single weight tensor broadcasted from the trainer.

        Args:
            name: Name of the parameter to update
            dtype: Data type of the tensor
            shape: Shape of the tensor
        """

    @abc.abstractmethod
    def update_weights(self, weights: Iterator[tuple[str, torch.Tensor]]) -> None:
        """Update model weights from an iterator.

        This method should handle the actual weight broadcasting/updating
        using NCCL communication.

        Args:
            weights: Iterator yielding (parameter_name, tensor) tuples
        """
