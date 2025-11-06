# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed service registry for TorchRL.

This module provides a service registry for managing distributed actors
(tokenizers, replay buffers, etc.) that can be accessed across workers.

Example:
    >>> from torchrl.services import get_services
    >>>
    >>> # Worker 1: Register a tokenizer service
    >>> services = get_services()
    >>> services.register("tokenizer", TokenizerClass, num_cpus=1, num_gpus=0.1)
    >>>
    >>> # Worker 2: Access the same tokenizer
    >>> services = get_services()
    >>> tokenizer = services["tokenizer"]
    >>> result = tokenizer.encode.remote(text)
"""
from __future__ import annotations

from torchrl.services.base import ServiceBase
from torchrl.services.ray_service import RayService

__all__ = ["ServiceBase", "RayService", "get_services"]


def get_services(backend: str = "ray", **init_kwargs) -> ServiceBase:
    """Get a distributed service registry.

    This function creates or retrieves a service registry for managing distributed
    actors across workers. Services registered by one worker are immediately visible
    to all other workers in the cluster.

    Args:
        backend: Service backend to use. Currently only "ray" is supported.
        **init_kwargs: Backend-specific initialization arguments.
            For Ray:
                - ray_init_config (dict, optional): Arguments to pass to ray.init()
                - namespace (str, optional): Ray namespace for service isolation.
                    Defaults to "torchrl_services".

    Returns:
        ServiceBase: A service registry instance.

    Raises:
        ValueError: If an unsupported backend is specified.
        ImportError: If the required backend library is not installed.

    Examples:
        >>> # Basic usage - register and access services
        >>> services = get_services()
        >>> services.register("tokenizer", TokenizerClass, num_cpus=1)
        >>> tokenizer = services["tokenizer"]
        >>>
        >>> # With custom Ray initialization
        >>> services = get_services(
        ...     backend="ray",
        ...     ray_init_config={"address": "auto"},
        ...     namespace="my_experiment"
        ... )
        >>>
        >>> # Check if service exists
        >>> if "tokenizer" in services:
        ...     tokenizer = services["tokenizer"]
        >>>
        >>> # List all registered services
        >>> service_names = services.list()
    """
    if backend != "ray":
        raise ValueError(
            f"Unsupported backend: {backend}. Currently only 'ray' is supported."
        )

    return RayService(**init_kwargs)
