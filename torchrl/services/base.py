# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ServiceBase(ABC):
    """Base class for distributed service registries.

    A service registry manages distributed actors/services that can be accessed
    across multiple workers. Common use cases include:

    - Tokenizers shared across inference workers
    - Replay buffers for distributed training
    - Model registries for centralized model storage
    - Metrics aggregators

    The registry provides a dict-like interface for registering and accessing
    services by name.
    """

    @abstractmethod
    def register(self, name: str, service_factory: type, *args, **kwargs) -> Any:
        """Register a service factory and create the service actor.

        This method registers a service with the given name and immediately
        creates the corresponding actor. The service becomes globally visible
        to all workers in the cluster.

        Args:
            name: Unique identifier for the service. This name is used to
                retrieve the service later.
            service_factory: Class to instantiate as a remote actor.
            *args: Positional arguments to pass to the service constructor.
            **kwargs: Keyword arguments for both actor configuration and
                service constructor. Actor configuration options are backend-specific
                (e.g., num_cpus, num_gpus for Ray).

        Returns:
            The remote actor handle.

        Raises:
            ValueError: If a service with this name already exists.
        """

    @abstractmethod
    def get(self, name: str) -> Any:
        """Get a service by name.

        Retrieves a previously registered service. If the service was registered
        by another worker, this method will find it in the distributed registry.

        Args:
            name: Service identifier.

        Returns:
            The remote actor handle for the service.

        Raises:
            KeyError: If the service is not found.
        """

    @abstractmethod
    def __contains__(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name: Service identifier.

        Returns:
            True if the service exists, False otherwise.
        """

    @abstractmethod
    def list(self) -> list[str]:
        """List all registered service names.

        Returns:
            List of service names currently registered in the cluster.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the service registry.

        This removes all registered services and cleans up associated resources.
        After calling reset(), the registry will be empty and all service actors
        will be terminated.

        Warning:
            This is a destructive operation. All services will be terminated and
            any ongoing work will be interrupted.
        """

    def __getitem__(self, name: str) -> Any:
        """Dict-like access: services["tokenizer"]."""
        return self.get(name)

    def __setitem__(self, name: str, service_factory: type) -> None:
        """Dict-like registration: services["tokenizer"] = TokenizerClass.

        Note: This only supports service_factory without additional arguments.
        For full control, use register() method instead.
        """
        self.register(name, service_factory)
