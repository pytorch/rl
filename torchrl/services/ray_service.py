# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torchrl._utils import logger
from torchrl.services.base import ServiceBase

RAY_ERR = None
try:
    import ray

    _has_ray = True
except ImportError as err:
    _has_ray = False
    RAY_ERR = err


class _ServiceRegistryActor:
    """Internal actor that maintains the list of registered services.

    This is a lightweight actor (1 CPU) that tracks which services have been
    registered in a namespace. This ensures we only list our own services,
    not other named actors in Ray.
    """

    def __init__(self):
        self._services: set[str] = set()

    def add(self, name: str) -> None:
        """Add a service to the registry."""
        self._services.add(name)

    def remove(self, name: str) -> None:
        """Remove a service from the registry."""
        self._services.discard(name)

    def list(self) -> list[str]:
        """List all registered services."""
        return sorted(self._services)

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()

    def contains(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services


class RayService(ServiceBase):
    """Ray-based distributed service registry.

    This class uses Ray's named actors feature to provide truly distributed
    service discovery. When a service is registered by any worker, it becomes
    immediately accessible to all other workers in the Ray cluster.

    Services are registered as Ray actors with globally unique names. This
    ensures that:
    1. Services persist independently of the registering worker
    2. All workers see the same services instantly
    3. No custom synchronization is needed

    Args:
        ray_init_config (dict, optional): Configuration for ray.init(). Only
            used if Ray is not already initialized. Common options:
            - address (str): Ray cluster address, or "auto" to auto-detect
            - num_cpus (int): Number of CPUs to use
            - num_gpus (int): Number of GPUs to use
        namespace (str, optional): Ray namespace for service isolation. Services
            in different namespaces are isolated from each other. Defaults to
            "torchrl_services".

    Examples:
        >>> # Basic usage
        >>> services = RayService()
        >>> services.register("tokenizer", TokenizerClass, num_cpus=1)
        >>> tokenizer = services["tokenizer"]
        >>>
        >>> # With Ray options for dynamic configuration
        >>> actor = services.register(
        ...     "model",
        ...     ModelClass,
        ...     num_cpus=2,
        ...     num_gpus=1,
        ...     memory=10 * 1024**3,
        ...     max_concurrency=4
        ... )
        >>>
        >>> # Check and retrieve
        >>> if "tokenizer" in services:
        ...     tok = services["tokenizer"]
        >>>
        >>> # List all services
        >>> print(services.list())
        ['tokenizer', 'model']
    """

    def __init__(
        self,
        ray_init_config: dict | None = None,
        namespace: str = "torchrl_services",
    ):
        if not _has_ray:
            raise ImportError(
                "Ray is required for RayService. Install with: pip install ray"
            ) from RAY_ERR

        self._namespace = namespace
        self._ensure_ray_initialized(ray_init_config)
        self._registry_actor = self._get_or_create_registry_actor()

    def _ensure_ray_initialized(self, ray_init_config: dict | None = None):
        """Initialize Ray if not already initialized."""
        if not ray.is_initialized():
            config = ray_init_config or {}
            # Ensure namespace is set
            if "namespace" not in config:
                config["namespace"] = self._namespace

            logger.info(f"Initializing Ray with namespace '{self._namespace}'")
            ray.init(**config)
        else:
            # Ray already initialized - check if namespace matches
            context = ray.get_runtime_context()
            current_namespace = context.namespace
            if current_namespace != self._namespace:
                logger.warning(
                    f"Ray already initialized with namespace '{current_namespace}', "
                    f"but RayService is using namespace '{self._namespace}'. "
                    f"Services may not be visible across namespaces."
                )

    def _make_service_name(self, name: str) -> str:
        """Create the full actor name with namespace prefix."""
        return f"{self._namespace}::service::{name}"

    def _get_registry_actor_name(self) -> str:
        """Get the name of the registry actor for this namespace."""
        return f"{self._namespace}::_registry"

    def _get_or_create_registry_actor(self):
        """Get or create the registry actor for this namespace."""
        registry_name = self._get_registry_actor_name()

        try:
            # Try to get existing registry
            registry = ray.get_actor(registry_name, namespace=self._namespace)
            return registry
        except ValueError:
            # Registry doesn't exist, create it
            RemoteRegistry = ray.remote(_ServiceRegistryActor)
            registry = RemoteRegistry.options(
                name=registry_name,
                namespace=self._namespace,
                lifetime="detached",
                num_cpus=1,
            ).remote()
            logger.info(
                f"Created service registry actor for namespace '{self._namespace}'"
            )
            return registry

    def register(self, name: str, service_factory: type, *args, **kwargs) -> Any:
        """Register a service and create a named Ray actor.

        This method creates a Ray actor with a globally unique name. The actor
        becomes immediately visible to all workers in the cluster.

        Args:
            name: Service identifier. Must be unique within the namespace.
            service_factory: Class to instantiate as a Ray actor.
            *args: Positional arguments for the service constructor.
            **kwargs: Both Ray actor options (num_cpus, num_gpus, memory, etc.)
                and service constructor arguments. Ray will filter out the actor
                options it recognizes.

        Returns:
            The Ray actor handle.

        Raises:
            ValueError: If a service with this name already exists.

        Examples:
            >>> services = RayService()
            >>>
            >>> # Basic registration
            >>> tokenizer = services.register("tokenizer", TokenizerClass)
            >>>
            >>> # With Ray resource specification
            >>> buffer = services.register(
            ...     "buffer",
            ...     ReplayBuffer,
            ...     num_cpus=2,
            ...     num_gpus=0,
            ...     size=1000000
            ... )
            >>>
            >>> # With advanced Ray options
            >>> model = services.register(
            ...     "model",
            ...     ModelClass,
            ...     num_cpus=4,
            ...     num_gpus=1,
            ...     memory=20 * 1024**3,
            ...     max_concurrency=10,
            ...     max_restarts=3,
            ... )
        """
        full_name = self._make_service_name(name)

        # Check if service already exists in our registry
        if ray.get(self._registry_actor.contains.remote(name)):
            raise ValueError(
                f"Service '{name}' already exists in namespace '{self._namespace}'. "
                f"Use a different name or retrieve the existing service with get()."
            )

        # Create the Ray remote class
        # First, make it a remote class
        remote_cls = ray.remote(service_factory)

        # Then apply options including the name
        options = {
            "name": full_name,
            "namespace": self._namespace,
            "lifetime": "detached",
        }

        # Extract Ray-specific options from kwargs
        ray_options = [
            "num_cpus",
            "num_gpus",
            "memory",
            "object_store_memory",
            "resources",
            "accelerator_type",
            "max_concurrency",
            "max_restarts",
            "max_task_retries",
            "max_pending_calls",
            "scheduling_strategy",
        ]

        for opt in ray_options:
            if opt in kwargs:
                options[opt] = kwargs.pop(opt)

        # Apply options and create the actor
        remote_actor = remote_cls.options(**options).remote(*args, **kwargs)

        # Add to registry
        ray.get(self._registry_actor.add.remote(name))

        logger.info(
            f"Registered service '{name}' as Ray actor '{full_name}' "
            f"with options: {options}"
        )

        return remote_actor

    def get(self, name: str) -> Any:
        """Get a service by name.

        Retrieves a service actor by name. The service can have been registered
        by any worker in the cluster.

        Args:
            name: Service identifier.

        Returns:
            The Ray actor handle.

        Raises:
            KeyError: If the service is not found.

        Examples:
            >>> services = RayService()
            >>> tokenizer = services.get("tokenizer")
            >>> # Use the actor
            >>> result = ray.get(tokenizer.encode.remote("Hello world"))
        """
        # Check registry first
        if not ray.get(self._registry_actor.contains.remote(name)):
            raise KeyError(
                f"Service '{name}' not found in namespace '{self._namespace}'. "
                f"Available services: {self.list()}"
            )

        full_name = self._make_service_name(name)

        try:
            actor = ray.get_actor(full_name, namespace=self._namespace)
            return actor
        except ValueError as e:
            # Service in registry but actor missing - inconsistency
            logger.warning(
                f"Service '{name}' in registry but actor not found. "
                f"Removing from registry."
            )
            ray.get(self._registry_actor.remove.remote(name))
            raise KeyError(
                f"Service '{name}' actor not found (removed from registry). "
                f"Available services: {self.list()}"
            ) from e

    def __contains__(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name: Service identifier.

        Returns:
            True if the service exists, False otherwise.

        Examples:
            >>> services = RayService()
            >>> if "tokenizer" in services:
            ...     tokenizer = services["tokenizer"]
            ... else:
            ...     services.register("tokenizer", TokenizerClass)
        """
        return ray.get(self._registry_actor.contains.remote(name))

    def list(self) -> list[str]:
        """List all registered service names.

        Returns a list of all services in the current namespace. This includes
        services registered by any worker.

        Returns:
            List of service names (without namespace prefix).

        Examples:
            >>> services = RayService()
            >>> services.register("tokenizer", TokenizerClass)
            >>> services.register("buffer", ReplayBuffer)
            >>> print(services.list())
            ['buffer', 'tokenizer']
        """
        return ray.get(self._registry_actor.list.remote())

    def reset(self) -> None:
        """Reset the service registry by terminating all actors.

        This method:
        1. Terminates all service actors in the current namespace
        2. Clears the registry actor's internal state

        After calling reset(), all services will be removed and their actors
        will be killed. Any ongoing work will be interrupted.

        Warning:
            This is a destructive operation that affects all workers in the
            namespace. Use with caution.

        Examples:
            >>> services = RayService(namespace="experiment")
            >>> services.register("tokenizer", TokenizerClass)
            >>> print(services.list())
            ['tokenizer']
            >>> services.reset()
            >>> print(services.list())
            []
        """
        service_names = self.list()

        for name in service_names:
            full_name = self._make_service_name(name)
            try:
                actor = ray.get_actor(full_name, namespace=self._namespace)
                ray.kill(actor)
                logger.info(f"Terminated service '{name}' (actor: {full_name})")
            except ValueError:
                # Actor already gone or doesn't exist
                logger.warning(f"Service '{name}' not found during reset")
            except Exception as e:
                logger.warning(f"Failed to terminate service '{name}': {e}")

        # Clear the registry
        ray.get(self._registry_actor.clear.remote())

        logger.info(
            f"Reset complete for namespace '{self._namespace}'. Terminated {len(service_names)} services."
        )

    def shutdown(self, raise_on_error: bool = True) -> None:
        """Shutdown the RayService by shutting down the Ray cluster."""
        try:
            self.reset()
            # kill the registry actor
            registry_actor = ray.get_actor(
                self._get_registry_actor_name(), namespace=self._namespace
            )
            ray.kill(registry_actor, no_restart=True)
        except Exception as e:
            if raise_on_error:
                raise e
            else:
                logger.warning(f"Error shutting down RayService: {e}")

    def register_with_options(
        self,
        name: str,
        service_factory: type,
        actor_options: dict[str, Any],
        **constructor_kwargs,
    ) -> Any:
        """Register a service with explicit separation of Ray options and constructor args.

        This is a convenience method that makes it explicit which arguments are for
        Ray actor configuration vs. the service constructor. It's functionally
        equivalent to `register()` but more readable for complex configurations.

        Args:
            name: Service identifier.
            service_factory: Class to instantiate as a Ray actor.
            actor_options: Dictionary of Ray actor options (num_cpus, num_gpus, etc.).
            **constructor_kwargs: Arguments to pass to the service constructor.

        Returns:
            The Ray actor handle.

        Examples:
            >>> services = RayService()
            >>>
            >>> # Explicit separation of concerns
            >>> model = services.register_with_options(
            ...     "model",
            ...     ModelClass,
            ...     actor_options={
            ...         "num_cpus": 4,
            ...         "num_gpus": 1,
            ...         "memory": 20 * 1024**3,
            ...         "max_concurrency": 10
            ...     },
            ...     model_path="/path/to/checkpoint",
            ...     batch_size=32
            ... )
            >>>
            >>> # Equivalent to:
            >>> # services.register(
            >>> #     "model", ModelClass,
            >>> #     num_cpus=4, num_gpus=1, memory=20*1024**3, max_concurrency=10,
            >>> #     model_path="/path/to/checkpoint", batch_size=32
            >>> # )
        """
        # Merge actor_options into kwargs for register()
        merged_kwargs = {**actor_options, **constructor_kwargs}
        return self.register(name, service_factory, **merged_kwargs)
