# ServiceBase

*class*torchrl.services.ServiceBase[[source]](../../_modules/torchrl/services/base.html#ServiceBase)

Base class for distributed service registries.

A service registry manages distributed actors/services that can be accessed
across multiple workers. Common use cases include:

- Tokenizers shared across inference workers
- Replay buffers for distributed training
- Model registries for centralized model storage
- Metrics aggregators

The registry provides a dict-like interface for registering and accessing
services by name.

*abstract*get(*name: str*) → Any[[source]](../../_modules/torchrl/services/base.html#ServiceBase.get)

Get a service by name.

Retrieves a previously registered service. If the service was registered
by another worker, this method will find it in the distributed registry.

Parameters:

**name** - Service identifier.

Returns:

The remote actor handle for the service.

Raises:

**KeyError** - If the service is not found.

get_client(*name: str*) → Any[[source]](../../_modules/torchrl/services/base.html#ServiceBase.get_client)

Get the restricted client for a registered [`Service`](torchrl.services.Service.html#torchrl.services.Service).

This method is additive so existing custom registry backends remain
instantiable. Backends that support owner/client discovery override it.

Parameters:

**name** - Service identifier.

Raises:

**NotImplementedError** - If this registry backend does not support
 service-client discovery.

*abstract*list() → list[str][[source]](../../_modules/torchrl/services/base.html#ServiceBase.list)

List all registered service names.

Returns:

List of service names currently registered in the cluster.

*abstract*register(*name: str*, *service_factory: type*, **args*, ***kwargs*) → Any[[source]](../../_modules/torchrl/services/base.html#ServiceBase.register)

Register a service factory and create the service actor.

This method registers a service with the given name and immediately
creates the corresponding actor. The service becomes globally visible
to all workers in the cluster.

Parameters:

- **name** - Unique identifier for the service. This name is used to
retrieve the service later.
- **service_factory** - Class to instantiate as a remote actor.
- ***args** - Positional arguments to pass to the service constructor.
- ****kwargs** - Keyword arguments for both actor configuration and
service constructor. Actor configuration options are backend-specific
(e.g., num_cpus, num_gpus for Ray).

Returns:

The remote actor handle.

Raises:

**ValueError** - If a service with this name already exists.

*abstract*reset() → None[[source]](../../_modules/torchrl/services/base.html#ServiceBase.reset)

Reset the service registry.

This removes all registered services and cleans up registry-owned
resources. Externally-owned [`Service`](torchrl.services.Service.html#torchrl.services.Service) instances are removed from
discovery but are never shut down by the registry.

Warning

This is a destructive operation. All services will be terminated and
any ongoing work will be interrupted.