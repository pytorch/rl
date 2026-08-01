# RayService

*class*torchrl.services.RayService(*ray_init_config: dict | None = None*, *namespace: str = 'torchrl_services'*)[[source]](../../_modules/torchrl/services/ray_service.html#RayService)

Ray-based distributed service registry.

This class uses Ray's named actors feature to provide truly distributed
service discovery. When a service is registered by any worker, it becomes
immediately accessible to all other workers in the Ray cluster.

Services are registered as Ray actors with globally unique names. This
ensures that:
1. Services persist independently of the registering worker
2. All workers see the same services instantly
3. No custom synchronization is needed

Parameters:

- **ray_init_config** (*dict**,**optional*) - Configuration for ray.init(). Only
used if Ray is not already initialized. Common options:
- address (str): Ray cluster address, or "auto" to auto-detect
- num_cpus (int): Number of CPUs to use
- num_gpus (int): Number of GPUs to use
- **namespace** (*str**,**optional*) - Ray namespace for service isolation. Services
in different namespaces are isolated from each other. Defaults to
"torchrl_services".

Examples

```
>>> # Basic usage
>>> services = RayService()
>>> services.register("tokenizer", TokenizerClass, num_cpus=1)
>>> tokenizer = services["tokenizer"]
>>>
>>> # With Ray options for dynamic configuration
>>> actor = services.register(
... "model",
... ModelClass,
... num_cpus=2,
... num_gpus=1,
... memory=10 * 1024**3,
... max_concurrency=4
... )
>>>
>>> # Check and retrieve
>>> if "tokenizer" in services:
... tok = services["tokenizer"]
>>>
>>> # List all services
>>> print(services.list())
['tokenizer', 'model']
```

get(*name: str*) → Any[[source]](../../_modules/torchrl/services/ray_service.html#RayService.get)

Get a service by name.

Retrieves a service actor by name. The service can have been registered
by any worker in the cluster.

Parameters:

**name** - Service identifier.

Returns:

The Ray actor handle.

Raises:

**KeyError** - If the service is not found.

Examples

```
>>> services = RayService()
>>> tokenizer = services.get("tokenizer")
>>> # Use the actor
>>> result = ray.get(tokenizer.encode.remote("Hello world"))
```

list() → list[str][[source]](../../_modules/torchrl/services/ray_service.html#RayService.list)

List all registered service names.

Returns a list of all services in the current namespace. This includes
services registered by any worker.

Returns:

List of service names (without namespace prefix).

Examples

```
>>> services = RayService()
>>> services.register("tokenizer", TokenizerClass)
>>> services.register("buffer", ReplayBuffer)
>>> print(services.list())
['buffer', 'tokenizer']
```

register(*name: str*, *service_factory: type*, **args*, ***kwargs*) → Any[[source]](../../_modules/torchrl/services/ray_service.html#RayService.register)

Register a service and create a named Ray actor.

This method creates a Ray actor with a globally unique name. The actor
becomes immediately visible to all workers in the cluster.

Parameters:

- **name** - Service identifier. Must be unique within the namespace.
- **service_factory** - Class to instantiate as a Ray actor.
- ***args** - Positional arguments for the service constructor.
- ****kwargs** - Both Ray actor options (num_cpus, num_gpus, memory, etc.)
and service constructor arguments. Ray will filter out the actor
options it recognizes.

Returns:

The Ray actor handle.

Raises:

**ValueError** - If a service with this name already exists.

Examples

```
>>> services = RayService()
>>>
>>> # Basic registration
>>> tokenizer = services.register("tokenizer", TokenizerClass)
>>>
>>> # With Ray resource specification
>>> buffer = services.register(
... "buffer",
... ReplayBuffer,
... num_cpus=2,
... num_gpus=0,
... size=1000000
... )
>>>
>>> # With advanced Ray options
>>> model = services.register(
... "model",
... ModelClass,
... num_cpus=4,
... num_gpus=1,
... memory=20 * 1024**3,
... max_concurrency=10,
... max_restarts=3,
... )
```

register_with_options(*name: str*, *service_factory: type*, *actor_options: dict[str, Any]*, ***constructor_kwargs*) → Any[[source]](../../_modules/torchrl/services/ray_service.html#RayService.register_with_options)

Register a service with explicit separation of Ray options and constructor args.

This is a convenience method that makes it explicit which arguments are for
Ray actor configuration vs. the service constructor. It's functionally
equivalent to register() but more readable for complex configurations.

Parameters:

- **name** - Service identifier.
- **service_factory** - Class to instantiate as a Ray actor.
- **actor_options** - Dictionary of Ray actor options (num_cpus, num_gpus, etc.).
- ****constructor_kwargs** - Arguments to pass to the service constructor.

Returns:

The Ray actor handle.

Examples

```
>>> services = RayService()
>>>
>>> # Explicit separation of concerns
>>> model = services.register_with_options(
... "model",
... ModelClass,
... actor_options={
... "num_cpus": 4,
... "num_gpus": 1,
... "memory": 20 * 1024**3,
... "max_concurrency": 10
... },
... model_path="/path/to/checkpoint",
... batch_size=32
... )
>>>
>>> # Equivalent to:
>>> # services.register(
>>> # "model", ModelClass,
>>> # num_cpus=4, num_gpus=1, memory=20*1024**3, max_concurrency=10,
>>> # model_path="/path/to/checkpoint", batch_size=32
>>> # )
```

reset() → None[[source]](../../_modules/torchrl/services/ray_service.html#RayService.reset)

Reset the service registry by terminating all actors.

This method:
1. Terminates all service actors in the current namespace
2. Clears the registry actor's internal state

After calling reset(), all services will be removed and their actors
will be killed. Any ongoing work will be interrupted.

Warning

This is a destructive operation that affects all workers in the
namespace. Use with caution.

Examples

```
>>> services = RayService(namespace="experiment")
>>> services.register("tokenizer", TokenizerClass)
>>> print(services.list())
['tokenizer']
>>> services.reset()
>>> print(services.list())
[]
```

shutdown(*raise_on_error: bool = True*) → None[[source]](../../_modules/torchrl/services/ray_service.html#RayService.shutdown)

Shutdown the RayService by shutting down the Ray cluster.