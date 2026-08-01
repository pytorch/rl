# get_services

*class*torchrl.services.get_services(*backend: str = 'ray'*, ***init_kwargs*)[[source]](../../_modules/torchrl/services.html#get_services)

Get a distributed service registry.

This function creates or retrieves a service registry for managing distributed
actors across workers. Services registered by one worker are immediately visible
to all other workers in the cluster.

Parameters:

- **backend** - Service backend to use. Currently only "ray" is supported.
- ****init_kwargs** -

Backend-specific initialization arguments.
For Ray:

- ray_init_config (dict, optional): Arguments to pass to ray.init()
- namespace (str, optional): Ray namespace for service isolation.
Defaults to "torchrl_services".

Returns:

A service registry instance.

Return type:

[ServiceBase](torchrl.services.ServiceBase.html#torchrl.services.ServiceBase)

Raises:

- **ValueError** - If an unsupported backend is specified.
- **ImportError** - If the required backend library is not installed.

Examples

```
>>> # Basic usage - register and access services
>>> services = get_services()
>>> services.register("tokenizer", TokenizerClass, num_cpus=1)
>>> tokenizer = services["tokenizer"]
>>>
>>> # With custom Ray initialization
>>> services = get_services(
... backend="ray",
... ray_init_config={"address": "auto"},
... namespace="my_experiment"
... )
>>>
>>> # Check if service exists
>>> if "tokenizer" in services:
... tokenizer = services["tokenizer"]
>>>
>>> # List all registered services
>>> service_names = services.list()
```