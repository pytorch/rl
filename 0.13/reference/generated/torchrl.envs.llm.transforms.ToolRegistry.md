# ToolRegistry

*class*torchrl.envs.llm.transforms.ToolRegistry(*services: Sequence[[ToolService](torchrl.envs.llm.transforms.ToolService.html#torchrl.envs.llm.transforms.ToolService)] = ()*)[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#ToolRegistry)

Registry for managing available tool services.

This class maintains a collection of tool services that can be looked up
by name for execution.

Parameters:

**services** (*Sequence**[*[*ToolService*](torchrl.envs.llm.transforms.ToolService.html#torchrl.envs.llm.transforms.ToolService)*]**,**optional*) - Initial services to register.
Defaults to an empty sequence.

Examples

```
>>> class AddService:
... name = "add"
... schema_in = {"a": int, "b": int}
... schema_out = {"result": int}
... def __call__(self, a, b, **kwargs):
... return {"result": a + b}
>>> registry = ToolRegistry([AddService()])
>>> service = registry.get("add")
>>> result = service(a=1, b=2)
>>> print(result)
{"result": 3}
```

get(*name: str*) → [ToolService](torchrl.envs.llm.transforms.ToolService.html#torchrl.envs.llm.transforms.ToolService)[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#ToolRegistry.get)

Retrieve a service by name.

Parameters:

**name** (*str*) - The name of the service to retrieve.

Returns:

The requested service.

Return type:

[ToolService](torchrl.envs.llm.transforms.ToolService.html#torchrl.envs.llm.transforms.ToolService)

Raises:

**KeyError** - If the service is not found.

register(*service: [ToolService](torchrl.envs.llm.transforms.ToolService.html#torchrl.envs.llm.transforms.ToolService)*) → None[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#ToolRegistry.register)

Register a new service.

Parameters:

**service** ([*ToolService*](torchrl.envs.llm.transforms.ToolService.html#torchrl.envs.llm.transforms.ToolService)) - The service to register.