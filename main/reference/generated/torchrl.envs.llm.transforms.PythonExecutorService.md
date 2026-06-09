# PythonExecutorService

*class*torchrl.envs.llm.transforms.PythonExecutorService(*pool_size: int = 32*, *timeout: float = 10.0*)[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#PythonExecutorService)

Ray actor that manages a pool of persistent Python interpreters.

This service allows multiple environments to share a pool of Python
interpreters, reducing resource usage and improving efficiency.

Parameters:

- **pool_size** (*int*) - Number of Python interpreter processes to maintain.
- **timeout** (*float*) - Timeout for code execution in seconds.

Examples

```
>>> # Register the service
>>> from torchrl.services import get_services
>>> services = get_services(backend="ray")
>>> services.register(
... "python_executor",
... PythonExecutorService,
... pool_size=32,
... timeout=10.0,
... num_cpus=32,
... max_concurrency=32
... )
>>>
>>> # Use in transform
>>> env = env.append_transform(
... PythonInterpreter(services="ray")
... )
```

cleanup()[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#PythonExecutorService.cleanup)

Cleanup all processes in the pool.

execute(*code: str*) → dict[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#PythonExecutorService.execute)

Execute Python code using next available process (round-robin).

Parameters:

**code** - Python code to execute.

Returns:

Execution result with keys 'success', 'stdout', 'stderr', 'returncode'.

Return type:

dict