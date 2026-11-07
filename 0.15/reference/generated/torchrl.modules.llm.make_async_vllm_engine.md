# make_async_vllm_engine

*class*torchrl.modules.llm.make_async_vllm_engine(***, *model_name: str*, *num_devices: int | None = None*, *num_replicas: int = 1*, *verbose: bool = True*, *compile: bool = True*, *enable_fp32_output: bool = False*, *tensor_parallel_size: int | None = None*, *data_parallel_size: int | None = None*, *pipeline_parallel_size: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#make_async_vllm_engine)

Create an async vLLM engine service.

Keyword Arguments:

- **model_name** (*str*) - The model name to pass to vLLM.
- **num_devices** (*int**,**optional*) - Number of devices to use, per replica.
- **num_replicas** (*int*) - Number of engine replicas to create.
- **verbose** (*bool**,**optional*) - Whether to enable verbose logging with throughput statistics. Defaults to True.
- **compile** (*bool**,**optional*) - Whether to enable model compilation for better performance. Defaults to True.
- **enable_fp32_output** (*bool**,**optional*) - Whether to enable FP32 output for the final layer. Defaults to False.
This can help with numerical stability for certain models. Requires model-specific support in
torchrl.modules.llm.backends.vllm._models.
- **tensor_parallel_size** (*int**,**optional*) - Number of devices to use, per replica. Defaults to None.
- **data_parallel_size** (*int**,**optional*) - Number of data parallel groups to use. Defaults to None.
- **pipeline_parallel_size** (*int**,**optional*) - Number of pipeline parallel groups to use. Defaults to None.
- **enable_prefix_caching** (*bool**,**optional*) - Whether to enable vLLM prefix
caching. Defaults to `False` to avoid reusing prompt KV caches
across online weight updates.
- ****kwargs** - Additional arguments passed to AsyncEngineArgs.

Returns:

The launched engine service.

Return type:

[AsyncVLLM](torchrl.modules.llm.AsyncVLLM.html#torchrl.modules.llm.AsyncVLLM)

Raises:

- **RuntimeError** - If no CUDA devices are available.
- **ValueError** - If invalid device configuration is provided.

Example

```
>>> # Create a single-GPU async engine
>>> service = make_async_vllm_engine("Qwen/Qwen2.5-3B")
>>>
>>> # Create a 2-GPU tensor parallel async engine with 2 replicas
>>> service = make_async_vllm_engine("Qwen/Qwen2.5-3B", num_devices=2, num_replicas=2)
>>> # Generate text
>>> result = service.generate("Hello, world!", sampling_params)
>>>
>>> # Create with FP32 output enabled
>>> service = make_async_vllm_engine("Qwen/Qwen2.5-3B", enable_fp32_output=True)
```