# AsyncVLLM

*class*torchrl.modules.llm.AsyncVLLM(*engine_args: AsyncEngineArgs*, *num_replicas: int = 1*, *actor_class=None*, *enable_prefix_caching: bool | None = None*)[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM)

A service that manages multiple async vLLM engine actors for distributed inference.

This is the main entry point for async vLLM inference in TorchRL. It manages multiple
vLLM engine replicas running as Ray actors, providing load balancing, weight updates,
and a unified interface for text generation.

The service automatically handles Ray actor lifecycle management, GPU allocation through
placement groups, and provides both synchronous and asynchronous generation interfaces
that are compatible with the standard vLLM API.

Parameters:

- **engine_args** (*AsyncEngineArgs*) - Configuration for the vLLM engines.
- **num_replicas** (*int**,**optional*) - Number of engine replicas to create. Defaults to 1.
- **actor_class** (*optional*) - Custom Ray actor class. Defaults to the internal actor implementation.
- **enable_prefix_caching** (*bool**,**optional*) -

Whether to enable prefix caching.
`None` (default) respects `engine_args.enable_prefix_caching` when it
is set, and falls back to `False` otherwise.

Note

Prefix caching used to be discouraged with online weight updates
because cached KV prefixes are keyed by prompt content, not by the
weights that produced them. Caches are now reset automatically after
each weight update (see `reset_prefix_cache()`), and truncated
prompt log-probs from cached prefixes are zero-padded by the vLLM
wrapper, so enabling it for online RL is supported.

Example

```
>>> from torchrl.modules.llm import AsyncVLLM
>>> from vllm import SamplingParams
>>>
>>> # Simple usage - single GPU, single replica
>>> service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-3B")
>>>
>>> # Advanced usage - multi-GPU tensor parallel with multiple replicas
>>> service = AsyncVLLM.from_pretrained(
... "Qwen/Qwen2.5-7B",
... num_devices=2, # Use 2 GPUs for tensor parallelism
... num_replicas=2, # Create 2 replicas for higher throughput
... max_model_len=4096
... )
>>>
>>> # Generate text
>>> sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
>>> result = service.generate("Hello, world!", sampling_params)
>>> print(result.outputs[0].text)
>>>
>>> # Alternative: using AsyncEngineArgs directly for advanced configuration
>>> from vllm import AsyncEngineArgs
>>> engine_args = AsyncEngineArgs(
... model="Qwen/Qwen2.5-3B",
... tensor_parallel_size=2
... )
>>> service = AsyncVLLM.launch(engine_args, num_replicas=2)
```

Note

**Architecture and Design**

The AsyncVLLM service implements a distributed inference architecture with the following key components:

1. **Ray Actor Management**: Each replica runs as a separate Ray actor with dedicated GPU resources.
The service creates a placement group to ensure optimal GPU allocation and co-location of
tensor-parallel workers on the same node when possible.
2. **Load Balancing**: Generation requests are distributed across replicas using random selection
by default, or can target specific replicas using the actor_index parameter.
3. **Weight Synchronization**: The service supports weight updates across all replicas through
NCCL communication groups, enabling integration with distributed training workflows.
4. **Resource Management**: Automatic GPU allocation and cleanup through Ray placement groups,
with proper shutdown procedures to prevent resource leaks.
5. **API Compatibility**: Provides the same interface as vLLM's synchronous LLM.generate()
method, making it a drop-in replacement for async workloads.

**Ray Integration**

The service leverages Ray's actor model for distributed execution. Each replica is an independent
Ray actor that can be scheduled on different nodes. The service handles actor lifecycle,
monitors readiness, and provides centralized access to all replicas.

**Performance Considerations**

- Prefix caching is disabled by default (conservative); when enabled, caches are
reset automatically after each weight update
- Tensor parallelism is supported for large models that don't fit on single GPUs
- Multiple replicas allow concurrent processing of different requests
- Native vLLM batching is used within each replica for optimal throughput

**Error Handling**

The service includes timeout support, graceful shutdown procedures, and best-effort
request cleanup on failures. Ray's fault tolerance mechanisms provide additional
resilience for long-running inference workloads.

collective_rpc(*method: str*, *timeout: float | None = None*, *args: tuple = ()*, *kwargs: dict | None = None*) → list[Any][[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.collective_rpc)

Forward an RPC to all actors.

Parameters:

- **method** (*str*) - Method name to call.
- **timeout** (*float**|**None*) - Timeout for the RPC call.
- **args** (*tuple*) - Arguments to pass to the method.
- **kwargs** (*dict**|**None*) - Keyword arguments to pass to the method.

Returns:

Ray futures for all RPC calls.

Return type:

list[Any]

create_load_balancer(*strategy: Literal['requests', 'kv-cache'] | Sequence[Literal['prefix-aware', 'requests', 'kv-cache', 'round-robin']] | None = None*, ***kwargs*) → LoadBalancer[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.create_load_balancer)

Create a load balancer for this AsyncVLLM service.

Parameters:

- **strategy** - Load balancing strategy or sequence of strategies in fallback order.
Default: ["prefix-aware", "requests"] - tries cache-aware routing first,
then load balancing. Single strategies: "requests", "kv-cache"
Strategy sequences: ["prefix-aware", "requests", "round-robin"]
- ****kwargs** - Additional arguments passed to LoadBalancer constructor.

Returns:

Configured load balancer instance. This is stored in the AsyncVLLM instance.

Return type:

LoadBalancer

Examples

```
>>> service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-3B", num_replicas=3)
```

```
>>> # Use smart defaults (prefix-aware -> requests)
>>> lb = service.create_load_balancer()
>>> selected_actor_index = lb.select_actor(prompt="Hello world")
```

```
>>> # Simple single strategy
>>> lb = service.create_load_balancer("requests")
>>> selected_actor_index = lb.select_actor()
```

```
>>> # Custom strategy hierarchy
>>> lb = service.create_load_balancer(
... ["prefix-aware", "kv-cache", "round-robin"],
... prefix_length=16,
... overload_threshold=2.0
... )
>>> selected_actor_index = lb.select_actor(prompt="Hello world")
```

*classmethod*from_pretrained(*model_name: str*, *num_devices: int | None = None*, *num_replicas: int = 1*, *verbose: bool = True*, *compile: bool = True*, *enable_fp32_output: bool = False*, ***kwargs*) → AsyncVLLM[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.from_pretrained)

Create an AsyncVLLM instance from a pretrained model.

This is a convenience method that combines model loading and service launching
in a single call, similar to how other ML libraries work.

Parameters:

- **model_name** (*str*) - The model name to pass to vLLM.
- **num_devices** (*int**,**optional*) - Number of devices to use, per replica.
- **num_replicas** (*int*) - Number of engine replicas to create.
- **verbose** (*bool**,**optional*) - Whether to enable verbose logging with throughput statistics. Defaults to True.
- **compile** (*bool**,**optional*) - Whether to enable model compilation for better performance. Defaults to True.
- **enable_fp32_output** (*bool**,**optional*) - Whether to enable FP32 output for the final layer. Defaults to False.
- ****kwargs** - Additional arguments passed to AsyncEngineArgs.

Returns:

The launched async vLLM service.

Return type:

AsyncVLLM

Example

```
>>> # Simple usage with defaults
>>> service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-3B")
>>>
>>> # Multi-GPU tensor parallel with multiple replicas
>>> service = AsyncVLLM.from_pretrained(
... "Qwen/Qwen2.5-7B",
... num_devices=2,
... num_replicas=2,
... max_model_len=4096
... )
>>>
>>> # Generate text
>>> from vllm import SamplingParams
>>> result = service.generate("Hello, world!", SamplingParams(max_tokens=50))
>>>
>>> # Enable FP32 output for better numerical stability
>>> service = AsyncVLLM.from_pretrained(
... "Qwen/Qwen2.5-3B",
... enable_fp32_output=True
... )
```

generate(*prompts: Any = None*, *sampling_params: SamplingParams | None = None*, ***, *prompt_token_ids: list[int] | list[list[int]] | None = None*, *use_tqdm: bool = True*, *lora_request: Any = None*, *prompt_adapter_request: Any = None*, *guided_options_request: Any = None*, *timeout_seconds: float | None = None*, *actor_index: int | None = None*) → RequestOutput | list[RequestOutput][[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.generate)

Generate text using one of the actors with vLLM.LLM.generate interface.

This method provides the same interface as vLLM.LLM.generate for seamless
compatibility between sync and async engines. It can be used to generate text
within multiple threads / actors. If actor_index is not provided, the load balancer
will be used to select the actor.

generate is a blocking method, so it will wait for the generation to complete.

Parameters:

- **prompts** (*String**,**TokensPrompt**, or**list**of**these*) - Input prompts for generation.
- **sampling_params** (*SamplingParams*) - SamplingParams object for controlling generation behavior.
- **prompt_token_ids** (*list**[**int**]**|**list**[**list**[**int**]**]*) - Alternative to prompts - token IDs for generation.
- **use_tqdm** (*bool*) - Whether to show progress bar (not used in async engine).
- **lora_request** (*Any*) - LoRA request for adapter-based generation.
- **prompt_adapter_request** (*Any*) - Prompt adapter request.
- **guided_options_request** (*Any*) - Guided decoding options.
- **timeout_seconds** (*float**|**None*) - Timeout for generation in seconds.
- **actor_index** (*int**|**None*) - Specific actor to use (random if None).

Returns:

Generated outputs from vLLM.

Return type:

RequestOutput | list[RequestOutput]

get_cache_usage(*actor_index: int | None = None*) → float | list[float][[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.get_cache_usage)

Get the KV cache usage for one or all actors.

Parameters:

**actor_index** (*int**|**None*) - Index of specific actor, or None for all actors.

Returns:

Cache usage fraction for the specified actor,

or list of usage fractions for all actors if actor_index is None.

Return type:

float | list[float]

get_master_address() → str[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.get_master_address)

Get the master address for weight synchronization.

get_master_port() → int[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.get_master_port)

Get the master port for weight synchronization.

get_model_metadata() → dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]][[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.get_model_metadata)

Get model parameter metadata.

Note: This requires the model to be loaded. For now, we return an empty dict
and expect the metadata to be provided externally during weight updates.

get_num_unfinished_requests(*actor_index: int | None = None*) → int | list[int][[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.get_num_unfinished_requests)

Get the number of unfinished requests for one or all actors.

Parameters:

**actor_index** (*int**|**None*) - Index of specific actor, or None for all actors.

Returns:

Number of unfinished requests for the specified actor,

or list of counts for all actors if actor_index is None.

Return type:

int | list[int]

get_random_actor_index() → int[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.get_random_actor_index)

Get a random actor index.

get_tp_size() → int[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.get_tp_size)

Get the tensor parallel size.

init_weight_update_group() → None[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.init_weight_update_group)

Initialize the weight update communication group using vLLM's native WeightTransferConfig API.

This sets up NCCL weight transfer on both the trainer side and all vLLM worker actors.
The trainer is rank 0 and vLLM workers are ranks 1+.

*classmethod*launch(*engine_args: AsyncEngineArgs*, *num_replicas: int = 1*) → AsyncVLLM[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.launch)

Launch a new AsyncVLLMEngineService.

Parameters:

- **engine_args** (*AsyncEngineArgs*) - Arguments for creating the AsyncLLMEngine instances.
- **num_replicas** (*int*) - Number of actor replicas to create.

Returns:

The launched service.

Return type:

AsyncVLLMEngineService

reset_prefix_cache() → None[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.reset_prefix_cache)

Reset the KV prefix cache on all replicas.

Called automatically after each weight update: cached prefixes are
keyed by prompt content, not by the weights that produced them, so
they are stale once new weights are loaded. This is a no-op when
prefix caching is disabled.

shutdown()[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.shutdown)

Shutdown all actors and clean up resources.

update_weights(*weights: Iterator[tuple[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]]*) → None[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_async.html#AsyncVLLM.update_weights)

Update model weights across all replicas using vLLM's native weight transfer API.

Parameters:

**weights** - Iterator yielding (parameter_name, tensor) tuples