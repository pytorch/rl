# AsyncSGLang

*class*torchrl.modules.llm.AsyncSGLang(*server_url: str | None = None*, *model_path: str | None = None*, *tp_size: int = 1*, *dp_size: int = 1*, *timeout: float = 300.0*, ***server_kwargs: Any*)[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang)

Server-based SGLang inference service for TorchRL.

AsyncSGLang provides a unified interface for text generation using SGLang servers,
supporting both managed (subprocess) and external server modes. It integrates
seamlessly with TorchRL's RL training workflows through NCCL-based weight
synchronization.

Key Features:

- HTTP-based generation via SGLang's native /generate API
- Cache-aware load balancing through SGLang Router
- NCCL-based weight synchronization for RL training
- Support for both managed and external server modes
- Compatible interface with vLLM backends for easy migration

Parameters:

- **server_url** - URL of an external SGLang server (e.g., "[http://localhost:30000](http://localhost:30000)").
If None, a managed server will be launched.
- **model_path** - Path or name of the model to load (for managed mode).
- **tp_size** - Tensor parallel size (default: 1).
- **dp_size** - Data parallel size (default: 1).
- **timeout** - Request timeout in seconds (default: 300).
- ****server_kwargs** - Additional arguments passed to SGLang server launch.

Examples

```
>>> # Connect to an existing SGLang server
>>> service = AsyncSGLang.connect("http://localhost:30000")
>>> result = service.generate("Hello, world!")
>>>
>>> # Launch a managed SGLang server
>>> service = AsyncSGLang.from_pretrained("Qwen/Qwen2.5-3B")
>>> result = service.generate("Hello, world!")
>>>
>>> # With custom parameters
>>> service = AsyncSGLang.from_pretrained(
... "Qwen/Qwen2.5-7B",
... tp_size=2,
... max_model_len=4096
... )
```

Note

For RL training with weight updates, use the weight synchronization
methods after initializing the NCCL communication group.

*classmethod*connect(*server_url: str*) → AsyncSGLang[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.connect)

Connect to an existing SGLang server.

Parameters:

**server_url** - URL of the SGLang server (e.g., "[http://localhost:30000](http://localhost:30000)")

Returns:

Connected service instance

Return type:

AsyncSGLang

Raises:

**ConnectionError** - If the server is not reachable

flush_cache() → bool[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.flush_cache)

Flush the radix cache on the server.

This is automatically triggered when weights are updated.

Returns:

True if cache was flushed successfully

Return type:

bool

*classmethod*from_pretrained(*model_name: str*, *tp_size: int = 1*, *dp_size: int = 1*, ***kwargs: Any*) → AsyncSGLang[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.from_pretrained)

Create an AsyncSGLang instance by launching a managed server.

Parameters:

- **model_name** - Model name or path to load
- **tp_size** - Tensor parallel size
- **dp_size** - Data parallel size
- ****kwargs** - Additional server arguments

Returns:

Service with managed server

Return type:

AsyncSGLang

Example

```
>>> service = AsyncSGLang.from_pretrained(
... "Qwen/Qwen2.5-3B",
... tp_size=2,
... max_model_len=4096
... )
```

generate(*prompts: str | list[str] | None = None*, *sampling_params: dict[str, Any] | None = None*, ***, *input_ids: list[int] | list[list[int]] | None = None*, *return_logprobs: bool = False*, *return_text: bool = True*, *timeout: float | None = None*, ***kwargs: Any*) → dict[str, Any] | list[dict[str, Any]][[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.generate)

Generate text completions from text prompts or token IDs.

You can provide either prompts (text) OR input_ids (tokens), but not both.

Parameters:

- **prompts** - Input text prompt(s) for generation. Mutually exclusive with input_ids.
- **sampling_params** - Sampling parameters (temperature, top_p, max_tokens, etc.)
- **input_ids** - Input token ID(s) for generation. Can be a single list of ints
or a list of lists for batch generation. Mutually exclusive with prompts.
- **return_logprobs** - Whether to return log probabilities
- **return_text** - Whether to return generated text
- **timeout** - Request timeout in seconds
- ****kwargs** - Additional sampling parameters (temperature, max_new_tokens, etc.)
These are merged into sampling_params for convenience.

Returns:

Generation results with 'text', 'output_ids', 'meta_info'

Return type:

dict or list[dict]

Example

```
>>> # Generate from text
>>> result = service.generate(
... "What is the capital of France?",
... {"temperature": 0.7, "max_tokens": 100}
... )
>>> print(result["text"])
```

```
>>> # Generate from token IDs
>>> result = service.generate(
... input_ids=[1, 2, 3, 4],
... sampling_params={"max_tokens": 50}
... )
>>> print(result["output_ids"])
```

```
>>> # Using kwargs for sampling params
>>> result = service.generate("Hello", max_new_tokens=50, temperature=0.7)
```

generate_batch(*prompts: list[str]*, *sampling_params: dict[str, Any] | None = None*, ***kwargs: Any*) → list[dict[str, Any]][[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.generate_batch)

Generate text completions for a batch of prompts.

This is an alias for generate() with a list of prompts.

Parameters:

- **prompts** - List of input prompts
- **sampling_params** - Sampling parameters
- ****kwargs** - Additional arguments passed to generate()

Returns:

List of generation results

Return type:

list[dict]

get_dp_size() → int[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.get_dp_size)

Get the data parallel size.

get_master_address() → str[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.get_master_address)

Get the master address for weight synchronization.

get_master_port() → int[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.get_master_port)

Get the master port for weight synchronization.

get_model_metadata() → dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]][[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.get_model_metadata)

Get model parameter metadata.

Note: This requires fetching from the server. For now, returns empty dict
and expects metadata to be provided externally.

get_tp_size() → int[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.get_tp_size)

Get the tensor parallel size.

init_weight_update_group(*master_address: str | None = None*, *master_port: int | None = None*) → None[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.init_weight_update_group)

Initialize the NCCL weight update group via SGLang's HTTP API.

This calls the SGLang server's /init_weights_update_group endpoint
to set up NCCL communication for weight synchronization.

Parameters:

- **master_address** - Master address for NCCL (default: "localhost")
- **master_port** - Master port for NCCL (auto-assigned if None)

*property*server_url*: str*

Get the server URL.

shutdown() → None[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.shutdown)

Shutdown the managed SGLang server if running.

update_weights(*weights: Iterator[tuple[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]]*) → None[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.update_weights)

Update model weights via NCCL broadcast.

This method coordinates with the SGLang server to broadcast weights
from the trainer (rank 0) to all workers.

Parameters:

**weights** - Iterator yielding (parameter_name, tensor) tuples

update_weights_from_distributed(*name: str*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*, *shape: tuple[int, ...]*) → None[[source]](../../_modules/torchrl/modules/llm/backends/sglang/sglang_server.html#AsyncSGLang.update_weights_from_distributed)

Signal the server to receive a weight update via NCCL broadcast.

This calls SGLang's /update_weights_from_distributed endpoint to
coordinate weight reception.

Parameters:

- **name** - Name of the parameter to update
- **dtype** - Data type of the tensor
- **shape** - Shape of the tensor