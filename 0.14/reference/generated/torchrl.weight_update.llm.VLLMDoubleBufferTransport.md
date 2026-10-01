# VLLMDoubleBufferTransport

*class*torchrl.weight_update.llm.VLLMDoubleBufferTransport(*remote_addr: str*, *local_addr: str | None = None*, *num_threads: int = 1*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferTransport)

Transport for vLLM using double-buffered memory-mapped storage.

This transport writes weights to a shared directory and reads them back
using TensorDict's memory-mapping capabilities.

Parameters:

- **remote_addr** - Directory path where sender writes weights.
- **local_addr** - Directory path where receiver reads weights.
If None, uses same path as remote_addr (for local testing).
- **num_threads** - Number of threads for memmap operations.

check_connection() → bool[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferTransport.check_connection)

Check if the transport is ready.

For file-based transport, always returns True.

receive_weights(*timeout: float | None = None*, ***, *weights: Any = None*, *model: Any = None*, *strategy: Any = None*) → Any | None[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferTransport.receive_weights)

Reads the weights from the shared directory.

Parameters:

- **timeout** - Ignored (file-based transport is instant).
- **weights** - Ignored.
- **model** - Ignored.
- **strategy** - Ignored.

Returns:

TensorDict with flattened keys containing the weights.

send_weights(*model_id: str*, *weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferTransport.send_weights)

Writes the weights to a shared directory.

Parameters:

- **model_id** - Identifier for the model (used for logging).
- **weights** - TensorDict or dict of weights to write.