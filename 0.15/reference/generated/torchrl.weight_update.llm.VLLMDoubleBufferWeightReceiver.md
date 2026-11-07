# VLLMDoubleBufferWeightReceiver

*class*torchrl.weight_update.llm.VLLMDoubleBufferWeightReceiver(*scheme: [VLLMDoubleBufferSyncScheme](torchrl.weight_update.llm.VLLMDoubleBufferSyncScheme.html#torchrl.weight_update.llm.VLLMDoubleBufferSyncScheme)*, *vllm_engine*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferWeightReceiver)

Receives weights in a vLLM worker using double-buffered storage.

This receiver reads weights from a shared directory and loads them into
the vLLM engine using the engine's load_weights interface.

Example

```
>>> receiver = scheme.create_receiver(vllm_engine)
>>>
>>> # Poll for new weights
>>> if receiver.poll_and_apply():
... print("Weights updated!")
```

apply_weights(*weights: [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)*, *inplace: bool = True*) → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferWeightReceiver.apply_weights)

Apply weights to vLLM engine using RPC.

This method uses RPC to tell all vLLM workers to load weights from
the shared storage directory. Similar to how AsyncVLLM._update_weights_with_nccl_broadcast_simple
uses collective_rpc to coordinate workers.

Parameters:

- **weights** - TensorDict with flattened keys containing weights.
- **inplace** - Whether to apply weights in place. Default is True.

poll_and_apply(*timeout: float = 180.0*) → bool[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferWeightReceiver.poll_and_apply)

Poll for and apply weights from shared storage.

Parameters:

**timeout** - Not used for file-based transport (kept for API compatibility).

Returns:

True if weights were successfully read and applied, False otherwise.