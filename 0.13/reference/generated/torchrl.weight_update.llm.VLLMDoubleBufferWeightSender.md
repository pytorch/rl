# VLLMDoubleBufferWeightSender

*class*torchrl.weight_update.llm.VLLMDoubleBufferWeightSender(*scheme: [VLLMDoubleBufferSyncScheme](torchrl.weight_update.llm.VLLMDoubleBufferSyncScheme.html#torchrl.weight_update.llm.VLLMDoubleBufferSyncScheme)*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferWeightSender)

Sends weights to vLLM workers using double-buffered storage.

This sender extracts weights from a training model and writes them to
a shared directory using TensorDict.memmap.

Example

```
>>> sender = scheme.create_sender()
>>> sender.register_model(policy_model)
>>>
>>> # During training loop
>>> sender.update_weights() # Writes current weights to shared storage
```

register_model(*model: Any*) → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferWeightSender.register_model)

Register the model to extract weights from.

Parameters:

**model** - The model to extract weights from (e.g., TransformersWrapper).

update_weights(*weights: Any | None = None*) → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_double_buffer.html#VLLMDoubleBufferWeightSender.update_weights)

Extract and write weights to shared storage.

Parameters:

**weights** - Optional weights to send. If None, extracts from registered model.