# SGLangWeightSender

*class*torchrl.weight_update.llm.SGLangWeightSender(*scheme: [SGLangWeightSyncScheme](torchrl.weight_update.llm.SGLangWeightSyncScheme.html#torchrl.weight_update.llm.SGLangWeightSyncScheme)*)[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangWeightSender)

Sends weights to SGLang workers using NCCL broadcast.

Parameters:

**scheme** - The SGLangWeightSyncScheme configuration.

flush_cache() → bool[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangWeightSender.flush_cache)

Flush the SGLang server's radix cache after weight update.

Returns:

True if cache was flushed successfully.

Return type:

bool

init_all_workers_group(*model_metadata: dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]]*) → None[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangWeightSender.init_all_workers_group)

Initialize the NCCL communication group.

Parameters:

**model_metadata** - Dict mapping param names to (dtype, shape) tuples.

register_collector(*collector*) → None[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangWeightSender.register_collector)

Register a collector for automatic policy version increment.

After each `update_weights()` call, `collector.increment_version()`
is called automatically.

register_model(*model: Any*) → None[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangWeightSender.register_model)

Register the model for weight extraction.

Parameters:

**model** - The PyTorch model to sync weights from.

shutdown() → None[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangWeightSender.shutdown)

Release resources held by the sender.

update_weights(*weights: dict[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | None = None*) → None[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangWeightSender.update_weights)

Broadcast weights to SGLang workers.

Parameters:

**weights** - Optional dict of weights. If None, extracts from registered model.