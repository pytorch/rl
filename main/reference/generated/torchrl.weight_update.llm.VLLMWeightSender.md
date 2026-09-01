# VLLMWeightSender

*class*torchrl.weight_update.llm.VLLMWeightSender(*scheme: [VLLMWeightSyncScheme](torchrl.weight_update.llm.VLLMWeightSyncScheme.html#torchrl.weight_update.llm.VLLMWeightSyncScheme)*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightSender)

Sends weights to vLLM workers using collective communication.

**RPC + Collective Implementation**

This class implements both layers:

1. **RPC Layer**: Currently uses Ray remote calls (implicit in test setup)
- Can be extended to other RPC backends (torch.distributed.rpc, gRPC)
- In the test, Ray actors provide the RPC mechanism
2. **Collective Layer**: Uses VLLMCollectiveTransport for NCCL broadcast
- Broadcasts weights from trainer (rank 0) to workers (ranks 1+)
- High-bandwidth GPU-to-GPU transfer

**Extending RPC Backends**

To use a different RPC backend, subclass and override coordination:

```
class TorchRPCVLLMSender(VLLMWeightSender):
 def update_weights(self, weights=None):
 # Custom RPC: Signal workers to prepare
 for worker in self.workers:
 torch.distributed.rpc.rpc_async(worker, "prepare_receive")

 # Then do collective (unchanged)
 super().update_weights(weights)
```

init_all_workers_group(*model_metadata: dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]]*, *vllm_engine: Any | None = None*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightSender.init_all_workers_group)

Initialize the collective communication group using vLLM's native API.

Parameters:

- **model_metadata** - Dict mapping param names to (dtype, shape) tuples.
- **vllm_engine** - Optional vLLM engine for RPC coordination. Required for NCCL broadcasts.

register_collector(*collector*) → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightSender.register_collector)

Register a collector for automatic policy version increment.

After each `update_weights()` call, `collector.increment_version()`
is called automatically.

register_model(*model: Any*) → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightSender.register_model)

Register the model to extract weights from.

shutdown() → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightSender.shutdown)

Release resources held by the sender.

update_weights(*weights: Any | None = None*) → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightSender.update_weights)

Extract and send weights to vLLM workers using native weight transfer API.

Parameters:

**weights** - Optional weights to send. If None, extracts from registered model.