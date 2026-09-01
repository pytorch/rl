# VLLMWeightReceiver

*class*torchrl.weight_update.llm.VLLMWeightReceiver(*scheme: [VLLMWeightSyncScheme](torchrl.weight_update.llm.VLLMWeightSyncScheme.html#torchrl.weight_update.llm.VLLMWeightSyncScheme)*, *vllm_engine*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightReceiver)

Receives weights in a vLLM worker using collective communication.

**RPC + Collective Implementation**

This class implements both layers:

1. **RPC Layer**: Currently uses Ray for coordination
- init() in test uses Ray ray.get_actor() to find trainer
- Fetches metadata via Ray remote call
- Signals readiness to participate in collective
2. **Collective Layer**: Participates in NCCL broadcast
- Receives weights via collective operations
- vLLM engine applies weights internally during broadcast

**Extending RPC Backends**

To use a different RPC backend:

```
class TorchRPCVLLMReceiver(VLLMWeightReceiver):
 def init(self):
 # Custom RPC: Get metadata from trainer
 metadata = torch.distributed.rpc.rpc_sync(
 "trainer",
 lambda: get_metadata()
 )

 # Then init collective (unchanged)
 self.receiver.init_all_workers_group(metadata)
```

Note

The RPC and collective layers are loosely coupled. The RPC layer
ensures all ranks are ready before the collective starts, but the
actual data transfer is independent of the RPC mechanism.

apply_weights(*weights: Any*, *inplace: bool = True*) → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightReceiver.apply_weights)

Apply weights to vLLM engine.

Parameters:

- **weights** - The weights to apply.
- **inplace** - Whether to apply weights in place. Default is True.

Note: For vLLM, weights are applied automatically during the collective
broadcast operation. This method is a no-op but kept for API consistency.

init_all_workers_group(*model_metadata: dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]]*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightReceiver.init_all_workers_group)

Initialize the collective communication group using vLLM's native API.

Parameters:

**model_metadata** - Dict mapping param names to (dtype, shape) tuples.

poll_and_apply(*timeout: float = 0.1*) → bool[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMWeightReceiver.poll_and_apply)

Poll for and apply weights.

Returns:

False - vLLM uses push-based updates via collectives, not polling.