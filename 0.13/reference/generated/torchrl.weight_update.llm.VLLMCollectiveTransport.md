# VLLMCollectiveTransport

*class*torchrl.weight_update.llm.VLLMCollectiveTransport(*master_address: str*, *master_port: int*, *rank: int | None*, *world_size: int*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *vllm_engine: Any | None = None*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMCollectiveTransport)

Transport for vLLM using vLLM's native WeightTransferConfig API (vLLM 0.17+).

This transport uses vLLM's built-in NCCL weight transfer engine to broadcast
weights from a trainer (rank 0) to vLLM workers (ranks 1+).

Parameters:

- **master_address** - Address of the master node for distributed init.
- **master_port** - Port of the master node for distributed init.
- **rank** - Rank of this process (0 for trainer, 1+ for vLLM workers).
- **world_size** - Total number of processes (1 + num_replicas * gpus_per_replica).
- **device** - Device to use for communication (typically cuda:0).
- **vllm_engine** - Optional vLLM engine reference (for receiver side).

check_connection() → bool[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMCollectiveTransport.check_connection)

Check if the communication group is initialized.

init_all_workers_group(*model_metadata: dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]]*, *gpus_per_replica: int | None = None*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMCollectiveTransport.init_all_workers_group)

Initialize the collective communication group using vLLM's native API.

Parameters:

- **model_metadata** - Dict mapping param names to (dtype, shape) tuples.
- **gpus_per_replica** - GPUs per replica (for rank_offset calculation). Inferred if not provided.

receive_weights(*timeout: float | None = None*, ***, *weights: Any = None*, *model: Any = None*, *strategy: Any = None*) → Any | None[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMCollectiveTransport.receive_weights)

Receive weights from broadcaster.

Returns:

None - vLLM handles weight application internally via native API.

send_weights(*model_id: str*, *weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMCollectiveTransport.send_weights)

Send weights to all workers using vLLM's native weight transfer API.

Parameters:

- **model_id** - ID of the model (used for logging).
- **weights** - TensorDict or dict of weights to broadcast.

shutdown() → None[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#VLLMCollectiveTransport.shutdown)

Release trainer-side resources used for weight synchronization.