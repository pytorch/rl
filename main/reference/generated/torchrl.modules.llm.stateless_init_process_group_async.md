# stateless_init_process_group_async

*class*torchrl.modules.llm.stateless_init_process_group_async(*master_address: str | None*, *master_port: str | None*, *rank: int*, *world_size: int*, *device*)[[source]](../../_modules/torchrl/modules/llm/backends/vllm/vllm_utils.html#stateless_init_process_group_async)

Initializes a stateless process group for distributed communication (async version).

Creates a StatelessProcessGroup instance without relying on the global
process group in torch.distributed. This approach is recommended for
initializing data-plane communication (NCCL) between external processes
(e.g., training processes) and vLLM workers.

Parameters:

- **master_address** (*str**|**None*) - The address of the master node. Defaults to "localhost" if not specified.
- **master_port** (*str**|**None*) - The port used by the master node. Automatically assigns an open port if not specified.
- **rank** (*int*) - The rank of the current process.
- **world_size** (*int*) - The total number of processes in the distributed group.
- **device** - The device to use for communication.

Returns:

A PyNcclCommunicator instance initialized with the created StatelessProcessGroup.

Return type:

PyNcclCommunicator