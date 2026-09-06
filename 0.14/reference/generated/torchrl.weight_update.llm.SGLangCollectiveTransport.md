# SGLangCollectiveTransport

*class*torchrl.weight_update.llm.SGLangCollectiveTransport(*server_url: str*, *master_address: str*, *master_port: int*, *rank: int*, *world_size: int*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *timeout: float = 300.0*, *flush_cache_on_update: bool | None = None*, *pause_mode: Literal['abort', 'retract', 'in_place'] | None = None*)[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangCollectiveTransport)

Transport for SGLang using NCCL collective communication.

This transport coordinates with SGLang servers via HTTP and performs
weight transfer via NCCL broadcast.

Parameters:

- **server_url** - URL of the SGLang server.
- **master_address** - Address for NCCL initialization.
- **master_port** - Port for NCCL initialization.
- **rank** - Rank of this process (0 for trainer).
- **world_size** - Total number of processes.
- **device** - Device to use for communication.
- **timeout** - HTTP request timeout in seconds.
- **flush_cache_on_update** - Whether to ask the server to flush its radix
(prefix) cache as part of each weight update. `None` (default)
flushes exactly when the pause mode is `"abort"` - the only mode
under which SGLang can honor the flush, since it requires an idle
scheduler and retracted requests stay queued. `True` requires the
`"abort"` pause mode and raises otherwise. The
`TORCHRL_SGLANG_WEIGHT_SYNC_FLUSH_CACHE` environment variable,
when set, overrides this in either direction (downgraded with a
warning when the pause mode cannot honor it).
- **pause_mode** - How to pause generation for the update: `"abort"` cancels
in-flight requests (callers must tolerate transient generation
failures), `"retract"` re-queues them, `"in_place"` freezes
them. `None` (default) defers to the
`TORCHRL_SGLANG_PAUSE_GENERATION_MODE` environment variable and
falls back to `"retract"`.

check_connection() → bool[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangCollectiveTransport.check_connection)

Check if the communication group is initialized.

init_all_workers_group(*model_metadata: dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]]*) → None[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangCollectiveTransport.init_all_workers_group)

Initialize the NCCL communication group.

For the trainer (rank 0), this:
1. Creates a torch.distributed process group via TCP rendezvous (rank 0 is master)
2. Signals the SGLang server via HTTP to create a matching process group
3. Both sides rendezvous via the TCP store and form an NCCL group

The SGLang server uses `init_custom_process_group` internally which
creates a `torch.distributed` process group (not SGLang's standalone
`StatelessProcessGroup` + `PyNcclCommunicator`). The trainer must
use the same mechanism so both sides join the same NCCL collective.

Parameters:

**model_metadata** - Dict mapping param names to (dtype, shape) tuples.

send_weights(*model_id: str*, *weights: dict[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]*) → None[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangCollectiveTransport.send_weights)

Broadcast weights to SGLang server via NCCL.

SGLang's `/update_weights_from_distributed` endpoint expects a single
request with lists of all parameter names, dtypes, and shapes. The
server then enters a broadcast-receive loop for each parameter in
order. The trainer must broadcast each tensor in the same order,
concurrently with the server receiving.

Parameters:

- **model_id** - Identifier for the model (for logging).
- **weights** - Dict mapping parameter names to tensors.

shutdown() → None[[source]](../../_modules/torchrl/weight_update/llm/sglang_nccl.html#SGLangCollectiveTransport.shutdown)

Release trainer-side resources used for weight synchronization.