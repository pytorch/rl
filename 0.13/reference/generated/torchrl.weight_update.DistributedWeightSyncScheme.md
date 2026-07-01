# DistributedWeightSyncScheme

*class*torchrl.weight_update.DistributedWeightSyncScheme(*backend: str = 'gloo'*, *sync: bool = True*, *timeout: float = 3600.0*)[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedWeightSyncScheme)

Weight synchronization for torch.distributed.

This scheme uses torch.distributed primitives (send/recv) to synchronize
weights across distributed workers. Each worker gets its own transport,
following the same pattern as multiprocess collectors.

The scheme can create its own TCPStore for coordination if one is not provided.
Use get_store_info() after init_on_sender() to get connection details for workers.

Parameters:

- **backend** (*str*) - The distributed backend ("gloo", "nccl", etc.)
- **sync** (*bool*) - If True, weight updates are synchronous (blocking receive).
If False, a background thread monitors the store and applies weight
updates automatically. Defaults to True.
- **timeout** (*float*) - Timeout in seconds for TCPStore operations.
Defaults to 3600.0 (1 hour).

apply_weights(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *inplace: bool = True*) → None

Apply weights to the model.

Parameters:

- **weights** - The weights to apply.
- **inplace** - Whether to apply weights in place. Default is True.

connect(***, *worker_idx: int | None = None*, *weights: Any | None = None*) → None

Method to be called once the workers have started.

Triggers a rendez-vous for the workers to receive their copy of the weights.

Dispatches to _setup_connection_and_weights_on_sender_impl() or _setup_connection_and_weights_on_receiver_impl()
based on which initialization was performed.

*property*context*: Any | None*

Get the context object (e.g., collector), if available.

Returns:

The context object if available, None otherwise.

create_transport(***kwargs*) → [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedWeightSyncScheme.create_transport)

Create distributed transport for a specific worker.

init_on_receiver(***, *model_id: str*, *context: Any = None*, ***kwargs*) → None

Initialize on worker process (receiver side).

This method is called once in each worker's initialization.

Parameters:

- **model_id** - Identifier for the model being synchronized
- **context** - Optional context object (e.g., inner collector)
- ****kwargs** - Alternative to context (model, etc.)

init_on_sender(**args*, ***kwargs*) → None

Initialize on the main process (sender side).

This method is called once in the collector's _run_processes() method,
after workers have been started and are ready to receive messages.

*property*model*: Any | None*

Get the model associated with this scheme.

Returns:

The model if set, None otherwise.

*property*model_id*: str | None*

Get the model ID for this scheme.

Returns:

The model ID if set, None otherwise.

prepare_weights(*weights: Any*, *model_id: str*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy)*, *context: Any = None*) → Any

Prepare weights for sending.

This method handles weight extraction, conversion, and any scheme-specific
preparation (e.g., cache lookups for SharedMemWeightSyncScheme).

Parameters:

- **weights** - Raw weights input (can be None, nn.Module, TensorDict, dict, str reference, etc.)
- **model_id** - The model identifier (e.g., "policy")
- **strategy** - WeightStrategy for extracting/converting weights
- **context** - Optional context (e.g., collector) for model resolution

Returns:

Prepared weights ready to send via transport

receive(*timeout: float | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None

Check for and apply new weights (non-blocking).

This method is called in the worker's main loop to check if
new weights have been sent. If weights are available, they
are applied to the registered model immediately, and the update
is cascaded to any sub-collectors via context.update_policy_weights_().

Parameters:

**timeout** - Maximum time to wait for weights (seconds).
None means no timeout (blocking). Some transports may not
support timeout and will raise ValueError if specified.

Returns:

The received weights if available, None otherwise.

Note: For SharedMemWeightSyncScheme, this always returns None
since workers automatically see updates via shared memory.

*property*receiver_transport*: [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend) | None*

Get the receiver transport.

Returns:

The receiver transport.

send(*weights: Any = None*, *worker_ids: int | list[int] | None = None*) → None

Send weights synchronously to workers.

This method:
1. Prepares weights (extracts from model if weights=None)
2. Sends to specified workers (or all if worker_ids=None)
3. Waits for acknowledgments from those workers
4. Returns when workers have applied the weights

Parameters:

- **weights** - Weights to send. Can be:
- None: Extract from model via context.get_model(model_id)
- nn.Module: Extract weights from module
- TensorDict: Use directly
- dict: Convert to TensorDict
- **worker_ids** - Which workers to send to:
- None: Send to all workers (default)
- int: Send to single worker
- list[int]: Send to specific workers

Note: This is a blocking call that ensures specified workers are updated
before returning.

*property*sender_transports*: dict[int, [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)]*

Get the sender transports.

Returns:

The sender transports.

*property*shared_transport*: [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend) | None*

Get the shared transport.

Returns:

The shared transport.

shutdown() → None[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedWeightSyncScheme.shutdown)

Stop background receiver thread and clean up.

*property*weights*: Any | None*

Get the current weights, if available.

Returns:

The weights as TensorDict if available, None otherwise.

*property*worker_idx*: int | None*

Get the worker index for this scheme.

Returns:

The worker index if set, None otherwise.