# RayWeightSyncScheme

*class*torchrl.weight_update.RayWeightSyncScheme(*strategy: Literal['tensordict', 'state_dict'] = 'tensordict'*, *backend: str = 'gloo'*)[[source]](../../_modules/torchrl/weight_update/_ray.html#RayWeightSyncScheme)

Weight synchronization for Ray distributed computing.

This scheme uses torch.distributed to synchronize weights across distributed
workers (Ray actors). The process group is initialized during the first
`synchronize_weights()` call, with the sender as rank 0 and workers as
rank `worker_idx + 1`.

Each remote collector gets its own transport, following the same pattern
as multiprocess collectors.

Parameters:

- **strategy** (*str*) - The weight transmission strategy ("state_dict" or "tensordict").
Defaults to "tensordict".
- **backend** (*str*) - The torch.distributed backend to use ("gloo" or "nccl").
Defaults to "gloo".

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

*property*connection_info_name*: str*

Get the name of the Ray actor storing connection info.

Returns a unique name based on model_id to avoid collisions when
multiple schemes are used with different models.

Returns:

The connection info actor name.

*property*context*: Any | None*

Get the context object (e.g., collector), if available.

Returns:

The context object if available, None otherwise.

create_transport(***, *remote_actor=None*, *worker_idx: int | None = None*, *remote_collector=None*, ***kwargs*) → [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)[[source]](../../_modules/torchrl/weight_update/_ray.html#RayWeightSyncScheme.create_transport)

Create Ray-based transport for a specific remote actor.

Parameters:

- **remote_actor** - The Ray actor handle for the remote collector/transform.
- **worker_idx** - The worker index for this remote actor.
- **remote_collector** - Legacy alias for remote_actor.
- ****kwargs** - Additional transport configuration.

Returns:

RayTransport configured for this specific remote actor.

*classmethod*from_backend(*backend: Literal['none', 'direct', 'shared', 'thread', 'process', 'multiprocessing', 'distributed', 'rpc', 'ray']*, ***kwargs*) → [WeightSyncScheme](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)

Construct a weight-sync scheme from its deployment backend.

Parameters:

- **backend** - One of `none`, `shared`, `process`, `distributed`,
`rpc`, or `ray`. `direct`, `thread`, and
`multiprocessing` are accepted aliases.
- ****kwargs** - Arguments forwarded to the concrete scheme constructor.

Returns:

The selected concrete scheme.

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

shutdown() → None

Shutdown the scheme and release resources.

This method stops any background threads and cleans up connections.
It is safe to call multiple times. Subclasses should override this
method to add custom cleanup logic, but should call super().shutdown()
to ensure base cleanup is performed.

*property*weights*: Any | None*

Get the current weights, if available.

Returns:

The weights as TensorDict if available, None otherwise.

*property*worker_idx*: int | None*

Get the worker index for this scheme.

Returns:

The worker index if set, None otherwise.