# WeightSyncScheme

*class*torchrl.weight_update.WeightSyncScheme(*strategy: Literal['state_dict', 'tensordict'] = 'tensordict'*)[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme)

Configuration for how to synchronize ONE model across workers.

A scheme manages synchronization of ONE model across workers.
The collector maintains a dict of {model_id: scheme} pairs.

This class directly handles both sender and receiver functionality,
with behavior determined by whether init_on_sender() or init_on_receiver()
was called.

apply_weights(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *inplace: bool = True*) → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme.apply_weights)

Apply weights to the model.

Parameters:

- **weights** - The weights to apply.
- **inplace** - Whether to apply weights in place. Default is True.

connect(***, *worker_idx: int | None = None*) → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme.connect)

connect(***, *weights: Any | None = None*) → None

Method to be called once the workers have started.

Triggers a rendez-vous for the workers to receive their copy of the weights.

Dispatches to _setup_connection_and_weights_on_sender_impl() or _setup_connection_and_weights_on_receiver_impl()
based on which initialization was performed.

*property*context*: Any | None*

Get the context object (e.g., collector), if available.

Returns:

The context object if available, None otherwise.

*abstract*create_transport(***kwargs*) → [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme.create_transport)

Create transport for communication.

Parameters:

****kwargs** - Transport-specific configuration parameters.

Returns:

A transport backend instance.

Note

This is used internally by init_on_sender/init_on_receiver.

init_on_receiver(*model_id: str*, *context: Any*, ***kwargs*) → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme.init_on_receiver)

init_on_receiver(*model_id: str*, *context: None = None*, ***, *worker_idx: int = ...*, *model: Any | None = None*, ***kwargs*) → None

Initialize on worker process (receiver side).

This method is called once in each worker's initialization.

Parameters:

- **model_id** - Identifier for the model being synchronized
- **context** - Optional context object (e.g., inner collector)
- ****kwargs** - Alternative to context (model, etc.)

init_on_sender(***, *model_id: str*, *context: Any*) → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme.init_on_sender)

init_on_sender(***, *params_map: dict[int, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *model_id: str | None = None*) → None

init_on_sender(***, *params_map: dict[int, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*) → None

init_on_sender(***, *weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *devices: list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)]*) → None

init_on_sender(***, *weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *devices: list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)]*, *model_id: str | None = None*) → None

init_on_sender(***, *model: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *devices: list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)]*) → None

init_on_sender(***, *model: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *devices: list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)]*, *model_id: str | None = None*) → None

init_on_sender(***, *weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *device_map_fn: Callable[[int, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *num_workers: int*) → None

init_on_sender(***, *model: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *device_map_fn: Callable[[int, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *num_workers: int*, *model_id: str | None = None*) → None

init_on_sender()

Initialize on the main process (sender side).

This method is called once in the collector's _run_processes() method,
after workers have been started and are ready to receive messages.

*property*model*: Any | None*

Get the model object, if available.

Returns:

The model object if available, None otherwise.

*property*model_id*: str | None*

Get the model ID for this scheme.

Returns:

The model ID if set, None otherwise.

prepare_weights(*weights: Any*, *model_id: str*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy)*, *context: Any = None*) → Any[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme.prepare_weights)

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

receive(*timeout: float | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme.receive)

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

send(*weights: Any = None*, *worker_ids: int | list[int] | None = None*) → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme.send)

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

shutdown() → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightSyncScheme.shutdown)

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