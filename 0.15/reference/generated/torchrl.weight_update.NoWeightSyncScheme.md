# NoWeightSyncScheme

*class*torchrl.weight_update.NoWeightSyncScheme(*strategy: Literal['state_dict', 'tensordict'] = 'tensordict'*)[[source]](../../_modules/torchrl/weight_update/_noupdate.html#NoWeightSyncScheme)

No-op weight synchronization scheme.

This scheme disables weight synchronization entirely.

apply_weights(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *inplace: bool = True*) → None

Apply weights to the model.

Parameters:

- **weights** - The weights to apply.
- **inplace** - Whether to apply weights in place. Default is True.

connect(***, *worker_idx: int | None = None*) → None[[source]](../../_modules/torchrl/weight_update/_noupdate.html#NoWeightSyncScheme.connect)

No-op synchronize - does nothing.

*property*context*: Any | None*

Get the context object (e.g., collector), if available.

Returns:

The context object if available, None otherwise.

create_transport(***kwargs*) → [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)[[source]](../../_modules/torchrl/weight_update/_noupdate.html#NoWeightSyncScheme.create_transport)

Create a no-op transport.

Note

This is used internally by init_on_sender/init_on_receiver.

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

Get the model object, if available.

Returns:

The model object if available, None otherwise.

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

receive(*timeout: float | None = None*) → None[[source]](../../_modules/torchrl/weight_update/_noupdate.html#NoWeightSyncScheme.receive)

No-op receive - always returns None.

*property*receiver_transport*: [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend) | None*

Get the receiver transport.

Returns:

The receiver transport.

send(*weights: Any = None*, *worker_ids: int | list[int] | None = None*) → None[[source]](../../_modules/torchrl/weight_update/_noupdate.html#NoWeightSyncScheme.send)

No-op send - does nothing.

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