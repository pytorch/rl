# TransportBackend

*class*torchrl.weight_update.TransportBackend(**args*, ***kwargs*)[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#TransportBackend)

Abstract interface for different communication mechanisms.

receive_weights(*timeout: float | None = None*, ***, *weights: Any = None*, *model: Any = None*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) | None = None*) → Any | None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#TransportBackend.receive_weights)

Receive weights from the sender and apply them to the model.

Parameters:

- **timeout** - Maximum time to wait for weights (seconds).
None means no timeout (blocking). Some transports may not
support timeout and will raise ValueError if specified.
- **weights** - Pre-allocated weight buffer to receive into.
- **model** - The model to apply weights to.
- **strategy** - Strategy for applying weights to the model.

Returns:

The received/applied weights, or None if timeout/no weights available.

send_weights(*weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#TransportBackend.send_weights)

Send weights to the receiver.

setup_connection_and_weights_on_receiver(***, *worker_idx: int*, *weights: Any = None*, *model: Any = None*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) | None = None*) → Any[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#TransportBackend.setup_connection_and_weights_on_receiver)

Synchronize weights on worker side before collection starts.

This is called once in each worker after initialization to receive
the initial weights. This is a no-op (weights are received via
receive_weights).

Parameters:

- **worker_idx** - The worker index.
- **weights** - Pre-allocated weight buffer to receive into.
- **model** - The model to apply weights to.
- **strategy** - Strategy for applying weights to the model.

Returns:

The received weights (for SharedMemTransport) or None.

setup_connection_and_weights_on_sender() → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#TransportBackend.setup_connection_and_weights_on_sender)

Synchronize weights on sender side before collection starts.

This is called once after workers are initialized to send the initial
weights. This can be a no-op (weights are sent via
send_weights).