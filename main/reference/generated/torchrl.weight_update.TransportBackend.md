# TransportBackend

*class*torchrl.weight_update.TransportBackend(**args*, ***kwargs*)[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#TransportBackend)

Core data-plane interface for weight communication mechanisms.

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