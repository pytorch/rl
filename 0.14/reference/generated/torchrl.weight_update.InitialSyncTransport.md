# InitialSyncTransport

*class*torchrl.weight_update.InitialSyncTransport(**args*, ***kwargs*)[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#InitialSyncTransport)

Optional initial-connection capability for weight transports.

setup_connection_and_weights_on_receiver(***, *worker_idx: int*, *weights: Any = None*, *model: Any = None*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) | None = None*) → Any[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#InitialSyncTransport.setup_connection_and_weights_on_receiver)

Synchronize weights on the receiver before collection starts.

setup_connection_and_weights_on_sender() → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#InitialSyncTransport.setup_connection_and_weights_on_sender)

Synchronize weights on sender side before collection starts.

This is called once after workers are initialized to send the initial
weights. This can be a no-op (weights are sent via
send_weights).