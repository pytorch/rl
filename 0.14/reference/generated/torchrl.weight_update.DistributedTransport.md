# DistributedTransport

*class*torchrl.weight_update.DistributedTransport(***, *weights_buffer: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *store: [Store](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.Store) = None*, *rank: int | None = None*, *sync: bool = True*)[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedTransport)

torch.distributed transport for communicating with a single distributed worker.

This transport handles weight updates for ONE specific distributed worker via
torch.distributed send/recv. Multiple transports are created for multiple workers,
following the same pattern as multiprocess collectors.

receive_initial_weights() → Any[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedTransport.receive_initial_weights)

Receive initial weights during connect() without TCPStore signaling.

This is used for the initial weight sync during connect() to avoid
interfering with the main collection loop's TCPStore-based coordination.

Returns:

The received weights TensorDict.

receive_weights(*timeout: float | None = None*, ***, *weights: Any = None*, *model: Any = None*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) | None = None*) → Any | None[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedTransport.receive_weights)

Receive weights via torch.distributed and apply them to the model.

The surrounding collector loop is responsible for checking the TCPStore
for the "update_weights" instruction. When this method is called we
assume that a weight update has been requested and the sender has
already performed the corresponding `send()`.

Parameters:

- **timeout** - Maximum time to wait for weights (seconds). If None,
blocks until weights are received.
- **weights** - Pre-allocated weight buffer to receive into.
- **model** - The model to apply weights to.
- **strategy** - Strategy for applying weights to the model.

Returns:

The received weights, or None if timeout expires.

send_initial_weights(*weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedTransport.send_initial_weights)

Send initial weights during connect() without TCPStore signaling.

This is used for the initial weight sync during connect() to avoid
interfering with the main collection loop's TCPStore-based coordination.

send_weights(*weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedTransport.send_weights)

Send weights to the distributed worker.

send_weights_async(*weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedTransport.send_weights_async)

Send weights to distributed worker without waiting for acknowledgment.

Use wait_ack() to wait for acknowledgment after sending to all workers.

setup_connection_and_weights_on_receiver(***, *worker_idx: int*, *weights: Any = None*, *model: Any = None*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) | None = None*) → Any[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedTransport.setup_connection_and_weights_on_receiver)

No-op for DistributedTransport - handled by scheme.

setup_connection_and_weights_on_sender() → None[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedTransport.setup_connection_and_weights_on_sender)

No-op for DistributedTransport - handled by scheme.

wait_ack() → None[[source]](../../_modules/torchrl/weight_update/_distributed.html#DistributedTransport.wait_ack)

Wait for acknowledgment from distributed worker.