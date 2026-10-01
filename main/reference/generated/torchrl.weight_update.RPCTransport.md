# RPCTransport

*class*torchrl.weight_update.RPCTransport(*collector_info=None*, *collector_rref=None*, *collector_class=None*, *worker_rank=None*)[[source]](../../_modules/torchrl/weight_update/_rpc.html#RPCTransport)

RPC transport for communicating with a single RPC remote collector.

This transport handles weight updates for ONE specific remote collector via
torch.distributed primitives (send/recv) with RPC used for signaling.
Multiple transports are created for multiple collectors, following the same
pattern as the DistributedCollector.

receive_weights(*timeout: float | None = None*, ***, *weights: Any = None*, *model: Any = None*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) | None = None*) → Any | None[[source]](../../_modules/torchrl/weight_update/_rpc.html#RPCTransport.receive_weights)

Receive weights from sender using torch.distributed.

Parameters:

- **timeout** - Maximum time to wait for weights (seconds). If None,
blocks until weights are received.
- **weights** - Pre-allocated weight buffer to receive into.
- **model** - The model to apply weights to.
- **strategy** - Strategy for applying weights to the model.

Returns:

The received weights, or None if timeout expires.

send_weights(*weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/_rpc.html#RPCTransport.send_weights)

Send weights to the remote collector using torch.distributed.

Uses torch.distributed.send() for the actual weight transfer and RPC
for signaling the remote collector to receive.

Order is critical to avoid deadlock:
1. Signal receiver via RPC to start recv() (non-blocking)
2. Send weights via torch.distributed (blocking until recv completes)

send_weights_async(*weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/_rpc.html#RPCTransport.send_weights_async)

Send weights to remote collector asynchronously.

Uses torch.distributed.isend() for the actual weight transfer and RPC
for signaling. Use wait_ack() to wait for completion.

Order is critical to avoid deadlock:
1. Signal receiver via RPC to start recv() (non-blocking)
2. Send weights via torch.distributed.isend() (non-blocking)
3. wait_ack() waits for both to complete

setup_connection_and_weights_on_receiver(***, *worker_idx: int*, *weights: Any = None*, *model: Any = None*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) | None = None*) → Any[[source]](../../_modules/torchrl/weight_update/_rpc.html#RPCTransport.setup_connection_and_weights_on_receiver)

No-op for RPCTransport - weights are received via receive().

setup_connection_and_weights_on_sender() → None[[source]](../../_modules/torchrl/weight_update/_rpc.html#RPCTransport.setup_connection_and_weights_on_sender)

No-op for RPCTransport - weights are sent via send_weights().

wait_ack() → None[[source]](../../_modules/torchrl/weight_update/_rpc.html#RPCTransport.wait_ack)

Wait for both the RPC call and the distributed send to complete.