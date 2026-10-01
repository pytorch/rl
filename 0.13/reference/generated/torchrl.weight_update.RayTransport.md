# RayTransport

*class*torchrl.weight_update.RayTransport(***, *remote_actor=None*, *worker_idx: int | None = None*, *backend: str = 'gloo'*, *connection_info_name: str = 'connection_info'*, *model_id: str | None = None*)[[source]](../../_modules/torchrl/weight_update/_ray.html#RayTransport)

Ray transport for communicating with a single Ray actor.

This transport handles weight updates for ONE specific remote actor
using torch.distributed for efficient weight transfer. Ray is used for
signaling/coordination, while the actual weight data is transferred via
torch.distributed send/recv operations.

Multiple transports are created for multiple actors, following the
same pattern as multiprocess collectors.

Parameters:

- **remote_actor** - The Ray actor handle for the remote collector/transform.
- **worker_idx** (*int**,**optional*) - The worker index for this remote actor.
Defaults to 0.
- **backend** (*str*) - The torch.distributed backend to use ("gloo" or "nccl").
Defaults to "gloo".
- **connection_info_name** (*str*) - Name of the Ray actor storing connection info.
Defaults to "connection_info".
- **model_id** (*str**,**optional*) - The model identifier for weight synchronization.

receive_weights(*timeout: float | None = None*, ***, *weights: Any = None*, *model: Any = None*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) | None = None*) → Any | None[[source]](../../_modules/torchrl/weight_update/_ray.html#RayTransport.receive_weights)

Receive weights from sender via torch.distributed.

Parameters:

- **timeout** - Maximum time to wait for weights (seconds). If None,
blocks until weights are received.
- **weights** - Pre-allocated weight buffer to receive into.
- **model** - The model to apply weights to.
- **strategy** - Strategy for applying weights to the model.

Returns:

The received weights, or None if timeout expires.

send_weights(*weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/_ray.html#RayTransport.send_weights)

Send weights to the remote actor via torch.distributed.

This method:
1. Signals the remote actor to start receiving via Ray remote call
2. Sends weights via torch.distributed.isend
3. Waits for both to complete

Parameters:

**weights** - The weights to send (typically a TensorDict).

send_weights_async(*weights: Any*) → None[[source]](../../_modules/torchrl/weight_update/_ray.html#RayTransport.send_weights_async)

Send weights to Ray actor without waiting for completion.

Use `wait_ack()` to wait for completion after sending to all actors.

Parameters:

**weights** - The weights to send (typically a TensorDict).

set_model(*model: Any*) → None[[source]](../../_modules/torchrl/weight_update/_ray.html#RayTransport.set_model)

Set the model for receiving weights.

Parameters:

**model** - The model to receive weights into.

setup_connection_and_weights_on_receiver(***, *worker_idx: int*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy) | None = None*, *model: Any | None = None*, *weights: Any | None = None*) → Any[[source]](../../_modules/torchrl/weight_update/_ray.html#RayTransport.setup_connection_and_weights_on_receiver)

Join torch.distributed process group and receive initial weights.

This method:
1. Retrieves connection info from the shared Ray actor
2. Initializes torch.distributed process group with rank=worker_idx+1
3. Receives weights if model is stateful

Parameters:

- **worker_idx** (*int*) - The worker index for this transport.
- **strategy** ([*WeightStrategy*](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy)*,**optional*) - The weight transmission strategy.
- **model** (*nn.Module**or**compatible**,**optional*) - The model to receive weights for.
- **weights** (*TensorDict**,**optional*) - Pre-allocated buffer for receiving weights.

Returns:

The received weights (TensorDict) if model is stateful, None otherwise.

setup_connection_and_weights_on_sender() → None[[source]](../../_modules/torchrl/weight_update/_ray.html#RayTransport.setup_connection_and_weights_on_sender)

Initialize torch.distributed on sender side for this worker's rank.

This is called by the scheme after it has created the connection info
Ray actor. The actual `init_process_group` happens in the scheme since
it's a collective operation that needs to happen for rank 0.

Note

This method exists for interface compatibility but the real work
happens in the scheme's `_setup_distributed_connection_sender()`.

wait_ack() → None[[source]](../../_modules/torchrl/weight_update/_ray.html#RayTransport.wait_ack)

Wait for Ray actor to finish applying weights.

Raises:

**RuntimeError** - If no pending future exists (i.e., `send_weights_async()`
 was not called before this method).