# SharedMemTransport

*class*torchrl.weight_update.SharedMemTransport[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemTransport)

Shared memory transport for in-place weight updates.

This transport uses queue-based buffer distribution for initialization, then
updates shared memory tensors directly for subsequent weight updates.
Workers automatically see weight updates without explicit communication.

Initialization flow:
- Shared memory buffers are created and sent to workers via per-worker queues
- Workers receive the buffer reference and apply weights to their models
- Subsequent updates are pure in-place shared memory (zero-copy)

Both CPU and CUDA tensors maintain shared references when sent through mp.Queue.

receive_weights(*timeout: float | None = None*, ***, *weights: Any = None*, *model: Any = None*, *strategy: Any = None*) → Any | None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemTransport.receive_weights)

Apply shared memory weights to the model.

For shared memory, weights are already available (passed via the weights arg).
This method applies them to the model, matching the pattern of other transports.

Parameters:

- **timeout** - Ignored (shared memory access is instant).
- **weights** - The shared memory buffer containing current weights.
- **model** - The model to apply weights to.
- **strategy** - Strategy for applying weights.

Returns:

The applied weights, or None if not applied.

register_weights(*params_map: dict[int, Queue]*, *init_queues: dict[int, Queue]*, *per_worker: bool = False*) → None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemTransport.register_weights)

Initialize per-worker queues for shared memory buffer distribution.

Parameters:

- **params_map** - Mapping of worker_idx to weights TensorDict.
- **init_queues** - Mapping of worker_idx to initialization queues.
- **per_worker** - If True, each worker gets its own independent weight copy
(for distinct policy factories). If False, workers on the same
device share the same weight buffer (device-based deduplication).

send_ack(*message: str = 'updated'*) → None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemTransport.send_ack)

No-op for shared memory - no acknowledgment needed.

send_weights(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict[int, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *worker_ids: list[int] | None = None*) → None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemTransport.send_weights)

Update weights in-place in shared memory.

Parameters:

- **weights** - New weights to send. Can be:
- TensorDictBase: Same weights broadcast to all workers (or subset via worker_ids)
- dict[int, TensorDictBase]: Per-worker weights (keys are worker indices)
- **worker_ids** - Which workers to update (None = all workers). Only used when
weights is a TensorDictBase (broadcast mode).

Raises:

**ValueError** - If weights type is unsupported.

setup_connection_and_weights_on_receiver(***, *worker_idx: int | None = None*, *weights: Any = None*, *model: Any = None*, *strategy: Any = None*, *timeout: float = 60.0*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemTransport.setup_connection_and_weights_on_receiver)

Receive shared memory buffer reference from sender via their per-worker queues.

Each worker reads from its own dedicated queue, to avoid race conditions.

Parameters:

- **worker_idx** - The worker index.
- **weights** - Ignored (weights come from queue).
- **model** - Ignored.
- **strategy** - Ignored.
- **timeout** - Timeout for reading from queue.

Returns:

The shared memory weights TensorDict.

setup_connection_and_weights_on_sender() → None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemTransport.setup_connection_and_weights_on_sender)

Send shared memory buffer reference to workers via their per-worker queues.

Both CPU and CUDA tensors maintain shared references through queues.
Each worker reads from its own dedicated queue, to avoid race conditions.

*property*unique_weights*: list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*

Get the unique weights.

Returns:

The unique weights.