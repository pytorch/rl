# MPTransport

*class*torchrl.weight_update.MPTransport(*weight_queue*, *ack_queue=None*, *timeout: float = 10.0*)[[source]](../../_modules/torchrl/weight_update/_mp.html#MPTransport)

Multiprocessing transport using queues.

This transport uses queues for weight distribution and synchronization.
Similar to SharedMemTransport's queue-based approach, MPTransport uses
queues to send initial weights to workers during synchronization.

Initialization flow:
- synchronize_weights() extracts weights and sends to all workers via queues
- Workers receive the initial weights via setup_connection_and_weights_on_receiver()
- Subsequent updates use send_weights_async() followed by acknowledgments

Parameters:

- **weight_queue** (*mp.Queue*) - The queue to use for sending weights.
- **ack_queue** (*mp.Queue*) - The queue to use for receiving acknowledgments.
- **timeout** (*float*) - The timeout for waiting for acknowledgment. Default is 10 seconds.

receive_weights(*timeout: float | None = None*, ***, *weights: Any = None*, *model: Any = None*, *strategy: Any = None*) → Any | None[[source]](../../_modules/torchrl/weight_update/_mp.html#MPTransport.receive_weights)

Receive weights from the queue (used in worker process).

This method only handles weight update messages. Other messages
(like "close", "continue", etc.) are ignored and should be handled
by the main worker loop.

Parameters:

- **timeout** - Maximum time to wait for weights (seconds).
None means use the transport's default timeout.
- **weights** - Ignored (weights come from queue).
- **model** - The model to apply weights to.
- **strategy** - Strategy for applying weights to the model.

Returns:

The received weights, or None if no data available.

send_weights_async(*weights: Any*, *model_id: str = 'policy'*) → None[[source]](../../_modules/torchrl/weight_update/_mp.html#MPTransport.send_weights_async)

Send weights through the queue without waiting for acknowledgment.

Use wait_ack() to wait for acknowledgment after sending to all workers.

setup_connection_and_weights_on_receiver(***, *worker_idx: int*, *weights: Any = None*, *model: Any = None*, *strategy: Any = None*) → Any[[source]](../../_modules/torchrl/weight_update/_mp.html#MPTransport.setup_connection_and_weights_on_receiver)

Receive initial weights from sender during worker initialization.

This method blocks waiting for the initial weights to be sent from the main process
via queue. Similar to SharedMemTransport.setup_connection_and_weights_on_receiver() which receives
shared memory buffer references via queues, this receives the actual weights via queues.

The received weights are then applied to the worker's model by the scheme's synchronize_weights().

Parameters:

- **worker_idx** - The worker index (used for logging/debugging).
- **weights** - Ignored (weights come from queue).
- **model** - Ignored.
- **strategy** - Ignored.

Returns:

The received weights if available, None otherwise (weights will come later via receive()).

setup_connection_and_weights_on_sender() → None[[source]](../../_modules/torchrl/weight_update/_mp.html#MPTransport.setup_connection_and_weights_on_sender)

No-op for MPTransport - weights are sent via scheme's synchronize_weights().

The actual sending happens in MultiProcessWeightSyncScheme._setup_connection_and_weights_on_sender_impl(), which:
1. Extracts weights from the context (e.g., collector.policy)
2. Calls send_weights_async() on all worker transports
3. Sends initial weights through queues to all workers

This is similar to SharedMemTransport.setup_connection_and_weights_on_sender() which
sends shared memory buffer references via queues.