# SharedMemWeightSyncScheme

*class*torchrl.weight_update.SharedMemWeightSyncScheme(*strategy: str = 'tensordict'*, *sync: bool = True*, *per_worker_weights: bool = False*, *atomic: bool | None = None*)[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme)

Weight synchronization using shared memory.

This scheme uses shared memory buffers for weight updates, so no weights
are pickled or sent through a pipe. A background thread on the receiver
side listens for "receive" instructions from the sender. When an
instruction arrives, the thread applies the shared buffer to the model and
sends an acknowledgment.

Two modes are available, selected by `atomic`:

- `atomic=False` (the current behavior, also used when `atomic` is
not passed): the shared buffers are the sender's own weights, and each
worker binds them as its parameters. The workers see every in-place
change on the sender, including an optimizer step, without any
`send()` call. Because those writes land while a
worker's policy is running, a policy call can read a mix of two weight
versions.
- `atomic=True`: the sender copies the weights into a shared staging
buffer on `send()` (one per device, or one per worker with
`per_worker_weights=True`), and each worker copies that buffer into
its own private parameters while holding the lock that the receiving
collector (or `InferenceServer`) holds around
each policy call. A policy call therefore always reads a single, complete
version of the weights, and workers only see new weights after
`send()` (for instance through `update_policy_weights_()`). Memory
grows by one copy of the weights per worker plus the staging buffers,
and each update costs one extra copy on the sender and on every
worker.

The default changes to `atomic=True` in v0.16. Code that trains the
sender's policy and relies on the workers seeing the change without calling
`update_policy_weights_()` must then pass `atomic=False` or call
`update_policy_weights_()` after each update.

Parameters:

- **strategy** (*str**,**optional*) - The weight transmission strategy,
`"tensordict"` or `"state_dict"`. Defaults to `"tensordict"`.
- **sync** (*bool**,**optional*) - If `True`, `send()` blocks until every
receiver acknowledges. If `False`, `send()` returns immediately;
with `atomic=True` the next `send()` first waits for the
previous acknowledgments, so that a staging buffer is never
overwritten while a worker copies it. With `atomic=True` each
wait lasts at most `WEIGHT_SYNC_TIMEOUT` seconds per worker, so a
dead worker does not block the sender. Defaults to `True`.
- **per_worker_weights** (*bool**,**optional*) - If `True`, each worker
maintains independent weights (for distinct policy factories). If
`False`, workers on the same device share the same weight buffer.
This flag is auto-set to `True` when distinct policy factories are
detected. Defaults to `False`.
- **atomic** (*bool**,**optional*) - If `True`, workers keep a private copy of the
weights and apply each update between two policy calls, as
described above. If `False`, workers read the sender's weights in
place. `None` behaves like `False` and emits a
`FutureWarning` when the scheme is initialized on the sender,
since the default becomes `True` in v0.16. Defaults to `None`.

Example

```
>>> from torchrl.weight_update import SharedMemWeightSyncScheme
>>> scheme = SharedMemWeightSyncScheme(atomic=True)
>>> scheme.atomic
True
>>> # Pass it to a multiprocess collector, then push each new set of
>>> # weights to the workers explicitly:
>>> # collector = MultiSyncCollector(
>>> # [make_env] * 2, policy, frames_per_batch=64, total_frames=-1,
>>> # weight_sync_schemes={"policy": scheme},
>>> # )
>>> # for data in collector:
>>> # ... # optimizer step on policy
>>> # collector.update_policy_weights_()
```

apply_weights(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *inplace: bool = True*) → None

Apply weights to the model.

Parameters:

- **weights** - The weights to apply.
- **inplace** - Whether to apply weights in place. Default is True.

connect(***, *worker_idx: int | None = None*, *weights: Any | None = None*) → None

Method to be called once the workers have started.

Triggers a rendez-vous for the workers to receive their copy of the weights.

Dispatches to _setup_connection_and_weights_on_sender_impl() or _setup_connection_and_weights_on_receiver_impl()
based on which initialization was performed.

*property*context*: Any | None*

Get the context object (e.g., collector), if available.

Returns:

The context object if available, None otherwise.

create_transport(***kwargs*) → [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme.create_transport)

Create shared memory transport.

Returns the shared transport instance that all workers will use.
Since this is shared memory, there's only one transport shared by all workers.

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

prepare_weights(*weights: Any*, *model_id: str*, *strategy: [WeightStrategy](torchrl.weight_update.WeightStrategy.html#torchrl.weight_update.WeightStrategy)*, *context: Any = None*) → Any[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme.prepare_weights)

Prepare weights for SharedMemWeightSyncScheme.

When weights=None, we extract fresh weights from the model and update
the shared memory buffer in-place so workers see the change.

Parameters:

- **weights** - Raw weights input
- **model_id** - The model identifier
- **strategy** - WeightStrategy for extracting/converting weights
- **context** - Optional context (e.g., collector) for cache lookup

Returns:

Shared memory weights ready to send

receive(*timeout: float | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme.receive)

Apply the shared buffer to the model, as the background thread does.

The application holds the receiving context's weight-update lock, so a
manual call cannot race with a policy call or with the background
receiver thread.

*property*receiver_transport*: [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend) | None*

Get the receiver transport.

Returns:

The receiver transport.

send(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict[int, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | None = None*, *worker_ids: int | list[int] | None = None*) → None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme.send)

Send weights via shared memory (in-place update).

For SharedMemWeightSyncScheme:
1. With `atomic=True`, waits until every worker has finished

> copying the previous update, so that a staging buffer is never
> written while a worker reads it

1. Updates the shared memory buffer(s) in-place
2. _send_instruction() tells workers to apply the new weights (with
`atomic=True`, between two policy calls)
3. If sync=True, waits for acknowledgments from all workers (with
`atomic=True`, at most `WEIGHT_SYNC_TIMEOUT` seconds per worker)

Parameters:

- **weights** - Weights to send. Can be:
- None: Extract weights from the model
- TensorDictBase: Same weights broadcast to all workers
- dict[int, TensorDictBase]: Per-worker weights (keys are worker indices)
- **worker_ids** - Which workers to notify (None = all workers).

*property*sender_transports*: dict[int, [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend)]*

Get the sender transports.

Returns:

The sender transports.

*property*shared_transport*: [TransportBackend](torchrl.weight_update.TransportBackend.html#torchrl.weight_update.TransportBackend) | None*

Get the shared transport.

Returns:

The shared transport.

shutdown() → None[[source]](../../_modules/torchrl/weight_update/_shared.html#SharedMemWeightSyncScheme.shutdown)

Stop the background receiver thread and clean up.

*property*weights*: Any | None*

Get the current weights from shared memory.

For SharedMemWeightSyncScheme:
- On sender side: weights are in transport's _unique_weights
- On receiver side: weights are in _receiver_shared_weights (stored during connect())

Returns:

The weights TensorDict if available, None otherwise.

*property*worker_idx*: int | None*

Get the worker index for this scheme.

Returns:

The worker index if set, None otherwise.