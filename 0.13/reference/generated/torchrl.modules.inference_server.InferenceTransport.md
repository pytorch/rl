# InferenceTransport

*class*torchrl.modules.inference_server.InferenceTransport[[source]](../../_modules/torchrl/modules/inference_server/_transport.html#InferenceTransport)

Abstract base class for inference server transport backends.

A transport handles the communication between actor-side clients and the
server-side inference loop. Concrete implementations provide the mechanism
for submitting requests, draining batches, and routing results back.

Subclasses must implement `submit()`, `drain()`, `wait_for_work()`,
and `resolve()`.

client() → [InferenceClient](torchrl.modules.inference_server.InferenceClient.html#torchrl.modules.inference_server.InferenceClient)[[source]](../../_modules/torchrl/modules/inference_server/_transport.html#InferenceTransport.client)

Return an actor-side [`InferenceClient`](torchrl.modules.inference_server.InferenceClient.html#torchrl.modules.inference_server.InferenceClient) bound to this transport.

*abstract*drain(*max_items: int*) → tuple[list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], list][[source]](../../_modules/torchrl/modules/inference_server/_transport.html#InferenceTransport.drain)

Drain up to *max_items* pending requests from the queue.

Called on the server side. Returns a pair `(inputs, callbacks)` where
`inputs` is a list of TensorDicts and `callbacks` is a list of
opaque objects that `resolve()` knows how to fulfil.

Parameters:

**max_items** (*int*) - maximum number of items to dequeue.

Returns:

Tuple of (inputs, callbacks).

*abstract*resolve(*callback*, *result: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/modules/inference_server/_transport.html#InferenceTransport.resolve)

Send a result back to the actor that submitted the request.

Parameters:

- **callback** - an opaque handle returned by `drain()`.
- **result** (*TensorDictBase*) - the inference output for this request.

*abstract*resolve_exception(*callback*, *exc: BaseException*) → None[[source]](../../_modules/torchrl/modules/inference_server/_transport.html#InferenceTransport.resolve_exception)

Propagate an exception back to the actor that submitted the request.

Parameters:

- **callback** - an opaque handle returned by `drain()`.
- **exc** (*BaseException*) - the exception to propagate.

*abstract*submit(*td: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → Future[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/modules/inference_server/_transport.html#InferenceTransport.submit)

Submit a single inference request.

Called on the actor side. Returns a `Future`
(or future-like object) that will be resolved with the inference result.

Parameters:

**td** (*TensorDictBase*) - a single (unbatched) input tensordict.

Returns:

A Future that resolves to the output TensorDictBase.

*abstract*wait_for_work(*timeout: float*) → None[[source]](../../_modules/torchrl/modules/inference_server/_transport.html#InferenceTransport.wait_for_work)

Block until new work is available or *timeout* seconds elapse.

Called on the server side before `drain()`.

Parameters:

**timeout** (*float*) - maximum seconds to wait.