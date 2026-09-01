# ThreadingTransport

*class*torchrl.modules.inference_server.ThreadingTransport[[source]](../../_modules/torchrl/modules/inference_server/_threading.html#ThreadingTransport)

In-process transport for actors that are threads.

Uses a shared list protected by a `threading.Condition` as the
request queue and `Future` objects for response
routing.

This is the simplest backend and is appropriate when all actors live in the
same process (e.g. running in a `ThreadPoolExecutor`).

drain(*max_items: int*) → tuple[list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], list[Future]][[source]](../../_modules/torchrl/modules/inference_server/_threading.html#ThreadingTransport.drain)

Dequeue up to *max_items* pending requests.

drain_with_timing(*max_items: int*) → tuple[list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], list[Future], list[float]][[source]](../../_modules/torchrl/modules/inference_server/_threading.html#ThreadingTransport.drain_with_timing)

Dequeue requests with actor-side submission timestamps.

resolve(*callback: Future*, *result: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/modules/inference_server/_threading.html#ThreadingTransport.resolve)

Set the result on the actor's Future.

resolve_exception(*callback: Future*, *exc: BaseException*) → None[[source]](../../_modules/torchrl/modules/inference_server/_threading.html#ThreadingTransport.resolve_exception)

Set an exception on the actor's Future.

submit(*td: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → Future[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/modules/inference_server/_threading.html#ThreadingTransport.submit)

Enqueue a request and return a Future for the result.

wait_for_work(*timeout: float*) → None[[source]](../../_modules/torchrl/modules/inference_server/_threading.html#ThreadingTransport.wait_for_work)

Block until at least one request is enqueued or *timeout* elapses.