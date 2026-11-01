# MPTransport

*class*torchrl.modules.inference_server.MPTransport(*ctx: BaseContext | None = None*)[[source]](../../_modules/torchrl/modules/inference_server/_mp.html#MPTransport)

Cross-process transport using `multiprocessing` queues.

Response routing uses per-actor queues (one per `client()` call) so
that no `mp.Queue` object is ever serialised through another queue.
Clients must be created with `client()` **before** spawning child
processes.

Parameters:

**ctx** - a multiprocessing context (e.g. `mp.get_context("spawn")`).
Defaults to `mp.get_context("spawn")`.

Example

```
>>> import multiprocessing as mp
>>> transport = MPTransport()
>>> client = transport.client() # creates response queue
>>> p = mp.Process(target=actor_fn, args=(client,))
>>> p.start() # queue inherited
```

client() → _QueueInferenceClient[[source]](../../_modules/torchrl/modules/inference_server/_mp.html#MPTransport.client)

Create an actor-side client with a dedicated response queue.

Must be called in the parent process **before** spawning children.

Returns:

A `_QueueInferenceClient` that can be passed to a child
process as an argument to `multiprocessing.Process`.