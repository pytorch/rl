# MPTransport

*class*torchrl.modules.inference_server.MPTransport(*ctx: BaseContext | None = None*, ***, *use_manager: bool = False*)[[source]](../../_modules/torchrl/modules/inference_server/_mp.html#MPTransport)

Cross-process transport using `multiprocessing` queues.

Response routing uses per-actor queues (one per `client()` call) so
that no `mp.Queue` object is ever serialised through another queue.
Clients must be created with `client()` **before** spawning child
processes.

Parameters:

- **ctx** - a multiprocessing context (e.g. `mp.get_context("spawn")`).
Defaults to `mp.get_context("spawn")`.
- **use_manager** (*bool**,**optional*) - if `True`, back the request and
response queues with a multiprocessing manager. This is useful
when clients are forwarded through another spawned process.
Defaults to `False`.

Example

```
>>> import multiprocessing as mp
>>> transport = MPTransport()
>>> client = transport.client() # creates response queue
>>> p = mp.Process(target=actor_fn, args=(client,))
>>> p.start() # queue inherited
```

client() → MailboxClient[[source]](../../_modules/torchrl/modules/inference_server/_mp.html#MPTransport.client)

Create an actor-side client with a dedicated response queue.

Must be called in the parent process **before** spawning children.
In particular, a manager-backed transport (`use_manager=True`)
loses its manager handle when pickled, so calling this on an
unpickled copy raises a `RuntimeError`.

Returns:

A `_QueueInferenceClient` that can be passed to a child
process as an argument to `multiprocessing.Process`.

close() → None[[source]](../../_modules/torchrl/modules/inference_server/_mp.html#MPTransport.close)

Release transport resources.

Shuts down the multiprocessing manager backing the request/response
queues when the transport was built with `use_manager=True`
(a no-op otherwise). The process that owns the transport must call
this once the server and all clients are done with it:
[`ProcessInferenceServer`](torchrl.modules.inference_server.ProcessInferenceServer.html#torchrl.modules.inference_server.ProcessInferenceServer)
does not close the transport on `shutdown()`.