# RayTransport

*class*torchrl.modules.inference_server.RayTransport(***, *max_queue_size: int = 1000*)[[source]](../../_modules/torchrl/modules/inference_server/_ray.html#RayTransport)

Transport using Ray queues for distributed inference.

Uses `ray.util.queue.Queue` for both request submission and response
routing. Per-actor response queues ensure correct result routing without
serialising Queue objects through other queues.

Ray is imported lazily at instantiation time; importing the class itself
does not require Ray.

Keyword Arguments:

**max_queue_size** (*int*) - maximum size of the request queue.
Default: `1000`.

Example

```
>>> import ray
>>> ray.init()
>>> transport = RayTransport()
>>> client = transport.client()
>>> # pass *client* to a Ray actor for remote inference requests
```

client() → MailboxClient[[source]](../../_modules/torchrl/modules/inference_server/_ray.html#RayTransport.client)

Create an actor-side client with a dedicated Ray response queue.

Returns:

A `_QueueInferenceClient` that can be used inside any Ray
actor or the driver process.