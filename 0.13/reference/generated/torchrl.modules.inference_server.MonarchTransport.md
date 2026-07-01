# MonarchTransport

*class*torchrl.modules.inference_server.MonarchTransport(***, *max_queue_size: int = 1000*)[[source]](../../_modules/torchrl/modules/inference_server/_monarch.html#MonarchTransport)

Transport using Monarch for distributed inference on GPU clusters.

Uses Monarch's actor model and RDMA-capable channels for efficient
cross-node communication. Monarch is imported lazily at instantiation
time; importing the class itself does not require Monarch.

Note

This transport requires `monarch` to be installed. It is designed
for large-scale GPU clusters where Monarch is the preferred
communication layer.

Keyword Arguments:

**max_queue_size** (*int*) - maximum size of the request queue.
Default: `1000`.

client() → _QueueInferenceClient[[source]](../../_modules/torchrl/modules/inference_server/_monarch.html#MonarchTransport.client)

Create an actor-side client with a dedicated response queue.

Returns:

A `_QueueInferenceClient` that can be passed to a Monarch
actor.