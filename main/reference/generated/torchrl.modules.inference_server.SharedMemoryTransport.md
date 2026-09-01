# SharedMemoryTransport

*class*torchrl.modules.inference_server.SharedMemoryTransport(*request_spec: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *response_spec: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *num_slots: int*, *ctx: BaseContext | None = None*, *copy_result: bool = True*)[[source]](../../_modules/torchrl/modules/inference_server/_shared_memory.html#SharedMemoryTransport)

Cross-process transport backed by shared-memory TensorDict slots.

Unlike [`MPTransport`](torchrl.modules.inference_server.MPTransport.html#torchrl.modules.inference_server.MPTransport), which
pickles full request/response TensorDicts through multiprocessing queues,
this transport preallocates two CPU shared-memory slot banks (one for
requests, one for responses) and passes only slot indices through the
queues. This removes per-request serialization of large payloads (e.g.
image observations) from the hot path.

A slot is owned by exactly one in-flight request: the client acquires a
slot from a shared free-slot pool, copies the request tensors into it,
and releases it once the response has been read. `num_slots` therefore
bounds the number of concurrently in-flight requests; when all slots are
busy, `submit()` blocks until one is
released.

Device rules: slots live in CPU shared memory, clients must submit CPU
tensors (a CUDA leaf raises a `ValueError`), and the server owns
all device transfers - batches are moved to the policy device by
[`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer) and results
are copied back into the CPU response slots by `resolve()`.

Parameters:

- **request_spec** (*TensorDictBase*) - a representative single request. Its
keys, shapes, dtypes, and batch size define the request slot
layout. All leaves must be CPU tensors.
- **response_spec** (*TensorDictBase*) - a representative single response
(including any server-added keys to forward, such as
`"policy_version"`). All leaves must be CPU tensors.

Keyword Arguments:

- **num_slots** (*int*) - number of preallocated slots, i.e. the maximum
number of concurrently in-flight requests across all clients.
- **ctx** (*multiprocessing context**,**optional*) - the multiprocessing context
used for the control queues. Defaults to
`mp.get_context("spawn")`.
- **copy_result** (*bool**,**optional*) - if `True` (default),
`Future.result()` returns a clone of the response slot. If
`False`, it returns a view into the shared response slot that
is only valid until the slot is reused by a later request;
callers must consume (or copy) it before submitting again.

Note

Only the keys declared in the specs are transmitted: extra keys on
submitted tensordicts and on model outputs are silently dropped, and
a missing declared key raises a `KeyError`. Non-tensor leaves
are not supported; encode small metadata as tensors (e.g. static
instruction ids) or use [`MPTransport`](torchrl.modules.inference_server.MPTransport.html#torchrl.modules.inference_server.MPTransport).

Note

As with [`MPTransport`](torchrl.modules.inference_server.MPTransport.html#torchrl.modules.inference_server.MPTransport), clients must be created with
`client()` in the owning process **before** spawning child
processes, so that their response queues and the shared slot banks
are inherited by the workers.

Example

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules.inference_server import (
... InferenceServer,
... SharedMemoryTransport,
... )
>>> request_spec = TensorDict({"observation": torch.zeros(4)})
>>> response_spec = TensorDict(
... {
... "action": torch.zeros(2),
... "policy_version": torch.zeros((), dtype=torch.long),
... }
... )
>>> transport = SharedMemoryTransport(
... request_spec, response_spec, num_slots=8
... )
>>> client = transport.client() # create before spawning workers
>>> policy = TensorDictModule(
... torch.nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
... )
>>> with InferenceServer(policy, transport, max_batch_size=4):
... result = client(TensorDict({"observation": torch.randn(4)}))
>>> assert result["action"].shape == (2,)
```

client() → _SharedMemorySlotClient[[source]](../../_modules/torchrl/modules/inference_server/_shared_memory.html#SharedMemoryTransport.client)

Create an actor-side client with a dedicated response queue.

Must be called in the owning process **before** spawning children so
that the response queue and the shared slot banks are inherited.

Returns:

A `_SharedMemorySlotClient` that can be passed to a child
process as an argument to `multiprocessing.Process`.

drain_with_timing(*max_items: int*) → tuple[list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], list[tuple[tuple[int, int], int]], list[float | None]][[source]](../../_modules/torchrl/modules/inference_server/_shared_memory.html#SharedMemoryTransport.drain_with_timing)

Dequeue request headers and return views into the request slots.

resolve(*callback: tuple[tuple[int, int], int]*, *result: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/modules/inference_server/_shared_memory.html#SharedMemoryTransport.resolve)

Copy the result into the response slot and notify the client.

Result tensors on a non-CPU device are copied back to the CPU slots
leaf-by-leaf, so no CUDA tensor ever crosses a queue.

resolve_exception(*callback: tuple[tuple[int, int], int]*, *exc: BaseException*) → None[[source]](../../_modules/torchrl/modules/inference_server/_shared_memory.html#SharedMemoryTransport.resolve_exception)

Propagate an exception; the client releases the slot on receipt.