# ProcessSlotTransport

*class*torchrl.modules.inference_server.ProcessSlotTransport(*request_spec: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *response_spec: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *num_slots: int*, *ctx: BaseContext | None = None*, *copy_result: bool = True*)[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport)

Fixed-slot shared-memory transport for environment worker processes.

Each client owns one CPU shared-memory request/response slot. A worker
copies an observation into its slot and releases a process-shared
semaphore; the inference server sweeps ready slots in round-robin order,
batches their tensor views, writes actions back, and wakes the matching
workers. Only synchronization signals cross process boundaries on the
inference hot path.

This transport allows environment workers and a
[`ProcessInferenceServer`](torchrl.modules.inference_server.ProcessInferenceServer.html#torchrl.modules.inference_server.ProcessInferenceServer) to
communicate without routing observations or actions through the driver.
Each client permits one in-flight request, which naturally applies
per-environment backpressure.

Parameters:

- **request_spec** (*TensorDictBase*) - representative request whose keys,
shapes, dtypes and batch size define each request slot. Leaves
must be CPU tensors.
- **response_spec** (*TensorDictBase*) - representative response, including
server-added keys such as `"policy_version"`. Leaves must be
CPU tensors.

Keyword Arguments:

- **num_slots** (*int*) - number of fixed slots and maximum number of clients.
- **ctx** (*multiprocessing context**,**optional*) - context used for process
synchronization primitives. Defaults to `"spawn"`.
- **copy_result** (*bool**,**optional*) - whether clients clone responses before
returning them. Defaults to `True`. If `False`, a response is
a borrowed view valid only until that client submits again.

Note

Create at most one client per environment worker. Unlike queue-based
transports, clients do not need registration with the already-running
server because every slot and signal is allocated at construction.

Note

[`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer) serves this
transport with one batched pass per sweep: ready slots are gathered
straight into a host staging batch (pinned when the policy runs on
CUDA), copied to the policy device without blocking, and the responses
are copied back and scattered into the response slots with one copy
per leaf. One CUDA event per pass replaces device-wide synchronization.

Example

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules.inference_server import (
... InferenceServer,
... ProcessSlotTransport,
... )
>>> transport = ProcessSlotTransport(
... TensorDict({"observation": torch.zeros(4)}),
... TensorDict(
... {
... "action": torch.zeros(2),
... "policy_version": torch.zeros((), dtype=torch.long),
... }
... ),
... num_slots=4,
... )
>>> client = transport.client()
>>> policy = TensorDictModule(
... torch.nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
... )
>>> with InferenceServer(policy, transport, max_batch_size=4):
... result = client(TensorDict({"observation": torch.randn(4)}))
>>> result["action"].shape
torch.Size([2])
```

client() → _ProcessSlotClient[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.client)

Create a client bound to the next unused slot.

drain(*max_items: int*) → tuple[list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], list[int]][[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.drain)

Sweep ready slots in round-robin order.

drain_slots(*max_items: int*) → tuple[list[int], list[float]][[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.drain_slots)

Claim ready slots in round-robin order without copying their payloads.

The requests stay in the slot bank until `gather_requests()`
collates them.

Parameters:

**max_items** (*int*) - maximum number of slots to claim.

Returns:

The claimed slot indices and their submission timestamps.

drain_with_timing(*max_items: int*) → tuple[list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], list[int], list[float | None]][[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.drain_with_timing)

Sweep ready slots and return request submission timestamps.

gather_requests(*slots: list[int]*, *out: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.gather_requests)

Collate request slots into `out[:len(slots)]` with one gather per leaf.

Parameters:

- **slots** (*list**of**int*) - slots to collate, typically the ones returned
by `drain_slots()`; row `i` of `out` receives
`slots[i]`.
- **out** (*TensorDictBase*) - batch allocated with `request_batch()`
(possibly pinned) holding at least `len(slots)` rows.

request_batch(*capacity: int*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.request_batch)

Allocate a private, contiguous CPU batch of `capacity` requests.

The batch has the request slot layout (including the interaction-type
key) and is the staging area that `gather_requests()` fills.

Parameters:

**capacity** (*int*) - number of rows.

resolve(*callback: int*, *result: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.resolve)

Copy a response into its slot and wake the owning worker.

resolve_batch(*slots: list[int]*, *results: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.resolve_batch)

Write a batch of responses into their slots and wake the owning workers.

Parameters:

- **slots** (*list**of**int*) - slots served by the pass; row `i` of
`results` is written to `slots[i]`.
- **results** (*TensorDictBase*) - batch of `len(slots)` responses whose
leaves match the response layout (shapes and dtypes). Undeclared
keys are dropped and a missing declared key raises a
`KeyError`.

resolve_exception(*callback: int*, *exc: BaseException*) → None[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.resolve_exception)

Send a model exception to the owning worker and wake it.

response_batch(*capacity: int*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.response_batch)

Allocate a private, contiguous CPU batch of `capacity` responses.

The batch has the response slot layout and is the staging area that
`resolve_batch()` scatters into the slots.

Parameters:

**capacity** (*int*) - number of rows.

submit(*td: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*)[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.submit)

Reject unbound submissions; callers must first obtain a client.

wait_for_work(*timeout: float*) → None[[source]](../../_modules/torchrl/modules/inference_server/_process_slot.html#ProcessSlotTransport.wait_for_work)

Wait until an environment worker marks a request slot ready.