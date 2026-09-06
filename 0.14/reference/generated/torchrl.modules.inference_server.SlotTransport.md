# SlotTransport

*class*torchrl.modules.inference_server.SlotTransport(*num_slots: int*, ***, *preallocate: bool = False*)[[source]](../../_modules/torchrl/modules/inference_server/_slot.html#SlotTransport)

Lock-free, in-process transport using per-env slots.

Each actor thread owns a dedicated *slot*. Submitting an observation
writes to the slot without any lock (each slot is accessed by exactly
one writer thread). The server sweeps slots to find ready ones, collects
observations, runs the model, and writes actions back via per-slot events.

This eliminates:

- The shared `threading.Lock` that `ThreadingTransport` uses for
every `submit()` and `drain()`.
- `concurrent.futures.Future` allocations (one per inference request).

The trade-off is that the number of slots is fixed at construction time
(equal to the number of environments).

Parameters:

**num_slots** (*int*) - number of slots (one per environment / actor thread).

Keyword Arguments:

**preallocate** (*bool**,**optional*) - if `True`, a contiguous observation
buffer of shape `[num_slots, ...]` is allocated on the first
submit. Subsequent submits copy into the buffer in-place
(`update_`). Defaults to `False` because the extra copy
into the buffer is not currently compensated by the batching
path (`lazy_stack` still calls `torch.stack`).

Note

This transport is only suitable for in-process threading scenarios
(the default for [`AsyncBatchedCollector`](torchrl.collectors.AsyncBatchedCollector.html#torchrl.collectors.AsyncBatchedCollector)
with `policy_backend="threading"`).

client() → _SlotClient[[source]](../../_modules/torchrl/modules/inference_server/_slot.html#SlotTransport.client)

Create a slot-bound client for one actor thread.

drain(*max_items: int*) → tuple[list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], list[int]][[source]](../../_modules/torchrl/modules/inference_server/_slot.html#SlotTransport.drain)

Sweep slots and return (observations, slot_ids) for ready ones.

drain_with_timing(*max_items: int*) → tuple[list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], list[int], list[float | None]][[source]](../../_modules/torchrl/modules/inference_server/_slot.html#SlotTransport.drain_with_timing)

Sweep slots and include actor-side submission timestamps.

resolve(*callback: int*, *result: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/modules/inference_server/_slot.html#SlotTransport.resolve)

Write the action into the slot and wake the waiting env thread.

resolve_exception(*callback: int*, *exc: BaseException*) → None[[source]](../../_modules/torchrl/modules/inference_server/_slot.html#SlotTransport.resolve_exception)

Propagate an exception to the waiting env thread.

submit(*td: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*)[[source]](../../_modules/torchrl/modules/inference_server/_slot.html#SlotTransport.submit)

Not supported - use `client()` to get a slot-bound callable.

wait_for_work(*timeout: float*) → None[[source]](../../_modules/torchrl/modules/inference_server/_slot.html#SlotTransport.wait_for_work)

Block until at least one slot has a ready observation.