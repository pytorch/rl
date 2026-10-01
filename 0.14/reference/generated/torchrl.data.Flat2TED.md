# Flat2TED

*class*torchrl.data.Flat2TED(*done_key='done'*, *shift_key='shift'*, *is_full_key='is_full'*, *done_keys=('done', 'truncated', 'terminated')*, *reward_keys=('reward',)*)[[source]](../../_modules/torchrl/data/replay_buffers/utils.html#Flat2TED)

A storage loading hook to deserialize flattened TED data to TED format.

Parameters:

- **done_key** (*NestedKey**,**optional*) - the key where the done states should be read.
Defaults to `("next", "done")`.
- **shift_key** (*NestedKey**,**optional*) - the key where the shift will be written.
Defaults to "shift".
- **is_full_key** (*NestedKey**,**optional*) - the key where the is_full attribute will be written.
Defaults to "is_full".
- **done_keys** (*Tuple**[**NestedKey**]**,**optional*) - a tuple of nested keys indicating the done entries.
Defaults to [`DEFAULT_DONE_KEYS`](../data_replaybuffers.html#torchrl.data.DEFAULT_DONE_KEYS), i.e.
`("done", "truncated", "terminated")`.
- **reward_keys** (*Tuple**[**NestedKey**]**,**optional*) - a tuple of nested keys indicating the reward entries.
Defaults to ("reward",)

Examples

```
>>> import tempfile
>>>
>>> from tensordict import TensorDict
>>>
>>> from torchrl.collectors import Collector
>>> from torchrl.data import ReplayBuffer, TED2Flat, LazyMemmapStorage, Flat2TED
>>> from torchrl.envs import GymEnv
>>> import torch
>>>
>>> env = GymEnv("CartPole-v1")
>>> env.set_seed(0)
>>> torch.manual_seed(0)
>>> collector = Collector(env, policy=env.rand_step, total_frames=200, frames_per_batch=200)
>>> rb = ReplayBuffer(storage=LazyMemmapStorage(200))
>>> rb.register_save_hook(TED2Flat())
>>> with tempfile.TemporaryDirectory() as tmpdir:
... for i, data in enumerate(collector):
... rb.extend(data)
... rb.dumps(tmpdir)
... # load the data to represent it
... td = TensorDict.load(tmpdir + "/storage/")
...
... rb_load = ReplayBuffer(storage=LazyMemmapStorage(200))
... rb_load.register_load_hook(Flat2TED())
... rb_load.load(tmpdir)
... print("storage after loading", rb_load[:])
... assert (rb[:] == rb_load[:]).all()
storage after loading TensorDict(
 fields={
 action: MemoryMappedTensor(shape=torch.Size([200, 2]), device=cpu, dtype=torch.int64, is_shared=False),
 collector: TensorDict(
 fields={
 traj_ids: MemoryMappedTensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([200]),
 device=cpu,
 is_shared=False),
 done: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: MemoryMappedTensor(shape=torch.Size([200, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([200]),
 device=cpu,
 is_shared=False),
 observation: MemoryMappedTensor(shape=torch.Size([200, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([200]),
 device=cpu,
 is_shared=False)
```