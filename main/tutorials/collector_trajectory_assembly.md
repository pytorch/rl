Note

Go to the end
to download the full example code.

# Collectors Deep Dive: Trajectory Assembly

**Author**: [Vincent Moens](https://github.com/vmoens)

 What you will learn

- Why collectors return fixed-size batches that mix multiple trajectories
- How `split_trajectories()` reassembles them into padded, per-episode tensors
- What `("collector", "traj_ids")` and `("collector", "mask")` mean
- How `done` and `truncated` interact with trajectory splitting
- When to use `as_nested=True` for memory-efficient ragged batches
- How to request complete trajectories with `trajs_per_batch`
- How to store complete trajectories in a replay buffer

 Prerequisites

- [TorchRL](https://github.com/pytorch/rl) and
[gymnasium](https://gymnasium.farama.org) installed
- Familiarity with [`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector)
(see [the data-collection tutorial](getting-started-3.html#gs-storage-collector))

```
import torch
from torchrl.collectors import Collector
from torchrl.collectors.utils import split_trajectories
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv
from torchrl.modules import RandomPolicy

torch.manual_seed(0)
```

```
<torch._C.Generator object at 0x7efc91ebc230>
```

## Why collectors return fixed-size chunks

In reinforcement learning, episodes can have wildly different lengths.
A CartPole episode may last 10 steps or 500, depending on the policy.
To keep training loops predictable, TorchRL collectors always return
batches of exactly `frames_per_batch` transitions, regardless of how
many episodes those transitions span.

This means a single batch will typically contain **fragments of
multiple trajectories** stitched together. Let's see this in practice.

```
env = GymEnv("CartPole-v1")
env.set_seed(0)

policy = RandomPolicy(env.action_spec)
collector = Collector(env, policy, frames_per_batch=200, total_frames=-1)

for data in collector:
 print(data)
 break
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([200, 2]), device=cpu, dtype=torch.int64, is_shared=False),
 collector: TensorDict(
 fields={
 traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([200]),
 device=None,
 is_shared=False),
 done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([200, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([200]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([200, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([200]),
 device=None,
 is_shared=False)
```

The batch has exactly 200 transitions. Let's inspect its trajectory
IDs -- each integer labels which episode a given transition belongs to:

```
print(data["collector", "traj_ids"])
```

```
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,
 9, 9, 9, 9, 9, 9, 9, 9])
```

Multiple trajectory IDs appear because several short episodes were
packed into a single 200-frame batch. The `("next", "done")` key
marks where each episode ends:

```
print(data["next", "done"].squeeze(-1))
```

```
tensor([False, False, False, False, False, False, False, False, False, False,
 True, False, False, False, False, False, False, False, False, False,
 False, False, True, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, True,
 False, False, False, False, False, False, False, False, False, False,
 False, True, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, True, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 True, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, True, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, True, False, False, False,
 False, False, False, False, False, False, False, False, False, True,
 False, False, False, False, False, False, False, False, False, False])
```

We can count how many complete episodes fell within this batch:

```
n_episodes = data["next", "done"].sum().item()
print(f"This batch of {data.shape[0]} frames contains {n_episodes} episodes.")
```

```
This batch of 200 frames contains 9 episodes.
```

## Reassembling trajectories with `split_trajectories`

For many algorithms (especially those involving recurrent networks or
episode-level returns), you need data organized **per episode**, not
as a flat interleaved stream. [`split_trajectories()`](../reference/generated/torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories)
takes a flat batch with `("collector", "traj_ids")` and returns a
zero-padded `TensorDict` of shape `(num_trajectories, max_length)`.

```
split_data = split_trajectories(data)

print(split_data)
print(f"Shape: {split_data.shape} → (num_trajectories, max_episode_length)")
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([10, 63, 2]), device=cpu, dtype=torch.int64, is_shared=False),
 collector: TensorDict(
 fields={
 mask: Tensor(shape=torch.Size([10, 63]), device=cpu, dtype=torch.bool, is_shared=False),
 traj_ids: Tensor(shape=torch.Size([10, 63]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([10, 63]),
 device=None,
 is_shared=False),
 done: Tensor(shape=torch.Size([10, 63, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([10, 63, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([10, 63, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([10, 63, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 63, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 63, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10, 63]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([10, 63, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 63, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 63, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10, 63]),
 device=None,
 is_shared=False)
Shape: torch.Size([10, 63]) → (num_trajectories, max_episode_length)
```

Because episodes have different lengths, shorter ones are padded with
zeros. The `("collector", "mask")` key tells you which time-steps
contain real data (`True`) and which are padding (`False`):

```
print(split_data["collector", "mask"])
```

```
tensor([[ True, True, True, True, True, True, True, True, True, True,
 True, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, True, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False]])
```

When computing losses on this padded tensor, **always multiply by the
mask** (or index with it) so that padding does not leak into your
gradients. This is especially important for recurrent models.

## `done` vs `truncated` and the mask

TorchRL distinguishes two flavours of episode termination:

- `("next", "done")` is `True` whenever an episode ends, for any
reason.
- `("next", "truncated")` is `True` only when the episode was cut
short by an external limit (a time limit, or the collector running
out of frames before the environment signalled a natural end).

When a trajectory is still in-flight at the edge of a batch, its last
step will be `truncated=True, done=True`. `split_trajectories`
handles this correctly: the mask covers exactly the valid steps,
and the `done` / `truncated` flags are preserved so that you can
treat natural terminations and artificial truncations differently in
your value-function bootstrap.

```
print("done shape: ", split_data["next", "done"].shape)
print("truncated shape:", split_data["next", "truncated"].shape)
```

```
done shape: torch.Size([10, 63, 1])
truncated shape: torch.Size([10, 63, 1])
```

## Padded vs nested tensors

By default `split_trajectories` zero-pads to the length of the
longest trajectory. If your episodes vary a lot in length this wastes
memory. Passing `as_nested=True` returns a
[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) backed by nested tensors instead:

```
padded = split_trajectories(data, as_nested=False)
nested = split_trajectories(data, as_nested=True)

print(f"Padded shape : {padded.shape}")
print(f"Nested result: {type(nested).__name__}, batch_size={nested.batch_size}")
```

```
Padded shape : torch.Size([10, 63])
Nested result: TensorDict, batch_size=torch.Size([10, -1])
```

**Recommendation:** use the default (padded) for simplicity and broad
compatibility. Switch to `as_nested=True` when episode lengths are
highly variable and memory is a concern.

## Getting complete trajectories with `trajs_per_batch`

Sometimes you want the collector itself to hand you **complete
episodes** rather than fixed-frame chunks. The `trajs_per_batch`
argument tells the collector to buffer partial trajectories internally
and yield only once it has accumulated the requested number of
finished episodes.

```
collector_trajs = Collector(
 env,
 policy,
 frames_per_batch=200,
 total_frames=-1,
 trajs_per_batch=5,
 traj_format="padded",
)

for traj_data in collector_trajs:
 print(traj_data)
 break
print(f"Shape: {traj_data.shape} → (trajs_per_batch, max_episode_length)")
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 55, 2]), device=cpu, dtype=torch.int64, is_shared=False),
 collector: TensorDict(
 fields={
 mask: Tensor(shape=torch.Size([5, 55]), device=cpu, dtype=torch.bool, is_shared=False),
 traj_ids: Tensor(shape=torch.Size([5, 55]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([5, 55]),
 device=None,
 is_shared=False),
 done: Tensor(shape=torch.Size([5, 55, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([5, 55, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 55, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([5, 55, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 55, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([5, 55, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5, 55]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([5, 55, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 55, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([5, 55, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5, 55]),
 device=None,
 is_shared=False)
Shape: torch.Size([5, 55]) → (trajs_per_batch, max_episode_length)
```

Every row is a **complete** episode. The mask confirms this -- each
trajectory starts at step 0 and runs until the episode's natural (or
truncated) end:

```
print(traj_data["collector", "mask"])
```

```
tensor([[ True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False],
 [ True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True, True, True, True, True, True,
 True, True, True, True, True],
 [ True, True, True, True, True, True, True, True, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False]])
```

`traj_format` controls the batch layout. `"padded"` (the current
default -- it will change to `"cat"` in torchrl v0.16) stacks the
episodes with zero padding as above. `"cat"` concatenates them along
time instead: the batch is flat and unpadded, episodes are contiguous
and delimited by `("next", "done")` and `("collector", "traj_ids")`.
Prefer it when episode lengths vary widely or frames are large (e.g.
images), since no memory is spent on padding:

```
collector_trajs.shutdown()
collector_cat = Collector(
 env,
 policy,
 frames_per_batch=200,
 total_frames=-1,
 trajs_per_batch=5,
 traj_format="cat",
)

traj_data_cat = next(iter(collector_cat))
print(f"Shape: {traj_data_cat.shape} → (sum of the 5 episode lengths,)")
print(traj_data_cat["next", "done"].squeeze(-1))
```

```
Shape: torch.Size([137]) → (sum of the 5 episode lengths,)
tensor([False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, True, False, False, False, False,
 False, False, False, False, False, False, False, False, False, True,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, True, False, False, False,
 False, False, False, False, False, False, False, False, False, False,
 False, True, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, True])
```

## Storing transitions and sampling trajectory slices

In off-policy training the standard pattern is to store **flat
transitions** in a [`ReplayBuffer`](../reference/generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) and let a
[`SliceSampler`](../reference/generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) carve out contiguous
sub-sequences that respect episode boundaries. The sampler uses
`("next", "done")` to locate where episodes end, so you never get a
slice that straddles two unrelated trajectories.

This is the approach used in the
[Recurrent DQN tutorial](dqn_with_rnn.html#rnn-tuto).

See also

The [replay buffer tutorial](rb_tutorial.html#tuto-rb-traj) covers trajectory
storage in more depth, including alternative samplers such as
[`PrioritizedSliceSampler`](../reference/generated/torchrl.data.replay_buffers.PrioritizedSliceSampler.html#torchrl.data.replay_buffers.PrioritizedSliceSampler) and
[`SliceSamplerWithoutReplacement`](../reference/generated/torchrl.data.replay_buffers.SliceSamplerWithoutReplacement.html#torchrl.data.replay_buffers.SliceSamplerWithoutReplacement).

```
from torchrl.data import SliceSampler

rb = ReplayBuffer(
 storage=LazyTensorStorage(max_size=10_000),
 sampler=SliceSampler(
 slice_len=16,
 end_key=("next", "done"),
 ),
 batch_size=32,
)
```

We extend the buffer with the **flat** collector batch (`data`, shape
`(200,)`), not with the pre-assembled trajectory tensor. The
`SliceSampler` reads the `("next", "done")` flags in this flat
storage to figure out where episodes start and stop.

```
rb.extend(data)

print(f"Buffer length after one batch: {len(rb)}")

sample = rb.sample()
print(sample)
```

```
Buffer length after one batch: 200
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([32, 2]), device=cpu, dtype=torch.int64, is_shared=False),
 collector: TensorDict(
 fields={
 traj_ids: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([32, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([32, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False)
```

With `batch_size=32` and `slice_len=16` the sampler must draw
exactly `32 // 16 = 2` contiguous trajectory slices per call:

```
traj_ids = sample["collector", "traj_ids"]
print(f"Unique trajectories in sample: {traj_ids.unique().numel()}")
```

```
Unique trajectories in sample: 2
```

Each sampled batch contains contiguous slices of 16 steps drawn from
the stored transitions. A typical training loop looks like this:

```
collector = Collector(env, policy, frames_per_batch=200, ...)
rb = ReplayBuffer(
 storage=LazyTensorStorage(max_size=100_000),
 sampler=SliceSampler(slice_len=16, end_key=("next", "done")),
 batch_size=64,
)

for batch in collector:
 rb.extend(batch)
 for _ in range(n_optim):
 sample = rb.sample()
 loss = loss_fn(sample)
 loss.backward()
 optim.step()
```

## Asynchronous collection with `collector.start()`

When a replay buffer is passed directly to the collector, you can
decouple collection from training entirely using
[`start()`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector.start). The collector runs in a
background thread and writes flat transitions into the buffer
continuously while your training loop samples from it.

```
import time

from torchrl.collectors import Collector

rb_async = ReplayBuffer(
 storage=LazyTensorStorage(max_size=10_000),
 sampler=SliceSampler(
 slice_len=16,
 end_key=("next", "done"),
 ),
 shared=True,
)

collector_async = Collector(
 env,
 policy,
 replay_buffer=rb_async,
 frames_per_batch=200,
 total_frames=-1,
)

collector_async.start()
```

The collector is now filling `rb_async` in the background with flat
transitions. The `SliceSampler` will carve contiguous 16-step slices
out of this flat storage, respecting episode boundaries.

```
for _ in range(10):
 time.sleep(0.1)
 if len(rb_async) > 0:
 break

print(f"Buffer length after background collection: {len(rb_async)}")

if len(rb_async) >= 16:
 sample = rb_async.sample(batch_size=32)
 print(sample)
```

```
Buffer length after background collection: 200
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([32, 2]), device=cpu, dtype=torch.int64, is_shared=False),
 collector: TensorDict(
 fields={
 traj_ids: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([32, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([32, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False)
```

When you are done, shut the collector down:

```
collector_async.async_shutdown()
```

This pattern is especially useful when environment stepping is slow
(e.g. physics simulators or LLM inference): the training loop never
idles waiting for new data, and the buffer is always fresh.

## Conclusion

In this tutorial we covered how TorchRL collectors handle trajectories:

- Collectors return **fixed-size batches** that interleave fragments of
multiple episodes.
- [`split_trajectories()`](../reference/generated/torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) reassembles them
into a `(num_trajectories, max_length)` padded tensor with a mask.
- `done` marks any episode end; `truncated` flags artificial
cut-offs. The mask covers valid time-steps only.
- `as_nested=True` gives memory-efficient ragged tensors.
- `trajs_per_batch` makes the collector yield complete episodes
directly.
- Complete episodes slot naturally into a
[`ReplayBuffer`](../reference/generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer).
- Passing a replay buffer and calling
[`start()`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector.start) enables fully
asynchronous collection in a background thread.

### Useful next resources

- [Get started with data collection](getting-started-3.html#gs-storage) -- basic collector
and replay-buffer workflow.
- [Recurrent DQN tutorial](dqn_with_rnn.html#rnn-tuto) -- training a recurrent
policy where per-episode data is essential.
- [TorchRL documentation](https://pytorch.org/rl/)

**Total running time of the script:** (0 minutes 0.375 seconds)

[`Download Jupyter notebook: collector_trajectory_assembly.ipynb`](../_downloads/a2c17acd7f5b44ec53e851e28c3416ac/collector_trajectory_assembly.ipynb)

[`Download Python source code: collector_trajectory_assembly.py`](../_downloads/0063ebb2e8d21875dbc585b77f39b550/collector_trajectory_assembly.py)

[`Download zipped: collector_trajectory_assembly.zip`](../_downloads/b450b188869de2a7935af7eeba0b1071/collector_trajectory_assembly.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)