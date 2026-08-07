Note

Go to the end
to download the full example code.

# Get started with data collection and storage

**Author**: [Vincent Moens](https://github.com/vmoens)

Note

To run this tutorial in a notebook, add an installation cell
at the beginning containing:

> ```
> !pip install tensordict
> !pip install torchrl
> ```

```
import tempfile
```

There is no learning without data. In supervised learning, users are
accustomed to using [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) and the like
to integrate data in their training loop.
Dataloaders are iterable objects that provide you with the data that you will
be using to train your model.

TorchRL approaches the problem of dataloading in a similar manner, although
it is surprisingly unique in the ecosystem of RL libraries. TorchRL's
dataloaders are referred to as `DataCollectors`. Most of the time,
data collection does not stop at the collection of raw data,
as the data needs to be stored temporarily in a buffer
(or equivalent structure for on-policy algorithms) before being consumed
by the [loss module](getting-started-2.html#gs-optim). This tutorial will explore
these two classes.

## Data collectors

[`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) is the main construction entry point
for TorchRL data collection. It can build a direct, local multi-process, Ray,
RPC, or distributed collector without changing the class imported by the
training code. At a fundamental level, a collector is a straightforward
class responsible for executing your policy within the environment,
resetting the environment when necessary, and providing batches of a
predefined size. Unlike the [`rollout()`](../reference/generated/torchrl.envs.EnvBase.html#id2) method
demonstrated in [the env tutorial](getting-started-0.html#gs-env-ted), collectors do not
reset between consecutive batches of data. Consequently, two successive
batches of data may contain elements from the same trajectory.

The basic arguments you need to pass to your collector are the size of the
batches you want to collect (`frames_per_batch`), the length (possibly
infinite) of the iterator, the policy and the environment. For simplicity,
we will use a dummy, random policy in this example.

```
import torch

from torchrl.collectors import Collector
from torchrl.envs import GymEnv
from torchrl.modules import RandomPolicy

torch.manual_seed(0)

env = GymEnv("CartPole-v1")
env.set_seed(0)

policy = RandomPolicy(env.action_spec)
collector = Collector(env, policy, frames_per_batch=200, total_frames=-1)
```

We now expect that our collector will deliver batches of size `200` no
matter what happens during collection. In other words, we may have multiple
trajectories in this batch! The `total_frames` indicates how long the
collector should be. A value of `-1` will produce a never
ending collector.

Let's iterate over the collector to get a sense
of what this data looks like:

```
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

As you can see, our data is augmented with some collector-specific metadata
grouped in a `"collector"` sub-tensordict that we did not see during
[environment rollouts](getting-started-0.html#gs-env-ted-rollout). This is useful to keep track of
the trajectory ids. In the following list, each item marks the trajectory
number the corresponding transition belongs to:

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

Changing the execution topology

The default above collects directly in the training process. Scale the same
entry point to local worker processes with `num_collectors`. `sync=True`
waits for every worker and is a natural fit for on-policy algorithms;
`sync=False` delivers the first available worker batch and is usually used
for off-policy training:

```
def make_env():
 return GymEnv("CartPole-v1")

process_collector = Collector(
 make_env,
 policy,
 num_collectors=4,
 sync=False,
 frames_per_batch=200,
 total_frames=-1,
)
```

Distributed placement uses the same constructor. Backend-specific resources
and launcher settings belong in `backend_options`:

```
ray_collector = Collector(
 make_env,
 policy,
 backend="ray",
 num_collectors=4,
 backend_options={
 "remote_configs": {"num_cpus": 1, "num_gpus": 0}
 },
 frames_per_batch=200,
 total_frames=-1,
)
```

The resulting object retains its concrete implementation type, but the
iteration, replay-buffer, lifecycle, and weight-update APIs stay the same.
See ref_collectors for RPC, distributed, submitit, and scoped backend
selection.

> specific technique to solve a problem in a given number of interactions with
> the environment (the `total_frames` argument in the collector).
> For this reason, most training loops in our examples look like this:
> 
> 
> 
> 
> > ```
> > >>> for data in collector:
> > ... # your algorithm here
> > ```
> 
> 
> 
> 
> Now that we have explored how to collect data, we would like to know how to
> store it. In RL, the typical setting is that the data is collected, stored
> temporarily and cleared after a little while given some heuristic:
> first-in first-out or other. A typical pseudo-code would look like this:
> 
> 
> 
> 
> ```
> >>> for data in collector:
> ... storage.store(data)
> ... for i in range(n_optim):
> ... sample = storage.sample()
> ... loss_val = loss_fn(sample)
> ... loss_val.backward()
> ... optim.step() # etc
> ```
> 
> 
> 
> 
> 
> The parent class that stores the data in TorchRL
> is referred to as [`ReplayBuffer`](../reference/generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer). TorchRL's replay
> buffers are composable: you can edit the storage type, their sampling
> technique, the writing heuristic or the transforms applied to them. We will
> leave the fancy stuff for a dedicated in-depth tutorial. The generic replay
> buffer only needs to know what storage it has to use. In general, we
> recommend a [`TensorStorage`](../reference/generated/torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage) subclass, which will work
> fine in most cases. We'll be using
> [`LazyMemmapStorage`](../reference/generated/torchrl.data.replay_buffers.LazyMemmapStorage.html#torchrl.data.replay_buffers.LazyMemmapStorage)
> in this tutorial, which enjoys two nice properties: first, being "lazy",
> you don't need to explicitly tell it what your data looks like in advance.
> Second, it uses [`MemoryMappedTensor`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) as a backend to save
> your data on disk in an efficient way. The only thing you need to know is
> how big you want your buffer to be.

```
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer

buffer_scratch_dir = tempfile.TemporaryDirectory().name

buffer = ReplayBuffer(
 storage=LazyMemmapStorage(max_size=1000, scratch_dir=buffer_scratch_dir)
)
```

Populating the buffer can be done via the
[`add()`](../reference/generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.add) (single element) or
[`extend()`](../reference/generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.extend) (multiple elements) methods. Using
the data we just collected, we initialize and populate the buffer in one go:

```
indices = buffer.extend(data)
```

We can check that the buffer now has the same number of elements as what
we got from the collector:

```
assert len(buffer) == collector.frames_per_batch
```

The only thing left to know is how to gather data from the buffer.
Naturally, this relies on the [`sample()`](../reference/generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample)
method. Because we did not specify that sampling had to be done without
repetitions, it is not guaranteed that the samples gathered from our buffer
will be unique:

```
sample = buffer.sample(batch_size=30)
print(sample)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([30, 2]), device=cpu, dtype=torch.int64, is_shared=False),
 collector: TensorDict(
 fields={
 traj_ids: Tensor(shape=torch.Size([30]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([30]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([30, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([30, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([30, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([30, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([30, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([30, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([30]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([30, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([30, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([30, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([30]),
 device=cpu,
 is_shared=False)
```

Again, our sample looks exactly the same as the data we gathered from the
collector!

## Next steps

- Scale collection with `Collector(num_collectors=N, sync=...)` or select
Ray, RPC, and distributed execution through `Collector(backend=...)`.
See the [API reference](../reference/collectors.html#ref-collectors) for the full selection and
configuration rules.
- Check the dedicated [Replay Buffer tutorial](rb_tutorial.html#rb-tuto) to know
more about the options you have when building a buffer, or the
[API reference](../reference/data.html#ref-data) which covers all the features in
details. Replay buffers have countless features such as multithreaded
sampling, prioritized experience replay, and many more...
- We left out the capacity of replay buffers to be iterated over for
simplicity. Try it out for yourself: build a buffer and indicate its
batch-size in the constructor, then try to iterate over it. This is
equivalent to calling `rb.sample()` within a loop!
- For trajectory-based training (recurrent policies, decision transformers),
see [Complete trajectory collection with trajs_per_batch](../reference/collectors_replay.html#collectors-replay-trajs) -- it shows how to use
`trajs_per_batch` with a [`SliceSampler`](../reference/generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) to store
and sample clean trajectory slices from the replay buffer, especially
with multi-process collectors. The underlying contract -- how episode
boundaries are recovered from the stored data -- is documented in
[Trajectory boundaries](../reference/data_layout.html#ref-traj-boundaries).

**Total running time of the script:** (0 minutes 0.079 seconds)

[`Download Jupyter notebook: getting-started-3.ipynb`](../_downloads/5cb0ffc0980a276546c9aeed94b0aa13/getting-started-3.ipynb)

[`Download Python source code: getting-started-3.py`](../_downloads/999b4bf62144c175c6f8c50bf823fd79/getting-started-3.py)

[`Download zipped: getting-started-3.zip`](../_downloads/fbaa3b91355e869868174ff484447abe/getting-started-3.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)