Note

Go to the end
to download the full example code.

# Using pretrained models

This tutorial explains how to use pretrained models in TorchRL.

```
import tempfile
```

At the end of this tutorial, you will be capable of using pretrained models
for efficient image representation, and fine-tune them.

TorchRL provides pretrained models that are to be used either as transforms or as
components of the policy. As the semantic is the same, they can be used interchangeably
in one or the other context. In this tutorial, we will be using R3M ([https://arxiv.org/abs/2203.12601](https://arxiv.org/abs/2203.12601)),
but other models (e.g. VIP) will work equally well.

```
import torch.cuda
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.envs import Compose, R3MTransform, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import Actor

is_fork = multiprocessing.get_start_method() == "fork"
device = (
 torch.device(0)
 if torch.cuda.is_available() and not is_fork
 else torch.device("cpu")
)
```

Let us first create an environment. For the sake of simplicity, we will be using
a common gym environment. In practice, this will work in more challenging, embodied
AI contexts (e.g. have a look at our Habitat wrappers).

```
base_env = GymEnv("Ant-v4", from_pixels=True, device=device)
```

Let us fetch our pretrained model. We ask for the pretrained version of the model through the
download=True flag. By default this is turned off.
Next, we will append our transform to the environment. In practice, what will happen is that
each batch of data collected will go through the transform and be mapped on a "r3m_vec" entry
in the output tensordict. Our policy, consisting of a single layer MLP, will then read this vector and compute
the corresponding action.

```
r3m = R3MTransform(
 "resnet50",
 in_keys=["pixels"],
 download=False, # Turn to true for real-life testing
)
env_transformed = TransformedEnv(base_env, r3m)
net = nn.Sequential(
 nn.LazyLinear(128, device=device),
 nn.Tanh(),
 nn.Linear(128, base_env.action_spec.shape[-1], device=device),
)
policy = Actor(net, in_keys=["r3m_vec"])
```

Let's check the number of parameters of the policy:

```
print("number of params:", len(list(policy.parameters())))
```

```
number of params: 4
```

We collect a rollout of 32 steps and print its output:

```
rollout = env_transformed.rollout(32, policy)
print("rollout with transform:", rollout)
```

```
rollout with transform: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([32, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 r3m_vec: Tensor(shape=torch.Size([32, 2048]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 r3m_vec: Tensor(shape=torch.Size([32, 2048]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False)
```

For fine tuning, we integrate the transform in the policy after making the parameters
trainable. In practice, it may be wiser to restrict this to a subset of the parameters (say the last layer
of the MLP).

```
r3m.train()
policy = TensorDictSequential(r3m, policy)
print("number of params after r3m is integrated:", len(list(policy.parameters())))
```

```
number of params after r3m is integrated: 163
```

Again, we collect a rollout with R3M. The structure of the output has changed slightly, as now
the environment returns pixels (and not an embedding). The embedding "r3m_vec" is an intermediate
result of our policy.

```
rollout = base_env.rollout(32, policy)
print("rollout, fine tuning:", rollout)
```

```
rollout, fine tuning: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([32, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 pixels: Tensor(shape=torch.Size([32, 480, 480, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 r3m_vec: Tensor(shape=torch.Size([32, 2048]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False)
```

The easiness with which we have swapped the transform from the env to the policy
is due to the fact that both behave like TensorDictModule: they have a set of "in_keys" and
"out_keys" that make it easy to read and write output in different context.

To conclude this tutorial, let's have a look at how we could use R3M to read
images stored in a replay buffer (e.g. in an offline RL context). First, let's build our dataset:

```
from torchrl.data import LazyMemmapStorage, ReplayBuffer

buffer_scratch_dir = tempfile.TemporaryDirectory().name
storage = LazyMemmapStorage(1000, scratch_dir=buffer_scratch_dir)
rb = ReplayBuffer(storage=storage, transform=Compose(lambda td: td.to(device), r3m))
```

We can now collect the data (random rollouts for our purpose) and fill the replay
buffer with it:

```
total = 0
while total < 1000:
 tensordict = base_env.rollout(1000)
 rb.extend(tensordict)
 total += tensordict.numel()
```

Let's check what our replay buffer storage looks like. It should not contain the "r3m_vec" entry
since we haven't used it yet:

```
print("stored data:", storage._storage)
```

```
stored data: TensorDict(
 fields={
 action: MemoryMappedTensor(shape=torch.Size([1000, 8]), device=cpu, dtype=torch.float32, is_shared=True),
 done: MemoryMappedTensor(shape=torch.Size([1000, 1]), device=cpu, dtype=torch.bool, is_shared=True),
 next: TensorDict(
 fields={
 done: MemoryMappedTensor(shape=torch.Size([1000, 1]), device=cpu, dtype=torch.bool, is_shared=True),
 pixels: MemoryMappedTensor(shape=torch.Size([1000, 480, 480, 3]), device=cpu, dtype=torch.uint8, is_shared=True),
 reward: MemoryMappedTensor(shape=torch.Size([1000, 1]), device=cpu, dtype=torch.float32, is_shared=True),
 terminated: MemoryMappedTensor(shape=torch.Size([1000, 1]), device=cpu, dtype=torch.bool, is_shared=True),
 truncated: MemoryMappedTensor(shape=torch.Size([1000, 1]), device=cpu, dtype=torch.bool, is_shared=True)},
 batch_size=torch.Size([1000]),
 device=cpu,
 is_shared=False),
 pixels: MemoryMappedTensor(shape=torch.Size([1000, 480, 480, 3]), device=cpu, dtype=torch.uint8, is_shared=True),
 terminated: MemoryMappedTensor(shape=torch.Size([1000, 1]), device=cpu, dtype=torch.bool, is_shared=True),
 truncated: MemoryMappedTensor(shape=torch.Size([1000, 1]), device=cpu, dtype=torch.bool, is_shared=True)},
 batch_size=torch.Size([1000]),
 device=cpu,
 is_shared=False)
```

When sampling, the data will go through the R3M transform, giving us the processed data that we wanted.
In this way, we can train an algorithm offline on a dataset made of images:

```
batch = rb.sample(32)
print("data after sampling:", batch)
```

```
data after sampling: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([32, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 pixels: Tensor(shape=torch.Size([32, 480, 480, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 r3m_vec: Tensor(shape=torch.Size([32, 2048]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False)
Directory '/tmp/tmpooack9jr' deleted successfully.
```

**Total running time of the script:** (0 minutes 32.186 seconds)

[`Download Jupyter notebook: pretrained_models.ipynb`](../_downloads/3ba23a9a93590a43a3046afd5406df88/pretrained_models.ipynb)

[`Download Python source code: pretrained_models.py`](../_downloads/ae7b3885bfc446e5c7ca2102306be4c2/pretrained_models.py)

[`Download zipped: pretrained_models.zip`](../_downloads/35e7f5fd87ede5f9e265a17f465badd1/pretrained_models.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)