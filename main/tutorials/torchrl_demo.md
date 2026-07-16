Note

Go to the end
to download the full example code.

# Introduction to TorchRL

Get started with reinforcement learning in PyTorch.

TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch.
This tutorial provides a hands-on introduction to its main components.

**Key features:**

- **PyTorch-native**: Seamless integration with PyTorch's ecosystem
- **Modular**: Easily swap components and build custom pipelines
- **Efficient**: Optimized for both research and production
- **Comprehensive**: Environments, modules, losses, collectors, and more

By the end of this tutorial, you'll understand how TorchRL's components
work together to build RL training pipelines. Let's start with a quick
example to see what's possible:

```

```

## Quick Start

Before diving into the details, here's a taste of what TorchRL can do.
In just a few lines, we can create an environment, build a policy, and
collect a trajectory:

```
import torch
from torchrl.envs import GymEnv
from torchrl.modules import MLP, QValueActor

env = GymEnv("CartPole-v1")
actor = QValueActor(
 MLP(
 in_features=env.observation_spec["observation"].shape[-1],
 out_features=2,
 num_cells=[64, 64],
 ),
 in_keys=["observation"],
 spec=env.action_spec,
)
rollout = env.rollout(max_steps=200, policy=actor)
print(
 f"Collected {rollout.shape[0]} steps, total reward: {rollout['next', 'reward'].sum().item():.0f}"
)
```

```
Collected 10 steps, total reward: 10
```

That's it! We wrapped a Gym environment, created a Q-value actor with an
MLP backbone, and used [`rollout()`](../reference/generated/torchrl.envs.EnvBase.html#id2) to collect
a full trajectory. The result is a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
containing observations, actions, rewards, and more.

Now let's understand each component in detail.

## TensorDict: The Data Backbone

At the heart of TorchRL is [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) - a dictionary-like
container that holds tensors and supports batched operations. Think of it as
a "tensor of dictionaries" or a "dictionary of tensors" that knows about its
batch dimensions.

Why TensorDict? In RL, we constantly pass around groups of related tensors:
observations, actions, rewards, done flags, next observations, etc. TensorDict
keeps these organized and lets us manipulate them as a unit.

```
from tensordict import TensorDict

# Create a TensorDict representing a batch of 4 transitions
batch_size = 4
data = TensorDict(
 obs=torch.randn(batch_size, 3),
 action=torch.randn(batch_size, 2),
 reward=torch.randn(batch_size, 1),
 batch_size=[batch_size],
)
print(data)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 obs: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4]),
 device=None,
 is_shared=False)
```

TensorDicts support all the operations you'd expect from PyTorch tensors.
You can index them, slice them, move them between devices, and stack them
together - all while keeping the dictionary structure intact:

```
# Indexing works just like tensors - grab the first transition
print("First element:", data[0])
print("Slice:", data[:2])

# Device transfer moves all contained tensors
data_cpu = data.to("cpu")

# Stacking is especially useful for building trajectories
data2 = data.clone()
stacked = torch.stack([data, data2], dim=0)
print("Stacked shape:", stacked.batch_size)
```

```
First element: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 obs: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
Slice: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 obs: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)
Stacked shape: torch.Size([2, 4])
```

TensorDicts can also be nested, which is useful for organizing complex
observations (e.g., an agent that receives both image pixels and vector
state) or for separating "current" from "next" step data:

```
nested = TensorDict(
 observation=TensorDict(
 pixels=torch.randn(4, 3, 84, 84),
 vector=torch.randn(4, 10),
 batch_size=[4],
 ),
 action=torch.randn(4, 2),
 batch_size=[4],
)
print(nested)
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: TensorDict(
 fields={
 pixels: Tensor(shape=torch.Size([4, 3, 84, 84]), device=cpu, dtype=torch.float32, is_shared=False),
 vector: Tensor(shape=torch.Size([4, 10]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([4]),
 device=None,
 is_shared=False)
```

## Environments

TorchRL provides a unified interface for RL environments. Whether you're
using Gym, DMControl, IsaacGym, or other simulators, the API stays the same:
environments accept and return TensorDicts.

**Creating Environments**

The simplest way to create an environment is with [`GymEnv`](../reference/generated/torchrl.envs.GymEnv.html#torchrl.envs.GymEnv),
which wraps any Gymnasium (or legacy Gym) environment:

```
from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")
print("Action spec:", env.action_spec)
print("Observation spec:", env.observation_spec)
```

```
Action spec: BoundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous)
Observation spec: Composite(
 observation: BoundedContinuous(
 shape=torch.Size([3]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous),
 device=None,
 shape=torch.Size([]),
 data_cls=None)
```

Every environment has *specs* that describe the shape and bounds of
observations, actions, rewards, and done flags. These specs are essential
for building correctly-shaped networks and for validating data.

The environment interaction follows a familiar pattern - reset, then step:

```
td = env.reset()
print("Reset output:", td)

# Sample a random action and take a step
td["action"] = env.action_spec.rand()
td = env.step(td)
print("Step output:", td)
```

```
Reset output: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
Step output: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Notice that [`step()`](../reference/generated/torchrl.envs.EnvBase.html#id4) returns the same TensorDict
with additional keys filled in: the `"next"` sub-TensorDict contains the
resulting observation, reward, and done flag.

**Transforms**

Just like torchvision transforms for images, TorchRL provides transforms
for environments. These modify observations, actions, or rewards in a
composable way. Common uses include normalizing observations, stacking
frames, or adding step counters:

```
from torchrl.envs import Compose, StepCounter, TransformedEnv

env = TransformedEnv(
 GymEnv("Pendulum-v1"),
 Compose(
 StepCounter(max_steps=200), # Track steps and auto-terminate
 ),
)
print("Transformed env:", env)
```

```
Transformed env: TransformedEnv(
 env=GymEnv(env=Pendulum-v1, batch_size=torch.Size([]), device=None),
 transform=Compose(
 StepCounter(keys=[])))
```

**Batched Environments**

RL algorithms are data-hungry. Running multiple environment instances in
parallel can dramatically speed up data collection. TorchRL's
[`ParallelEnv`](../reference/generated/torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) runs environments in separate processes,
returning batched TensorDicts:

Note

`ParallelEnv` uses multiprocessing. The `mp_start_method` parameter
controls how processes are spawned: `"fork"` (Linux default) is fast but
can have issues with some libraries; `"spawn"` (Windows/macOS default)
is safer but requires code to be guarded with `if __name__ == "__main__"`.

```
from torchrl.envs import ParallelEnv

def make_env():
 return GymEnv("Pendulum-v1")

# Run 4 environments in parallel
vec_env = ParallelEnv(4, make_env)
td = vec_env.reset()
print("Batched reset:", td.batch_size)

td["action"] = vec_env.action_spec.rand()
td = vec_env.step(td)
print("Batched step:", td.batch_size)

vec_env.close()
```

```
Batched reset: torch.Size([4])
Batched step: torch.Size([4])
```

The batch dimension (4 in this case) propagates through all tensors,
making it easy to process multiple environments with a single forward pass.

## Modules and Policies

TorchRL extends PyTorch's `nn.Module` system with modules that read from
and write to TensorDicts. This makes it easy to build policies that
integrate seamlessly with the environment interface.

**TensorDictModule**

The core building block is [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule). It wraps
any `nn.Module` and specifies which TensorDict keys to read as inputs and
which keys to write as outputs:

```
from tensordict.nn import TensorDictModule
from torch import nn

module = nn.Linear(3, 2)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["action"])

# The module reads "observation" and writes "action"
td = TensorDict(observation=torch.randn(4, 3), batch_size=[4])
td_module(td)
print(td) # Now has "action" key
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4]),
 device=None,
 is_shared=False)
```

This pattern has a powerful benefit: modules become composable. You can
chain them together, and each module only needs to know about its own
input/output keys.

**Built-in Networks**

TorchRL includes common network architectures used in RL. These are
regular PyTorch modules that you can wrap with TensorDictModule:

```
from torchrl.modules import ConvNet, MLP

# MLP for vector observations - specify input/output dims and hidden layers
mlp = MLP(in_features=64, out_features=10, num_cells=[128, 128])
print(mlp(torch.randn(4, 64)).shape)

# ConvNet for image observations - outputs a flat feature vector
cnn = ConvNet(num_cells=[32, 64], kernel_sizes=[8, 4], strides=[4, 2])
print(cnn(torch.randn(4, 3, 84, 84)).shape)
```

```
torch.Size([4, 10])
torch.Size([4, 5184])
```

**Probabilistic Policies**

Many RL algorithms (PPO, SAC, etc.) use stochastic policies that output
probability distributions over actions. TorchRL provides
[`ProbabilisticTensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) to sample from
distributions and optionally compute log-probabilities:

```
from tensordict.nn import (
 ProbabilisticTensorDictModule,
 ProbabilisticTensorDictSequential,
)
from torchrl.modules import NormalParamExtractor, TanhNormal

# The network outputs mean and std (via NormalParamExtractor)
net = nn.Sequential(
 nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 4), NormalParamExtractor()
)
backbone = TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"])

# Combine backbone with a distribution sampler
policy = ProbabilisticTensorDictSequential(
 backbone,
 ProbabilisticTensorDictModule(
 in_keys=["loc", "scale"],
 out_keys=["action"],
 distribution_class=TanhNormal,
 return_log_prob=True,
 ),
)

td = TensorDict(observation=torch.randn(4, 3), batch_size=[4])
policy(td)
print("Sampled action:", td["action"].shape)
print("Log prob:", td["action_log_prob"].shape)
```

```
Sampled action: torch.Size([4, 2])
Log prob: torch.Size([4])
```

The `TanhNormal` distribution squashes samples to [-1, 1], which is useful
for continuous control. The log-probability accounts for this transformation,
which is crucial for policy gradient methods.

## Data Collection

In RL, we need to repeatedly collect experience from the environment.
While you can write your own rollout loop, TorchRL's *collectors* handle
this efficiently, including batching, device management, and multi-process
collection.

The [`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) collects data
synchronously - it waits for a batch to be ready before returning:

```
from torchrl.collectors import Collector

# A simple deterministic policy for demonstration
actor = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])

collector = Collector(
 create_env_fn=lambda: GymEnv("Pendulum-v1"),
 policy=actor,
 frames_per_batch=200, # Collect 200 frames per iteration
 total_frames=1000, # Stop after 1000 total frames
)

for batch in collector:
 print(
 f"Collected batch: {batch.shape}, reward: {batch['next', 'reward'].mean():.2f}"
 )

collector.shutdown()
```

```
Collected batch: torch.Size([200]), reward: -8.52
Collected batch: torch.Size([200]), reward: -8.37
Collected batch: torch.Size([200]), reward: -8.70
Collected batch: torch.Size([200]), reward: -9.28
Collected batch: torch.Size([200]), reward: -8.86
```

For async collection (useful when training takes longer than collecting),
see [`MultiAsyncCollector`](../reference/generated/torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector).

## Replay Buffers

Most RL algorithms don't learn from experience immediately - they store
transitions in a buffer and sample mini-batches for training. TorchRL's
replay buffers handle this efficiently:

```
from torchrl.data import LazyTensorStorage, ReplayBuffer

buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000))

# Add a batch of experience
buffer.extend(
 TensorDict(obs=torch.randn(100, 4), action=torch.randn(100, 2), batch_size=[100])
)

# Sample a mini-batch for training
sample = buffer.sample(32)
print("Sampled batch:", sample.batch_size)
```

```
Sampled batch: torch.Size([32])
```

The [`LazyTensorStorage`](../reference/generated/torchrl.data.replay_buffers.LazyTensorStorage.html#torchrl.data.replay_buffers.LazyTensorStorage) allocates memory lazily based
on the first batch added. For prioritized experience replay (used in DQN
variants), use [`PrioritizedReplayBuffer`](../reference/generated/torchrl.data.PrioritizedReplayBuffer.html#torchrl.data.PrioritizedReplayBuffer):

```
from torchrl.data import PrioritizedReplayBuffer

buffer = PrioritizedReplayBuffer(
 alpha=0.6, # Priority exponent
 beta=0.4, # Importance sampling exponent
 storage=LazyTensorStorage(max_size=10000),
)
buffer.extend(TensorDict(obs=torch.randn(100, 4), batch_size=[100]))

# Use return_info=True to get sampling metadata (indices, weights)
sample, info = buffer.sample(32, return_info=True)
print("Prioritized sample indices:", info["index"][:5], "...") # First 5 indices
```

```
Prioritized sample indices: tensor([36, 66, 0, 19, 34]) ...
```

## Loss Functions

The final piece is the objective function. TorchRL provides loss classes
for major RL algorithms, encapsulating the often-complex loss computations:

- [`DQNLoss`](../reference/generated/torchrl.objectives.DQNLoss.html#torchrl.objectives.DQNLoss) - Deep Q-Networks
- [`DDPGLoss`](../reference/generated/torchrl.objectives.DDPGLoss.html#torchrl.objectives.DDPGLoss) - Deep Deterministic Policy Gradient
- [`SACLoss`](../reference/generated/torchrl.objectives.SACLoss.html#torchrl.objectives.SACLoss) - Soft Actor-Critic
- [`PPOLoss`](../reference/generated/torchrl.objectives.PPOLoss.html#torchrl.objectives.PPOLoss) - Proximal Policy Optimization
- [`TD3Loss`](../reference/generated/torchrl.objectives.TD3Loss.html#torchrl.objectives.TD3Loss) - Twin Delayed DDPG

Here's how to set up a DQN loss. We create a Q-network wrapped in a
[`QValueActor`](../reference/generated/torchrl.modules.QValueActor.html#torchrl.modules.QValueActor), which handles action selection:

```
from torchrl.objectives import DQNLoss

qnet = TensorDictModule(
 nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)),
 in_keys=["observation"],
 out_keys=["action_value"],
)

# QValueActor wraps the Q-network to select actions and output chosen values
from torchrl.data import Categorical

actor = QValueActor(qnet, in_keys=["observation"], spec=Categorical(n=2))
loss_fn = DQNLoss(actor, action_space="categorical")
```

The loss function expects batches with specific keys. Let's create a
dummy batch to see it in action:

```
batch = TensorDict(
 observation=torch.randn(32, 4),
 action=torch.randint(0, 2, (32,)),
 next=TensorDict(
 observation=torch.randn(32, 4),
 reward=torch.randn(32, 1),
 done=torch.zeros(32, 1, dtype=torch.bool),
 terminated=torch.zeros(32, 1, dtype=torch.bool),
 batch_size=[32],
 ),
 batch_size=[32],
)

loss_td = loss_fn(batch)
print("Loss:", loss_td["loss"])
```

```
Loss: tensor(0.7207, grad_fn=<MeanBackward0>)
```

The loss function handles target network updates, Bellman backup
computation, and all the bookkeeping needed for stable training.

## Putting It All Together

Now let's see how all these components work together in a complete
training loop. We'll train a simple DQN agent on CartPole:

```
torch.manual_seed(0)

# 1. Create the environment
env = GymEnv("CartPole-v1")

# 2. Build a Q-network and wrap it as a policy
qnet = TensorDictModule(
 nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2)),
 in_keys=["observation"],
 out_keys=["action_value"],
)
policy = QValueActor(qnet, in_keys=["observation"], spec=env.action_spec)

# 3. Set up the data collector
collector = Collector(
 create_env_fn=lambda: GymEnv("CartPole-v1"),
 policy=policy,
 frames_per_batch=100,
 total_frames=2000,
)

# 4. Create a replay buffer
buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000))

# 5. Set up the loss and optimizer (pass the QValueActor, not just the network)
loss_fn = DQNLoss(policy, action_space=env.action_spec)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# 6. Training loop: collect -> store -> sample -> train
for i, batch in enumerate(collector):
 # Store collected experience
 buffer.extend(batch)

 # Wait until we have enough data
 if len(buffer) < 100:
 continue

 # Sample a batch and compute the loss
 sample = buffer.sample(64)
 loss = loss_fn(sample)

 # Standard PyTorch optimization step
 optimizer.zero_grad()
 loss["loss"].backward()
 optimizer.step()

 if i % 5 == 0:
 print(f"Step {i}: loss={loss['loss'].item():.3f}")

collector.shutdown()
env.close()
```

```
Step 0: loss=1.016
Step 5: loss=0.619
Step 10: loss=0.383
Step 15: loss=0.250
```

This is a minimal example - a production DQN would include target network
updates, epsilon-greedy exploration, and more. Check out the full
implementations in `sota-implementations/dqn/`.

## What's Next?

This tutorial covered the basics. TorchRL has much more to offer:

**Tutorials:**

- [PPO Tutorial](../tutorials/coding_ppo.html) - Train PPO on MuJoCo
- [DQN Tutorial](../tutorials/coding_dqn.html) - Deep Q-Learning from scratch
- [Multi-Agent RL](../tutorials/multiagent_ppo.html) - Cooperative and competitive agents

**SOTA Implementations:**

The [sota-implementations/](https://github.com/pytorch/rl/tree/main/sota-implementations)
folder contains production-ready implementations of:

- PPO, A2C, SAC, TD3, DDPG, DQN
- Offline RL: CQL, IQL, Decision Transformer
- Multi-agent: IPPO, QMIX, MADDPG
- LLM training: GRPO, Expert Iteration

**Advanced Features:**

- Distributed training with Ray and RPC
- Offline RL datasets (D4RL, Minari)
- Model-based RL (Dreamer)
- LLM integration for RLHF

**Resources:**

- [API Reference](https://pytorch.org/rl/reference/index.html)
- [GitHub](https://github.com/pytorch/rl)
- [Contributing Guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md)

**Total running time of the script:** (0 minutes 5.073 seconds)

[`Download Jupyter notebook: torchrl_demo.ipynb`](../_downloads/36fe09d5d4546649ee1a029c7144936e/torchrl_demo.ipynb)

[`Download Python source code: torchrl_demo.py`](../_downloads/566627e1cd97def8cf2a3b4720332591/torchrl_demo.py)

[`Download zipped: torchrl_demo.zip`](../_downloads/99d42909723ba57785105ef8a42c1535/torchrl_demo.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)