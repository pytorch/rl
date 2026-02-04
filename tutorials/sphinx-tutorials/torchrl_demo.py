"""
Introduction to TorchRL
=======================

Get started with reinforcement learning in PyTorch.
"""

###############################################################################
# TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch.
# This tutorial provides a hands-on introduction to its main components.
#
# **Key features:**
#
# - **PyTorch-native**: Seamless integration with PyTorch's ecosystem
# - **Modular**: Easily swap components and build custom pipelines
# - **Efficient**: Optimized for both research and production
# - **Comprehensive**: Environments, modules, losses, collectors, and more
#
# By the end of this tutorial, you'll understand how TorchRL's components
# work together to build RL training pipelines. Let's start with a quick
# example to see what's possible:

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

###############################################################################
# Quick Start
# -----------
#
# Before diving into the details, here's a taste of what TorchRL can do.
# In just a few lines, we can create an environment, build a policy, and
# collect a trajectory:

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

###############################################################################
# That's it! We wrapped a Gym environment, created a Q-value actor with an
# MLP backbone, and used :meth:`~torchrl.envs.EnvBase.rollout` to collect
# a full trajectory. The result is a :class:`~tensordict.TensorDict`
# containing observations, actions, rewards, and more.
#
# Now let's understand each component in detail.
#
# TensorDict: The Data Backbone
# -----------------------------
#
# At the heart of TorchRL is :class:`~tensordict.TensorDict` - a dictionary-like
# container that holds tensors and supports batched operations. Think of it as
# a "tensor of dictionaries" or a "dictionary of tensors" that knows about its
# batch dimensions.
#
# Why TensorDict? In RL, we constantly pass around groups of related tensors:
# observations, actions, rewards, done flags, next observations, etc. TensorDict
# keeps these organized and lets us manipulate them as a unit.

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

###############################################################################
# TensorDicts support all the operations you'd expect from PyTorch tensors.
# You can index them, slice them, move them between devices, and stack them
# together - all while keeping the dictionary structure intact:

# Indexing works just like tensors - grab the first transition
print("First element:", data[0])
print("Slice:", data[:2])

# Device transfer moves all contained tensors
data_cpu = data.to("cpu")

# Stacking is especially useful for building trajectories
data2 = data.clone()
stacked = torch.stack([data, data2], dim=0)
print("Stacked shape:", stacked.batch_size)

###############################################################################
# TensorDicts can also be nested, which is useful for organizing complex
# observations (e.g., an agent that receives both image pixels and vector
# state) or for separating "current" from "next" step data:

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

###############################################################################
# Environments
# ------------
#
# TorchRL provides a unified interface for RL environments. Whether you're
# using Gym, DMControl, IsaacGym, or other simulators, the API stays the same:
# environments accept and return TensorDicts.
#
# **Creating Environments**
#
# The simplest way to create an environment is with :class:`~torchrl.envs.GymEnv`,
# which wraps any Gymnasium (or legacy Gym) environment:

from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")
print("Action spec:", env.action_spec)
print("Observation spec:", env.observation_spec)

###############################################################################
# Every environment has *specs* that describe the shape and bounds of
# observations, actions, rewards, and done flags. These specs are essential
# for building correctly-shaped networks and for validating data.
#
# The environment interaction follows a familiar pattern - reset, then step:

td = env.reset()
print("Reset output:", td)

# Sample a random action and take a step
td["action"] = env.action_spec.rand()
td = env.step(td)
print("Step output:", td)

###############################################################################
# Notice that :meth:`~torchrl.envs.EnvBase.step` returns the same TensorDict
# with additional keys filled in: the ``"next"`` sub-TensorDict contains the
# resulting observation, reward, and done flag.
#
# **Transforms**
#
# Just like torchvision transforms for images, TorchRL provides transforms
# for environments. These modify observations, actions, or rewards in a
# composable way. Common uses include normalizing observations, stacking
# frames, or adding step counters:

from torchrl.envs import Compose, StepCounter, TransformedEnv

env = TransformedEnv(
    GymEnv("Pendulum-v1"),
    Compose(
        StepCounter(max_steps=200),  # Track steps and auto-terminate
    ),
)
print("Transformed env:", env)

###############################################################################
# **Batched Environments**
#
# RL algorithms are data-hungry. Running multiple environment instances in
# parallel can dramatically speed up data collection. TorchRL's
# :class:`~torchrl.envs.ParallelEnv` runs environments in separate processes,
# returning batched TensorDicts:
#
# .. note::
#    By default, ``ParallelEnv`` uses ``fork`` on Linux and ``spawn`` on
#    Windows/macOS. You can override this with ``mp_start_method``.
#    ``spawn`` is safer but requires code to be in ``if __name__ == "__main__"``.

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

###############################################################################
# The batch dimension (4 in this case) propagates through all tensors,
# making it easy to process multiple environments with a single forward pass.
#
# Modules and Policies
# --------------------
#
# TorchRL extends PyTorch's ``nn.Module`` system with modules that read from
# and write to TensorDicts. This makes it easy to build policies that
# integrate seamlessly with the environment interface.
#
# **TensorDictModule**
#
# The core building block is :class:`~tensordict.nn.TensorDictModule`. It wraps
# any ``nn.Module`` and specifies which TensorDict keys to read as inputs and
# which keys to write as outputs:

from tensordict.nn import TensorDictModule
from torch import nn

module = nn.Linear(3, 2)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["action"])

# The module reads "observation" and writes "action"
td = TensorDict(observation=torch.randn(4, 3), batch_size=[4])
td_module(td)
print(td)  # Now has "action" key

###############################################################################
# This pattern has a powerful benefit: modules become composable. You can
# chain them together, and each module only needs to know about its own
# input/output keys.
#
# **Built-in Networks**
#
# TorchRL includes common network architectures used in RL. These are
# regular PyTorch modules that you can wrap with TensorDictModule:

from torchrl.modules import ConvNet, MLP

# MLP for vector observations - specify input/output dims and hidden layers
mlp = MLP(in_features=64, out_features=10, num_cells=[128, 128])
print(mlp(torch.randn(4, 64)).shape)

# ConvNet for image observations - outputs a flat feature vector
cnn = ConvNet(num_cells=[32, 64], kernel_sizes=[8, 4], strides=[4, 2])
print(cnn(torch.randn(4, 3, 84, 84)).shape)

###############################################################################
# **Probabilistic Policies**
#
# Many RL algorithms (PPO, SAC, etc.) use stochastic policies that output
# probability distributions over actions. TorchRL provides
# :class:`~tensordict.nn.ProbabilisticTensorDictModule` to sample from
# distributions and optionally compute log-probabilities:

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

###############################################################################
# The ``TanhNormal`` distribution squashes samples to [-1, 1], which is useful
# for continuous control. The log-probability accounts for this transformation,
# which is crucial for policy gradient methods.
#
# Data Collection
# ---------------
#
# In RL, we need to repeatedly collect experience from the environment.
# While you can write your own rollout loop, TorchRL's *collectors* handle
# this efficiently, including batching, device management, and multi-process
# collection.
#
# The :class:`~torchrl.collectors.SyncDataCollector` collects data
# synchronously - it waits for a batch to be ready before returning:

from torchrl.collectors import SyncDataCollector

# A simple deterministic policy for demonstration
actor = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])

collector = SyncDataCollector(
    create_env_fn=lambda: GymEnv("Pendulum-v1"),
    policy=actor,
    frames_per_batch=200,  # Collect 200 frames per iteration
    total_frames=1000,  # Stop after 1000 total frames
)

for batch in collector:
    print(
        f"Collected batch: {batch.shape}, reward: {batch['next', 'reward'].mean():.2f}"
    )

collector.shutdown()

###############################################################################
# For async collection (useful when training takes longer than collecting),
# see :class:`~torchrl.collectors.MultiaSyncDataCollector`.
#
# Replay Buffers
# --------------
#
# Most RL algorithms don't learn from experience immediately - they store
# transitions in a buffer and sample mini-batches for training. TorchRL's
# replay buffers handle this efficiently:

from torchrl.data import LazyTensorStorage, ReplayBuffer

buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000))

# Add a batch of experience
buffer.extend(
    TensorDict(obs=torch.randn(100, 4), action=torch.randn(100, 2), batch_size=[100])
)

# Sample a mini-batch for training
sample = buffer.sample(32)
print("Sampled batch:", sample.batch_size)

###############################################################################
# The :class:`~torchrl.data.LazyTensorStorage` allocates memory lazily based
# on the first batch added. For prioritized experience replay (used in DQN
# variants), use :class:`~torchrl.data.PrioritizedReplayBuffer`:

from torchrl.data import PrioritizedReplayBuffer

buffer = PrioritizedReplayBuffer(
    alpha=0.6,  # Priority exponent
    beta=0.4,  # Importance sampling exponent
    storage=LazyTensorStorage(max_size=10000),
)
buffer.extend(TensorDict(obs=torch.randn(100, 4), batch_size=[100]))
sample = buffer.sample(32)
print("Prioritized sample with indices:", sample["index"])

###############################################################################
# Loss Functions
# --------------
#
# The final piece is the objective function. TorchRL provides loss classes
# for major RL algorithms, encapsulating the often-complex loss computations:
#
# - :class:`~torchrl.objectives.DQNLoss` - Deep Q-Networks
# - :class:`~torchrl.objectives.DDPGLoss` - Deep Deterministic Policy Gradient
# - :class:`~torchrl.objectives.SACLoss` - Soft Actor-Critic
# - :class:`~torchrl.objectives.PPOLoss` - Proximal Policy Optimization
# - :class:`~torchrl.objectives.TD3Loss` - Twin Delayed DDPG
#
# Here's how to set up a DQN loss. First, we create a Q-network that maps
# observations to action values:

from torchrl.objectives import DQNLoss

qnet = TensorDictModule(
    nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)),
    in_keys=["observation"],
    out_keys=["action_value"],
)

loss_fn = DQNLoss(qnet, action_space=2)

###############################################################################
# The loss function expects batches with specific keys. Let's create a
# dummy batch to see it in action:

batch = TensorDict(
    observation=torch.randn(32, 4),
    action=torch.randint(0, 2, (32, 1)),
    next=TensorDict(
        observation=torch.randn(32, 4),
        reward=torch.randn(32, 1),
        done=torch.zeros(32, 1, dtype=torch.bool),
        batch_size=[32],
    ),
    batch_size=[32],
)

loss_td = loss_fn(batch)
print("Loss:", loss_td["loss"])

###############################################################################
# The loss function handles target network updates, Bellman backup
# computation, and all the bookkeeping needed for stable training.
#
# Putting It All Together
# -----------------------
#
# Now let's see how all these components work together in a complete
# training loop. We'll train a simple DQN agent on CartPole:

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
collector = SyncDataCollector(
    create_env_fn=lambda: GymEnv("CartPole-v1"),
    policy=policy,
    frames_per_batch=100,
    total_frames=2000,
)

# 4. Create a replay buffer
buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000))

# 5. Set up the loss and optimizer
loss_fn = DQNLoss(qnet, action_space=env.action_spec)
optimizer = torch.optim.Adam(qnet.parameters(), lr=1e-3)

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

###############################################################################
# This is a minimal example - a production DQN would include target network
# updates, epsilon-greedy exploration, and more. Check out the full
# implementations in ``sota-implementations/dqn/``.
#
# What's Next?
# ------------
#
# This tutorial covered the basics. TorchRL has much more to offer:
#
# **Tutorials:**
#
# - `PPO Tutorial <../tutorials/coding_ppo.html>`_ - Train PPO on MuJoCo
# - `DQN Tutorial <../tutorials/coding_dqn.html>`_ - Deep Q-Learning from scratch
# - `Multi-Agent RL <../tutorials/multiagent_ppo.html>`_ - Cooperative and competitive agents
#
# **SOTA Implementations:**
#
# The `sota-implementations/ <https://github.com/pytorch/rl/tree/main/sota-implementations>`_
# folder contains production-ready implementations of:
#
# - PPO, A2C, SAC, TD3, DDPG, DQN
# - Offline RL: CQL, IQL, Decision Transformer
# - Multi-agent: IPPO, QMIX, MADDPG
# - LLM training: GRPO, Expert Iteration
#
# **Advanced Features:**
#
# - Distributed training with Ray and RPC
# - Offline RL datasets (D4RL, Minari)
# - Model-based RL (Dreamer)
# - LLM integration for RLHF
#
# **Resources:**
#
# - `API Reference <https://pytorch.org/rl/reference/index.html>`_
# - `GitHub <https://github.com/pytorch/rl>`_
# - `Contributing Guide <https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md>`_
#
