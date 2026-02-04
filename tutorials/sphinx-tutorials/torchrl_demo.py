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
# Let's start with a quick example to see what TorchRL can do:

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

###############################################################################
# Quick Start
# -----------
#
# Here's a complete RL rollout in just a few lines:

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
# That's it! We created an environment, a Q-value actor, and collected a
# trajectory. Now let's dive into the components.
#
# TensorDict: The Data Backbone
# -----------------------------
#
# :class:`~tensordict.TensorDict` is the foundation of TorchRL. It's a
# dictionary-like container for tensors that supports batching, indexing,
# and device transfer.

from tensordict import TensorDict

# Create a TensorDict with keyword arguments
batch_size = 4
data = TensorDict(
    obs=torch.randn(batch_size, 3),
    action=torch.randn(batch_size, 2),
    reward=torch.randn(batch_size, 1),
    batch_size=[batch_size],
)
print(data)

###############################################################################
# TensorDicts support familiar operations:

# Indexing
print("First element:", data[0])
print("Slice:", data[:2])

# Device transfer
data_cpu = data.to("cpu")

# Stacking trajectories
data2 = data.clone()
stacked = torch.stack([data, data2], dim=0)
print("Stacked shape:", stacked.batch_size)

###############################################################################
# Nested TensorDicts are useful for organizing observations:

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
# TorchRL provides wrappers for popular RL environments. All environments
# return TensorDicts.
#
# **Creating Environments**

from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")
print("Action spec:", env.action_spec)
print("Observation spec:", env.observation_spec)

# Reset and step
td = env.reset()
print("Reset output:", td)

td["action"] = env.action_spec.rand()
td = env.step(td)
print("Step output:", td)

###############################################################################
# **Transforms**
#
# Transforms modify environment inputs/outputs, similar to torchvision transforms:

from torchrl.envs import Compose, StepCounter, TransformedEnv

env = TransformedEnv(
    GymEnv("Pendulum-v1"),
    Compose(
        StepCounter(max_steps=200),  # Add step count, auto-terminate
    ),
)
print("Transformed env:", env)

###############################################################################
# **Batched Environments**
#
# Run multiple environments in parallel for faster data collection:
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
# Modules and Policies
# --------------------
#
# TorchRL provides neural network modules that work with TensorDicts.
#
# **TensorDictModule**
#
# Wrap any ``nn.Module`` to read/write TensorDict keys:

from tensordict.nn import TensorDictModule
from torch import nn

module = nn.Linear(3, 2)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["action"])

td = TensorDict(observation=torch.randn(4, 3), batch_size=[4])
td_module(td)
print(td)  # Now has "action" key

###############################################################################
# **Built-in Networks**
#
# TorchRL includes common architectures:

from torchrl.modules import ConvNet, MLP

# MLP for vector observations
mlp = MLP(in_features=64, out_features=10, num_cells=[128, 128])
print(mlp(torch.randn(4, 64)).shape)

# ConvNet for image observations
cnn = ConvNet(num_cells=[32, 64], kernel_sizes=[8, 4], strides=[4, 2])
print(cnn(torch.randn(4, 3, 84, 84)).shape)

###############################################################################
# **Probabilistic Policies**
#
# For stochastic policies (e.g., PPO, SAC):

from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)
from torchrl.modules import NormalParamExtractor, TanhNormal

# Network outputs mean and std
net = nn.Sequential(
    nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 4), NormalParamExtractor()
)
backbone = TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"])

# Sample from distribution
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
print("Log prob:", td["sample_log_prob"].shape)

###############################################################################
# Data Collection
# ---------------
#
# Collectors gather experience from environments efficiently.

from torchrl.collectors import SyncDataCollector

# Create a simple policy
actor = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])

# Collect data
collector = SyncDataCollector(
    create_env_fn=lambda: GymEnv("Pendulum-v1"),
    policy=actor,
    frames_per_batch=200,
    total_frames=1000,
)

for batch in collector:
    print(
        f"Collected batch: {batch.shape}, reward: {batch['next', 'reward'].mean():.2f}"
    )

collector.shutdown()

###############################################################################
# Replay Buffers
# --------------
#
# Store and sample experience for training:

from torchrl.data import LazyTensorStorage, ReplayBuffer

buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000))

# Add experience
buffer.extend(
    TensorDict(obs=torch.randn(100, 4), action=torch.randn(100, 2), batch_size=[100])
)

# Sample a batch
sample = buffer.sample(32)
print("Sampled batch:", sample.batch_size)

###############################################################################
# **Prioritized Replay**

from torchrl.data import PrioritizedReplayBuffer

buffer = PrioritizedReplayBuffer(
    alpha=0.6,
    beta=0.4,
    storage=LazyTensorStorage(max_size=10000),
)
buffer.extend(TensorDict(obs=torch.randn(100, 4), batch_size=[100]))
sample = buffer.sample(32)
print("Prioritized sample with indices:", sample["index"])

###############################################################################
# Loss Functions
# --------------
#
# TorchRL provides loss functions for common RL algorithms:
#
# - :class:`~torchrl.objectives.DQNLoss` - Deep Q-Networks
# - :class:`~torchrl.objectives.DDPGLoss` - Deep Deterministic Policy Gradient
# - :class:`~torchrl.objectives.SACLoss` - Soft Actor-Critic
# - :class:`~torchrl.objectives.PPOLoss` - Proximal Policy Optimization
# - :class:`~torchrl.objectives.TD3Loss` - Twin Delayed DDPG
#
# Here's a simple DQN example:

from torchrl.objectives import DQNLoss

# Create Q-network
qnet = TensorDictModule(
    nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)),
    in_keys=["observation"],
    out_keys=["action_value"],
)

loss_fn = DQNLoss(qnet, action_space=2)

# Compute loss on a batch
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
# A Complete Training Loop
# ------------------------
#
# Here's how all the pieces fit together:

torch.manual_seed(0)

# 1. Environment
env = GymEnv("CartPole-v1")

# 2. Policy (Q-network for DQN)
qnet = TensorDictModule(
    nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2)),
    in_keys=["observation"],
    out_keys=["action_value"],
)
policy = QValueActor(qnet, in_keys=["observation"], spec=env.action_spec)

# 3. Collector
collector = SyncDataCollector(
    create_env_fn=lambda: GymEnv("CartPole-v1"),
    policy=policy,
    frames_per_batch=100,
    total_frames=2000,
)

# 4. Replay Buffer
buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000))

# 5. Loss and Optimizer
loss_fn = DQNLoss(qnet, action_space=env.action_spec)
optimizer = torch.optim.Adam(qnet.parameters(), lr=1e-3)

# Training loop
for i, batch in enumerate(collector):
    buffer.extend(batch)
    if len(buffer) < 100:
        continue

    # Sample and train
    sample = buffer.sample(64)
    loss = loss_fn(sample)
    optimizer.zero_grad()
    loss["loss"].backward()
    optimizer.step()

    if i % 5 == 0:
        print(f"Step {i}: loss={loss['loss'].item():.3f}")

collector.shutdown()
env.close()

###############################################################################
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
