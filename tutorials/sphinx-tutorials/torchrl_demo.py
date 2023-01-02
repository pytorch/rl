# -*- coding: utf-8 -*-
"""
Introduction to TorchRL
============================
This demo was presented at ICML 2022 on the industry demo day.
"""
##############################################################################
# It gives a good overview of TorchRL functionalities. Feel free to reach out
# to vmoens@fb.com or submit issues if you have questions or comments about
# it.
#
# TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch.
#
# https://github.com/pytorch/rl
#
# The PyTorch ecosystem team (Meta) has decided to invest in that library to
# provide a leading platform to develop RL solutions in research settings.
#
# It provides pytorch and **python-first**, low and high level
# **abstractions** # for RL that are intended to be efficient, documented and
# properly tested.
# The code is aimed at supporting research in RL. Most of it is written in
# python in a highly modular way, such that researchers can easily swap
# components, transform them or write new ones with little effort.
#
# This repo attempts to align with the existing pytorch ecosystem libraries
# in that it has a dataset pillar (torchrl/envs), transforms, models, data
# utilities (e.g. collectors and containers), etc. TorchRL aims at having as
# few dependencies as possible (python standard library, numpy and pytorch).
# Common environment libraries (e.g. OpenAI gym) are only optional.
#
# **Content**:
#    .. aafig::
#
#      "torchrl"
#      │
#      ├── "collectors"
#      │   └── "collectors.py"
#      ├── "data"
#      │   ├── "tensor_specs.py"
#      │   ├── "postprocs"
#      │   │  └── "postprocs.py"
#      │   └── "replay_buffers"
#      │      ├── "replay_buffers.py"
#      │      └── "storages.py"
#      ├── "envs"
#      │   ├── "common.py"
#      │   ├── "env_creator.py"
#      │   ├── "gym_like.py"
#      │   ├── "vec_env.py"
#      │   ├── "libs"
#      │   │  ├── "dm_control.py"
#      │   │  └── "gym.py"
#      │   └── "transforms"
#      │      ├── "functional.py"
#      │      └── "transforms.py"
#      ├── "modules"
#      │   ├── "distributions"
#      │   │  ├── "continuous.py"
#      │   │  └── "discrete.py"
#      │   ├── "models"
#      │   │  ├── "models.py"
#      │   │  └── "exploration.py"
#      │   └── "tensordict_module"
#      │      ├── "actors.py"
#      │      ├── "common.py"
#      │      ├── "exploration.py"
#      │      ├── "probabilistic.py"
#      │      └── "sequence.py"
#      ├── "objectives"
#      │   ├── "common.py"
#      │   ├── "ddpg.py"
#      │   ├── "dqn.py"
#      │   ├── "functional.py"
#      │   ├── "ppo.py"
#      │   ├── "redq.py"
#      │   ├── "reinforce.py"
#      │   ├── "sac.py"
#      │   ├── "utils.py"
#      │   └── "value"
#      │      ├── "advantages.py"
#      │      ├── "functional.py"
#      │      ├── "pg.py"
#      │      ├── "utils.py"
#      │      └── "vtrace.py"
#      ├── "record"
#      │   └── "recorder.py"
#      └── "trainers"
#          ├── "loggers"
#          │  ├── "common.py"
#          │  ├── "csv.py"
#          │  ├── "mlflow.py"
#          │  ├── "tensorboard.py"
#          │  └── "wandb.py"
#          ├── "trainers.py"
#          └── "helpers"
#             ├── "collectors.py"
#             ├── "envs.py"
#             ├── "loggers.py"
#             ├── "losses.py"
#             ├── "models.py"
#             ├── "replay_buffer.py"
#             └── "trainers.py"
#
# Unlike other domains, RL is less about media than *algorithms*. As such, it
# is harder to make truly independent components.
#
# What TorchRL is not:
#
# * a collection of algorithms: we do not intend to provide SOTA implementations of RL algorithms,
#   but we provide these algorithms only as examples of how to use the library.
#
# * a research framework: modularity in TorchRL comes in two flavours. First, we try
#   to build re-usable components, such that they can be easily swapped with each other.
#   Second, we make our best such that components can be used independently of the rest
#   of the library.
#
# TorchRL has very few core dependencies, predominantly PyTorch and numpy. All
# other dependencies (gym, torchvision, wandb / tensorboard) are optional.
#
# Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TensorDict
# ------------------------------

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import torch
from tensordict import TensorDict

###############################################################################
# Let's create a TensorDict.

batch_size = 5
tensordict = TensorDict(
    source={
        "key 1": torch.zeros(batch_size, 3),
        "key 2": torch.zeros(batch_size, 5, 6, dtype=torch.bool),
    },
    batch_size=[batch_size],
)
print(tensordict)

###############################################################################
# You can index a TensorDict as well as query keys.

print(tensordict[2])
print(tensordict["key 1"] is tensordict.get("key 1"))

###############################################################################
# The following shows how to stack multiple TensorDicts.

tensordict1 = TensorDict(
    source={
        "key 1": torch.zeros(batch_size, 1),
        "key 2": torch.zeros(batch_size, 5, 6, dtype=torch.bool),
    },
    batch_size=[batch_size],
)

tensordict2 = TensorDict(
    source={
        "key 1": torch.ones(batch_size, 1),
        "key 2": torch.ones(batch_size, 5, 6, dtype=torch.bool),
    },
    batch_size=[batch_size],
)

tensordict = torch.stack([tensordict1, tensordict2], 0)
tensordict.batch_size, tensordict["key 1"]

###############################################################################
# Here are some other functionalities of TensorDict.

print(
    "view(-1): ",
    tensordict.view(-1).batch_size,
    tensordict.view(-1).get("key 1").shape,
)

print("to device: ", tensordict.to("cpu"))

# print("pin_memory: ", tensordict.pin_memory())

print("share memory: ", tensordict.share_memory_())

print(
    "permute(1, 0): ",
    tensordict.permute(1, 0).batch_size,
    tensordict.permute(1, 0).get("key 1").shape,
)

print(
    "expand: ",
    tensordict.expand(3, *tensordict.batch_size).batch_size,
    tensordict.expand(3, *tensordict.batch_size).get("key 1").shape,
)

###############################################################################
# You can create a **nested TensorDict** as well.

tensordict = TensorDict(
    source={
        "key 1": torch.zeros(batch_size, 3),
        "key 2": TensorDict(
            source={"sub-key 1": torch.zeros(batch_size, 2, 1)},
            batch_size=[batch_size, 2],
        ),
    },
    batch_size=[batch_size],
)
tensordict

###############################################################################
# Replay buffers
# ------------------------------

from torchrl.data import PrioritizedReplayBuffer, ReplayBuffer

###############################################################################

rb = ReplayBuffer(100, collate_fn=lambda x: x)
rb.add(1)
rb.sample(1)

###############################################################################

rb.extend([2, 3])
rb.sample(3)

###############################################################################

rb = PrioritizedReplayBuffer(100, alpha=0.7, beta=1.1, collate_fn=lambda x: x)
rb.add(1)
rb.sample(1)
rb.update_priority(1, 0.5)

###############################################################################
# Here are examples of using a replaybuffer with tensordicts.

collate_fn = torch.stack
rb = ReplayBuffer(100, collate_fn=collate_fn)
rb.add(TensorDict({"a": torch.randn(3)}, batch_size=[]))
len(rb)

###############################################################################

rb.extend(TensorDict({"a": torch.randn(2, 3)}, batch_size=[2]))
print(len(rb))
print(rb.sample(10))
print(rb.sample(2).contiguous())

###############################################################################

torch.manual_seed(0)
from torchrl.data import TensorDictPrioritizedReplayBuffer

rb = TensorDictPrioritizedReplayBuffer(
    100, alpha=0.7, beta=1.1, priority_key="td_error"
)
rb.extend(TensorDict({"a": torch.randn(2, 3)}, batch_size=[2]))
tensordict_sample = rb.sample(2).contiguous()
tensordict_sample

###############################################################################

tensordict_sample["index"]

###############################################################################

tensordict_sample["td_error"] = torch.rand(2)
rb.update_priority(tensordict_sample)

for i, val in enumerate(rb._sum_tree):
    print(i, val)
    if i == len(rb):
        break

import gym

###############################################################################
# Envs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from torchrl.envs.libs.gym import GymEnv, GymWrapper

gym_env = gym.make("Pendulum-v1")
env = GymWrapper(gym_env)
env = GymEnv("Pendulum-v1")

###############################################################################

tensordict = env.reset()
env.rand_step(tensordict)

###############################################################################
# Changing environments config
# ------------------------------

env = GymEnv("Pendulum-v1", frame_skip=3, from_pixels=True, pixels_only=False)
env.reset()

###############################################################################

env.close()
del env

###############################################################################

from torchrl.envs import (
    Compose,
    NoopResetEnv,
    ObservationNorm,
    ToTensorImage,
    TransformedEnv,
)

base_env = GymEnv("Pendulum-v1", frame_skip=3, from_pixels=True, pixels_only=False)
env = TransformedEnv(base_env, Compose(NoopResetEnv(3), ToTensorImage()))
env.append_transform(ObservationNorm(in_keys=["pixels"], loc=2, scale=1))

###############################################################################
# Transforms
# ------------------------------

from torchrl.envs import (
    Compose,
    NoopResetEnv,
    ObservationNorm,
    ToTensorImage,
    TransformedEnv,
)

base_env = GymEnv("Pendulum-v1", frame_skip=3, from_pixels=True, pixels_only=False)
env = TransformedEnv(base_env, Compose(NoopResetEnv(3), ToTensorImage()))
env.append_transform(ObservationNorm(in_keys=["pixels"], loc=2, scale=1))

###############################################################################

env.reset()

###############################################################################

print("env: ", env)
print("last transform parent: ", env.transform[2].parent)

###############################################################################
# Vectorized Environments
# ------------------------------

from torchrl.envs import ParallelEnv

base_env = ParallelEnv(
    4,
    lambda: GymEnv("Pendulum-v1", frame_skip=3, from_pixels=True, pixels_only=False),
)
env = TransformedEnv(
    base_env, Compose(NoopResetEnv(3), ToTensorImage())
)  # applies transforms on batch of envs
env.append_transform(ObservationNorm(in_keys=["pixels"], loc=2, scale=1))
env.reset()

###############################################################################

env.action_spec

###############################################################################
# Modules
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Models
# ------------------------------
#
# Example of a MLP model:

from torch import nn

###############################################################################

from torchrl.modules import ConvNet, MLP
from torchrl.modules.models.utils import SquashDims

net = MLP(num_cells=[32, 64], out_features=4, activation_class=nn.ELU)
print(net)
print(net(torch.randn(10, 3)).shape)

###############################################################################
# Example of a CNN model:

cnn = ConvNet(
    num_cells=[32, 64],
    kernel_sizes=[8, 4],
    strides=[2, 1],
    aggregator_class=SquashDims,
)
print(cnn)
print(cnn(torch.randn(10, 3, 32, 32)).shape)  # last tensor is squashed

###############################################################################
# TensorDictModules
# ------------------------------

from tensordict.nn import TensorDictModule

tensordict = TensorDict({"key 1": torch.randn(10, 3)}, batch_size=[10])
module = nn.Linear(3, 4)
td_module = TensorDictModule(module, in_keys=["key 1"], out_keys=["key 2"])
td_module(tensordict)
print(tensordict)

###############################################################################
# Sequences of Modules
# ------------------------------

from tensordict.nn import TensorDictSequential

backbone_module = nn.Linear(5, 3)
backbone = TensorDictModule(
    backbone_module, in_keys=["observation"], out_keys=["hidden"]
)
actor_module = nn.Linear(3, 4)
actor = TensorDictModule(actor_module, in_keys=["hidden"], out_keys=["action"])
value_module = MLP(out_features=1, num_cells=[4, 5])
value = TensorDictModule(value_module, in_keys=["hidden", "action"], out_keys=["value"])

sequence = TensorDictSequential(backbone, actor, value)
print(sequence)

###############################################################################

print(sequence.in_keys, sequence.out_keys)

###############################################################################

tensordict = TensorDict(
    {"observation": torch.randn(3, 5)},
    [3],
)
backbone(tensordict)
actor(tensordict)
value(tensordict)

###############################################################################

tensordict = TensorDict(
    {"observation": torch.randn(3, 5)},
    [3],
)
sequence(tensordict)
print(tensordict)

###############################################################################
# Functional Programming (Ensembling / Meta-RL)
# ----------------------------------------------

from tensordict.nn import make_functional

params = make_functional(sequence)
len(list(sequence.parameters()))  # functional modules have no parameters

###############################################################################

sequence(tensordict, params)

###############################################################################

import functorch

params_expand = params.expand(4)
tensordict_exp = functorch.vmap(sequence, (None, 0))(tensordict, params_expand)
print(tensordict_exp)

###############################################################################
# Specialized Classes
# ------------------------------

torch.manual_seed(0)
from torchrl.data import BoundedTensorSpec
from torchrl.modules import SafeModule

spec = BoundedTensorSpec(-torch.ones(3), torch.ones(3))
base_module = nn.Linear(5, 3)
module = SafeModule(
    module=base_module, spec=spec, in_keys=["obs"], out_keys=["action"], safe=True
)
tensordict = TensorDict({"obs": torch.randn(5)}, batch_size=[])
module(tensordict)["action"]

###############################################################################

tensordict = TensorDict({"obs": torch.randn(5) * 100}, batch_size=[])
module(tensordict)["action"]  # safe=True projects the result within the set

###############################################################################

from torchrl.modules import Actor

base_module = nn.Linear(5, 3)
actor = Actor(base_module, in_keys=["obs"])
tensordict = TensorDict({"obs": torch.randn(5)}, batch_size=[])
actor(tensordict)  # action is the default value

from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)

###############################################################################

# Probabilistic modules
from torchrl.modules import NormalParamWrapper, TanhNormal

td = TensorDict({"input": torch.randn(3, 5)}, [3])
net = NormalParamWrapper(nn.Linear(5, 4))  # splits the output in loc and scale
module = TensorDictModule(net, in_keys=["input"], out_keys=["loc", "scale"])
td_module = ProbabilisticTensorDictSequential(
    module,
    ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=False,
    ),
)
td_module(td)
print(td)

###############################################################################

# returning the log-probability
td = TensorDict({"input": torch.randn(3, 5)}, [3])
td_module = ProbabilisticTensorDictSequential(
    module,
    ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    ),
)
td_module(td)
print(td)

###############################################################################

# Sampling vs mode / mean
from torchrl.envs.utils import set_exploration_mode

td = TensorDict({"input": torch.randn(3, 5)}, [3])

torch.manual_seed(0)
with set_exploration_mode("random"):
    td_module(td)
    print("random:", td["action"])

with set_exploration_mode("mode"):
    td_module(td)
    print("mode:", td["action"])

with set_exploration_mode("mean"):
    td_module(td)
    print("mean:", td["action"])

###############################################################################
# Using Environments and Modules
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from torchrl.envs.utils import step_mdp

env = GymEnv("Pendulum-v1")

action_spec = env.action_spec
actor_module = nn.Linear(3, 1)
actor = SafeModule(
    actor_module, spec=action_spec, in_keys=["observation"], out_keys=["action"]
)

torch.manual_seed(0)
env.set_seed(0)

max_steps = 100
tensordict = env.reset()
tensordicts = TensorDict({}, [max_steps])
for i in range(max_steps):
    actor(tensordict)
    tensordicts[i] = env.step(tensordict)
    tensordict = step_mdp(tensordict)  # roughly equivalent to obs = next_obs
    if env.is_done:
        break

tensordicts_prealloc = tensordicts.clone()
print("total steps:", i)
print(tensordicts)

###############################################################################

# equivalent
torch.manual_seed(0)
env.set_seed(0)

max_steps = 100
tensordict = env.reset()
tensordicts = []
for _ in range(max_steps):
    actor(tensordict)
    tensordicts.append(env.step(tensordict))
    tensordict = step_mdp(tensordict)  # roughly equivalent to obs = next_obs
    if env.is_done:
        break
tensordicts_stack = torch.stack(tensordicts, 0)
print("total steps:", i)
print(tensordicts_stack)

###############################################################################

(tensordicts_stack == tensordicts_prealloc).all()

###############################################################################

# helper
torch.manual_seed(0)
env.set_seed(0)
tensordict_rollout = env.rollout(policy=actor, max_steps=max_steps)
tensordict_rollout

###############################################################################

(tensordict_rollout == tensordicts_prealloc).all()

from tensordict.nn import TensorDictModule

# Collectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from torchrl.collectors import MultiaSyncDataCollector, MultiSyncDataCollector

###############################################################################

from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv

# EnvCreator makes sure that we can send a lambda function from process to process
parallel_env = ParallelEnv(3, EnvCreator(lambda: GymEnv("Pendulum-v1")))
create_env_fn = [parallel_env, parallel_env]

actor_module = nn.Linear(3, 1)
actor = TensorDictModule(actor_module, in_keys=["observation"], out_keys=["action"])

# Sync data collector
devices = ["cpu", "cpu"]

collector = MultiSyncDataCollector(
    create_env_fn=create_env_fn,  # either a list of functions or a ParallelEnv
    policy=actor,
    total_frames=240,
    max_frames_per_traj=-1,  # envs are terminating, we don't need to stop them early
    frames_per_batch=60,  # we want 60 frames at a time (we have 3 envs per sub-collector)
    passing_devices=devices,  # len must match len of env created
    devices=devices,
)

###############################################################################

for i, d in enumerate(collector):
    if i == 0:
        print(d)  # trajectories are split automatically in [6 workers x 10 steps]
    collector.update_policy_weights_()  # make sure that our policies have the latest weights if working on multiple devices
print(i)
collector.shutdown()
del collector

###############################################################################

# async data collector: keeps working while you update your model
collector = MultiaSyncDataCollector(
    create_env_fn=create_env_fn,  # either a list of functions or a ParallelEnv
    policy=actor,
    total_frames=240,
    max_frames_per_traj=-1,  # envs are terminating, we don't need to stop them early
    frames_per_batch=60,  # we want 60 frames at a time (we have 3 envs per sub-collector)
    passing_devices=devices,  # len must match len of env created
    devices=devices,
)

for i, d in enumerate(collector):
    if i == 0:
        print(d)  # trajectories are split automatically in [6 workers x 10 steps]
    collector.update_policy_weights_()  # make sure that our policies have the latest weights if working on multiple devices
print(i)
collector.shutdown()
del collector

###############################################################################
# Objectives
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# TorchRL delivers meta-RL compatible loss functions
# Disclaimer: This APi may change in the future
from torchrl.objectives import DDPGLoss

actor_module = nn.Linear(3, 1)
actor = TensorDictModule(actor_module, in_keys=["observation"], out_keys=["action"])


class ConcatModule(nn.Linear):
    def forward(self, obs, action):
        return super().forward(torch.cat([obs, action], -1))


value_module = ConcatModule(4, 1)
value = TensorDictModule(
    value_module, in_keys=["observation", "action"], out_keys=["state_action_value"]
)

loss_fn = DDPGLoss(actor, value, gamma=0.99)

###############################################################################

tensordict = TensorDict(
    {
        "observation": torch.randn(10, 3),
        "next": {"observation": torch.randn(10, 3)},
        "reward": torch.randn(10, 1),
        "action": torch.randn(10, 1),
        "done": torch.zeros(10, 1, dtype=torch.bool),
    },
    batch_size=[10],
    device="cpu",
)
loss_td = loss_fn(tensordict)

###############################################################################

print(loss_td)

###############################################################################

print(tensordict)

###############################################################################
# State of the Library
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# TorchRL is currently an **alpha-release**: there may be bugs and there is no
# guarantee about BC-breaking changes. We should be able to move to a beta-release
# by the end of the year. Our roadmap to get there comprises:
#
# - Distributed solutions
# - Offline RL
# - Greater support for meta-RL
# - Multi-task and hierarchical RL
#
# Contributing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We are actively looking for contributors and early users. If you're working in
# RL (or just curious), try it! Give us feedback: what will make the success of
# TorchRL is how well it covers researchers needs. To do that, we need their input!
# Since the library is nascent, it is a great time for you to shape it the way you want!

###############################################################################
# Installing the Library
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The library is on PyPI: *pip install torchrl*
