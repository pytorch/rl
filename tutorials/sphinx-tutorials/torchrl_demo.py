"""
Introduction to TorchRL
=======================
A comprehensive overview of TorchRL's core features and capabilities.
"""
##############################################################################
# This tutorial provides a hands-on introduction to TorchRL's main components.
# Feel free to submit issues on `GitHub <https://github.com/pytorch/rl>`__ if
# you have questions or feedback.
#
# TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch,
# developed by Meta as part of the PyTorch ecosystem.
#
# Key features:
#
# - **PyTorch-first**: Native integration with PyTorch's tensor operations and autograd
# - **Modular design**: Easily swap components and build custom RL pipelines
# - **Efficient**: Optimized for both research prototyping and production use
# - **Minimal dependencies**: Core library requires only PyTorch and numpy
#
# TorchRL follows PyTorch ecosystem conventions with familiar abstractions:
# environments (like datasets), transforms, modules, and data utilities.
#
# **Library Structure**:
#
# - ``torchrl.collectors``: Data collection utilities (single/multi-process)
# - ``torchrl.data``: Replay buffers, tensor specs, datasets
# - ``torchrl.envs``: Environment wrappers (Gym, DMControl, etc.) and transforms
# - ``torchrl.modules``: Neural network modules and actors
# - ``torchrl.objectives``: RL loss functions (PPO, SAC, DQN, etc.)
# - ``torchrl.record``: Logging utilities (TensorBoard, WandB, etc.)
#
# See the `API Reference <https://pytorch.org/rl/reference/index.html>`__ for
# complete documentation.
#
# Data
# ----
#
# TensorDict
# ~~~~~~~~~~

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

from torch import multiprocessing

# TorchRL prefers spawn method for safety across platforms
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

# sphinx_gallery_end_ignore

import torch
from tensordict import TensorDict

###############################################################################
# Let's create a TensorDict. The constructor accepts many different formats, like passing a dict
# or with keyword arguments:

batch_size = 5
data = TensorDict(
    key1=torch.zeros(batch_size, 3),
    key2=torch.zeros(batch_size, 5, 6, dtype=torch.bool),
    batch_size=[batch_size],
)
print(data)

###############################################################################
# You can index a TensorDict along its ``batch_size``, as well as query keys.

print(data[2])
print(data["key1"] is data.get("key1"))

###############################################################################
# The following shows how to stack multiple TensorDicts. This is particularly useful when writing rollout loops!

data1 = TensorDict(
    {
        "key1": torch.zeros(batch_size, 1),
        "key2": torch.zeros(batch_size, 5, 6, dtype=torch.bool),
    },
    batch_size=[batch_size],
)

data2 = TensorDict(
    {
        "key1": torch.ones(batch_size, 1),
        "key2": torch.ones(batch_size, 5, 6, dtype=torch.bool),
    },
    batch_size=[batch_size],
)

data = torch.stack([data1, data2], 0)
data.batch_size, data["key1"]

###############################################################################
# Here are some other functionalities of TensorDict: viewing, permute, sharing memory or expanding.

print(
    "view(-1): ",
    data.view(-1).batch_size,
    data.view(-1).get("key1").shape,
)

print("to device: ", data.to("cpu"))

print(
    "permute(1, 0): ",
    data.permute(1, 0).batch_size,
    data.permute(1, 0).get("key1").shape,
)

print(
    "expand: ",
    data.expand(3, *data.batch_size).batch_size,
    data.expand(3, *data.batch_size).get("key1").shape,
)

###############################################################################
# You can create a **nested data** as well.

data = TensorDict(
    source={
        "key1": torch.zeros(batch_size, 3),
        "key2": TensorDict(
            source={"sub_key1": torch.zeros(batch_size, 2, 1)},
            batch_size=[batch_size, 2],
        ),
    },
    batch_size=[batch_size],
)
data

###############################################################################
# Replay buffers
# --------------
#
# :ref:`Replay buffers <ref_buffers>` are a crucial component in many RL algorithms. TorchRL provides a range of replay buffer implementations.
# Most basic features will work with any data scturcture (list, tuples, dict) but to use the replay buffers to their
# full extend and with fast read and write access, TensorDict APIs should be preferred.

from torchrl.data import PrioritizedReplayBuffer, ReplayBuffer

rb = ReplayBuffer(collate_fn=lambda x: x)

###############################################################################
# Adding can be done with :meth:`~torchrl.data.ReplayBuffer.add` (n=1)
# or :meth:`~torchrl.data.ReplayBuffer.extend` (n>1).
rb.add(1)
rb.sample(1)
rb.extend([2, 3])
rb.sample(3)

###############################################################################
# Prioritized Replay Buffers can also be used:
#

rb = PrioritizedReplayBuffer(alpha=0.7, beta=1.1, collate_fn=lambda x: x)
rb.add(1)
rb.sample(1)
rb.update_priority(1, 0.5)

###############################################################################
# Here are examples of using a replaybuffer with data_stack.
# Using them makes it easy to abstract away the behaviour of the replay buffer for multiple use cases.

collate_fn = torch.stack
rb = ReplayBuffer(collate_fn=collate_fn)
rb.add(TensorDict({"a": torch.randn(3)}, batch_size=[]))
len(rb)

rb.extend(TensorDict({"a": torch.randn(2, 3)}, batch_size=[2]))
print(len(rb))
print(rb.sample(10))
print(rb.sample(2).contiguous())

torch.manual_seed(0)
from torchrl.data import TensorDictPrioritizedReplayBuffer

rb = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=1.1, priority_key="td_error")
rb.extend(TensorDict({"a": torch.randn(2, 3)}, batch_size=[2]))
data_sample = rb.sample(2).contiguous()
print(data_sample)

print(data_sample["index"])

data_sample["td_error"] = torch.rand(2)
rb.update_tensordict_priority(data_sample)

for i, val in enumerate(rb._sampler._sum_tree):
    print(i, val)
    if i == len(rb):
        break

###############################################################################
# Envs
# ----
# TorchRL provides a range of :ref:`environment <Environment-API>` wrappers and utilities.
#
# Gym Environment
# ~~~~~~~~~~~~~~~

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

from torchrl.envs.libs.gym import GymEnv, GymWrapper, set_gym_backend

gym_env = gym.make("Pendulum-v1")
env = GymWrapper(gym_env)
env = GymEnv("Pendulum-v1")

data = env.reset()
env.rand_step(data)

###############################################################################
# Changing environments config
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

env = GymEnv("Pendulum-v1", frame_skip=3, from_pixels=True, pixels_only=False)
env.reset()

env.close()
del env

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
# Environment Transforms
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Transforms act like Gym wrappers but with an API closer to torchvision's ``torch.distributions``' transforms.
# There is a wide range of :ref:`transforms <transforms>` to choose from.

from torchrl.envs import (
    Compose,
    NoopResetEnv,
    ObservationNorm,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)

base_env = GymEnv("HalfCheetah-v4", frame_skip=3, from_pixels=True, pixels_only=False)
env = TransformedEnv(base_env, Compose(NoopResetEnv(3), ToTensorImage()))
env = env.append_transform(ObservationNorm(in_keys=["pixels"], loc=2, scale=1))

env.reset()

print("env: ", env)
print("last transform parent: ", env.transform[2].parent)

###############################################################################
# Vectorized Environments
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Vectorized / parallel environments can provide some significant speed-ups.
#

from torchrl.envs import ParallelEnv


def make_env():
    # You can control whether to use gym or gymnasium for your env
    with set_gym_backend("gym"):
        return GymEnv("Pendulum-v1", frame_skip=3, from_pixels=True, pixels_only=False)


base_env = ParallelEnv(
    4,
    make_env,
    mp_start_method="spawn",
)
env = TransformedEnv(
    base_env, Compose(StepCounter(), ToTensorImage())
)  # applies transforms on batch of envs
env.append_transform(ObservationNorm(in_keys=["pixels"], loc=2, scale=1))
env.reset()

print(env.action_spec)

env.close()
del env

###############################################################################
# Modules
# -------
#
# Multiple :ref:`modules <ref_modules>`  (utils, models and wrappers) can be found in the library.
#
# Models
# ~~~~~~
#
# Example of a MLP model:

from torch import nn
from torchrl.modules import ConvNet, MLP
from torchrl.modules.models.utils import SquashDims

net = MLP(num_cells=[32, 64], out_features=4, activation_class=nn.ELU)
print(net)
print(net(torch.randn(10, 3)).shape)

###############################################################################
# Example of a CNN model:
#

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
# ~~~~~~~~~~~~~~~~~
#
# :ref:`Some modules <tdmodules>` are specifically designed to work with tensordict inputs.
#

from tensordict.nn import TensorDictModule

data = TensorDict({"key1": torch.randn(10, 3)}, batch_size=[10])
module = nn.Linear(3, 4)
td_module = TensorDictModule(module, in_keys=["key1"], out_keys=["key2"])
td_module(data)
print(data)

###############################################################################
# Sequences of Modules
# ~~~~~~~~~~~~~~~~~~~~
#
# Making sequences of modules is made easy by :class:`~tensordict.nn.TensorDictSequential`:
#

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

print(sequence.in_keys, sequence.out_keys)

data = TensorDict(
    {"observation": torch.randn(3, 5)},
    [3],
)
backbone(data)
actor(data)
value(data)

data = TensorDict(
    {"observation": torch.randn(3, 5)},
    [3],
)
sequence(data)
print(data)

###############################################################################
# Functional Programming (Ensembling / Meta-RL)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Functional calls have never been easier. Extract the parameters with :func:`~tensordict.from_module`, and
# replace them with :meth:`~tensordict.TensorDict.to_module`:

from tensordict import from_module

params = from_module(sequence)
print("extracted params", params)

###############################################################################
# functional call using tensordict:

with params.to_module(sequence):
    data = sequence(data)

###############################################################################
# VMAP
# ~~~~
#
# Fast execution of multiple copies of a similar architecture is key to train your models fast.
# :func:`~torch.vmap` is tailored to do just that:
#

from torch import vmap

params_expand = params.expand(4)


def exec_sequence(params, data):
    with params.to_module(sequence):
        return sequence(data)


tensordict_exp = vmap(exec_sequence, (0, None))(params_expand, data)
print(tensordict_exp)

###############################################################################
# Specialized Classes
# ~~~~~~~~~~~~~~~~~~~
#
# TorchRL provides also some specialized modules that run checks on the output values.

torch.manual_seed(0)
from torchrl.data import Bounded
from torchrl.modules import SafeModule

spec = Bounded(-torch.ones(3), torch.ones(3))
base_module = nn.Linear(5, 3)
module = SafeModule(
    module=base_module, spec=spec, in_keys=["obs"], out_keys=["action"], safe=True
)
data = TensorDict({"obs": torch.randn(5)}, batch_size=[])
module(data)["action"]

data = TensorDict({"obs": torch.randn(5) * 100}, batch_size=[])
module(data)["action"]  # safe=True projects the result within the set

###############################################################################
# The :class:`~torchrl.modules.Actor` class has has a predefined output key (``"action"``):
#

from torchrl.modules import Actor

base_module = nn.Linear(5, 3)
actor = Actor(base_module, in_keys=["obs"])
data = TensorDict({"obs": torch.randn(5)}, batch_size=[])
actor(data)  # action is the default value

from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)

###############################################################################
# Working with probabilistic models is also made easy thanks to the ``tensordict.nn`` API:
#
from torchrl.modules import NormalParamExtractor, TanhNormal

td = TensorDict({"input": torch.randn(3, 5)}, [3])
net = nn.Sequential(
    nn.Linear(5, 4), NormalParamExtractor()
)  # splits the output in loc and scale
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
# Controlling randomness and sampling strategies is achieved via a context manager,
# :class:`~torchrl.envs.set_exploration_type`:
#
from torchrl.envs.utils import ExplorationType, set_exploration_type

td = TensorDict({"input": torch.randn(3, 5)}, [3])

torch.manual_seed(0)
with set_exploration_type(ExplorationType.RANDOM):
    td_module(td)
    print("random:", td["action"])

with set_exploration_type(ExplorationType.DETERMINISTIC):
    td_module(td)
    print("mode:", td["action"])

###############################################################################
# Using Environments and Modules
# ------------------------------
#
# Let us see how environments and modules can be combined:

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
data = env.reset()
data_stack = TensorDict(batch_size=[max_steps])
for i in range(max_steps):
    actor(data)
    data_stack[i] = env.step(data)
    if data["done"].any():
        break
    data = step_mdp(data)  # roughly equivalent to obs = next_obs

tensordicts_prealloc = data_stack.clone()
print("total steps:", i)
print(data_stack)

###############################################################################

# equivalent
torch.manual_seed(0)
env.set_seed(0)

max_steps = 100
data = env.reset()
data_stack = []
for _ in range(max_steps):
    actor(data)
    data_stack.append(env.step(data))
    if data["done"].any():
        break
    data = step_mdp(data)  # roughly equivalent to obs = next_obs
tensordicts_stack = torch.stack(data_stack, 0)
print("total steps:", i)
print(tensordicts_stack)

###############################################################################

(tensordicts_stack == tensordicts_prealloc).all()

###############################################################################

torch.manual_seed(0)
env.set_seed(0)
tensordict_rollout = env.rollout(policy=actor, max_steps=max_steps)
tensordict_rollout


(tensordict_rollout == tensordicts_prealloc).all()

from tensordict.nn import TensorDictModule

###############################################################################
# Collectors
# ----------
#
# We also provide a set of :ref:`data collectors <ref_collectors>`, that automaticall gather as many frames per batch as required.
# They work from single-node, single worker to multi-nodes, multi-workers settings.

from torchrl.collectors import MultiaSyncDataCollector, MultiSyncDataCollector

from torchrl.envs import EnvCreator, SerialEnv
from torchrl.envs.libs.gym import GymEnv

###############################################################################
# EnvCreator makes sure that we can send a lambda function from process to process
# We use a :class:`~torchrl.envs.SerialEnv` for simplicity (single worker), but for larger jobs a
# :class:`~torchrl.envs.ParallelEnv` (multi-workers) would be better suited.
#
# .. note:: Multiprocessed envs and multiprocessed collectors can be combined!
#

parallel_env = SerialEnv(
    3,
    EnvCreator(lambda: GymEnv("Pendulum-v1")),
)
create_env_fn = [parallel_env, parallel_env]

actor_module = nn.Linear(3, 1)
actor = TensorDictModule(actor_module, in_keys=["observation"], out_keys=["action"])

###############################################################################
# Sync multiprocessed data collector
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

devices = ["cpu", "cpu"]

collector = MultiSyncDataCollector(
    create_env_fn=create_env_fn,  # either a list of functions or a ParallelEnv
    policy=actor,
    total_frames=240,
    max_frames_per_traj=-1,  # envs are terminating, we don't need to stop them early
    frames_per_batch=60,  # we want 60 frames at a time (we have 3 envs per sub-collector)
    device=devices,
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
# Async multiprocessed data collector
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This class allows you to collect data while the model is training. This is particularily useful in off-policy settings
# as it decouples the inference and the model trainning. Data is delived in a first-ready-first-served basis (workers
# will queue their results):

collector = MultiaSyncDataCollector(
    create_env_fn=create_env_fn,  # either a list of functions or a ParallelEnv
    policy=actor,
    total_frames=240,
    max_frames_per_traj=-1,  # envs are terminating, we don't need to stop them early
    frames_per_batch=60,  # we want 60 frames at a time (we have 3 envs per sub-collector)
    device=devices,
)

for i, d in enumerate(collector):
    if i == 0:
        print(d)  # trajectories are split automatically in [6 workers x 10 steps]
    collector.update_policy_weights_()  # make sure that our policies have the latest weights if working on multiple devices
print(i)
collector.shutdown()
del collector
del create_env_fn
del parallel_env

###############################################################################
# Objectives
# ----------
# :ref:`Objectives <ref_objectives>` are the main entry points when coding up a new algorithm.
#

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

loss_fn = DDPGLoss(actor, value)
loss_fn.make_value_estimator(loss_fn.default_value_estimator, gamma=0.99)

###############################################################################

data = TensorDict(
    {
        "observation": torch.randn(10, 3),
        "next": {
            "observation": torch.randn(10, 3),
            "reward": torch.randn(10, 1),
            "done": torch.zeros(10, 1, dtype=torch.bool),
        },
        "action": torch.randn(10, 1),
    },
    batch_size=[10],
    device="cpu",
)
loss_td = loss_fn(data)

print(loss_td)

print(data)

###############################################################################
#
# Installing the Library
# ----------------------
#
# The library is on PyPI: *pip install torchrl*
# See the `README <https://github.com/pytorch/rl/blob/main/README.md>`_ for more information.
#
# Contributing
# ------------
#
# We are actively looking for contributors and early users. If you're working in
# RL (or just curious), try it! Give us feedback: what will make the success of
# TorchRL is how well it covers researchers needs. To do that, we need their input!
# Since the library is nascent, it is a great time for you to shape it the way you want!
#
# See the `Contributing guide <https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md>`_ for more info.
#
