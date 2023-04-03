# -*- coding: utf-8 -*-
"""
Coding DDPG using TorchRL
=========================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

"""
##############################################################################
# This tutorial will guide you through the steps to code DDPG from scratch.
#
# DDPG (`Deep Deterministic Policy Gradient <https://arxiv.org/abs/1509.02971>_`_)
# is a simple continuous control algorithm. It consists in learning a
# parametric value function for an action-observation pair, and
# then learning a policy that outputs actions that maximise this value
# function given a certain observation.
#
# This tutorial is more  than the PPO tutorial: it covers
# multiple topics that were left aside. We strongly advise the reader to go
# through the PPO tutorial first before trying out this one. The goal is to
# show how flexible torchrl is when it comes to writing scripts that can cover
# multiple use cases.
#
# Key learnings:
#
# - how to build an environment in TorchRL, including transforms
#   (e.g. data normalization) and parallel execution;
# - how to design a policy and value network;
# - how to collect data from your environment efficiently and store them
#   in a replay buffer;
# - how to store trajectories (and not transitions) in your replay buffer);
# - and finally how to evaluate your model.
#
# This tutorial assumes the reader is familiar with some of TorchRL primitives,
# such as :class:`tensordict.TensorDict` and
# :class:`tensordict.nn.TensorDictModules`, although it should be
# sufficiently transparent to be understood without a deep understanding of
# these classes.
#
# We do not aim at giving a SOTA implementation of the algorithm, but rather
# to provide a high-level illustration of TorchRL features in the context of
# this algorithm.
#
# Imports
# -------
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

from copy import deepcopy

import numpy as np
import torch
import torch.cuda
import tqdm
from matplotlib import pyplot as plt
from tensordict.nn import TensorDictModule
from torch import nn, optim
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import CompositeSpec, TensorDictReplayBuffer
from torchrl.data.postprocs import MultiStep
from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    ObservationNorm,
    ParallelEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.utils import set_exploration_mode, step_mdp
from torchrl.modules import (
    MLP,
    OrnsteinUhlenbeckProcessWrapper,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.modules.distributions.continuous import TanhDelta
from torchrl.objectives.utils import hold_out_net
from torchrl.trainers import Recorder

###############################################################################
# Environment
# -----------
#
# In most algorithms, the first thing that needs to be taken care of is the
# construction of the environmet as it conditions the remainder of the
# training script.
#
# For this example, we will be using the ``"cheetah"`` task. The goal is to make
# a half-cheetah run as fast as possible.
#
# In TorchRL, one can create such a task by relying on dm_control or gym:
#
# .. code-block:: python
#
#    env = GymEnv("HalfCheetah-v4")
#
# or
#
# .. code-block:: python
#
#    env = DMControlEnv("cheetah", "run")
#
# By default, these environment disable rendering. Training from states is
# usually easier than training from images. To keep things simple, we focus
# on learning from states only. To pass the pixels to the tensordicts that
# are collected by :func:`env.step()`, simply pass the ``from_pixels=True``
# argument to the constructor:
#
# .. code-block:: python
#
#    env = GymEnv("HalfCheetah-v4", from_pixels=True, pixels_only=True)
#
# We write a :func:`make_env` helper funciton that will create an environment
# with either one of the two backends considered above (dm-control or gym).
#

env_library = None
env_name = None


def make_env():
    """Create a base env."""
    global env_library
    global env_name

    if backend == "dm_control":
        env_name = "cheetah"
        env_task = "run"
        env_args = (env_name, env_task)
        env_library = DMControlEnv
    elif backend == "gym":
        env_name = "HalfCheetah-v4"
        env_args = (env_name,)
        env_library = GymEnv
    else:
        raise NotImplementedError

    env_kwargs = {
        "device": device,
        "frame_skip": frame_skip,
        "from_pixels": from_pixels,
        "pixels_only": from_pixels,
    }
    env = env_library(*env_args, **env_kwargs)
    return env


###############################################################################
# Transforms
# ^^^^^^^^^^
#
# Now that we have a base environment, we may want to modify its representation
# to make it more policy-friendly. In TorchRL, transforms are appended to the
# base environment in a specialized :class:`torchr.envs.TransformedEnv` class.
#
# - It is common in DDPG to rescale the reward using some heuristic value. We
#   will multiply the reward by 5 in this example.
#
# - If we are using :mod:`dm_control`, it is also important to build an interface
#   between the simulator which works with double precision numbers, and our
#   script which presumably uses single precision ones. This transformation goes
#   both ways: when calling :func:`env.step`, our actions will need to be
#   represented in double precision, and the output will need to be transformed
#   to single precision.
#   The :class:`torchrl.envs.DoubleToFloat` transform does exactly this: the
#   ``in_keys`` list refers to the keys that will need to be transformed from
#   double to float, while the ``in_keys_inv`` refers to those that need to
#   be transformed to double before being passed to the environment.
#
# - We concatenate the state keys together using the :class:`torchrl.envs.CatTensors`
#   transform.
#
# - Finally, we also leave the possibility of normalizing the states: we will
#   take care of computing the normalizing constants later on.
#


def make_transformed_env(
    env,
):
    """Apply transforms to the env (such as reward scaling and state normalization)."""

    env = TransformedEnv(env)

    # we append transforms one by one, although we might as well create the
    # transformed environment using the `env = TransformedEnv(base_env, transforms)`
    # syntax.
    env.append_transform(RewardScaling(loc=0.0, scale=reward_scaling))

    double_to_float_list = []
    double_to_float_inv_list = []
    if env_library is DMControlEnv:
        # DMControl requires double-precision
        double_to_float_list += [
            "reward",
            "action",
        ]
        double_to_float_inv_list += ["action"]

    # We concatenate all states into a single "observation_vector"
    # even if there is a single tensor, it'll be renamed in "observation_vector".
    # This facilitates the downstream operations as we know the name of the
    # output tensor.
    # In some environments (not half-cheetah), there may be more than one
    # observation vector: in this case this code snippet will concatenate them
    # all.
    selected_keys = list(env.observation_spec.keys())
    out_key = "observation_vector"
    env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))

    # we normalize the states, but for now let's just instantiate a stateless
    # version of the transform
    env.append_transform(ObservationNorm(in_keys=[out_key], standard_normal=True))

    double_to_float_list.append(out_key)
    env.append_transform(
        DoubleToFloat(
            in_keys=double_to_float_list, in_keys_inv=double_to_float_inv_list
        )
    )

    return env


###############################################################################
# Normalization of the observations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To compute the normalizing statistics, we run an arbitrary number of random
# steps in the environment and compute the mean and standard deviation of the
# collected observations. The :func:`ObservationNorm.init_stats()` method can
# be used for this purpose. To get the summary statistics, we create a dummy
# environment and run it for a given number of steps, collect data over a given
# number of steps and compute its summary statistics.
#


def get_env_stats():
    """Gets the stats of an environment."""
    proof_env = make_transformed_env(make_env())
    proof_env.set_seed(seed)
    t = proof_env.transform[2]
    t.init_stats(init_env_steps)
    transform_state_dict = t.state_dict()
    proof_env.close()
    return transform_state_dict


###############################################################################
# Parallel execution
# ^^^^^^^^^^^^^^^^^^
#
# The following helper function allows us to run environments in parallel.
# Running environments in parallel can significantly speed up the collection
# throughput. When using transformed environment, we need to choose whether we
# want to execute the transform individually for each environment, or
# centralize the data and transform it in batch. Both approaches are easy to
# code:
#
# .. code-block:: python
#
#    env = ParallelEnv(
#        lambda: TransformedEnv(GymEnv("HalfCheetah-v4"), transforms),
#        num_workers=4
#    )
#    env = TransformedEnv(
#        ParallelEnv(lambda: GymEnv("HalfCheetah-v4"), num_workers=4),
#        transforms
#    )
#
# To leverage the vectorization capabilities of PyTorch, we adopt
# the first method:
#


def parallel_env_constructor(
    transform_state_dict,
):
    if env_per_collector == 1:

        def make_t_env():
            env = make_transformed_env(make_env())
            env.transform[2].init_stats(3)
            env.transform[2].loc.copy_(transform_state_dict["loc"])
            env.transform[2].scale.copy_(transform_state_dict["scale"])
            return env

        env_creator = EnvCreator(make_t_env)
        return env_creator

    parallel_env = ParallelEnv(
        num_workers=env_per_collector,
        create_env_fn=EnvCreator(lambda: make_env()),
        create_env_kwargs=None,
        pin_memory=False,
    )
    env = make_transformed_env(parallel_env)
    # we call `init_stats` for a limited number of steps, just to instantiate
    # the lazy buffers.
    env.transform[2].init_stats(3, cat_dim=1, reduce_dim=[0, 1])
    env.transform[2].load_state_dict(transform_state_dict)
    return env


###############################################################################
# Building the model
# ------------------
#
# We now turn to the setup of the model and loss function. DDPG requires a
# value network, trained to estimate the value of a state-action pair, and a
# parametric actor that learns how to select actions that maximize this value.
# In this tutorial, we will be using two independent networks for these
# components.
#
# Recall that building a torchrl module requires two steps:
#
# - writing the :class:`torch.nn.Module` that will be used as network
# - wrapping the network in a :class:`tensordict.nn.TensorDictModule` where the
#   data flow is handled by specifying the input and output keys.
#
# In more complex scenarios, :class:`tensordict.nn.TensorDictSequential` can
# also be used.
#
# In :func:`make_ddpg_actor`, we use a :class:`torchrl.modules.ProbabilisticActor`
# object to wrap our policy network. Since DDPG is a deterministic algorithm,
# this is not strictly necessary. We rely on this class to map the output
# action to the appropriate domain. Alternatively, one could perfectly use a
# non-linearity such as :class:`torch.tanh` to map the output to the right
# domain.
#
# The Q-Value network is wrapped in a :class:`torchrl.modules.ValueOperator`
# that automatically sets the ``out_keys`` to ``"state_action_value`` for q-value
# networks and ``state_value`` for other value networks.
#
# Since we use lazy modules, it is necessary to materialize the lazy modules
# before being able to move the policy from device to device and achieve other
# operations. Hence, it is good practice to run the modules with a small
# sample of data. For this purpose, we generate fake data from the
# environment specs.
#


def make_ddpg_actor(
    transform_state_dict,
    device="cpu",
):
    proof_environment = make_transformed_env(make_env())
    proof_environment.transform[2].init_stats(3)
    proof_environment.transform[2].load_state_dict(transform_state_dict)

    env_specs = proof_environment.specs
    out_features = env_specs["input_spec"]["action"].shape[0]

    actor_net = MLP(
        num_cells=[num_cells] * num_layers,
        activation_class=nn.Tanh,
        out_features=out_features,
    )
    in_keys = ["observation_vector"]
    out_keys = ["param"]

    actor_module = TensorDictModule(actor_net, in_keys=in_keys, out_keys=out_keys)

    # We use a ProbabilisticActor to make sure that we map the network output
    # to the right space using a TanhDelta distribution.
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["param"],
        spec=CompositeSpec(action=env_specs["input_spec"]["action"]),
        safe=True,
        distribution_class=TanhDelta,
        distribution_kwargs={
            "min": env_specs["input_spec"]["action"].space.minimum,
            "max": env_specs["input_spec"]["action"].space.maximum,
        },
    ).to(device)

    q_net = MLP(
        num_cells=[num_cells] * num_layers,
        activation_class=nn.Tanh,
        out_features=1,
    )

    in_keys = in_keys + ["action"]
    qnet = ValueOperator(
        in_keys=in_keys,
        module=q_net,
    ).to(device)

    # init: since we have lazy layers, we should run the network
    # once to initialize them
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.fake_tensordict()
        td = td.expand((*td.shape, 2))
        td = td.to(device)
        actor(td)
        qnet(td)

    return actor, qnet


###############################################################################
# Evaluator: building your recorder object
# ----------------------------------------
#
# As the training data is obtained using some exploration strategy, the true
# performance of our algorithm needs to be assessed in deterministic mode. We
# do this using a dedicated class, ``Recorder``, which executes the policy in
# the environment at a given frequency and returns some statistics obtained
# from these simulations.
#
# The following helper function builds this object:


def make_recorder(actor_model_explore, transform_state_dict):
    base_env = make_env()
    recorder = make_transformed_env(base_env)
    recorder.transform[2].init_stats(3)
    recorder.transform[2].load_state_dict(transform_state_dict)

    recorder_obj = Recorder(
        record_frames=1000,
        frame_skip=frame_skip,
        policy_exploration=actor_model_explore,
        recorder=recorder,
        exploration_mode="mean",
        record_interval=record_interval,
    )
    return recorder_obj


###############################################################################
# Replay buffer
# -------------
#
# Replay buffers come in two flavors: prioritized (where some error signal
# is used to give a higher likelihood of sampling to some items than others)
# and regular, circular experience replay.
#
# TorchRL replay buffers are composable: one can pick up the storage, sampling
# and writing strategies. It is also possible to
# store tensors on physical memory using a memory-mapped array. The following
# function takes care of creating the replay buffer with the desired
# hyperparameters:
#


def make_replay_buffer(buffer_size, prefetch=3):
    if prb:
        sampler = PrioritizedSampler(
            max_capacity=buffer_size,
            alpha=0.7,
            beta=0.5,
        )
    else:
        sampler = RandomSampler()
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(
            buffer_size,
            scratch_dir=buffer_scratch_dir,
            device=device,
        ),
        sampler=sampler,
        pin_memory=False,
        prefetch=prefetch,
    )
    return replay_buffer


###############################################################################
# Hyperparameters
# ---------------
#
# After having written our helper functions, it is time to set the
# experiment hyperparameters:

###############################################################################
# Environment
# ^^^^^^^^^^^

# The backend can be gym or dm_control
backend = "gym"

exp_name = "cheetah"

# frame_skip batches multiple step together with a single action
# If > 1, the other frame counts (e.g. frames_per_batch, total_frames) need to
# be adjusted to have a consistent total number of frames collected across
# experiments.
frame_skip = 2
from_pixels = False
# Scaling the reward helps us control the signal magnitude for a more
# efficient learning.
reward_scaling = 5.0

# Number of random steps used as for stats computation using ObservationNorm
init_env_steps = 1000

# Exploration: Number of frames before OU noise becomes null
annealing_frames = 1000000 // frame_skip

###############################################################################
# Collection
# ^^^^^^^^^^

# We will execute the policy on cuda if available
device = (
    torch.device("cpu") if torch.cuda.device_count() == 0 else torch.device("cuda:0")
)

# Number of environments in each data collector
env_per_collector = 2

# Total frames we will use during training. Scale up to 500K - 1M for a more
# meaningful training
total_frames = 5000 // frame_skip
# Number of frames returned by the collector at each iteration of the outer loop
frames_per_batch = env_per_collector * 1000 // frame_skip
max_frames_per_traj = 1000 // frame_skip
init_random_frames = 0
# We'll be using the MultiStep class to have a less myopic representation of
# upcoming states
n_steps_forward = 3

# record every 10 batch collected
record_interval = 10

###############################################################################
# Optimizer and optimization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

lr = 5e-4
weight_decay = 0.0
# UTD: Number of iterations of the inner loop
update_to_data = 32
batch_size = 128

###############################################################################
# Model
# ^^^^^

gamma = 0.99
tau = 0.005  # Decay factor for the target network

# Network specs
num_cells = 64
num_layers = 2

###############################################################################
# Replay buffer
# ^^^^^^^^^^^^^

# If True, a Prioritized replay buffer will be used
prb = True
# Number of frames stored in the buffer
buffer_size = min(total_frames, 1000000 // frame_skip)
buffer_scratch_dir = "/tmp/"

seed = 0

###############################################################################
# Initialization
# --------------
#
# To initialize the experiment, we first acquire the observation statistics,
# then build the networks, wrap them in an exploration wrapper (following the
# seminal DDPG paper, we used an Ornstein-Uhlenbeck process to add noise to the
# sampled actions).


# Seeding
torch.manual_seed(seed)
np.random.seed(seed)

###############################################################################
# Normalization stats
# ^^^^^^^^^^^^^^^^^^^

transform_state_dict = get_env_stats()

###############################################################################
# Models: policy and q-value network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

actor, qnet = make_ddpg_actor(
    transform_state_dict=transform_state_dict,
    device=device,
)
if device == torch.device("cpu"):
    actor.share_memory()

###############################################################################
# We create a copy of the q-value network to be used as target network

qnet_target = deepcopy(qnet).requires_grad_(False)

###############################################################################
# The policy is wrapped in a :class:`torchrl.modules.OrnsteinUhlenbeckProcessWrapper`
# exploration module:

actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
    actor,
    annealing_num_steps=annealing_frames,
).to(device)
if device == torch.device("cpu"):
    actor_model_explore.share_memory()

###############################################################################
# Parallel environment creation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We pass the stats computed earlier to normalize the output of our
# environment:

create_env_fn = parallel_env_constructor(
    transform_state_dict=transform_state_dict,
)

###############################################################################
# Data collector
# ^^^^^^^^^^^^^^
#
# TorchRL provides specialized classes to help you collect data by executing
# the policy in the environment. These "data collectors" iteratively compute
# the action to be executed at a given time, then execute a step in the
# environment and reset it when required.
# Data collectors are designed to help developers have a tight control
# on the number of frames per batch of data, on the (a)sync nature of this
# collection and on the resources allocated to the data collection (e.g. GPU,
# number of workers etc).
#
# Here we will use
# :class:`torchrl.collectors.MultiaSyncDataCollector`, a data collector that
# will be executed in an async manner (i.e. data will be collected while
# the policy is being optimized). With the :class:`MultiaSyncDataCollector`,
# multiple workers are running rollouts separately. When a batch is asked, it
# is gathered from the first worker that can provide it.
#
# The parameters to specify are:
#
# - the list of environment creation functions,
# - the policy,
# - the total number of frames before the collector is considered empty,
# - the maximum number of frames per trajectory (useful for non-terminating
#   environments, like dm_control ones).
#
# One should also pass:
#
# - the number of frames in each batch collected,
# - the number of random steps executed independently from the policy,
# - the devices used for policy execution
# - the devices used to store data before the data is passed to the main
#   process.
#
# Collectors also accept post-processing hooks.
# For instance, the :class:`torchrl.data.postprocs.MultiStep` class passed as
# ``postproc`` makes it so that the rewards of the ``n`` upcoming steps are
# summed (with some discount factor) and the next observation is changed to
# be the n-step forward observation. One could pass other transforms too:
# using :class:`tensordict.nn.TensorDictModule` and
# :class:`tensordict.nn.TensorDictSequential` we can seamlessly append a
# wide range of transforms to our collector.

if n_steps_forward > 0:
    multistep = MultiStep(n_steps=n_steps_forward, gamma=gamma)
else:
    multistep = None

collector = MultiaSyncDataCollector(
    create_env_fn=[create_env_fn, create_env_fn],
    policy=actor_model_explore,
    total_frames=total_frames,
    max_frames_per_traj=max_frames_per_traj,
    frames_per_batch=frames_per_batch,
    init_random_frames=init_random_frames,
    reset_at_each_iter=False,
    postproc=multistep,
    split_trajs=True,
    devices=[device, device],  # device for execution
    storing_devices=[device, device],  # device where data will be stored and passed
    pin_memory=False,
    update_at_each_batch=False,
    exploration_mode="random",
)

collector.set_seed(seed)

###############################################################################
# Replay buffer
# ^^^^^^^^^^^^^
#

replay_buffer = make_replay_buffer(buffer_size, prefetch=3)

###############################################################################
# Recorder
# ^^^^^^^^

recorder = make_recorder(actor_model_explore, transform_state_dict)

###############################################################################
# Optimizer
# ^^^^^^^^^
#
# Finally, we will use the Adam optimizer for the policy and value network,
# with the same learning rate for both.

optimizer_actor = optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_qnet = optim.Adam(qnet.parameters(), lr=lr, weight_decay=weight_decay)
total_collection_steps = total_frames // frames_per_batch

scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_actor, T_max=total_collection_steps
)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_qnet, T_max=total_collection_steps
)

###############################################################################
# Time to train the policy
# ------------------------
#
# Some notes about the following training loop:
#
# - :func:`torchrl.objectives.utils.hold_out_net` is a TorchRL context manager
#   that temporarily sets :func:`torch.Tensor.requires_grad_()` to False for
#   a designated set of network parameters. This is used to
#   prevent :func:`torch.Tensor.backward()`` from writing gradients on
#   parameters that need not to be differentiated given the loss at hand.
# - The value network is designed using the
#   :class:`torchrl.modules.ValueOperator` subclass from
#   :class:`tensordict.nn.TensorDictModule` class. As explained earlier,
#   this class will write a ``"state_action_value"`` entry if one of its
#   ``in_keys`` is named ``"action"``, otherwise it will assume that only the
#   state-value is returned and the output key will simply be ``"state_value"``.
#   In the case of DDPG, the value if of the state-action pair,
#   hence the ``"state_action_value"`` will be used.
# - The :func:`torchrl.envs.utils.step_mdp(tensordict)` helper function is the
#   equivalent of the ``obs = next_obs`` command found in multiple RL
#   algorithms. It will return a new :class:`tensordict.TensorDict` instance
#   that contains all the data that will need to be used in the next iteration.
#   This makes it possible to pass this new tensordict to the policy or
#   value network.
# - When using prioritized replay buffer, a priority key is added to the
#   sampled tensordict (named ``"td_error"`` by default). Then, this
#   TensorDict will be fed back to the replay buffer using the
#   :func:`torchrl.data.replay_buffers.TensorDictReplayBuffer.update_tensordict_priority`
#   method. Under the hood, this method will read the index present in the
#   TensorDict as well as the priority value, and update its list of priorities
#   at these indices.
# - TorchRL provides optimized versions of the loss functions (such as this one)
#   where one only needs to pass a sampled tensordict and obtains a dictionary
#   of losses and metadata in return (see :mod:`torchrl.objectives` for more
#   context). Here we write the full loss function in the optimization loop
#   for transparency.
#   Similarly, the target network updates are written explicitly but
#   TorchRL provides a couple of dedicated classes for this
#   (see :class:`torchrl.objectives.SoftUpdate` and
#   :class:`torchrl.objectives.HardUpdate`).
# - After each collection of data, we call :func:`collector.update_policy_weights_()`,
#   which will update the policy network weights on the data collector. If the
#   code is executed on cpu or with a single cuda device, this part can be
#   omitted. If the collector is executed on another device, then its weights
#   must be synced with those on the main, training process and this method
#   should be incorporated in the training loop (ideally early in the loop in
#   async settings, and at the end of it in sync settings).

rewards = []
rewards_eval = []

# Main loop
norm_factor_training = (
    sum(gamma**i for i in range(n_steps_forward)) if n_steps_forward else 1
)

collected_frames = 0
pbar = tqdm.tqdm(total=total_frames)
r0 = None
for i, tensordict in enumerate(collector):

    # update weights of the inference policy
    collector.update_policy_weights_()

    if r0 is None:
        r0 = tensordict["next", "reward"].mean().item()
    pbar.update(tensordict.numel())

    # extend the replay buffer with the new data
    if ("collector", "mask") in tensordict.keys(True):
        # if multi-step, a mask is present to help filter padded values
        current_frames = tensordict["collector", "mask"].sum()
        tensordict = tensordict[tensordict.get(("collector", "mask"))]
    else:
        tensordict = tensordict.view(-1)
        current_frames = tensordict.numel()
    collected_frames += current_frames
    replay_buffer.extend(tensordict.cpu())

    # optimization steps
    if collected_frames >= init_random_frames:
        for _ in range(update_to_data):
            # sample from replay buffer
            sampled_tensordict = replay_buffer.sample(batch_size).clone()

            # compute loss for qnet and backprop
            with hold_out_net(actor):
                # get next state value
                next_tensordict = step_mdp(sampled_tensordict)
                qnet_target(actor(next_tensordict))
                next_value = next_tensordict["state_action_value"]
                assert not next_value.requires_grad
            value_est = (
                sampled_tensordict["next", "reward"]
                + gamma * (1 - sampled_tensordict["next", "done"].float()) * next_value
            )
            value = qnet(sampled_tensordict)["state_action_value"]
            value_loss = (value - value_est).pow(2).mean()
            # we write the td_error in the sampled_tensordict for priority update
            # because the indices of the samples is tracked in sampled_tensordict
            # and the replay buffer will know which priorities to update.
            sampled_tensordict["td_error"] = (value - value_est).pow(2).detach()
            value_loss.backward()

            optimizer_qnet.step()
            optimizer_qnet.zero_grad()

            # compute loss for actor and backprop:
            # the actor must maximise the state-action value, hence the loss
            # is the neg value of this.
            sampled_tensordict_actor = sampled_tensordict.select(*actor.in_keys)
            with hold_out_net(qnet):
                qnet(actor(sampled_tensordict_actor))
            actor_loss = -sampled_tensordict_actor["state_action_value"]
            actor_loss.mean().backward()

            optimizer_actor.step()
            optimizer_actor.zero_grad()

            # update qnet_target params
            for (p_in, p_dest) in zip(qnet.parameters(), qnet_target.parameters()):
                p_dest.data.copy_(tau * p_in.data + (1 - tau) * p_dest.data)
            for (b_in, b_dest) in zip(qnet.buffers(), qnet_target.buffers()):
                b_dest.data.copy_(tau * b_in.data + (1 - tau) * b_dest.data)

            # update priority
            if prb:
                replay_buffer.update_tensordict_priority(sampled_tensordict)

    rewards.append(
        (
            i,
            tensordict["next", "reward"].mean().item()
            / norm_factor_training
            / frame_skip,
        )
    )
    td_record = recorder(None)
    if td_record is not None:
        rewards_eval.append((i, td_record["r_evaluation"].item()))
    if len(rewards_eval):
        pbar.set_description(
            f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}), reward eval: reward: {rewards_eval[-1][1]: 4.4f}"
        )

    # update the exploration strategy
    actor_model_explore.step(current_frames)
    if collected_frames >= init_random_frames:
        scheduler1.step()
        scheduler2.step()

collector.shutdown()
del collector

###############################################################################
# Experiment results
# ------------------
#
# We make a simple plot of the average rewards during training. We can observe
# that our policy learned quite well to solve the task.
#
# **Note**: As already mentioned above, to get a more reasonable performance,
# use a greater value for ``total_frames`` e.g. 1M.

plt.figure()
plt.plot(*zip(*rewards), label="training")
plt.plot(*zip(*rewards_eval), label="eval")
plt.legend()
plt.xlabel("iter")
plt.ylabel("reward")
plt.tight_layout()

###############################################################################
# Sampling trajectories and using TD(lambda)
# ------------------------------------------
#
# TD(lambda) is known to be less biased than the regular TD-error we used in
# the previous example. To use it, however, we need to sample trajectories and
# not single transitions.
#
# We modify the previous example to make this possible.
#
# The first modification consists in building a replay buffer that stores
# trajectories (and not transitions).
#
# Specifically, we'll collect trajectories of (at most)
# 250 steps (note that the total trajectory length is actually 1000 frames, but
# we collect batches of 500 transitions obtained over 2 environments running in
# parallel, hence only 250 steps per trajectory are collected at any given
# time). Hence, we'll divide our replay buffer size by 250:

buffer_size = 100000 // frame_skip // 250
print("the new buffer size is", buffer_size)
batch_size_traj = max(4, batch_size // 250)
print("the new batch size for trajectories is", batch_size_traj)

n_steps_forward = 0  # disable multi-step for simplicity

###############################################################################
# The following code is identical to the initialization we made earlier:

torch.manual_seed(seed)
np.random.seed(seed)

# get stats for normalization
transform_state_dict = get_env_stats()

# Actor and qnet instantiation
actor, qnet = make_ddpg_actor(
    transform_state_dict=transform_state_dict,
    device=device,
)
if device == torch.device("cpu"):
    actor.share_memory()

# Target network
qnet_target = deepcopy(qnet).requires_grad_(False)

# Exploration wrappers:
actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
    actor,
    annealing_num_steps=annealing_frames,
).to(device)
if device == torch.device("cpu"):
    actor_model_explore.share_memory()

# Environment setting:
create_env_fn = parallel_env_constructor(
    transform_state_dict=transform_state_dict,
)
# Batch collector:
collector = MultiaSyncDataCollector(
    create_env_fn=[create_env_fn, create_env_fn],
    policy=actor_model_explore,
    total_frames=total_frames,
    max_frames_per_traj=max_frames_per_traj,
    frames_per_batch=frames_per_batch,
    init_random_frames=init_random_frames,
    reset_at_each_iter=False,
    postproc=None,
    split_trajs=False,
    devices=[device, device],  # device for execution
    storing_devices=[device, device],  # device where data will be stored and passed
    seed=None,
    pin_memory=False,
    update_at_each_batch=False,
    exploration_mode="random",
)
collector.set_seed(seed)

# Replay buffer:
replay_buffer = make_replay_buffer(buffer_size, prefetch=0)

# trajectory recorder
recorder = make_recorder(actor_model_explore, transform_state_dict)

# Optimizers
optimizer_actor = optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_qnet = optim.Adam(qnet.parameters(), lr=lr, weight_decay=weight_decay)
total_collection_steps = total_frames // frames_per_batch

scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_actor, T_max=total_collection_steps
)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_qnet, T_max=total_collection_steps
)

###############################################################################
# The training loop needs to be slightly adapted.
# First, whereas before extending the replay buffer we used to flatten the
# collected data, this won't be the case anymore. To understand why, let's
# check the output shape of the data collector:

for data in collector:
    print(data.shape)
    break

###############################################################################
# We see that our data has shape ``[2, 250]`` as expected: 2 envs, each
# returning 250 frames.
#
# Let's import the td_lambda function:
#

from torchrl.objectives.value.functional import vec_td_lambda_advantage_estimate

lmbda = 0.95

###############################################################################
# The training loop is roughly the same as before, with the exception that we
# don't flatten the collected data. Also, the sampling from the replay buffer
# is slightly different: We will collect at minimum four trajectories, compute
# the returns (TD(lambda)), then sample from these the values we'll be using
# to compute gradients. This ensures that do not have batches that are
# 'too big' but still compute an accurate return.
#

rewards = []
rewards_eval = []

# Main loop
norm_factor_training = (
    sum(gamma**i for i in range(n_steps_forward)) if n_steps_forward else 1
)

collected_frames = 0
# # if tqdm is to be used
# pbar = tqdm.tqdm(total=total_frames)
r0 = None
for i, tensordict in enumerate(collector):

    # update weights of the inference policy
    collector.update_policy_weights_()

    if r0 is None:
        r0 = tensordict["next", "reward"].mean().item()

    # extend the replay buffer with the new data
    current_frames = tensordict.numel()
    collected_frames += current_frames
    replay_buffer.extend(tensordict.cpu())

    # optimization steps
    if collected_frames >= init_random_frames:
        for _ in range(update_to_data):
            # sample from replay buffer
            sampled_tensordict = replay_buffer.sample(batch_size_traj)
            # reset the batch size temporarily, and exclude index
            # whose shape is incompatible with the new size
            index = sampled_tensordict.get("index")
            sampled_tensordict.exclude("index", inplace=True)

            # compute loss for qnet and backprop
            with hold_out_net(actor):
                # get next state value
                next_tensordict = step_mdp(sampled_tensordict)
                qnet_target(actor(next_tensordict.view(-1))).view(
                    sampled_tensordict.shape
                )
                next_value = next_tensordict["state_action_value"]
                assert not next_value.requires_grad

            # This is the crucial part: we'll compute the TD(lambda)
            # instead of a simple single step estimate
            done = sampled_tensordict["next", "done"]
            reward = sampled_tensordict["next", "reward"]
            value = qnet(sampled_tensordict.view(-1)).view(sampled_tensordict.shape)[
                "state_action_value"
            ]
            advantage = vec_td_lambda_advantage_estimate(
                gamma,
                lmbda,
                value,
                next_value,
                reward,
                done,
                time_dim=sampled_tensordict.ndim - 1,
            )
            # we sample from the values we have computed
            rand_idx = torch.randint(0, advantage.numel(), (batch_size,))
            value_loss = advantage.view(-1)[rand_idx].pow(2).mean()

            # we write the td_error in the sampled_tensordict for priority update
            # because the indices of the samples is tracked in sampled_tensordict
            # and the replay buffer will know which priorities to update.
            value_loss.backward()

            optimizer_qnet.step()
            optimizer_qnet.zero_grad()

            # compute loss for actor and backprop: the actor must maximise the state-action value, hence the loss is the neg value of this.
            sampled_tensordict_actor = sampled_tensordict.select(*actor.in_keys)
            with hold_out_net(qnet):
                qnet(actor(sampled_tensordict_actor.view(-1))).view(
                    sampled_tensordict.shape
                )
            actor_loss = -sampled_tensordict_actor["state_action_value"]
            actor_loss.view(-1)[rand_idx].mean().backward()

            optimizer_actor.step()
            optimizer_actor.zero_grad()

            # update qnet_target params
            for (p_in, p_dest) in zip(qnet.parameters(), qnet_target.parameters()):
                p_dest.data.copy_(tau * p_in.data + (1 - tau) * p_dest.data)
            for (b_in, b_dest) in zip(qnet.buffers(), qnet_target.buffers()):
                b_dest.data.copy_(tau * b_in.data + (1 - tau) * b_dest.data)

            # update priority
            sampled_tensordict.batch_size = [batch_size_traj]
            sampled_tensordict["td_error"] = advantage.detach().pow(2).mean(1)
            sampled_tensordict["index"] = index
            if prb:
                replay_buffer.update_tensordict_priority(sampled_tensordict)

    rewards.append(
        (
            i,
            tensordict["next", "reward"].mean().item()
            / norm_factor_training
            / frame_skip,
        )
    )
    td_record = recorder(None)
    if td_record is not None:
        rewards_eval.append((i, td_record["r_evaluation"].item()))
    #     if len(rewards_eval):
    #         pbar.set_description(f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}), reward eval: reward: {rewards_eval[-1][1]: 4.4f}")

    # update the exploration strategy
    actor_model_explore.step(current_frames)
    if collected_frames >= init_random_frames:
        scheduler1.step()
        scheduler2.step()

collector.shutdown()
del create_env_fn
del collector

###############################################################################
# We can observe that using TD(lambda) made our results considerably more
# stable for a similar training speed:
#
# **Note**: As already mentioned above, to get a more reasonable performance,
# use a greater value for ``total_frames`` e.g. 1000000.

plt.figure()
plt.plot(*zip(*rewards), label="training")
plt.plot(*zip(*rewards_eval), label="eval")
plt.legend()
plt.xlabel("iter")
plt.ylabel("reward")
plt.tight_layout()
plt.title("TD-labmda DDPG results")
