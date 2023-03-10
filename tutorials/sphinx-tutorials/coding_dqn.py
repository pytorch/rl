# -*- coding: utf-8 -*-
"""
Coding a pixel-based DQN using TorchRL
======================================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

"""

##############################################################################
# This tutorial will guide you through the steps to code DQN to solve the
# CartPole task from scratch. DQN
# (`Deep Q-Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_) was
# the founding work in deep reinforcement learning.
# On a high level, the algorithm is quite simple: Q-learning consists in learning a table of
# state-action values in such a way that, when encountering any particular state,
# we know which action to pick just by searching for the action with the
# highest value. This simple setting requires the actions and states to be
# discrete, otherwise a lookup table cannot be built.
#
# DQN uses a neural network that encodes a map from the state-action space to
# a value (scalar) space, which amortizes the cost of storing and exploring all
# the possible state-action combinations: if a state has not been seen in the
# past, we can still pass it in conjunction with the various actions available
# through our neural network and get an interpolated value for each of the
# actions available.
#
# We will solve the classic control problem of the cart pole. From the
# Gymnasium doc from where this environment is retrieved:
#
# | A pole is attached by an un-actuated joint to a cart, which moves along a
# | frictionless track. The pendulum is placed upright on the cart and the goal
# | is to balance the pole by applying forces in the left and right direction
# | on the cart.
#
# .. figure:: /_static/img/cartpole_demo.gif
#    :alt: Cart Pole
#
# **Prerequisites**: We encourage you to get familiar with torchrl through the
# `PPO tutorial <https://pytorch.org/rl/tutorials/coding_ppo.html>`_ first.
# This tutorial is more complex and full-fleshed, but it may be .
#
# In this tutorial, you will learn:
#
# - how to build an environment in TorchRL, including transforms (e.g. data
#   normalization, frame concatenation, resizing and turning to grayscale)
#   and parallel execution. Unlike what we did in the
#   `DDPG tutorial <https://pytorch.org/rl/tutorials/coding_ddpg.html>`_, we
#   will normalize the pixels and not the state vector.
# - how to design a QValue actor, i.e. an actor that estimates the action
#   values and picks up the action with the highest estimated return;
# - how to collect data from your environment efficiently and store them
#   in a replay buffer;
# - how to store trajectories (and not transitions) in your replay buffer),
#   and how to estimate returns using TD(lambda);
# - how to make a module functional and use ;
# - and finally how to evaluate your model.
#
# This tutorial assumes the reader is familiar with some of TorchRL
# primitives, such as :class:`tensordict.TensorDict` and
# :class:`tensordict.TensorDictModules`, although it
# should be sufficiently transparent to be understood without a deep
# understanding of these classes.
#
# We do not aim at giving a SOTA implementation of the algorithm, but rather
# to provide a high-level illustration of TorchRL features in the context
# of this algorithm.

# sphinx_gallery_start_ignore
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import torch
import tqdm
from functorch import vmap
from matplotlib import pyplot as plt
from tensordict import TensorDict
from tensordict.nn import get_functional
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import EnvCreator, ParallelEnv, RewardScaling, StepCounter
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    CatTensors,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.utils import set_exploration_mode, step_mdp
from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


###############################################################################
# Hyperparameters
# ---------------
#
# Let's start with our hyperparameters. The following setting should work well
# in practice, and the performance of the algorithm should hopefully not be
# too sensitive to slight variations of these.

device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

###############################################################################
# Optimizer
# ^^^^^^^^^

# the learning rate of the optimizer
lr = 2e-3
# the beta parameters of Adam
betas = (0.9, 0.999)
# Optimization steps per batch collected (aka UPD or updates per data)
n_optim = 8

###############################################################################
# DQN parameters
# ^^^^^^^^^^^^^^

###############################################################################
# gamma decay factor
gamma = 0.99

###############################################################################
# lambda decay factor (see second the part with TD(:math:`\lambda`)
lmbda = 0.95

###############################################################################
# Smooth target network update decay parameter.
# This loosely corresponds to a 1/(1-tau) interval with hard target network
# update
tau = 0.005

###############################################################################
# Data collection and replay buffer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Values to be used for proper training have been commented.
#
# Total frames collected in the environment. In other implementations, the
# user defines a maximum number of episodes.
# This is harder to do with our data collectors since they return batches
# of N collected frames, where N is a constant.
# However, one can easily get the same restriction on number of episodes by
# breaking the training loop when a certain number
# episodes has been collected.
total_frames = 5000  # 500000

###############################################################################
# Random frames used to initialize the replay buffer.
init_random_frames = 100  # 1000

###############################################################################
# Frames in each batch collected.
frames_per_batch = 32  # 128

###############################################################################
# Frames sampled from the replay buffer at each optimization step
batch_size = 32  # 256

###############################################################################
# Size of the replay buffer in terms of frames
buffer_size = min(total_frames, 100000)

###############################################################################
# Number of environments run in parallel in each data collector
num_workers = 2  # 8
num_collectors = 2  # 4


###############################################################################
# Environment and exploration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We set the initial and final value of the epsilon factor in Epsilon-greedy
# exploration.
# Since our policy is deterministic, exploration is crucial: without it, the
# only source of randomness would be the environment reset.

eps_greedy_val = 0.1
eps_greedy_val_env = 0.005

###############################################################################
# To speed up learning, we set the bias of the last layer of our value network
# to a predefined value (this is not mandatory)
init_bias = 2.0

###############################################################################
# **Note**: for fast rendering of the tutorial ``total_frames`` hyperparameter
# was set to a very low number. To get a reasonable performance, use a greater
# value e.g. 500000
#
# Building the environment
# ------------------------
#
# Our environment builder has two arguments:
#
# - ``parallel``: determines whether multiple environments have to be run in
#   parallel. We stack the transforms after the
#   :class:`torchrl.envs.ParallelEnv` to take advantage
#   of vectorization of the operations on device, although this would
#   technically work with every single environment attached to its own set of
#   transforms.
# - ``observation_norm_state_dict`` will contain the normalizing constants for
#   the :class:`torchrl.envs.ObservationNorm` tranform.
#
# We will be using five transforms:
#
# - :class:`torchrl.envs.ToTensorImage` will convert a ``[W, H, C]`` uint8
#   tensor in a floating point tensor in the ``[0, 1]`` space with shape
#   ``[C, W, H]``;
# - :class:`torchrl.envs.RewardScaling` to reduce the scale of the return;
# - :class:`torchrl.envs.GrayScale` will turn our image into grayscale;
# - :class:`torchrl.envs.Resize` will resize the image in a 64x64 format;
# - :class:`torchrl.envs.CatFrames` will concatenate an arbitrary number of
#   successive frames (``N=4``) in a single tensor along the channel dimension.
#   This is useful as a single image does not carry information about the
#   motion of the cartpole. Some memory about past observations and actions
#   is needed, either via a recurrent neural network or using a stack of
#   frames.
# - :class:`torchrl.envs.ObservationNorm` which will normalize our observations
#   given some custom summary statistics.
#


def make_env(parallel=False, observation_norm_state_dict=None):
    if observation_norm_state_dict is None:
        observation_norm_state_dict = {"standard_normal": True}
    if parallel:
        base_env = ParallelEnv(
            num_workers,
            EnvCreator(
                lambda: GymEnv(
                    "CartPole-v1", from_pixels=True, pixels_only=True, device=device
                )
            ),
        )
    else:
        base_env = GymEnv(
            "CartPole-v1", from_pixels=True, pixels_only=True, device=device
        )

    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),  # to count the steps of each trajectory
            ToTensorImage(),
            RewardScaling(loc=0.0, scale=0.1),
            GrayScale(),
            Resize(64, 64),
            CatFrames(4, in_keys=["pixels"], dim=-3),
            ObservationNorm(in_keys=["pixels"], **observation_norm_state_dict),
        ),
    )
    return env


###############################################################################
# Compute normalizing constants
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To normalize images, we don't want to normalize each pixel independently
# with a full ``[C, W, H]`` normalizing mask, but with simpler ``[C, 1, 1]``
# shaped loc and scale parameters. We will be using the ``reduce_dim`` argument
# of :func:`torchrl.envs.ObservationNorm.init_stats` to instruct which
# dimensions must be reduced, and the ``keep_dims`` parameter to ensure that
# not all dimensions disappear in the process:

test_env = make_env()
test_env.transform[-1].init_stats(
    num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
)
observation_norm_state_dict = test_env.transform[-1].state_dict()

###############################################################################
# let's check that normalizing constants have a size of ``[C, 1, 1]`` where
# ``C=4`` (because of :class:`torchrl.envs.CatFrames`).
print(observation_norm_state_dict)

###############################################################################
# Building the model (Deep Q-network)
# -----------------------------------
#
# The following function builds a :class:`torchrl.modules.DuelingCnnDQNet`
# object which is a simple CNN followed by a two-layer MLP. The only trick used
# here is that the action values (i.e. left and right action value) are
# computed using
#
# .. math::
#
#    val = b(obs) + v(obs) - \mathbb{E}[v(obs)]
#
# where :math:`b` is a :math:`\# obs \rightarrow 1` function and :math:`v` is a
# :math:`\# obs \rightarrow num_actions` function.
#
# Our network is wrapped in a :class:`torchrl.modules.QValueActor`, which will read the state-action
# values, pick up the one with the maximum value and write all those results
# in the input :class:`tensordict.TensorDict`.
#
# Target parameters
# ^^^^^^^^^^^^^^^^^
#
# Many off-policy RL algorithms use the concept of "target parameters" when it
# comes to estimate the value of the ``t+1`` state or state-action pair.
# The target parameters are lagged copies of the model parameters. Because
# their predictions mismatch those of the current model configuration, they
# help learning by putting a pessimistic bound on the value being estimated.
# This is a powerful trick (known as "Double Q-Learning") that is ubiquitous
# in similar algorithms.
#
# Functionalizing modules
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# One of the features of torchrl is its usage of functional modules: as the
# same architecture is often used with multiple sets of parameters (e.g.
# trainable and target parameters), we functionalize the modules and isolate
# the various sets of parameters in separate tensordicts.
#
# To this aim, we use :func:`tensordict.nn.get_functional`, which augments
# our modules with some extra feature that make them compatible with parameters
# passed in the ``TensorDict`` format.


def make_model(dummy_env):
    cnn_kwargs = {
        "num_cells": [32, 64, 64],
        "kernel_sizes": [6, 4, 3],
        "strides": [2, 2, 1],
        "activation_class": nn.ELU,
        # This can be used to reduce the size of the last layer of the CNN
        # "squeeze_output": True,
        # "aggregator_class": nn.AdaptiveAvgPool2d,
        # "aggregator_kwargs": {"output_size": (1, 1)},
    }
    mlp_kwargs = {
        "depth": 2,
        "num_cells": [
            64,
            64,
        ],
        "activation_class": nn.ELU,
    }
    net = DuelingCnnDQNet(
        dummy_env.action_spec.shape[-1], 1, cnn_kwargs, mlp_kwargs
    ).to(device)
    net.value[-1].bias.data.fill_(init_bias)

    actor = QValueActor(net, in_keys=["pixels"], spec=dummy_env.action_spec).to(device)
    # init actor: because the model is composed of lazy conv/linear layers,
    # we must pass a fake batch of data through it to instantiate them.
    tensordict = dummy_env.fake_tensordict()
    actor(tensordict)

    # Make functional:
    # here's an explicit way of creating the parameters and buffer tensordict.
    # Alternatively, we could have used `params = make_functional(actor)` from
    # tensordict.nn
    params = TensorDict({k: v for k, v in actor.named_parameters()}, [])
    buffers = TensorDict({k: v for k, v in actor.named_buffers()}, [])
    params = params.update(buffers)
    params = params.unflatten_keys(".")  # creates a nested TensorDict
    factor = get_functional(actor)

    # creating the target parameters is fairly easy with tensordict:
    params_target = params.clone().detach()

    # we wrap our actor in an EGreedyWrapper for data collection
    actor_explore = EGreedyWrapper(
        actor,
        annealing_num_steps=total_frames,
        eps_init=eps_greedy_val,
        eps_end=eps_greedy_val_env,
    )

    return factor, actor, actor_explore, params, params_target


(
    factor,
    actor,
    actor_explore,
    params,
    params_target,
) = make_model(test_env)

###############################################################################
# We represent the parameters and targets as flat structures, but unflattening
# them is quite easy:

params_flat = params.flatten_keys(".")

###############################################################################
# We will be using the adam optimizer:

optim = torch.optim.Adam(list(params_flat.values()), lr, betas=betas)

###############################################################################
# We create a test environment for evaluation of the policy:

test_env = make_env(
    parallel=False, observation_norm_state_dict=observation_norm_state_dict
)
# sanity check:
print(actor_explore(test_env.reset()))

###############################################################################
# Collecting and storing data
# ---------------------------
#
# Replay buffers
# ^^^^^^^^^^^^^^
#
# Replay buffers play a central role in off-policy RL algorithms such as DQN.
# They constitute the dataset we will be sampling from during training.
#
# Here, we will use a regular sampling strategy, although a prioritized RB
# could improve the performance significantly.
#
# We place the storage on disk using
# :class:`torchrl.data.replay_buffers.storages.LazyMemmapStorage` class. This
# storage is created in a lazy manner: it will only be instantiated once the
# first batch of data is passed to it.
#
# The only requirement of this storage is that the data passed to it at write
# time must always have the same shape.

replay_buffer = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(buffer_size),
    prefetch=n_optim,
)

###############################################################################
# Data collector
# ^^^^^^^^^^^^^^
#
# As in `PPO <https://pytorch.org/rl/tutorials/coding_ppo.html>` and
# `DDPG <https://pytorch.org/rl/tutorials/coding_ddpg.html>`, we will be using
# a data collector as a dataloader in the outer loop.
#
# We choose the following configuration: we will be running a series of
# parallel environments synchronously in parallel in different collectors,
# themselves running in parallel but asynchronously.
# The advantage of this configuration is that we can balance the amount of
# compute that is executed in batch with what we want to be executed
# asynchronously. We encourage the reader to experiment how the collection
# speed is impacted by modifying the number of collectors (ie the number of
# environment constructors passed to the collector) and the number of
# environment executed in parallel in each collector (controlled by the
# ``num_workers`` hyperparameter).
#
# When building the collector, we can choose on which device we want the
# environment and policy to execute the operations through the ``device``
# keyword argument. The ``storing_devices`` argument will modify the
# location of the data being collected: if the batches that we are gathering
# have a considerable size, we may want to store them on a different location
# than the device where the computation is happening. For asynchronous data
# collectors such as ours, different storing devices mean that the data that
# we collect won't sit on the same device each time, which is something that
# out training loop must account for. For simplicity, we set the devices to
# the same value for all sub-collectors.

data_collector = MultiaSyncDataCollector(
    # ``num_collectors`` collectors, each with an set of `num_workers` environments being run in parallel
    [
        make_env(
            parallel=True, observation_norm_state_dict=observation_norm_state_dict
        ),
    ]
    * num_collectors,
    policy=actor_explore,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    # this is the default behaviour: the collector runs in ``"random"`` (or explorative) mode
    exploration_mode="random",
    # We set the all the devices to be identical. Below is an example of
    # heterogeneous devices
    devices=[device] * num_collectors,
    storing_devices=[device] * num_collectors,
    # devices=[f"cuda:{i}" for i in range(1, 1 + num_collectors)],
    # storing_devices=[f"cuda:{i}" for i in range(1, 1 + num_collectors)],
    split_trajs=False,
)

###############################################################################
# Training loop of a regular DQN
# ------------------------------
#
# We'll start with a simple implementation of DQN where the returns are
# computed without bootstrapping, i.e.
#
# .. math::
#
#       Q_{t}(s, a) = R(s, a) + \gamma * V_{t+1}(s)
#
# where :math:`Q(s, a)` is the Q-value of the current state-action pair,
# :math:`R(s, a)` is the result of the reward function, and :math:`V(s)` is a
# value function that returns 0 for terminating states.
#
# We store the logs in a defaultdict:

logs_exp1 = defaultdict(list)
prev_traj_count = 0

pbar = tqdm.tqdm(total=total_frames)
for j, data in enumerate(data_collector):
    current_frames = data.numel()
    pbar.update(current_frames)
    data = data.view(-1)

    # We store the values on the replay buffer, after placing them on CPU.
    # When called for the first time, this will instantiate our storage
    # object which will print its content.
    replay_buffer.extend(data.cpu())

    # some logging
    if len(logs_exp1["frames"]):
        logs_exp1["frames"].append(current_frames + logs_exp1["frames"][-1])
    else:
        logs_exp1["frames"].append(current_frames)

    if data["next", "done"].any():
        done = data["next", "done"].squeeze(-1)
        logs_exp1["traj_lengths"].append(
            data["next", "step_count"][done].float().mean().item()
        )

    # check that we have enough data to start training
    if sum(logs_exp1["frames"]) > init_random_frames:
        for _ in range(n_optim):
            # sample from the RB and send to device
            sampled_data = replay_buffer.sample(batch_size)
            sampled_data = sampled_data.to(device, non_blocking=True)

            # collect data from RB
            reward = sampled_data["next", "reward"].squeeze(-1)
            done = sampled_data["next", "done"].squeeze(-1).to(reward.dtype)
            action = sampled_data["action"].clone()

            # Compute action value (of the action actually taken) at time t
            # By default, TorchRL uses one-hot encodings for discrete actions
            sampled_data_out = sampled_data.select(*actor.in_keys)
            sampled_data_out = factor(sampled_data_out, params=params)
            action_value = sampled_data_out["action_value"]
            action_value = (action_value * action.to(action_value.dtype)).sum(-1)
            with torch.no_grad():
                # compute best action value for the next step, using target parameters
                tdstep = step_mdp(sampled_data)
                next_value = factor(
                    tdstep.select(*actor.in_keys),
                    params=params_target,
                )["chosen_action_value"].squeeze(-1)
                exp_value = reward + gamma * next_value * (1 - done)
            assert exp_value.shape == action_value.shape
            # we use MSE loss but L1 or smooth L1 should also work
            error = nn.functional.mse_loss(exp_value, action_value).mean()
            error.backward()

            gv = nn.utils.clip_grad_norm_(list(params_flat.values()), 1)

            optim.step()
            optim.zero_grad()

            # update of the target parameters
            params_target.apply(
                lambda p_target, p_orig: p_orig * tau + p_target * (1 - tau),
                params.detach(),
                inplace=True,
            )

        actor_explore.step(current_frames)

        # Logging
        logs_exp1["grad_vals"].append(float(gv))
        logs_exp1["losses"].append(error.item())
        logs_exp1["values"].append(action_value.mean().item())
        logs_exp1["traj_count"].append(
            prev_traj_count + data["next", "done"].sum().item()
        )
        prev_traj_count = logs_exp1["traj_count"][-1]

        if j % 10 == 0:
            with set_exploration_mode("mode"), torch.no_grad():
                # execute a rollout. The `set_exploration_mode("mode")` has no effect here since the policy is deterministic, but we add it for completeness
                eval_rollout = test_env.rollout(
                    max_steps=10000,
                    policy=actor,
                ).cpu()
            logs_exp1["traj_lengths_eval"].append(eval_rollout.shape[-1])
            logs_exp1["evals"].append(eval_rollout["next", "reward"].sum().item())
            if len(logs_exp1["mavgs"]):
                logs_exp1["mavgs"].append(
                    logs_exp1["evals"][-1] * 0.05 + logs_exp1["mavgs"][-1] * 0.95
                )
            else:
                logs_exp1["mavgs"].append(logs_exp1["evals"][-1])
            logs_exp1["traj_count_eval"].append(logs_exp1["traj_count"][-1])
            pbar.set_description(
                f"error: {error: 4.4f}, value: {action_value.mean(): 4.4f}, test return: {logs_exp1['evals'][-1]: 4.4f}"
            )

    # update policy weights
    data_collector.update_policy_weights_()

###############################################################################
# We write a custom plot function to display the performance of our algorithm
#


def plot(logs, name):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(
        logs["frames"][-len(logs["evals"]) :],
        logs["evals"],
        label="return (eval)",
    )
    plt.plot(
        logs["frames"][-len(logs["mavgs"]) :],
        logs["mavgs"],
        label="mavg of returns (eval)",
    )
    plt.xlabel("frames collected")
    plt.ylabel("trajectory length (= return)")
    plt.subplot(2, 3, 2)
    plt.plot(
        logs["traj_count"][-len(logs["evals"]) :],
        logs["evals"],
        label="return",
    )
    plt.plot(
        logs["traj_count"][-len(logs["mavgs"]) :],
        logs["mavgs"],
        label="mavg",
    )
    plt.xlabel("trajectories collected")
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.plot(logs["frames"][-len(logs["losses"]) :], logs["losses"])
    plt.xlabel("frames collected")
    plt.title("loss")
    plt.subplot(2, 3, 4)
    plt.plot(logs["frames"][-len(logs["values"]) :], logs["values"])
    plt.xlabel("frames collected")
    plt.title("value")
    plt.subplot(2, 3, 5)
    plt.plot(
        logs["frames"][-len(logs["grad_vals"]) :],
        logs["grad_vals"],
    )
    plt.xlabel("frames collected")
    plt.title("grad norm")
    if len(logs["traj_lengths"]):
        plt.subplot(2, 3, 6)
        plt.plot(logs["traj_lengths"])
        plt.xlabel("batches")
        plt.title("traj length (training)")
    plt.savefig(name)
    if is_notebook():
        plt.show()


###############################################################################
# The performance of the policy can be measured as the length of trajectories.
# As we can see on the results of the :func:`plot` function, the performance
# of the policy increases, albeit slowly.
#
# .. code-block:: python
#
#    plot(logs_exp1, "dqn_td0.png")
#
# .. figure:: /_static/img/dqn_td0.png
#    :alt: Cart Pole results with TD(0)
#

print("shutting down")
data_collector.shutdown()
del data_collector

###############################################################################
# DQN with TD(:math:`\lambda`)
# ----------------------------
#
# We can improve the above algorithm by getting a better estimate of the
# return, using not only the next state value but the whole sequence of rewards
# and values that follow a particular step.
#
# TorchRL provides a vectorized version of TD(lambda) named
# :func:`torchrl.objectives.value.functional.vec_td_lambda_advantage_estimate`.
# We'll use this to obtain a target value that the value network will be
# trained to match.
#
# The big difference in this implementation is that we'll store entire
# trajectories and not single steps in the replay buffer. This will be done
# automatically as long as we're not "flattening" the tensordict collected:
# by keeping a shape ``[Batch x timesteps]`` and giving this
# to the RB, we'll be creating a replay buffer of size
# ``[Capacity x timesteps]``.


from torchrl.objectives.value.functional import vec_td_lambda_advantage_estimate

###############################################################################
# We reset the actor parameters:
#

(
    factor,
    actor,
    actor_explore,
    params,
    params_target,
) = make_model(test_env)
params_flat = params.flatten_keys(".")

optim = torch.optim.Adam(list(params_flat.values()), lr, betas=betas)
test_env = make_env(
    parallel=False, observation_norm_state_dict=observation_norm_state_dict
)
print(actor_explore(test_env.reset()))

###############################################################################
# Data: Replay buffer and collector
# ---------------------------------
#
# We need to build a new replay buffer of the appropriate size:
#

max_size = frames_per_batch // num_workers

replay_buffer = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(-(-buffer_size // max_size)),
    prefetch=n_optim,
)

data_collector = MultiaSyncDataCollector(
    [
        make_env(
            parallel=True, observation_norm_state_dict=observation_norm_state_dict
        ),
    ]
    * num_collectors,
    policy=actor_explore,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    exploration_mode="random",
    devices=[device] * num_collectors,
    storing_devices=[device] * num_collectors,
    # devices=[f"cuda:{i}" for i in range(1, 1 + num_collectors)],
    # storing_devices=[f"cuda:{i}" for i in range(1, 1 + num_collectors)],
    split_trajs=False,
)


logs_exp2 = defaultdict(list)
prev_traj_count = 0

###############################################################################
# Training loop
# -------------
#
# There are very few differences with the training loop above:
#
# - The tensordict received by the collector is used as-is, without being
#   flattened (recall the ``data.view(-1)`` above), to keep the temporal
#   relation between consecutive steps.
# - We use :func:`vec_td_lambda_advantage_estimate` to compute the target
#   value.

pbar = tqdm.tqdm(total=total_frames)
for j, data in enumerate(data_collector):
    current_frames = data.numel()
    pbar.update(current_frames)

    replay_buffer.extend(data.cpu())
    if len(logs_exp2["frames"]):
        logs_exp2["frames"].append(current_frames + logs_exp2["frames"][-1])
    else:
        logs_exp2["frames"].append(current_frames)

    if data["next", "done"].any():
        done = data["next", "done"].squeeze(-1)
        logs_exp2["traj_lengths"].append(
            data["next", "step_count"][done].float().mean().item()
        )

    if sum(logs_exp2["frames"]) > init_random_frames:
        for _ in range(n_optim):
            sampled_data = replay_buffer.sample(batch_size // max_size)
            sampled_data = sampled_data.clone().to(device, non_blocking=True)

            reward = sampled_data["next", "reward"]
            done = sampled_data["next", "done"].to(reward.dtype)
            action = sampled_data["action"].clone()

            sampled_data_out = sampled_data.select(*actor.in_keys)
            sampled_data_out = vmap(factor, (0, None))(sampled_data_out, params)
            action_value = sampled_data_out["action_value"]
            action_value = (action_value * action.to(action_value.dtype)).sum(-1, True)
            with torch.no_grad():
                tdstep = step_mdp(sampled_data)
                next_value = vmap(factor, (0, None))(
                    tdstep.select(*actor.in_keys), params
                )
                next_value = next_value["chosen_action_value"]
            error = vec_td_lambda_advantage_estimate(
                gamma,
                lmbda,
                action_value,
                next_value,
                reward,
                done,
            ).pow(2)
            error = error.mean()
            error.backward()

            gv = nn.utils.clip_grad_norm_(list(params_flat.values()), 1)

            optim.step()
            optim.zero_grad()

            # update of the target parameters
            params_target.apply(
                lambda p_target, p_orig: p_orig * tau + p_target * (1 - tau),
                params.detach(),
                inplace=True,
            )

        actor_explore.step(current_frames)

        # Logging
        logs_exp2["grad_vals"].append(float(gv))

        logs_exp2["losses"].append(error.item())
        logs_exp2["values"].append(action_value.mean().item())
        logs_exp2["traj_count"].append(
            prev_traj_count + data["next", "done"].sum().item()
        )
        prev_traj_count = logs_exp2["traj_count"][-1]
        if j % 10 == 0:
            with set_exploration_mode("mode"), torch.no_grad():
                # execute a rollout. The `set_exploration_mode("mode")` has
                # no effect here since the policy is deterministic, but we add
                # it for completeness
                eval_rollout = test_env.rollout(
                    max_steps=10000,
                    policy=actor,
                ).cpu()
            logs_exp2["traj_lengths_eval"].append(eval_rollout.shape[-1])
            logs_exp2["evals"].append(eval_rollout["next", "reward"].sum().item())
            if len(logs_exp2["mavgs"]):
                logs_exp2["mavgs"].append(
                    logs_exp2["evals"][-1] * 0.05 + logs_exp2["mavgs"][-1] * 0.95
                )
            else:
                logs_exp2["mavgs"].append(logs_exp2["evals"][-1])
            logs_exp2["traj_count_eval"].append(logs_exp2["traj_count"][-1])
            pbar.set_description(
                f"error: {error: 4.4f}, value: {action_value.mean(): 4.4f}, test return: {logs_exp2['evals'][-1]: 4.4f}"
            )

    # update policy weights
    data_collector.update_policy_weights_()


###############################################################################
# TD(:math:`\lambda`) performs significantly better than TD(0) because it
# retrieves a much less biased estimate of the state-action value.
#
# .. code-block:: python
#
#    plot(logs_exp2, "dqn_tdlambda.png")
#
# .. figure:: /_static/img/dqn_tdlambda.png
#    :alt: Cart Pole results with TD(lambda)
#


print("shutting down")
data_collector.shutdown()
del data_collector

###############################################################################
# Let's compare the results on a single plot. Because the TD(lambda) version
# works better, we'll have fewer episodes collected for a given number of
# frames (as there are more frames per episode).
#
# **Note**: As already mentioned above, to get a more reasonable performance,
# use a greater value for ``total_frames`` e.g. 500000.


def plot_both():
    frames_td0 = logs_exp1["frames"]
    frames_tdlambda = logs_exp2["frames"]
    evals_td0 = logs_exp1["evals"]
    evals_tdlambda = logs_exp2["evals"]
    mavgs_td0 = logs_exp1["mavgs"]
    mavgs_tdlambda = logs_exp2["mavgs"]
    traj_count_td0 = logs_exp1["traj_count_eval"]
    traj_count_tdlambda = logs_exp2["traj_count_eval"]

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.plot(frames_td0[-len(evals_td0) :], evals_td0, label="return (td0)", alpha=0.5)
    plt.plot(
        frames_tdlambda[-len(evals_tdlambda) :],
        evals_tdlambda,
        label="return (td(lambda))",
        alpha=0.5,
    )
    plt.plot(frames_td0[-len(mavgs_td0) :], mavgs_td0, label="mavg (td0)")
    plt.plot(
        frames_tdlambda[-len(mavgs_tdlambda) :],
        mavgs_tdlambda,
        label="mavg (td(lambda))",
    )
    plt.xlabel("frames collected")
    plt.ylabel("trajectory length (= return)")

    plt.subplot(1, 2, 2)
    plt.plot(
        traj_count_td0[-len(evals_td0) :],
        evals_td0,
        label="return (td0)",
        alpha=0.5,
    )
    plt.plot(
        traj_count_tdlambda[-len(evals_tdlambda) :],
        evals_tdlambda,
        label="return (td(lambda))",
        alpha=0.5,
    )
    plt.plot(traj_count_td0[-len(mavgs_td0) :], mavgs_td0, label="mavg (td0)")
    plt.plot(
        traj_count_tdlambda[-len(mavgs_tdlambda) :],
        mavgs_tdlambda,
        label="mavg (td(lambda))",
    )
    plt.xlabel("trajectories collected")
    plt.legend()

    plt.savefig("dqn.png")


###############################################################################
# .. code-block:: python
#
#    plot_both()
#
# .. figure:: /_static/img/dqn.png
#    :alt: Cart Pole results from the TD(:math:`lambda`) trained policy.
#
# Finally, we generate a new video to check what the algorithm has learnt.
# If all goes well, the duration should be significantly longer than with a
# random rollout.
#
# To get the raw pixels of the rollout, we insert a
# :class:`torchrl.envs.CatTensors` transform that precedes all others and copies
# the ``"pixels"`` key onto a ``"pixels_save"`` key. This is necessary because
# the other transforms that modify this key will update its value in-place in
# the output tensordict.
#

test_env.transform.insert(0, CatTensors(["pixels"], "pixels_save", del_keys=False))
eval_rollout = test_env.rollout(max_steps=10000, policy=actor, auto_reset=True).cpu()

# sphinx_gallery_start_ignore
import imageio

imageio.mimwrite("cartpole.gif", eval_rollout["pixels_save"].numpy(), fps=30)
# sphinx_gallery_end_ignore

del test_env

###############################################################################
# The video of the rollout can be saved using the imageio package:
#
# .. code-block::
#
#   import imageio
#   imageio.mimwrite('cartpole.mp4', eval_rollout["pixels_save"].numpy(), fps=30);
#
# .. figure:: /_static/img/cartpole.gif
#    :alt: Cart Pole results from the TD(:math:`\lambda`) trained policy.

###############################################################################
# Conclusion and possible improvements
# ------------------------------------
#
# In this tutorial we have learnt:
#
# - How to train a policy that read pixel-based states, what transforms to
#   include and how to normalize the data;
# - How to create a policy that picks up the action with the highest value
#   with :class:`torchrl.modules.QValueNetwork`;
# - How to build a multiprocessed data collector;
# - How to train a DQN with TD(:math:`\lambda`) returns.
#
# We have seen that using TD(:math:`\lambda`) greatly improved the performance
# of DQN. Other possible improvements could include:
#
# - Using the Multi-Step post-processing. Multi-step will project an action
#   to the nth following step, and create a discounted sum of the rewards in
#   between. This trick can make the algorithm noticebly less myopic. To use
#   this, simply create the collector with
#
#       from torchrl.data.postprocs.postprocs import MultiStep
#       collector = CollectorClass(..., postproc=MultiStep(gamma, n))
#
#   where ``n`` is the number of looking-forward steps. Pay attention to the
#   fact that the ``gamma`` factor has to be corrected by the number of
#   steps till the next observation when being passed to
#   ``vec_td_lambda_advantage_estimate``:
#
#       gamma = gamma ** tensordict["steps_to_next_obs"]
# - A prioritized replay buffer could also be used. This will give a
#   higher priority to samples that have the worst value accuracy.
# - A distributional loss (see ``torchrl.objectives.DistributionalDQNLoss``
#   for more information).
# - More fancy exploration techniques, such as NoisyLinear layers and such
#   (check ``torchrl.modules.NoisyLinear``, which is fully compatible with the
#   ``MLP`` class used in our Dueling DQN).
