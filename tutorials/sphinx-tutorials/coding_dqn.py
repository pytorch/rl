# -*- coding: utf-8 -*-
"""
TorchRL trainer: A DQN example
==============================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

"""

##############################################################################
# TorchRL provides a generic :class:`~torchrl.trainers.Trainer` class to handle
# your training loop. The trainer executes a nested loop where the outer loop
# is the data collection and the inner loop consumes this data or some data
# retrieved from the replay buffer to train the model.
# At various points in this training loop, hooks can be attached and executed at
# given intervals.
#
# In this tutorial, we will be using the trainer class to train a DQN algorithm
# to solve the CartPole task from scratch.
#
# Main takeaways:
#
# - Building a trainer with its essential components: data collector, loss
#   module, replay buffer and optimizer.
# - Adding hooks to a trainer, such as loggers, target network updaters and such.
#
# The trainer is fully customisable and offers a large set of functionalities.
# The tutorial is organised around its construction.
# We will be detailing how to build each of the components of the library first,
# and then put the pieces together using the :class:`~torchrl.trainers.Trainer`
# class.
#
# Along the road, we will also focus on some other aspects of the library:
#
# - how to build an environment in TorchRL, including transforms (e.g. data
#   normalization, frame concatenation, resizing and turning to grayscale)
#   and parallel execution. Unlike what we did in the
#   `DDPG tutorial <https://pytorch.org/rl/tutorials/coding_ddpg.html>`_, we
#   will normalize the pixels and not the state vector.
# - how to design a :class:`~torchrl.modules.QValueActor` object, i.e. an actor
#   that estimates the action values and picks up the action with the highest
#   estimated return;
# - how to collect data from your environment efficiently and store them
#   in a replay buffer;
# - how to use multi-step, a simple preprocessing step for off-policy algorithms;
# - and finally how to evaluate your model.
#
# **Prerequisites**: We encourage you to get familiar with torchrl through the
# `PPO tutorial <https://pytorch.org/rl/tutorials/coding_ppo.html>`_ first.
#
# DQN
# ---
#
# DQN (`Deep Q-Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_) was
# the founding work in deep reinforcement learning.
#
# On a high level, the algorithm is quite simple: Q-learning consists in
# learning a table of state-action values in such a way that, when
# encountering any particular state, we know which action to pick just by
# searching for the one with the highest value. This simple setting
# requires the actions and states to be
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
# We do not aim at giving a SOTA implementation of the algorithm, but rather
# to provide a high-level illustration of TorchRL features in the context
# of this algorithm.

# sphinx_gallery_start_ignore
import tempfile
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import os
import uuid

import torch
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.envs import EnvCreator, ParallelEnv, RewardScaling, StepCounter
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor

from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers.csv import CSVLogger
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)


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
# Let's get started with the various pieces we need for our algorithm:
#
# - An environment;
# - A policy (and related modules that we group under the "model" umbrella);
# - A data collector, which makes the policy play in the environment and
#   delivers training data;
# - A replay buffer to store the training data;
# - A loss module, which computes the objective function to train our policy
#   to maximise the return;
# - An optimizer, which performs parameter updates based on our loss.
#
# Additional modules include a logger, a recorder (executes the policy in
# "eval" mode) and a target network updater. With all these components into
# place, it is easy to see how one could misplace or misuse one component in
# the training script. The trainer is there to orchestrate everything for you!
#
# Building the environment
# ------------------------
#
# First let's write a helper function that will output an environment. As usual,
# the "raw" environment may be too simple to be used in practice and we'll need
# some data transformation to expose its output to the policy.
#
# We will be using five transforms:
#
# - :class:`~torchrl.envs.StepCounter` to count the number of steps in each trajectory;
# - :class:`~torchrl.envs.transforms.ToTensorImage` will convert a ``[W, H, C]`` uint8
#   tensor in a floating point tensor in the ``[0, 1]`` space with shape
#   ``[C, W, H]``;
# - :class:`~torchrl.envs.transforms.RewardScaling` to reduce the scale of the return;
# - :class:`~torchrl.envs.transforms.GrayScale` will turn our image into grayscale;
# - :class:`~torchrl.envs.transforms.Resize` will resize the image in a 64x64 format;
# - :class:`~torchrl.envs.transforms.CatFrames` will concatenate an arbitrary number of
#   successive frames (``N=4``) in a single tensor along the channel dimension.
#   This is useful as a single image does not carry information about the
#   motion of the cartpole. Some memory about past observations and actions
#   is needed, either via a recurrent neural network or using a stack of
#   frames.
# - :class:`~torchrl.envs.transforms.ObservationNorm` which will normalize our observations
#   given some custom summary statistics.
#
# In practice, our environment builder has two arguments:
#
# - ``parallel``: determines whether multiple environments have to be run in
#   parallel. We stack the transforms after the
#   :class:`~torchrl.envs.ParallelEnv` to take advantage
#   of vectorization of the operations on device, although this would
#   technically work with every single environment attached to its own set of
#   transforms.
# - ``obs_norm_sd`` will contain the normalizing constants for
#   the :class:`~torchrl.envs.ObservationNorm` transform.
#


def make_env(
    parallel=False,
    obs_norm_sd=None,
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:
        base_env = ParallelEnv(
            num_workers,
            EnvCreator(
                lambda: GymEnv(
                    "CartPole-v1",
                    from_pixels=True,
                    pixels_only=True,
                    device=device,
                )
            ),
        )
    else:
        base_env = GymEnv(
            "CartPole-v1",
            from_pixels=True,
            pixels_only=True,
            device=device,
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
            ObservationNorm(in_keys=["pixels"], **obs_norm_sd),
        ),
    )
    return env


###############################################################################
# Compute normalizing constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To normalize images, we don't want to normalize each pixel independently
# with a full ``[C, W, H]`` normalizing mask, but with simpler ``[C, 1, 1]``
# shaped set of normalizing constants (loc and scale parameters).
# We will be using the ``reduce_dim`` argument
# of :meth:`~torchrl.envs.ObservationNorm.init_stats` to instruct which
# dimensions must be reduced, and the ``keep_dims`` parameter to ensure that
# not all dimensions disappear in the process:
#


def get_norm_stats():
    test_env = make_env()
    test_env.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    print("state dict of the observation norm:", obs_norm_sd)
    return obs_norm_sd


###############################################################################
# Building the model (Deep Q-network)
# -----------------------------------
#
# The following function builds a :class:`~torchrl.modules.DuelingCnnDQNet`
# object which is a simple CNN followed by a two-layer MLP. The only trick used
# here is that the action values (i.e. left and right action value) are
# computed using
#
# .. math::
#
#    \mathbb{v} = b(obs) + v(obs) - \mathbb{E}[v(obs)]
#
# where :math:`\mathbb{v}` is our vector of action values,
# :math:`b` is a :math:`\mathbb{R}^n \rightarrow 1` function and :math:`v` is a
# :math:`\mathbb{R}^n \rightarrow \mathbb{R}^m` function, for
# :math:`n = \# obs` and :math:`m = \# actions`.
#
# Our network is wrapped in a :class:`~torchrl.modules.QValueActor`,
# which will read the state-action
# values, pick up the one with the maximum value and write all those results
# in the input :class:`tensordict.TensorDict`.
#


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

    # we wrap our actor in an EGreedyWrapper for data collection
    actor_explore = EGreedyWrapper(
        actor,
        annealing_num_steps=total_frames,
        eps_init=eps_greedy_val,
        eps_end=eps_greedy_val_env,
    )

    return actor, actor_explore


###############################################################################
# Collecting and storing data
# ---------------------------
#
# Replay buffers
# ~~~~~~~~~~~~~~
#
# Replay buffers play a central role in off-policy RL algorithms such as DQN.
# They constitute the dataset we will be sampling from during training.
#
# Here, we will use a regular sampling strategy, although a prioritized RB
# could improve the performance significantly.
#
# We place the storage on disk using
# :class:`~torchrl.data.replay_buffers.storages.LazyMemmapStorage` class. This
# storage is created in a lazy manner: it will only be instantiated once the
# first batch of data is passed to it.
#
# The only requirement of this storage is that the data passed to it at write
# time must always have the same shape.


def get_replay_buffer(buffer_size, n_optim, batch_size):
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(buffer_size),
        prefetch=n_optim,
    )
    return replay_buffer


###############################################################################
# Data collector
# ~~~~~~~~~~~~~~
#
# As in `PPO <https://pytorch.org/rl/tutorials/coding_ppo.html>`_ and
# `DDPG <https://pytorch.org/rl/tutorials/coding_ddpg.html>`_, we will be using
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


def get_collector(
    obs_norm_sd,
    num_collectors,
    actor_explore,
    frames_per_batch,
    total_frames,
    device,
):
    data_collector = MultiaSyncDataCollector(
        [
            make_env(parallel=True, obs_norm_sd=obs_norm_sd),
        ]
        * num_collectors,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # this is the default behaviour: the collector runs in ``"random"`` (or explorative) mode
        exploration_mode="random",
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=device,
        storing_device=device,
        split_trajs=False,
        postproc=MultiStep(gamma=gamma, n_steps=5),
    )
    return data_collector


###############################################################################
# Loss function
# -------------
#
# Building our loss function is straightforward: we only need to provide
# the model and a bunch of hyperparameters to the DQNLoss class.
#
# Target parameters
# ~~~~~~~~~~~~~~~~~
#
# Many off-policy RL algorithms use the concept of "target parameters" when it
# comes to estimate the value of the next state or state-action pair.
# The target parameters are lagged copies of the model parameters. Because
# their predictions mismatch those of the current model configuration, they
# help learning by putting a pessimistic bound on the value being estimated.
# This is a powerful trick (known as "Double Q-Learning") that is ubiquitous
# in similar algorithms.
#


def get_loss_module(actor, gamma):
    loss_module = DQNLoss(actor, gamma=gamma, delay_value=True)
    target_updater = SoftUpdate(loss_module)
    return loss_module, target_updater


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
# ~~~~~~~~~

# the learning rate of the optimizer
lr = 2e-3
# weight decay
wd = 1e-5
# the beta parameters of Adam
betas = (0.9, 0.999)
# Optimization steps per batch collected (aka UPD or updates per data)
n_optim = 8

###############################################################################
# DQN parameters
# ~~~~~~~~~~~~~~
# gamma decay factor
gamma = 0.99

###############################################################################
# Smooth target network update decay parameter.
# This loosely corresponds to a 1/tau interval with hard target network
# update
tau = 0.02

###############################################################################
# Data collection and replay buffer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. note::
#   Values to be used for proper training have been commented.
#
# Total frames collected in the environment. In other implementations, the
# user defines a maximum number of episodes.
# This is harder to do with our data collectors since they return batches
# of N collected frames, where N is a constant.
# However, one can easily get the same restriction on number of episodes by
# breaking the training loop when a certain number
# episodes has been collected.
total_frames = 5_000  # 500000

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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# .. note::
#   For fast rendering of the tutorial ``total_frames`` hyperparameter
#   was set to a very low number. To get a reasonable performance, use a greater
#   value e.g. 500000
#

###############################################################################
# Building a Trainer
# ------------------
#
# TorchRL's :class:`~torchrl.trainers.Trainer` class constructor takes the
# following keyword-only arguments:
#
# - ``collector``
# - ``loss_module``
# - ``optimizer``
# - ``logger``: A logger can be
# - ``total_frames``: this parameter defines the lifespan of the trainer.
# - ``frame_skip``: when a frame-skip is used, the collector must be made
#   aware of it in order to accurately count the number of frames
#   collected etc. Making the trainer aware of this parameter is not
#   mandatory but helps to have a fairer comparison between settings where
#   the total number of frames (budget) is fixed but the frame-skip is
#   variable.

stats = get_norm_stats()
test_env = make_env(parallel=False, obs_norm_sd=stats)
# Get model
actor, actor_explore = make_model(test_env)
loss_module, target_net_updater = get_loss_module(actor, gamma)
target_net_updater.init_()

collector = get_collector(
    stats, num_collectors, actor_explore, frames_per_batch, total_frames, device
)
optimizer = torch.optim.Adam(
    loss_module.parameters(), lr=lr, weight_decay=wd, betas=betas
)
exp_name = f"dqn_exp_{uuid.uuid1()}"
tmpdir = tempfile.TemporaryDirectory()
logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir.name)
warnings.warn(f"log dir: {logger.experiment.log_dir}")

###############################################################################
# We can control how often the scalars should be logged. Here we set this
# to a low value as our training loop is short:

log_interval = 500

trainer = Trainer(
    collector=collector,
    total_frames=total_frames,
    frame_skip=1,
    loss_module=loss_module,
    optimizer=optimizer,
    logger=logger,
    optim_steps_per_batch=n_optim,
    log_interval=log_interval,
)

###############################################################################
# Registering hooks
# ~~~~~~~~~~~~~~~~~
#
# Registering hooks can be achieved in two separate ways:
#
# - If the hook has it, the :meth:`~torchrl.trainers.TrainerHookBase.register`
#   method is the first choice. One just needs to provide the trainer as input
#   and the hook will be registered with a default name at a default location.
#   For some hooks, the registration can be quite complex: :class:`~torchrl.trainers.ReplayBufferTrainer`
#   requires 3 hooks (``extend``, ``sample`` and ``update_priority``) which
#   can be cumbersome to implement.
buffer_hook = ReplayBufferTrainer(
    get_replay_buffer(buffer_size, n_optim, batch_size=batch_size),
    flatten_tensordicts=True,
)
buffer_hook.register(trainer)
weight_updater = UpdateWeights(collector, update_weights_interval=1)
weight_updater.register(trainer)
recorder = Recorder(
    record_interval=100,  # log every 100 optimization steps
    record_frames=1000,  # maximum number of frames in the record
    frame_skip=1,
    policy_exploration=actor_explore,
    environment=test_env,
    exploration_mode="mode",
    log_keys=[("next", "reward")],
    out_keys={("next", "reward"): "rewards"},
    log_pbar=True,
)
recorder.register(trainer)

###############################################################################
# - Any callable (including :class:`~torchrl.trainers.TrainerHookBase`
#   subclasses) can be registered using :meth:`~torchrl.trainers.Trainer.register_op`.
#   In this case, a location must be explicitly passed (). This method gives
#   more control over the location of the hook but it also requires more
#   understanding of the Trainer mechanism.
#   Check the `trainer documentation <https://pytorch.org/rl/reference/trainers.html>`_
#   for a detailed description of the trainer hooks.
#
trainer.register_op("post_optim", target_net_updater.step)

###############################################################################
# We can log the training rewards too. Note that this is of limited interest
# with CartPole, as rewards are always 1. The discounted sum of rewards is
# maximised not by getting higher rewards but by keeping the cart-pole alive
# for longer.
# This will be reflected by the `total_rewards` value displayed in the
# progress bar.
#
log_reward = LogReward(log_pbar=True)
log_reward.register(trainer)

###############################################################################
# .. note::
#   It is possible to link multiple optimizers to the trainer if needed.
#   In this case, each optimizer will be tied to a field in the loss
#   dictionary.
#   Check the :class:`~torchrl.trainers.OptimizerHook` to learn more.
#
# Here we are, ready to train our algorithm! A simple call to
# ``trainer.train()`` and we'll be getting our results logged in.
#
trainer.train()

###############################################################################
# We can now quickly check the CSVs with the results.


def print_csv_files_in_folder(folder_path):
    """
    Find all CSV files in a folder and prints the first 10 lines of each file.

    Args:
        folder_path (str): The relative path to the folder.

    """
    csv_files = []
    output_str = ""
    for dirpath, _, filenames in os.walk(folder_path):
        for file in filenames:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, file))
    for csv_file in csv_files:
        output_str += f"File: {csv_file}\n"
        with open(csv_file, "r") as f:
            for i, line in enumerate(f):
                if i == 10:
                    break
                output_str += line.strip() + "\n"
        output_str += "\n"
    print(output_str)


print_csv_files_in_folder(logger.experiment.log_dir)

###############################################################################
# Conclusion and possible improvements
# ------------------------------------
#
# In this tutorial we have learned:
#
# - How to write a Trainer, including building its components and registering
#   them in the trainer;
# - How to code a DQN algorithm, including how to create a policy that picks
#   up the action with the highest value with
#   :class:`~torchrl.modules.QValueNetwork`;
# - How to build a multiprocessed data collector;
#
# Possible improvements to this tutorial could include:
#
# - A prioritized replay buffer could also be used. This will give a
#   higher priority to samples that have the worst value accuracy.
#   Learn more on the
#   `replay buffer section <https://pytorch.org/rl/reference/data.html#composable-replay-buffers>`_
#   of the documentation.
# - A distributional loss (see :class:`~torchrl.objectives.DistributionalDQNLoss`
#   for more information).
# - More fancy exploration techniques, such as :class:`~torchrl.modules.NoisyLinear` layers and such.
