# -*- coding: utf-8 -*-
"""
Get started with your onw first training loop
=============================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _gs_first_training:

"""

#################################
# Time to wrap up everything we've learned in this Getting Started series.
#
# In this tutorial, we will be writing the most basic training loop there is
# using only components we have presented in the previous lessons.
#
# We'll be using DQN with a CartPole environment as a prototypical example.
#
# We will be voluntarily keeping the verbosity to its minimum, only linking
# each section to the related tutorial.
#
# Building the environment
# ------------------------
#
# We'll be using a gym environment with a :class:`~torchrl.envs.StepCounter`
# transform. These features are presented in
# :ref:`the environment tutorial <gs_env_ted>`.
#
import torch

torch.manual_seed(0)

import time

from torchrl.envs import GymEnv, StepCounter, TransformedEnv

env = TransformedEnv(GymEnv("CartPole-v1"), StepCounter())
env.set_seed(0)

from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

#################################
# Designing a policy
# ------------------
#
# The next step is to build our policy. We'll be making a regular, deterministic
# version used within the :ref:`loss module <gs_optim>` and during
# :ref:`exploration <gs_logging>`, and one augmented by an exploration module
# for :ref:`inference <gs_storage>`.

from torchrl.modules import EGreedyModule, MLP, QValueModule

value_mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[64, 64])
value_net = Mod(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = Seq(value_net, QValueModule(env.action_spec))
# TODO: explain annealing in other tuto
exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=100_000, eps_init=0.5
)
policy_explore = Seq(policy, exploration_module)


#################################
# Data Collector and replay buffer
# --------------------------------
#
# Here comes the data part: we need a
# :ref:`data collector <gs_storage_collector>` to easily get batches of data
# and a :ref:`replay buffer <gs_storage_rb>` to store that data for training.
#

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

init_rand_steps = 5000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(env, policy, frames_per_batch=frames_per_batch, total_frames=-1, init_random_frames=init_rand_steps)
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

from torch.optim import Adam

#################################
# Loss module and optimizer
# -------------------------
#

from torchrl.objectives import DQNLoss

loss = DQNLoss(value_network=policy, action_space=env.action_spec)
optim = Adam(loss.parameters(), lr=0.02)

#################################
# Logger
# ------
# TODO

from torchrl._utils import logger as torchrl_logger

#################################
# Training loop
# -------------
#
# Instead of fixing a specific number of iterations to run, we will keep on
# training the network until it reaches a certain performance (arbitrarily
# defined as 200 steps in the environment -- with CartPole, success is defined
# as having longer trajectories).
#

total_count = 0
total_episodes = 0
t0 = time.time()
for i, data in enumerate(collector):
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            exploration_module.step(data.numel())
            if i % 10:
                torchrl_logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_length > 200:
        break

t1 = time.time()

torchrl_logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)
