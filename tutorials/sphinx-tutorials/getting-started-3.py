# -*- coding: utf-8 -*-
"""
Get started with data collection and storage
============================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _gs_storage:

.. note:: To run this tutorial in a notebook, add an installation cell
  at the beginning containing:

    .. code-block::

        !pip install tensordict
        !pip install torchrl

"""

#################################
#
# There is no learning without data. In supervised learning, users are
# accustomed to using :class:`~torch.utils.data.DataLoader` and the like
# to integrate data in their training loop.
# Dataloaders are iterable objects that provide you with the data that you will
# be using to train your model.
#
# TorchRL approaches the problem of dataloading in a similar manner, although
# it is surprisingly unique in the ecosystem of RL libraries. TorchRL's
# dataloaders are referred to as ``DataCollectors``. Most of the time,
# data collection does not stop at the collection of raw data,
# as the data needs to be stored temporarily in a buffer
# (or equivalent structure for on-policy sota-implementations) before being consumed
# by the :ref:`loss module <gs_optim>`. This tutorial will explore
# these two classes.
#
# Data collectors
# ---------------
#
# .. _gs_storage_collector:
#
#
# The primary data collector discussed here is the
# :class:`~torchrl.collectors.SyncDataCollector`, which is the focus of this
# documentation. At a fundamental level, a collector is a straightforward
# class responsible for executing your policy within the environment,
# resetting the environment when necessary, and providing batches of a
# predefined size. Unlike the :meth:`~torchrl.envs.EnvBase.rollout` method
# demonstrated in :ref:`the env tutorial <gs_env_ted>`, collectors do not
# reset between consecutive batches of data. Consequently, two successive
# batches of data may contain elements from the same trajectory.
#
# The basic arguments you need to pass to your collector are the size of the
# batches you want to collect (``frames_per_batch``), the length (possibly
# infinite) of the iterator, the policy and the environment. For simplicity,
# we will use a dummy, random policy in this example.

import torch

torch.manual_seed(0)

from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv
from torchrl.envs.utils import RandomPolicy

env = GymEnv("CartPole-v1")
env.set_seed(0)

policy = RandomPolicy(env.action_spec)
collector = SyncDataCollector(env, policy, frames_per_batch=200, total_frames=-1)

#################################
# We now expect that our collector will deliver batches of size ``200`` no
# matter what happens during collection. In other words, we may have multiple
# trajectories in this batch! The ``total_frames`` indicates how long the
# collector should be. A value of ``-1`` will produce a never
# ending collector.
#
# Let's iterate over the collector to get a sense
# of what this data looks like:

for data in collector:
    print(data)
    break

#################################
# As you can see, our data is augmented with some collector-specific metadata
# grouped in a ``"collector"`` sub-tensordict that we did not see during
# :ref:`environment rollouts <gs_env_ted_rollout>`. This is useful to keep track of
# the trajectory ids. In the following list, each item marks the trajectory
# number the corresponding transition belongs to:

print(data["collector", "traj_ids"])

#################################
# Data collectors are very useful when it comes to coding state-of-the-art
# sota-implementations, as performance is usually measured by the capability of a
# specific technique to solve a problem in a given number of interactions with
# the environment (the ``total_frames`` argument in the collector).
# For this reason, most training loops in our examples look like this:
#
#   >>> for data in collector:
#   ...     # your algorithm here
#
#
# Replay Buffers
# --------------
#
# .. _gs_storage_rb:
#
# Now that we have explored how to collect data, we would like to know how to
# store it. In RL, the typical setting is that the data is collected, stored
# temporarily and cleared after a little while given some heuristic:
# first-in first-out or other. A typical pseudo-code would look like this:
#
#   >>> for data in collector:
#   ...     storage.store(data)
#   ...     for i in range(n_optim):
#   ...         sample = storage.sample()
#   ...         loss_val = loss_fn(sample)
#   ...         loss_val.backward()
#   ...         optim.step() # etc
#
# The parent class that stores the data in TorchRL
# is referred to as :class:`~torchrl.data.ReplayBuffer`. TorchRL's replay
# buffers are composable: you can edit the storage type, their sampling
# technique, the writing heuristic or the transforms applied to them. We will
# leave the fancy stuff for a dedicated in-depth tutorial. The generic replay
# buffer only needs to know what storage it has to use. In general, we
# recommend a :class:`~torchrl.data.TensorStorage` subclass, which will work
# fine in most cases. We'll be using
# :class:`~torchrl.data.replay_buffers.LazyMemmapStorage`
# in this tutorial, which enjoys two nice properties: first, being "lazy",
# you don't  need to explicitly tell it what your data looks like in advance.
# Second, it uses :class:`~tensordict.MemoryMappedTensor` as a backend to save
# your data on disk in an efficient way. The only thing you need to know is
# how big you want your buffer to be.

from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer

buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=1000))

#################################
# Populating the buffer can be done via the
# :meth:`~torchrl.data.ReplayBuffer.add` (single element) or
# :meth:`~torchrl.data.ReplayBuffer.extend` (multiple elements) methods. Using
# the data we just collected, we initialize and populate the buffer in one go:

indices = buffer.extend(data)

#################################
# We can check that the buffer now has the same number of elements than what
# we got from the collector:

assert len(buffer) == collector.frames_per_batch

#################################
# The only thing left to know is how to gather data from the buffer.
# Naturally, this relies on the :meth:`~torchrl.data.ReplayBuffer.sample`
# method. Because we did not specify that sampling had to be done without
# repetitions, it is not guaranteed that the samples gathered from our buffer
# will be unique:

sample = buffer.sample(batch_size=30)
print(sample)

#################################
# Again, our sample looks exactly the same as the data we gathered from the
# collector!
#
# Next steps
# ----------
#
# - You can have look at other multirpocessed
#   collectors such as :class:`~torchrl.collectors.collectors.MultiSyncDataCollector` or
#   :class:`~torchrl.collectors.collectors.MultiaSyncDataCollector`.
# - TorchRL also offers distributed collectors if you have multiple nodes to
#   use for inference. Check them out in the
#   :ref:`API reference <ref_collectors>`.
# - Check the dedicated :ref:`Replay Buffer tutorial <rb_tuto>` to know
#   more about the options you have when building a buffer, or the
#   :ref:`API reference <ref_data>` which covers all the features in
#   details. Replay buffers have countless features such as multithreaded
#   sampling, prioritized experience replay, and many more...
# - We left out the capacity of replay buffers to be iterated over for
#   simplicity. Try it out for yourself: build a buffer and indicate its
#   batch-size in the constructor, then try to iterate over it. This is
#   equivalent to calling ``rb.sample()`` within a loop!
#
