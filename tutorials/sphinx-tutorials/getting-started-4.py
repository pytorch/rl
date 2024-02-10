# -*- coding: utf-8 -*-
"""
Get started with logging
========================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _gs_logging:

.. note:: To run this tutorial in a notebook, add an installation cell
  at the beginning containing:

    .. code-block::

        !pip install tensordict
        !pip install torchrl

"""

#####################################
# The final chapter of this series before we orchestrate everything in a
# training script is to learn about logging.
#
# Loggers
# -------
#
# Logging is crucial for reporting your results to the outside world and for
# you to check that your algorithm is learning properly. TorchRL has several
# loggers that interface with custom backends such as
# wandb (:class:`~torchrl.record.loggers.wandb.WandbLogger`),
# tensorboard (:class:`~torchrl.record.loggers.tensorboard.TensorBoardLogger`) or a lightweight and
# portable CSV logger (:class:`~torchrl.record.loggers.csv.CSVLogger`) that you can use
# pretty much everywhere.
#
# Loggers are located in the ``torchrl.record`` module and the various classes
# can be found in the :ref:`API reference <ref_loggers>`.
#
# We tried to keep the loggers APIs as similar as we could, given the
# differences in the underlying backends. While execution of the loggers will
# mostly be interchangeable, their instantiation can differ.
#
# Usually, building a logger requires
# at least an experiment name and possibly a logging directory and other
# hyperapameters.
#

from torchrl.record import CSVLogger

logger = CSVLogger(exp_name="my_exp")

#####################################
# Once the logger is instantiated, the only thing left to do is call the
# logging methods! For example, :meth:`~torchrl.record.CSVLogger.log_scalar`
# is used in several places across the training examples to log values such as
# reward, loss value or time elapsed for executing a piece of code.

logger.log_scalar("my_scalar", 0.4)

#####################################
# Recording videos
# ----------------
#
# Finally, it can come in handy to record videos of a simulator. Some
# environments (e.g., Atari games) are already rendered as images whereas
# others require you to create them as such. Fortunately, in most common cases,
# rendering and recording videos isn't too difficult.
#
# Let's first see how we can create a Gym environment that outputs images
# alongside its observations. :class:`~torchrl.envs.GymEnv` accept two keywords
# for this purpose: ``from_pixels=True`` will make the env ``step`` function
# write a ``"pixels"`` entry containing the images corresponding to your
# observations, and the ``pixels_only=False`` will indicate that you want the
# observations to be returned as well.
#

from torchrl.envs import GymEnv

env = GymEnv("CartPole-v1", from_pixels=True, pixels_only=False)

print(env.rollout(max_steps=3))

from torchrl.envs import TransformedEnv

#####################################
# We now have built an environment that renders images with its observations.
# To record videos, we will need to combine that environment with a recorder
# and the logger (the logger providing the backend to save the video).
# This will happen within a transformed environment, like the one we saw in
# the :ref:`first tutorial <gs_env_ted>`.

from torchrl.record import VideoRecorder

recorder = VideoRecorder(logger, tag="my_video")
record_env = TransformedEnv(env, recorder)

#####################################
# When running this environment, all the ``"pixels"`` entries will be saved in
# a local buffer and dumped in a video on demand (it is important that you
# call this method when appropriate):

rollout = record_env.rollout(max_steps=3)
# Uncomment this line to save the video on disk:
# recorder.dump()

#####################################
# In this specific case, the video format can be chosen when instantiating
# the CSVLogger.
#
# This is all we wanted to cover in the getting started tutorial.
# You should now be ready to code your
# :ref:`first training loop with TorchRL <gs_first_training>`!
#
