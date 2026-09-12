"""
MicroDuck: train skills, then compose behaviors
===================================================

**Author**: `TorchRL contributors <https://github.com/pytorch/rl>`_

.. _microduck_skills_tuto:

Our duck needs two kinds of practice: moving its legs, and choosing where to
go. We'll teach it to stand, walk, sidestep and jump, then give it a destination.
A second policy will learn when to use each of those skills to get there.
Both policies learn with :class:`~torchrl.trainers.algorithms.PPOTrainer`.

You can start here, or explore the robot and its tasks first in :doc:`microduck`.

What you will learn
-------------------

- Teach one policy several ways to move.
- Reuse those skills in a new task without retraining the walker.
- Train a second policy to choose skills that bring the duck closer to its goal.

Allow 10–15 minutes to follow along. We'll try a little training at each stage,
then load saved policies to see what more practice can achieve. You won't need
to wait for a duck to learn to walk before trying the navigation task.

Watch the composed behavior
---------------------------

Here's where we're headed: a duck using its learned skills to approach a
waypoint at ``(0.5, 0.3)`` metres. You can play the recording now, before running
any code. The waypoint isn't drawn, and the duck returns to the start between
episodes. The :doc:`integration tutorial <microduck>` shows each skill on its own.

The videos and trained policies are available in
`torchrl/microduck-skills <https://huggingface.co/torchrl/microduck-skills>`_.
"""

from __future__ import annotations

from IPython.display import Video

Video(
    url="https://huggingface.co/torchrl/microduck-skills/resolve/"
    "43ffe3b725b6a6853ba8e4a54deb06ae17606e5e/videos/navigation.mp4",
    width=640,
    html_attributes='controls muted loop playsinline preload="metadata" '
    'style="max-width: 100%"',
)

# %%
# Run the tutorial
# ----------------
#
# To follow along, use a TorchRL checkout with ``mujoco``, ``huggingface_hub``
# and the ``utils`` extra installed. The robot assets download on the first run.

import functools as ft
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchrl
from huggingface_hub import hf_hub_download
from torchrl.checkpoint import Checkpoint
from torchrl.envs import MicroDuckController, MicroDuckEnv
from torchrl.envs.transforms import ClosedLoopMultiAction
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.render import load_checkpoint
from torchrl.trainers.algorithms import PPOTrainer

# The example shares task and network definitions with the full training run.
REPO_ROOT = Path(torchrl.__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from examples.microduck.ppo_mujoco import (  # noqa: E402
    make_env,
    make_models,
    make_tasks,
)
from examples.microduck.train_skills import (  # noqa: E402
    load_walker,
    make_navigation_models,
    SKILL_PRESETS,
    WaypointMicroDuck,
)

# %%
# Start with a little practice
# -----------------------------
#
# These short runs let us try both stages in a few moments. Learning a steady
# gait takes much longer; we'll pick up from a trained walker when we get there.

fast = os.environ.get("TORCHRL_TUTORIALS_FAST", "0") == "1"
low_frames = 64 if fast else 1024
high_frames = 64 if fast else 512
epochs = 1 if fast else 4
torch.manual_seed(0)
torch.set_num_threads(1)

# %%
# 1. Define the skills and low-level policy
# -----------------------------------------
#
# A :class:`~torchrl.envs.MicroDuckTask` contains command ranges, reward weights
# and gait parameters. The shared preset list defines these six tasks, in order:
#
# .. code-block:: python
#
#    SKILL_PRESETS = [
#        {"preset": "standing_task"},
#        {"preset": "tracking_task", "speed": 0.2},
#        {"preset": "tracking_task", "speed": -0.2},
#        {"preset": "sidestep_task", "speed": 0.15},
#        {"preset": "sidestep_task", "speed": -0.15},
#        {"preset": "jump_task", "weight": 3.0},
#    ]
#
# At reset, the environment samples a task and holds it for the episode.
# ``make_models`` builds a task embedding and GRU shared by a Gaussian actor
# and a value head. The actor reads ``observation`` and ``task_id`` and writes
# 14 normalized joint actions. Its recurrent state travels in the TensorDict.

skill_tasks = torch.stack(make_tasks(SKILL_PRESETS))
low_env = make_env(
    {
        "backend": "mujoco",
        "device": "cpu",
        "num_envs": 1,
        "parallel": False,
        "download": True,
        "action_scale": 1.0,
        "tasks": SKILL_PRESETS,
        "max_episode_steps": 32 if fast else 500,
    }
)
walker, low_critic = make_models(low_env, hidden_size=32, initial_policy_scale=1.0)

# %%
# 2. Train the low-level policy
# -----------------------------
#
# Let's give the duck its first practice session with
# :meth:`~torchrl.trainers.algorithms.PPOTrainer.from_env`. Two choices matter
# here: keep consecutive steps together so the GRU can use its memory, and
# normalize advantages separately for each task. Otherwise, tasks with larger
# reward variation can drown out the learning signal from the others.

low_trainer = PPOTrainer.from_env(
    low_env,
    actor=walker,
    critic=low_critic,
    total_frames=low_frames,
    frames_per_batch=64 if fast else 256,
    minibatch_size=64,
    sub_traj_len=64,
    gae_kwargs={"group_key": "task_id", "average_gae": True},
    loss_kwargs={"entropy_coeff": 0.01},
    num_epochs=epochs,
    progress_bar=False,
)
low_trainer.train()

# %%
# Reuse a saved walker
# --------------------
#
# A few updates won't make an accomplished walker. We've saved one that has
# practiced for ten million steps in
# `torchrl/microduck-skills <https://huggingface.co/torchrl/microduck-skills>`_.
# It stayed upright in all 48 ten-second evaluation episodes. It still drifts
# and doesn't always match the requested speed, as the skill videos show.
# Our next policy will have to work with the movements it actually learned.
#
# ``load_walker`` restores the trained network and its task library, then freezes
# its weights. Keeping the original task order matters: skill 1 must still mean
# the same thing to the walker. We also keep its action scale. To try your own
# walker, set ``MICRODUCK_WALKER_CHECKPOINT`` to its checkpoint path.

checkpoint_path = os.environ.get("MICRODUCK_WALKER_CHECKPOINT") or hf_hub_download(
    repo_id="torchrl/microduck-skills",
    filename="walker.ckpt",
    revision="4191d7d25c4fd58a5c6e6395fcf8217459fdd073",
)
checkpoint = load_checkpoint(checkpoint_path, weights_only=True)
walker, skill_tasks = load_walker(checkpoint)
action_scale = checkpoint["config"]["env"]["action_scale"]

# %%
# 3. Give the walker a new task
# -----------------------------
#
# Now give the duck somewhere to go. ``WaypointMicroDuck`` tells it where the
# goal is relative to its position, and which way it's facing. Getting closer
# earns a reward, reaching the waypoint earns a bonus, and falling costs it::
#
#    reward = 10 * (old_distance - distance) + arrived - fallen - 0.001
#
# The episode ends when the duck arrives or falls.
#
# ``MicroDuckController`` adapts skill indices to the walker's original inputs.
# ``ClosedLoopMultiAction`` holds that decision for up to five physical steps,
# recomputing joint actions from fresh observations at every step.

make_task_env = ft.partial(
    WaypointMicroDuck,
    backend="mujoco",
    download=True,
    tasks=MicroDuckEnv.standing_task(),
    action_scale=action_scale,
    max_episode_steps=64 if fast else 500,
    seed=0,
)
controller = MicroDuckController(
    walker,
    skill_tasks,
    group_key=None,
    reset_key=None,
    control_period_s=0.02,
)
training_env = ClosedLoopMultiAction.from_env(
    make_task_env(),
    controller,
    steps=5,
    reward_aggregation="sum",
)
check_env_specs(training_env)

# %%
# The action spec now describes a skill index, rather than joint targets.
# The adapter slices the original 56-value robot observation for the walker;
# the high-level policy sees the six extra waypoint/orientation values too.
# The controller owns the walker's GRU state and gait clock under ``_controller``.

assert training_env.action_key == "skill"
assert training_env.full_action_spec["skill"].n == 6
assert training_env.observation_spec["observation"].shape[-1] == 62

# %%
# 4. Train the skill selector
# ---------------------------
#
# ``make_navigation_models`` creates two ordinary 64–64 MLPs: an actor with
# six categorical logits, and a critic with one value output. Their interfaces
# are ``observation -> skill`` and ``observation -> state_value``.
#
# PPO sees only this actor. The walker belongs to the environment and is absent
# from the high-level optimizer. Its inference is deterministic, with gradients
# disabled. The new actor has no recurrent state, so we can shuffle individual
# transitions. ``gamma=0.99`` discounts **one high-level decision**, whose reward
# is the sum of the physical-step rewards executed during that decision.

high_actor, high_critic = make_navigation_models(training_env)
high_trainer = PPOTrainer.from_env(
    training_env,
    actor=high_actor,
    critic=high_critic,
    total_frames=high_frames,
    frames_per_batch=32 if fast else 128,
    minibatch_size=64,
    loss_kwargs={"entropy_coeff": 0.01},
    gamma=0.99,
    num_epochs=epochs,
    progress_bar=False,
)
high_trainer.train()

# %%
# 5. Roll out the composed policy
# -------------------------------
#
# Let's see how a practiced skill selector handles the task. Load the one
# trained with our walker for one million decisions. It reached this waypoint
# in 26 of 32 evaluation episodes (81.25%), with a mean final distance of 9.3 cm.
# This does not establish navigation to arbitrary goals. The
# `model repository <https://huggingface.co/torchrl/microduck-skills>`_ includes
# both final videos, evaluation metrics and training configuration.
#
# A selector depends on its walker. When using your own walker checkpoint,
# keep the selector you just trained instead of loading this published pair.

if not os.environ.get("MICRODUCK_WALKER_CHECKPOINT"):
    navigation_path = hf_hub_download(
        repo_id="torchrl/microduck-skills",
        filename="navigation.ckpt",
        revision="6330ae68b8cf00cda62feb661e50df327f12ebcd",
    )
    Checkpoint(policy=high_actor).load(navigation_path)

# %%
# Give the duck up to ten seconds to reach the waypoint, and plot its distance
# after each skill choice. A successful trip should bring the curve below
# 5 cm, our arrival threshold.

evaluation_env = ClosedLoopMultiAction.from_env(
    make_task_env(max_episode_steps=500),
    controller,
    steps=5,
)
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    trajectory = evaluation_env.rollout(
        100,
        high_actor,
        break_when_any_done=True,
    )
evaluation_env.close()
distance = trajectory["next", "observation"][..., -6:-4].norm(dim=-1)
fig, ax = plt.subplots()
ax.plot(distance.flatten().cpu())
ax.set(xlabel="High-level decision", ylabel="Distance to goal (m)")
fig.tight_layout()
plt.show()

# %%
# Run a full experiment locally
# -----------------------------
#
# To train your own walker and skill selector, give them more time to practice.
# The companion script runs both stages and evaluates the result::
#
#    python -m examples.microduck.train_skills --num-envs 16 \
#        --low-level-frames 10000000 --high-level-frames 1000000 \
#        --output-dir ~/microduck-training
#
# Resume an interrupted low-level stage with ``--resume`` and the same worker
# count. To reuse a walker (or resume its high-level stage), supply its checkpoint::
#
#    python -m examples.microduck.train_skills \
#        --walker-checkpoint ~/microduck-training/walker.ckpt \
#        --output-dir ~/microduck-training --resume
#
# Check ``skills.json`` to see whether the duck stays upright and follows its
# commands, and ``navigation.json`` to see how often it reaches the goal.
# ``walker.ckpt`` saves the policy and its task library for your next experiment.

# %%
# Reuse the deployment for 5-vs-5 football
# ----------------------------------------
#
# Given a football environment exposing ten agents under ``"agents"``, the
# same weights can control every player, with independent recurrent state:
#
# .. code-block:: python
#
#    from torchrl.envs import microduck_skill_env
#
#    football_training_env = microduck_skill_env(
#        football_env, walker, skill_tasks,
#        group_key="agents", steps=5, control_period_s=0.02,
#    )
#    # For 64 matches, 10 players each:
#    # action: ("agents", "skill")                 [64, 10]
#    # state:  ("agents", "_controller", ...)      [64, 10, ...]
#
# Each player's observation must begin with the MicroDuck observation layout;
# ``fallen`` is its raw respawn signal. The preset resets that player's state
# without resetting its neighbours and accumulates fall reports per decision.
# The football environment provides physics, teams and rewards. Team policies,
# critics and opponents are ordinary multi-agent training choices; see
# :doc:`/tutorials/multiagent_ppo` for a complete PPO example.

# %%
# Conclusion
# ----------
#
# A task-conditioned policy can become part of a new environment. The high-level
# actor chooses skills, and the controller translates them into feedback-based
# joint actions. Both levels use the same PPO trainer. For another robot, replace
# the MicroDuck adapter with :class:`~torchrl.modules.LowLevelController` and an
# adapter matching that policy's inputs; the execution transform stays the same.
#
# Further reading
# ---------------
#
# - :doc:`microduck` covers task sampling, custom rewards and simulator backends.
# - :class:`~torchrl.envs.MicroDuckEnv` and :class:`~torchrl.envs.MicroDuckTask`
#   document task presets, reward terms and simulator backends.
# - :class:`~torchrl.modules.LowLevelController` documents batching, routing and state.
# - :meth:`~torchrl.trainers.algorithms.PPOTrainer.from_env` documents PPO options.
# - :class:`~torchrl.objectives.value.GAE` documents task-wise advantage normalization.
