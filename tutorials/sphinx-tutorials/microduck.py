"""
MicroDuck: train skills, then choose how to use them
====================================================

**Author**: `TorchRL contributors <https://github.com/pytorch/rl>`_

.. _microduck_tuto:

A low-level policy controls MicroDuck's joints. A high-level policy chooses
which skill it should perform. We will train both with
:class:`~torchrl.trainers.algorithms.PPOTrainer`, using the same walker inside
a new task: approach a waypoint.

What you will learn
-------------------

- Train a recurrent policy conditioned on a library of locomotion tasks.
- Deploy it with :class:`~torchrl.envs.MicroDuckController` and
  :class:`~torchrl.envs.transforms.ClosedLoopMultiAction`.
- Train a high-level skill selector with ordinary PPO, then run a rollout.

Allow 10–20 minutes to read and try this tutorial. It uses one CPU simulator
and short training budgets; these demonstrate learning updates and deployment,
but do not produce a competent walker. The full training launcher at the end
uses the same components with larger budgets, parallel workers and evaluation.

Run this notebook from a TorchRL checkout with ``mujoco`` and the ``utils``
extra installed. ``download=True`` fetches the pinned robot assets into
``~/.cache/torchrl/microduck`` on the first run.
"""

from __future__ import annotations

import functools as ft
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchrl
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
# Small budgets for an interactive run
# ------------------------------------
#
# Both the script and generated notebook execute top to bottom. One simulator
# means no worker processes and no ``__main__`` guards. Documentation builds
# set ``TORCHRL_TUTORIALS_FAST=1`` and execute 64 steps per training stage.

fast = os.environ.get("TORCHRL_TUTORIALS_FAST", "0") == "1"
low_frames = 64 if fast else 1024
high_frames = 64 if fast else 512
epochs = 1 if fast else 4
torch.set_num_threads(1)
torch.manual_seed(0)

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
# :meth:`~torchrl.trainers.algorithms.PPOTrainer.from_env` supplies the collector,
# PPO loss, Adam optimizer, GAE and minibatches. We keep the two recurrent-policy
# choices explicit: sample consecutive 64-step windows, and normalize advantages
# separately within each task. A walking reward should not set the advantage
# scale for a jumping task.

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
walker.eval().requires_grad_(False)
walker.zero_grad(set_to_none=True)

# %%
# Reuse a saved walker
# --------------------
#
# The short update above demonstrates training. To deploy weights from a full
# run, set ``MICRODUCK_WALKER_CHECKPOINT`` to its ``walker.ckpt`` before executing
# the notebook or script. This also works in a documentation build: ``fast``
# changes the training budget, not which checkpoint is loaded.
#
# ``load_walker`` reconstructs the recorded network and restores the ordered
# task library: skill indices address learned embeddings, so reordering the
# tasks would change their meaning. We also reuse the saved action scale.
# The checkpoint can be a local training output or a file downloaded from a
# model repository; see the companion example's checkpoint-sharing instructions.

checkpoint_path = os.environ.get("MICRODUCK_WALKER_CHECKPOINT")
action_scale = 1.0
if checkpoint_path:
    checkpoint = load_checkpoint(checkpoint_path, weights_only=True)
    walker, skill_tasks = load_walker(checkpoint)
    action_scale = checkpoint["config"]["env"]["action_scale"]

# %%
# 3. Give the walker a new task
# -----------------------------
#
# ``WaypointMicroDuck`` reuses MicroDuck's physics and adds six observation
# values: displacement to the goal and robot orientation. Its reward is progress
# toward ``(0.5, 0.3)`` metres, plus an arrival bonus and a fall penalty::
#
#    reward = 10 * (old_distance - distance) + arrived - fallen - 0.001
#
# The episode ends on arrival or a fall. The task class contains observation,
# reward and termination definitions; it has no policy-execution loop.
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
# Training closes its environment, so evaluation creates a fresh instance.
# The rollout records high-level skill decisions; each decision invokes the
# walker repeatedly inside the environment. Plot distance to the waypoint to
# see what the composed policy actually did. The short run is a pipeline check;
# successful navigation needs trained skills and a longer high-level run.

evaluation_env = ClosedLoopMultiAction.from_env(
    make_task_env(),
    controller,
    steps=5,
)
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    trajectory = evaluation_env.rollout(
        20 if fast else 100,
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
# The companion launcher shares these task and model definitions. It trains a
# larger GRU using parallel native MuJoCo workers, logs CSV metrics, saves
# resumable trainer state, and evaluates every skill before deployment::
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
# ``walker.ckpt`` stores architecture, weights and the ordered task definitions.
# ``skills.json`` measures survival and command tracking; ``navigation.json``
# measures arrival rate and final distance. Use these evaluations to assess
# learned behavior. Larger budgets alone do not establish that training succeeded.

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
# - :class:`~torchrl.envs.MicroDuckEnv` and :class:`~torchrl.envs.MicroDuckTask`
#   document task presets, reward terms and simulator backends.
# - :class:`~torchrl.modules.LowLevelController` documents batching, routing and state.
# - :meth:`~torchrl.trainers.algorithms.PPOTrainer.from_env` documents PPO options.
# - :class:`~torchrl.objectives.value.GAE` documents task-wise advantage normalization.
