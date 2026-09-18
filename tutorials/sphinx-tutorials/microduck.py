"""
MicroDuck: tasks, rewards and simulation
============================================

**Author**: `TorchRL contributors <https://github.com/pytorch/rl>`_

.. _microduck_tuto:

MicroDuck is a small open-hardware biped by Pollen Robotics. Let's get it moving.
We'll choose a few tasks, look at what the robot sees, and adjust what it gets
rewarded for. Then we'll try a walking controller that needs no training.

In :class:`~torchrl.envs.MicroDuckEnv`, standing, walking, sidestepping and jumping
share one environment. Changing the behavior starts with changing its commands
and reward weights.

What you will learn
-------------------

- Build a task library and choose which task each simulator runs.
- Inspect observations, actions and reward diagnostics.
- Build the recurrent, task-conditioned policy used to train a skill library.
- Add a reward term and run the supplied gait controller.
- Deploy a frozen skill artifact inside a high-level environment.
- Choose a simulator backend and record a rollout.

Allow 10–15 minutes to read and try the tutorial on a CPU. The full PPO recipe
linked below uses the same environment and policy objects.

Watch the learned skills
------------------------

This recorded grid streams from
`torchrl/microduck-skills <https://huggingface.co/torchrl/microduck-skills>`_.
The rows show standing / forward / backward, sidestep left / sidestep right /
hop forward, then turn left / turn right / hop in place. The camera follows
position while keeping its orientation fixed, so the turning skills visibly
change heading. Head pitch is labeled in every tile.

This nine-skill prior survived all 288 held-out ten-second episodes. Both hops
repeatedly take off and land, with 43.8% and 44.0% airtime; forward hopping
reaches 0.285 m/s. In-place hopping still drifts at 0.044 m/s. The
`model card <https://huggingface.co/torchrl/microduck-skills>`_ records the
evaluation, training history and limitations. These are learned policies;
the gait controller we will run here is hand-written.

You can play the recording now, before running any code.
"""

from __future__ import annotations

from IPython.display import Video

Video(
    url="https://huggingface.co/torchrl/microduck-skills/resolve/"
    "01ebcefca08850231edc0eb428a0151474559a85/"
    "priors/nine-skills-20260913/videos/skills.mp4",
    width=800,
    html_attributes='controls muted loop playsinline preload="metadata" '
    'style="max-width: 100%"',
)

# %%
# Set up the simulator
# --------------------
#
# To follow along, use a TorchRL checkout with ``mujoco`` and the ``utils`` extra
# installed. The robot assets download on the first run.

import os
import sys
from pathlib import Path

import torch
import torchrl
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import (
    Compose,
    InitTracker,
    MicroDuckEnv,
    MicroDuckSkillEnv,
    MicroDuckTaskSampler,
    TransformedEnv,
)
from torchrl.envs.utils import check_env_specs
from torchrl.modules import get_primers_from_module
from torchrl.modules.tensordict_module.zoo import MicroDuckSkillPolicy, MicroDuckSkills

fast = os.environ.get("TORCHRL_TUTORIALS_FAST", "0") == "1"
rollout_steps = 100 if fast else 300
torch.manual_seed(0)
torch.set_num_threads(1)

# %%
# A task is data
# --------------
#
# A :class:`~torchrl.envs.MicroDuckTask` is a tensorclass containing a planar
# command box ``(vx, vy)``, reset noise, warm-start settings, a gait clock,
# reward weights and parameters. Presets fill all fields and accept overrides.
# Stack them into a library: each simulator will hold one row per episode.

library = [
    MicroDuckEnv.standing_task(),
    MicroDuckEnv.tracking_task(
        0.2, warm_start_velocity=(0.05, 0.25), warm_start_fraction=0.5
    ),
    MicroDuckEnv.sidestep_task(0.15),
    MicroDuckEnv.jump_task(weight=2.0),
]
tasks = torch.stack(library)
tasks.command_low, tasks.command_high

# %%
# Every row contains one weight per registered reward term. The standing preset
# disables stepping rewards; the jumping preset enables hopping rewards and
# disables the vertical-velocity cost. Compare three columns below: alternating
# foot contacts (``phase_contact``), jumping, and vertical-velocity cost.
# The rows are standing, walking, sidestepping, and jumping, in library order.

terms = list(MicroDuckEnv.REWARD_TERMS)
columns = [terms.index(name) for name in ("phase_contact", "jump", "lin_vel_z")]
tasks.reward_weights[:, columns]

# %%
# Observations, actions and episode tasks
# ---------------------------------------
#
# The native MuJoCo backend runs on CPU. One simulator has batch size ``[1]``.
# Its policy interface is:
#
# .. list-table::
#    :header-rows: 1
#
#    * - Key
#      - Shape
#      - Meaning
#    * - ``observation``
#      - ``[1, 56]``
#      - Gravity, velocities, command, joint state, gait clock, previous action
#    * - ``task_id``
#      - ``[1, 1]``
#      - Index into the task library
#    * - ``command``
#      - ``[1, 2]``
#      - Target forward and lateral velocities in metres per second
#    * - ``action``
#      - ``[1, 14]``
#      - Normalized joint-target offsets around the standing pose
#
# Actions are applied at 50 Hz; ``action_scale`` controls their physical range.
# Explicitly supply ``task_id`` at reset to select row 2 (sidestepping).

env = MicroDuckEnv(
    download=True, backend="mujoco", tasks=tasks, action_scale=1.0, seed=0
)
check_env_specs(env)
reset = env.reset(
    TensorDict(task_id=torch.tensor([[2]]), batch_size=[1]), set_state=True
)
rollout = env.rollout(20, tensordict=reset, auto_reset=False, break_when_any_done=True)
env.close()
rollout["task_id"].unique(), reset["command"]

# %%
# The task is held for the episode. Without an explicit index, resets sample
# using the tasks' ``weight`` fields. The random actions above only demonstrate
# the interface; they do not produce a useful gait.
#
# A :class:`~torchrl.envs.MicroDuckTaskSampler` can replace those probabilities
# for a curriculum. This mixture samples only walking and sidestepping:

mixed = TransformedEnv(
    MicroDuckEnv(download=True, backend="mujoco", tasks=tasks, seed=0),
    MicroDuckTaskSampler([0.0, 1.0, 1.0, 0.0], seed=0),
)
drawn = [mixed.reset()["task_id"].item() for _ in range(6)]
mixed.close()
drawn

# %%
# Let's put two ducks side by side: one walking forward, the other sidestepping.
# Pin a task index to each simulator so they keep their assignments after resets.

paired = TransformedEnv(
    MicroDuckEnv(
        download=True, backend="mujoco", num_envs=2, parallel=False, tasks=tasks
    ),
    MicroDuckTaskSampler.fixed([1, 2]),
)
paired_reset = paired.reset()
paired.close()
paired_reset["task_id"], paired_reset["command"]

# %%
# Build the policy that learns the skills
# ---------------------------------------
#
# :class:`~torchrl.modules.tensordict_module.zoo.MicroDuckSkillPolicy` is the
# recurrent TensorDict policy used by the published artifact. It reads the
# observation and ``task_id`` directly, so the task library controls both the
# reward and the learned task embedding. ``InitTracker`` supplies ``is_init``;
# the primer derived from the policy carries its GRU state between steps.

skill_policy = MicroDuckSkillPolicy(
    hidden_size=128,
    num_tasks=tasks.shape[0],
    observation_dim=MicroDuckEnv.OBSERVATION_DIM,
    num_actions=MicroDuckEnv.NUM_JOINTS,
)
training_env = TransformedEnv(
    MicroDuckEnv(download=True, backend="mujoco", tasks=tasks, seed=0),
    Compose(InitTracker(), get_primers_from_module(skill_policy)),
)
check_env_specs(training_env)
policy_output = skill_policy(training_env.reset())
training_env.close()
policy_output["action"].shape, policy_output["next", "recurrent_state"].shape

# %%
# The actor is now ready for a collector and PPO loss; no MicroDuck-specific
# wrapper is needed during low-level training. The complete recurrent PPO
# recipe is ``examples/microduck/ppo_mujoco.py``. After training, package this
# policy with the same ``tasks`` and ``action_scale`` as a
# :class:`~torchrl.modules.tensordict_module.zoo.MicroDuckSkills` artifact.
#
# %%
# Design a reward term
# --------------------
#
# Each step computes a features TensorDict: body velocity, uprightness, joint
# errors, contacts, foot heights, gait phase and more. A reward term maps those
# features and the task's parameters to one value per simulator.
#
# Here we reward a small yaw rate. We give a subclass its own registries so this
# experiment leaves other MicroDuck environments and saved task libraries usable.
# Register terms **before** constructing that class's tasks: registration adds
# an entry to every task's weight vector.


class HeadingMicroDuck(MicroDuckEnv):
    REWARD_TERMS = MicroDuckEnv.REWARD_TERMS.copy()
    REWARD_PARAMS = MicroDuckEnv.REWARD_PARAMS.copy()


@HeadingMicroDuck.register_reward("heading", weight=0.0, heading_std=0.5)
def heading(features: TensorDictBase, params: TensorDictBase) -> torch.Tensor:
    yaw_rate = features["angular_velocity"][..., 2]
    return torch.exp(-yaw_rate.square() / params["heading_std"].square())


steady = HeadingMicroDuck.tracking_task(
    0.2, reward_weights={"heading": 1.0}, heading_std=0.3
)

# %%
# The new term defaults to weight zero; ``steady`` switches it on and narrows
# its tolerance. By default terms are rates multiplied by the control period;
# one-off rewards such as fall penalties use ``per_second=False``.
# ``diagnostics=True`` exposes each weighted contribution separately, making it
# possible to see whether the reward you intended is the reward being collected.

diag_env = HeadingMicroDuck(
    download=True, backend="mujoco", tasks=steady, diagnostics=True, seed=0
)
diagnostics = diag_env.rollout(20)["next"]
diag_env.close()
{
    "heading contribution": diagnostics["diagnostic_reward_heading"].mean().item(),
    "total reward": diagnostics["reward"].mean().item(),
}

# %%
# Run the supplied gait controller
# --------------------------------
#
# ``MicroDuckGaitActor`` is a hand-written controller in the examples directory.
# A phase oscillator drives the legs, and torso-pitch feedback helps balance.
# It reads the same observations as a learned policy and works with the ordinary
# :meth:`~torchrl.envs.EnvBase.rollout` method. No training is needed.

REPO_ROOT = Path(torchrl.__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from examples.microduck.heuristic_gait import (  # noqa: E402
    gait_metrics,
    MicroDuckGaitActor,
)

gait = MicroDuckGaitActor()
gait_task = MicroDuckEnv.tracking_task(0.03, **gait.config.task_kwargs())
gait_env = MicroDuckEnv(
    download=True, backend="mujoco", tasks=gait_task, diagnostics=True, seed=0
)
gait_rollout = gait_env.rollout(rollout_steps, gait, break_when_any_done=True)
gait_env.close()
gait_metrics(gait_rollout).to_dict()

# %%
# Inspect survival, forward speed and left/right swing phases. These diagnostics
# help distinguish actual walking from staying upright or sliding. The controller
# is useful for checking the simulator and reward design; the training tutorial
# learns a broader task-conditioned skill library.
#
# Turn a skill policy into environment dynamics
# ----------------------------------------------
#
# A task-conditioned low-level policy can perform several locomotion skills;
# it is not itself the policy that solves a football match or another game.
# :class:`~torchrl.modules.tensordict_module.zoo.MicroDuckSkills` keeps that
# policy together with the ordered task library that gives each ``task_id`` its
# meaning and with the motor scale used during training.
#
# :class:`~torchrl.envs.MicroDuckSkillEnv` then makes the frozen skill policy
# part of the environment. A separate high-level policy acts on this new
# environment by choosing a skill. During each high-level transition, the
# controller recomputes joint targets from fresh observations:
#
# .. code-block:: text
#
#    high-level policy --skill id--> MicroDuckSkillEnv
#                                      |
#                                      v
#                           MicroDuckSkillController
#                                      |
#                         observation + task_id
#                                      v
#                              skill_policy --joint target--> physics
#
# Load the published artifact, build the joint-level task, and explicitly
# promote it to the skill-level MDP. ``group_key=None`` is appropriate here
# because this is a single duck at the root; multi-duck games use an ``agents``
# group instead.

skills = MicroDuckSkills.from_pretrained()
skill_task = MicroDuckEnv.tracking_task(0.2)
joint_env = MicroDuckEnv(
    download=True,
    backend="mujoco",
    tasks=skill_task,
    action_scale=skills.action_scale,
    seed=0,
)
skill_env = MicroDuckSkillEnv.from_env(
    joint_env,
    skills,
    skill_ids=[0, 1, 2],
    control_steps_per_decision=5,
    group_key=None,
)
check_env_specs(skill_env)
skill_rollout = skill_env.rollout(4)
skill_env.close()
skill_rollout["skill"], skill_rollout["next", "reward"]

# %%
# The high-level rollout has four decisions but may contain up to twenty
# physical steps. The exact low-level policy, task-id meanings and action scale
# remain encapsulated in ``skills``; the high-level policy only sees the
# skill-level action and observation specs.
#
# Choose a backend
# ----------------
#
# The task and policy interfaces stay the same across backends. Use the native
# CPU backend for this notebook. In a standalone training script, native workers
# can run in parallel; MJX and ``mujoco-torch`` instead batch the physics on an
# accelerator. These alternatives need their respective optional dependencies:
#
# .. code-block:: python
#
#    # CPU worker processes (standalone script):
#    MicroDuckEnv(download=True, backend="mujoco", num_envs=16, parallel=True)
#    # Accelerator batches:
#    MicroDuckEnv(download=True, backend="mjx", num_envs=1024, device="cuda")
#    MicroDuckEnv(
#        download=True, backend="mujoco-torch", num_envs=1024,
#        device="cuda", compile_step=True,
#    )
#
# MicroDuck's upstream ``mjlab`` tasks can also run through
# :class:`~torchrl.envs.MJLabWrapper`; those have their own task definitions.
#
# Record your own rollout (optional)
# ----------------------------------
#
# The video at the top plays without rendering. To record a new experiment,
# enable pixel observations and attach a :class:`~torchrl.record.VideoRecorder`.
# Run this optional snippet in a notebook with local MuJoCo rendering available:
#
# .. code-block:: python
#
#    from IPython.display import HTML
#    from torchrl.record import VideoRecorder
#
#    recorder = VideoRecorder(logger=None, tag="gait", skip=2, make_grid=False)
#    video_env = TransformedEnv(
#        MicroDuckEnv(
#            download=True, backend="mujoco", tasks=gait_task,
#            from_pixels=True, camera_id=-1, seed=0,
#            render_width=480, render_height=360,
#        ),
#        recorder,
#    )
#    video_env.rollout(100, gait, break_when_any_done=True)
#    animation = recorder.to_animation(interval=40, clear=True)
#    video_env.close()
#    HTML(animation.to_jshtml())
#
# With a logger, ``recorder.dump()`` can instead send video to W&B, TensorBoard
# or disk through :class:`~torchrl.record.CSVLogger`.
#
# Conclusion
# ----------
#
# Tasks are rows of data: a command, reward weights and parameters. Samplers
# decide who runs each row; diagnostics show what happened. The same interface
# supports the supplied gait, a learned skill policy, and batched simulators.
# A follow-up tutorial trains that low-level policy with PPO; a high-level
# policy can then learn when to choose each frozen skill in a new task.
#
# Further reading
# ---------------
#
# - :class:`~torchrl.envs.MicroDuckTask` documents all task fields.
# - :class:`~torchrl.envs.MicroDuckEnv` documents presets and reward terms.
# - :class:`~torchrl.modules.tensordict_module.zoo.MicroDuckSkills` documents
#   loading and packaging the deployable skill artifact.
# - :class:`~torchrl.envs.MicroDuckSkillEnv` documents the skill-level MDP.
# - :class:`~torchrl.envs.MujocoEnv` documents simulator backends.
# - :doc:`rlrender` covers rendering checkpoints outside training.
