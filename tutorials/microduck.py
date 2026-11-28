"""
MicroDuck: tasks as data, rewards as a registry, one policy for all of them
==========================================================================

**Author**: `TorchRL contributors <https://github.com/pytorch/rl>`_

.. _microduck_tuto:

MicroDuck is a small open-hardware biped by Pollen Robotics. TorchRL ships it
as :class:`~torchrl.envs.MicroDuckEnv`, a MuJoCo locomotion environment whose
tasks (standing, walking at a commanded speed, sidestepping, hopping) are rows
of a tensorclass rather than subclasses or flags. Every simulator of a batch
holds one row, picked at reset, and the reward is a registry of terms that
each row weights. This tutorial walks through that design and the tools
around it: how to select tasks, how to add a reward term, how to standardize
advantages within each task, how to switch simulation backends, how to run
the closed-form gait controller that ships with the example, and how to film
it.

What you will learn
-------------------

- how a :class:`~torchrl.envs.MicroDuckTask` library is built from presets and
  stacked with :func:`torch.stack`;
- how the env picks one task per simulator at reset, and how
  :class:`~torchrl.envs.MicroDuckTaskSampler` pins or mixes tasks;
- how the same task code runs on the native MuJoCo bindings, on MJX and on
  ``mujoco-torch``;
- how to run the closed-form gait policy of the MicroDuck example and record
  it with :class:`~torchrl.record.VideoRecorder`;
- how to standardize advantages within each task with
  :class:`~torchrl.objectives.value.GAE` and its ``group_key``;
- how to register a reward term of your own with
  :meth:`~torchrl.envs.MicroDuckEnv.register_reward`;
- where the end-to-end PPO recipe that learns all of this lives.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

import torchrl
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import MicroDuckEnv, MicroDuckTaskSampler, TransformedEnv
from torchrl.modules import MLP
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder

if importlib.util.find_spec("mujoco") is None:
    raise ImportError("This tutorial requires the `mujoco` Python package.")

# %%
# Assets and a fast mode
# ----------------------
#
# The walking MJCF and its meshes live in the ``microduck_rl`` repository and
# are not vendored in TorchRL. ``download=True`` fetches the pinned commit into
# ``~/.cache/torchrl/microduck`` the first time (or set ``MICRODUCK_RL_ROOT``
# to a checkout). ``TORCHRL_TUTORIALS_FAST=1`` shortens the rollouts for the
# docs build.

TUTORIAL_FAST = os.environ.get("TORCHRL_TUTORIALS_FAST", "0") == "1"
ROLLOUT_STEPS = 100 if TUTORIAL_FAST else 300
RENDER_WIDTH, RENDER_HEIGHT = (320, 240) if TUTORIAL_FAST else (480, 360)

# %%
# A task is data
# --------------
#
# :class:`~torchrl.envs.MicroDuckTask` is a tensorclass: the planar command box
# ``(vx, vy)`` to track, the warm start and reset noise, the gait clock, one
# weight per registered reward term, the term parameters, a sampling weight
# and a name. The presets on :class:`~torchrl.envs.MicroDuckEnv` fill every
# field and accept overrides by name, so a task is one call.

library = [
    MicroDuckEnv.standing_task(),
    MicroDuckEnv.tracking_task(
        0.2, warm_start_velocity=(0.05, 0.25), warm_start_fraction=0.5
    ),
    MicroDuckEnv.sidestep_task(0.15),
    MicroDuckEnv.jump_task(weight=2.0),
]
tasks = torch.stack(library)
print(tasks.shape, list(tasks.name))
print("command boxes:", tasks.command_low.tolist(), tasks.command_high.tolist())

# %%
# Stacking is the structural validation: every task carries the full weight
# vector over :attr:`~torchrl.envs.MicroDuckEnv.REWARD_TERMS`, so the presets
# differ only in which terms they switch on. The standing row zeroes the gait
# terms; the jump row turns the hop terms on and the vertical-velocity cost
# off.

terms = list(MicroDuckEnv.REWARD_TERMS)
for name, weights in zip(tasks.name, tasks.reward_weights):
    active = {term: round(float(w), 2) for term, w in zip(terms, weights) if w != 0}
    print(f"{name:14s} {active}")

# %%
# One row per simulator
# ---------------------
#
# The env takes the library and every simulator of the batch holds one row for
# the duration of an episode. At reset, the rows being reset read ``task_id``
# from the reset TensorDict when it is there, and otherwise draw one with the
# tasks' ``weight`` field. The observation carries the command and the task id
# (``task_id`` is also in the ``state_spec``, which is what lets a
# :class:`~torchrl.envs.TransformedEnv` forward it at reset).

env = MicroDuckEnv(
    download=True, backend="mujoco", tasks=tasks, action_scale=1.0, seed=0
)
print(env.observation_spec["task_id"], env.observation_spec["command"])

reset = env.reset(TensorDict(task_id=torch.tensor([[2]]), batch_size=[1]))
print("pinned task:", reset["task_id"].item(), reset["command"].tolist())

rollout = env.rollout(20, tensordict=reset, auto_reset=False)
print("held for the episode:", rollout["task_id"].unique().tolist())

# %%
# :class:`~torchrl.envs.MicroDuckTaskSampler` writes ``task_id`` at reset for
# you: with weights when the mixture should differ from the library's (a
# curriculum), or with :meth:`~torchrl.envs.MicroDuckTaskSampler.fixed` to give
# every simulator its own task, which is how the example films four tasks side
# by side.

mixed = TransformedEnv(env, MicroDuckTaskSampler([0.0, 1.0, 1.0, 0.0], seed=0))
drawn = [mixed.reset()["task_id"].item() for _ in range(6)]
print("drawn from the walking rows only:", drawn)
mixed.close()

# %%
# Backends
# --------
#
# The task is written once against :class:`~torchrl.envs.MujocoEnv`, so the
# backend is a constructor argument. ``"mujoco"`` runs the official C bindings,
# one simulator per process with :class:`~torchrl.envs.ParallelEnv` when
# ``num_envs > 1`` (the CPU fallback used here); ``"mjx"`` and
# ``"mujoco-torch"`` vectorize ``num_envs`` simulators inside the simulator and
# are how the env is meant to run at scale on an accelerator, with
# ``compile_step=True`` compiling the ``mujoco-torch`` physics step. The
# observation, action, reward and termination are identical on all three.
#
# .. code-block:: python
#
#    MicroDuckEnv(download=True, backend="mujoco", num_envs=16, parallel=True)
#    MicroDuckEnv(download=True, backend="mjx", num_envs=1024, device="cuda")
#    MicroDuckEnv(download=True, num_envs=1024, device="cuda", compile_step=True)
#
# MicroDuck's upstream training environments are written for ``mjlab``;
# :class:`~torchrl.envs.MJLabWrapper` runs those directly, which is a
# different task definition from the one in this tutorial.

for backend in ("mjx", "mujoco-torch"):
    module = {"mjx": "mujoco.mjx", "mujoco-torch": "mujoco_torch"}[backend]
    print(backend, "available" if importlib.util.find_spec(module) else "not installed")

# %%
# The closed-form gait
# --------------------
#
# The MicroDuck example ships a hand-written walking controller,
# ``MicroDuckGaitActor``: a bilateral phase oscillator on the env's gait clock
# drives the hip, knee, ankle and lateral targets while a proportional
# controller on the torso pitch keeps the robot upright, all read from the
# observation. It is a :class:`~tensordict.nn.TensorDictModuleBase`, so it is a
# policy like any other: ``env.rollout(steps, gait)`` walks. It lives in
# ``examples/microduck/heuristic_gait.py``, next to the PPO script that can
# use it as a prior (``policy.from_prior=true``).

EXAMPLES_DIR = Path(torchrl.__file__).resolve().parents[1] / "examples" / "microduck"
sys.path.insert(0, str(EXAMPLES_DIR))
from heuristic_gait import gait_metrics, MicroDuckGaitActor  # noqa: E402

gait = MicroDuckGaitActor()
gait_env = MicroDuckEnv(
    download=True,
    backend="mujoco",
    tasks=MicroDuckEnv.tracking_task(0.03, **gait.config.task_kwargs()),
    diagnostics=True,
    seed=0,
)
gait_rollout = gait_env.rollout(ROLLOUT_STEPS, gait, break_when_any_done=True)
metrics = gait_metrics(gait_rollout)
print(
    f"survived={bool(metrics['survived'])} forward_speed={float(metrics['forward_speed']):+.3f} m/s "
    f"swing phases: left={int(metrics['left_swing_phases'])} right={int(metrics['right_swing_phases'])}"
)
gait_env.close()

# %%
# Film it
# -------
#
# ``from_pixels=True`` adds a rendered ``pixels`` observation, and a
# :class:`~torchrl.record.VideoRecorder` appended to the env collects the
# frames of every step. With a logger it writes the video where the logger
# lives (W&B, TensorBoard, an mp4 through :class:`~torchrl.record.CSVLogger`);
# here :meth:`~torchrl.record.VideoRecorder.to_animation` turns the frames
# into an animation that Sphinx-Gallery embeds below.

recorder = VideoRecorder(logger=None, tag="microduck_gait", skip=2, make_grid=False)
video_env = TransformedEnv(
    MicroDuckEnv(
        download=True,
        backend="mujoco",
        tasks=MicroDuckEnv.tracking_task(0.03, **gait.config.task_kwargs()),
        from_pixels=True,
        render_width=RENDER_WIDTH,
        render_height=RENDER_HEIGHT,
        camera_id=-1,
        seed=0,
    ),
    recorder,
)
video_env.rollout(ROLLOUT_STEPS, gait, break_when_any_done=True)
gait_animation = recorder.to_animation(
    title="Closed-form MicroDuck gait", interval=40, clear=True
)
video_env.close()

# %%
# Advantages standardized within each task
# ----------------------------------------
#
# A multi-task batch mixes rewards of very different scales: a walking row
# collects the gait terms, the standing row does not. Standardizing the
# advantages over the whole batch lets the high-variance tasks set the scale
# and shrinks the others' learning signal. :class:`~torchrl.objectives.value.GAE`
# (and the TD estimators) take ``group_key``, the tensordict entry of an
# integer id per batch element, and standardize within its groups instead.
# Here two pinned episodes, one per task, form the batch.

value_net = TensorDictModule(
    MLP(in_features=MicroDuckEnv.OBSERVATION_DIM, out_features=1, num_cells=[64]),
    in_keys=["observation"],
    out_keys=["state_value"],
)
episodes = []
for task_id in (0, 1):
    start = env.reset(TensorDict(task_id=torch.tensor([[task_id]]), batch_size=[1]))
    episodes.append(
        env.rollout(
            ROLLOUT_STEPS, tensordict=start, auto_reset=False, break_when_any_done=False
        )[0]
    )
batch = torch.stack(episodes)  # (2 tasks, T)

per_task = GAE(
    gamma=0.99,
    lmbda=0.95,
    value_network=value_net,
    average_gae=True,
    group_key="task_id",
)
global_norm = GAE(gamma=0.99, lmbda=0.95, value_network=value_net, average_gae=True)
with torch.no_grad():
    grouped = per_task(batch.clone())["advantage"]
    pooled = global_norm(batch.clone())["advantage"]
for task_id, name in enumerate(tasks.name[:2]):
    print(
        f"{name:14s} per-task: mean={grouped[task_id].mean():+.3f} std={grouped[task_id].std():.3f} | "
        f"pooled: mean={pooled[task_id].mean():+.3f} std={pooled[task_id].std():.3f}"
    )

# %%
# The example's ``ppo.per_task_advantage`` option is exactly
# ``GAE(average_gae=True, group_key="task_id")`` with the loss's own
# normalization switched off.

# %%
# Designing a reward term
# -----------------------
#
# Every step the env computes one features TensorDict (body-frame velocity,
# uprightness, joint errors, contacts, foot heights, base height, gait phase,
# command, previous action). A reward term is a function of those features and
# of the per-env task parameters, registered once on the class; from then on
# every task carries a weight for it (zero by default) and any parameters it
# declared. Here a term that rewards keeping the heading, with a task that
# switches it on.


@MicroDuckEnv.register_reward("heading", weight=0.0, heading_std=0.5)
def heading(features, params):
    yaw_rate = features["angular_velocity"][..., 2]
    return torch.exp(-yaw_rate.square() / params["heading_std"].square())


steady = MicroDuckEnv.tracking_task(
    0.2, reward_weights={"heading": 1.0}, heading_std=0.3
)
print(
    "heading weight:",
    float(steady.reward_weights[-1]),
    "(the newest term is the last entry)",
)
print("params:", sorted(steady.params.keys()))

# %%
# Tasks built before a registration have a shorter weight vector and are
# rejected by the env, so register terms first; ``diagnostics=True`` then
# exposes every weighted term under ``diagnostic_reward_<name>``.

env.close()
diag_env = MicroDuckEnv(
    download=True,
    backend="mujoco",
    tasks=steady,
    diagnostics=True,
    action_scale=1.0,
    seed=0,
)
diag_rollout = diag_env.rollout(20)
print(
    "heading term per step:",
    diag_rollout["next", "diagnostic_reward_heading"].mean().item(),
    "| total reward:",
    diag_rollout["next", "reward"].mean().item(),
)
diag_env.close()

# %%
# Training end to end
# -------------------
#
# ``examples/microduck/ppo_mujoco.py`` trains a recurrent PPO policy on a task
# library from scratch: a GRU actor-critic conditioned on the task id, whole
# episodes replayed from a :class:`~torchrl.data.TensorDictReplayBuffer`, one
# :class:`~torchrl.collectors.Evaluator` per task, the per-task GAE above, and
# unified checkpoints that ``rlrender`` reads. The library is the Hydra list
# ``env.tasks``, one preset per entry:
#
# .. code-block:: bash
#
#    python examples/microduck/ppo_mujoco.py env.download=true \
#        'env.tasks=[{preset:standing_task},{preset:tracking_task,speed:0.2},{preset:tracking_task,speed:-0.2},{preset:sidestep_task,speed:0.15},{preset:sidestep_task,speed:-0.15},{preset:jump_task,weight:3.0}]' \
#        evaluation.video.interval=4 logger.entity=YOUR_ENTITY
#
# ``evaluation.video.interval`` logs, every fourth evaluation, a 2x2 video of
# four tasks filmed in parallel, one simulator per tile pinned with
# :meth:`~torchrl.envs.MicroDuckTaskSampler.fixed`. With the shipped defaults
# the policy walks both ways within a few million transitions, sidesteps by
# about six million and hops shortly after; ``policy.from_prior=true`` starts
# from the closed-form gait instead, a quick debugging start that only knows
# forward walking.

# %%
# Conclusion and further reading
# ------------------------------
#
# Keeping the task as data buys three things: a batch of simulators can run
# different tasks with no code path per task, a new behaviour is a new row
# (weights, parameters, command box) rather than a new environment, and the
# tools around the env (samplers, per-task advantages, evaluators, videos)
# only ever need the task id.
#
# .. seealso::
#
#    - :class:`~torchrl.envs.MicroDuckEnv` and :class:`~torchrl.envs.MicroDuckTask`
#      for every field and preset.
#    - :class:`~torchrl.envs.MujocoEnv` for the base class and its backends.
#    - :class:`~torchrl.objectives.value.GAE` for ``group_key``.
#    - :ref:`rlrender_tuto` for rendering checkpoints outside training.
#    - ``examples/microduck/README.md`` for the training recipe and the results
#      of the multi-task runs.
