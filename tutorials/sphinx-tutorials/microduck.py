"""
MicroDuck: train low-level skills and deploy a high-level policy
================================================================

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
it. We then train or load a recurrent skill policy and deploy it behind a
high-level PPO actor that learns waypoint navigation.

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
- how to train and reload a recurrent low-level policy;
- how to deploy skills with :class:`~torchrl.envs.MicroDuckController` and
  :class:`~torchrl.envs.transforms.ClosedLoopMultiAction`;
- how to train a high-level skill selector and reuse the deployment over
  multiple agents.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import torchrl
from omegaconf import OmegaConf
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.data import Composite, Unbounded
from torchrl.envs import (
    MicroDuckController,
    MicroDuckEnv,
    MicroDuckTaskSampler,
    TransformedEnv,
)
from torchrl.envs.transforms import ClosedLoopMultiAction
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.render import load_checkpoint

REPO_ROOT = Path(torchrl.__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "microduck"
sys.path.insert(0, str(REPO_ROOT))
from examples.microduck.heuristic_gait import (  # noqa: E402
    gait_metrics,
    MicroDuckGaitActor,
)
from examples.microduck.ppo_mujoco import (  # noqa: E402
    make_env,
    make_models,
    make_render_policy,
    make_tasks,
    save_checkpoint,
    train_ppo,
)

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

env = MicroDuckEnv(
    download=True, backend="mujoco", tasks=tasks, action_scale=1.0, seed=0
)
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
# Train the low-level skills
# --------------------------
#
# The low-level actor maps ``(observation, task_id, recurrent_state)`` to 14
# joint targets. ``examples/microduck/ppo_mujoco.py`` trains it with recurrent
# PPO, complete-episode replay and the per-task GAE described above.
#
# Start a full training run from a TorchRL checkout. This command trains one
# policy on six tasks and saves both its parameters and the configuration that
# defines the task-id order:
#
# .. code-block:: bash
#
#    python examples/microduck/ppo_mujoco.py env.download=true \
#        env.backend=mujoco env.parallel=true env.num_envs=16 \
#        'env.tasks=[{preset:standing_task},{preset:tracking_task,speed:0.2},{preset:tracking_task,speed:-0.2},{preset:sidestep_task,speed:0.15},{preset:sidestep_task,speed:-0.15},{preset:jump_task,weight:3.0}]' \
#        ppo.total_transitions=10000000 \
#        evaluation.best_checkpoint_path=microduck_skills.ckpt \
#        logger.backend=csv
#
# Evaluate the individual skills before deploying them: survival, command
# tracking and foot contacts matter more than a finite PPO loss. See the
# example README for locomotion recipes and evaluation results.
#
# To use that checkpoint for the rest of this tutorial:
#
# .. code-block:: bash
#
#    MICRODUCK_WALKER_CHECKPOINT=microduck_skills.ckpt \
#        MICRODUCK_HIGH_LEVEL_FRAMES=1000000 \
#        python tutorials/sphinx-tutorials/microduck.py
#
# With no checkpoint supplied, the following block runs a small recurrent PPO
# update and round-trips its checkpoint. This checks the complete pipeline;
# a few hundred transitions do not produce a trained walker. Fast mode
# shortens both training stages further.

walker_checkpoint = os.environ.get("MICRODUCK_WALKER_CHECKPOINT")
if walker_checkpoint:
    payload = load_checkpoint(walker_checkpoint)
else:
    low_cfg = OmegaConf.load(EXAMPLES_DIR / "config.yaml")
    low_cfg.env.update(
        backend="mujoco",
        device="cpu",
        num_envs=1,
        parallel=False,
        download=True,
        max_episode_steps=32 if TUTORIAL_FAST else 64,
        tasks=[
            {"preset": "standing_task"},
            {"preset": "tracking_task", "speed": 0.2},
            {"preset": "tracking_task", "speed": -0.2},
            {"preset": "sidestep_task", "speed": 0.15},
            {"preset": "sidestep_task", "speed": -0.15},
            {"preset": "jump_task", "weight": 3.0},
        ],
    )
    low_cfg.policy.hidden_size = 32
    low_cfg.policy.initial_policy_scale = 1.0
    policy_kwargs = {
        "hidden_size": 32,
        "policy_head": "gaussian",
        "initial_policy_scale": 1.0,
    }
    low_env = make_env(low_cfg.env)
    low_actor, low_critic = make_models(low_env, **policy_kwargs)
    try:
        low_history = train_ppo(
            low_env,
            low_actor,
            low_critic,
            total_transitions=64 if TUTORIAL_FAST else 512,
            transitions_per_update=64 if TUTORIAL_FAST else 256,
            max_episode_steps=low_cfg.env.max_episode_steps,
            epochs=1 if TUTORIAL_FAST else 2,
            minibatch_trajectories=2,
            per_task_advantage=True,
        )
        with TemporaryDirectory() as directory:
            path = save_checkpoint(
                Path(directory) / "walker.ckpt",
                low_actor,
                low_critic,
                transitions=int(low_history[-1]["progress/transitions"]),
                policy_kwargs=policy_kwargs,
                metrics={},
                config=OmegaConf.to_container(low_cfg, resolve=True),
            )
            payload = load_checkpoint(path)
    finally:
        low_env.close()

# %%
# Rebuild the checkpoint's architecture and task library
# ------------------------------------------------------
#
# The checkpoint stores the policy architecture and the original task
# definitions. ``make_render_policy`` reconstructs that architecture; the
# explicit ``load_state_dict`` below loads its weights. The temporary model
# environment installs the policy's ordinary GRU primer while constructing it.
# Deployment will install a separately namespaced primer on the new task.
#
# Keep the original task order: the walker's learned embedding associates
# ``task_id=2`` with the third training task, even if only some tasks are offered
# to the high-level actor.

model_env = make_env(
    checkpoint=payload,
    cfg={"backend": "mujoco", "device": "cpu", "parallel": False},
    download=True,
    num_envs=1,
)
try:
    walker = make_render_policy(model_env, checkpoint=payload)
    walker.load_state_dict(payload["model_state_dict"])
    walker.eval()
finally:
    model_env.close()

skill_tasks = torch.stack(make_tasks(payload["config"]["env"]["tasks"]))

# %%
# Give the walker a new job: reach a waypoint
# -------------------------------------------
#
# This small task uses the same robot and physical actions. It changes the
# reward to progress toward a fixed world-space waypoint, and adds the goal
# displacement and orientation to the observation. The high-level policy can
# use these six extra values to choose a skill; the low-level adapter receives
# only the original MicroDuck observation.
#
# All stepping, contacts and falls still come from ``MicroDuckEnv``. This class
# defines the new task, with no controller execution or recurrent-state code.


class WaypointMicroDuck(MicroDuckEnv):
    """Tutorial task: approach (0.5, 0.3) metres, terminating on arrival or a fall."""

    goal = (0.5, 0.3)

    def _make_obs_spec(self) -> Composite:
        spec = super()._make_obs_spec()
        spec["observation"] = Unbounded(
            (self.num_envs, self.OBSERVATION_DIM + 6),
            dtype=self.dtype,
            device=self.device,
        )
        return spec

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        obs = super()._build_obs_dict(state)
        qpos = state["qpos"].to(self.dtype)
        delta = qpos.new_tensor(self.goal) - qpos[..., :2]
        obs["observation"] = torch.cat(
            (obs["observation"], delta, qpos[..., 3:7]), dim=-1
        )
        return obs

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        before, after = state["qpos"][..., :2], next_state["qpos"][..., :2]
        goal = after.new_tensor(self.goal)
        old_distance = (before - goal).norm(dim=-1, keepdim=True)
        distance = (after - goal).norm(dim=-1, keepdim=True)
        fallen = super()._compute_done(state, next_state)
        return (
            10 * (old_distance - distance)
            + (distance < 0.05).to(self.dtype)
            - fallen.to(self.dtype)
            - 0.001
        )

    def _compute_done(
        self, state: TensorDictBase, next_state: TensorDictBase
    ) -> torch.Tensor:
        position = next_state["qpos"][..., :2]
        distance = (position - position.new_tensor(self.goal)).norm(
            dim=-1, keepdim=True
        )
        return super()._compute_done(state, next_state) | (distance < 0.05)


# %%
# Deploy with two objects
# -----------------------
#
# ``MicroDuckController`` supplies the robot's observation adapter and declares
# its gait-clock state. ``ClosedLoopMultiAction`` supplies the shared execution
# loop. Each high-level decision selects a task index and runs the walker for
# up to five physical steps, recomputing joint targets from fresh feedback.
#
# This single-robot task ends on a fall, so episode resets suffice:
# ``group_key=None, reset_key=None``. Match the checkpoint's physical action
# scale. The standard MicroDuck step is 0.02 s, so five steps make one
# high-level decision last up to 0.1 s.

task_env = WaypointMicroDuck(
    download=True,
    backend="mujoco",
    tasks=MicroDuckEnv.standing_task(),
    action_scale=payload["config"]["env"]["action_scale"],
    max_episode_steps=64 if TUTORIAL_FAST else 500,
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
    task_env, controller, steps=5, reward_aggregation="sum"
)
check_env_specs(training_env)

assert training_env.action_key == "skill"
assert training_env.full_action_spec["skill"].n == len(skill_tasks)
# The high-level observation includes the six waypoint/orientation values.
assert training_env.observation_spec["observation"].shape[-1] == (
    MicroDuckEnv.OBSERVATION_DIM + 6
)

td = training_env.reset()
td["skill"] = torch.zeros(training_env.batch_size, dtype=torch.long)
transition = training_env.step(td)
td = training_env.step_mdp(transition)
hidden = td["_controller", "recurrent_state"]
# hidden.shape == [1, 1, hidden_size]: env, GRU layer, features.
# This includes the final physical step's update; the next decision resumes it.

# %%
# Train the high-level skill selector with ordinary PPO
# -----------------------------------------------------
#
# The actor below outputs categorical skill indices. The collector only
# receives this actor; the walker lives inside ``training_env``.
# ``gamma=0.99`` now discounts one high-level decision, whose reward is the sum
# of executed physical-step rewards. Low-level inference is deterministic
# and has gradients disabled by default.

num_skills = training_env.full_action_spec["skill"].n
obs_dim = training_env.observation_spec["observation"].shape[-1]
high_actor = ProbabilisticActor(
    module=TensorDictModule(
        MLP(in_features=obs_dim, out_features=num_skills, num_cells=[64, 64]),
        in_keys=["observation"],
        out_keys=["logits"],
    ),
    in_keys=["logits"],
    out_keys=["skill"],
    spec=training_env.full_action_spec_unbatched,
    distribution_class=Categorical,
    return_log_prob=True,
)
high_critic = TensorDictModule(
    MLP(in_features=obs_dim, out_features=1, num_cells=[64, 64]),
    in_keys=["observation"],
    out_keys=["state_value"],
)
loss = ClipPPOLoss(
    high_actor, high_critic, normalize_advantage=True, entropy_coeff=0.01
)
loss.set_keys(action="skill")
loss.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)
optimizer = torch.optim.Adam(loss.parameters(), lr=3e-4)

walker_before = [p.detach().clone() for p in walker.parameters()]
actor_before = [p.detach().clone() for p in high_actor.parameters()]
collector = Collector(
    training_env,
    high_actor,
    frames_per_batch=32 if TUTORIAL_FAST else 128,
    total_frames=int(
        os.environ.get("MICRODUCK_HIGH_LEVEL_FRAMES", 64 if TUTORIAL_FAST else 1024)
    ),
)
high_history = []
try:
    for batch in collector:
        with torch.no_grad():
            loss.value_estimator(
                batch,
                params=loss.critic_network_params,
                target_params=loss.target_critic_network_params,
            )
        for _ in range(1 if TUTORIAL_FAST else 4):
            losses = loss(batch.reshape(-1))
            objective = (
                losses["loss_objective"]
                + losses["loss_critic"]
                + losses["loss_entropy"]
            )
            assert torch.isfinite(objective)
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
        high_history.append(batch["next", "reward"].mean().item())
        collector.update_policy_weights_()
finally:
    collector.shutdown(close_env=False)

assert any(
    not torch.equal(before, after)
    for before, after in zip(actor_before, high_actor.parameters())
)
for before, after in zip(walker_before, walker.parameters()):
    torch.testing.assert_close(before, after)
    assert after.grad is None

# %%
# Evaluate the composed policy
# ----------------------------
#
# The high-level actor is also deterministic for evaluation. Increasing the
# training budgets and loading a validated skill checkpoint is necessary to
# judge navigation performance; the short default run checks learning and
# deployment mechanics, not waypoint success.

with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    navigation = training_env.rollout(
        20 if TUTORIAL_FAST else 100, high_actor, break_when_any_done=True
    )
distance_to_goal = navigation["next", "observation"][..., -6:-4].norm(dim=-1)
# Inspect distance_to_goal and navigation["next", "terminated"] when evaluating.
training_env.close()

# %%
# The same walker in a 5-vs-5 task
# --------------------------------
#
# Given a task environment with ten agents under ``"agents"``, the deployment
# is the same. Each agent's observation must start with the MicroDuck
# observation layout, and ``"fallen"`` must be its raw respawn signal. The
# simulator's action scale and physical period must match the trained walker.
# The task environment supplies football physics, observations and rewards;
# the generic controller does not define teams or a football simulator.
#
# .. code-block:: python
#
#    from torchrl.envs import microduck_skill_env
#
#    football_training_env = microduck_skill_env(
#        football_env, walker, skill_tasks,
#        group_key="agents", steps=5, control_period_s=0.02,
#    )
#    # For 64 parallel matches with 10 players:
#    football_training_env.full_action_spec["agents", "skill"].shape
#    # torch.Size([64, 10])
#
#    td = football_training_env.reset()
#    td.update(football_training_env.full_action_spec.rand())
#    td = football_training_env.step_mdp(football_training_env.step(td))
#    td["agents", "_controller", "recurrent_state"].shape
#    # torch.Size([64, 10, 1, hidden_size])
#
# One walker shares its weights over 640 controller rows, while every row
# keeps its own recurrent state and gait clock. A fall resets that player's
# state to its declared defaults without resetting its neighbours. The preset
# also appends the previous skill to the high-level observation and reports
# whether each agent fell during the decision.
#
# For a shared per-agent actor, the PPO setup above changes its keys and
# output size; ``MLP`` already operates on every leading batch dimension:
#
# .. code-block:: python
#
#    obs_key = ("agents", "observation")
#    action_key = ("agents", "skill")
#    n = football_training_env.full_action_spec[action_key].n
#    actor = ProbabilisticActor(
#        module=TensorDictModule(
#            MLP(
#                in_features=football_training_env.observation_spec[obs_key].shape[-1],
#                out_features=n, num_cells=[64, 64],
#            ),
#            in_keys=[obs_key], out_keys=[("agents", "logits")],
#        ),
#        in_keys={"logits": ("agents", "logits")},
#        out_keys=[action_key],
#        spec=football_training_env.full_action_spec_unbatched,
#        distribution_class=Categorical,
#        return_log_prob=True,
#    )
#    value_key = ("agents", "state_value")
#    critic = TensorDictModule(
#        MLP(
#            in_features=football_training_env.observation_spec[obs_key].shape[-1],
#            out_features=1, num_cells=[64, 64],
#        ),
#        in_keys=[obs_key], out_keys=[value_key],
#    )
#    loss = ClipPPOLoss(actor, critic, normalize_advantage=False, entropy_coeff=0.01)
#    loss.set_keys(
#        action=action_key,
#        reward=("agents", "reward"),
#        value=value_key,
#        done=("agents", "done"),
#        terminated=("agents", "terminated"),
#    )
#    loss.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)
#
# The same collector/update loop applies. For this local critic, every player
# gets its own value prediction. A centralized critic can instead read the
# global match state. When episodes end at match level, add the corresponding
# per-agent done keys to each collected batch before computing GAE:
#
# .. code-block:: python
#
#    reward = batch["next", "agents", "reward"]
#    for key in ("done", "terminated"):
#        batch["next", "agents", key] = (
#            batch["next", key].unsqueeze(-1).expand_as(reward)
#        )
#
# Team rewards, opponent policies and self-play are high-level training
# choices. See the multi-agent PPO tutorial for those training conventions.


# %%
# Conclusion and further reading
# ------------------------------
#
# Task rows define the low-level skills. Their checkpoint preserves the
# walker's parameters, architecture and task-id order. ``MicroDuckController``
# adapts those skills to a new task, and ``ClosedLoopMultiAction`` executes them
# with fresh feedback and independent recurrent state. The high-level actor
# then trains with an ordinary collector and PPO loss.
#
# .. seealso::
#
#    - :class:`~torchrl.envs.MicroDuckEnv` and :class:`~torchrl.envs.MicroDuckTask`
#      for every field and preset.
#    - :class:`~torchrl.envs.MicroDuckController` and
#      :class:`~torchrl.envs.transforms.ClosedLoopMultiAction` for deployment.
#    - :class:`~torchrl.modules.LowLevelController` for other robots and
#      arbitrary TensorDict policies.
#    - :class:`~torchrl.objectives.value.GAE` for ``group_key``.
#    - :ref:`rlrender_tuto` for rendering checkpoints outside training.
#    - ``low_level_controller.py`` for continuous high-level commands, and
#      ``multiagent_ppo.py`` for per-agent PPO.
#    - ``examples/microduck/README.md`` for the locomotion training recipes.
