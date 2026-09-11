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
- how to train both levels with PPOTrainer, save and resume local runs,
  and keep documentation builds short;
- how to deploy skills with :class:`~torchrl.envs.MicroDuckController` and
  :class:`~torchrl.envs.transforms.ClosedLoopMultiAction`;
- how to train a high-level skill selector and reuse the deployment over
  multiple agents.
"""

from __future__ import annotations

import functools as ft
import importlib.util
import json
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
from torchrl.checkpoint import Checkpoint, GlobalRNGState
from torchrl.collectors import Collector, Evaluator
from torchrl.data import (
    Composite,
    LazyTensorStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    Unbounded,
)
from torchrl.envs import (
    MicroDuckController,
    MicroDuckEnv,
    MicroDuckTaskSampler,
    TransformedEnv,
)
from torchrl.envs.transforms import ClosedLoopMultiAction
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, set_recurrent_mode
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers import CSVLogger
from torchrl.render import load_checkpoint
from torchrl.trainers import BatchSubSampler
from torchrl.trainers.algorithms import PPOTrainer

REPO_ROOT = Path(torchrl.__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "microduck"
sys.path.insert(0, str(REPO_ROOT))
from examples.microduck.heuristic_gait import (  # noqa: E402
    gait_metrics,
    MicroDuckGaitActor,
)
from examples.microduck.ppo_mujoco import (  # noqa: E402
    evaluation_metrics,
    make_env,
    make_evaluator,
    make_models,
    make_render_policy,
    make_tasks,
    save_checkpoint,
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
# docs build and reduces each PPO stage to 64 steps.

if __name__ == "__main__":
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

if __name__ == "__main__":
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

if __name__ == "__main__":
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

if __name__ == "__main__":
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

if __name__ == "__main__":
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

if __name__ == "__main__":
    for backend in ("mjx", "mujoco-torch"):
        module = {"mjx": "mujoco.mjx", "mujoco-torch": "mujoco_torch"}[backend]
        print(
            backend,
            "available" if importlib.util.find_spec(module) else "not installed",
        )

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

if __name__ == "__main__":
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

if __name__ == "__main__":
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

if __name__ == "__main__":
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
                ROLLOUT_STEPS,
                tensordict=start,
                auto_reset=False,
                break_when_any_done=False,
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


if __name__ == "__main__":
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

if __name__ == "__main__":
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
# Both stages use :class:`~torchrl.trainers.algorithms.PPOTrainer`. It owns
# GAE, optimization, gradient clipping, logging and collector weight updates.
# We supply the environment, networks, loss and minibatch strategy.
#
# Outside documentation mode this is a long CPU training recipe: 10 million
# physical transitions for the walker, then one million high-level decisions.
# Native MuJoCo uses parallel CPU workers on macOS; no CUDA backend is needed.
# These are training budgets, not a guarantee that all six skills will succeed.
# Evaluate survival and command tracking before interpreting navigation results.
#
# .. code-block:: bash
#
#    MICRODUCK_OUTPUT_DIR=$HOME/microduck-training \
#        python tutorials/sphinx-tutorials/microduck.py
#
#    # Change the budgets or number of simulator workers:
#    MICRODUCK_LOW_LEVEL_FRAMES=20000000 MICRODUCK_HIGH_LEVEL_FRAMES=2000000 \
#        MICRODUCK_NUM_ENVS=8 MICRODUCK_OUTPUT_DIR=$HOME/microduck-training-longer \
#        python tutorials/sphinx-tutorials/microduck.py
#
# Documentation and tutorial CI set ``TORCHRL_TUTORIALS_FAST=1``: one simulator,
# 64 steps per stage and one optimization epoch. This takes precedence over
# the training-budget overrides. The same PPOTrainer code runs in both modes.
# A main guard keeps spawned MuJoCo workers from running the tutorial again.

if __name__ == "__main__":
    temporary_output = TemporaryDirectory() if TUTORIAL_FAST else None
    output_dir = Path(
        temporary_output.name
        if TUTORIAL_FAST
        else os.environ.get("MICRODUCK_OUTPUT_DIR", "microduck-training")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    num_envs = 1 if TUTORIAL_FAST else int(os.environ.get("MICRODUCK_NUM_ENVS", 16))
    low_frames = (
        64
        if TUTORIAL_FAST
        else int(os.environ.get("MICRODUCK_LOW_LEVEL_FRAMES", 10_000_000))
    )
    high_frames = (
        64
        if TUTORIAL_FAST
        else int(os.environ.get("MICRODUCK_HIGH_LEVEL_FRAMES", 1_000_000))
    )
    resume = not TUTORIAL_FAST and os.environ.get("MICRODUCK_RESUME", "0") == "1"
    torch.set_num_threads(1)
    torch.manual_seed(0)

    walker_checkpoint = os.environ.get("MICRODUCK_WALKER_CHECKPOINT")
    if walker_checkpoint:
        payload = load_checkpoint(walker_checkpoint)
    else:
        low_env_cfg = OmegaConf.load(EXAMPLES_DIR / "config.yaml").env
        low_env_cfg.update(
            backend="mujoco",
            device="cpu",
            num_envs=num_envs,
            parallel=num_envs > 1,
            download=True,
            max_episode_steps=32 if TUTORIAL_FAST else 500,
            tasks=[
                {"preset": "standing_task"},
                {"preset": "tracking_task", "speed": 0.2},
                {"preset": "tracking_task", "speed": -0.2},
                {"preset": "sidestep_task", "speed": 0.15},
                {"preset": "sidestep_task", "speed": -0.15},
                {"preset": "jump_task", "weight": 3.0},
            ],
        )
        policy_kwargs = {
            "hidden_size": 32 if TUTORIAL_FAST else 128,
            "policy_head": "gaussian",
            "initial_policy_scale": 1.0,
        }
        low_env = make_env(low_env_cfg)
        low_actor, low_critic = make_models(low_env, **policy_kwargs)
        low_loss = ClipPPOLoss(
            low_actor,
            low_critic,
            functional=False,
            normalize_advantage=False,
            entropy_coeff=0.01,
        )
        low_gae = GAE(
            gamma=0.99,
            lmbda=0.95,
            value_network=low_critic,
            average_gae=True,
            group_key="task_id",
            shifted=True,
            deactivate_vmap=True,
        )
        low_collector = Collector(
            low_env,
            low_actor,
            frames_per_batch=64 if TUTORIAL_FAST else num_envs * 1024,
            total_frames=low_frames,
        )
        low_trainer = PPOTrainer(
            collector=low_collector,
            total_frames=low_frames,
            frame_skip=1,
            loss_module=low_loss,
            optimizer=torch.optim.Adam(low_loss.parameters(), lr=3e-4),
            gae=set_recurrent_mode(True)(low_gae),
            num_epochs=1 if TUTORIAL_FAST else 5,
            optim_steps_per_batch=1 if TUTORIAL_FAST else 8,
            clip_norm=1.0,
            progress_bar=not TUTORIAL_FAST,
            logger=CSVLogger("low_level", log_dir=str(output_dir)),
            log_interval=1,
            log_timings=True,
            checkpoint=Checkpoint(rng=GlobalRNGState()),
            save_trainer_file=output_dir / "low_level.trainer",
            save_trainer_interval=100_000,
        )
        # Keep contiguous time windows for truncated backpropagation through the
        # GRU. Each window starts from its collected state; is_init resets it at
        # episode boundaries. Do not flatten time into independent transitions.
        BatchSubSampler(
            batch_size=64 if TUTORIAL_FAST else num_envs * 128,
            sub_traj_len=64 if TUTORIAL_FAST else 128,
        ).register(low_trainer)
        if resume:
            low_trainer.load_from_file(output_dir / "low_level.trainer")
        try:
            low_trainer.train()
            path = save_checkpoint(
                output_dir / "walker.ckpt",
                low_actor,
                low_critic,
                transitions=low_trainer.collected_frames,
                policy_kwargs=policy_kwargs,
                metrics={},
                config={"env": OmegaConf.to_container(low_env_cfg, resolve=True)},
            )
            payload = load_checkpoint(path)
        finally:
            low_collector.shutdown()

# %%
# Checkpoints and resuming
# ------------------------
#
# ``low_level.trainer`` and ``high_level.trainer`` contain the optimizer,
# collector, model, RNG and training counters. ``walker.ckpt`` additionally
# records the architecture and ordered task definitions for deployment.
# CSV losses, rewards and timings live under ``low_level/`` and ``high_level/``.
# Keep the same worker count and network/batch settings when resuming.
# Native MuJoCo resets simulator episodes after a restart; restored model and
# optimizer state continue learning, but the continuation is not bit-exact.
#
# .. code-block:: bash
#
#    # Resume an interrupted low-level stage, then run the high-level stage:
#    MICRODUCK_RESUME=1 MICRODUCK_OUTPUT_DIR=$HOME/microduck-training \
#        python tutorials/sphinx-tutorials/microduck.py
#
#    # Skip low-level training, or resume an interrupted high-level stage:
#    MICRODUCK_WALKER_CHECKPOINT=$HOME/microduck-training/walker.ckpt \
#        MICRODUCK_RESUME=1 MICRODUCK_OUTPUT_DIR=$HOME/microduck-training \
#        python tutorials/sphinx-tutorials/microduck.py
#
# Omit ``MICRODUCK_RESUME`` for a new high-level run from a supplied walker.

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

if __name__ == "__main__":
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
# Evaluate each skill before deploying it
# ---------------------------------------
#
# Pin one task at a time and evaluate deterministic rollouts on fresh resets.
# ``skills.json`` reports survival, episode length, velocity-tracking error
# and airborne time for jumping. A short documentation run only exercises
# these measurements; it cannot establish that the walker has learned them.

if __name__ == "__main__":
    skill_results = []
    jump_index = list(MicroDuckEnv.REWARD_TERMS).index("jump")
    for task_id, task in enumerate(skill_tasks.unbind(0)):
        evaluator = make_evaluator(
            make_env(
                checkpoint=payload,
                cfg={"backend": "mujoco", "task_id": task_id, "diagnostics": True},
                download=True,
                num_envs=1,
                parallel=False,
            ),
            walker,
            label=str(task_id),
            jumping=bool(task.reward_weights[jump_index] > 0),
            num_episodes=1 if TUTORIAL_FAST else 8,
            steps=32 if TUTORIAL_FAST else 500,
        )
        try:
            skill_results.append(evaluator.evaluate())
        finally:
            evaluator.shutdown()
    skill_metrics = evaluation_metrics(skill_results)
    (output_dir / "skills.json").write_text(json.dumps(skill_metrics, indent=2))

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

if __name__ == "__main__":
    make_task_env = ft.partial(
        WaypointMicroDuck,
        download=True,
        backend="mujoco",
        tasks=MicroDuckEnv.standing_task(),
        action_scale=payload["config"]["env"]["action_scale"],
        max_episode_steps=64 if TUTORIAL_FAST else 500,
        seed=0,
    )
    task_env = make_task_env(num_envs=num_envs, parallel=num_envs > 1)
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
# hidden.shape == [*training_env.batch_size, 1, hidden_size].
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
# ``functional=False`` lets the trainer's GAE read the critic's live parameters.
# The replay buffer shuffles transitions without replacement for each epoch;
# ``optim_steps_per_batch=None`` consumes all minibatches. Unlike the walker,
# this actor has no recurrent state, so flattening time is appropriate here.

if __name__ == "__main__":
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
        high_actor,
        high_critic,
        functional=False,
        normalize_advantage=True,
        entropy_coeff=0.01,
    )
    optimizer = torch.optim.Adam(loss.parameters(), lr=3e-4)

    collector = Collector(
        training_env,
        high_actor,
        frames_per_batch=32 if TUTORIAL_FAST else num_envs * 128,
        total_frames=high_frames,
    )
    replay = TensorDictReplayBuffer(
        storage=LazyTensorStorage(32 if TUTORIAL_FAST else num_envs * 128),
        sampler=SamplerWithoutReplacement(),
        batch_size=32 if TUTORIAL_FAST else 256,
    )
    high_trainer = PPOTrainer(
        collector=collector,
        total_frames=high_frames,
        frame_skip=1,  # Count high-level decisions, not physical substeps.
        loss_module=loss,
        optimizer=optimizer,
        action_key="skill",
        gamma=0.99,
        lmbda=0.95,
        replay_buffer=replay,
        optim_steps_per_batch=None,
        num_epochs=1 if TUTORIAL_FAST else 4,
        clip_norm=1.0,
        progress_bar=not TUTORIAL_FAST,
        log_actions=False,
        logger=CSVLogger("high_level", log_dir=str(output_dir)),
        log_interval=1,
        log_timings=True,
        checkpoint=Checkpoint(rng=GlobalRNGState()),
        save_trainer_file=output_dir / "high_level.trainer",
        save_trainer_interval=10_000,
    )
    if resume and walker_checkpoint:
        high_trainer.load_from_file(output_dir / "high_level.trainer")
    walker_before = [p.detach().clone() for p in walker.parameters()]
    actor_before = [p.detach().clone() for p in high_actor.parameters()]
    try:
        high_trainer.train()
    finally:
        collector.shutdown()

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
# training budgets alone does not establish navigation success. Inspect goal
# distance and arrival rate over multiple episodes and seeds. The documentation
# run checks the training/deployment pipeline, not learned skill quality.
# PPOTrainer closes its collector environment, so evaluation uses a fresh one.
# ``navigation.json`` records arrival rate and final distance across resets.


def navigation_metrics(trajectories: TensorDictBase) -> dict[str, float]:
    lengths = trajectories["collector", "mask"].sum(-1)
    distance = trajectories["next", "observation"][..., -6:-4].norm(dim=-1)
    final_distance = distance.gather(-1, (lengths - 1).unsqueeze(-1))
    return {
        "arrival_rate": (final_distance < 0.05).float().mean().item(),
        "final_distance_m": final_distance.mean().item(),
    }


if __name__ == "__main__":
    evaluation_env = ClosedLoopMultiAction.from_env(
        make_task_env(), controller, steps=5, reward_aggregation="sum"
    )
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        navigation = evaluation_env.rollout(
            20 if TUTORIAL_FAST else 100, high_actor, break_when_any_done=True
        )
    distance_to_goal = navigation["next", "observation"][..., -6:-4].norm(dim=-1)
    # Inspect distance_to_goal and navigation["next", "terminated"] when evaluating.
    evaluator = Evaluator(
        evaluation_env,
        high_actor,
        num_trajectories=1 if TUTORIAL_FAST else 32,
        max_steps=20 if TUTORIAL_FAST else 100,
        metrics_fn=navigation_metrics,
    )
    try:
        navigation_results = evaluator.evaluate()
        (output_dir / "navigation.json").write_text(
            json.dumps(navigation_results, indent=2)
        )
    finally:
        evaluator.shutdown()
    if temporary_output is not None:
        temporary_output.cleanup()

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
#    loss = ClipPPOLoss(
#        actor, critic, functional=False,
#        normalize_advantage=False, entropy_coeff=0.01,
#    )
#    loss.set_keys(
#        action=action_key,
#        reward=("agents", "reward"),
#        value=value_key,
#        done=("agents", "done"),
#        terminated=("agents", "terminated"),
#    )
#    loss.make_value_estimator(gamma=0.99, lmbda=0.95)  # PPO defaults to GAE.
#
# The same PPOTrainer applies. For this local critic, every player
# gets its own value prediction. A centralized critic can instead read the
# global match state. When episodes end at match level, add the corresponding
# per-agent done keys to each collected batch before computing GAE:
#
# .. code-block:: python
#
#    def match_boundaries(batch):
#        reward = batch["next", "agents", "reward"]
#        for key in ("done", "terminated"):
#            batch["next", "agents", key] = (
#                batch["next", key].unsqueeze(-1).expand_as(reward)
#            )
#        return batch
#
#    collector = Collector(
#        football_training_env, actor, frames_per_batch=4096,
#        total_frames=1_000_000,
#    )
#    trainer = PPOTrainer(
#        collector=collector, total_frames=1_000_000, frame_skip=1,
#        loss_module=loss,
#        optimizer=torch.optim.Adam(loss.parameters(), lr=3e-4),
#        gae=loss.value_estimator,
#        action_key=action_key, observation_key=obs_key,
#        reward_key=("agents", "reward"),
#        episode_reward_key=("agents", "reward"),
#        done_key=("agents", "done"),
#        terminated_key=("agents", "terminated"),
#        num_epochs=4, optim_steps_per_batch=1, clip_norm=1.0,
#        log_actions=False,
#    )
#    trainer.register_op("batch_process", match_boundaries)
#    trainer.train()
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
# then trains with the same PPOTrainer used for low-level learning.
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
