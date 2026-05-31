"""
MuJoCo scripted manipulation with macro actions
===============================================

The first step in controlling a complex robot is rarely an end-to-end RL
policy. Closed-loop controllers, scripted motion primitives, and imitation data
are often used first to make the task reliable enough for later reward-driven
fine tuning. This tutorial uses a compact MuJoCo manipulation scene to show how
such a controller can be written in TorchRL.

We will control a ball-to-bowl task without learning: a policy emits a small
TensorDict that describes *which* macro to run, and TorchRL transforms expand
that macro into the low-level actuator sequence consumed by the MuJoCo
environment. The same pattern can later provide demonstrations, curricula, or a
safe initialization for residual RL.

What you will learn
-------------------

- how to instantiate and render a custom MuJoCo environment;
- what a primitive action TensorDict contains and how it differs from a raw
  actuator action;
- how :class:`~torchrl.envs.URScriptPrimitiveTransform` expands primitives such
  as ``movej`` and ``movel`` into fixed-length action sequences;
- how :class:`~torchrl.envs.MultiAction` executes those sequences in a base
  environment;
- how to write and render a scripted pick-and-place macro;
- why construction-time scene randomization is a useful first step before
  imitation learning or RL.
"""

from __future__ import annotations

import importlib.util
import os

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import (
    BallBowlEnv,
    Compose,
    MultiAction,
    TransformedEnv,
    URScriptPrimitive,
    URScriptPrimitiveTransform,
    step_mdp,
)
from torchrl.record import PixelRenderTransform, VideoRecorder

_has_mujoco = importlib.util.find_spec("mujoco") is not None

if not _has_mujoco:
    raise ImportError("This tutorial requires the `mujoco` Python package.")


# %%
# Environment
# -----------
#
# For this tutorial, we introduce a custom MuJoCo-based
# :class:`~torchrl.envs.BallBowlEnv` that exposes the state needed for scripted
# manipulation: robot joints, gripper joints, the pinch site, the ball pose, the
# bowl target and a success flag.
#
# When a MuJoCo Menagerie checkout is present and detectable through
# ``TORCHRL_MUJOCO_MENAGERIE_PATH``, the tutorial uses a proper UR5e arm with a
# Robotiq 2F-85 gripper. If those assets are absent, it falls back to a
# lightweight scene built only from MuJoCo primitive geoms: a six-revolute-joint
# arm with a Universal-Robots-like kinematic layout, a simple two-finger gripper,
# a free ball, and a bowl. In this tutorial, "UR-style" means that the actuator
# interface follows the usual six arm joint targets plus a gripper command.
#
# For the rendered documentation, TorchRL's docs build fetches a sparse MuJoCo
# Menagerie checkout and sets ``TORCHRL_MUJOCO_MENAGERIE_PATH``. If you want the
# same UR5e + Robotiq view locally, clone the relevant assets and point the
# environment to them:
#
#   .. code-block:: bash
#
#      git clone --depth=1 --filter=blob:none --sparse \
#          https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
#      git -C /tmp/menagerie sparse-checkout set \
#          universal_robots_ur5e robotiq_2f85
#      export TORCHRL_MUJOCO_MENAGERIE_PATH=/tmp/menagerie
#
# The low-level action is always seven-dimensional: the first six entries are
# arm joint-position targets and the last entry is the gripper command. The
# primitive scene uses a gripper command in meters, whereas the Menagerie
# Robotiq actuator uses the native ``0`` to ``255`` range.

MENAGERIE_PATH = os.environ.get(BallBowlEnv.MENAGERIE_ENV_VAR)
ROBOT_MODEL = "menagerie_ur5e" if MENAGERIE_PATH else "primitive"
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 255.0 if MENAGERIE_PATH else 0.038
RENDER_WIDTH = 480
RENDER_HEIGHT = 360
VIDEO_INTERVAL_MS = 55

# %%
# The following dictionary is the only branch that distinguishes the lightweight
# tutorial scene from the optional Menagerie scene. The rest of the tutorial only
# depends on the shared observation and action interface.

env_kwargs = {
    "robot_model": ROBOT_MODEL,
    "menagerie_path": MENAGERIE_PATH,
}

# %%
# TorchRL records videos through two reusable transforms:
# :class:`~torchrl.record.PixelRenderTransform` calls the environment's
# ``render`` method and stores the result under ``"pixels"``;
# :class:`~torchrl.record.VideoRecorder` accumulates these pixels and can either
# dump them through a logger or expose them as a notebook/Sphinx animation.
#
# The small helpers used to avoid repeating the same boilerplate are hidden in
# the rendered tutorial. They are ordinary TorchRL plumbing: create an env with a
# pixel-rendering transform, pack primitive TensorDicts, and execute an expanded
# primitive while the recorder collects frames.

# sphinx_gallery_start_ignore

def make_video_env(
    *,
    seed: int,
    max_episode_steps: int,
    macro_steps: int,
) -> tuple[
    TransformedEnv,
    BallBowlEnv,
    URScriptPrimitiveTransform,
    VideoRecorder,
    TensorDictBase,
]:
    base_env = BallBowlEnv(
        seed=seed,
        max_episode_steps=max_episode_steps,
        **env_kwargs,
    )
    recorder = VideoRecorder(
        logger=None,
        tag="pixels",
        in_keys=["pixels"],
        skip=1,
        make_grid=False,
    )
    env = TransformedEnv(
        base_env,
        Compose(
            PixelRenderTransform(width=RENDER_WIDTH, height=RENDER_HEIGHT),
            recorder,
        ),
    )
    primitive_transform = URScriptPrimitiveTransform(
        macro_steps=macro_steps,
        cartesian_solver=base_env._cartesian_pose_to_joint_target,
        open_gripper_ctrl=GRIPPER_OPEN,
        close_gripper_ctrl=GRIPPER_CLOSE,
    )
    observation = env.reset()
    return env, base_env, primitive_transform, recorder, observation


def repeat_last_frame(recorder: VideoRecorder, count: int) -> None:
    if recorder.obs:
        recorder.obs.extend(recorder.obs[-1].clone() for _ in range(count - 1))


def action_from_robot_qpos(robot_qpos: torch.Tensor, gripper: float) -> torch.Tensor:
    action = torch.zeros(*robot_qpos.shape[:-1], 7, dtype=robot_qpos.dtype)
    action[..., :6] = robot_qpos
    action[..., -1] = float(gripper)
    return action


def make_primitive_td(
    observation: TensorDictBase,
    primitive_id: int | URScriptPrimitive,
    *,
    target_pose: torch.Tensor | None = None,
    target_qpos: torch.Tensor | None = None,
    gripper: float = 0.0,
) -> TensorDictBase:
    batch_size = observation.batch_size
    if target_pose is None:
        target_pose = torch.zeros(*batch_size, 7)
    if target_qpos is None:
        target_qpos = action_from_robot_qpos(observation["robot_qpos"], gripper)
    td = observation.clone()
    td["primitive_id"] = torch.full(
        (*batch_size, 1), int(primitive_id), dtype=torch.long
    )
    td["target_pose"] = target_pose
    td["target_qpos"] = target_qpos
    td["gripper"] = torch.full((*batch_size, 1), float(gripper))
    return td


def pose_at(xyz: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros(*xyz.shape[:-1], 4, dtype=xyz.dtype, device=xyz.device)
    quat[..., 0] = 1.0
    return torch.cat([xyz, quat], dim=-1)


def step_primitive(
    env: TransformedEnv,
    observation: TensorDictBase,
    primitive: TensorDictBase,
) -> TensorDictBase:
    primitive.update(observation.select(*env.observation_keys))
    transition = env.step(primitive)
    return step_mdp(transition)


def record_primitive(
    env: TransformedEnv,
    transform: URScriptPrimitiveTransform,
    observation: TensorDictBase,
    primitive: TensorDictBase,
) -> TensorDictBase:
    primitive.update(observation.select(*env.observation_keys))
    expanded = transform.inv(primitive)
    for action in expanded["action"][0]:
        transition = env.step(observation.clone().set("action", action.view(1, 7)))
        observation = step_mdp(transition)
    return observation

# sphinx_gallery_end_ignore

# %%
# We can now instantiate the base environment exactly like any other TorchRL
# environment and inspect the shape of its observations and actions.

env = BallBowlEnv(seed=0, max_episode_steps=200, **env_kwargs)
obs = env.reset()
assert obs["robot_qpos"].shape[-1] == 6
assert env.action_spec.shape[-1] == 7
env.close()

# %%
# This first video is a reset frame recorded with
# :class:`~torchrl.record.VideoRecorder`. Keeping it as a video rather than a
# static image lets the rendered tutorial use the same path for every visual.

initial_env, _, _, initial_recorder, _ = make_video_env(
    seed=0,
    max_episode_steps=20,
    macro_steps=8,
)
repeat_last_frame(initial_recorder, 12)
initial_scene_animation = initial_recorder.to_animation(
    title="Initial BallBowl scene",
    interval=VIDEO_INTERVAL_MS,
    clear=True,
)
initial_env.close()

# %%
# A plain rollout with random low-level actions already works because
# ``BallBowlEnv`` is a regular TorchRL environment. It is not, however, a useful
# robot policy: the random action is a raw seven-dimensional actuator command,
# so it has no notion of "move above the ball" or "open the gripper".

random_env, _, _, random_recorder, observation = make_video_env(
    seed=1,
    max_episode_steps=80,
    macro_steps=8,
)
for _ in range(24):
    random_action = random_env.action_spec.rand()
    transition = random_env.step(observation.clone().set("action", random_action))
    observation = step_mdp(transition)
random_rollout_animation = random_recorder.to_animation(
    title="Random low-level actuator commands",
    interval=VIDEO_INTERVAL_MS,
    clear=True,
)
random_env.close()


# %%
# Primitive actions: the 10000-foot view
# --------------------------------------
#
# The names ``movej`` and ``movel`` come from Universal Robots' URScript
# convention:
#
# - ``movej`` means "move in joint space". The command specifies target joint
#   positions, and the controller interpolates from the current joint
#   configuration to those targets. It is useful for returning to a home pose or
#   moving through a posture that is known to be collision-free.
# - ``movel`` means "move linearly in Cartesian space". The command specifies a
#   target end-effector pose, and the controller follows a straight-line path for
#   the tool center point. It is useful for approach, descend, lift and place
#   motions where the gripper should move along a predictable line.
#
# In this TorchRL tutorial both commands are implemented as macro primitives
# that expand into fixed-length low-level actuator targets. ``movej`` linearly
# interpolates joint targets. ``movel`` interpolates Cartesian targets and calls
# an inverse-kinematics solver to convert each waypoint into joint targets for
# the MuJoCo environment.
#
# A primitive action is still a TensorDict. The policy-facing keys are:
#
# - ``primitive_id``: an integer id represented by
#   :class:`~torchrl.envs.URScriptPrimitive`. For the preset used here,
#   ``URScriptPrimitive.WAIT`` is ``0``, ``MOVEJ`` is ``1``, ``MOVEL`` is ``2``,
#   ``OPEN_GRIPPER`` is ``3`` and ``CLOSE_GRIPPER`` is ``4``. The enum gives
#   readable policy code, and ``int(primitive)`` gives the tensor value consumed
#   by the transform.
# - ``target_qpos``: a seven-dimensional low-level target for ``movej``. The
#   first six values are joint targets and the last value is the gripper command.
# - ``target_pose``: an ``xyz + quaternion`` Cartesian target for ``movel``.
#   In this environment, ``movel`` solves for the ``pinch`` site. The orientation
#   part is included for a standard pose shape, but the default solver optimizes
#   the position only.
# - ``gripper``: an optional scalar override for the gripper command emitted by
#   the primitive.
#
# .. admonition:: Solver and adapter roles
#
#    :class:`~torchrl.envs.URScriptPrimitiveTransform` is a preset of the more
#    general :class:`~torchrl.envs.MacroPrimitiveTransform`. The transform owns
#    the TensorDict plumbing. The *adapter* explains how to read and write the
#    low-level action vector, while the *solver* explains how to turn a Cartesian
#    target into joint targets. For a Franka, a mobile manipulator, or a custom
#    gripper, keep the macro pattern but provide an adapter and solver matching
#    that robot's action space.

primitive_transform = URScriptPrimitiveTransform(
    macro_steps=8,
    open_gripper_ctrl=GRIPPER_OPEN,
    close_gripper_ctrl=GRIPPER_CLOSE,
)

movej_td = TensorDict(
    {
        "primitive_id": torch.tensor([[int(URScriptPrimitive.MOVEJ)]]),
        "target_qpos": torch.zeros(1, 7),
        "target_pose": torch.zeros(1, 7),
        "robot_qpos": obs["robot_qpos"],
        "gripper_qpos": obs["gripper_qpos"],
    },
    batch_size=[1],
)
expanded = primitive_transform.inv(movej_td)
assert expanded["action"].shape == torch.Size([1, 8, 7])


# %%
# Executing individual primitives with MultiAction
# ------------------------------------------------
#
# :class:`~torchrl.envs.MultiAction` is the transform that executes a sequence of
# low-level actions. If the primitive transform writes ``"action"`` with shape
# ``[batch, macro_steps, 7]``, ``MultiAction`` consumes the middle dimension by
# stepping the base environment ``macro_steps`` times.
#
# .. note:: Transform order for actions
#
#    In a :class:`~torchrl.envs.TransformedEnv`, inverse transforms run from
#    right to left when an action enters the environment. The composition below
#    is therefore read as: first expand the primitive into an action sequence,
#    then let ``MultiAction`` execute that sequence.

macro_env = TransformedEnv(
    BallBowlEnv(seed=2, max_episode_steps=400, **env_kwargs),
    Compose(
        MultiAction(stack_rewards=False),
        URScriptPrimitiveTransform(
            macro_steps=20,
            open_gripper_ctrl=GRIPPER_OPEN,
            close_gripper_ctrl=GRIPPER_CLOSE,
        ),
    ),
)

observation = macro_env.reset()
movej_target = action_from_robot_qpos(observation["robot_qpos"], GRIPPER_OPEN)
movej_target[..., 0] = movej_target[..., 0] + 0.35
observation = step_primitive(
    macro_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.MOVEJ,
        target_qpos=movej_target,
        gripper=GRIPPER_OPEN,
    ),
)
observation = step_primitive(
    macro_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.CLOSE_GRIPPER,
        gripper=GRIPPER_CLOSE,
    ),
)
assert "success" in observation.keys()
macro_env.close()

# %%
# The next video records the same pair of primitives at the low-level MuJoCo step
# rate. The rendered clip is produced by :meth:`~torchrl.record.VideoRecorder.to_animation`.

primitive_env, _, primitive_transform, primitive_recorder, observation = make_video_env(
    seed=3,
    max_episode_steps=120,
    macro_steps=18,
)
movej_target = action_from_robot_qpos(observation["robot_qpos"], GRIPPER_OPEN)
movej_target[..., 0] = movej_target[..., 0] + 0.35
observation = record_primitive(
    primitive_env,
    primitive_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.MOVEJ,
        target_qpos=movej_target,
        gripper=GRIPPER_OPEN,
    ),
)
observation = record_primitive(
    primitive_env,
    primitive_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.CLOSE_GRIPPER,
        gripper=GRIPPER_CLOSE,
    ),
)
primitive_animation = primitive_recorder.to_animation(
    title="One movej primitive followed by one close-gripper primitive",
    interval=VIDEO_INTERVAL_MS,
    clear=True,
)
primitive_env.close()


# %%
# A scripted pick-and-place macro
# -------------------------------
#
# We can now write the ball-to-bowl controller as a sequence of small decisions.
# The controller uses current observations for the ball and bowl positions. This
# makes it more robust than an open-loop joint trajectory, but it is still a
# calibrated scripted policy: if the scene geometry, contacts, or camera-to-robot
# calibration change too much, the macro will need retuning or a learned residual
# controller.
#
# The waypoints below are deliberately introduced one by one. Each waypoint uses
# ``movel`` to place the gripper pinch site at a Cartesian target, except the
# final ``movej`` that returns to the initial joint configuration.

task_env, _, task_transform, task_recorder, observation = make_video_env(
    seed=4,
    max_episode_steps=500,
    macro_steps=20,
)
initial_robot_qpos = observation["robot_qpos"].clone()
ball = observation["ball_pos"]
bowl = observation["bowl_pos"]

# 1. Move above the ball with an open gripper. The positive z offset gives the
# arm clearance before descending.
above_ball = pose_at(ball + torch.tensor([[0.0, 0.0, 0.14]]))
observation = record_primitive(
    task_env,
    task_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=above_ball,
        gripper=GRIPPER_OPEN,
    ),
)

# 2. Descend to the ball. This offset is close to the ball center height; in a
# real setup it would be calibrated from gripper geometry and contact behavior.
grip_ball = pose_at(ball + torch.tensor([[0.0, 0.0, 0.045]]))
observation = record_primitive(
    task_env,
    task_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=grip_ball,
        gripper=GRIPPER_OPEN,
    ),
)

# 3. Close the gripper around the ball.
observation = record_primitive(
    task_env,
    task_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.CLOSE_GRIPPER,
        gripper=GRIPPER_CLOSE,
    ),
)

# 4. Lift the ball before moving sideways. This avoids scraping the ball across
# the table and separates manipulation errors from transport errors.
lift_ball = pose_at(ball + torch.tensor([[0.0, 0.0, 0.18]]))
observation = record_primitive(
    task_env,
    task_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=lift_ball,
        gripper=GRIPPER_CLOSE,
    ),
)

# 5. Move above the bowl, still holding the gripper command closed.
above_bowl = pose_at(bowl + torch.tensor([[0.0, 0.0, 0.16]]))
observation = record_primitive(
    task_env,
    task_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=above_bowl,
        gripper=GRIPPER_CLOSE,
    ),
)

# 6. Lower to the bowl target and open the gripper.
place_in_bowl = pose_at(bowl + torch.tensor([[0.0, 0.0, 0.075]]))
observation = record_primitive(
    task_env,
    task_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=place_in_bowl,
        gripper=GRIPPER_CLOSE,
    ),
)
observation = record_primitive(
    task_env,
    task_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.OPEN_GRIPPER,
        gripper=GRIPPER_OPEN,
    ),
)

# 7. Return to the initial joint configuration. This illustrates that a macro
# policy can mix Cartesian primitives and joint-space primitives.
home_target = action_from_robot_qpos(initial_robot_qpos, GRIPPER_OPEN)
observation = record_primitive(
    task_env,
    task_transform,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitive.MOVEJ,
        target_qpos=home_target,
        gripper=GRIPPER_OPEN,
    ),
)
scripted_macro_animation = task_recorder.to_animation(
    title="Scripted macro: pick the ball, place it at the bowl, return home",
    interval=VIDEO_INTERVAL_MS,
    clear=True,
)

assert "success" in observation.keys()
task_env.close()


# %%
# Construction-time scene randomization
# -------------------------------------
#
# Randomization is a bridge between a single calibrated script and a policy that
# survives heterogeneous scenes. A macro controller may work well when ball and
# bowl poses are known and remain within a small workspace. RL or imitation
# learning can then train over many placements, lighting conditions or object
# sizes, and learn corrections for contact-rich details that a hand-written
# script misses.
#
# The simplest randomization is construction-time randomization: make a family
# of environment factories whose XML is built with different object placements.
# This is easy and deterministic, but it is not as flexible as reset-time
# randomization. Reset-time randomization of static bodies requires modeling
# those bodies as movable joints or mocap targets and checking that new poses are
# collision-free.

# sphinx_gallery_start_ignore

def make_randomized_env(index: int) -> BallBowlEnv:
    offset = 0.025 * float(index)
    if ROBOT_MODEL == "menagerie_ur5e":
        ball_position = BallBowlEnv.MENAGERIE_BALL_POSITION
        bowl_position = BallBowlEnv.MENAGERIE_BOWL_POSITION
    else:
        ball_position = (0.32, -0.14, 0.035)
        bowl_position = (0.28, 0.19, 0.01)
    return BallBowlEnv(
        ball_position=(
            ball_position[0],
            ball_position[1] + offset,
            ball_position[2],
        ),
        bowl_position=(
            bowl_position[0],
            bowl_position[1] - offset,
            bowl_position[2],
        ),
        seed=index,
        **env_kwargs,
    )

# sphinx_gallery_end_ignore

randomized_envs = [make_randomized_env(index) for index in range(4)]
randomized_observations = [randomized_env.reset() for randomized_env in randomized_envs]
assert not torch.equal(
    randomized_observations[0]["ball_pos"][..., 1],
    randomized_observations[1]["ball_pos"][..., 1],
)

randomization_recorder = VideoRecorder(
    logger=None,
    tag="randomized",
    in_keys=["pixels"],
    skip=1,
    make_grid=False,
)
for randomized_env in randomized_envs:
    pixels = randomized_env.render(width=RENDER_WIDTH, height=RENDER_HEIGHT)
    randomization_recorder._apply_transform(pixels)
    repeat_last_frame(randomization_recorder, 12)
randomization_animation = randomization_recorder.to_animation(
    title="Construction-time randomization of ball and bowl placement",
    interval=VIDEO_INTERVAL_MS,
    clear=True,
)

for randomized_env in randomized_envs:
    randomized_env.close()


# %%
# Conclusion and further reading
# ------------------------------
#
# Macro actions are useful when a policy should choose *what* to do at a slower
# time scale while a scripted controller handles short-horizon actuation. They
# are not a replacement for learning in difficult contact-rich settings: they
# still depend on calibration, solver quality and reachable waypoints. Their
# practical value is that they provide a strong baseline, a source of
# demonstrations, and a structured action space for residual policies or RL fine
# tuning.
#
# .. seealso::
#
#    - :class:`~torchrl.envs.BallBowlEnv` for the custom MuJoCo task used here.
#    - :class:`~torchrl.envs.MujocoEnv` for the base class behind custom MuJoCo
#      environments.
#    - :class:`~torchrl.envs.MacroPrimitiveTransform` for robot-agnostic macro
#      expansion with custom adapters and solvers.
#    - :class:`~torchrl.envs.URScriptPrimitiveTransform` for the URScript-style
#      preset used in this tutorial.
#    - :class:`~torchrl.envs.MultiAction` for executing batched low-level action
#      sequences.
#    - :class:`~torchrl.record.PixelRenderTransform` and
#      :class:`~torchrl.record.VideoRecorder` for rendering and recording
#      simulator videos.
