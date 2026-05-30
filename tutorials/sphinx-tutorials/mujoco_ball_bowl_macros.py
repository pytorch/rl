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

import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.envs import (
    BallBowlEnv,
    Compose,
    MultiAction,
    TransformedEnv,
    URScriptPrimitiveTransform,
    step_mdp,
)

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
# The default scene is intentionally lightweight. It is built only from MuJoCo
# primitive geoms: a six-revolute-joint arm with a kinematic layout inspired by
# Universal Robots manipulators, a simple two-finger gripper, a free ball, and a
# bowl. In this tutorial, "UR-style" means that the actuator interface follows
# the usual six arm joint targets plus a gripper command. It is not a vendored
# Universal Robots mesh asset.
#
# If you want a more realistic visual model, clone the relevant MuJoCo Menagerie
# assets and point the environment to them. The environment will then compose a
# UR5e arm and a Robotiq 2F-85 gripper locally, while the tutorial code below
# stays unchanged:
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
RENDER_WIDTH = 520
RENDER_HEIGHT = 390

# ``env_kwargs`` is the only branch that distinguishes the lightweight tutorial
# scene from the optional Menagerie scene. The rest of the tutorial only depends
# on the shared observation and action interface.
env_kwargs = {
    "robot_model": ROBOT_MODEL,
    "menagerie_path": MENAGERIE_PATH,
}


def show_frame(
    env: BallBowlEnv | TransformedEnv,
    title: str,
    *,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
) -> None:
    """Render one RGB frame and show it as a Sphinx-Gallery figure."""
    try:
        frame = env.render(width=width, height=height)[0].cpu().numpy()
    except Exception as err:
        torchrl_logger.warning("MuJoCo rendering failed: %s", err)
        frame = torch.zeros(height, width, 3, dtype=torch.uint8).numpy()
    _, axis = plt.subplots(figsize=(6.0, 4.5))
    axis.imshow(frame)
    axis.set_axis_off()
    axis.set_title(title)


env = BallBowlEnv(seed=0, max_episode_steps=200, **env_kwargs)
obs = env.reset()
assert obs["robot_qpos"].shape[-1] == 6
assert env.action_spec.shape[-1] == 7

show_frame(env, "Initial BallBowl scene")

# %%
# A plain rollout with random low-level actions already works because
# ``BallBowlEnv`` is a regular TorchRL environment. It is not, however, a useful
# robot policy: the random action is a raw seven-dimensional actuator command,
# so it has no notion of "move above the ball" or "open the gripper".

random_rollout = env.rollout(4)
assert random_rollout.get(("next", "reward")).shape[-1] == 1
show_frame(env, "After four random low-level actions")


# %%
# Primitive actions: the 10000-foot view
# --------------------------------------
#
# A primitive action is still a TensorDict. The policy-facing keys are:
#
# - ``primitive_id``: an integer id, not a string. For the preset used here,
#   ``0`` is ``wait``, ``1`` is ``movej``, ``2`` is ``movel``, ``3`` is
#   ``open_gripper`` and ``4`` is ``close_gripper``. A policy can keep nicer
#   Python names or an enum on its side, but the environment receives tensors.
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
        "primitive_id": torch.tensor([[URScriptPrimitiveTransform.MOVEJ]]),
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
# Helper functions for primitive policies
# ---------------------------------------
#
# The next helpers are the small amount of glue a scripted policy needs. The
# policy reads the latest observation, chooses a primitive id and fills whichever
# target field that primitive needs. This is also what a learned high-level
# policy would output, except that a model would predict these tensors instead
# of hand-writing them.


def action_from_robot_qpos(robot_qpos: torch.Tensor, gripper: float) -> torch.Tensor:
    """Pack six arm joint targets and one gripper command as a 7D action."""
    action = torch.zeros(*robot_qpos.shape[:-1], 7, dtype=robot_qpos.dtype)
    action[..., :6] = robot_qpos
    action[..., -1] = float(gripper)
    return action


def make_primitive_td(
    observation: TensorDictBase,
    primitive_id: int,
    *,
    target_pose: torch.Tensor | None = None,
    target_qpos: torch.Tensor | None = None,
    gripper: float = 0.0,
) -> TensorDictBase:
    """Create one batched primitive TensorDict from the latest observation."""
    batch_size = observation.batch_size
    if target_pose is None:
        target_pose = torch.zeros(*batch_size, 7)
    if target_qpos is None:
        target_qpos = action_from_robot_qpos(observation["robot_qpos"], gripper)
    td = observation.clone()
    td["primitive_id"] = torch.full(
        (*batch_size, 1), primitive_id, dtype=torch.long
    )
    td["target_pose"] = target_pose
    td["target_qpos"] = target_qpos
    td["gripper"] = torch.full((*batch_size, 1), float(gripper))
    return td


def pose_at(xyz: torch.Tensor) -> torch.Tensor:
    """Pack an xyz target with an identity quaternion in wxyz order."""
    quat = torch.zeros(*xyz.shape[:-1], 4, dtype=xyz.dtype, device=xyz.device)
    quat[..., 0] = 1.0
    return torch.cat([xyz, quat], dim=-1)


def step_primitive(
    env: TransformedEnv,
    observation: TensorDictBase,
    primitive: TensorDictBase,
) -> TensorDictBase:
    """Run one primitive and return the next observation."""
    primitive.update(observation.select(*env.observation_keys))
    transition = env.step(primitive)
    return step_mdp(transition)


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
    BallBowlEnv(seed=1, max_episode_steps=400, **env_kwargs),
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
show_frame(macro_env, "Macro environment before a primitive")

# A single ``movej`` primitive changes the joint target directly. Here we rotate
# the first joint a little, keeping the gripper open. The policy chooses the
# number ``MOVEJ`` and a joint target; the transform handles interpolation.
movej_target = action_from_robot_qpos(observation["robot_qpos"], GRIPPER_OPEN)
movej_target[..., 0] = movej_target[..., 0] + 0.35
observation = step_primitive(
    macro_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEJ,
        target_qpos=movej_target,
        gripper=GRIPPER_OPEN,
    ),
)
show_frame(macro_env, "After one movej primitive")

# Gripper primitives are just as small: the policy selects ``CLOSE_GRIPPER`` and
# the transform interpolates the last actuator entry to the configured close
# command.
observation = step_primitive(
    macro_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.CLOSE_GRIPPER,
        gripper=GRIPPER_CLOSE,
    ),
)
show_frame(macro_env, "After one close-gripper primitive")


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

task_env = TransformedEnv(
    BallBowlEnv(seed=2, max_episode_steps=500, **env_kwargs),
    Compose(
        MultiAction(stack_rewards=False),
        URScriptPrimitiveTransform(
            macro_steps=24,
            open_gripper_ctrl=GRIPPER_OPEN,
            close_gripper_ctrl=GRIPPER_CLOSE,
        ),
    ),
)

observation = task_env.reset()
initial_robot_qpos = observation["robot_qpos"].clone()
ball = observation["ball_pos"]
bowl = observation["bowl_pos"]
show_frame(task_env, "Scripted macro: start")

# 1. Move above the ball with an open gripper. The positive z offset gives the
# arm clearance before descending.
above_ball = pose_at(ball + torch.tensor([[0.0, 0.0, 0.14]]))
observation = step_primitive(
    task_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEL,
        target_pose=above_ball,
        gripper=GRIPPER_OPEN,
    ),
)
show_frame(task_env, "Scripted macro: above the ball")

# 2. Descend to the ball. This offset is close to the ball center height; in a
# real setup it would be calibrated from gripper geometry and contact behavior.
grip_ball = pose_at(ball + torch.tensor([[0.0, 0.0, 0.045]]))
observation = step_primitive(
    task_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEL,
        target_pose=grip_ball,
        gripper=GRIPPER_OPEN,
    ),
)
show_frame(task_env, "Scripted macro: descend to the ball")

# 3. Close the gripper around the ball.
observation = step_primitive(
    task_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.CLOSE_GRIPPER,
        gripper=GRIPPER_CLOSE,
    ),
)
show_frame(task_env, "Scripted macro: close around the ball")

# 4. Lift the ball before moving sideways. This avoids scraping the ball across
# the table and separates manipulation errors from transport errors.
lift_ball = pose_at(ball + torch.tensor([[0.0, 0.0, 0.18]]))
observation = step_primitive(
    task_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEL,
        target_pose=lift_ball,
        gripper=GRIPPER_CLOSE,
    ),
)
show_frame(task_env, "Scripted macro: lift")

# 5. Move above the bowl, still holding the gripper command closed.
above_bowl = pose_at(bowl + torch.tensor([[0.0, 0.0, 0.16]]))
observation = step_primitive(
    task_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEL,
        target_pose=above_bowl,
        gripper=GRIPPER_CLOSE,
    ),
)
show_frame(task_env, "Scripted macro: above the bowl")

# 6. Lower to the bowl target and open the gripper.
place_in_bowl = pose_at(bowl + torch.tensor([[0.0, 0.0, 0.075]]))
observation = step_primitive(
    task_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEL,
        target_pose=place_in_bowl,
        gripper=GRIPPER_CLOSE,
    ),
)
observation = step_primitive(
    task_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.OPEN_GRIPPER,
        gripper=GRIPPER_OPEN,
    ),
)
show_frame(task_env, "Scripted macro: release at the bowl")

# 7. Return to the initial joint configuration. This illustrates that a macro
# policy can mix Cartesian primitives and joint-space primitives.
home_target = action_from_robot_qpos(initial_robot_qpos, GRIPPER_OPEN)
observation = step_primitive(
    task_env,
    observation,
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEJ,
        target_qpos=home_target,
        gripper=GRIPPER_OPEN,
    ),
)
show_frame(task_env, "Scripted macro: return home")

assert "success" in observation.keys()


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


randomized_envs = [make_randomized_env(index) for index in range(2)]
randomized_observations = [randomized_env.reset() for randomized_env in randomized_envs]
assert not torch.equal(
    randomized_observations[0]["ball_pos"][..., 1],
    randomized_observations[1]["ball_pos"][..., 1],
)

show_frame(randomized_envs[0], "Randomized scene 0")
show_frame(randomized_envs[1], "Randomized scene 1")

for randomized_env in randomized_envs:
    randomized_env.close()
task_env.close()
macro_env.close()
env.close()


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
