"""
MuJoCo scripted manipulation with macro actions
===============================================

The first step in controlling a complex robot is rarely an end-to-end RL
policy. Closed-loop controllers, scripted motion primitives, and imitation data
are often used first to make the task reliable enough for later reward-driven
fine tuning. This tutorial uses a compact MuJoCo manipulation scene to show how
a controller can be written in TorchRL without hiding the TensorDict plumbing.

We will control a cube-to-bowl task without learning: a policy emits a small
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
- how to write and render a scripted contact-rich cube-to-bowl macro;
- why construction-time scene randomization is a useful first step before
  imitation learning or RL.
"""

from __future__ import annotations

import importlib.util
import os

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import (
    CubeBowlEnv,
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
# :class:`~torchrl.envs.CubeBowlEnv` that exposes the state needed for scripted
# manipulation: robot joints, gripper joints, the pinch site, the cube pose, the
# bowl target and a success flag.
#
# The task reward is intentionally sparse. It is ``1`` only when the cube center
# is within the environment's placement tolerance of the bowl target coordinate,
# and ``0`` otherwise. This makes reward checks easy to interpret: any non-zero
# reward means the cube is actually at the target, not merely closer to it.
#
# When a MuJoCo Menagerie checkout is present through
# ``TORCHRL_MUJOCO_MENAGERIE_PATH``, the tutorial uses a proper UR5e arm with a
# Robotiq 2F-85 gripper. If those assets are absent, it falls back to a
# lightweight scene built only from MuJoCo primitive geoms: a six-revolute-joint
# arm with a Universal-Robots-like kinematic layout, a simple two-finger gripper,
# a free cube, and a bowl. In this tutorial, "UR-style" means that the actuator
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
#
# There is a single runtime branch: if Menagerie is available we use the
# Menagerie robot, otherwise we use the lightweight primitive robot. Everything
# below is written against the shared observation and action interface.

MENAGERIE_PATH = os.environ.get(CubeBowlEnv.MENAGERIE_ENV_VAR)
ROBOT_MODEL = "menagerie_ur5e" if MENAGERIE_PATH else "primitive"
ENV_KWARGS = {
    "robot_model": ROBOT_MODEL,
    "menagerie_path": MENAGERIE_PATH,
}
RENDER_WIDTH = 480
RENDER_HEIGHT = 360
VIDEO_INTERVAL_MS = 55

# ``movel`` uses MuJoCo damped-least-squares IK. The primitive robot is simple
# enough for the defaults; the Menagerie scene benefits from more IK iterations
# and an orientation term so the Robotiq pads remain aligned with the cube.
IK_KWARGS = (
    {
        "iterations": 220,
        "orientation_weight": 1.0,
        "step_size": 0.7,
        "damping": 1e-4,
    }
    if ROBOT_MODEL == "menagerie_ur5e"
    else {}
)

# %%
# We can instantiate the base environment exactly like any other TorchRL
# environment and inspect the shape of its observations and actions.

env = CubeBowlEnv(seed=0, max_episode_steps=200, **ENV_KWARGS)
obs = env.reset()
assert obs["robot_qpos"].shape[-1] == 6
assert env.action_spec.shape[-1] == 7
assert env.gripper_open_ctrl == 0.0
assert env.low_level_action(obs["robot_qpos"]).shape[-1] == 7
env.close()

# %%
# A small recording helper keeps the tutorial examples short while still using
# TorchRL's reusable rendering stack. The helper creates a normal
# :class:`~torchrl.envs.TransformedEnv` with two transforms:
# :class:`~torchrl.record.PixelRenderTransform` stores the latest MuJoCo render
# under ``"pixels"``, and :class:`~torchrl.record.VideoRecorder` accumulates
# those frames for Sphinx.


def make_recording_env(
    *,
    seed: int,
    max_episode_steps: int,
    recorder_skip: int = 1,
) -> tuple[TransformedEnv, CubeBowlEnv, VideoRecorder, TensorDictBase]:
    base_env = CubeBowlEnv(
        seed=seed,
        max_episode_steps=max_episode_steps,
        **ENV_KWARGS,
    )
    recorder = VideoRecorder(
        logger=None,
        tag="pixels",
        in_keys=["pixels"],
        skip=recorder_skip,
        make_grid=False,
    )
    transformed_env = TransformedEnv(
        base_env,
        Compose(
            PixelRenderTransform(width=RENDER_WIDTH, height=RENDER_HEIGHT),
            recorder,
        ),
    )
    observation = transformed_env.reset()
    return transformed_env, base_env, recorder, observation


# %%
# The first clip keeps the robot still for a few simulator steps. It shows the
# scene that every later rollout starts from.

initial_env, initial_base_env, initial_recorder, observation = make_recording_env(
    seed=0,
    max_episode_steps=20,
)
for _ in range(12):
    hold_action = initial_base_env.low_level_action(
        observation["robot_qpos"], initial_base_env.gripper_open_ctrl
    )
    transition = initial_env.step(observation.clone().set("action", hold_action))
    observation = step_mdp(transition)
initial_scene_animation = initial_recorder.to_animation(
    title="Initial cube-to-bowl scene",
    interval=VIDEO_INTERVAL_MS,
    clear=True,
)
initial_env.close()

# %%
# A plain rollout with random low-level actions already works because
# ``CubeBowlEnv`` is a regular TorchRL environment. It is not, however, a useful
# robot policy: the random action is a raw seven-dimensional actuator command,
# so it has no notion of "move above the cube" or "open the gripper".

random_env, _, random_recorder, observation = make_recording_env(
    seed=1,
    max_episode_steps=80,
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
# that expand into low-level actuator targets. ``movej`` linearly interpolates
# joint targets. ``movel`` calls the environment's MuJoCo IK helper to convert a
# Cartesian target for the ``pinch`` site into joint targets.
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
# - ``gripper``: an optional scalar override for the gripper command emitted by
#   the primitive.
#
# The environment constructs the URScript-style transform for us. This avoids
# tutorial-side solver closures and gripper-range constants: the transform knows
# this environment's IK method, action keys, and gripper commands.

base_env = CubeBowlEnv(seed=2, max_episode_steps=50, **ENV_KWARGS)
observation = base_env.reset()
primitive_transform = base_env.make_urscript_transform(
    macro_steps=8,
    ik_kwargs=IK_KWARGS,
)

# A high-level primitive TensorDict can be built by name. The helper fills the
# TensorDict keys used by the transform and keeps nested-key support in one
# place.
movej_target = base_env.low_level_action(
    observation["robot_qpos"], base_env.gripper_open_ctrl
)
movej_target[..., 0] = movej_target[..., 0] + 0.35
movej_primitive = primitive_transform.make_primitive(
    observation,
    URScriptPrimitive.MOVEJ,
    target_qpos=movej_target,
    gripper=base_env.gripper_open_ctrl,
)

# Expanding the primitive yields the low-level actuator sequence that MuJoCo
# consumes. No environment step has happened yet.
expanded_action = primitive_transform.action_sequence(movej_primitive)
assert expanded_action.shape == torch.Size([1, 8, 7])
base_env.close()


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

base_env = CubeBowlEnv(seed=3, max_episode_steps=400, **ENV_KWARGS)
primitive_transform = base_env.make_urscript_transform(
    macro_steps=20,
    ik_kwargs=IK_KWARGS,
)
macro_env = TransformedEnv(
    base_env,
    Compose(
        MultiAction(stack_rewards=False),
        primitive_transform,
    ),
)

observation = macro_env.reset()
movej_target = base_env.low_level_action(
    observation["robot_qpos"], base_env.gripper_open_ctrl
)
movej_target[..., 0] = movej_target[..., 0] + 0.35
observation = step_mdp(
    macro_env.step(
        primitive_transform.make_primitive(
            observation,
            URScriptPrimitive.MOVEJ,
            target_qpos=movej_target,
            gripper=base_env.gripper_open_ctrl,
        )
    )
)
observation = step_mdp(
    macro_env.step(
        primitive_transform.make_primitive(
            observation,
            URScriptPrimitive.CLOSE_GRIPPER,
            gripper=base_env.gripper_close_ctrl,
        )
    )
)
assert "success" in observation.keys()
macro_env.close()

# %%
# For the remaining videos we execute the expanded action sequence explicitly.
# This is the same sequence that ``MultiAction`` consumes, but an explicit loop
# lets us collect per-low-level-step diagnostics such as gripper-to-cube
# distance, cube lift and sparse reward. Those diagnostics are useful when
# debugging a scripted contact policy.


def step_low_level_action(
    env: TransformedEnv,
    observation: TensorDictBase,
    action: torch.Tensor,
) -> tuple[TensorDictBase, TensorDictBase]:
    transition = env.step(observation.clone().set("action", action))
    next_observation = step_mdp(transition)
    next_observation["last_reward"] = transition["next", "reward"]
    return next_observation, transition


# ``run_primitive`` is intentionally visible: it is the complete bridge between
# a high-level primitive TensorDict and MuJoCo low-level stepping. Applications
# that do not need per-step diagnostics can replace this loop with
# ``MultiAction`` as shown above.


def run_primitive(
    env: TransformedEnv,
    base_env: CubeBowlEnv,
    transform: URScriptPrimitiveTransform,
    observation: TensorDictBase,
    primitive_id: URScriptPrimitive,
    *,
    target_pose: torch.Tensor | None = None,
    target_qpos: torch.Tensor | None = None,
    gripper: float | torch.Tensor | None = None,
) -> tuple[TensorDictBase, TensorDictBase]:
    sequence = transform.action_sequence(
        observation,
        primitive_id,
        target_pose=target_pose,
        target_qpos=target_qpos,
        gripper=gripper,
    )
    start_cube = observation["cube_pos"].clone()
    start_cube_z = start_cube[..., 2:3].clone()
    min_gripper_distance = torch.full_like(start_cube[..., :1], float("inf"))
    max_cube_displacement = torch.zeros_like(start_cube[..., :1])
    max_cube_lift = torch.zeros_like(start_cube[..., :1])
    max_reward = torch.zeros_like(start_cube[..., :1])
    last_reward = torch.zeros_like(start_cube[..., :1])

    for action in sequence[0]:
        observation, transition = step_low_level_action(
            env,
            observation,
            action.view(1, 7),
        )
        min_gripper_distance = torch.minimum(
            min_gripper_distance, base_env.gripper_cube_distance(observation)
        )
        cube_displacement = (observation["cube_pos"] - start_cube).norm(
            dim=-1, keepdim=True
        )
        cube_lift = observation["cube_pos"][..., 2:3] - start_cube_z
        max_cube_displacement = torch.maximum(max_cube_displacement, cube_displacement)
        max_cube_lift = torch.maximum(max_cube_lift, cube_lift)
        last_reward = transition["next", "reward"]
        max_reward = torch.maximum(max_reward, last_reward)

    metrics = TensorDict(
        {
            "min_gripper_cube_distance": min_gripper_distance,
            "cube_displacement": max_cube_displacement,
            "cube_lift": max_cube_lift,
            "max_reward": max_reward,
            "last_reward": last_reward,
        },
        batch_size=observation.batch_size,
    )
    return observation, metrics


# %%
# The next video records one ``movej`` primitive followed by one gripper
# primitive at the low-level MuJoCo step rate.

primitive_env, primitive_base_env, primitive_recorder, observation = make_recording_env(
    seed=4,
    max_episode_steps=120,
)
primitive_transform = primitive_base_env.make_urscript_transform(
    macro_steps=18,
    ik_kwargs=IK_KWARGS,
)
movej_target = primitive_base_env.low_level_action(
    observation["robot_qpos"], primitive_base_env.gripper_open_ctrl
)
movej_target[..., 0] = movej_target[..., 0] + 0.35
observation, _ = run_primitive(
    primitive_env,
    primitive_base_env,
    primitive_transform,
    observation,
    URScriptPrimitive.MOVEJ,
    target_qpos=movej_target,
    gripper=primitive_base_env.gripper_open_ctrl,
)
observation, _ = run_primitive(
    primitive_env,
    primitive_base_env,
    primitive_transform,
    observation,
    URScriptPrimitive.CLOSE_GRIPPER,
    gripper=primitive_base_env.gripper_close_ctrl,
)
primitive_animation = primitive_recorder.to_animation(
    title="One movej primitive followed by one close-gripper primitive",
    interval=VIDEO_INTERVAL_MS,
    clear=True,
)
primitive_env.close()


# %%
# Sparse reward sanity check
# --------------------------
#
# Before tuning contact-rich grasping, it is useful to isolate the reward
# condition itself. The shortest possible check is: initialize the cube at the
# bowl target, emit one ``wait`` primitive so the environment takes one step, and
# assert that the next reward is exactly ``1``.
#
# This is a diagnostic sequence, not a manipulation policy. It verifies that the
# coordinate-based success predicate and the primitive-to-action expansion agree
# on what a solved state looks like.

reward_env = CubeBowlEnv(seed=5, max_episode_steps=3, **ENV_KWARGS)
reward_observation = reward_env.reset(
    TensorDict({"cube_pos": reward_env._target_pos().clone()}, batch_size=[1])
)
reward_transform = reward_env.make_urscript_transform(macro_steps=1)
reward_action = reward_transform.action_sequence(
    reward_observation,
    URScriptPrimitive.WAIT,
    gripper=reward_env.gripper_open_ctrl,
)[:, 0]
reward_transition = reward_env.step(
    reward_observation.clone().set("action", reward_action)
)
assert reward_transition["next", "reward"].item() == 1.0
assert reward_transition["next", "success"].all()
reward_env.close()


# %%
# A scripted cube-to-bowl macro
# -----------------------------
#
# We can now write the cube-to-bowl controller as a sequence of small decisions.
# The controller uses current observations for the cube and bowl positions. This
# makes it more robust than an open-loop joint trajectory, but it is still a
# calibrated scripted policy: if the scene geometry, contacts, or camera-to-robot
# calibration change too much, the macro will need retuning or a learned residual
# controller.
#
# The Menagerie policy below uses the real UR5e + Robotiq model when the assets
# are available. It approaches the cube, closes the gripper near the cube, lifts
# high enough to clear the receiving bowl, carries the cube over the bowl, drops
# it, and returns the robot close to its starting configuration. The bowl has
# four full-height sides. The assertions are intentionally strict: the gripper
# must reach the cube, the cube must lift and move while the gripper is closed,
# the robot must return home, and the final sparse reward must be exactly ``1``.
#
# When Menagerie assets are not available, the same macro API is demonstrated on
# the lightweight primitive scene with a shorter pick-and-place sequence.

task_env, task_base_env, task_recorder, observation = make_recording_env(
    seed=6,
    max_episode_steps=8000 if ROBOT_MODEL == "menagerie_ur5e" else 700,
    recorder_skip=8 if ROBOT_MODEL == "menagerie_ur5e" else 1,
)


def make_task_transform(
    macro_steps: int,
    settle_steps: int = 0,
) -> URScriptPrimitiveTransform:
    return task_base_env.make_urscript_transform(
        macro_steps=macro_steps,
        settle_steps=settle_steps,
        ik_kwargs=IK_KWARGS,
    )


def update_closed_motion(
    reference_cube: torch.Tensor,
    observation: TensorDictBase,
    current_motion: torch.Tensor,
    current_lift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cube_motion = (observation["cube_pos"] - reference_cube).norm(
        dim=-1, keepdim=True
    )
    cube_lift = observation["cube_pos"][..., 2:3] - reference_cube[..., 2:3]
    return torch.maximum(current_motion, cube_motion), torch.maximum(
        current_lift, cube_lift
    )


initial_robot_qpos = observation["robot_qpos"].clone()
gripper_quat = observation["pinch_quat"].clone()
gripper_open = task_base_env.gripper_open_ctrl
gripper_close = task_base_env.gripper_close_ctrl
grasp_distance = torch.full_like(observation["cube_pos"][..., :1], float("inf"))
cube_motion_while_closed = torch.zeros_like(grasp_distance)
cube_lift_while_closed = torch.zeros_like(grasp_distance)
max_reward = torch.zeros_like(grasp_distance)
last_reward = torch.zeros_like(grasp_distance)

if ROBOT_MODEL == "menagerie_ur5e":
    approach_transform = make_task_transform(180, settle_steps=60)
    close_transform = make_task_transform(160, settle_steps=80)
    lift_transform = make_task_transform(120, settle_steps=60)
    carry_transform = make_task_transform(80, settle_steps=20)
    drop_transform = make_task_transform(100, settle_steps=40)
    open_transform = make_task_transform(100, settle_steps=20)
    retreat_transform = make_task_transform(120, settle_steps=60)
    home_transform = make_task_transform(250, settle_steps=800)

    # Settle at the reset posture with the gripper open.
    for _ in range(20):
        hold_action = task_base_env.low_level_action(
            observation["robot_qpos"], gripper_open
        )
        observation, transition = step_low_level_action(
            task_env, observation, hold_action
        )
        last_reward = transition["next", "reward"]
        max_reward = torch.maximum(max_reward, last_reward)

    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        open_transform,
        observation,
        URScriptPrimitive.OPEN_GRIPPER,
        gripper=gripper_open,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])

    cube = observation["cube_pos"].clone()
    bowl = observation["bowl_pos"].clone()
    above_cube = task_base_env.pose_at(
        cube + cube.new_tensor([[0.0, 0.0, 0.18]]), gripper_quat
    )
    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        approach_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=above_cube,
        gripper=gripper_open,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])

    grasp_cube = task_base_env.pose_at(
        cube + cube.new_tensor([[0.0, 0.0, -0.005]]), gripper_quat
    )
    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        approach_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=grasp_cube,
        gripper=gripper_open,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])

    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        close_transform,
        observation,
        URScriptPrimitive.CLOSE_GRIPPER,
        gripper=gripper_close,
    )
    grasp_distance = torch.minimum(
        grasp_distance, metrics["min_gripper_cube_distance"]
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])
    closed_reference_cube = observation["cube_pos"].clone()
    cube_motion_while_closed, cube_lift_while_closed = update_closed_motion(
        closed_reference_cube,
        observation,
        cube_motion_while_closed,
        cube_lift_while_closed,
    )

    cube = observation["cube_pos"].clone()
    pinch_to_cube = observation["pinch_pos"].clone() - cube
    lift_cube = task_base_env.pose_at(
        cube + pinch_to_cube + cube.new_tensor([[0.0, 0.0, 0.20]]),
        gripper_quat,
    )
    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        lift_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=lift_cube,
        gripper=gripper_close,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])
    cube_motion_while_closed, cube_lift_while_closed = update_closed_motion(
        closed_reference_cube,
        observation,
        cube_motion_while_closed,
        cube_lift_while_closed,
    )

    start_y = observation["cube_pos"][..., 1:2].clone()
    target_y = bowl[..., 1:2]
    for waypoint in range(1, 5):
        alpha = float(waypoint) / 4.0
        desired_cube = torch.cat(
            [
                bowl[..., :1],
                start_y + alpha * (target_y - start_y),
                torch.full_like(bowl[..., 2:3], 0.24),
            ],
            dim=-1,
        )
        cube = observation["cube_pos"].clone()
        pinch_to_cube = observation["pinch_pos"].clone() - cube
        carry_cube = task_base_env.pose_at(desired_cube + pinch_to_cube, gripper_quat)
        observation, metrics = run_primitive(
            task_env,
            task_base_env,
            carry_transform,
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=carry_cube,
            gripper=gripper_close,
        )
        max_reward = torch.maximum(max_reward, metrics["max_reward"])
        cube_motion_while_closed, cube_lift_while_closed = update_closed_motion(
            closed_reference_cube,
            observation,
            cube_motion_while_closed,
            cube_lift_while_closed,
        )

    cube = observation["cube_pos"].clone()
    pinch_to_cube = observation["pinch_pos"].clone() - cube
    drop_cube = torch.cat(
        [
            bowl[..., :1],
            bowl[..., 1:2],
            torch.full_like(bowl[..., 2:3], 0.13),
        ],
        dim=-1,
    )
    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        drop_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=task_base_env.pose_at(drop_cube + pinch_to_cube, gripper_quat),
        gripper=gripper_close,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])
    cube_motion_while_closed, cube_lift_while_closed = update_closed_motion(
        closed_reference_cube,
        observation,
        cube_motion_while_closed,
        cube_lift_while_closed,
    )

    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        open_transform,
        observation,
        URScriptPrimitive.OPEN_GRIPPER,
        gripper=gripper_open,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])

    # Give the cube time to settle in the bowl before the arm retreats.
    for _ in range(240):
        hold_action = task_base_env.low_level_action(
            observation["robot_qpos"], gripper_open
        )
        observation, transition = step_low_level_action(
            task_env, observation, hold_action
        )
        last_reward = transition["next", "reward"]
        max_reward = torch.maximum(max_reward, last_reward)

    retreat_xyz = torch.cat(
        [
            observation["pinch_pos"][..., :2],
            torch.full_like(observation["pinch_pos"][..., 2:3], 0.26),
        ],
        dim=-1,
    )
    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        retreat_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=task_base_env.pose_at(retreat_xyz, gripper_quat),
        gripper=gripper_open,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])

    home_target = task_base_env.low_level_action(initial_robot_qpos, gripper_open)
    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        home_transform,
        observation,
        URScriptPrimitive.MOVEJ,
        target_qpos=home_target,
        gripper=gripper_open,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])
    final_wait_action = home_target
    final_wait_steps = 800
    robot_home_tolerance = 0.03
else:
    task_transform = make_task_transform(20)
    cube = observation["cube_pos"].clone()
    bowl = observation["bowl_pos"].clone()

    observation, _ = run_primitive(
        task_env,
        task_base_env,
        task_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=task_base_env.pose_at(cube + cube.new_tensor([[0.0, 0.0, 0.14]])),
        gripper=gripper_open,
    )
    observation, _ = run_primitive(
        task_env,
        task_base_env,
        task_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=task_base_env.pose_at(cube + cube.new_tensor([[0.0, 0.0, 0.02]])),
        gripper=gripper_open,
    )
    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        task_transform,
        observation,
        URScriptPrimitive.CLOSE_GRIPPER,
        gripper=gripper_close,
    )
    grasp_distance = metrics["min_gripper_cube_distance"].clone()
    closed_reference_cube = observation["cube_pos"].clone()

    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        task_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=task_base_env.pose_at(cube + cube.new_tensor([[0.0, 0.0, 0.18]])),
        gripper=gripper_close,
    )
    cube_motion_while_closed = metrics["cube_displacement"].clone()
    cube_lift_while_closed = metrics["cube_lift"].clone()
    max_reward = metrics["max_reward"].clone()

    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        task_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=task_base_env.pose_at(bowl + bowl.new_tensor([[0.0, 0.0, 0.18]])),
        gripper=gripper_close,
    )
    cube_motion_while_closed, cube_lift_while_closed = update_closed_motion(
        closed_reference_cube,
        observation,
        cube_motion_while_closed,
        cube_lift_while_closed,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])

    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        task_transform,
        observation,
        URScriptPrimitive.MOVEL,
        target_pose=task_base_env.pose_at(bowl + bowl.new_tensor([[0.0, 0.0, 0.08]])),
        gripper=gripper_close,
    )
    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        task_transform,
        observation,
        URScriptPrimitive.OPEN_GRIPPER,
        gripper=gripper_open,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])

    home_target = task_base_env.low_level_action(initial_robot_qpos, gripper_open)
    observation, metrics = run_primitive(
        task_env,
        task_base_env,
        task_transform,
        observation,
        URScriptPrimitive.MOVEJ,
        target_qpos=home_target,
        gripper=gripper_open,
    )
    max_reward = torch.maximum(max_reward, metrics["max_reward"])
    final_wait_action = home_target
    final_wait_steps = 80
    robot_home_tolerance = 1.0

for _ in range(final_wait_steps):
    observation, transition = step_low_level_action(
        task_env,
        observation,
        final_wait_action,
    )
    last_reward = transition["next", "reward"]
    max_reward = torch.maximum(max_reward, last_reward)

robot_home_error = (observation["robot_qpos"] - initial_robot_qpos).norm(
    dim=-1, keepdim=True
)
scripted_macro_animation = task_recorder.to_animation(
    title="Scripted macro: pick the cube into the bowl",
    interval=VIDEO_INTERVAL_MS,
    clear=True,
)

assert grasp_distance.item() <= 0.025
assert cube_motion_while_closed.item() >= 0.05
assert cube_lift_while_closed.item() >= 0.08
assert robot_home_error.item() <= robot_home_tolerance
assert max_reward.item() == 1.0
assert last_reward.item() == 1.0
assert observation["success"].all()
task_env.close()


# %%
# Construction-time scene randomization
# -------------------------------------
#
# Randomization is a bridge between a single calibrated script and a policy that
# survives heterogeneous scenes. A macro controller may work well when cube and
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


def make_randomized_env(index: int) -> CubeBowlEnv:
    offset = 0.025 * float(index)
    if ROBOT_MODEL == "menagerie_ur5e":
        cube_position = CubeBowlEnv.MENAGERIE_CUBE_POSITION
        bowl_position = CubeBowlEnv.MENAGERIE_BOWL_POSITION
    else:
        cube_position = (0.32, -0.14, 0.035)
        bowl_position = (0.28, 0.19, 0.01)
    return CubeBowlEnv(
        cube_position=(
            cube_position[0],
            cube_position[1] + offset,
            cube_position[2],
        ),
        bowl_position=(
            bowl_position[0],
            bowl_position[1] - offset,
            bowl_position[2],
        ),
        seed=index,
        **ENV_KWARGS,
    )


randomized_envs = [make_randomized_env(index) for index in range(4)]
randomized_observations = [randomized_env.reset() for randomized_env in randomized_envs]
assert not torch.equal(
    randomized_observations[0]["cube_pos"][..., 1],
    randomized_observations[1]["cube_pos"][..., 1],
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
#    - :class:`~torchrl.envs.CubeBowlEnv` for the custom MuJoCo task used here.
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
