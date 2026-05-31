"""
MuJoCo scripted manipulation with macro actions
===============================================

The first step in controlling a complex robot is rarely an end-to-end RL
policy. Closed-loop controllers, scripted motion primitives, and imitation data
are often used first to make the task reliable enough for later reward-driven
fine tuning. This tutorial uses a compact MuJoCo manipulation scene to show how
such a controller can be written in TorchRL.

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
# When a MuJoCo Menagerie checkout is present and detectable through
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
# The Menagerie scene uses a receiving bowl with four full-height sides.

MENAGERIE_PATH = os.environ.get(CubeBowlEnv.MENAGERIE_ENV_VAR)
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
    settle_steps: int = 0,
    recorder_skip: int = 1,
    orientation_weight: float = 0.0,
    local_env_kwargs: dict | None = None,
) -> tuple[
    TransformedEnv,
    CubeBowlEnv,
    URScriptPrimitiveTransform,
    VideoRecorder,
    TensorDictBase,
]:
    if local_env_kwargs is None:
        local_env_kwargs = env_kwargs
    base_env = CubeBowlEnv(
        seed=seed,
        max_episode_steps=max_episode_steps,
        **local_env_kwargs,
    )
    recorder = VideoRecorder(
        logger=None,
        tag="pixels",
        in_keys=["pixels"],
        skip=recorder_skip,
        make_grid=False,
    )
    env = TransformedEnv(
        base_env,
        Compose(
            PixelRenderTransform(width=RENDER_WIDTH, height=RENDER_HEIGHT),
            recorder,
        ),
    )

    def cartesian_solver(
        target_pose: torch.Tensor, start_action: torch.Tensor
    ) -> torch.Tensor:
        return base_env._cartesian_pose_to_joint_target(
            target_pose,
            start_action,
            iterations=160,
            orientation_weight=orientation_weight,
        )

    primitive_transform = URScriptPrimitiveTransform(
        macro_steps=macro_steps,
        settle_steps=settle_steps,
        cartesian_solver=cartesian_solver,
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


def pose_with_quat(xyz: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    quat = quat.expand(*xyz.shape[:-1], 4)
    return torch.cat([xyz, quat], dim=-1)


def xyz_like(
    reference: torch.Tensor, values: tuple[float, float, float]
) -> torch.Tensor:
    return torch.tensor([values], dtype=reference.dtype, device=reference.device)


def gripper_cube_distance(observation: TensorDictBase) -> torch.Tensor:
    cube_pos = observation["cube_pos"]
    half_size = torch.full_like(cube_pos, CubeBowlEnv.OBJECT_HALF_SIZE)

    def pad_to_cube(pad_pos: torch.Tensor) -> torch.Tensor:
        q = (pad_pos - cube_pos).abs() - half_size
        outside = q.clamp_min(0.0).norm(dim=-1, keepdim=True)
        inside = q.max(dim=-1, keepdim=True).values.clamp_max(0.0)
        return outside + inside

    left_distance = pad_to_cube(observation["gripper_left_pad_pos"])
    right_distance = pad_to_cube(observation["gripper_right_pad_pos"])
    return torch.minimum(left_distance, right_distance).clamp_min(0.0)


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
    start_cube = observation["cube_pos"].clone()
    start_cube_z = start_cube[..., 2:3].clone()
    min_pinch_distance = torch.full_like(start_cube[..., :1], float("inf"))
    min_gripper_distance = torch.full_like(start_cube[..., :1], float("inf"))
    max_cube_displacement = torch.zeros_like(start_cube[..., :1])
    max_cube_lift = torch.zeros_like(start_cube[..., :1])
    max_reward = torch.zeros_like(start_cube[..., :1])
    for action in expanded["action"][0]:
        transition = env.step(observation.clone().set("action", action.view(1, 7)))
        observation = step_mdp(transition)
        observation["last_reward"] = transition["next", "reward"]
        pinch_distance = (
            observation["pinch_pos"] - observation["cube_pos"]
        ).norm(dim=-1, keepdim=True)
        min_pinch_distance = torch.minimum(min_pinch_distance, pinch_distance)
        min_gripper_distance = torch.minimum(
            min_gripper_distance, gripper_cube_distance(observation)
        )
        cube_displacement = (observation["cube_pos"] - start_cube).norm(
            dim=-1, keepdim=True
        )
        cube_lift = observation["cube_pos"][..., 2:3] - start_cube_z
        max_cube_displacement = torch.maximum(max_cube_displacement, cube_displacement)
        max_cube_lift = torch.maximum(max_cube_lift, cube_lift)
        max_reward = torch.maximum(max_reward, transition["next", "reward"])
    observation["primitive_min_pinch_cube_distance"] = min_pinch_distance
    observation["primitive_min_gripper_cube_distance"] = min_gripper_distance
    observation["primitive_cube_displacement"] = max_cube_displacement
    observation["primitive_cube_lift"] = max_cube_lift
    observation["primitive_max_reward"] = max_reward
    return observation

# sphinx_gallery_end_ignore

# %%
# We can now instantiate the base environment exactly like any other TorchRL
# environment and inspect the shape of its observations and actions.

env = CubeBowlEnv(seed=0, max_episode_steps=200, **env_kwargs)
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
    CubeBowlEnv(seed=2, max_episode_steps=400, **env_kwargs),
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

reward_env = CubeBowlEnv(seed=5, max_episode_steps=3, **env_kwargs)
reward_observation = reward_env.reset(
    TensorDict({"cube_pos": reward_env._target_pos().clone()}, batch_size=[1])
)
reward_transform = URScriptPrimitiveTransform(
    macro_steps=1,
    open_gripper_ctrl=GRIPPER_OPEN,
    close_gripper_ctrl=GRIPPER_CLOSE,
)
reward_primitive = make_primitive_td(
    reward_observation,
    URScriptPrimitive.WAIT,
    gripper=GRIPPER_OPEN,
)
reward_primitive.update(reward_observation.select(*reward_env.observation_keys))
reward_action = reward_transform.inv(reward_primitive)["action"][:, 0]
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
# are available. It approaches the cube, closes the gripper near the cube, then
# lifts it high enough to clear the receiving bowl, carries it over the bowl,
# drops it, and returns the robot close to its starting configuration. The bowl
# has four full-height sides. The assertions are intentionally strict: the
# gripper must reach the cube, the cube must lift and move while the gripper is
# closed, the robot must return home, and the final sparse reward must be
# exactly ``1``.
#
# When Menagerie assets are not available, the same macro API is demonstrated on
# the lightweight primitive scene with a simple pick-and-place sequence.

# sphinx_gallery_start_ignore

if ROBOT_MODEL == "menagerie_ur5e":
    task_env, task_base_env, _, task_recorder, observation = make_video_env(
        seed=4,
        max_episode_steps=5000,
        macro_steps=8,
        recorder_skip=8,
    )

    def task_solver(
        target_pose: torch.Tensor, start_action: torch.Tensor
    ) -> torch.Tensor:
        return task_base_env._cartesian_pose_to_joint_target(
            target_pose,
            start_action,
            iterations=220,
            orientation_weight=1.0,
            step_size=0.7,
            damping=1e-4,
        )

    def task_transform(
        macro_steps: int, settle_steps: int = 0
    ) -> URScriptPrimitiveTransform:
        return URScriptPrimitiveTransform(
            macro_steps=macro_steps,
            settle_steps=settle_steps,
            cartesian_solver=task_solver,
            open_gripper_ctrl=GRIPPER_OPEN,
            close_gripper_ctrl=GRIPPER_CLOSE,
        )

    approach_transform = task_transform(macro_steps=180, settle_steps=60)
    close_transform = task_transform(macro_steps=160, settle_steps=80)
    lift_transform = task_transform(macro_steps=120, settle_steps=60)
    carry_transform = task_transform(macro_steps=80, settle_steps=20)
    drop_transform = task_transform(macro_steps=100, settle_steps=40)
    open_transform = task_transform(macro_steps=100, settle_steps=20)
    retreat_transform = task_transform(macro_steps=120, settle_steps=60)
    home_transform = task_transform(macro_steps=250, settle_steps=800)

    task_gripper_open = GRIPPER_OPEN
    task_gripper_close = GRIPPER_CLOSE
    initial_robot_qpos = observation["robot_qpos"].clone()
    gripper_quat = observation["pinch_quat"].clone()
    grasp_distance = torch.full_like(observation["cube_pos"][..., :1], float("inf"))
    cube_motion_while_closed = torch.zeros_like(grasp_distance)
    cube_lift_while_closed = torch.zeros_like(grasp_distance)
    max_reward = torch.zeros_like(grasp_distance)

    def primitive_reward(observation: TensorDictBase) -> torch.Tensor:
        return observation["primitive_max_reward"]

    def closed_motion(
        reference_cube: torch.Tensor, observation: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            (observation["cube_pos"] - reference_cube).norm(dim=-1, keepdim=True),
            observation["cube_pos"][..., 2:3] - reference_cube[..., 2:3],
        )

    for _ in range(20):
        transition = task_env.step(
            observation.clone().set(
                "action",
                action_from_robot_qpos(observation["robot_qpos"], task_gripper_open),
            )
        )
        observation = step_mdp(transition)
        observation["last_reward"] = transition["next", "reward"]
        max_reward = torch.maximum(max_reward, transition["next", "reward"])

    observation = record_primitive(
        task_env,
        open_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.OPEN_GRIPPER,
            gripper=task_gripper_open,
        ),
    )
    max_reward = torch.maximum(max_reward, primitive_reward(observation))

    cube = observation["cube_pos"].clone()
    bowl = observation["bowl_pos"].clone()
    above_cube = pose_with_quat(
        cube + xyz_like(cube, (0.0, 0.0, 0.18)), gripper_quat
    )
    observation = record_primitive(
        task_env,
        approach_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=above_cube,
            gripper=task_gripper_open,
        ),
    )
    max_reward = torch.maximum(max_reward, primitive_reward(observation))

    grasp_cube = pose_with_quat(
        cube + xyz_like(cube, (0.0, 0.0, -0.005)), gripper_quat
    )
    observation = record_primitive(
        task_env,
        approach_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=grasp_cube,
            gripper=task_gripper_open,
        ),
    )
    max_reward = torch.maximum(max_reward, primitive_reward(observation))

    observation = record_primitive(
        task_env,
        close_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.CLOSE_GRIPPER,
            gripper=task_gripper_close,
        ),
    )
    grasp_distance = torch.minimum(
        grasp_distance, observation["primitive_min_gripper_cube_distance"]
    )
    max_reward = torch.maximum(max_reward, primitive_reward(observation))
    closed_reference_cube = observation["cube_pos"].clone()
    cube_motion, cube_lift = closed_motion(closed_reference_cube, observation)
    cube_motion_while_closed = torch.maximum(cube_motion_while_closed, cube_motion)
    cube_lift_while_closed = torch.maximum(cube_lift_while_closed, cube_lift)

    cube = observation["cube_pos"].clone()
    pinch_to_cube = observation["pinch_pos"].clone() - cube
    lift_cube = pose_with_quat(
        cube + pinch_to_cube + xyz_like(cube, (0.0, 0.0, 0.20)),
        gripper_quat,
    )
    observation = record_primitive(
        task_env,
        lift_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=lift_cube,
            gripper=task_gripper_close,
        ),
    )
    max_reward = torch.maximum(max_reward, primitive_reward(observation))
    cube_motion, cube_lift = closed_motion(closed_reference_cube, observation)
    cube_motion_while_closed = torch.maximum(cube_motion_while_closed, cube_motion)
    cube_lift_while_closed = torch.maximum(cube_lift_while_closed, cube_lift)

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
        carry_cube = pose_with_quat(desired_cube + pinch_to_cube, gripper_quat)
        observation = record_primitive(
            task_env,
            carry_transform,
            observation,
            make_primitive_td(
                observation,
                URScriptPrimitive.MOVEL,
                target_pose=carry_cube,
                gripper=task_gripper_close,
            ),
        )
        max_reward = torch.maximum(max_reward, primitive_reward(observation))
        cube_motion, cube_lift = closed_motion(closed_reference_cube, observation)
    cube_motion_while_closed = torch.maximum(cube_motion_while_closed, cube_motion)
    cube_lift_while_closed = torch.maximum(cube_lift_while_closed, cube_lift)

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
    observation = record_primitive(
        task_env,
        drop_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=pose_with_quat(drop_cube + pinch_to_cube, gripper_quat),
            gripper=task_gripper_close,
        ),
    )
    max_reward = torch.maximum(max_reward, primitive_reward(observation))
    cube_motion, cube_lift = closed_motion(closed_reference_cube, observation)
    cube_motion_while_closed = torch.maximum(cube_motion_while_closed, cube_motion)
    cube_lift_while_closed = torch.maximum(cube_lift_while_closed, cube_lift)

    observation = record_primitive(
        task_env,
        open_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.OPEN_GRIPPER,
            gripper=task_gripper_open,
        ),
    )
    max_reward = torch.maximum(max_reward, primitive_reward(observation))

    for _ in range(240):
        transition = task_env.step(
            observation.clone().set(
                "action",
                action_from_robot_qpos(observation["robot_qpos"], task_gripper_open),
            )
        )
        observation = step_mdp(transition)
        observation["last_reward"] = transition["next", "reward"]
        max_reward = torch.maximum(max_reward, transition["next", "reward"])

    retreat_xyz = torch.cat(
        [
            observation["pinch_pos"][..., :2],
            torch.full_like(observation["pinch_pos"][..., 2:3], 0.26),
        ],
        dim=-1,
    )
    observation = record_primitive(
        task_env,
        retreat_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=pose_with_quat(retreat_xyz, gripper_quat),
            gripper=task_gripper_open,
        ),
    )
    max_reward = torch.maximum(max_reward, primitive_reward(observation))

    home_target = action_from_robot_qpos(initial_robot_qpos, task_gripper_open)
    observation = record_primitive(
        task_env,
        home_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEJ,
            target_qpos=home_target,
            gripper=task_gripper_open,
        ),
    )
    max_reward = torch.maximum(max_reward, primitive_reward(observation))
    final_wait_action = home_target
    final_wait_steps = 800
    robot_home_tolerance = 0.03
else:
    task_env, _, task_transform, task_recorder, observation = make_video_env(
        seed=4,
        max_episode_steps=700,
        macro_steps=20,
    )
    task_gripper_open = GRIPPER_OPEN
    task_gripper_close = GRIPPER_CLOSE
    initial_robot_qpos = observation["robot_qpos"].clone()
    cube = observation["cube_pos"].clone()
    bowl = observation["bowl_pos"].clone()

    above_cube = pose_at(cube + xyz_like(cube, (0.0, 0.0, 0.14)))
    observation = record_primitive(
        task_env,
        task_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=above_cube,
            gripper=task_gripper_open,
        ),
    )
    grip_cube = pose_at(cube + xyz_like(cube, (0.0, 0.0, 0.02)))
    observation = record_primitive(
        task_env,
        task_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=grip_cube,
            gripper=task_gripper_open,
        ),
    )
    observation = record_primitive(
        task_env,
        task_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.CLOSE_GRIPPER,
            gripper=task_gripper_close,
        ),
    )
    grasp_distance = observation["primitive_min_gripper_cube_distance"].clone()

    lift_cube = pose_at(cube + xyz_like(cube, (0.0, 0.0, 0.18)))
    observation = record_primitive(
        task_env,
        task_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=lift_cube,
            gripper=task_gripper_close,
        ),
    )
    cube_motion_while_closed = observation["primitive_cube_displacement"].clone()
    cube_lift_while_closed = observation["primitive_cube_lift"].clone()
    max_reward = observation["primitive_max_reward"].clone()

    above_bowl = pose_at(bowl + xyz_like(bowl, (0.0, 0.0, 0.18)))
    observation = record_primitive(
        task_env,
        task_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=above_bowl,
            gripper=task_gripper_close,
        ),
    )
    cube_motion_while_closed = torch.maximum(
        cube_motion_while_closed, observation["primitive_cube_displacement"]
    )
    cube_lift_while_closed = torch.maximum(
        cube_lift_while_closed, observation["primitive_cube_lift"]
    )
    max_reward = torch.maximum(max_reward, observation["primitive_max_reward"])

    place_in_bowl = pose_at(bowl + xyz_like(bowl, (0.0, 0.0, 0.08)))
    observation = record_primitive(
        task_env,
        task_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEL,
            target_pose=place_in_bowl,
            gripper=task_gripper_close,
        ),
    )
    observation = record_primitive(
        task_env,
        task_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.OPEN_GRIPPER,
            gripper=task_gripper_open,
        ),
    )
    max_reward = torch.maximum(max_reward, observation["primitive_max_reward"])

    home_target = action_from_robot_qpos(initial_robot_qpos, task_gripper_open)
    observation = record_primitive(
        task_env,
        task_transform,
        observation,
        make_primitive_td(
            observation,
            URScriptPrimitive.MOVEJ,
            target_qpos=home_target,
            gripper=task_gripper_open,
        ),
    )
    final_wait_action = home_target
    final_wait_steps = 80
    robot_home_tolerance = 1.0

for _ in range(final_wait_steps):
    transition = task_env.step(
        observation.clone().set(
            "action",
            final_wait_action,
        )
    )
    observation = step_mdp(transition)
    observation["last_reward"] = transition["next", "reward"]
    max_reward = torch.maximum(max_reward, transition["next", "reward"])

robot_home_error = (observation["robot_qpos"] - initial_robot_qpos).norm(
    dim=-1, keepdim=True
)

# sphinx_gallery_end_ignore

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
assert observation["last_reward"].item() == 1.0
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

# sphinx_gallery_start_ignore

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
        **env_kwargs,
    )

# sphinx_gallery_end_ignore

randomized_envs = [make_randomized_env(index) for index in range(4)]
randomized_observations = [randomized_env.reset() for randomized_env in randomized_envs]
assert not torch.equal(
    randomized_observations[0]["cube_pos"][..., 1],
    randomized_observations[1]["cube_pos"][..., 1],
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
    title="Construction-time randomization of cube and bowl placement",
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
