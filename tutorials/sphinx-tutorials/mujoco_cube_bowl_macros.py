"""
MuJoCo scripted manipulation with human-readable robot actions
==============================================================

The first step in controlling a complex robot is rarely an end-to-end RL
policy. Closed-loop controllers, scripted motion primitives, and imitation data
are often used first to make the task reliable enough for later reward-driven
fine tuning. This tutorial uses a MuJoCo Menagerie UR5e arm with a Robotiq
2F-85 gripper and shows how to ask the robot to do human-readable things such
as "reach this pose", "close the gripper", or "go home" while TorchRL handles
the low-level action plumbing.

We will control a cube-to-bowl task without learning. A policy emits a single
``RobotAction`` object under the normal ``"action"`` key, and a TorchRL
transform expands that high-level command into the low-level actuator sequence
consumed by MuJoCo. The same pattern can later provide demonstrations,
curricula, or a safe initialization for residual RL.

What you will learn
-------------------

- how to instantiate a Menagerie-backed custom MuJoCo environment;
- how to write a small TensorDict-backed ``RobotAction`` with readable
  factories;
- how :class:`~torchrl.envs.URScriptPrimitiveTransform` turns those commands
  into fixed-length low-level actions;
- how to write a scripted contact-rich cube-to-bowl policy as an explicit list
  of poses and gripper commands;
- how to reset and reuse one environment across scripted scenes.
"""

from __future__ import annotations

import importlib.util
import os
from typing import ClassVar, Iterator, Literal

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import CubeBowlEnv, URScriptPrimitive, step_mdp

_has_mujoco = importlib.util.find_spec("mujoco") is not None

if not _has_mujoco:
    raise ImportError("This tutorial requires the `mujoco` Python package.")


# %%
# Environment
# -----------
#
# The scene uses the MuJoCo Menagerie UR5e and Robotiq 2F-85 models. Set
# ``TORCHRL_MUJOCO_MENAGERIE_PATH`` to a checkout containing the
# ``universal_robots_ur5e`` and ``robotiq_2f85`` assets:
#
#   .. code-block:: bash
#
#      git clone --depth=1 --filter=blob:none --sparse \
#          https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
#      git -C /tmp/menagerie sparse-checkout set \
#          universal_robots_ur5e robotiq_2f85
#      export TORCHRL_MUJOCO_MENAGERIE_PATH=/tmp/menagerie
#
# :class:`~torchrl.envs.CubeBowlEnv` exposes the state needed for scripted
# manipulation: robot joints, gripper joints, the pinch site, the cube pose, the
# bowl target and a success flag. The task reward is intentionally sparse. It is
# ``1`` only when the cube center is within the environment's placement
# tolerance of the bowl target coordinate, and ``0`` otherwise.

MENAGERIE_PATH = os.environ.get(CubeBowlEnv.MENAGERIE_ENV_VAR)
if MENAGERIE_PATH is None:
    raise RuntimeError(
        f"Set {CubeBowlEnv.MENAGERIE_ENV_VAR} to a MuJoCo Menagerie checkout "
        "before running this tutorial."
    )

MAX_EPISODE_STEPS = 12000
IK_KWARGS = {
    "iterations": 220,
    "orientation_weight": 1.0,
    "step_size": 0.7,
    "damping": 1e-4,
}


# %%
# A human-readable RobotAction
# ----------------------------
#
# A readable manipulation policy should not have to emit magic flat tensors or
# loose collections of root TensorDict keys. It can emit one structured object
# under ``"action"``. The factories below are the API a human writes:
#
# - ``RobotAction.reach_pose(position=..., quaternion=...)``;
# - ``RobotAction.close_gripper()``;
# - ``RobotAction.reach_pose(..., gripper="closed")``;
# - ``RobotAction.home(joints=initial_robot_qpos, gripper="open")``.
#
# Each factory returns a structured TensorDict. ``mode`` says what primitive
# should run, ``position`` and ``quaternion`` define Cartesian targets,
# ``joints`` defines joint-space targets, and ``gripper`` is a symbolic code:
# ``"keep"``, ``"open"`` or ``"closed"``. The Menagerie-specific numeric
# gripper command is applied by the transform helper, not by user code.

GripperCommand = Literal["keep", "open", "closed"]


class RobotAction:
    WAIT: ClassVar[int] = int(URScriptPrimitive.WAIT)
    REACH_JOINTS: ClassVar[int] = int(URScriptPrimitive.MOVEJ)
    REACH_POSE: ClassVar[int] = int(URScriptPrimitive.MOVEL)
    OPEN_GRIPPER: ClassVar[int] = int(URScriptPrimitive.OPEN_GRIPPER)
    CLOSE_GRIPPER: ClassVar[int] = int(URScriptPrimitive.CLOSE_GRIPPER)

    GRIPPER_KEEP: ClassVar[int] = -1
    GRIPPER_OPEN: ClassVar[int] = 0
    GRIPPER_CLOSED: ClassVar[int] = 1

    @classmethod
    def reach_pose(
        cls,
        *,
        position: torch.Tensor,
        quaternion: torch.Tensor | None = None,
        gripper: GripperCommand = "keep",
        steps: int = 16,
        settle_steps: int = 0,
    ) -> TensorDictBase:
        """Ask the end effector to reach a Cartesian pose."""
        position = _as_batch(position, 3)
        if quaternion is None:
            quaternion = _identity_quaternion_like(position)
        else:
            quaternion = _as_batch(quaternion, 4).to(
                dtype=position.dtype, device=position.device
            )
        return cls._make(
            mode=cls.REACH_POSE,
            position=position,
            quaternion=quaternion,
            gripper=gripper,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def reach_joints(
        cls,
        *,
        joints: torch.Tensor,
        gripper: GripperCommand = "keep",
        steps: int = 16,
        settle_steps: int = 0,
    ) -> TensorDictBase:
        """Ask the robot arm to reach a joint configuration."""
        joints = _as_batch(joints, 6)
        return cls._make(
            mode=cls.REACH_JOINTS,
            joints=joints,
            gripper=gripper,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def home(
        cls,
        *,
        joints: torch.Tensor,
        gripper: GripperCommand = "open",
        steps: int = 16,
        settle_steps: int = 0,
    ) -> TensorDictBase:
        """Return to a saved reset joint configuration."""
        return cls.reach_joints(
            joints=joints,
            gripper=gripper,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def open_gripper(
        cls,
        *,
        steps: int = 16,
        settle_steps: int = 0,
    ) -> TensorDictBase:
        """Open the gripper without changing the arm target."""
        return cls._make(
            mode=cls.OPEN_GRIPPER,
            gripper="open",
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def close_gripper(
        cls,
        *,
        steps: int = 16,
        settle_steps: int = 0,
    ) -> TensorDictBase:
        """Close the gripper without changing the arm target."""
        return cls._make(
            mode=cls.CLOSE_GRIPPER,
            gripper="closed",
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def wait(
        cls,
        *,
        gripper: GripperCommand = "keep",
        steps: int = 1,
        settle_steps: int = 0,
    ) -> TensorDictBase:
        """Repeat the current target for a number of simulator steps."""
        return cls._make(
            mode=cls.WAIT,
            gripper=gripper,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def _make(
        cls,
        *,
        mode: int,
        position: torch.Tensor | None = None,
        quaternion: torch.Tensor | None = None,
        joints: torch.Tensor | None = None,
        gripper: GripperCommand = "keep",
        steps: int = 16,
        settle_steps: int = 0,
    ) -> TensorDictBase:
        if position is not None:
            position = _as_batch(position, 3)
            batch_size = position.shape[:-1]
            dtype = position.dtype
            device = position.device
        elif joints is not None:
            joints = _as_batch(joints, 6)
            batch_size = joints.shape[:-1]
            dtype = joints.dtype
            device = joints.device
        else:
            batch_size = torch.Size([1])
            dtype = torch.get_default_dtype()
            device = torch.device("cpu")
            position = torch.zeros(batch_size + (3,), dtype=dtype, device=device)

        if position is None:
            position = torch.zeros(batch_size + (3,), dtype=dtype, device=device)
        if quaternion is None:
            quaternion = _identity_quaternion_like(position)
        else:
            quaternion = _as_batch(quaternion, 4).to(dtype=dtype, device=device)
        if joints is None:
            joints = torch.zeros(batch_size + (6,), dtype=dtype, device=device)

        return TensorDict(
            {
                "mode": torch.full(
                    batch_size + (1,), mode, dtype=torch.long, device=device
                ),
                "position": position,
                "quaternion": quaternion,
                "joints": joints,
                "gripper": torch.full(
                    batch_size + (1,),
                    _gripper_code(gripper),
                    dtype=torch.long,
                    device=device,
                ),
                "steps": torch.full(
                    batch_size + (1,), steps, dtype=torch.long, device=device
                ),
                "settle_steps": torch.full(
                    batch_size + (1,),
                    settle_steps,
                    dtype=torch.long,
                    device=device,
                ),
            },
            batch_size=batch_size,
        )


def _as_batch(value: torch.Tensor, last_dim: int) -> torch.Tensor:
    if value.shape[-1] != last_dim:
        raise ValueError(f"Expected trailing dimension {last_dim}, got {value.shape}.")
    if value.ndim == 1:
        return value.unsqueeze(0)
    return value


def _identity_quaternion_like(position: torch.Tensor) -> torch.Tensor:
    quaternion = torch.zeros(
        position.shape[:-1] + (4,), dtype=position.dtype, device=position.device
    )
    quaternion[..., 0] = 1.0
    return quaternion


def _gripper_code(gripper: GripperCommand) -> int:
    if gripper == "keep":
        return RobotAction.GRIPPER_KEEP
    if gripper == "open":
        return RobotAction.GRIPPER_OPEN
    if gripper == "closed":
        return RobotAction.GRIPPER_CLOSED
    raise ValueError(f"Unknown gripper command {gripper!r}.")


# %%
# Create the MuJoCo scene
# -----------------------
#
# We start by constructing the cube-and-bowl task once. The environment owns the
# MuJoCo model, action spec, reset logic and helper methods that translate
# robot-arm commands into low-level actuator targets.

env = CubeBowlEnv(
    seed=0,
    max_episode_steps=MAX_EPISODE_STEPS,
    robot_model="menagerie_ur5e",
    menagerie_path=MENAGERIE_PATH,
)
observation = env.reset()
assert observation["robot_qpos"].shape[-1] == 6
assert env.action_spec.shape[-1] == 7
assert env.low_level_action(observation["robot_qpos"]).shape[-1] == 7


# %%
# Primitive expansion
# -------------------
#
# ``URScriptPrimitiveTransform`` expands a structured ``RobotAction`` into a
# low-level action sequence. The transform also supports ``execute=True`` for
# direct use in :class:`~torchrl.envs.TransformedEnv`. Keeping the expansion
# visible is useful when we want diagnostics for each low-level simulator step.

primitive_transform = env.make_urscript_transform(
    macro_steps=16,
    ik_kwargs=IK_KWARGS,
)
example_action = RobotAction.reach_pose(
    position=observation["cube_pos"]
    + observation["cube_pos"].new_tensor([[0.0, 0.0, 0.12]]),
    quaternion=observation["pinch_quat"],
    gripper="open",
    steps=20,
)
assert example_action["position"].shape[-1] == 3
assert primitive_transform.action_sequence(
    observation.clone().set("action", example_action)
).shape[-1] == 7


# %%
# Policy shape
# ------------
#
# A scripted policy can be just a generator of explicit robot commands. This is
# the minimal shape we want users to see: build a list of poses and gripper
# commands, then have the policy place the next ``RobotAction`` under the normal
# action key. ``TensorDict.set`` is the single-entry form of replacing
# ``td["action"]``.


def gen_small_actions(td: TensorDictBase) -> Iterator[TensorDictBase]:
    joints = td["robot_qpos"].clone()
    joints[..., 0] = joints[..., 0] + 0.35
    yield RobotAction.reach_joints(joints=joints, gripper="open", steps=18)
    yield RobotAction.close_gripper(steps=18)


small_actions = gen_small_actions(observation)


def small_policy(td: TensorDictBase) -> TensorDictBase:
    return td.set("action", next(small_actions))


# %%
# Action execution helper
# -----------------------
#
# The helper below is deliberately small: it reads ``td["action"]``, expands the
# command with the primitive transform, executes the resulting low-level action
# sequence, and returns diagnostics used by the assertions. Applications that do
# not need per-low-level-step diagnostics can instead construct
# ``URScriptPrimitiveTransform(..., execute=True)`` inside a transformed env.


def step_low_level_action(
    td: TensorDictBase,
    action: torch.Tensor,
) -> tuple[TensorDictBase, TensorDictBase]:
    transition = env.step(td.clone().set("action", action))
    next_td = step_mdp(transition)
    return next_td, transition


def run_robot_policy(
    td: TensorDictBase,
    policy_fn,
) -> tuple[TensorDictBase, TensorDictBase, TensorDictBase]:
    action_td = policy_fn(td.clone())
    robot_action = action_td["action"]
    mode = int(robot_action["mode"].reshape(-1)[0].item())
    steps = int(robot_action["steps"].reshape(-1)[0].item())
    settle_steps = int(robot_action["settle_steps"].reshape(-1)[0].item())
    gripper_code = int(robot_action["gripper"].reshape(-1)[0].item())
    if gripper_code == RobotAction.GRIPPER_OPEN:
        gripper = env.gripper_open_ctrl
    elif gripper_code == RobotAction.GRIPPER_CLOSED:
        gripper = env.gripper_close_ctrl
    else:
        gripper = None

    transform = env.make_urscript_transform(
        macro_steps=steps,
        settle_steps=settle_steps,
        ik_kwargs=IK_KWARGS,
    )
    if mode == RobotAction.REACH_POSE:
        sequence = transform.action_sequence(
            td,
            URScriptPrimitive.MOVEL,
            target_pose=torch.cat(
                [robot_action["position"], robot_action["quaternion"]], dim=-1
            ),
            gripper=gripper,
        )
    elif mode == RobotAction.REACH_JOINTS:
        target_qpos = env.low_level_action(robot_action["joints"], gripper)
        sequence = transform.action_sequence(
            td,
            URScriptPrimitive.MOVEJ,
            target_qpos=target_qpos,
            gripper=gripper,
        )
    elif mode == RobotAction.OPEN_GRIPPER:
        sequence = transform.action_sequence(
            td, URScriptPrimitive.OPEN_GRIPPER, gripper=gripper
        )
    elif mode == RobotAction.CLOSE_GRIPPER:
        sequence = transform.action_sequence(
            td, URScriptPrimitive.CLOSE_GRIPPER, gripper=gripper
        )
    else:
        sequence = transform.action_sequence(td, URScriptPrimitive.WAIT, gripper=gripper)

    start_cube = td["cube_pos"].clone()
    start_cube_z = start_cube[..., 2:3].clone()
    min_gripper_distance = torch.full_like(start_cube[..., :1], float("inf"))
    max_cube_displacement = torch.zeros_like(start_cube[..., :1])
    max_cube_lift = torch.zeros_like(start_cube[..., :1])
    max_action_reward = torch.zeros_like(start_cube[..., :1])
    action_last_reward = torch.zeros_like(start_cube[..., :1])

    for low_level_action in sequence[0]:
        td, transition = step_low_level_action(td, low_level_action.view(1, 7))
        min_gripper_distance = torch.minimum(
            min_gripper_distance, env.gripper_cube_distance(td)
        )
        cube_displacement = (td["cube_pos"] - start_cube).norm(dim=-1, keepdim=True)
        cube_lift = td["cube_pos"][..., 2:3] - start_cube_z
        max_cube_displacement = torch.maximum(max_cube_displacement, cube_displacement)
        max_cube_lift = torch.maximum(max_cube_lift, cube_lift)
        action_last_reward = transition["next", "reward"]
        max_action_reward = torch.maximum(max_action_reward, action_last_reward)

    metrics = TensorDict(
        {
            "min_gripper_cube_distance": min_gripper_distance,
            "cube_displacement": max_cube_displacement,
            "cube_lift": max_cube_lift,
            "max_reward": max_action_reward,
            "last_reward": action_last_reward,
        },
        batch_size=td.batch_size,
    )
    return td, robot_action, metrics


# %%
# A scripted cube-to-bowl macro
# -----------------------------
#
# We can now write the cube-to-bowl controller as a readable sequence of manual
# robot commands. The generator below is deliberately explicit: it names the
# poses we want the pinch site to reach and interleaves them with gripper
# commands.

initial_robot_qpos = observation["robot_qpos"].clone()
grasp_distance = torch.full_like(observation["cube_pos"][..., :1], float("inf"))
cube_motion_while_closed = torch.zeros_like(grasp_distance)
cube_lift_while_closed = torch.zeros_like(grasp_distance)
max_reward = torch.zeros_like(grasp_distance)
last_reward = torch.zeros_like(grasp_distance)
closed_reference_cube: torch.Tensor | None = None
policy_state: dict[str, TensorDictBase] = {"td": observation}


def pose_at(
    xyz: torch.Tensor, quat: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if quat is None:
        quat = _identity_quaternion_like(xyz)
    else:
        quat = quat.to(dtype=xyz.dtype, device=xyz.device).expand(
            xyz.shape[:-1] + (4,)
        )
    return xyz, quat


def gen_actions(td: TensorDictBase) -> Iterator[TensorDictBase]:
    cube = td["cube_pos"].clone()
    bowl = td["bowl_pos"].clone()
    quat = td["pinch_quat"].clone()
    grasp_offset = cube.new_tensor([[0.0, 0.0, -0.016]])

    # Action 0: Keep the arm at the reset joint target and let the scene settle.
    yield RobotAction.reach_joints(
        joints=td["robot_qpos"].clone(),
        gripper="open",
        steps=1,
        settle_steps=19,
    )
    # Action 1: Fully open the gripper before approaching the cube.
    yield RobotAction.open_gripper(steps=100, settle_steps=20)

    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    bowl = current_td["bowl_pos"].clone()
    quat = current_td["pinch_quat"].clone()
    position, quaternion = pose_at(cube + cube.new_tensor([[0.0, 0.0, 0.18]]), quat)
    # Action 2: Move the open gripper above the cube.
    yield RobotAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="open",
        steps=180,
        settle_steps=60,
    )

    position, quaternion = pose_at(cube + grasp_offset, quat)
    # Action 3: Lower the open gripper to the grasp pose around the cube.
    yield RobotAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="open",
        steps=180,
        settle_steps=60,
    )

    # Action 4: Close the gripper to grasp the cube.
    yield RobotAction.close_gripper(steps=160, settle_steps=80)

    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    pinch_to_cube = current_td["pinch_pos"].clone() - cube
    position, quaternion = pose_at(
        cube + pinch_to_cube + cube.new_tensor([[0.0, 0.0, 0.20]]),
        quat,
    )
    # Action 5: Lift the grasped cube above the table.
    yield RobotAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="closed",
        steps=120,
        settle_steps=60,
    )

    current_td = policy_state["td"]
    start_y = current_td["cube_pos"][..., 1:2].clone()
    target_y = bowl[..., 1:2]

    # Action 6: Carry the cube one quarter of the way toward the bowl.
    alpha = 0.25
    desired_cube = torch.cat(
        [
            bowl[..., :1],
            start_y + alpha * (target_y - start_y),
            torch.full_like(bowl[..., 2:3], 0.24),
        ],
        dim=-1,
    )
    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    pinch_to_cube = current_td["pinch_pos"].clone() - cube
    position, quaternion = pose_at(desired_cube + pinch_to_cube, quat)
    yield RobotAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="closed",
        steps=80,
        settle_steps=20,
    )

    # Action 7: Carry the cube halfway toward the bowl.
    alpha = 0.50
    desired_cube = torch.cat(
        [
            bowl[..., :1],
            start_y + alpha * (target_y - start_y),
            torch.full_like(bowl[..., 2:3], 0.24),
        ],
        dim=-1,
    )
    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    pinch_to_cube = current_td["pinch_pos"].clone() - cube
    position, quaternion = pose_at(desired_cube + pinch_to_cube, quat)
    yield RobotAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="closed",
        steps=80,
        settle_steps=20,
    )

    # Action 8: Carry the cube three quarters of the way toward the bowl.
    alpha = 0.75
    desired_cube = torch.cat(
        [
            bowl[..., :1],
            start_y + alpha * (target_y - start_y),
            torch.full_like(bowl[..., 2:3], 0.24),
        ],
        dim=-1,
    )
    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    pinch_to_cube = current_td["pinch_pos"].clone() - cube
    position, quaternion = pose_at(desired_cube + pinch_to_cube, quat)
    yield RobotAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="closed",
        steps=80,
        settle_steps=20,
    )

    # Action 9: Carry the cube above the bowl center.
    alpha = 1.0
    desired_cube = torch.cat(
        [
            bowl[..., :1],
            start_y + alpha * (target_y - start_y),
            torch.full_like(bowl[..., 2:3], 0.24),
        ],
        dim=-1,
    )
    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    pinch_to_cube = current_td["pinch_pos"].clone() - cube
    position, quaternion = pose_at(desired_cube + pinch_to_cube, quat)
    yield RobotAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="closed",
        steps=80,
        settle_steps=20,
    )

    drop_cube = torch.cat(
        [
            bowl[..., :1],
            bowl[..., 1:2],
            torch.full_like(bowl[..., 2:3], 0.13),
        ],
        dim=-1,
    )
    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    pinch_to_cube = current_td["pinch_pos"].clone() - cube
    position, quaternion = pose_at(drop_cube + pinch_to_cube, quat)
    # Action 10: Lower the grasped cube into the bowl.
    yield RobotAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="closed",
        steps=100,
        settle_steps=40,
    )

    # Action 11: Open the gripper to release the cube.
    yield RobotAction.open_gripper(steps=100, settle_steps=20)
    # Action 12: Wait with the gripper open so the cube can settle in the bowl.
    yield RobotAction.wait(gripper="open", steps=240)

    current_td = policy_state["td"]
    retreat_xyz = torch.cat(
        [
            current_td["pinch_pos"][..., :2],
            torch.full_like(current_td["pinch_pos"][..., 2:3], 0.26),
        ],
        dim=-1,
    )
    position, quaternion = pose_at(retreat_xyz, quat)
    # Action 13: Retreat upward and away from the released cube.
    yield RobotAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="open",
        steps=120,
        settle_steps=60,
    )

    # Action 14: Return the arm to its reset joint configuration.
    yield RobotAction.home(
        joints=initial_robot_qpos,
        gripper="open",
        steps=250,
        settle_steps=1600,
    )
    # Action 15: Hold the reset pose while the final reward is measured.
    yield RobotAction.wait(gripper="open", steps=800)


def update_closed_motion(reference_cube: torch.Tensor, td: TensorDictBase) -> None:
    global cube_motion_while_closed, cube_lift_while_closed
    cube_motion = (td["cube_pos"] - reference_cube).norm(dim=-1, keepdim=True)
    cube_lift = td["cube_pos"][..., 2:3] - reference_cube[..., 2:3]
    cube_motion_while_closed = torch.maximum(cube_motion_while_closed, cube_motion)
    cube_lift_while_closed = torch.maximum(cube_lift_while_closed, cube_lift)


script_actions = gen_actions(observation)


def policy(td: TensorDictBase) -> TensorDictBase:
    policy_state["td"] = td
    action = next(script_actions)
    return td.set("action", action)


while True:
    try:
        observation, action, metrics = run_robot_policy(observation, policy)
    except StopIteration:
        break
    max_reward = torch.maximum(max_reward, metrics["max_reward"])
    last_reward = metrics["last_reward"]

    mode = int(action["mode"].reshape(-1)[0].item())
    gripper = int(action["gripper"].reshape(-1)[0].item())
    if mode == RobotAction.CLOSE_GRIPPER:
        grasp_distance = torch.minimum(
            grasp_distance, metrics["min_gripper_cube_distance"]
        )
        closed_reference_cube = observation["cube_pos"].clone()
    if closed_reference_cube is not None and gripper == RobotAction.GRIPPER_CLOSED:
        cube_motion_while_closed = torch.maximum(
            cube_motion_while_closed, metrics["cube_displacement"]
        )
        cube_lift_while_closed = torch.maximum(
            cube_lift_while_closed, metrics["cube_lift"]
        )
        update_closed_motion(closed_reference_cube, observation)

robot_home_error = (observation["robot_qpos"] - initial_robot_qpos).norm(
    dim=-1, keepdim=True
)

assert grasp_distance.item() <= 0.025, grasp_distance.item()
assert cube_motion_while_closed.item() >= 0.05, cube_motion_while_closed.item()
assert cube_lift_while_closed.item() >= 0.08, cube_lift_while_closed.item()
assert robot_home_error.item() <= 0.04, robot_home_error.item()
assert max_reward.item() == 1.0, (
    max_reward.item(),
    observation["cube_pos"],
    observation["bowl_pos"],
)
assert last_reward.item() == 1.0, last_reward.item()
assert observation["success"].all()


# %%
# Reset-time randomization with the same environment
# --------------------------------------------------
#
# A calibrated script is a useful baseline, but policies eventually need to
# survive heterogeneous scenes. We can already vary the cube placement without
# creating another environment by passing a new ``cube_pos`` at reset time.

randomized_observations = []
for index in range(4):
    offset = 0.025 * float(index)
    cube_pos = torch.as_tensor(
        env.cube_position,
        dtype=observation["cube_pos"].dtype,
        device=observation["cube_pos"].device,
    ).view(1, 3)
    cube_pos[..., 1] = cube_pos[..., 1] + offset
    randomized_observations.append(
        env.reset(TensorDict({"cube_pos": cube_pos}, batch_size=[1]))
    )
assert not torch.equal(
    randomized_observations[0]["cube_pos"][..., 1],
    randomized_observations[1]["cube_pos"][..., 1],
)


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

env.close()
