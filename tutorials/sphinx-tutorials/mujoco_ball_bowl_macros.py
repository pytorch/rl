"""
MuJoCo scripted manipulation with macro actions
===============================================

This tutorial introduces a compact MuJoCo manipulation task and shows how to
build high-level robot macros with TorchRL transforms.

What you will learn
-------------------

- how to instantiate a custom MuJoCo environment backed by a compact MJCF;
- how to expand a URScript-style primitive into a fixed-length low-level action
  sequence;
- how to execute that sequence with :class:`~torchrl.envs.MultiAction`;
- how to randomize scene construction by creating environment factories.
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
# By default, ``BallBowlEnv`` uses only primitive MJCF geoms: a UR-style
# six-joint arm, a simple two-finger gripper, a free ball, and a bowl. If you
# have a local MuJoCo Menagerie checkout, set
# ``TORCHRL_MUJOCO_MENAGERIE_PATH`` to use the more detailed UR5e + Robotiq
# 2F-85 model without vendoring the meshes in TorchRL:
#
#   .. code-block:: bash
#
#      git clone --depth=1 --filter=blob:none --sparse \
#          https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
#      git -C /tmp/menagerie sparse-checkout set \
#          universal_robots_ur5e robotiq_2f85
#      export TORCHRL_MUJOCO_MENAGERIE_PATH=/tmp/menagerie
#
# The base action remains a low-level 7D actuator command: six joint-position
# targets and one gripper command.

MENAGERIE_PATH = os.environ.get(BallBowlEnv.MENAGERIE_ENV_VAR)
ROBOT_MODEL = "menagerie_ur5e" if MENAGERIE_PATH else "primitive"
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 255.0 if MENAGERIE_PATH else 0.038

env_kwargs = {
    "robot_model": ROBOT_MODEL,
    "menagerie_path": MENAGERIE_PATH,
}

env = BallBowlEnv(seed=0, max_episode_steps=200, **env_kwargs)
obs = env.reset()
assert obs["robot_qpos"].shape[-1] == 6
assert env.action_spec.shape[-1] == 7

random_rollout = env.rollout(4)
assert random_rollout.get(("next", "reward")).shape[-1] == 1


# %%
# Expanding one primitive by hand
# -------------------------------
#
# A primitive action is stored in a TensorDict. ``primitive_id`` selects the
# macro; ``target_qpos`` is used by ``movej``; ``target_pose`` is used by
# ``movel``; and ``gripper`` optionally overrides the gripper command emitted at
# every low-level step.

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
# Executing primitives with MultiAction
# -------------------------------------
#
# In a transformed environment, inverse transforms run from right to left.
# Therefore ``URScriptPrimitiveTransform`` first expands the policy-facing
# primitive into an action sequence, then ``MultiAction`` unrolls that sequence
# through the base environment.

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
        target_qpos = torch.zeros(*batch_size, 7)
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


observation = macro_env.reset()
ball = observation["ball_pos"]
bowl = observation["bowl_pos"]

waypoints = [
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEL,
        target_pose=pose_at(ball + torch.tensor([[0.0, 0.0, 0.12]])),
        gripper=GRIPPER_OPEN,
    ),
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEL,
        target_pose=pose_at(ball + torch.tensor([[0.0, 0.0, 0.035]])),
        gripper=GRIPPER_OPEN,
    ),
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.CLOSE_GRIPPER,
        gripper=GRIPPER_CLOSE,
    ),
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEL,
        target_pose=pose_at(ball + torch.tensor([[0.0, 0.0, 0.16]])),
        gripper=GRIPPER_CLOSE,
    ),
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.MOVEL,
        target_pose=pose_at(bowl + torch.tensor([[0.0, 0.0, 0.12]])),
        gripper=GRIPPER_CLOSE,
    ),
    make_primitive_td(
        observation,
        URScriptPrimitiveTransform.OPEN_GRIPPER,
        gripper=GRIPPER_OPEN,
    ),
]

for primitive in waypoints:
    primitive.update(observation.select(*macro_env.observation_keys))
    transition = macro_env.step(primitive)
    observation = step_mdp(transition)

assert "success" in observation.keys()


# %%
# Construction-time scene randomization
# -------------------------------------
#
# For simple tasks, a practical first randomization strategy is to build env
# factories with different object placements. Reset-time randomization of static
# scene bodies can be added later by giving those bodies joints or mocap targets.


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

for randomized_env in randomized_envs:
    randomized_env.close()
macro_env.close()
env.close()


# %%
# Conclusion and further reading
# ------------------------------
#
# Macro actions are useful when a policy should choose *what* to do at a slower
# time scale while a scripted controller handles short horizon actuation. The
# same structure can seed imitation datasets, bootstrap residual policies, or
# define curricula for reinforcement learning. For more context, see the API
# references for :class:`~torchrl.envs.MujocoEnv`,
# :class:`~torchrl.envs.URScriptPrimitiveTransform`, and
# :class:`~torchrl.envs.MultiAction`.
