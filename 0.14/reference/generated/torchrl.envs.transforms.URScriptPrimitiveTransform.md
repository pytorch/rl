# URScriptPrimitiveTransform

*class*torchrl.envs.transforms.URScriptPrimitiveTransform(**args: Any*, *execute: bool = False*, *multi_action_dim: int = 1*, *stack_rewards: bool = True*, *stack_observations: bool = False*, ***kwargs: Any*)[[source]](../../_modules/torchrl/envs/custom/mujoco/_ur_primitives.html#URScriptPrimitiveTransform)

URScript-style preset of [`MacroPrimitiveTransform`](torchrl.envs.transforms.MacroPrimitiveTransform.html#torchrl.envs.transforms.MacroPrimitiveTransform).

This specialization is scoped to six-joint UR-style arms with a scalar
gripper command. The policy-facing action is a [`RobotMacroAction`](torchrl.envs.transforms.RobotMacroAction.html#torchrl.envs.transforms.RobotMacroAction)
placed under `action_key`. The transform reads the arm and gripper joint
observations to build the interpolation start, maps each primitive to a
seven-dimensional joint-position + gripper destination (running Cartesian
inverse kinematics for `reach_pose`), and delegates fixed-length
interpolation / execution to the generic base.

Parameters:

- **execute** - if `True`, return `Compose(MultiAction(...), transform)`.
- **action_key** - key carrying the macro action and the expanded low-level
sequence.
- **robot_qpos_key** - observation key for the six arm joints.
- **gripper_qpos_key** - observation key for the gripper joints.
- **macro_steps** - interpolated low-level actions per primitive.
- **settle_steps** - repeated final actions appended per primitive.
- **action_dim** - low-level action dimension (six joints + one gripper).
- **cartesian_solver** - optional `CartesianSolver`
mapping `(target_pose, start_action)` to a low-level action.
A plain two-argument callable is accepted; the optional
`orientation_mask` and `waypoints` keyword arguments are only
forwarded when the solver declares them (they are required for
[`RobotMacroAction.reach_pose()`](torchrl.envs.transforms.RobotMacroAction.html#torchrl.envs.transforms.RobotMacroAction.reach_pose) partial orientation
constraints and `path="cartesian"` respectively). When omitted,
the transform uses a parent env's
`_cartesian_pose_to_joint_target` hook.
- **open_gripper_ctrl** - low-level gripper command for an open gripper.
- **close_gripper_ctrl** - low-level gripper command for a closed gripper.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs import RobotMacroAction, URScriptPrimitiveTransform
>>> td = TensorDict({
... "action": RobotMacroAction.reach_joints(joints=torch.ones(1, 6), steps=2),
... "robot_qpos": torch.zeros(1, 6),
... "gripper_qpos": torch.zeros(1, 2),
... }, batch_size=[1])
>>> URScriptPrimitiveTransform().inv(td)["action"].shape
torch.Size([1, 2, 7])
```

action_sequence(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *primitive_id: int | IntEnum | None = None*, ***, *target_pose: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *target_qpos: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *gripper: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *steps: int | None = None*, *settle_steps: int | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/custom/mujoco/_ur_primitives.html#URScriptPrimitiveTransform.action_sequence)

Expand a UR primitive into its low-level sequence without executing.

current_action(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *batch_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*, *action_dim: int*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/custom/mujoco/_ur_primitives.html#URScriptPrimitiveTransform.current_action)

Return the low-level action used as the interpolation start.

The base implementation starts every macro from the zero action: in the
inverse path `action_key` carries the incoming macro action (the
*target*), so it must not be read back here as the start. Subclasses that
can read the controlled state from observations (e.g. joint positions)
override this hook.

low_level_action(*robot_qpos: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *gripper: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/custom/mujoco/_ur_primitives.html#URScriptPrimitiveTransform.low_level_action)

Build a low-level joint-position + gripper action.

make_primitive(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *primitive_id: int | IntEnum*, ***, *target_pose: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *target_qpos: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *gripper: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *steps: int | None = None*, *settle_steps: int | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/custom/mujoco/_ur_primitives.html#URScriptPrimitiveTransform.make_primitive)

Return a copy of `tensordict` carrying one [`RobotMacroAction`](torchrl.envs.transforms.RobotMacroAction.html#torchrl.envs.transforms.RobotMacroAction).

Maps a URScript primitive id (and its pose / joint / gripper arguments)
onto a [`RobotMacroAction`](torchrl.envs.transforms.RobotMacroAction.html#torchrl.envs.transforms.RobotMacroAction) placed under `action_key`.

primitive_enum

alias of [`URScriptPrimitive`](torchrl.envs.transforms.URScriptPrimitive.html#torchrl.envs.transforms.URScriptPrimitive)

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/custom/mujoco/_ur_primitives.html#URScriptPrimitiveTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform