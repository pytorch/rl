# CartesianSolver

*class*torchrl.envs.transforms.CartesianSolver(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/custom/mujoco/_ur_primitives.html#CartesianSolver)

Contract of the Cartesian inverse-kinematics hook used by `movel`.

A Cartesian solver maps a target end-effector pose to a low-level
joint-position action. It is the sanctioned extension point for custom
inverse-kinematics behavior in the macro-action stack: pass one to
`URScriptPrimitiveTransform` via the
`cartesian_solver` argument, or let the transform fall back to a parent
environment's `_cartesian_pose_to_joint_target` hook (e.g.
[`CubeBowlEnv`](torchrl.envs.CubeBowlEnv.html#torchrl.envs.CubeBowlEnv)).

The call signature is:

```
solver(target_pose, start_action, *, orientation_mask=None, waypoints=None)
```

Parameters:

- **target_pose** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - target end-effector pose of shape
`(*batch, 7)`: three position coordinates followed by a
`(w, x, y, z)` unit quaternion, all in the world frame. A zero
(or otherwise invalid) quaternion means "position only": all three
rotational degrees of freedom are free.
- **start_action** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - current low-level action of shape
`(*batch, action_dim)`. The leading `action_dim - 1` entries
are joint positions used to seed the solve; the trailing entry is
the gripper command and must be copied through unchanged.

Keyword Arguments:

- **orientation_mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - per-axis weights of shape
`(*batch, 3)` applied to the world-frame rotation error. A zero
entry leaves rotation about that world axis unconstrained; e.g.
`(1.0, 1.0, 0.0)` constrains rotations about the world x and y
axes (keep a tool axis parallel to world z, i.e. "stay level")
while leaving the spin about world z free. Non-finite entries mean
"no mask" for that batch element. Solvers that only support the
position-only / full-6D endpoints may omit this parameter from
their signature; the transform then raises when a macro action
requests a partial constraint.
- **waypoints** (*int**,**optional*) - when provided, solve the inverse kinematics
along a straight-line Cartesian path from the current end-effector
pose to `target_pose` and return the whole joint-space sequence
of shape `(*batch, waypoints, action_dim)` instead of a single
endpoint of shape `(*batch, action_dim)`. Constraints (full or
partial orientation) must hold at every waypoint, not only at the
endpoint. Solvers that do not support per-waypoint solving may
omit this parameter; the transform then raises when a macro action
requests `path="cartesian"`.

Returns:

the low-level action(s) realizing the target pose:
`(*batch, action_dim)` without `waypoints`, or
`(*batch, waypoints, action_dim)` with it.

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

A plain two-argument callable `(target_pose, start_action) -> action` is
a valid (endpoint-only, fully-constrained-or-free) solver;
`URScriptPrimitiveTransform` inspects the signature
and only forwards the keyword arguments the solver declares.

Examples

```
>>> import torch
>>> def keep_level_solver(target_pose, start_action, *, orientation_mask=None, waypoints=None):
... # A stub that ignores kinematics and returns the seed action:
... # a real solver would run damped least squares, weighting the
... # world-frame rotation error rows by ``orientation_mask``.
... if waypoints is not None:
... return start_action.unsqueeze(-2).expand(
... *start_action.shape[:-1], waypoints, start_action.shape[-1]
... ).clone()
... return start_action.clone()
>>> from torchrl.envs import CartesianSolver
>>> isinstance(keep_level_solver, CartesianSolver)
True
```