.. currentmodule:: torchrl.envs

.. _macro-primitives-guide:

Macro-control primitives
========================

Macro-control transforms let a policy emit a *destination* or a *semantic
command* at a slow time scale while TorchRL expands it into a sequence of
ordinary low-level actions before the base environment sees it. They are useful
for scripted baselines, demonstrations, curriculum tasks, and policies that
should choose *what to do next* rather than every actuator value at every
simulator step.

This page gives the design view behind :class:`MacroPrimitiveTransform` and
:class:`URScriptPrimitiveTransform`. For a complete manipulation walkthrough,
see the :doc:`cube-to-bowl tutorial <../tutorials/mujoco_cube_bowl_macros>`.

Choosing a macro transform
--------------------------

Given a new environment, first identify the boundary between the policy-facing
action and the action consumed by ``env.step``.

1. **The policy already knows a low-level action destination.**
   Use :class:`MacroPrimitiveTransform` with :class:`MacroAction`. The transform
   only interpolates from the current action to the requested target and, if
   ``execute=True``, :class:`MultiAction` executes the resulting sequence.

2. **The policy wants a task-space or semantic destination.**
   Keep the policy-facing action semantic, and add an adapter or solver that
   maps the semantic target to a low-level action target. The generic transform
   supports this through ``adapter`` and ``solver`` hooks. For simple MuJoCo
   Cartesian commands, ``solver="mujoco_dls_ik"`` can call an environment
   method such as ``_cartesian_pose_to_joint_target``.

3. **The environment has a stable domain vocabulary.**
   If many examples should use the same command language, provide a specialized
   preset. :class:`URScriptPrimitiveTransform` is one such preset: it extends
   the generic macro transform with UR-style arm primitives and gripper
   commands while keeping that robot-specific logic out of
   :class:`MacroPrimitiveTransform`.

4. **The policy emits an already-expanded action sequence.**
   Use :class:`MultiAction` directly, or use a macro transform with
   ``execute=False`` to inspect the sequence and then decide how to execute it.

The main rule is that policies should normally place an explicit action object
or primitive TensorDict under ``td["action"]`` and call standard TorchRL APIs
(``rollout``, ``step`` or ``step_and_maybe_reset``). Calling
``transform.action_sequence(...)`` is useful for debugging and tests, but it is
not the usual policy interface.

The generic data path
---------------------

A macro transform is an inverse action transform. It runs before the base
environment step:

.. code-block:: text

   policy action
      |
      v
   MacroAction, RobotAction or primitive TensorDict
      |
      v
   MacroPrimitiveTransform.inv(...)
      |
      v
   low-level action sequence, shape (..., T, action_dim)
      |
      v
   MultiAction executes T base-env steps when execute=True

With ``execute=True``, the constructor returns a :class:`Compose` containing
``MultiAction`` and the primitive transform. With ``execute=False``, the
transform only expands the macro action and leaves execution to the caller.

What is the target tensor?
--------------------------

The most important design point is that a macro target is not automatically a
pose, a quaternion, or a named simulator state. It is the quantity that the
transform knows how to turn into the base environment's low-level action.

For :meth:`MacroAction.reach_action`, the target is already a destination in the
base environment action coordinates. Its trailing dimension, dtype, device and
valid range come from ``base_env.action_spec``. The observation and state specs
are the companion lookup tables for semantic quantities that can be used to
compute that action-space target:

.. code-block:: python

   low_level_spec = base_env.action_spec
   target_shape = low_level_spec.shape
   target_dtype = low_level_spec.dtype
   target_device = low_level_spec.device
   observation_spec = base_env.observation_spec
   state_spec = base_env.state_spec

.. code-block:: python

   target_action = low_level_spec.zero()
   values = torch.as_tensor(
       [0.16, -0.14, 0.10, -0.10, 0.08, -0.08],
       dtype=target_action.dtype,
       device=target_action.device,
   )
   target_action[..., :values.numel()] = values
   target_action = low_level_spec.project(target_action)

   td["action"] = MacroAction.reach_action(
       target_action,
       steps=24,
       settle_steps=8,
   )

The transform-facing action spec also exposes the raw primitive fields. After a
macro transform is appended, ``env.full_action_spec`` contains keys such as
``primitive_id``, ``target_qpos`` and ``target_pose``. These specs are useful
when you want to emit primitive TensorDicts directly instead of using the
structured helpers :class:`MacroAction` or :class:`RobotAction`.

The environment observation and state specs tell you which semantic quantities
are available to build the target. In many tasks the target can be derived from
observations: a cube position is an observation, an attitude error is an
observation, or the current robot joint state is an observation. The target does
not have to be an observation itself, though. For a humanoid, a low-level action
target is simply an actuator-control destination. If you want a body pose
instead, you must add a controller, an IK solver, or a domain-specific adapter
that maps that pose to the low-level action space.

Example 1: humanoid open-loop poses
-----------------------------------

The humanoid example has no robot-specific adapter. Each macro action is a
low-level actuator destination expressed directly in the humanoid action space.
In the example, ``target_action`` is therefore not a humanoid body pose: it is a
tensor with the same shape and dtype as ``base_env.action_spec.zero()``. Its
entries are MuJoCo actuator control destinations, and the transform interpolates
between the current control vector and that destination.

.. code-block:: python

   target_action = base_env.action_spec.zero()
   target_action[..., :6] = torch.as_tensor(
       [0.16, -0.14, 0.10, -0.10, 0.08, -0.08],
       dtype=target_action.dtype,
       device=target_action.device,
   )
   target_action = base_env.action_spec.project(target_action)

   td["action"] = MacroAction.reach_action(
       target_action,
       steps=24,
       settle_steps=8,
   )

The transform is generic:

.. code-block:: python

   env = base_env.append_transform(
       MacroPrimitiveTransform(
           action_dim=base_env.action_spec.shape[-1],
           execute=True,
           stack_rewards=True,
           stack_observations=False,
       )
   )

This is the minimal pattern: inspect ``base_env.action_spec``, build a valid
low-level target with ``zero()`` and ``project(...)``, wrap it in
:class:`MacroAction`, then execute the environment through normal TorchRL
rollout logic.

.. raw:: html

   <video controls muted loop playsinline width="480">
     <source src="../_static/videos/macro_humanoid.mp4" type="video/mp4">
   </video>

Example 2: satellite attitude slews
-----------------------------------

The satellite policy is still using :meth:`MacroAction.reach_action`, so its
target must also live in the low-level action space. For
:class:`SatelliteEnv`, the low-level action is the normalized commanded rate of
each control-moment-gyroscope gimbal. With the default four-CMG model,
``base_env.action_spec`` has trailing dimension four and bounded values in
``[-1, 1]``. The physical gimbal-rate command is obtained by scaling that
action inside the environment.

The target attitude itself is not the action. It is reset state named
``target_quat``. The environment exposes observations that make the target
usable by a policy:

* ``quat_err`` has shape ``(..., 3)`` and is the logarithmic attitude error from
  the current bus attitude to ``target_quat``.
* ``bus_omega`` has shape ``(..., 3)`` and is the current bus angular velocity.
* ``gimbal_angles`` has shape ``(..., 2 * num_cmgs)`` and stores a sin/cos
  encoding of the current gimbal angles.

The scripted policy computes a desired bus angular acceleration from the
attitude error and angular velocity, maps that through the instantaneous CMG
Jacobian, projects the result into ``base_env.action_spec``, and only then emits
the macro action:

.. code-block:: python

   quat_err = td["quat_err"]
   bus_omega = td["bus_omega"]
   gimbal_obs = td["gimbal_angles"]
   num_gimbals = base_env.action_spec.shape[-1]
   gimbal_angles = torch.atan2(
       gimbal_obs[..., :num_gimbals],
       gimbal_obs[..., num_gimbals:],
   )

   gimbal_axes, rotor_axes_ref = pyramid_4cmg_geometry(
       device=quat_err.device,
       dtype=quat_err.dtype,
   )
   jacobian = cmg_jacobian(gimbal_angles, gimbal_axes, rotor_axes_ref, 1.0)

   desired_bus_accel = attitude_gain * quat_err - rate_gain * bus_omega
   gimbal_rate = -torch.linalg.pinv(jacobian).matmul(
       desired_bus_accel.unsqueeze(-1)
   )
   gimbal_rate_target = base_env.action_spec.project(
       gimbal_rate.squeeze(-1) / base_env.action_scale
   )

   td["action"] = MacroAction.reach_action(
       gimbal_rate_target,
       steps=36,
       settle_steps=8,
   )

Here ``gimbal_rate_target`` has exactly the same trailing shape, dtype and range
as the satellite action spec. The controller is deliberately small and local: it
is enough to demonstrate the macro transform, but it is not a full guidance,
navigation and control stack. In particular, the CMG Jacobian can become poorly
conditioned; the environment exposes ``manipulability`` for diagnostics and a
more robust policy could use it to avoid singular gimbal configurations.

The target attitude frame is part of :class:`SatelliteEnv`: it is a non-colliding
RGB visual frame driven by ``target_quat`` at reset. The satellite body has a
smaller RGB body frame and semi-transparent geoms so the attitude error can be
seen in the viewer and in videos.

.. raw:: html

   <video controls muted loop playsinline width="480">
     <source src="../_static/videos/macro_satellite.mp4" type="video/mp4">
   </video>

Example 3: cube-to-bowl robot primitives
----------------------------------------

The cube-to-bowl task uses a domain-specific preset because the natural policy
language is not the low-level MuJoCo action. The base robot action is a
seven-dimensional vector: six arm joint-position targets plus one gripper
command. A policy writer usually does not want to hand-code those seven numbers
when the observations already contain the cube, bowl and gripper poses.

The policy therefore writes readable :class:`RobotAction` objects such as
``reach_pose``, ``open_gripper`` and ``close_gripper``. The environment
constructs the matching transform through
:meth:`CubeBowlEnv.make_urscript_transform`:

.. code-block:: python

   env = base_env.append_transform(
       base_env.make_urscript_transform(
           macro_steps=28,
           settle_steps=8,
           execute=True,
           stack_rewards=True,
           stack_observations=False,
       )
   )

A Cartesian command stays semantic at the policy boundary. In this task, the
semantic target is easy to derive from observations: ``cube_pos`` and
``bowl_pos`` are ``(..., 3)`` positions in the observation spec, and
``pinch_quat`` is the current end-effector orientation. For example, a hover
pose above the cube can be written directly in observation coordinates:

.. code-block:: python

   cube = td["cube_pos"]
   hover_offset = torch.as_tensor(
       [0.0, 0.0, 0.12],
       dtype=cube.dtype,
       device=cube.device,
   )
   target_position = cube + hover_offset

   td["action"] = RobotAction.reach_pose(
       position=target_position,
       quaternion=td["pinch_quat"],
       gripper="open",
       steps=36,
       settle_steps=8,
   )

The transform maps the ``position`` and ``quaternion`` fields to a low-level
joint target using the env-provided MuJoCo damped-least-squares IK helper, fills
the gripper command, interpolates, and executes the low-level joint-position
plus gripper sequence. If you emit raw primitive TensorDicts instead of
:class:`RobotAction`, the transformed action spec documents the same fields:
``target_pose`` has trailing dimension seven ``(x, y, z, qw, qx, qy, qz)``,
``target_qpos`` has trailing dimension seven for joint-position plus gripper
commands, and ``gripper`` is a scalar command.

.. raw:: html

   <video controls muted loop playsinline width="480">
     <source src="../_static/videos/macro_cube_bowl.mp4" type="video/mp4">
   </video>

Designing target-driven macros for a new environment
----------------------------------------------------

For a new environment, design the macro target from the specs outward:

1. Inspect ``base_env.action_spec``. If your high-level decision can be written
   as a destination in this exact space, use :meth:`MacroAction.reach_action`
   and :class:`MacroPrimitiveTransform`. Build targets with ``zero()`` and
   ``project(...)`` so shape, dtype, device and bounds stay valid.

2. Inspect ``base_env.observation_spec`` and ``base_env.state_spec``. These tell
   you what semantic quantities are available to compute a target. A target may
   be a known object coordinate, an error vector to a reset-time goal, a current
   robot pose, or any controller output derived from observations.

3. Decide whether the policy target is low-level or semantic. Low-level targets
   go straight to :meth:`MacroAction.reach_action`. Semantic targets need a
   solver or adapter that maps them to the low-level action space before
   interpolation.

4. Specialize only when the semantic language is reusable. A one-off scripted
   policy can compute ``target_action`` directly. A robot family that should
   consistently speak in poses, joints and gripper commands should expose a
   domain action such as :class:`RobotAction` and a preset such as
   :class:`URScriptPrimitiveTransform`.

A custom preset usually subclasses :class:`MacroPrimitiveTransform` and supplies
three pieces of domain knowledge:

* a primitive-id library, for example ``WAIT``, ``MOVEJ``, ``MOVEL`` and any
  domain-specific commands;
* an adapter that reads the policy-facing TensorDict fields, returns the
  current low-level action, and exposes the transform-facing action specs;
* optionally, a solver that maps semantic targets such as Cartesian poses to
  low-level action targets.

The policy-facing API should still look like ordinary TorchRL code:

.. code-block:: python

   td = env.reset()
   td["action"] = MyDomainAction.reach_goal(goal_from_observation(td))
   _, td = env.step_and_maybe_reset(td)

The transform owns the expansion and execution. The policy owns the target.

Comparison
----------

.. list-table:: Three open-loop macro styles
   :header-rows: 1
   :widths: 16 21 19 24 20

   * - Example
     - Policy-facing action
     - Transform
     - What the target means
     - Where shape and dtype come from
   * - Humanoid
     - :meth:`MacroAction.reach_action <MacroAction.reach_action>` with actuator
       targets
     - :class:`MacroPrimitiveTransform`
     - A low-level MuJoCo actuator-control destination
     - ``base_env.action_spec``
   * - Satellite
     - :meth:`MacroAction.reach_action <MacroAction.reach_action>` with
       gimbal-rate targets
     - :class:`MacroPrimitiveTransform`
     - A normalized CMG gimbal-rate command computed from ``quat_err``,
       ``bus_omega`` and gimbal observations
     - ``base_env.action_spec`` for the target; ``observation_spec`` and
       ``state_spec`` for the attitude quantities used to compute it
   * - Cube bowl
     - :class:`RobotAction` commands such as ``reach_pose`` and ``close_gripper``
     - :class:`URScriptPrimitiveTransform` from
       :meth:`CubeBowlEnv.make_urscript_transform`
     - A semantic Cartesian pose, joint target or gripper command; the transform
       maps it to the seven-dimensional robot action
     - ``observation_spec`` for object and gripper poses; transformed
       ``full_action_spec`` for raw primitive fields

When to specialize
------------------

Start with :class:`MacroPrimitiveTransform` when the environment action space is
already the natural macro destination. Specialize only when the command language
adds domain meaning that should be reusable and documented, such as Cartesian
end-effector targets, gripper commands, tool frames, or environment-defined home
poses. This keeps the generic transform small while letting environment families
provide ergonomic presets.
