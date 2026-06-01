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
   MacroAction or primitive TensorDict
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

Example 1: humanoid open-loop poses
-----------------------------------

The humanoid example has no robot-specific adapter. Each macro action is a
low-level actuator destination expressed directly in the humanoid action space.
The policy emits one :class:`MacroAction` per pose:

.. code-block:: python

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

This is the minimal pattern: interpolate in action space and execute the
sequence through normal TorchRL rollout logic.

.. raw:: html

   <video controls muted loop playsinline width="480">
     <source src="../_static/videos/macro_humanoid.mp4" type="video/mp4">
   </video>

Example 2: satellite attitude slews
-----------------------------------

The satellite example also uses :class:`MacroPrimitiveTransform`, but the target
action is computed by a small feedback controller. The env exposes
``quat_err``, ``bus_omega`` and CMG gimbal observations. The scripted policy
computes a desired bus acceleration, maps it through the instantaneous CMG
Jacobian, projects the gimbal-rate command into the action spec, and emits a
macro destination:

.. code-block:: python

   td["action"] = MacroAction.reach_action(
       gimbal_rate_target,
       steps=36,
       settle_steps=8,
   )

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

The cube-to-bowl task uses a domain-specific preset. The policy writes readable
:class:`RobotAction` objects such as ``reach_pose``, ``open_gripper`` and
``close_gripper``. The environment constructs the matching transform through
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

A Cartesian command stays semantic at the policy boundary:

.. code-block:: python

   td["action"] = RobotAction.reach_pose(
       position=target_position,
       quaternion=target_quaternion,
       gripper="closed",
       gripper_command=close_command,
       steps=36,
       settle_steps=8,
   )

The transform maps Cartesian targets to joint targets using the env-provided
MuJoCo damped-least-squares IK helper, fills the gripper command, interpolates,
and executes the low-level joint-position plus gripper sequence.

.. raw:: html

   <video controls muted loop playsinline width="480">
     <source src="../_static/videos/macro_cube_bowl.mp4" type="video/mp4">
   </video>

Comparison
----------

.. list-table:: Three open-loop macro styles
   :header-rows: 1
   :widths: 18 22 22 20 18

   * - Example
     - Policy-facing action
     - Transform
     - Low-level target source
     - Execution API
   * - Humanoid
     - :class:`MacroAction.reach_action <MacroAction>` with actuator targets
     - :class:`MacroPrimitiveTransform`
     - The target is already in action space
     - ``rollout(policy=...)``
   * - Satellite
     - :class:`MacroAction.reach_action <MacroAction>` with gimbal-rate targets
     - :class:`MacroPrimitiveTransform`
     - A scripted feedback controller computes the target from env observations
     - ``step_and_maybe_reset`` until attitude error is small
   * - Cube bowl
     - :class:`RobotAction` commands such as ``reach_pose`` and ``close_gripper``
     - :class:`URScriptPrimitiveTransform` from
       :meth:`CubeBowlEnv.make_urscript_transform`
     - Env-specific IK and gripper helpers map semantic targets to low-level
       commands
     - One ``step_and_maybe_reset`` per high-level primitive

When to specialize
------------------

Start with :class:`MacroPrimitiveTransform` when the environment action space is
already the natural macro destination. Specialize only when the command language
adds domain meaning that should be reusable and documented, such as Cartesian
end-effector targets, gripper commands, tool frames, or environment-defined home
poses. This keeps the generic transform small while letting environment families
provide ergonomic presets.
