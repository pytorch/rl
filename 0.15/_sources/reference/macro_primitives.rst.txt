.. currentmodule:: torchrl.envs.transforms

.. _macro-primitives-guide:

Macro-control primitives
========================

Macro-control transforms let a policy emit a *destination* or a *semantic
command* at a slow time scale while TorchRL expands it into a sequence of
ordinary low-level actions before the base environment sees it. They are useful
for scripted baselines, demonstrations, curriculum tasks, and policies that
should choose *what to do next* rather than every actuator value at every
simulator step.

This page describes the design behind :class:`MacroPrimitiveTransform`,
:class:`SatelliteAttitudeTransform` and :class:`URScriptPrimitiveTransform`.
For a complete manipulation walkthrough, see the
:doc:`cube-to-bowl tutorial <../tutorials/mujoco_cube_bowl_macros>`.

The central design choice
-------------------------

The shared primitive is **not** a universal ``pose -> controls`` solver. The
shared primitive is the action-sequence plumbing:

.. code-block:: text

   td["action"] = environment-specific target command
      |
      v
   macro transform inverse action path
      |
      v
   low-level action sequence, shape (..., T, action_dim)
      |
      v
   MultiAction executes T base-env steps when execute=True

The environment-specific part is the meaning of the target command:

* for a humanoid demo, the target is a low-level actuator-control vector;
* for the satellite, the target is an attitude quaternion and the transform
  computes a CMG gimbal-rate command;
* for the cube-to-bowl robot, the target can be an end-effector pose, a joint
  target, or a gripper command.

Those targets have different units, observations, constraints and solvers. The
base :class:`MacroPrimitiveTransform` therefore owns interpolation, fixed-length
sequence construction and optional :class:`MultiAction` execution, while
specialized actions/transforms own target interpretation.

Concretely, the policy-facing command is a structured action object that
subclasses :class:`MacroAction` (a small ``mode`` + ``steps`` + ``settle_steps``
base). The single-target case adds one ``target`` field via
:class:`TargetMacroAction`; richer domains (a gripper arm) add their own fields.
A specialization is then just a :class:`MacroPrimitiveTransform` subclass that
overrides three hooks instead of plugging in adapter/solver/library objects:

* ``_resolve(td, action)`` -- map the macro action to a ``(start, target,
  steps, settle_steps)`` tuple of low-level tensors;
* ``current_action(td, ...)`` -- read the low-level action used as the
  interpolation start;
* ``transform_input_spec(...)`` -- advertise the policy-facing action spec.

The generic base implements all three for the in-action-space case, so an
environment whose action *is* the macro target needs no subclass at all.

What does "reach" mean?
-----------------------

A normal env action is consumed for one base-environment step. A macro action is
a destination plus a duration. "Reach this target in 24 steps" means:

1. read the current low-level action or current controlled state;
2. map the policy target to a low-level action destination;
3. interpolate from the start to the destination for ``steps`` base steps;
4. optionally repeat the final action for ``settle_steps`` more base steps.

So the policy is not asking the env to perform a one-step action that already
exists. It is asking the transform to create and execute the open-loop sequence
that moves toward that destination.

Choosing a macro transform
--------------------------

Given a new environment, identify the boundary between the policy-facing action
and the action consumed by ``env.step``.

1. **The policy target is already in low-level action coordinates.** Use
   :class:`MacroPrimitiveTransform` directly, or a small domain action such as
   :class:`HumanoidMacroAction` that stores a low-level target under ``td["action"]``.

2. **The policy target is semantic.** Write a domain transform that maps that
   semantic target to the low-level action space. The satellite exposes a target
   attitude through :class:`SatelliteAttitudeTransform`; the manipulation
   example uses :class:`RobotMacroAction` with
   :class:`URScriptPrimitiveTransform`.

3. **The policy already emits a sequence.** Use :class:`MultiAction` directly.
   A macro transform with ``execute=False`` is also useful for inspecting the
   sequence in tests.

The policy-facing code should remain ordinary TorchRL code:

.. code-block:: python

   td = env.reset()
   td.set(("action", "target"), target_from_observation(td))
   _, td = env.step_and_maybe_reset(td)

Random actions are still defined by the transformed ``full_action_spec``. That
spec is a :class:`~torchrl.data.Composite` of the policy-facing macro fields.
For the generic transform those fields are a primitive ``mode`` and a low-level
``target``; a domain preset keeps the same ``("action", "target")`` layout but
gives the target a semantic meaning (a target attitude quaternion for the
satellite). A random macro is valid by shape,
dtype and bounds, but it is not guaranteed to be meaningful for the task.

Example 1: humanoid actuator-control macros
-------------------------------------------

The humanoid example does not introduce a body-pose solver. Its target is a
low-level MuJoCo actuator-control destination with the same trailing dimension,
dtype, device and bounds as ``base_env.action_spec``. The helper action is
therefore named :class:`HumanoidMacroAction` rather than a generic pose action.

.. code-block:: python

   target_action = base_env.action_spec.zero()
   values = torch.as_tensor(
       [0.16, -0.14, 0.10, -0.10, 0.08, -0.08],
       dtype=target_action.dtype,
       device=target_action.device,
   )
   target_action[..., : values.numel()] = values
   target_action = base_env.action_spec.project(target_action)

   td["action"] = HumanoidMacroAction.reach_control(
       target_action,
       steps=24,
       settle_steps=8,
   )

``zero()`` provides a valid neutral vector for every actuator that the example
does not explicitly move. ``project(...)`` keeps the hand-written destination in
spec after assignment; for bounded action specs it clamps to the valid range.
``spec.rand()`` would also be valid, but it would be a random control vector,
not a demonstrative posture target.

The transform can be built directly, or through the env convenience method:

.. code-block:: python

   env = base_env.append_transform(
       base_env.make_control_transform(
           execute=True,
           stack_rewards=True,
           stack_observations=False,
       )
   )

.. raw:: html

   <video controls muted loop playsinline width="480">
     <source src="../_static/videos/macro_humanoid.mp4" type="video/mp4">
   </video>

Example 2: satellite attitude slews
-----------------------------------

The satellite target is a desired attitude frame represented by a unit
quaternion. ``SatelliteEnv`` stores reset-time target attitudes under
``td["target_quat"]`` with shape ``(..., 4)``. After appending
:class:`SatelliteAttitudeTransform`, the policy-facing action spec contains
``("action", "target")`` with the same shape, dtype and device. The readable way
to set it is the domain action object, exactly like the other two examples:

.. code-block:: python

   # ``target_quat`` is part of SatelliteEnv.state_spec and is copied to the
   # reset TensorDict. It is the desired attitude in (w, x, y, z) quaternion
   # convention.
   td.set("action", SatelliteMacroAction.slew_attitude(td["target_quat"]))

That is the whole policy-side command: "make the satellite attitude match this
target frame". The :class:`SatelliteAttitudeTransform` maps this semantic target
to the low-level action. It reads these :class:`~torchrl.envs.SatelliteEnv` observations:

* ``bus_quat``: current satellite attitude quaternion, shape ``(..., 4)``;
* ``bus_omega``: current body angular velocity, shape ``(..., 3)``;
* ``gimbal_angles``: sin/cos encoding of the CMG gimbal angles, shape
  ``(..., 2 * num_cmgs)``.

It computes the quaternion log error from ``bus_quat`` to the target attitude,
applies a small proportional-derivative steering law, maps the desired body
acceleration through the instantaneous CMG Jacobian, clamps the result to the
normalized ``[-1, 1]`` satellite action space, and lets
:class:`MacroPrimitiveTransform` interpolate the command sequence.

For quick experiments the transform also accepts a raw quaternion tensor
(``td["action"] = td["target_quat"]``) or a nested ``("action", "target")``
entry. :meth:`SatelliteMacroAction.slew_attitude
<SatelliteMacroAction.slew_attitude>` is the readable form and additionally lets
a script set ``steps`` or ``settle_steps`` per action.

The transform can be built through the env convenience method, which passes the
CMG count and action scale:

.. code-block:: python

   env = base_env.append_transform(
       base_env.make_attitude_transform(
           execute=True,
           stack_rewards=True,
           stack_observations=False,
       )
   )

The reset state still contains ``target_quat`` and ``init_bus_quat``. The target
attitude frame is also rendered by :class:`~torchrl.envs.SatelliteEnv` as a non-colliding RGB
visual frame, while the satellite body is semi-transparent so the attitude error
is visible.

This steering law is a compact demo controller, not a production guidance,
navigation and control stack. The CMG Jacobian can become poorly conditioned;
``manipulability`` remains exposed in the observations so a more robust policy
can monitor or avoid singular gimbal configurations.

.. raw:: html

   <video controls muted loop playsinline width="480">
     <source src="../_static/videos/macro_satellite.mp4" type="video/mp4">
   </video>

Example 3: cube-to-bowl robot primitives
----------------------------------------

The cube-to-bowl task uses a domain-specific preset because the natural policy
language is not the seven-dimensional MuJoCo action. The base action is six arm
joint-position targets plus one gripper command, while the task is naturally
described in terms of object and gripper poses.

A policy can therefore use observations such as ``cube_pos``, ``bowl_pos`` and
``pinch_quat`` to write readable :class:`RobotMacroAction` commands:

.. code-block:: python

   cube = td["cube_pos"]
   hover_offset = torch.as_tensor(
       [0.0, 0.0, 0.12],
       dtype=cube.dtype,
       device=cube.device,
   )

   td["action"] = RobotMacroAction.reach_pose(
       position=cube + hover_offset,
       quaternion=td["pinch_quat"],
       gripper="open",
       steps=36,
       settle_steps=8,
   )

Here the position is the coordinate of an object that needs to be moved by the
robot, expressed in the same world frame as the gripper observations. The
:meth:`~torchrl.envs.CubeBowlEnv.make_urscript_transform` preset maps that Cartesian pose to a
low-level joint target using the env-provided MuJoCo IK helper, fills the
requested gripper command, interpolates the seven-dimensional action sequence,
and executes it when ``execute=True``.

.. code-block:: python

   env = base_env.append_transform(
       base_env.make_urscript_transform(
           execute=True,
           stack_rewards=True,
           stack_observations=False,
       )
   )

.. raw:: html

   <video controls muted loop playsinline width="480">
     <source src="../_static/videos/macro_cube_bowl.mp4" type="video/mp4">
   </video>

Custom Cartesian solvers and partial pose constraints
-----------------------------------------------------

:class:`~torchrl.envs.URScriptPrimitiveTransform` maps ``reach_pose`` targets to
joint destinations through a Cartesian solver. The solver is the sanctioned
extension point for custom inverse-kinematics behavior; its contract is
documented by the :class:`~torchrl.envs.CartesianSolver` protocol. A plain
two-argument callable ``(target_pose, start_action) -> action`` remains a valid
solver: the transform inspects the signature and only forwards the optional
keyword arguments the solver declares.

Two flavors of Cartesian target need no extra arguments: a position-only target
(quaternion omitted, all three rotational degrees of freedom free) and a full
6D pose (all three rotational degrees of freedom locked). Many practical
constraints live in between: keep the gripper level, hold a rod vertically,
keep a camera pointing at a target with the roll free. These use one to two
rotational degrees of freedom and are expressed with ``orientation_mask``:

.. code-block:: python

   # Keep the tool axis aligned with the target orientation ("stay level"),
   # leave the spin about the world z axis free.
   td["action"] = RobotMacroAction.reach_pose(
       position=target_position,
       quaternion=td["pinch_quat"],
       orientation_mask=(1.0, 1.0, 0.0),
       steps=36,
   )

The mask entries are per-axis weights on the world-frame rotation error inside
the damped-least-squares loop: a zero entry removes the corresponding error row
so rotation about that world axis remains unconstrained, avoiding the
over-constrained full-quaternion solve near singularities.

Constraints applied only at the endpoint can still be violated mid-macro,
because the default expansion interpolates in joint space between the start
configuration and the endpoint solution. Passing ``path="cartesian"`` re-solves
the inverse kinematics at every interpolation waypoint along the straight-line
Cartesian path (linear in position, geodesic in orientation), so constraints
such as "stay on the line" and "stay level" hold along the whole macro action:

.. code-block:: python

   # Translate a vertically held rod along a line, no lateral motion.
   td["action"] = RobotMacroAction.reach_pose(
       position=target_position,
       quaternion=vertical_quat,
       orientation_mask=(1.0, 1.0, 0.0),
       path="cartesian",
       steps=36,
   )

The :meth:`~torchrl.envs.CubeBowlEnv.make_urscript_transform` preset honors
both keywords out of the box. A custom solver opts in by declaring the
corresponding keyword arguments; the sketch below shows the shape of a
partial-orientation damped-least-squares solver (compare
``CubeBowlEnv._cartesian_pose_to_joint_target`` for a complete MuJoCo
implementation):

.. code-block:: python

   def cartesian_solver(target_pose, start_action, *, orientation_mask=None, waypoints=None):
       # target_pose: (*batch, 7) position + (w, x, y, z) quaternion, world frame
       # start_action: (*batch, action_dim); trailing entry is the gripper
       q = start_action[..., :-1]
       weights = orientation_mask if orientation_mask is not None else ones3
       for wp_pose in interpolate_cartesian(fk(q), target_pose, waypoints or 1):
           for _ in range(iterations):
               pos_err = wp_pose[..., :3] - fk(q).position
               rot_err = weights * world_rotation_error(wp_pose[..., 3:], fk(q).rotation)
               err = concat([pos_err, rot_err])
               jac = concat([jac_pos(q), weights[..., None] * jac_rot(q)])
               q = q + damped_least_squares_step(jac, err)
       # single endpoint (*batch, action_dim) or a (*batch, waypoints, action_dim) path
       return assemble_actions(q, start_action)

If a macro action requests ``orientation_mask`` or ``path="cartesian"`` and the
configured solver does not declare the matching keyword, the transform raises a
``TypeError`` pointing at the :class:`~torchrl.envs.CartesianSolver` contract
rather than silently dropping the constraint.

Designing target-driven macros for a new environment
----------------------------------------------------

For a new environment, design from the specs outward:

1. Inspect ``base_env.action_spec``. This tells you the low-level shape, dtype,
   device and bounds that the transform must eventually emit.
2. Inspect ``base_env.observation_spec`` and ``base_env.state_spec``. These tell
   you which semantic quantities are available to compute a target, such as an
   object coordinate, a current body attitude, a joint state, or a reset-time
   goal.
3. Decide whether the policy target is low-level or semantic. Low-level targets
   can use :class:`MacroPrimitiveTransform` directly. Semantic targets need a
   domain transform and, when useful, a small action object for optional fields
   such as per-action durations.
4. Keep reusable domain logic in a specialized preset. A one-off scripted policy
   can compute a low-level target directly; a task family should expose a
   target-shaped action spec and transform that readers can reuse.

A custom preset usually supplies three pieces, matching the three transform
hooks:

* a policy-facing action spec whose fields look like desired states (e.g. a
  ``("action", "target")`` quaternion for the satellite), via
  ``transform_input_spec``;
* code that maps this desired state to a low-level action target, via
  ``_resolve`` (and ``current_action`` for the interpolation start);
* the base macro sequence expansion, inherited from
  :class:`MacroPrimitiveTransform`.

Comparison
----------

.. list-table:: Three open-loop macro styles
   :header-rows: 1
   :widths: 16 22 23 25 18

   * - Example
     - Policy-facing action
     - Transform
     - What the target means
     - Where shape and dtype come from
   * - Humanoid
     - :meth:`HumanoidMacroAction.reach_control <HumanoidMacroAction.reach_control>`
     - :meth:`HumanoidEnv.make_control_transform <torchrl.envs.HumanoidEnv.make_control_transform>`
       / :class:`MacroPrimitiveTransform`
     - A low-level MuJoCo actuator-control destination
     - ``base_env.action_spec``
   * - Satellite
     - :meth:`SatelliteMacroAction.slew_attitude <SatelliteMacroAction.slew_attitude>`
       (a ``("action", "target")`` quaternion)
     - :meth:`SatelliteEnv.make_attitude_transform <torchrl.envs.SatelliteEnv.make_attitude_transform>`
       / :class:`SatelliteAttitudeTransform`
     - A desired target attitude quaternion; the transform computes the
       normalized CMG gimbal-rate command
     - ``state_spec["target_quat"]`` and transformed
       ``full_action_spec[("action", "target")]`` for the target;
       ``observation_spec`` for current attitude and CMG state; base
       ``action_spec`` for final commands
   * - Cube bowl
     - :class:`RobotMacroAction` commands such as ``reach_pose`` and ``close_gripper``
     - :meth:`~torchrl.envs.CubeBowlEnv.make_urscript_transform` /
       :class:`URScriptPrimitiveTransform`
     - A semantic Cartesian pose, joint target or gripper command; the transform
       maps it to the seven-dimensional robot action
     - ``observation_spec`` for object and gripper poses; transformed
       ``full_action_spec`` for raw primitive fields

When to specialize
------------------

Start with :class:`MacroPrimitiveTransform` when the environment action space is
already the natural macro destination. Specialize when the command language adds
meaning that should be reusable and documented, such as target attitudes,
Cartesian end-effector targets, gripper commands, tool frames, or
environment-defined home poses. This keeps the generic transform small while
letting each environment expose a readable target-driven API.
