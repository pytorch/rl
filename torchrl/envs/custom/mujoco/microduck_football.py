# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Football for MicroDuck bipeds: two teams of ducks, a ball and a pitch.

The scene is built procedurally from the walking MJCF that
:class:`~torchrl.envs.MicroDuckEnv` resolves: :func:`build_football_scene`
attaches one copy of the robot per player, with the name prefix
``blue<i>/`` or ``red<i>/``, onto a pitch with walls, two goals, a ball and
two cameras, and returns the MJCF as text. :class:`MicroDuckFootballEnv` is
the multi-agent env over that scene, and :func:`~torchrl.envs.microduck_skill_env` runs a
trained joint-level MicroDuck controller under a coarser policy that picks
one of its locomotion tasks per duck.

Teams
    ``blue`` (team 0) attacks along ``+x`` and kicks off on the ``x < 0``
    half; ``red`` (team 1) attacks along ``-x``. Every per-agent quantity is
    expressed in the agent's own frame (planar vectors rotated by the
    agent's yaw) and in its team's frame (the attacking direction is ``+x``
    for both teams), so one set of parameters plays both sides.

Reward
    Per agent, the sum of a team term (the goal, shared by the whole team
    with opposite signs, and the ball's velocity along the team's attacking
    direction) and individual terms (own velocity toward the ball, a fall
    penalty, an action-rate cost). Weights are per second and multiplied by
    the control period except for the goal and the fall penalty.

Termination
    A goal ends the match; the match clock truncates it. A fallen duck does
    not end the match: it is put back upright on its kickoff slot (with the
    fall penalty) so the game goes on.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar, Literal, TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Binary, Bounded, Composite, Unbounded
from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.envs.custom.mujoco.base import _MujocoMeta, MujocoEnv
from torchrl.envs.custom.mujoco.microduck import (
    _body_frame_linear_velocity,
    _low_cost_collision_scene,
    _projected_gravity,
    MicroDuckEnv,
)

if TYPE_CHECKING:
    import mujoco

_has_mujoco = importlib.util.find_spec("mujoco") is not None

TEAM_NAMES: tuple[str, str] = ("blue", "red")
"""Name prefix of the two teams; team 0 attacks along ``+x``."""
FOOTBALL_NUMERIC: str = "football"
"""Name of the MJCF ``<numeric>`` element that records the pitch parameters."""
_WALL_THICKNESS = 0.02
_ROBOT_COMPILER_FLAGS = (
    "fitaabb",
    "inertiafromgeom",
    "balanceinertia",
    "boundmass",
    "boundinertia",
    "settotalmass",
)
_POST_RADIUS = 0.01


def kickoff_positions(
    players_per_team: int, pitch_length: float, pitch_width: float
) -> list[tuple[float, float]]:
    """Return the kickoff slot of every player of a team, in the team frame.

    The team frame has the attacking direction along ``+x`` and the team's
    own goal at ``x = -pitch_length / 2``. The first slot is the goalkeeper's,
    in front of the goal; the others form one or two rows toward the halfway
    line. Team 1 uses the same slots rotated by 180 degrees about the pitch
    center.

    Examples:
        >>> [tuple(round(v, 2) for v in slot) for slot in kickoff_positions(1, 3.0, 2.0)]
        [(-0.45, 0.0)]
        >>> [tuple(round(v, 2) for v in slot) for slot in kickoff_positions(5, 3.0, 2.0)]
        [(-1.26, 0.0), (-0.9, -0.6), (-0.9, 0.6), (-0.36, -0.4), (-0.36, 0.4)]
    """
    if players_per_team < 1:
        raise ValueError("players_per_team must be at least 1.")
    if players_per_team == 1:
        return [(-0.15 * pitch_length, 0.0)]

    def row(count: int, x: float, half_spread: float) -> list[tuple[float, float]]:
        if count == 1:
            return [(x, 0.0)]
        return [
            (x, -half_spread + 2.0 * half_spread * k / (count - 1))
            for k in range(count)
        ]

    slots = [(-0.42 * pitch_length, 0.0)]
    field = players_per_team - 1
    if field <= 2:
        slots += row(field, -0.15 * pitch_length, 0.2 * pitch_width)
    else:
        back = field // 2
        slots += row(back, -0.3 * pitch_length, 0.3 * pitch_width)
        slots += row(field - back, -0.12 * pitch_length, 0.2 * pitch_width)
    return slots


def _yaw_quaternion(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _quaternion_product(a: Sequence[float], b: Sequence[float]) -> list[float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _camera_axes(position: Sequence[float], target: Sequence[float]) -> list[float]:
    """Return the ``xyaxes`` of a camera at ``position`` looking at ``target``."""
    forward = [t - p for t, p in zip(target, position)]
    norm = math.sqrt(sum(v * v for v in forward))
    forward = [v / norm for v in forward]
    up = (0.0, 0.0, 1.0)
    right = [
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    ]
    norm = math.sqrt(sum(v * v for v in right))
    right = [v / norm for v in right]
    camera_up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    return right + camera_up


def _share_meshes(spec: mujoco.MjSpec) -> None:
    """Keep one copy of each mesh that was attached under several prefixes.

    :meth:`mujoco.MjSpec.attach` copies the child's meshes under the player's
    prefix, so a team of five carries five copies of every robot mesh. Point
    the other players' geoms at the first copy and delete the rest; the
    compiled model is unchanged apart from its mesh tables.
    """
    first: dict[str, str] = {}
    replacement: dict[str, str] = {}
    for mesh in spec.meshes:
        if "/" not in mesh.name:
            continue
        stem = mesh.name.split("/", 1)[1]
        kept = first.setdefault(stem, mesh.name)
        if kept != mesh.name:
            replacement[mesh.name] = kept
    for geom in spec.geoms:
        if geom.meshname in replacement:
            geom.meshname = replacement[geom.meshname]
    for mesh in list(spec.meshes):
        if mesh.name in replacement:
            spec.delete(mesh)


def build_football_scene(
    scene: str | Path,
    *,
    players_per_team: int = 5,
    pitch_length: float = 3.0,
    pitch_width: float = 2.0,
    goal_width: float = 0.6,
    goal_height: float = 0.25,
    goal_depth: float = 0.25,
    wall_height: float = 0.15,
    ball_radius: float = 0.035,
    ball_mass: float = 0.015,
    team_colors: Sequence[Sequence[float]] = (
        (0.2, 0.45, 0.95, 1.0),
        (0.95, 0.25, 0.2, 1.0),
    ),
    low_cost_collisions: bool = True,
    meshdir: str | Path | None = None,
) -> str:
    """Build the MJCF of a MicroDuck football match and return it as text.

    The robot is read from ``scene``, the ``scene_walk.xml`` that
    :meth:`~torchrl.envs.MicroDuckEnv.resolve_scene` locates (any MJCF with a
    single free root body, 14 actuators and a ``STAND`` keyframe works), and
    attached ``2 * players_per_team`` times with :class:`mujoco.MjSpec`:
    team ``blue`` on the ``x < 0`` half facing ``+x``, team ``red`` mirrored
    through the pitch center, each player on its :func:`kickoff_positions`
    slot and named ``blue<i>/`` or ``red<i>/`` (so ``blue0/left_foot`` is a
    site and ``red3/left_hip_yaw`` an actuator). Only the robot comes along:
    the scene's floor, lights, ground textures and keyframes are dropped
    before attaching, and the materials whose name contains ``shell`` take
    the team color. The players share one copy of the robot's meshes.

    Around the ducks: a grass plane with pitch markings, walls of
    ``wall_height`` that keep the ball (and the feet) in, two goals with
    posts, a crossbar and nets (the goal at ``+x`` is the one ``blue``
    attacks; sites ``red_goal`` and ``blue_goal`` mark the goal each team
    defends), a ``ball`` body with a free joint and a hollow-sphere
    inertia, two cameras (``broadcast`` from the touchline, id 0, and
    ``topdown``, id 1; the robots' own head cameras follow), a ``STAND``
    keyframe with every duck in its standing pose and a ``<numeric>``
    element named ``football`` holding ``pitch_length, pitch_width,
    goal_width, goal_height, goal_depth, ball_radius, players_per_team`` so
    that :class:`MicroDuckFootballEnv` reads the geometry back from the
    compiled model. The ball's joint comes last, so the state is the
    concatenation of the ducks' ``qpos``/``qvel`` blocks followed by the
    ball's.

    Args:
        scene (str or Path): path to the MicroDuck walking scene.

    Keyword Args:
        players_per_team (int, optional): ducks per team. Defaults to ``5``.
        pitch_length (float, optional): goal line to goal line, in meters.
            Defaults to ``3.0``.
        pitch_width (float, optional): touchline to touchline. Defaults to
            ``2.0``.
        goal_width (float, optional): distance between the posts. Defaults
            to ``0.6``.
        goal_height (float, optional): height of the crossbar. Defaults to
            ``0.25``.
        goal_depth (float, optional): depth of the net behind the goal line.
            Defaults to ``0.25``.
        wall_height (float, optional): height of the walls around the pitch.
            Defaults to ``0.15``.
        ball_radius (float, optional): ball radius; the upstream kick scene
            uses a 70 mm ball. Defaults to ``0.035``.
        ball_mass (float, optional): ball mass in kg. Defaults to ``0.015``.
        team_colors (sequence of two RGBA sequences, optional): shell colors
            of the two teams and tint of the net each team defends.
        low_cost_collisions (bool, optional): replace the robot's collision
            meshes with box proxies, as :class:`~torchrl.envs.MicroDuckEnv`
            does. Defaults to ``True``.
        meshdir (str or Path, optional): value of the ``meshdir`` compiler
            attribute in the returned MJCF. Defaults to the absolute path of
            the robot's mesh directory, so the text loads from anywhere on
            the same machine; pass a relative directory to export a scene to
            ship next to the meshes.

    Returns:
        The MJCF as a string.

    Examples:
        >>> import mujoco
        >>> from torchrl.envs import MicroDuckEnv
        >>> from torchrl.envs.custom.mujoco.microduck_football import build_football_scene
        >>> xml = build_football_scene(MicroDuckEnv.resolve_scene(download=True), players_per_team=2)  # doctest: +SKIP
        >>> model = mujoco.MjModel.from_xml_string(xml)  # doctest: +SKIP
        >>> model.nq, model.nu, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)  # doctest: +SKIP
        (91, 56, 'broadcast')
    """
    if not _has_mujoco:
        raise ImportError("build_football_scene requires the `mujoco` package.")
    import mujoco

    if players_per_team < 1:
        raise ValueError("players_per_team must be at least 1.")
    for name, value in (
        ("pitch_length", pitch_length),
        ("pitch_width", pitch_width),
        ("goal_width", goal_width),
        ("goal_height", goal_height),
        ("goal_depth", goal_depth),
        ("wall_height", wall_height),
        ("ball_radius", ball_radius),
        ("ball_mass", ball_mass),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}.")
    if goal_width >= pitch_width:
        raise ValueError("goal_width must be smaller than pitch_width.")
    if len(team_colors) != 2 or any(len(color) != 4 for color in team_colors):
        raise ValueError("team_colors must hold two RGBA colors.")
    scene = Path(scene).expanduser()
    if not scene.is_file():
        raise FileNotFoundError(scene)

    physics_scene = (
        _low_cost_collision_scene(scene.resolve())
        if low_cost_collisions
        else nullcontext(scene.resolve())
    )
    with physics_scene as robot_scene:
        robot = mujoco.MjSpec.from_file(str(robot_scene))
        roots = list(robot.worldbody.bodies)
        if len(roots) != 1:
            raise ValueError(
                "The robot scene must have exactly one body under the world body, "
                f"got {[body.name for body in roots]}."
            )
        stand = next((key for key in robot.keys if key.name == "STAND"), None)
        if stand is None:
            raise ValueError("The robot scene must define a `STAND` keyframe.")
        stand_qpos = [float(value) for value in stand.qpos]
        stand_ctrl = [float(value) for value in stand.ctrl]
        if len(stand_qpos) != 7 + MicroDuckEnv.NUM_JOINTS:
            raise ValueError(
                "Expected a robot with a free root joint and "
                f"{MicroDuckEnv.NUM_JOINTS} hinge joints, got a STAND keyframe of "
                f"{len(stand_qpos)} positions."
            )
        if meshdir is None:
            robot_meshdir = Path(robot.meshdir) if robot.meshdir else Path()
            if not robot_meshdir.is_absolute():
                robot_meshdir = Path(robot.modelfiledir) / robot_meshdir
            meshdir = robot_meshdir.resolve()

        spec = mujoco.MjSpec()
        spec.modelname = "microduck_football"
        spec.compiler.degree = False
        spec.compiler.autolimits = True
        # The compiler of the attaching spec applies to the robot's geoms too;
        # the collision proxies rely on the walking scene's fitting flags.
        for flag in _ROBOT_COMPILER_FLAGS:
            setattr(spec.compiler, flag, getattr(robot.compiler, flag))
        spec.option.timestep = robot.option.timestep
        spec.meshdir = str(meshdir)
        spec.stat.extent = max(pitch_length, pitch_width)
        spec.stat.center = [0.0, 0.0, 0.1]
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        spec.visual.global_.azimuth = 90
        spec.visual.global_.elevation = -35

        half_length, half_width = pitch_length / 2.0, pitch_width / 2.0
        spec.add_texture(
            name="sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.55, 0.7, 0.9],
            rgb2=[0.15, 0.2, 0.35],
            width=512,
            height=3072,
        )
        spec.add_texture(
            name="grass",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.22, 0.5, 0.2],
            rgb2=[0.27, 0.56, 0.24],
            width=300,
            height=300,
        )
        grass = spec.add_material(
            name="grass",
            texrepeat=[pitch_length * 2.0, pitch_width * 2.0],
            texuniform=True,
            reflectance=0.05,
        )
        grass.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grass"
        world = spec.worldbody
        world.add_light(
            pos=[0.0, 0.0, 3.0],
            dir=[0.0, 0.0, -1.0],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.7, 0.7, 0.7],
        )
        world.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[half_length + goal_depth + 1.0, half_width + 1.0, 0.05],
            material="grass",
        )

        def marking(name: str, pos: Sequence[float], size: Sequence[float]) -> None:
            world.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(pos),
                size=list(size),
                rgba=[1.0, 1.0, 1.0, 1.0],
                contype=0,
                conaffinity=0,
            )

        line = 0.006
        marking("halfway_line", (0.0, 0.0, 0.0005), (line, half_width, 0.0005))
        for sign, side in ((1.0, "left"), (-1.0, "right")):
            marking(
                f"{side}_touchline",
                (0.0, sign * (half_width - line), 0.0005),
                (half_length, line, 0.0005),
            )
        for sign, team in ((1.0, TEAM_NAMES[1]), (-1.0, TEAM_NAMES[0])):
            marking(
                f"{team}_goal_line",
                (sign * (half_length - line), 0.0, 0.0005),
                (line, half_width, 0.0005),
            )
        world.add_geom(
            name="center_spot",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.0, 0.0, 0.0005],
            size=[0.02, 0.0005, 0.0],
            rgba=[1.0, 1.0, 1.0, 1.0],
            contype=0,
            conaffinity=0,
        )

        def box(
            name: str,
            pos: Sequence[float],
            size: Sequence[float],
            rgba: Sequence[float],
        ) -> None:
            world.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(pos),
                size=list(size),
                rgba=list(rgba),
            )

        wall_rgba = (0.85, 0.85, 0.85, 0.5)

        def wall(name: str, pos: Sequence[float], size: Sequence[float]) -> None:
            box(name, pos, size, wall_rgba)

        thickness = _WALL_THICKNESS
        outer_x = half_length + goal_depth + 2.0 * thickness
        for sign, side in ((1.0, "left"), (-1.0, "right")):
            wall(
                f"{side}_wall",
                (0.0, sign * (half_width + thickness), wall_height / 2.0),
                (outer_x + thickness, thickness, wall_height / 2.0),
            )
        mouth = goal_width / 2.0 + _POST_RADIUS
        for sign, team in ((1.0, TEAM_NAMES[1]), (-1.0, TEAM_NAMES[0])):
            # The end wall leaves the goal mouth open; the goal box behind the
            # line is closed by a back net, two side nets and a roof.
            y_center = (mouth + half_width + 2.0 * thickness) / 2.0
            y_half = (half_width + 2.0 * thickness - mouth) / 2.0
            for lateral, side in ((1.0, "left"), (-1.0, "right")):
                wall(
                    f"{team}_{side}_end_wall",
                    (
                        sign * (half_length + thickness),
                        lateral * y_center,
                        wall_height / 2.0,
                    ),
                    (thickness, y_half, wall_height / 2.0),
                )
            net_rgba = list(team_colors[TEAM_NAMES.index(team)][:3]) + [0.3]
            box(
                f"{team}_back_net",
                (sign * (half_length + goal_depth + thickness), 0.0, goal_height / 2.0),
                (thickness, mouth + thickness, goal_height / 2.0),
                net_rgba,
            )
            for lateral, side in ((1.0, "left"), (-1.0, "right")):
                box(
                    f"{team}_{side}_net",
                    (
                        sign * (half_length + goal_depth / 2.0),
                        lateral * (mouth + thickness),
                        goal_height / 2.0,
                    ),
                    (goal_depth / 2.0, thickness, goal_height / 2.0),
                    net_rgba,
                )
            box(
                f"{team}_roof_net",
                (sign * (half_length + goal_depth / 2.0), 0.0, goal_height + thickness),
                (goal_depth / 2.0, mouth + 2.0 * thickness, thickness),
                net_rgba,
            )
            for lateral, side in ((1.0, "left"), (-1.0, "right")):
                world.add_geom(
                    name=f"{team}_{side}_post",
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=[_POST_RADIUS, 0.0, 0.0],
                    fromto=[
                        sign * half_length,
                        lateral * mouth,
                        0.0,
                        sign * half_length,
                        lateral * mouth,
                        goal_height,
                    ],
                    rgba=[1.0, 1.0, 1.0, 1.0],
                )
            world.add_geom(
                name=f"{team}_crossbar",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[_POST_RADIUS, 0.0, 0.0],
                fromto=[
                    sign * half_length,
                    -mouth,
                    goal_height,
                    sign * half_length,
                    mouth,
                    goal_height,
                ],
                rgba=[1.0, 1.0, 1.0, 1.0],
            )
            world.add_site(
                name=f"{team}_goal",
                pos=[sign * half_length, 0.0, goal_height / 2.0],
                size=[0.01, 0.01, 0.01],
                group=3,
            )
        world.add_site(
            name="pitch_center", pos=[0.0, 0.0, 0.0], size=[0.01, 0.01, 0.01], group=3
        )

        world.add_camera(
            name="broadcast",
            pos=[0.0, -(half_width + 0.5 * pitch_length), 0.45 * pitch_length],
            xyaxes=_camera_axes(
                (0.0, -(half_width + 0.5 * pitch_length), 0.45 * pitch_length),
                (0.0, 0.0, 0.05),
            ),
            fovy=45,
        )
        world.add_camera(
            name="topdown",
            pos=[0.0, 0.0, 0.9 * max(pitch_length, pitch_width)],
            xyaxes=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            fovy=45,
        )

        slots = kickoff_positions(players_per_team, pitch_length, pitch_width)
        key_qpos: list[float] = []
        key_ctrl: list[float] = []
        for team, name in enumerate(TEAM_NAMES):
            sign = 1.0 if team == 0 else -1.0
            yaw = 0.0 if team == 0 else math.pi
            color = [float(value) for value in team_colors[team]]
            for index, (slot_x, slot_y) in enumerate(slots):
                position = [sign * slot_x, sign * slot_y, 0.0]
                quaternion = _yaw_quaternion(yaw)
                frame = world.add_frame(pos=position, quat=quaternion)
                child = mujoco.MjSpec.from_file(str(robot_scene))
                for geom in list(child.worldbody.geoms):
                    child.delete(geom)
                for light in list(child.worldbody.lights):
                    child.delete(light)
                for key in list(child.keys):
                    child.delete(key)
                for material in list(child.materials):
                    if any(material.textures):
                        child.delete(material)
                    elif "shell" in material.name:
                        material.rgba = color
                for texture in list(child.textures):
                    child.delete(texture)
                spec.attach(child, prefix=f"{name}{index}/", frame=frame)
                # Root pose of the STAND keyframe, expressed in the world.
                root_x = position[0] + sign * stand_qpos[0]
                root_y = position[1] + sign * stand_qpos[1]
                key_qpos += [root_x, root_y, stand_qpos[2]]
                key_qpos += _quaternion_product(quaternion, stand_qpos[3:7])
                key_qpos += stand_qpos[7:]
                key_ctrl += stand_ctrl
        _share_meshes(spec)

        ball = world.add_body(name="ball", pos=[0.0, 0.0, ball_radius])
        ball.add_joint(name="ball_free", type=mujoco.mjtJoint.mjJNT_FREE)
        # Hollow sphere: I = 2/3 m r^2, as in the upstream kick scene.
        inertia = 2.0 / 3.0 * ball_mass * ball_radius**2
        ball.mass = ball_mass
        ball.inertia = [inertia, inertia, inertia]
        ball.explicitinertial = True
        ball.add_geom(
            name="ball_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[ball_radius, 0.0, 0.0],
            rgba=[1.0, 0.55, 0.0, 1.0],
            mass=0.0,
            condim=6,
            friction=[0.5, 0.005, 0.0001],
            contype=3,
            conaffinity=3,
        )
        key_qpos += [0.0, 0.0, ball_radius, 1.0, 0.0, 0.0, 0.0]
        spec.add_key(name="STAND", qpos=key_qpos, ctrl=key_ctrl)
        spec.add_numeric(
            name=FOOTBALL_NUMERIC,
            data=[
                pitch_length,
                pitch_width,
                goal_width,
                goal_height,
                goal_depth,
                ball_radius,
                float(players_per_team),
            ],
        )
        spec.compile()
        return spec.to_xml()


class _FootballMeta(_MujocoMeta):
    """Build (once) and cache the football scene before the env is batched."""

    def __call__(
        cls,
        scene: str | Path | None = None,
        *args: Any,
        microduck_root: str | Path | None = None,
        root: str | Path | None = None,
        download: bool | str = False,
        players_per_team: int | None = None,
        pitch: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        if scene is None:
            robot_scene = MicroDuckEnv.resolve_scene(
                microduck_root, root=root, download=download
            )
            scene = cls.write_scene(
                robot_scene,
                root=root,
                players_per_team=5 if players_per_team is None else players_per_team,
                **(pitch or {}),
            )
        elif players_per_team is not None or pitch is not None:
            raise ValueError(
                "players_per_team and pitch only apply when the scene is built here; "
                "a given scene fixes them."
            )
        return super().__call__(scene, *args, **kwargs)


class MicroDuckFootballEnv(MujocoEnv, metaclass=_FootballMeta):
    r"""Football between two teams of MicroDuck bipeds, as a multi-agent env.

    The scene comes from :func:`build_football_scene`: ``2 * players_per_team``
    copies of the robot that :class:`~torchrl.envs.MicroDuckEnv` resolves
    (``microduck_root``, ``MICRODUCK_RL_ROOT``, the installed package or a
    download), a ball, walls, two goals and cameras. Team ``blue`` (agents
    ``0`` to ``players_per_team - 1``) attacks along ``+x`` from the ``x < 0``
    half, team ``red`` (the remaining agents) attacks along ``-x``. The built
    MJCF is cached under ``root/football`` and its path is :attr:`scene_path`.

    Every agent acts and observes along an ``agents`` dimension:

    * ``("agents", "action")``: normalized offsets around the ``STAND``
      actuator targets, scaled by ``action_scale`` radians, shape
      ``(num_envs, num_agents, 14)``, exactly the action of
      :class:`~torchrl.envs.MicroDuckEnv`.
    * ``("agents", "observation")``: the 56 proprioceptive values of
      :class:`~torchrl.envs.MicroDuckEnv` in the same order (projected
      gravity, base angular velocity, body-frame linear velocity, a zero
      command, joint errors, joint velocities, a gait clock at
      :attr:`~torchrl.envs.MicroDuckEnv.GAIT_FREQUENCY_HZ`, the previous
      action) followed by ``8 * players_per_team + 10`` match features: the
      position on the pitch in the team frame (normalized to ``[-1, 1]``),
      the heading relative to the attacking direction (cosine, sine), the
      ball's relative position (planar, in the agent's frame) and height,
      the ball's relative velocity, the relative positions of the goal to
      attack and of the own goal, the relative position and velocity of
      every teammate then of every opponent (fixed order), and the fraction
      of the match left.
    * ``("agents", "reward")``: per-agent reward, ``(num_envs, num_agents, 1)``.
    * ``("agents", "fallen")``: whether the agent is down, because it fell
      during the step or has not stood up again yet.
    * ``ball_position`` (``(num_envs, 3)``) and ``goal`` (``(num_envs, 1)``,
      ``1`` when blue scored on this step, ``-1`` when red did, else ``0``).

    Reward terms, weighted by :attr:`REWARD_WEIGHTS` (overridden per key by
    ``reward_weights``); the per-second terms are multiplied by the control
    period:

    * ``goal`` (one-off): ``+w`` for the scoring team, ``-w`` for the other.
    * ``ball_progress``: the ball's velocity along the team's attacking
      direction, so the two teams' shaping cancels out.
    * ``approach_ball``: the agent's planar velocity toward the ball, capped
      at :attr:`APPROACH_SPEED_CAP` and off within a ball radius plus 5 cm.
    * ``fall`` (one-off, negative weight): paid on the step the agent goes down.
    * ``crowd`` (negative weight): number of other ducks within
      :attr:`CROWD_RADIUS` of the agent, so the team spreads out instead of
      piling onto the ball.
    * ``action_rate`` (negative weight): squared change of the action.

    A fall (base height below :attr:`~torchrl.envs.MicroDuckEnv.MIN_HEIGHT_RATIO`
    of the standing height, or tilt beyond
    :attr:`~torchrl.envs.MicroDuckEnv.MIN_UPRIGHT`) costs the duck
    ``respawn_delay_s`` seconds on the ground, its actions ignored, before it
    stands up again, still: where it fell, facing the goal it attacks
    (``respawn_mode="in_place"``), or on its kickoff slot (``"kickoff"``).
    Without ``respawn`` it stays down for the rest of the match. A goal
    terminates the match; ``max_episode_steps`` (1500 steps, 30 s at 50 Hz)
    truncates it. A non-finite state terminates it as well.

    The env runs on every :class:`~torchrl.envs.MujocoEnv` backend, but the
    default is ``"mujoco"``: the native bindings step a 5-a-side scene at
    several hundred control steps per second per worker process on a CPU,
    while the vectorized backends (``"mujoco-torch"``, built from its
    ``main`` branch, and ``"mjx"``) pay off on a GPU.

    Args:
        scene (str or Path, optional): path to a football MJCF written by
            :meth:`write_scene` or exported from :func:`build_football_scene`.
            When omitted, the scene is built from the MicroDuck robot located
            through ``microduck_root``, ``root`` and ``download`` (see
            :meth:`~torchrl.envs.MicroDuckEnv.resolve_scene`).

    Keyword Args:
        microduck_root (str or Path, optional): MicroDuck checkout, package
            directory or ``scene_walk.xml`` path. Defaults to the
            ``MICRODUCK_RL_ROOT`` environment variable, the installed package,
            or a download under ``root``.
        root (str or Path, optional): cache directory for downloads and built
            scenes. Defaults to ``~/.cache/torchrl/microduck``.
        download (bool or ``"force"``, optional): download the pinned
            MicroDuck commit when nothing else resolves. Defaults to
            ``False``.
        players_per_team (int, optional): ducks per team. Defaults to ``5``.
        pitch (Mapping[str, Any], optional): keyword arguments of
            :func:`build_football_scene` other than ``players_per_team``, for
            instance ``{"pitch_length": 2.0, "goal_width": 0.5}``.
        action_scale (float, optional): radians of actuator target per unit
            action. Defaults to ``1.0``, the scale of the from-scratch
            MicroDuck policies.
        reward_weights (Mapping[str, float], optional): weights replacing
            entries of :attr:`REWARD_WEIGHTS`.
        respawn (bool, optional): stand fallen ducks up again. Defaults to
            ``True``.
        respawn_mode (str, optional): ``"in_place"`` (default) stands a duck
            up where it fell, facing the goal it attacks; ``"kickoff"`` puts it
            back on its kickoff slot.
        respawn_delay_s (float, optional): seconds a fallen duck stays down,
            actions ignored, before standing up. Defaults to ``1.0``.
        spawn_noise (float, optional): uniform noise on the kickoff positions
            at reset, in meters. Defaults to ``0.1``.
        yaw_noise (float, optional): uniform noise on the kickoff heading, in
            radians. Defaults to ``0.3``.
        joint_reset_noise_scale (float, optional): uniform noise on the joint
            positions at reset, in radians. Defaults to ``0.02``.
        ball_noise (float, optional): uniform noise on the ball's kickoff
            position, in meters. Defaults to ``0.1``.
        backend (str, optional): ``"mujoco"`` (default), ``"mujoco-torch"``
            or ``"mjx"``.
        max_episode_steps (int, optional): match length in control steps.
            Defaults to ``1500``.
        \*\*kwargs: forwarded to :class:`~torchrl.envs.MujocoEnv`:
            ``num_envs``, ``parallel``, ``device``, ``seed``, ``from_pixels``,
            ``render_width``, ``render_height``, ``camera_id`` and so on.
            ``pixels_only`` is not supported.

    Examples:
        Two against two with random joint actions, 16 native simulators in
        worker processes:

        >>> import torch
        >>> from torchrl.envs import MicroDuckFootballEnv
        >>> env = MicroDuckFootballEnv(download=True, players_per_team=2, num_envs=16)  # doctest: +SKIP
        >>> rollout = env.rollout(20)  # doctest: +SKIP
        >>> rollout["agents", "observation"].shape  # doctest: +SKIP
        torch.Size([16, 20, 4, 82])
        >>> rollout["next", "agents", "reward"].shape, rollout["next", "goal"].shape  # doctest: +SKIP
        (torch.Size([16, 20, 4, 1]), torch.Size([16, 20, 1]))

        Film the broadcast camera:

        >>> from torchrl.record import CSVLogger, VideoRecorder
        >>> from torchrl.envs import TransformedEnv
        >>> env = TransformedEnv(  # doctest: +SKIP
        ...     MicroDuckFootballEnv(download=True, from_pixels=True, render_width=640, render_height=360),
        ...     VideoRecorder(CSVLogger("football", video_format="mp4"), tag="match"),
        ... )
        >>> env.rollout(500)  # doctest: +SKIP
        >>> env.transform.dump()  # doctest: +SKIP

        Export the MJCF for inspection, or load a scene built elsewhere:

        >>> path = MicroDuckFootballEnv.write_scene(  # doctest: +SKIP
        ...     MicroDuckEnv.resolve_scene(download=True), players_per_team=3
        ... )
        >>> env = MicroDuckFootballEnv(path, num_envs=4)  # doctest: +SKIP

    Reference:
        Pollen Robotics, MicroDuck (https://github.com/pollen-robotics/microduck).
        Liu et al., "From Motor Control to Team Play in Simulated Humanoid
        Football", Science Robotics 2022 (https://arxiv.org/abs/2105.12196),
        for the skill-based training recipe the example follows.
    """

    FRAME_SKIP = 10
    DEFAULT_BACKEND: ClassVar[BackendName] = "mujoco"
    NUM_JOINTS: ClassVar[int] = MicroDuckEnv.NUM_JOINTS
    PROPRIOCEPTION_DIM: ClassVar[int] = MicroDuckEnv.OBSERVATION_DIM
    """Size of the leading :class:`~torchrl.envs.MicroDuckEnv` observation block."""
    DUCK_NQ: ClassVar[int] = 7 + MicroDuckEnv.NUM_JOINTS
    DUCK_NV: ClassVar[int] = 6 + MicroDuckEnv.NUM_JOINTS
    BALL_NQ: ClassVar[int] = 7
    BALL_NV: ClassVar[int] = 6
    REWARD_WEIGHTS: ClassVar[dict[str, float]] = {
        "goal": 10.0,
        "ball_progress": 2.0,
        "approach_ball": 0.5,
        "fall": -1.0,
        "action_rate": -0.05,
        "crowd": -0.5,
    }
    """Default weight of every reward term; ``goal`` and ``fall`` are one-off."""
    APPROACH_SPEED_CAP: ClassVar[float] = 0.5
    """Cap on the speed toward the ball that ``approach_ball`` pays for, in m/s."""
    APPROACH_RADIUS_MARGIN: ClassVar[float] = 0.05
    CROWD_RADIUS: ClassVar[float] = 0.2
    """Planar distance under which another duck counts as crowding, in meters."""
    CAMERAS: ClassVar[tuple[str, str]] = ("broadcast", "topdown")
    """Names of the scene cameras, in id order."""

    def __init__(
        self,
        scene: str | Path,
        *,
        action_scale: float = 1.0,
        reward_weights: Mapping[str, float] | None = None,
        respawn: bool = True,
        respawn_mode: Literal["in_place", "kickoff"] = "in_place",
        respawn_delay_s: float = 1.0,
        spawn_noise: float = 0.1,
        yaw_noise: float = 0.3,
        joint_reset_noise_scale: float = 0.02,
        ball_noise: float = 0.1,
        backend: BackendName = "mujoco",
        max_episode_steps: int = 1500,
        **kwargs: Any,
    ) -> None:
        for forbidden in ("xml_path", "patch_xml"):
            if forbidden in kwargs:
                raise ValueError(
                    f"MicroDuckFootballEnv builds its scene itself; pass scene=... "
                    f"instead of {forbidden}=..."
                )
        if kwargs.get("pixels_only"):
            raise ValueError("MicroDuckFootballEnv does not support pixels_only=True.")
        if not math.isfinite(action_scale) or action_scale <= 0:
            raise ValueError("action_scale must be finite and positive.")
        for name, value in (
            ("spawn_noise", spawn_noise),
            ("yaw_noise", yaw_noise),
            ("joint_reset_noise_scale", joint_reset_noise_scale),
            ("ball_noise", ball_noise),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        weights = dict(self.REWARD_WEIGHTS)
        unknown = set(reward_weights or {}) - set(weights)
        if unknown:
            raise ValueError(
                f"reward_weights name unknown terms {sorted(unknown)}; the terms are "
                f"{tuple(weights)}."
            )
        weights.update(reward_weights or {})
        if not all(math.isfinite(value) for value in weights.values()):
            raise ValueError("Reward weights must be finite.")
        self.reward_weights = weights
        self.scene_path = Path(scene).expanduser().resolve()
        self.action_scale = float(action_scale)
        self.respawn = bool(respawn)
        if respawn_mode not in ("in_place", "kickoff"):
            raise ValueError("respawn_mode must be 'in_place' or 'kickoff'.")
        if not math.isfinite(respawn_delay_s) or respawn_delay_s < 0:
            raise ValueError("respawn_delay_s must be finite and non-negative.")
        self.respawn_mode = respawn_mode
        self.respawn_delay_s = float(respawn_delay_s)
        self.spawn_noise = float(spawn_noise)
        self.yaw_noise = float(yaw_noise)
        self.joint_reset_noise_scale = float(joint_reset_noise_scale)
        self.ball_noise = float(ball_noise)
        super().__init__(
            xml_path=self.scene_path,
            patch_xml=False,
            backend=backend,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
        self._respawn_delay_steps = int(
            round(self.respawn_delay_s / (self.frame_skip * self._backend.timestep))
        )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    @classmethod
    def write_scene(
        cls,
        robot_scene: str | Path,
        *,
        root: str | Path | None = None,
        **scene_kwargs: Any,
    ) -> Path:
        r"""Build the football MJCF with :func:`build_football_scene` and cache it.

        The text is written once under ``root/football`` (default
        ``~/.cache/torchrl/microduck/football``), named after a hash of its
        content, so identical parameters share a file and worker processes
        load a file instead of rebuilding the scene.

        Args:
            robot_scene (str or Path): the MicroDuck walking scene.

        Keyword Args:
            root (str or Path, optional): cache directory.
            \\*\\*scene_kwargs: forwarded to :func:`build_football_scene`.

        Returns:
            The path of the cached MJCF.
        """
        xml = build_football_scene(robot_scene, **scene_kwargs)
        cache_root = (
            Path("~/.cache/torchrl/microduck").expanduser()
            if root is None
            else Path(root).expanduser()
        )
        directory = cache_root / "football"
        digest = hashlib.sha1(xml.encode("utf-8")).hexdigest()[:16]
        path = directory / f"microduck_football-{digest}.xml"
        if not path.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            partial = directory / f".{path.name}.{os.getpid()}.tmp"
            partial.write_text(xml)
            partial.replace(path)
        return path

    def _configure_from_model(self) -> None:
        import mujoco

        model = self._backend.mj_model
        numeric = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_NUMERIC, FOOTBALL_NUMERIC
        )
        if numeric < 0:
            raise ValueError(
                f"The football scene must carry a <numeric name={FOOTBALL_NUMERIC!r}> "
                "element; build it with build_football_scene."
            )
        start = int(model.numeric_adr[numeric])
        size = int(model.numeric_size[numeric])
        if size != 7:
            raise ValueError(
                f"The {FOOTBALL_NUMERIC!r} numeric must hold 7 values, got {size}."
            )
        values = [float(value) for value in model.numeric_data[start : start + size]]
        (
            self.pitch_length,
            self.pitch_width,
            self.goal_width,
            self.goal_height,
            self.goal_depth,
            self.ball_radius,
        ) = values[:6]
        self.players_per_team = int(round(values[6]))
        num_agents = 2 * self.players_per_team
        self.num_agents = num_agents
        expected = (
            num_agents * self.DUCK_NQ + self.BALL_NQ,
            num_agents * self.DUCK_NV + self.BALL_NV,
            num_agents * self.NUM_JOINTS,
        )
        if (model.nq, model.nv, model.nu) != expected:
            raise ValueError(
                f"Expected {num_agents} MicroDucks and a ball, (nq, nv, nu)={expected}, "
                f"got {(model.nq, model.nv, model.nu)}."
            )
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if key_id < 0:
            raise ValueError("The football scene must define a `STAND` keyframe.")
        key_qpos = torch.as_tensor(model.key_qpos[key_id].copy())
        key_ctrl = torch.as_tensor(model.key_ctrl[key_id].copy())
        home_joints = key_qpos[7 : self.DUCK_NQ]
        home_ctrl = key_ctrl[: self.NUM_JOINTS]
        joint_ids = torch.as_tensor(
            model.actuator_trnid[: self.NUM_JOINTS, 0].copy()
        ).long()
        if (joint_ids < 0).any():
            raise ValueError("Every MicroDuck actuator must target a joint.")
        joint_limited = torch.as_tensor(
            model.jnt_limited[joint_ids.numpy()].copy()
        ).bool()
        joint_range = torch.as_tensor(model.jnt_range[joint_ids.numpy()].copy())
        joint_low = torch.where(joint_limited, joint_range[:, 0], home_ctrl - torch.pi)
        joint_high = torch.where(joint_limited, joint_range[:, 1], home_ctrl + torch.pi)
        # Every duck of the scene is the same robot: the first block of the
        # keyframe gives the standing pose of all of them.
        self._home_joints = home_joints.to(device=self.device, dtype=self.dtype)
        self._home_ctrl = home_ctrl.to(device=self.device, dtype=self.dtype)
        self._joint_low = joint_low.to(device=self.device, dtype=self.dtype)
        self._joint_high = joint_high.to(device=self.device, dtype=self.dtype)
        self._standing_height = float(key_qpos[2])

        # Team layout: agents [0, n) are blue and attack +x, [n, 2n) are red.
        team = torch.arange(num_agents, device=self.device) // self.players_per_team
        self._team = team
        self._team_sign = (1 - 2 * team).to(self.dtype)
        slots = torch.tensor(
            kickoff_positions(
                self.players_per_team, self.pitch_length, self.pitch_width
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self._kickoff_xy = torch.cat((slots, -slots), dim=0)
        self._kickoff_yaw = torch.where(
            team == 0,
            torch.zeros(num_agents, dtype=self.dtype, device=self.device),
            torch.full((num_agents,), math.pi, dtype=self.dtype, device=self.device),
        )
        # Fixed observation order of the other players: teammates in index
        # order without oneself, then every opponent in index order.
        agents = torch.arange(num_agents, device=self.device)
        same = team.unsqueeze(0) == team.unsqueeze(1)
        others = agents.unsqueeze(0) != agents.unsqueeze(1)
        teammates = (same & others).nonzero()[:, 1].view(num_agents, -1)
        opponents = (~same).nonzero()[:, 1].view(num_agents, -1)
        self._teammate_index = teammates
        self._opponent_index = opponents
        self.feature_dim = 8 * self.players_per_team + 10
        self.observation_dim = self.PROPRIOCEPTION_DIM + self.feature_dim
        foot_geoms = [
            f"{TEAM_NAMES[int(team[agent])]}{int(agent) % self.players_per_team}/{geom}"
            for agent in range(num_agents)
            for geom in MicroDuckEnv.FOOT_GEOMS
        ]
        foot_sites = [
            f"{TEAM_NAMES[int(team[agent])]}{int(agent) % self.players_per_team}/{site}"
            for agent in range(num_agents)
            for site in MicroDuckEnv.FOOT_SITES
        ]
        self._foot_geom_ids = self._mujoco_ids("geom", foot_geoms)
        self._foot_site_ids = self._mujoco_ids("site", foot_sites)

        shape = (self.num_envs, num_agents)
        self._previous_action = torch.zeros(
            *shape, self.NUM_JOINTS, dtype=self.dtype, device=self.device
        )
        self._fallen = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._down = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._down_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self._goal = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    def _make_specs(self) -> None:
        self._configure_from_model()
        super()._make_specs()
        shape = (self.num_envs, self.num_agents)
        self.action_spec = Composite(
            agents=Composite(
                action=Bounded(
                    low=-1.0,
                    high=1.0,
                    shape=(*shape, self.NUM_JOINTS),
                    dtype=self.dtype,
                    device=self.device,
                ),
                shape=shape,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )
        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(
                    shape=(*shape, 1), dtype=self.dtype, device=self.device
                ),
                shape=shape,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _make_obs_spec(self) -> Composite:
        shape = (self.num_envs, self.num_agents)
        return Composite(
            agents=Composite(
                observation=Unbounded(
                    shape=(*shape, self.observation_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                fallen=Binary(
                    n=1, shape=(*shape, 1), dtype=torch.bool, device=self.device
                ),
                shape=shape,
                device=self.device,
            ),
            ball_position=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            goal=Unbounded(
                shape=(self.num_envs, 1), dtype=torch.long, device=self.device
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _split_state(
        self, state: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(duck_qpos, duck_qvel, ball_qpos, ball_qvel)`` in :attr:`dtype`."""
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        split_q = self.num_agents * self.DUCK_NQ
        split_v = self.num_agents * self.DUCK_NV
        ducks_q = qpos[..., :split_q].reshape(
            self.num_envs, self.num_agents, self.DUCK_NQ
        )
        ducks_v = qvel[..., :split_v].reshape(
            self.num_envs, self.num_agents, self.DUCK_NV
        )
        return ducks_q, ducks_v, qpos[..., split_q:], qvel[..., split_v:]

    @staticmethod
    def _yaw(quaternion: torch.Tensor) -> torch.Tensor:
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = quaternion.unbind(-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))

    @staticmethod
    def _to_body_frame(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Rotate planar world vectors ``(..., 2)`` into the frame of heading ``yaw``."""
        cos, sin = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)
        x, y = vector[..., :1], vector[..., 1:2]
        return torch.cat((cos * x + sin * y, -sin * x + cos * y), dim=-1)

    def _fallen_ducks(
        self, ducks_q: torch.Tensor, ducks_v: torch.Tensor
    ) -> torch.Tensor:
        upright = -_projected_gravity(ducks_q[..., 3:7])[..., 2]
        finite = torch.isfinite(ducks_q).all(dim=-1) & torch.isfinite(ducks_v).all(
            dim=-1
        )
        return (
            (ducks_q[..., 2] < MicroDuckEnv.MIN_HEIGHT_RATIO * self._standing_height)
            | (upright < MicroDuckEnv.MIN_UPRIGHT)
            | ~finite
        )

    def _kickoff_qpos(self, n: int, *, noisy: bool) -> torch.Tensor:
        """Standing pose of every duck on its kickoff slot, ``(n, num_agents, DUCK_NQ)``."""
        shape = (n, self.num_agents)
        xy = self._kickoff_xy.expand(n, -1, -1)
        yaw = self._kickoff_yaw.expand(n, -1)
        joints = self._home_joints.expand(*shape, -1)
        if noisy:
            xy = (
                xy
                + (
                    torch.rand(
                        *shape,
                        2,
                        generator=self.rng,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * 2.0
                    - 1.0
                )
                * self.spawn_noise
            )
            yaw = (
                yaw
                + (
                    torch.rand(
                        *shape, generator=self.rng, device=self.device, dtype=self.dtype
                    )
                    * 2.0
                    - 1.0
                )
                * self.yaw_noise
            )
            joints = (
                joints
                + (
                    torch.rand(
                        *shape,
                        self.NUM_JOINTS,
                        generator=self.rng,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * 2.0
                    - 1.0
                )
                * self.joint_reset_noise_scale
            )
        half = yaw / 2.0
        zeros = torch.zeros_like(half)
        quaternion = torch.stack((half.cos(), zeros, zeros, half.sin()), dim=-1)
        height = torch.full(
            (*shape, 1), self._standing_height, dtype=self.dtype, device=self.device
        )
        return torch.cat((xy, height, quaternion, joints), dim=-1)

    def _sample_initial_state(
        self,
        n: int,
        tensordict: TensorDictBase | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del tensordict
        ducks = self._kickoff_qpos(n, noisy=True).reshape(n, -1)
        ball_xy = (
            torch.rand(n, 2, generator=self.rng, device=self.device, dtype=self.dtype)
            * 2.0
            - 1.0
        ) * self.ball_noise
        ball = torch.cat(
            (
                ball_xy,
                torch.full(
                    (n, 1), self.ball_radius, dtype=self.dtype, device=self.device
                ),
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]], dtype=self.dtype, device=self.device
                ).expand(n, -1),
            ),
            dim=-1,
        )
        qpos = torch.cat((ducks, ball), dim=-1).to(self._backend.qpos0.dtype)
        qvel = torch.zeros(
            n, self._backend.nv, dtype=self._backend.qvel0.dtype, device=self.device
        )
        return qpos, qvel

    def _on_reset_all(self, tensordict: TensorDictBase | None = None) -> None:
        self._previous_action.zero_()
        self._fallen.zero_()
        self._down.zero_()
        self._down_steps.zero_()
        self._goal.zero_()

    def _on_reset_mask(
        self,
        mask: torch.Tensor,
        tensordict: TensorDictBase | None = None,
    ) -> None:
        mask = mask.squeeze(-1) if mask.ndim == 2 else mask
        self._previous_action = torch.where(
            mask[:, None, None],
            torch.zeros_like(self._previous_action),
            self._previous_action,
        )
        self._fallen = torch.where(
            mask[:, None], torch.zeros_like(self._fallen), self._fallen
        )
        self._down = torch.where(
            mask[:, None], torch.zeros_like(self._down), self._down
        )
        self._down_steps = torch.where(
            mask[:, None], torch.zeros_like(self._down_steps), self._down_steps
        )
        self._goal = torch.where(
            mask[:, None], torch.zeros_like(self._goal), self._goal
        )

    # ------------------------------------------------------------------
    # Observation, reward, termination
    # ------------------------------------------------------------------

    def _clock(self) -> tuple[torch.Tensor, torch.Tensor]:
        elapsed = self._step_count.to(self.dtype) * (
            self.frame_skip * self._backend.timestep
        )
        phase = (
            MicroDuckEnv.GAIT_PHASE_OFFSET
            + 2.0 * math.pi * MicroDuckEnv.GAIT_FREQUENCY_HZ * elapsed
        )
        ramp = (elapsed / MicroDuckEnv.GAIT_RAMP_DURATION_S).clamp(max=1.0)
        return phase, ramp

    def _make_obs(self, state: TensorDictBase) -> torch.Tensor:
        ducks_q, ducks_v, ball_q, ball_v = self._split_state(state)
        num_envs, num_agents = ducks_q.shape[:2]
        quaternion = ducks_q[..., 3:7]
        phase, ramp = self._clock()
        clock = torch.stack((phase.sin(), phase.cos(), ramp), dim=-1)
        proprioception = torch.cat(
            (
                _projected_gravity(quaternion),
                ducks_v[..., 3:6],
                _body_frame_linear_velocity(quaternion, ducks_v[..., :3]),
                torch.zeros(
                    num_envs, num_agents, 2, dtype=self.dtype, device=self.device
                ),
                ducks_q[..., 7:] - self._home_joints,
                ducks_v[..., 6:],
                clock.unsqueeze(1).expand(-1, num_agents, -1),
                self._previous_action,
            ),
            dim=-1,
        )
        yaw = self._yaw(quaternion)
        sign = self._team_sign.view(1, num_agents, 1)
        xy = ducks_q[..., :2]
        vxy = ducks_v[..., :2]
        half = torch.tensor(
            [self.pitch_length / 2.0, self.pitch_width / 2.0],
            dtype=self.dtype,
            device=self.device,
        )
        position = sign * xy / half
        heading = sign * torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        ball_xy = ball_q[:, None, :2]
        ball_relative = self._to_body_frame(ball_xy - xy, yaw)
        ball_height = ball_q[:, None, 2:3].expand(-1, num_agents, -1)
        ball_velocity = self._to_body_frame(ball_v[:, None, :2] - vxy, yaw)
        goal_x = sign * (self.pitch_length / 2.0)
        goal_offset = torch.cat((goal_x, torch.zeros_like(goal_x)), dim=-1)
        attacking_goal = self._to_body_frame(goal_offset - xy, yaw)
        own_goal = self._to_body_frame(-goal_offset - xy, yaw)
        # Relative position and velocity of every other player in the
        # agent's frame, then the fixed teammates/opponents order.
        relative_xy = self._to_body_frame(
            xy.unsqueeze(1) - xy.unsqueeze(2), yaw.unsqueeze(-1)
        )
        relative_vxy = self._to_body_frame(
            vxy.unsqueeze(1) - vxy.unsqueeze(2), yaw.unsqueeze(-1)
        )
        others = torch.cat((relative_xy, relative_vxy), dim=-1)
        rows = torch.arange(num_agents, device=self.device).unsqueeze(-1)
        teammates = others[:, rows, self._teammate_index].flatten(-2)
        opponents = others[:, rows, self._opponent_index].flatten(-2)
        time_left = (
            1.0 - self._step_count.to(self.dtype) / self.max_episode_steps
        ).clamp_min(0.0)
        features = torch.cat(
            (
                position,
                heading,
                ball_relative,
                ball_height,
                ball_velocity,
                attacking_goal,
                own_goal,
                teammates,
                opponents,
                time_left.view(num_envs, 1, 1).expand(-1, num_agents, -1),
            ),
            dim=-1,
        )
        return torch.cat((proprioception, features), dim=-1)

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, Any]:
        out: dict[str, Any] = {
            "agents": TensorDict(
                {
                    "observation": self._make_obs(state),
                    "fallen": self._fallen.unsqueeze(-1).clone(),
                },
                batch_size=(self.num_envs, self.num_agents),
                device=self.device,
            ),
            "ball_position": state["qpos"][..., -self.BALL_NQ : -4].to(self.dtype),
            "goal": self._goal.clone(),
        }
        if self.from_pixels:
            out["pixels"] = self._render_pixels()
        return out

    def _goals(self, ball_q: torch.Tensor) -> torch.Tensor:
        """Return ``1`` where blue scored, ``-1`` where red did, ``0`` otherwise."""
        x, y, z = ball_q[..., 0], ball_q[..., 1], ball_q[..., 2]
        in_mouth = (y.abs() < self.goal_width / 2.0) & (z < self.goal_height)
        blue = in_mouth & (x > self.pitch_length / 2.0)
        red = in_mouth & (x < -self.pitch_length / 2.0)
        return (blue.long() - red.long()).unsqueeze(-1)

    def _rewards(
        self,
        ducks_q: torch.Tensor,
        ducks_v: torch.Tensor,
        ball_q: torch.Tensor,
        ball_v: torch.Tensor,
        action: torch.Tensor,
        fell: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.reward_weights
        dt = self.frame_skip * self._backend.timestep
        sign = self._team_sign.unsqueeze(0)
        progress = sign * ball_v[:, :1]
        relative = ball_q[:, None, :2] - ducks_q[..., :2]
        distance = relative.norm(dim=-1)
        unit = relative / distance.clamp_min(1e-6).unsqueeze(-1)
        toward = (
            (ducks_v[..., :2] * unit)
            .sum(-1)
            .clamp(-self.APPROACH_SPEED_CAP, self.APPROACH_SPEED_CAP)
        )
        approach = toward * (
            distance > self.ball_radius + self.APPROACH_RADIUS_MARGIN
        ).to(self.dtype)
        action_rate = (action - self._previous_action).square().sum(-1)
        xy = ducks_q[..., :2]
        spacing = (xy.unsqueeze(1) - xy.unsqueeze(2)).norm(dim=-1)
        neighbors = (spacing < self.CROWD_RADIUS).sum(-1).to(self.dtype) - 1.0
        per_second = (
            weights["ball_progress"] * progress
            + weights["approach_ball"] * approach
            + weights["action_rate"] * action_rate
            + weights["crowd"] * neighbors
        )
        one_off = weights["goal"] * goal.to(self.dtype) * sign + weights[
            "fall"
        ] * fell.to(self.dtype)
        return (per_second * dt + one_off).unsqueeze(-1)

    def _prepare_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(-1.0, 1.0)
        target = self._home_ctrl + self.action_scale * action
        target = target.clamp(self._joint_low, self._joint_high)
        return target.reshape(action.shape[0], -1)

    def _upright_qpos(self, ducks_q: torch.Tensor) -> torch.Tensor:
        """Standing pose of every duck where it is, facing the goal it attacks."""
        n = ducks_q.shape[0]
        margin = 2.0 * _WALL_THICKNESS + 0.1
        half = torch.tensor(
            [self.pitch_length / 2.0 - margin, self.pitch_width / 2.0 - margin],
            dtype=self.dtype,
            device=self.device,
        )
        xy = torch.minimum(torch.maximum(ducks_q[..., :2], -half), half)
        half_yaw = self._kickoff_yaw.expand(n, -1) / 2.0
        zeros = torch.zeros_like(half_yaw)
        quaternion = torch.stack((half_yaw.cos(), zeros, zeros, half_yaw.sin()), dim=-1)
        height = torch.full(
            (n, self.num_agents, 1),
            self._standing_height,
            dtype=self.dtype,
            device=self.device,
        )
        joints = self._home_joints.expand(n, self.num_agents, -1)
        return torch.cat((xy, height, quaternion, joints), dim=-1)

    def _respawn(
        self,
        state: TensorDictBase,
        ducks_q: torch.Tensor,
        ducks_v: torch.Tensor,
        fallen: torch.Tensor,
    ) -> None:
        """Stand the selected ducks up, still, where the respawn mode says."""
        if self.respawn_mode == "kickoff":
            home = self._kickoff_qpos(self.num_envs, noisy=False)
        else:
            home = self._upright_qpos(ducks_q)
        column = fallen.unsqueeze(-1)
        ducks_q = torch.where(column, home, ducks_q)
        ducks_v = torch.where(column, torch.zeros_like(ducks_v), ducks_v)
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        split_q = self.num_agents * self.DUCK_NQ
        split_v = self.num_agents * self.DUCK_NV
        qpos = torch.cat(
            (ducks_q.reshape(self.num_envs, -1), qpos[..., split_q:]), dim=-1
        )
        qvel = torch.cat(
            (ducks_v.reshape(self.num_envs, -1), qvel[..., split_v:]), dim=-1
        )
        self._backend.reset_mask(
            fallen.any(dim=-1),
            qpos.to(self._backend.qpos0.dtype),
            qvel.to(self._backend.qvel0.dtype),
        )

    def _compute_reward(self, state, action, next_state) -> torch.Tensor:
        raise NotImplementedError("MicroDuckFootballEnv computes rewards in _step.")

    def _compute_done(self, state, next_state) -> torch.Tensor:
        raise NotImplementedError("MicroDuckFootballEnv computes termination in _step.")

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        active = tensordict.get("_step", None)
        if active is not None and not active.all():
            result = self._skip_tensordict(tensordict)
            result.update(self.full_reward_spec.zero())
            if active.any():
                # Snapshot indexing is shared by the MuJoCo backends. Only
                # live matches advance physics, clocks, and respawn state.
                indices = active.nonzero(as_tuple=True)[0]
                live = self[indices]
                result[active] = live._step(tensordict[active].exclude("_step"))
                self[indices] = live
                self._render_counter += 1
                if self.from_pixels:
                    self._last_pixels = result["pixels"]
            return result
        action = tensordict["agents", "action"].to(self.dtype).clamp(-1.0, 1.0)
        # A duck that is down holds its standing targets until it gets up.
        action = torch.where(self._down.unsqueeze(-1), torch.zeros_like(action), action)
        self._backend.step(self._prepare_ctrl(action), self.frame_skip)
        self._step_count += 1
        self._render_counter += 1

        state = self._state_td()
        ducks_q, ducks_v, ball_q, ball_v = self._split_state(state)
        fallen = self._fallen_ducks(ducks_q, ducks_v)
        finite = torch.isfinite(state["qpos"]).all(dim=-1) & torch.isfinite(
            state["qvel"]
        ).all(dim=-1)
        goal = self._goals(ball_q)
        # The fall penalty is paid on the step a duck goes down, not while it
        # lies there.
        fell = fallen & ~self._down
        reward = self._rewards(ducks_q, ducks_v, ball_q, ball_v, action, fell, goal)
        down = fallen | self._down
        if self.respawn:
            # A duck that just fell waits the whole delay, the others count down.
            steps = torch.where(
                fell,
                torch.full_like(self._down_steps, self._respawn_delay_steps),
                (self._down_steps - 1).clamp_min(0),
            )
            ready = down & (steps <= 0)
            if bool(ready.any()):
                self._respawn(state, ducks_q, ducks_v, ready)
                state = self._state_td()
            self._down_steps = torch.where(ready, torch.zeros_like(steps), steps)
            down = down & ~ready
        # A duck that fell or lies down restarts with a zero previous action.
        self._previous_action = torch.where(
            (fell | down).unsqueeze(-1), torch.zeros_like(action), action
        )
        self._down = down
        self._fallen = fell | down
        self._goal = goal
        terminated = ((goal != 0) | ~finite.unsqueeze(-1)).to(torch.bool)
        truncated = (self._step_count >= self.max_episode_steps).unsqueeze(-1)
        obs = self._build_obs_dict(state)
        obs["agents"]["reward"] = reward
        return TensorDict(
            {
                **obs,
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=(self.num_envs,),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Contact helpers and snapshot indexing
    # ------------------------------------------------------------------

    def foot_contacts(self) -> torch.Tensor:
        """Return a ``(num_envs, num_agents, 2)`` boolean tensor of foot contacts."""
        contacts = self._backend.geom_contacts(self._foot_geom_ids)
        return contacts.reshape(self.num_envs, self.num_agents, 2)

    def foot_heights(self) -> torch.Tensor:
        """Return the ``(num_envs, num_agents, 2)`` heights of the foot sites."""
        positions = self._backend.site_positions(self._foot_site_ids)
        return positions[..., 2].reshape(self.num_envs, self.num_agents, 2)

    def _index_extra_state(self, index: slice | torch.Tensor) -> dict[str, Any]:
        return {
            "previous_action": self._previous_action[index].clone(),
            "fallen": self._fallen[index].clone(),
            "down": self._down[index].clone(),
            "down_steps": self._down_steps[index].clone(),
            "goal": self._goal[index].clone(),
        }

    def _load_indexed_extra_state(self, state: dict[str, Any]) -> None:
        self._previous_action = state["previous_action"].clone()
        self._fallen = state["fallen"].clone()
        self._down = state["down"].clone()
        self._down_steps = state["down_steps"].clone()
        self._goal = state["goal"].clone()

    def _set_indexed_extra_state(
        self,
        index: slice | torch.Tensor,
        source: MujocoEnv,
    ) -> None:
        if not isinstance(source, MicroDuckFootballEnv):
            raise TypeError(
                "MicroDuckFootballEnv snapshots can only be restored from a "
                "MicroDuckFootballEnv."
            )
        self._previous_action[index] = source._previous_action.to(self.device)
        self._fallen[index] = source._fallen.to(self.device)
        self._down[index] = source._down.to(self.device)
        self._down_steps[index] = source._down_steps.to(self.device)
        self._goal[index] = source._goal.to(self.device)
