# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Cube-to-bowl manipulation task for MuJoCo macro-control examples."""

from __future__ import annotations

import importlib.util
import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import Binary, Composite, Unbounded
from torchrl.envs.custom.mujoco._ur_primitives import URScriptPrimitiveTransform
from torchrl.envs.custom.mujoco.base import MujocoEnv
from torchrl.envs.transforms._base import Transform

_has_mujoco = importlib.util.find_spec("mujoco") is not None


class CubeBowlEnv(MujocoEnv):
    r"""UR-style cube-to-bowl manipulation scene.

    ``CubeBowlEnv`` is a compact MuJoCo task meant for tutorials on custom
    MuJoCo environments, action sequences, and scripted robot primitives. The
    scene composes the MuJoCo Menagerie UR5e arm and Robotiq 2F-85 gripper from
    a local Menagerie checkout, without vendoring their mesh assets in TorchRL.
    The low-level action is a 7D position command: six arm joint targets
    followed by one gripper command. Observations include
    privileged manipulation diagnostics such as the pinch pose and gripper pad
    positions. The task reward is sparse: it is ``1`` when the cube center is
    within ``placement_tolerance`` of the bowl target coordinate and ``0``
    otherwise. ``cube_pos`` and ``bowl_pos`` may also be provided in the
    reset TensorDict to start an episode from a different reachable layout.

    Args:
        menagerie_path: optional path to a local ``mujoco_menagerie`` checkout.
            When omitted, the ``TORCHRL_MUJOCO_MENAGERIE_PATH`` environment
            variable is used.
        cube_position: initial xyz position of the cube center.
        bowl_position: xyz position of the bowl body. The task target is the
            ``bowl_target`` site inside that body.
        placement_tolerance: success radius, in meters, around the bowl target.
        terminate_on_success: if ``True``, terminate when the cube reaches the
            bowl target.
        backend: physics backend. Defaults to ``"mujoco"`` because the tutorial
            task uses contacts and MuJoCo's site Jacobians for scripted IK.
        \*\*kwargs: forwarded to :class:`~torchrl.envs.MujocoEnv`.

    Examples:
        >>> from torchrl.envs import CubeBowlEnv  # doctest: +SKIP
        >>> env = CubeBowlEnv(max_episode_steps=50)  # doctest: +SKIP
        >>> td = env.rollout(3)  # doctest: +SKIP
    """

    DEFAULT_BACKEND = "mujoco"
    FRAME_SKIP = 5
    RESET_NOISE_SCALE = 0.0
    OBJECT_HALF_SIZE = 0.022
    CUBE_HALF_SIZE = OBJECT_HALF_SIZE
    ROBOT_QPOS_DIM = 6
    MENAGERIE_GRIPPER_QPOS_DIM = 8
    CUBE_QPOS_START = ROBOT_QPOS_DIM + MENAGERIE_GRIPPER_QPOS_DIM
    CUBE_QVEL_START = ROBOT_QPOS_DIM + MENAGERIE_GRIPPER_QPOS_DIM
    BOWL_TARGET_SITE_NAME = "bowl_target"
    MENAGERIE_ENV_VAR = "TORCHRL_MUJOCO_MENAGERIE_PATH"
    MENAGERIE_CUBE_POSITION = (0.45, -0.18, 0.035)
    MENAGERIE_BOWL_POSITION = (0.45, 0.08, 0.01)
    MENAGERIE_BOWL_TARGET_OFFSET = (0.0, 0.0, 0.015)
    # Calibration values for converting a desired pad-center distance into the
    # Menagerie Robotiq actuator's 0..255 command range.
    MENAGERIE_GRIPPER_OPEN_PAD_DISTANCE = 0.09313070774078369
    MENAGERIE_GRIPPER_CUBE_WIDTH_CTRL = 150.0
    MENAGERIE_GRIPPER_CUBE_WIDTH_PAD_DISTANCE = 0.04430961608886719
    MENAGERIE_GRIPPER_GRASP_MARGIN = 0.001
    MENAGERIE_PINCH_SITE_NAME = "gripper/pinch"
    MENAGERIE_LEFT_PAD_GEOM_NAMES = ("gripper/left_pad1", "gripper/left_pad2")
    MENAGERIE_RIGHT_PAD_GEOM_NAMES = ("gripper/right_pad1", "gripper/right_pad2")
    MENAGERIE_HOME_QPOS = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
    MENAGERIE_ROBOT_JOINT_NAMES = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )

    def __init__(
        self,
        *,
        menagerie_path: str | Path | None = None,
        cube_position: tuple[float, float, float] | None = None,
        bowl_position: tuple[float, float, float] | None = None,
        placement_tolerance: float = 0.06,
        terminate_on_success: bool = False,
        backend: Literal["mujoco-torch", "mjx", "mujoco"] = "mujoco",
        **kwargs,
    ) -> None:
        default_cube_position = self.MENAGERIE_CUBE_POSITION
        default_bowl_position = self.MENAGERIE_BOWL_POSITION
        self._gripper_qpos_dim = self.MENAGERIE_GRIPPER_QPOS_DIM
        self._pinch_site_name = self.MENAGERIE_PINCH_SITE_NAME
        self._bowl_target_offset = self.MENAGERIE_BOWL_TARGET_OFFSET
        self._robot_home_qpos = self.MENAGERIE_HOME_QPOS
        self._menagerie_path = self._resolve_menagerie_path(menagerie_path)
        if cube_position is None:
            cube_position = default_cube_position
        if bowl_position is None:
            bowl_position = default_bowl_position
        self.cube_position = tuple(float(v) for v in cube_position)
        self.bowl_position = tuple(float(v) for v in bowl_position)
        self.placement_tolerance = float(placement_tolerance)
        self.terminate_on_success = bool(terminate_on_success)
        self._cube_qpos_start = self.ROBOT_QPOS_DIM + self._gripper_qpos_dim
        self._cube_qvel_start = self.ROBOT_QPOS_DIM + self._gripper_qpos_dim
        self._pinch_site_id: int | None = None
        self._bowl_body_id: int | None = None
        self._bowl_target_site_id: int | None = None
        self._left_pad_geom_ids: tuple[int, ...] = ()
        self._right_pad_geom_ids: tuple[int, ...] = ()
        super().__init__(backend=backend, **kwargs)
        self._configure_qpos_layout()
        self._bowl_target_pos = torch.tensor(
            tuple(
                self.bowl_position[index] + self._bowl_target_offset[index]
                for index in range(3)
            ),
            dtype=self.dtype,
            device=self.device,
        ).view(1, 3)
        self._pinch_site_id = self._find_site_id(self._pinch_site_name)
        self._bowl_body_id = self._find_body_id("bowl")
        self._bowl_target_site_id = self._find_site_id(self.BOWL_TARGET_SITE_NAME)
        self._left_pad_geom_ids = self._find_geom_ids(
            self.MENAGERIE_LEFT_PAD_GEOM_NAMES
        )
        self._right_pad_geom_ids = self._find_geom_ids(
            self.MENAGERIE_RIGHT_PAD_GEOM_NAMES
        )

    def _configure_qpos_layout(self) -> None:
        model = getattr(self._backend, "_m", getattr(self._backend, "_m_mj", None))
        if not _has_mujoco or model is None:
            self._robot_qpos_indices = tuple(range(self.ROBOT_QPOS_DIM))
            self._robot_qvel_indices = tuple(range(self.ROBOT_QPOS_DIM))
            self._gripper_qpos_indices = tuple(
                range(self.ROBOT_QPOS_DIM, self._cube_qpos_start)
            )
            self._gripper_qvel_indices = tuple(
                range(self.ROBOT_QPOS_DIM, self._cube_qvel_start)
            )
            return

        import mujoco

        robot_joint_names = self.MENAGERIE_ROBOT_JOINT_NAMES
        robot_qpos_indices: list[int] = []
        robot_qvel_indices: list[int] = []
        for name in robot_joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"CubeBowlEnv XML is missing joint {name!r}.")
            robot_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
            robot_qvel_indices.append(int(model.jnt_dofadr[joint_id]))

        cube_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint"
        )
        if cube_joint_id < 0:
            raise ValueError("CubeBowlEnv XML is missing joint 'cube_freejoint'.")
        self._cube_qpos_start = int(model.jnt_qposadr[cube_joint_id])
        self._cube_qvel_start = int(model.jnt_dofadr[cube_joint_id])

        robot_joint_set = set(robot_qpos_indices)
        gripper_qpos_indices: list[int] = []
        gripper_qvel_indices: list[int] = []
        for joint_id in range(model.njnt):
            qpos_adr = int(model.jnt_qposadr[joint_id])
            if qpos_adr in robot_joint_set or qpos_adr == self._cube_qpos_start:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is not None and name.startswith("gripper/"):
                gripper_qpos_indices.append(qpos_adr)
                gripper_qvel_indices.append(int(model.jnt_dofadr[joint_id]))

        self._robot_qpos_indices = tuple(robot_qpos_indices)
        self._robot_qvel_indices = tuple(robot_qvel_indices)
        self._gripper_qpos_indices = tuple(gripper_qpos_indices)
        self._gripper_qvel_indices = tuple(gripper_qvel_indices)

    # ------------------------------------------------------------------
    # XML loading and construction-time scene randomization.
    # ------------------------------------------------------------------

    def _load_xml(self, xml_path: str | Path | None) -> str:
        if xml_path is not None:
            raise ValueError(
                "CubeBowlEnv composes a MuJoCo Menagerie UR5e scene; pass "
                "menagerie_path=... instead of xml_path=..."
            )
        return self._make_menagerie_ur5e_xml(self._menagerie_path)

    @classmethod
    def _resolve_menagerie_path(cls, menagerie_path: str | Path | None) -> Path:
        if menagerie_path is None:
            menagerie_path = os.environ.get(cls.MENAGERIE_ENV_VAR)
        if menagerie_path is None:
            raise ValueError(
                "CubeBowlEnv requires a local MuJoCo "
                f"Menagerie checkout. Pass menagerie_path=... or set "
                f"{cls.MENAGERIE_ENV_VAR}."
            )
        path = Path(menagerie_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo Menagerie path does not exist: {path}.")
        return path

    def _make_menagerie_ur5e_xml(self, menagerie_path: Path) -> str:
        ur_dir = menagerie_path / "universal_robots_ur5e"
        gripper_dir = menagerie_path / "robotiq_2f85"
        ur_xml = ur_dir / "ur5e.xml"
        gripper_xml = gripper_dir / "2f85.xml"
        if not ur_xml.exists() or not gripper_xml.exists():
            raise FileNotFoundError(
                "CubeBowlEnv requires "
                "universal_robots_ur5e/ur5e.xml and robotiq_2f85/2f85.xml "
                f"under {menagerie_path}."
            )
        root = ET.parse(ur_xml).getroot()
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            root.insert(0, compiler)
        compiler.set("meshdir", str((ur_dir / "assets").resolve()))
        gripper = self._load_prefixed_menagerie_gripper(gripper_xml, gripper_dir)
        self._merge_menagerie_gripper(root, gripper)
        self._insert_menagerie_task_scene(root)
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def _format_vec(values: tuple[float, float, float]) -> str:
        return " ".join(f"{value:.8g}" for value in values)

    @staticmethod
    def _ensure_child(root: ET.Element, tag: str) -> ET.Element:
        child = root.find(tag)
        if child is None:
            child = ET.Element(tag)
            root.append(child)
        return child

    @staticmethod
    def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element | None:
        for parent in root.iter():
            if child in list(parent):
                return parent
        return None

    @staticmethod
    def _named_elements(root: ET.Element, tags: set[str]) -> dict[str, str]:
        return {
            elem.attrib["name"]: f"gripper/{elem.attrib['name']}"
            for elem in root.iter()
            if elem.tag in tags and "name" in elem.attrib
        }

    @classmethod
    def _load_prefixed_menagerie_gripper(
        cls, gripper_xml: Path, gripper_dir: Path
    ) -> ET.Element:
        root = ET.parse(gripper_xml).getroot()
        classes: set[str] = set()
        for elem in root.iter():
            if elem.tag == "default" and "class" in elem.attrib:
                classes.add(elem.attrib["class"])
            if "class" in elem.attrib:
                classes.add(elem.attrib["class"])
            if "childclass" in elem.attrib:
                classes.add(elem.attrib["childclass"])
        class_map = {name: f"gripper_{name}" for name in classes}
        body_map = cls._named_elements(root, {"body"})
        joint_map = cls._named_elements(root, {"joint"})
        geom_map = cls._named_elements(root, {"geom"})
        site_map = cls._named_elements(root, {"site"})
        material_map = cls._named_elements(root, {"material"})
        actuator_map = cls._named_elements(
            root, {"general", "position", "motor", "velocity"}
        )
        tendon_map = cls._named_elements(root, {"fixed", "spatial", "tendon"})
        mesh_map: dict[str, str] = {}
        for elem in root.iter("mesh"):
            if "file" not in elem.attrib:
                continue
            old_name = elem.attrib.get("name") or Path(elem.attrib["file"]).stem
            mesh_map[old_name] = f"gripper/{old_name}"
            elem.set("name", mesh_map[old_name])
            elem.set(
                "file", str((gripper_dir / "assets" / elem.attrib["file"]).resolve())
            )

        reference_maps = {
            "mesh": mesh_map,
            "material": material_map,
            "joint": joint_map,
            "joint1": joint_map,
            "joint2": joint_map,
            "body1": body_map,
            "body2": body_map,
            "tendon": tendon_map,
            "site": site_map,
        }
        for elem in root.iter():
            if "class" in elem.attrib:
                elem.set("class", class_map[elem.attrib["class"]])
            if "childclass" in elem.attrib:
                elem.set("childclass", class_map[elem.attrib["childclass"]])
            if elem.tag == "body" and "name" in elem.attrib:
                elem.set("name", body_map[elem.attrib["name"]])
            elif elem.tag == "joint" and "name" in elem.attrib:
                elem.set("name", joint_map[elem.attrib["name"]])
            elif elem.tag == "geom" and "name" in elem.attrib:
                elem.set("name", geom_map[elem.attrib["name"]])
            elif elem.tag == "site" and "name" in elem.attrib:
                elem.set("name", site_map[elem.attrib["name"]])
            elif elem.tag == "material" and "name" in elem.attrib:
                elem.set("name", material_map[elem.attrib["name"]])
            elif elem.tag in {"general", "position", "motor", "velocity"}:
                if "name" in elem.attrib:
                    elem.set("name", actuator_map[elem.attrib["name"]])
            elif elem.tag in {"fixed", "spatial", "tendon"} and "name" in elem.attrib:
                elem.set("name", tendon_map[elem.attrib["name"]])
            for attr, mapping in reference_maps.items():
                if attr in elem.attrib and elem.attrib[attr] in mapping:
                    elem.set(attr, mapping[elem.attrib[attr]])
        return root

    @classmethod
    def _merge_menagerie_gripper(cls, root: ET.Element, gripper: ET.Element) -> None:
        root_default = cls._ensure_child(root, "default")
        gripper_default = gripper.find("default")
        if gripper_default is not None:
            for child in gripper_default:
                root_default.append(deepcopy(child))

        root_asset = cls._ensure_child(root, "asset")
        gripper_asset = gripper.find("asset")
        if gripper_asset is not None:
            for child in gripper_asset:
                root_asset.append(deepcopy(child))

        attachment_site = root.find(".//site[@name='attachment_site']")
        if attachment_site is None:
            raise ValueError("Menagerie UR5e model is missing attachment_site.")
        attachment_parent = cls._find_parent(root, attachment_site)
        if attachment_parent is None:
            raise ValueError("Could not find parent of attachment_site.")
        gripper_body = gripper.find("worldbody/body")
        if gripper_body is None:
            raise ValueError("Menagerie Robotiq 2F-85 model has no root body.")
        gripper_body = deepcopy(gripper_body)
        gripper_body.set("pos", attachment_site.attrib.get("pos", "0 0 0"))
        if "quat" in attachment_site.attrib:
            gripper_body.set("quat", attachment_site.attrib["quat"])
        children = list(attachment_parent)
        attachment_parent.insert(children.index(attachment_site) + 1, gripper_body)

        for section_name in ("contact", "tendon", "equality"):
            source = gripper.find(section_name)
            if source is None:
                continue
            target = cls._ensure_child(root, section_name)
            for child in source:
                target.append(deepcopy(child))

        root_actuator = cls._ensure_child(root, "actuator")
        gripper_actuator = gripper.find("actuator")
        if gripper_actuator is not None:
            for child in gripper_actuator:
                root_actuator.append(deepcopy(child))

    def _insert_menagerie_task_scene(self, root: ET.Element) -> None:
        worldbody = self._ensure_child(root, "worldbody")
        worldbody.insert(
            0,
            ET.Element(
                "camera",
                {
                    "name": "overview",
                    "pos": "1.8 -0.9 1.5",
                    "xyaxes": "0.563147 0.826357 0 -0.521717 0.355541 0.775501",
                    "fovy": "45",
                },
            ),
        )
        worldbody.insert(
            1,
            ET.Element(
                "geom",
                {
                    "name": "floor",
                    "type": "plane",
                    "size": "1.5 1.5 0.05",
                    "rgba": "0.8 0.85 0.8 1",
                },
            ),
        )
        bowl = self._make_bowl_body(
            bottom_radius=0.08,
            wall_offset=0.075,
            wall_half_height=0.04,
            target_height=self._bowl_target_offset[2],
        )
        bowl_bottom = bowl.find("geom[@name='bowl_bottom']")
        if bowl_bottom is not None:
            bowl_bottom.set("friction", "1 0.02 0.001")
        cube = self._make_cube_body()
        worldbody.append(bowl)
        worldbody.append(cube)
        for geom in root.iter("geom"):
            if geom.attrib.get("name") in (
                "gripper/left_pad1",
                "gripper/left_pad2",
                "gripper/right_pad1",
                "gripper/right_pad2",
            ):
                geom.set("type", "box")
                geom.set("size", "0.016 0.010 0.025")
                geom.set("friction", "5 0.1 0.001")
                geom.set("condim", "4")
                geom.set("solimp", "0.95 0.99 0.001")
                geom.set("solref", "0.004 1")
        actuator = root.find(".//general[@name='gripper/fingers_actuator']")
        if actuator is not None:
            actuator.set("forcerange", "-80 80")

    def _make_bowl_body(
        self,
        *,
        bottom_radius: float,
        wall_offset: float,
        wall_half_height: float,
        target_height: float,
    ) -> ET.Element:
        color = "0.25 0.65 0.55 0.7"
        bowl = ET.Element(
            "body", {"name": "bowl", "pos": self._format_vec(self.bowl_position)}
        )
        ET.SubElement(
            bowl,
            "geom",
            {
                "name": "bowl_bottom",
                "type": "cylinder",
                "size": f"{bottom_radius:.8g} 0.008",
                "rgba": color,
            },
        )
        for name, pos, size in (
            (
                "bowl_front",
                f"0 {-wall_offset:.8g} {wall_half_height:.8g}",
                f"{bottom_radius:.8g} 0.006 {wall_half_height:.8g}",
            ),
            (
                "bowl_back",
                f"0 {wall_offset:.8g} {wall_half_height:.8g}",
                f"{bottom_radius:.8g} 0.006 {wall_half_height:.8g}",
            ),
            (
                "bowl_left",
                f"{-wall_offset:.8g} 0 {wall_half_height:.8g}",
                f"0.006 {bottom_radius:.8g} {wall_half_height:.8g}",
            ),
            (
                "bowl_right",
                f"{wall_offset:.8g} 0 {wall_half_height:.8g}",
                f"0.006 {bottom_radius:.8g} {wall_half_height:.8g}",
            ),
        ):
            ET.SubElement(
                bowl,
                "geom",
                {
                    "name": name,
                    "type": "box",
                    "pos": pos,
                    "size": size,
                    "rgba": color,
                },
            )
        ET.SubElement(
            bowl,
            "site",
            {
                "name": self.BOWL_TARGET_SITE_NAME,
                "pos": f"0 0 {target_height:.8g}",
                "size": "0.01",
                "rgba": "0.1 0.9 0.1 0.7",
            },
        )
        return bowl

    def _make_cube_body(self) -> ET.Element:
        cube = ET.Element(
            "body", {"name": "cube", "pos": self._format_vec(self.cube_position)}
        )
        ET.SubElement(cube, "freejoint", {"name": "cube_freejoint"})
        ET.SubElement(
            cube,
            "geom",
            {
                "name": "cube_geom",
                "type": "box",
                "size": (
                    f"{self.OBJECT_HALF_SIZE:.8g} "
                    f"{self.OBJECT_HALF_SIZE:.8g} "
                    f"{self.OBJECT_HALF_SIZE:.8g}"
                ),
                "mass": "0.02",
                "friction": "3 0.05 0.001",
                "condim": "4",
                "solimp": "0.95 0.99 0.001",
                "solref": "0.004 1",
                "rgba": "0.95 0.35 0.15 1",
            },
        )
        return cube

    # ------------------------------------------------------------------
    # Specs and reset state.
    # ------------------------------------------------------------------

    def _make_specs(self) -> None:
        super()._make_specs()
        reset_position_spec = Composite(
            cube_pos=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            bowl_pos=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            shape=(self.num_envs,),
            device=self.device,
        )
        self.state_spec = reset_position_spec

    def _make_obs_spec(self) -> Composite:
        return Composite(
            robot_qpos=Unbounded(
                shape=(self.num_envs, self.ROBOT_QPOS_DIM),
                dtype=self.dtype,
                device=self.device,
            ),
            robot_qvel=Unbounded(
                shape=(self.num_envs, self.ROBOT_QPOS_DIM),
                dtype=self.dtype,
                device=self.device,
            ),
            gripper_qpos=Unbounded(
                shape=(self.num_envs, self._gripper_qpos_dim),
                dtype=self.dtype,
                device=self.device,
            ),
            gripper_qvel=Unbounded(
                shape=(self.num_envs, self._gripper_qpos_dim),
                dtype=self.dtype,
                device=self.device,
            ),
            pinch_pos=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            pinch_quat=Unbounded(
                shape=(self.num_envs, 4), dtype=self.dtype, device=self.device
            ),
            gripper_left_pad_pos=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            gripper_right_pad_pos=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            cube_pos=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            cube_quat=Unbounded(
                shape=(self.num_envs, 4), dtype=self.dtype, device=self.device
            ),
            bowl_pos=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            success=Binary(
                n=1,
                shape=(self.num_envs, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _sample_initial_state(
        self,
        n: int,
        tensordict: TensorDictBase | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos, qvel = super()._sample_initial_state(n, tensordict)
        if self._robot_home_qpos is not None:
            qpos[..., self._robot_qpos_indices] = torch.tensor(
                self._robot_home_qpos, dtype=qpos.dtype, device=qpos.device
            )
        cube_pos = torch.tensor(
            self.cube_position, dtype=qpos.dtype, device=qpos.device
        ).expand(n, 3)
        if tensordict is not None and "cube_pos" in tensordict.keys(True, True):
            cube_pos = tensordict.get("cube_pos").to(
                device=qpos.device, dtype=qpos.dtype
            )
            cube_pos = cube_pos.reshape(n, 3)
        qpos[..., self._cube_qpos_start : self._cube_qpos_start + 3] = cube_pos
        qpos[..., self._cube_qpos_start + 3 : self._cube_qpos_start + 7] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), dtype=qpos.dtype, device=qpos.device
        )
        qvel[..., self._cube_qvel_start : self._cube_qvel_start + 6] = 0.0
        bowl_pos = torch.tensor(
            self.bowl_position, dtype=qpos.dtype, device=qpos.device
        ).expand(n, 3)
        if tensordict is not None and "bowl_pos" in tensordict.keys(True, True):
            target_pos = tensordict.get("bowl_pos").to(
                device=qpos.device, dtype=qpos.dtype
            )
            target_pos = target_pos.reshape(n, 3)
            offset = torch.tensor(
                self._bowl_target_offset, dtype=qpos.dtype, device=qpos.device
            )
            bowl_pos = target_pos - offset
        self._set_bowl_body_position(bowl_pos)
        return qpos, qvel

    # ------------------------------------------------------------------
    # Observations, reward, and done.
    # ------------------------------------------------------------------

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        out = self._make_obs_split(state) if not self.pixels_only else {}
        if self.from_pixels:
            out["pixels"] = self._backend.render(
                camera_id=self.camera_id,
                width=self.render_width,
                height=self.render_height,
                background=self.RENDER_BACKGROUND,
            )
        return out

    def _make_obs_split(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        cube_pos = qpos[..., self._cube_qpos_start : self._cube_qpos_start + 3]
        bowl_pos = self._target_pos().expand(self.num_envs, 3)
        pinch_pos = self._pinch_pos().to(self.dtype)
        return {
            "robot_qpos": qpos[..., self._robot_qpos_indices].clone(),
            "robot_qvel": qvel[..., self._robot_qvel_indices].clone(),
            "gripper_qpos": qpos[..., self._gripper_qpos_indices].clone(),
            "gripper_qvel": qvel[..., self._gripper_qvel_indices].clone(),
            "pinch_pos": pinch_pos,
            "pinch_quat": self._pinch_quat().to(self.dtype),
            "gripper_left_pad_pos": self._pad_pos(
                self._left_pad_geom_ids, fallback=pinch_pos
            ).to(self.dtype),
            "gripper_right_pad_pos": self._pad_pos(
                self._right_pad_geom_ids, fallback=pinch_pos
            ).to(self.dtype),
            "cube_pos": cube_pos.clone(),
            "cube_quat": qpos[
                ..., self._cube_qpos_start + 3 : self._cube_qpos_start + 7
            ].clone(),
            "bowl_pos": bowl_pos.clone(),
            "success": self._success(cube_pos, bowl_pos),
        }

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state, action
        cube_pos = next_state["qpos"].to(self.dtype)[
            ..., self._cube_qpos_start : self._cube_qpos_start + 3
        ]
        target = self._target_pos().expand(self.num_envs, 3)
        return self._success(cube_pos, target).to(self.dtype)

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        if not self.terminate_on_success:
            return torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        cube_pos = next_state["qpos"].to(self.dtype)[
            ..., self._cube_qpos_start : self._cube_qpos_start + 3
        ]
        return self._success(cube_pos, self._target_pos().expand(self.num_envs, 3))

    def _success(self, cube_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = (cube_pos - target).norm(dim=-1, keepdim=True)
        return dist <= self.placement_tolerance

    # ------------------------------------------------------------------
    # Scripted-control helper API.
    # ------------------------------------------------------------------

    @property
    def gripper_open_ctrl(self) -> float:
        """Low-level command that opens the gripper."""
        return 0.0

    @property
    def gripper_close_ctrl(self) -> float:
        """Low-level command that closes the gripper for object grasping."""
        target_width = 2 * self.OBJECT_HALF_SIZE - self.MENAGERIE_GRIPPER_GRASP_MARGIN
        return self.gripper_ctrl_for_width(target_width)

    def gripper_ctrl_for_width(
        self, width: float | torch.Tensor
    ) -> float | torch.Tensor:
        """Return a low-level gripper command for an object width.

        Args:
            width: desired grasp width in meters.

        Returns:
            The corresponding low-level gripper command.

        Examples:
            >>> from torchrl.envs import CubeBowlEnv  # doctest: +SKIP
            >>> env = CubeBowlEnv()  # doctest: +SKIP
            >>> env.gripper_ctrl_for_width(2 * env.OBJECT_HALF_SIZE)  # doctest: +SKIP
        """
        return self._menagerie_gripper_ctrl_for_width(width)

    @classmethod
    def _menagerie_gripper_ctrl_for_width(
        cls, width: float | torch.Tensor
    ) -> float | torch.Tensor:
        ctrl_per_meter = cls.MENAGERIE_GRIPPER_CUBE_WIDTH_CTRL / (
            cls.MENAGERIE_GRIPPER_OPEN_PAD_DISTANCE
            - cls.MENAGERIE_GRIPPER_CUBE_WIDTH_PAD_DISTANCE
        )
        ctrl = (cls.MENAGERIE_GRIPPER_OPEN_PAD_DISTANCE - width) * ctrl_per_meter
        return cls._clamp_gripper_ctrl(ctrl, 0.0, 255.0)

    @staticmethod
    def _clamp_gripper_ctrl(
        ctrl: float | torch.Tensor,
        low: float,
        high: float,
    ) -> float | torch.Tensor:
        if isinstance(ctrl, torch.Tensor):
            return ctrl.clamp(min=low, max=high)
        return min(high, max(low, ctrl))

    @property
    def robot_home_qpos(self) -> tuple[float, ...] | None:
        """Environment-defined home joint configuration for scripted control."""
        return self._robot_home_qpos

    def low_level_action(
        self,
        robot_qpos: torch.Tensor,
        gripper: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a seven-dimensional low-level action.

        Args:
            robot_qpos: six robot joint targets.
            gripper: optional gripper command. If omitted,
                :attr:`gripper_open_ctrl` is used.

        Returns:
            A tensor whose last dimension is ``[six joints, gripper]``.

        Examples:
            >>> from torchrl.envs import CubeBowlEnv  # doctest: +SKIP
            >>> env = CubeBowlEnv()  # doctest: +SKIP
            >>> td = env.reset()  # doctest: +SKIP
            >>> env.low_level_action(td["robot_qpos"]).shape  # doctest: +SKIP
            torch.Size([1, 7])
        """
        action = torch.zeros(
            robot_qpos.shape[:-1] + (self.ROBOT_QPOS_DIM + 1,),
            dtype=robot_qpos.dtype,
            device=robot_qpos.device,
        )
        action[..., : self.ROBOT_QPOS_DIM] = robot_qpos[..., : self.ROBOT_QPOS_DIM]
        if gripper is None:
            action[..., -1] = self.gripper_open_ctrl
        elif isinstance(gripper, torch.Tensor):
            gripper = gripper.to(dtype=robot_qpos.dtype, device=robot_qpos.device)
            if gripper.numel() == 1:
                action[..., -1] = gripper.reshape(())
            else:
                action[..., -1:] = gripper.reshape(robot_qpos.shape[:-1] + (1,))
        else:
            action[..., -1] = float(gripper)
        return action

    @staticmethod
    def pose_at(
        xyz: torch.Tensor,
        quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack an ``xyz + quaternion`` pose tensor.

        Args:
            xyz: Cartesian position with trailing dimension ``3``.
            quat: optional quaternion with trailing dimension ``4``. If omitted,
                the identity quaternion is used.

        Examples:
            >>> import torch
            >>> from torchrl.envs import CubeBowlEnv
            >>> CubeBowlEnv.pose_at(torch.zeros(1, 3)).shape
            torch.Size([1, 7])
        """
        if quat is None:
            quat = torch.zeros(
                xyz.shape[:-1] + (4,), dtype=xyz.dtype, device=xyz.device
            )
            quat[..., 0] = 1.0
        else:
            quat = quat.to(dtype=xyz.dtype, device=xyz.device)
            quat = quat.expand(xyz.shape[:-1] + (4,))
        return torch.cat([xyz, quat], dim=-1)

    def gripper_cube_distance(self, observation: TensorDictBase) -> torch.Tensor:
        """Return the shortest gripper-pad distance to the cube surface.

        Args:
            observation: observation TensorDict emitted by this environment.

        Examples:
            >>> from torchrl.envs import CubeBowlEnv  # doctest: +SKIP
            >>> env = CubeBowlEnv()  # doctest: +SKIP
            >>> td = env.reset()  # doctest: +SKIP
            >>> env.gripper_cube_distance(td).shape  # doctest: +SKIP
            torch.Size([1, 1])
        """
        cube_pos = observation["cube_pos"]
        half_size = torch.full_like(cube_pos, self.OBJECT_HALF_SIZE)

        def pad_to_cube(pad_pos: torch.Tensor) -> torch.Tensor:
            q = (pad_pos - cube_pos).abs() - half_size
            outside = q.clamp_min(0.0).norm(dim=-1, keepdim=True)
            inside = q.max(dim=-1, keepdim=True).values.clamp_max(0.0)
            return outside + inside

        left_distance = pad_to_cube(observation["gripper_left_pad_pos"])
        right_distance = pad_to_cube(observation["gripper_right_pad_pos"])
        return torch.minimum(left_distance, right_distance).clamp_min(0.0)

    def make_urscript_transform(
        self,
        *,
        macro_steps: int = 16,
        settle_steps: int = 0,
        execute: bool = True,
        multi_action_dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
        action_key: NestedKey = "action",
        primitive_id_key: NestedKey = "primitive_id",
        target_qpos_key: NestedKey = "target_qpos",
        target_pose_key: NestedKey = "target_pose",
        gripper_key: NestedKey = "gripper",
        robot_qpos_key: NestedKey = "robot_qpos",
        gripper_qpos_key: NestedKey = "gripper_qpos",
        ik_kwargs: dict[str, float | int] | None = None,
    ) -> Transform:
        """Create a URScript-style primitive transform for this environment.

        The returned transform uses this environment's low-level gripper command
        range and Cartesian IK helper, so tutorials do not need to write custom
        solver closures.

        Args:
            macro_steps: number of interpolated low-level actions per primitive.
            settle_steps: number of repeated final actions appended per
                primitive.
            execute: if ``True`` (default), append a ``MultiAction`` executor
                around the primitive expansion so a high-level primitive can be
                passed directly to :meth:`step`. Pass ``False`` to inspect or
                manually execute the expanded action sequence.
            multi_action_dim: stack dimension consumed by ``MultiAction`` when
                ``execute=True``.
            stack_rewards: whether ``MultiAction`` should return each low-level
                reward when ``execute=True``.
            stack_observations: whether ``MultiAction`` should return each
                low-level observation when ``execute=True``.
            action_key: low-level action key consumed by the environment.
            primitive_id_key: primitive id key.
            target_qpos_key: joint target key for ``movej``.
            target_pose_key: Cartesian pose target key for ``movel``.
            gripper_key: optional gripper command key.
            robot_qpos_key: observation key for robot joints.
            gripper_qpos_key: observation key for gripper joints.
            ik_kwargs: optional keyword arguments forwarded to the MuJoCo DLS
                IK helper used by ``movel``.

        Examples:
            >>> from torchrl.envs import CubeBowlEnv  # doctest: +SKIP
            >>> from torchrl.envs import RobotMacroAction  # doctest: +SKIP
            >>> env = CubeBowlEnv()  # doctest: +SKIP
            >>> env = env.append_transform(env.make_urscript_transform(macro_steps=4))  # doctest: +SKIP
            >>> td = env.reset()  # doctest: +SKIP
            >>> offset = td["cube_pos"].new_tensor([[0.0, 0.0, 0.1]])  # doctest: +SKIP
            >>> td["action"] = RobotMacroAction.reach_pose(  # doctest: +SKIP
            ...     position=td["cube_pos"] + offset,
            ...     quaternion=td["pinch_quat"],
            ...     gripper="open",
            ... )
            >>> env.step(td).get(("next", "reward")).shape  # doctest: +SKIP
            torch.Size([1, 1])
        """
        if ik_kwargs is None:
            ik_kwargs = {}
        else:
            ik_kwargs = dict(ik_kwargs)

        def cartesian_solver(
            target_pose: torch.Tensor, start_action: torch.Tensor
        ) -> torch.Tensor:
            return self._cartesian_pose_to_joint_target(
                target_pose, start_action, **ik_kwargs
            )

        return URScriptPrimitiveTransform(
            macro_steps=macro_steps,
            settle_steps=settle_steps,
            execute=execute,
            multi_action_dim=multi_action_dim,
            stack_rewards=stack_rewards,
            stack_observations=stack_observations,
            action_key=action_key,
            primitive_id_key=primitive_id_key,
            target_qpos_key=target_qpos_key,
            target_pose_key=target_pose_key,
            gripper_key=gripper_key,
            robot_qpos_key=robot_qpos_key,
            gripper_qpos_key=gripper_qpos_key,
            cartesian_solver=cartesian_solver,
            open_gripper_ctrl=self.gripper_open_ctrl,
            close_gripper_ctrl=self.gripper_close_ctrl,
        )

    # ------------------------------------------------------------------
    # Privileged geometry helpers for scripted primitives.
    # ------------------------------------------------------------------

    def _target_pos(self) -> torch.Tensor:
        if self._bowl_target_site_id is None or not hasattr(self._backend, "_d"):
            return self._bowl_target_pos
        pos = torch.as_tensor(
            self._backend._d.site_xpos[self._bowl_target_site_id].copy(),
            dtype=self.dtype,
            device=self.device,
        )
        return pos.view(1, 3)

    def _set_bowl_body_position(self, bowl_pos: torch.Tensor) -> None:
        if bowl_pos.shape[0] != 1:
            raise RuntimeError("Per-reset `bowl_pos` overrides require `num_envs=1`.")
        offset = torch.tensor(
            self._bowl_target_offset, dtype=self.dtype, device=self.device
        )
        bowl_pos = bowl_pos.reshape(1, 3).to(dtype=self.dtype, device=self.device)
        self._bowl_target_pos = bowl_pos + offset.view(1, 3)
        model = getattr(self._backend, "_m", getattr(self._backend, "_m_mj", None))
        if model is None or self._bowl_body_id is None:
            return
        model.body_pos[self._bowl_body_id] = bowl_pos[0].detach().cpu().numpy()

    def _pinch_pos(self) -> torch.Tensor:
        if self._pinch_site_id is None or not hasattr(self._backend, "_d"):
            return torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        pos = torch.as_tensor(
            self._backend._d.site_xpos[self._pinch_site_id].copy(),
            dtype=self.dtype,
            device=self.device,
        )
        return pos.unsqueeze(0).expand(self.num_envs, 3).clone()

    def _pinch_quat(self) -> torch.Tensor:
        quat = torch.zeros(self.num_envs, 4, dtype=self.dtype, device=self.device)
        quat[..., 0] = 1.0
        if (
            not _has_mujoco
            or self._pinch_site_id is None
            or not hasattr(self._backend, "_d")
        ):
            return quat
        import mujoco

        quat_np = np.zeros(4)
        mujoco.mju_mat2Quat(
            quat_np, self._backend._d.site_xmat[self._pinch_site_id].copy()
        )
        quat_single = torch.as_tensor(quat_np, dtype=self.dtype, device=self.device)
        return quat_single.unsqueeze(0).expand(self.num_envs, 4).clone()

    def _pad_pos(
        self, geom_ids: tuple[int, ...], *, fallback: torch.Tensor
    ) -> torch.Tensor:
        if not geom_ids or not hasattr(self._backend, "_d"):
            return fallback.clone()
        pos = torch.as_tensor(
            self._backend._d.geom_xpos[list(geom_ids)].copy(),
            dtype=self.dtype,
            device=self.device,
        ).mean(0)
        return pos.unsqueeze(0).expand(self.num_envs, 3).clone()

    def _find_site_id(self, name: str) -> int | None:
        if not _has_mujoco or not hasattr(self._backend, "_m"):
            return None
        import mujoco

        site_id = mujoco.mj_name2id(self._backend._m, mujoco.mjtObj.mjOBJ_SITE, name)
        return None if site_id < 0 else int(site_id)

    def _find_body_id(self, name: str) -> int | None:
        if not _has_mujoco or not hasattr(self._backend, "_m"):
            return None
        import mujoco

        body_id = mujoco.mj_name2id(self._backend._m, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if body_id < 0 else int(body_id)

    def _find_geom_ids(self, names: tuple[str, ...]) -> tuple[int, ...]:
        if not _has_mujoco or not hasattr(self._backend, "_m"):
            return ()
        import mujoco

        ids = []
        for name in names:
            geom_id = mujoco.mj_name2id(
                self._backend._m, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if geom_id >= 0:
                ids.append(int(geom_id))
        return tuple(ids)

    def _cartesian_pose_to_joint_target(
        self,
        target_pose: torch.Tensor,
        start_action: torch.Tensor | None = None,
        *,
        iterations: int = 64,
        damping: float = 1e-4,
        step_size: float = 0.7,
        orientation_weight: float = 0.0,
    ) -> torch.Tensor:
        """Best-effort MuJoCo damped-least-squares IK for ``movel``.

        The solver optimizes the ``pinch`` site over the first six arm joints.
        If the active backend is not the official MuJoCo C backend, the input
        action is returned unchanged.
        """
        if start_action is None:
            start_action = torch.zeros(
                target_pose.shape[:-1] + (7,),
                dtype=self.dtype,
                device=target_pose.device,
            )
        if (
            not _has_mujoco
            or self._pinch_site_id is None
            or not hasattr(self._backend, "_m")
        ):
            return start_action
        if target_pose.shape[0] != 1:
            return start_action
        import mujoco

        model = self._backend._m
        data = mujoco.MjData(model)
        data.qpos[:] = self._backend._d.qpos.copy()
        data.qvel[:] = 0.0
        q = data.qpos.copy()
        if start_action.shape[-1] >= self.ROBOT_QPOS_DIM:
            q[list(self._robot_qpos_indices)] = (
                start_action[0, : self.ROBOT_QPOS_DIM].detach().cpu().double().numpy()
            )
        ctrl_low = model.actuator_ctrlrange[: self.ROBOT_QPOS_DIM, 0]
        ctrl_high = model.actuator_ctrlrange[: self.ROBOT_QPOS_DIM, 1]
        target = target_pose[0, :3].detach().cpu().double().numpy()
        target_quat = None
        target_mat = None
        if target_pose.shape[-1] >= 7 and orientation_weight > 0.0:
            quat = target_pose[0, 3:7].detach().cpu().double().numpy()
            norm = float(np.linalg.norm(quat))
            if norm > 1e-6:
                target_quat = quat / norm
                target_mat = np.zeros(9)
                mujoco.mju_quat2Mat(target_mat, target_quat)
                target_mat = target_mat.reshape(3, 3)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        err_dim = 6 if target_quat is not None else 3
        eye = np.eye(err_dim)
        for _ in range(iterations):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            pos_err = target - data.site_xpos[self._pinch_site_id]
            mujoco.mj_jacSite(model, data, jacp, jacr, self._pinch_site_id)
            if target_quat is None:
                err = pos_err
                jac = jacp[:, list(self._robot_qvel_indices)]
            else:
                rot_err = self._rotation_error(
                    target_mat, data.site_xmat[self._pinch_site_id].reshape(3, 3)
                )
                err = np.concatenate([pos_err, orientation_weight * rot_err])
                jac = np.concatenate(
                    [
                        jacp[:, list(self._robot_qvel_indices)],
                        orientation_weight * jacr[:, list(self._robot_qvel_indices)],
                    ],
                    axis=0,
                )
            if float(np.linalg.norm(err)) < 1e-4:
                break
            lhs = jac @ jac.T + damping * eye
            dq = jac.T @ np.linalg.solve(lhs, err)
            q[list(self._robot_qpos_indices)] += step_size * dq
            q[list(self._robot_qpos_indices)] = self._wrap_ctrl_range(
                q[list(self._robot_qpos_indices)], ctrl_low, ctrl_high
            )
        out = start_action.clone()
        out[0, : self.ROBOT_QPOS_DIM] = torch.as_tensor(
            q[list(self._robot_qpos_indices)], dtype=out.dtype, device=out.device
        )
        return out

    @staticmethod
    def _wrap_ctrl_range(
        qpos: np.ndarray, ctrl_low: np.ndarray, ctrl_high: np.ndarray
    ) -> np.ndarray:
        qpos = qpos.copy()
        period = 2.0 * np.pi
        for index, value in enumerate(qpos):
            low = ctrl_low[index]
            high = ctrl_high[index]
            if high - low >= period - 1e-4:
                while value < low:
                    value += period
                while value > high:
                    value -= period
            qpos[index] = np.clip(value, low, high)
        return qpos

    @staticmethod
    def _rotation_error(target_mat: np.ndarray, current_mat: np.ndarray) -> np.ndarray:
        rot = target_mat @ current_mat.T
        cos_angle = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(cos_angle))
        skew = np.array(
            [
                rot[2, 1] - rot[1, 2],
                rot[0, 2] - rot[2, 0],
                rot[1, 0] - rot[0, 1],
            ]
        )
        if angle < 1e-6:
            return 0.5 * skew
        sin_angle = float(np.sin(angle))
        if abs(sin_angle) < 1e-6:
            return np.zeros(3)
        return angle * skew / (2.0 * sin_angle)
