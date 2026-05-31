# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Ball-to-bowl manipulation task for MuJoCo macro-control examples."""

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
from torchrl.data.tensor_specs import Binary, Composite, Unbounded
from torchrl.envs.custom.mujoco._backends import resolve_xml_string
from torchrl.envs.custom.mujoco.base import MujocoEnv

_has_mujoco = importlib.util.find_spec("mujoco") is not None


class BallBowlEnv(MujocoEnv):
    r"""UR-style ball-to-bowl manipulation scene.

    ``BallBowlEnv`` is a compact MuJoCo task meant for tutorials on custom
    MuJoCo environments, action sequences, and scripted robot primitives. The
    default bundled MJCF uses only primitive geoms: a six-joint arm, a simple
    two-finger gripper, a free ball, and a static bowl. For demos, the
    environment can also compose the MuJoCo Menagerie UR5e arm and Robotiq
    2F-85 gripper from a local Menagerie checkout, without vendoring their mesh
    assets in TorchRL. The low-level action is a 7D position command: six arm
    joint targets followed by one gripper command.

    Args:
        robot_model: ``"primitive"`` for the bundled low-footprint model or
            ``"menagerie_ur5e"`` to compose a local MuJoCo Menagerie UR5e +
            Robotiq 2F-85 scene.
        menagerie_path: optional path to a local ``mujoco_menagerie`` checkout.
            When ``robot_model="menagerie_ur5e"`` and this is omitted, the
            ``TORCHRL_MUJOCO_MENAGERIE_PATH`` environment variable is used.
        ball_position: initial xyz position of the ball center.
        bowl_position: xyz position of the bowl body. The task target is the
            ``bowl_target`` site inside that body.
        placement_tolerance: success radius, in meters, around the bowl target.
        terminate_on_success: if ``True``, terminate when the ball reaches the
            bowl target.
        backend: physics backend. Defaults to ``"mujoco"`` because the tutorial
            task uses contacts and MuJoCo's site Jacobians for scripted IK.
        \*\*kwargs: forwarded to :class:`~torchrl.envs.MujocoEnv`.

    Examples:
        >>> from torchrl.envs import BallBowlEnv  # doctest: +SKIP
        >>> env = BallBowlEnv(max_episode_steps=50)  # doctest: +SKIP
        >>> td = env.rollout(3)  # doctest: +SKIP
    """

    DEFAULT_BACKEND = "mujoco"
    XML_PATH = "ball_bowl.xml"
    FRAME_SKIP = 5
    RESET_NOISE_SCALE = 0.0
    ROBOT_QPOS_DIM = 6
    GRIPPER_QPOS_DIM = 2
    BALL_QPOS_START = ROBOT_QPOS_DIM + GRIPPER_QPOS_DIM
    BALL_QVEL_START = ROBOT_QPOS_DIM + GRIPPER_QPOS_DIM
    PINCH_SITE_NAME = "pinch"
    BOWL_TARGET_SITE_NAME = "bowl_target"
    DEFAULT_BALL_POSITION = (0.34, -0.14, 0.035)
    DEFAULT_BOWL_POSITION = (0.28, 0.19, 0.01)
    BOWL_TARGET_OFFSET = (0.0, 0.0, 0.04)
    MENAGERIE_ENV_VAR = "TORCHRL_MUJOCO_MENAGERIE_PATH"
    MENAGERIE_BALL_POSITION = (0.45, -0.18, 0.035)
    MENAGERIE_BOWL_POSITION = (0.45, 0.2, 0.01)
    MENAGERIE_BOWL_TARGET_OFFSET = (0.0, 0.0, 0.05)
    MENAGERIE_GRIPPER_QPOS_DIM = 8
    MENAGERIE_PINCH_SITE_NAME = "gripper/pinch"
    MENAGERIE_HOME_QPOS = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)

    def __init__(
        self,
        *,
        robot_model: Literal["primitive", "menagerie_ur5e"] = "primitive",
        menagerie_path: str | Path | None = None,
        ball_position: tuple[float, float, float] | None = None,
        bowl_position: tuple[float, float, float] | None = None,
        placement_tolerance: float = 0.06,
        terminate_on_success: bool = False,
        backend: Literal["mujoco-torch", "mjx", "mujoco"] = "mujoco",
        **kwargs,
    ) -> None:
        self.robot_model = robot_model
        if robot_model == "primitive":
            default_ball_position = self.DEFAULT_BALL_POSITION
            default_bowl_position = self.DEFAULT_BOWL_POSITION
            self._gripper_qpos_dim = self.GRIPPER_QPOS_DIM
            self._pinch_site_name = self.PINCH_SITE_NAME
            self._bowl_target_offset = self.BOWL_TARGET_OFFSET
            self._robot_home_qpos = None
            self._menagerie_path = None
        elif robot_model == "menagerie_ur5e":
            default_ball_position = self.MENAGERIE_BALL_POSITION
            default_bowl_position = self.MENAGERIE_BOWL_POSITION
            self._gripper_qpos_dim = self.MENAGERIE_GRIPPER_QPOS_DIM
            self._pinch_site_name = self.MENAGERIE_PINCH_SITE_NAME
            self._bowl_target_offset = self.MENAGERIE_BOWL_TARGET_OFFSET
            self._robot_home_qpos = self.MENAGERIE_HOME_QPOS
            self._menagerie_path = self._resolve_menagerie_path(menagerie_path)
        else:
            raise ValueError(
                "robot_model must be one of 'primitive' or 'menagerie_ur5e', "
                f"got {robot_model!r}."
            )
        if ball_position is None:
            ball_position = default_ball_position
        if bowl_position is None:
            bowl_position = default_bowl_position
        self.ball_position = tuple(float(v) for v in ball_position)
        self.bowl_position = tuple(float(v) for v in bowl_position)
        self.placement_tolerance = float(placement_tolerance)
        self.terminate_on_success = bool(terminate_on_success)
        self._ball_qpos_start = self.ROBOT_QPOS_DIM + self._gripper_qpos_dim
        self._ball_qvel_start = self.ROBOT_QPOS_DIM + self._gripper_qpos_dim
        self._pinch_site_id: int | None = None
        self._bowl_target_site_id: int | None = None
        super().__init__(backend=backend, **kwargs)
        self._bowl_target_pos = torch.tensor(
            tuple(
                self.bowl_position[index] + self._bowl_target_offset[index]
                for index in range(3)
            ),
            dtype=self.dtype,
            device=self.device,
        ).view(1, 3)
        self._pinch_site_id = self._find_site_id(self._pinch_site_name)
        self._bowl_target_site_id = self._find_site_id(self.BOWL_TARGET_SITE_NAME)

    # ------------------------------------------------------------------
    # XML loading and construction-time scene randomization.
    # ------------------------------------------------------------------

    def _load_xml(self, xml_path: str | Path | None) -> str:
        if self.robot_model == "menagerie_ur5e":
            if xml_path is not None:
                raise ValueError(
                    "xml_path is only supported with robot_model='primitive'. "
                    "Pass menagerie_path for robot_model='menagerie_ur5e'."
                )
            return self._make_menagerie_ur5e_xml(self._menagerie_path)
        if xml_path is None:
            path_or_url: str | Path = Path(__file__).parent / "assets" / self.XML_PATH
        else:
            path_or_url = xml_path
        try:
            xml = resolve_xml_string(path_or_url)
        except OSError as err:
            raise FileNotFoundError(
                f"{type(self).__name__}: xml_path={path_or_url!r} could not be "
                "resolved."
            ) from err
        return self._patch_scene_xml(xml)

    def _patch_scene_xml(self, xml: str) -> str:
        root = ET.fromstring(xml)
        ball = root.find(".//body[@name='ball']")
        if ball is None:
            raise ValueError("BallBowlEnv XML is missing body name='ball'.")
        ball.set("pos", self._format_vec(self.ball_position))
        bowl = root.find(".//body[@name='bowl']")
        if bowl is None:
            raise ValueError("BallBowlEnv XML is missing body name='bowl'.")
        bowl.set("pos", self._format_vec(self.bowl_position))
        return ET.tostring(root, encoding="unicode")

    @classmethod
    def _resolve_menagerie_path(cls, menagerie_path: str | Path | None) -> Path:
        if menagerie_path is None:
            menagerie_path = os.environ.get(cls.MENAGERIE_ENV_VAR)
        if menagerie_path is None:
            raise ValueError(
                "robot_model='menagerie_ur5e' requires a local MuJoCo "
                f"Menagerie checkout. Pass menagerie_path=... or set "
                f"{cls.MENAGERIE_ENV_VAR}."
            )
        path = Path(menagerie_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"MuJoCo Menagerie path does not exist: {path}."
            )
        return path

    def _make_menagerie_ur5e_xml(self, menagerie_path: Path) -> str:
        ur_dir = menagerie_path / "universal_robots_ur5e"
        gripper_dir = menagerie_path / "robotiq_2f85"
        ur_xml = ur_dir / "ur5e.xml"
        gripper_xml = gripper_dir / "2f85.xml"
        if not ur_xml.exists() or not gripper_xml.exists():
            raise FileNotFoundError(
                "robot_model='menagerie_ur5e' requires "
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
    def _merge_menagerie_gripper(
        cls, root: ET.Element, gripper: ET.Element
    ) -> None:
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
        ball = self._make_ball_body()
        worldbody.append(bowl)
        worldbody.append(ball)

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

    def _make_ball_body(self) -> ET.Element:
        ball = ET.Element(
            "body", {"name": "ball", "pos": self._format_vec(self.ball_position)}
        )
        ET.SubElement(ball, "freejoint", {"name": "ball_freejoint"})
        ET.SubElement(
            ball,
            "geom",
            {
                "name": "ball_geom",
                "type": "sphere",
                "size": "0.025",
                "mass": "0.045",
                "rgba": "0.95 0.35 0.15 1",
            },
        )
        return ball

    # ------------------------------------------------------------------
    # Specs and reset state.
    # ------------------------------------------------------------------

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
            ball_pos=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            ball_quat=Unbounded(
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
            qpos[..., : self.ROBOT_QPOS_DIM] = torch.tensor(
                self._robot_home_qpos, dtype=qpos.dtype, device=qpos.device
            )
        ball_pos = torch.tensor(
            self.ball_position, dtype=qpos.dtype, device=qpos.device
        ).expand(n, 3)
        if tensordict is not None and "ball_pos" in tensordict.keys(True, True):
            ball_pos = tensordict.get("ball_pos").to(
                device=qpos.device, dtype=qpos.dtype
            )
            ball_pos = ball_pos.reshape(n, 3)
        qpos[..., self._ball_qpos_start : self._ball_qpos_start + 3] = ball_pos
        qpos[..., self._ball_qpos_start + 3 : self._ball_qpos_start + 7] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), dtype=qpos.dtype, device=qpos.device
        )
        qvel[..., self._ball_qvel_start : self._ball_qvel_start + 6] = 0.0
        return qpos, qvel

    # ------------------------------------------------------------------
    # Observations, reward, and done.
    # ------------------------------------------------------------------

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        out = self._make_obs_split(state) if not self.pixel_only else {}
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
        ball_pos = qpos[..., self._ball_qpos_start : self._ball_qpos_start + 3]
        bowl_pos = self._target_pos().expand(self.num_envs, 3)
        return {
            "robot_qpos": qpos[..., : self.ROBOT_QPOS_DIM].clone(),
            "robot_qvel": qvel[..., : self.ROBOT_QPOS_DIM].clone(),
            "gripper_qpos": qpos[
                ..., self.ROBOT_QPOS_DIM : self._ball_qpos_start
            ].clone(),
            "gripper_qvel": qvel[
                ..., self.ROBOT_QPOS_DIM : self._ball_qvel_start
            ].clone(),
            "pinch_pos": self._pinch_pos().to(self.dtype),
            "ball_pos": ball_pos.clone(),
            "ball_quat": qpos[
                ..., self._ball_qpos_start + 3 : self._ball_qpos_start + 7
            ].clone(),
            "bowl_pos": bowl_pos.clone(),
            "success": self._success(ball_pos, bowl_pos),
        }

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state, action
        ball_pos = next_state["qpos"].to(self.dtype)[
            ..., self._ball_qpos_start : self._ball_qpos_start + 3
        ]
        target = self._target_pos().expand(self.num_envs, 3)
        dist = (ball_pos - target).norm(dim=-1, keepdim=True)
        success = self._success(ball_pos, target).to(self.dtype)
        return -dist + success

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        if not self.terminate_on_success:
            return torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        ball_pos = next_state["qpos"].to(self.dtype)[
            ..., self._ball_qpos_start : self._ball_qpos_start + 3
        ]
        return self._success(ball_pos, self._target_pos().expand(self.num_envs, 3))

    def _success(self, ball_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = (ball_pos - target).norm(dim=-1, keepdim=True)
        return dist <= self.placement_tolerance

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

    def _pinch_pos(self) -> torch.Tensor:
        if self._pinch_site_id is None or not hasattr(self._backend, "_d"):
            return torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        pos = torch.as_tensor(
            self._backend._d.site_xpos[self._pinch_site_id].copy(),
            dtype=self.dtype,
            device=self.device,
        )
        return pos.unsqueeze(0).expand(self.num_envs, 3).clone()

    def _find_site_id(self, name: str) -> int | None:
        if not _has_mujoco or not hasattr(self._backend, "_m"):
            return None
        import mujoco

        site_id = mujoco.mj_name2id(self._backend._m, mujoco.mjtObj.mjOBJ_SITE, name)
        return None if site_id < 0 else int(site_id)

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
            q[: self.ROBOT_QPOS_DIM] = (
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
                jac = jacp[:, : self.ROBOT_QPOS_DIM]
            else:
                rot_err = self._rotation_error(
                    target_mat, data.site_xmat[self._pinch_site_id].reshape(3, 3)
                )
                err = np.concatenate([pos_err, orientation_weight * rot_err])
                jac = np.concatenate(
                    [
                        jacp[:, : self.ROBOT_QPOS_DIM],
                        orientation_weight * jacr[:, : self.ROBOT_QPOS_DIM],
                    ],
                    axis=0,
                )
            if float(np.linalg.norm(err)) < 1e-4:
                break
            lhs = jac @ jac.T + damping * eye
            dq = jac.T @ np.linalg.solve(lhs, err)
            q[: self.ROBOT_QPOS_DIM] += step_size * dq
            q[: self.ROBOT_QPOS_DIM] = self._wrap_ctrl_range(
                q[: self.ROBOT_QPOS_DIM], ctrl_low, ctrl_high
            )
        out = start_action.clone()
        out[0, : self.ROBOT_QPOS_DIM] = torch.as_tensor(
            q[: self.ROBOT_QPOS_DIM], dtype=out.dtype, device=out.device
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
