# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Any MuJoCo Menagerie robot as a TorchRL env.

`MuJoCo Menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_ is
Google DeepMind's collection of curated robot models: arms, hands, grippers,
quadrupeds, bipeds, humanoids, drones and mobile manipulators. It ships models
and scenes, not tasks. :class:`MenagerieEnv` loads any of them by name on the
:class:`~torchrl.envs.MujocoEnv` physics backends, resets to the model's
``home`` keyframe and exposes the raw simulator state; :class:`MenagerieTask`
holds the few task parameters that make sense for every robot, and a
:class:`~torchrl.envs.Transform` supplies the reward of a task of your own.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.envs.custom.mujoco.base import _MujocoMeta, MujocoEnv

_has_mujoco_menagerie = importlib.util.find_spec("mujoco_menagerie") is not None

MENAGERIE_ENV_VAR = "TORCHRL_MUJOCO_MENAGERIE_PATH"


@dataclass(frozen=True)
class MenagerieTask:
    """Task parameters of :class:`MenagerieEnv`.

    Menagerie ships robots, not tasks, so the defaults describe the bare
    simulator: reset around the model's ``home`` keyframe, observe the state,
    never terminate before the horizon and pay no reward. Non-zero weights turn
    the built-in reward terms on; leave them at zero to let a
    :class:`~torchrl.envs.Transform` write ``("next", "reward")`` instead.
    :meth:`MenagerieEnv.hold_pose_task` is the preset that turns them on.

    Args:
        keyframe (str, optional): name of the MJCF keyframe whose ``qpos`` and
            ``qvel`` the reset state is drawn around. ``None`` (default) uses
            the ``home`` keyframe when the model defines one and the model's
            ``qpos0`` at rest otherwise; a name the model does not define
            raises ``KeyError`` at construction.
        site_names (Sequence[str], optional): MuJoCo sites whose world
            positions are exposed as the ``site_positions`` observation,
            shaped ``(num_envs, len(site_names), 3)`` in this order. Empty
            (default) omits the entry.
        terminate_below_height (float, optional): if set, the episode
            terminates once the height (world ``z``) of the floating base drops
            below this value, in meters. Requires a free joint. ``None``
            (default) never terminates on height.
        pose_weight (float, optional): weight of the pose term,
            ``exp(-mean((q - q_key)^2) / pose_std^2)`` over the hinge and
            slide joints, where ``q_key`` is the reset keyframe. ``0.0``
            (default) turns the term off.
        pose_std (float, optional): scale of the pose term, in the joints'
            units. Defaults to ``0.5``.
        control_cost_weight (float, optional): weight of the control cost,
            minus the mean squared action after mapping each actuator's
            control range onto ``[-1, 1]``. ``0.0`` (default) turns the term
            off.
        alive_bonus (float, optional): constant paid at every step that does
            not terminate on height. Defaults to ``0.0``.

    Examples:
        >>> from dataclasses import replace
        >>> from torchrl.envs import MenagerieEnv, MenagerieTask
        >>> task = MenagerieTask(site_names=("imu",), terminate_below_height=0.15)
        >>> task.pose_weight, task.site_names
        (0.0, ('imu',))
        >>> standing = replace(MenagerieEnv.hold_pose_task(), alive_bonus=0.5)
        >>> standing.pose_weight, standing.alive_bonus
        (1.0, 0.5)
        >>> env = MenagerieEnv("unitree_go2", download=True, task=standing)  # doctest: +SKIP
    """

    keyframe: str | None = None
    site_names: Sequence[str] = ()
    terminate_below_height: float | None = None
    pose_weight: float = 0.0
    pose_std: float = 0.5
    control_cost_weight: float = 0.0
    alive_bonus: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.site_names, str):
            raise TypeError("site_names must be a sequence of site names, not a str.")
        object.__setattr__(self, "site_names", tuple(self.site_names))
        if not self.pose_std > 0:
            raise ValueError(f"pose_std must be positive, got {self.pose_std}.")


class _MenagerieMeta(_MujocoMeta):
    """Resolve, and if requested download, the model once before batching.

    :class:`~torchrl.envs.custom.mujoco.base._MujocoMeta` builds one env per
    worker for the native backend; resolving the XML here hands the workers a
    local path, so they never download concurrently.
    """

    def __call__(
        cls,
        robot: str,
        *args: Any,
        entry: str | None = None,
        menagerie_path: str | Path | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        xml = cls.resolve_model(
            robot, entry=entry, menagerie_path=menagerie_path, download=download
        )
        return super().__call__(
            robot, *args, entry=entry, menagerie_path=xml, download=False, **kwargs
        )


class MenagerieEnv(MujocoEnv, metaclass=_MenagerieMeta):
    r"""A MuJoCo Menagerie robot, loaded by name.

    Menagerie ships robots, not tasks, so this env is the bare simulator of
    one of its models. The action is the model's actuator control vector
    (position targets for most arms and hands, torques for most legged robots,
    in the units and ranges of the MJCF). The observation is the raw state:
    ``qpos``, ``qvel``, the model's ``sensordata`` when it defines sensors,
    and the world positions of the sites named in the task under
    ``site_positions``. A reset starts from the model's ``home`` keyframe plus
    ``reset_noise_scale`` uniform noise. The episode ends at
    ``max_episode_steps``, when the state stops being finite or, if the task
    asks for it, when a floating base drops below a height. The reward is the
    weighted sum of the :class:`MenagerieTask` terms, all off by default; a
    task of your own is a :class:`~torchrl.envs.Transform` that writes
    ``("next", "reward")`` from the observation.

    The model is resolved from ``menagerie_path`` (a ``mujoco_menagerie``
    checkout, the robot's directory inside one, or the XML itself), then from
    the :data:`MENAGERIE_ENV_VAR` environment variable, then from the cache of
    the ``mujoco-menagerie`` package (``pip install mujoco-menagerie``), which
    ``download=True`` lets fetch the robot. The package pins every robot to
    one Menagerie commit; a checkout is whatever revision it holds. The
    resolved XML is :attr:`model_path`, next to :attr:`robot`, :attr:`entry`
    and :attr:`task`. ``examples/menagerie/ppo_hold_pose.py`` trains a PPO
    hold-pose policy on any robot and plays it back with ``rlrender``.

    Menagerie's ``scene`` entry points put the robot on a floor with lights
    and are meant for the ``"mujoco"`` backend, the default here. The models
    Menagerie maintains for MJX (``scene_mjx``, ``mjx_scene`` and the like,
    with primitive collision geoms) also run on the ``"mujoco-torch"`` and
    ``"mjx"`` backends, which vectorize ``num_envs`` simulators on an
    accelerator; the other scenes use collision pairs those engines do not
    implement. Those engines may also evaluate acceleration-stage sensors
    (accelerometers, force and torque sensors) differently from MuJoCo.

    Args:
        robot (str): the Menagerie model directory, for example
            ``"unitree_go2"``, ``"franka_emika_panda"`` or ``"shadow_hand"``.

    Keyword Args:
        entry (str, optional): the top-level XML to load, by file stem:
            ``"scene"`` (the robot on a floor with lights), ``"scene_mjx"``
            where Menagerie provides one, or the robot alone (``"go2"``).
            ``None`` (default) loads ``scene.xml`` from a checkout and the
            registry's default scene from the package.
        menagerie_path (str or Path, optional): a ``mujoco_menagerie``
            checkout, the robot's directory inside one, or the XML itself.
            Defaults to the :data:`MENAGERIE_ENV_VAR` environment variable,
            then to the ``mujoco-menagerie`` package cache.
        download (bool, optional): whether the ``mujoco-menagerie`` package
            may download the robot into its cache when no other source
            resolves. Defaults to ``False``, in which case a missing robot
            raises ``FileNotFoundError`` describing every option.
        task (MenagerieTask, optional): the reset keyframe, the observed
            sites, the termination height and the reward weights. Defaults to
            ``MenagerieTask()``: the ``home`` keyframe, no sites, no
            termination on height and a zero reward. See
            :meth:`hold_pose_task`.
        backend (str, optional): ``"mujoco"`` (default) runs the official C
            bindings, one simulator per worker process with
            :class:`~torchrl.envs.ParallelEnv` when ``num_envs > 1`` (or in
            one process with :class:`~torchrl.envs.SerialEnv` when
            ``parallel=False``). ``"mujoco-torch"`` and ``"mjx"`` vectorize
            the ``num_envs`` simulators inside the engine and need one of the
            MJX-ready entry points.
        max_episode_steps (int, optional): truncation horizon. Defaults to
            ``1000``.
        \*\*kwargs: forwarded to :class:`~torchrl.envs.MujocoEnv`:
            ``num_envs``, ``device``, ``seed``, ``frame_skip``,
            ``reset_noise_scale``, ``dtype``, ``compile_step``,
            ``from_pixels``, ``render_width``, ``render_height``,
            ``camera_id`` and so on. ``xml_path`` and ``patch_xml`` are not
            accepted. Scenes are loaded unpatched, so ``camera_id`` indexes
            the cameras the scene defines; pass ``camera_id=-1`` for MuJoCo's
            free camera when it defines none.

    Examples:
        Load a quadruped through the package, fetching it on first use, and
        look at what the env exposes:

        >>> from torchrl.envs import MenagerieEnv, MenagerieTask
        >>> env = MenagerieEnv("unitree_go2", download=True, seed=0)  # doctest: +SKIP
        >>> td = env.reset()  # doctest: +SKIP
        >>> td["qpos"].shape, env.action_spec.shape  # doctest: +SKIP
        (torch.Size([1, 19]), torch.Size([1, 12]))
        >>> rollout = env.rollout(20)  # doctest: +SKIP

        The same robot from a local checkout, batched over worker processes:

        >>> env = MenagerieEnv(  # doctest: +SKIP
        ...     "unitree_go2",
        ...     menagerie_path="~/mujoco_menagerie",
        ...     num_envs=8,
        ...     parallel=True,
        ... )

        Its MJX scene on the vectorized torch engine, with the sensors it
        defines in the observation:

        >>> env = MenagerieEnv(  # doctest: +SKIP
        ...     "unitree_go2",
        ...     entry="scene_mjx",
        ...     backend="mujoco-torch",
        ...     num_envs=1024,
        ...     device="cuda",
        ...     compile_step=True,
        ... )
        >>> env.reset()["sensordata"].shape  # doctest: +SKIP
        torch.Size([1024, 43])

        A hold-pose task that ends the episode when the base falls:

        >>> env = MenagerieEnv(  # doctest: +SKIP
        ...     "unitree_go2",
        ...     download=True,
        ...     task=MenagerieEnv.hold_pose_task(
        ...         control_cost_weight=0.05, terminate_below_height=0.15, alive_bonus=0.5
        ...     ),
        ... )

        A task of your own: expose a site and write the reward from a
        transform, here the distance from a UR5e's flange to a target:

        >>> import torch
        >>> from torchrl.envs import Transform, TransformedEnv
        >>> class ReachReward(Transform):
        ...     def __init__(self, target):
        ...         super().__init__()
        ...         self.target = target
        ...     def _step(self, tensordict, next_tensordict):
        ...         flange = next_tensordict["site_positions"][..., 0, :]
        ...         distance = (flange - self.target).norm(dim=-1, keepdim=True)
        ...         next_tensordict["reward"] = -distance
        ...         return next_tensordict
        >>> env = TransformedEnv(  # doctest: +SKIP
        ...     MenagerieEnv(
        ...         "universal_robots_ur5e",
        ...         download=True,
        ...         task=MenagerieTask(site_names=("attachment_site",)),
        ...     ),
        ...     ReachReward(torch.tensor([0.4, 0.2, 0.5])),
        ... )
        >>> env.rollout(10)["next", "reward"].shape  # doctest: +SKIP
        torch.Size([1, 10, 1])

        Pixels from the free camera of a scene that defines none:

        >>> env = MenagerieEnv(  # doctest: +SKIP
        ...     "franka_emika_panda", download=True, from_pixels=True, camera_id=-1
        ... )
    """

    DEFAULT_BACKEND: ClassVar[BackendName] = "mujoco"

    def __init__(
        self,
        robot: str,
        *,
        entry: str | None = None,
        menagerie_path: str | Path | None = None,
        download: bool = False,
        task: MenagerieTask | None = None,
        backend: BackendName = "mujoco",
        max_episode_steps: int = 1000,
        **kwargs: Any,
    ) -> None:
        for forbidden in ("xml_path", "patch_xml"):
            if forbidden in kwargs:
                raise ValueError(
                    "MenagerieEnv loads the Menagerie model itself; pass "
                    f"robot=... and menagerie_path=... instead of {forbidden}=..."
                )
        self.robot = str(robot)
        self.entry = entry
        self.task = MenagerieTask() if task is None else task
        self.model_path = self.resolve_model(
            robot, entry=entry, menagerie_path=menagerie_path, download=download
        )
        super().__init__(
            xml_path=self.model_path,
            patch_xml=False,
            backend=backend,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Model resolution and task presets
    # ------------------------------------------------------------------

    @classmethod
    def resolve_model(
        cls,
        robot: str,
        *,
        entry: str | None = None,
        menagerie_path: str | Path | None = None,
        download: bool = False,
    ) -> Path:
        """Locate the XML of one Menagerie robot.

        Args:
            robot (str): the Menagerie model directory, for example
                ``"unitree_go2"``.

        Keyword Args:
            entry (str, optional): the top-level XML to load, by file stem.
                ``None`` (default) means ``scene`` from a checkout and the
                registry's default scene from the package.
            menagerie_path (str or Path, optional): a ``mujoco_menagerie``
                checkout, the robot's directory inside one, or the XML itself.
                Defaults to the :data:`MENAGERIE_ENV_VAR` environment variable,
                then to the ``mujoco-menagerie`` package cache.
            download (bool, optional): whether the ``mujoco-menagerie`` package
                may download the robot into its cache. Defaults to ``False``.

        Returns:
            The absolute path to the XML.

        Raises:
            FileNotFoundError: if the robot cannot be located without
                downloading, or the checkout lacks the robot or the entry. The
                package raises its own errors for a name or an entry that is
                not in its registry.
        """
        candidate = menagerie_path
        if candidate is None:
            candidate = os.environ.get(MENAGERIE_ENV_VAR) or None
        if candidate is not None:
            return cls._resolve_in_checkout(robot, entry, Path(candidate).expanduser())
        if not _has_mujoco_menagerie:
            raise FileNotFoundError(
                f"MenagerieEnv could not locate {robot!r}. Pass "
                "menagerie_path=<mujoco_menagerie checkout>, set the "
                f"{MENAGERIE_ENV_VAR} environment variable, or `pip install "
                "mujoco-menagerie` and pass download=True."
            )
        import mujoco_menagerie

        spec = mujoco_menagerie.get(robot)
        cache = mujoco_menagerie.Cache()
        if cache.root is None and not download and not cache.is_cached(spec):
            raise FileNotFoundError(
                f"{robot!r} is not in the mujoco-menagerie cache at {cache.dir}. "
                "Pass download=True to fetch it, or point menagerie_path or "
                f"{MENAGERIE_ENV_VAR} at a mujoco_menagerie checkout."
            )
        return Path(spec.xml(entry, cache)).resolve()

    @staticmethod
    def _resolve_in_checkout(robot: str, entry: str | None, path: Path) -> Path:
        if path.is_file():
            return path.resolve()
        if (path / robot).is_dir():
            robot_dir = path / robot
        elif path.is_dir() and path.name == robot:
            robot_dir = path
        else:
            raise FileNotFoundError(
                f"MenagerieEnv: no {robot!r} model directory under {path}."
            )
        xml = robot_dir / f"{'scene' if entry is None else entry}.xml"
        if not xml.is_file():
            entries = sorted(candidate.stem for candidate in robot_dir.glob("*.xml"))
            raise FileNotFoundError(
                f"MenagerieEnv: {robot!r} has no entry {xml.stem!r} under "
                f"{robot_dir}; available entries: {entries}."
            )
        return xml.resolve()

    @classmethod
    def hold_pose_task(
        cls,
        *,
        pose_weight: float = 1.0,
        pose_std: float = 0.5,
        control_cost_weight: float = 0.01,
        **overrides: Any,
    ) -> MenagerieTask:
        """Hold the reset keyframe pose under a control cost.

        The usual first task for a new robot: the reward is the pose term
        times ``pose_weight`` plus the control cost times
        ``control_cost_weight``. ``overrides`` set the other
        :class:`MenagerieTask` fields, typically ``terminate_below_height``
        and ``alive_bonus`` for a floating-base robot.
        """
        return MenagerieTask(
            pose_weight=pose_weight,
            pose_std=pose_std,
            control_cost_weight=control_cost_weight,
            **overrides,
        )

    # ------------------------------------------------------------------
    # Specs and observations
    # ------------------------------------------------------------------

    def _make_specs(self) -> None:
        self._configure_from_model()
        super()._make_specs()

    def _configure_from_model(self) -> None:
        import mujoco

        model = self._backend.mj_model
        task = self.task
        qpos0 = self._backend.qpos0
        qvel0 = self._backend.qvel0
        key_id = self._keyframe_id(model, task.keyframe)
        if key_id is None:
            self._reset_qpos = qpos0.clone()
            self._reset_qvel = torch.zeros_like(qvel0)
        else:
            self._reset_qpos = torch.as_tensor(
                model.key_qpos[key_id], dtype=qpos0.dtype, device=qpos0.device
            )
            self._reset_qvel = torch.as_tensor(
                model.key_qvel[key_id], dtype=qvel0.dtype, device=qvel0.device
            )
        joint_qpos_index = []
        free_joint_qpos_adr = None
        for joint_id in range(model.njnt):
            joint_type = model.jnt_type[joint_id]
            qpos_adr = int(model.jnt_qposadr[joint_id])
            if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                joint_qpos_index.append(qpos_adr)
            elif (
                joint_type == mujoco.mjtJoint.mjJNT_FREE and free_joint_qpos_adr is None
            ):
                free_joint_qpos_adr = qpos_adr
        self._joint_qpos_index = torch.tensor(
            joint_qpos_index, dtype=torch.long, device=self.device
        )
        self._free_joint_qpos_adr = free_joint_qpos_adr
        if task.terminate_below_height is not None and free_joint_qpos_adr is None:
            raise ValueError(
                "terminate_below_height needs a floating base, but the "
                f"{self.robot!r} model has no free joint."
            )
        self._site_ids = self._mujoco_ids("site", task.site_names)
        self._nsensordata = int(model.nsensordata)

    def _keyframe_id(self, model: Any, keyframe: str | None) -> int | None:
        import mujoco

        name = "home" if keyframe is None else keyframe
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            return int(key_id)
        if keyframe is None:
            return None
        keyframes = [model.key(index).name for index in range(model.nkey)]
        raise KeyError(
            f"The {self.robot!r} model has no keyframe {keyframe!r}; it defines "
            f"{keyframes}. Pass keyframe=None to reset to qpos0."
        )

    def _sample_initial_state(
        self,
        n: int,
        tensordict: TensorDictBase | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos, qvel = super()._sample_initial_state(n, tensordict)
        backend = self._backend
        return (
            qpos + (self._reset_qpos - backend.qpos0).to(qpos),
            qvel + (self._reset_qvel - backend.qvel0).to(qvel),
        )

    def _make_obs_spec(self) -> Composite:
        backend = self._backend
        spec = Composite(
            qpos=Unbounded(
                shape=(self.num_envs, backend.nq), dtype=self.dtype, device=self.device
            ),
            qvel=Unbounded(
                shape=(self.num_envs, backend.nv), dtype=self.dtype, device=self.device
            ),
            shape=(self.num_envs,),
            device=self.device,
        )
        if self._nsensordata:
            spec["sensordata"] = Unbounded(
                shape=(self.num_envs, self._nsensordata),
                dtype=self.dtype,
                device=self.device,
            )
        if self._site_ids:
            spec["site_positions"] = Unbounded(
                shape=(self.num_envs, len(self._site_ids), 3),
                dtype=self.dtype,
                device=self.device,
            )
        return spec

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        out = {} if self.pixels_only else self._make_obs_split(state)
        if self.from_pixels:
            out["pixels"] = self._render_pixels()
        return out

    def _make_obs_split(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        out = {
            "qpos": state["qpos"].to(self.dtype).clone(),
            "qvel": state["qvel"].to(self.dtype).clone(),
        }
        if self._nsensordata:
            out["sensordata"] = self._backend.sensordata.to(self.dtype).clone()
        if self._site_ids:
            out["site_positions"] = (
                self._backend.site_positions(self._site_ids).to(self.dtype).clone()
            )
        return out

    # ------------------------------------------------------------------
    # Reward and termination
    # ------------------------------------------------------------------

    def _fallen(self, qpos: torch.Tensor) -> torch.Tensor:
        height_limit = self.task.terminate_below_height
        if height_limit is None:
            return torch.zeros(self.num_envs, 1, dtype=torch.bool, device=qpos.device)
        adr = self._free_joint_qpos_adr
        return qpos[:, adr + 2 : adr + 3] < height_limit

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        task = self.task
        qpos = next_state["qpos"].to(self.dtype)
        reward = torch.zeros(self.num_envs, 1, dtype=self.dtype, device=qpos.device)
        if task.pose_weight and self._joint_qpos_index.numel():
            target = self._reset_qpos.to(qpos)[self._joint_qpos_index]
            error = (
                (qpos[:, self._joint_qpos_index] - target)
                .square()
                .mean(dim=-1, keepdim=True)
            )
            reward = reward + task.pose_weight * torch.exp(-error / task.pose_std**2)
        if task.control_cost_weight:
            low = self.action_spec.low
            high = self.action_spec.high
            normalized = (action - (high + low) / 2) / ((high - low) / 2)
            reward = reward - task.control_cost_weight * normalized.square().mean(
                dim=-1, keepdim=True
            )
        if task.alive_bonus:
            reward = reward + task.alive_bonus * (~self._fallen(qpos)).to(self.dtype)
        return reward

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        qpos = next_state["qpos"]
        qvel = next_state["qvel"]
        finite = torch.isfinite(qpos).all(dim=-1, keepdim=True) & torch.isfinite(
            qvel
        ).all(dim=-1, keepdim=True)
        return ~finite | self._fallen(qpos.to(self.dtype))
