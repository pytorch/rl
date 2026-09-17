# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Any MuJoCo Menagerie robot as a TorchRL env.

`MuJoCo Menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_ is
Google DeepMind's collection of curated robot models: arms, hands, grippers,
quadrupeds, bipeds, humanoids, drones and mobile manipulators. It ships models
and scenes, not tasks. :class:`MenagerieModelSource` locates one of them by
name in a checkout or in the ``mujoco-menagerie`` package cache, and
:class:`MenagerieEnv` is :class:`~torchrl.envs.MujocoModelEnv` over that
source: it resets to the model's ``home`` keyframe, exposes the raw simulator
state and takes a :class:`~torchrl.envs.MujocoModelTask`.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.envs.custom.mujoco.model_env import MujocoModelEnv, MujocoModelTask

_has_mujoco_menagerie = importlib.util.find_spec("mujoco_menagerie") is not None

MENAGERIE_ENV_VAR = "TORCHRL_MUJOCO_MENAGERIE_PATH"

MenagerieTask = MujocoModelTask


@dataclass(frozen=True)
class MenagerieModelSource:
    """One MuJoCo Menagerie robot, located by name.

    The robot is resolved from ``menagerie_path`` (a ``mujoco_menagerie``
    checkout, the robot's directory inside one, or the XML itself), then from
    the :data:`MENAGERIE_ENV_VAR` environment variable, then from the cache of
    the ``mujoco-menagerie`` package (``pip install mujoco-menagerie``), which
    ``download=True`` lets fetch the robot. The package pins every robot to
    one Menagerie commit; a checkout is whatever revision it holds.

    Args:
        robot (str): the Menagerie model directory, for example
            ``"unitree_go2"``, ``"franka_emika_panda"`` or ``"shadow_hand"``.
        entry (str, optional): the top-level XML to load, by file stem:
            ``"scene"`` (the robot on a floor with lights), ``"scene_mjx"``
            where Menagerie provides one, or the robot alone (``"go2"``).
            ``None`` (default) loads ``scene.xml`` from a checkout and the
            registry's default scene from the package.
        menagerie_path (str or Path, optional): a ``mujoco_menagerie``
            checkout, the robot's directory inside one, or the XML itself.
            Defaults to the :data:`MENAGERIE_ENV_VAR` environment variable,
            then to the ``mujoco-menagerie`` package cache.

    Examples:
        >>> from torchrl.envs import MenagerieModelSource, MujocoModelEnv
        >>> source = MenagerieModelSource("unitree_go2", entry="scene_mjx")
        >>> source.resolve(download=True).name  # doctest: +SKIP
        'scene_mjx.xml'
        >>> env = MujocoModelEnv(source, download=True, backend="mujoco-torch", num_envs=64)  # doctest: +SKIP
    """

    robot: str
    entry: str | None = None
    menagerie_path: str | Path | None = None

    def resolve(self, *, download: bool = False) -> Path:
        """Locate the robot's XML.

        Keyword Args:
            download (bool, optional): whether the ``mujoco-menagerie`` package
                may download the robot into its cache. Only consulted when
                neither ``menagerie_path`` nor the environment variable is
                set. Defaults to ``False``.

        Returns:
            The absolute path to the XML.

        Raises:
            FileNotFoundError: if the robot cannot be located without
                downloading, or the checkout lacks the robot or the entry. The
                package raises its own errors for a name or an entry that is
                not in its registry.
        """
        robot = self.robot
        candidate = self.menagerie_path
        if candidate is None:
            candidate = os.environ.get(MENAGERIE_ENV_VAR) or None
        if candidate is not None:
            return self._resolve_in_checkout(Path(candidate).expanduser())
        if not _has_mujoco_menagerie:
            raise FileNotFoundError(
                f"MenagerieModelSource could not locate {robot!r}. Pass "
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
        entry_point = spec.entry(self.entry)
        try:
            robot_dir = spec.path(cache)
        except mujoco_menagerie.MenagerieError as err:
            raise FileNotFoundError(
                f"{robot!r} could not be fetched from the mujoco-menagerie "
                f"package: {err}. Point menagerie_path or {MENAGERIE_ENV_VAR} at "
                "a mujoco_menagerie checkout instead."
            ) from err
        return (Path(robot_dir) / entry_point.file).resolve()

    def _resolve_in_checkout(self, path: Path) -> Path:
        robot = self.robot
        if path.is_file():
            if self.entry is not None and path.stem != self.entry:
                raise ValueError(
                    f"menagerie_path points at {path.name} but entry="
                    f"{self.entry!r} was requested; pass the robot directory or "
                    "drop entry."
                )
            return path.resolve()
        if (path / robot).is_dir():
            robot_dir = path / robot
        elif path.is_dir() and path.name == robot:
            robot_dir = path
        else:
            raise FileNotFoundError(
                f"MenagerieModelSource: no {robot!r} model directory under {path}."
            )
        xml = robot_dir / f"{'scene' if self.entry is None else self.entry}.xml"
        if not xml.is_file():
            entries = sorted(candidate.stem for candidate in robot_dir.glob("*.xml"))
            raise FileNotFoundError(
                f"MenagerieModelSource: {robot!r} has no entry {xml.stem!r} under "
                f"{robot_dir}; available entries: {entries}."
            )
        return xml.resolve()


class MenagerieEnv(MujocoModelEnv):
    r"""A MuJoCo Menagerie robot, loaded by name.

    :class:`~torchrl.envs.MujocoModelEnv` over a :class:`MenagerieModelSource`:
    the action is the model's actuator control vector, the observation the raw
    state (``qpos``, ``qvel``, ``sensordata`` when the model defines sensors,
    ``site_positions`` for the sites named in the task), a reset starts from
    the ``home`` keyframe, and the reward is the weighted sum of the
    :class:`~torchrl.envs.MujocoModelTask` terms, all off by default, so a
    :class:`~torchrl.envs.Transform` can supply the reward of a task of your
    own. ``examples/menagerie/ppo.py`` trains PPO to hold the pose of any
    robot or to walk a quadruped, and plays the policy back with ``rlrender``.

    The model is resolved from ``menagerie_path``, then from the
    :data:`MENAGERIE_ENV_VAR` environment variable, then from the cache of the
    ``mujoco-menagerie`` package, which ``download=True`` lets fetch the robot;
    see :class:`MenagerieModelSource`. The resolved XML is :attr:`model_path`,
    next to :attr:`robot`, :attr:`entry` and :attr:`task`; :attr:`reset_state`
    is the keyframe the reset draws around.

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
            may download the robot into its cache. Only consulted when neither
            ``menagerie_path`` nor the environment variable is set: a checkout
            that lacks the robot raises instead of falling back to the
            package. Defaults to ``False``, in which case a robot missing from
            the cache raises ``FileNotFoundError`` describing every option.
        task (MujocoModelTask, optional): the reset keyframe, the observed
            sites, the termination height and the reward weights. Defaults to
            ``MujocoModelTask()``: the ``home`` keyframe, no sites, no
            termination on height and a zero reward. See
            :meth:`~torchrl.envs.MujocoModelEnv.hold_pose_task`.
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

    def __init__(
        self,
        robot: str,
        *,
        entry: str | None = None,
        menagerie_path: str | Path | None = None,
        download: bool = False,
        task: MujocoModelTask | None = None,
        backend: BackendName = "mujoco",
        max_episode_steps: int = 1000,
        **kwargs: Any,
    ):
        self.robot = str(robot)
        self.entry = entry
        super().__init__(
            MenagerieModelSource(robot, entry=entry, menagerie_path=menagerie_path),
            download=download,
            task=task,
            backend=backend,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

    @classmethod
    def _resolve_before_batching(
        cls,
        robot: str,
        *,
        entry: str | None = None,
        menagerie_path: str | Path | None = None,
        download: bool = False,
        **kwargs: Any,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        source = MenagerieModelSource(robot, entry=entry, menagerie_path=menagerie_path)
        (xml,), kwargs = super()._resolve_before_batching(
            source, download=download, **kwargs
        )
        return (robot,), {**kwargs, "entry": entry, "menagerie_path": xml}

    @property
    def _model_name(self) -> str:
        return f"{self.robot} ({Path(str(self.model_path)).name})"

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

        The arguments are those of :class:`MenagerieModelSource`, whose
        :meth:`~MenagerieModelSource.resolve` this calls.
        """
        source = MenagerieModelSource(robot, entry=entry, menagerie_path=menagerie_path)
        return source.resolve(download=download)
