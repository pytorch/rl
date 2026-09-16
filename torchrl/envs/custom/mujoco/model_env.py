# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Any MuJoCo model as a TorchRL env, from a local file or a model source.

:class:`MujocoModelEnv` is the bare simulator of a model: it resets around the
model's ``home`` keyframe, exposes the raw state and takes a
:class:`MujocoModelTask` holding the few task parameters that make sense for
every model. The model comes from a path or a
:class:`~torchrl.envs.custom.mujoco.sources.ModelSource` such as
:class:`~torchrl.envs.GitHubModelSource`; :class:`~torchrl.envs.MenagerieEnv`
wraps the curated Menagerie source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.envs.custom.mujoco.base import MujocoEnv
from torchrl.envs.custom.mujoco.sources import _resolve_model_source, ModelSource


@dataclass(frozen=True)
class MujocoModelTask:
    """Task parameters of :class:`MujocoModelEnv`.

    A model ships no task, so the defaults describe the bare simulator: reset
    around the model's ``home`` keyframe, observe the state, never terminate
    before the horizon and pay no reward. Non-zero weights turn the built-in
    reward terms on; leave them at zero to let a
    :class:`~torchrl.envs.Transform` write ``("next", "reward")`` instead.
    :meth:`MujocoModelEnv.hold_pose_task` is the preset that turns them on.

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
            below this value, in meters. The base is the first free joint in
            model order, the robot's in scenes that carry one; a scene whose
            only free joint belongs to an object (a cube on a table) would
            track that object instead. ``None`` (default) never terminates on
            height.
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
        >>> from torchrl.envs import MujocoModelEnv, MujocoModelTask
        >>> task = MujocoModelTask(site_names=("imu",), terminate_below_height=0.15)
        >>> task.pose_weight, task.site_names
        (0.0, ('imu',))
        >>> standing = replace(MujocoModelEnv.hold_pose_task(), alive_bonus=0.5)
        >>> standing.pose_weight, standing.alive_bonus
        (1.0, 0.5)
        >>> env = MujocoModelEnv("~/robots/go2/scene.xml", task=standing)  # doctest: +SKIP
    """

    keyframe: str | None = None
    site_names: Sequence[str] = ()
    terminate_below_height: float | None = None
    pose_weight: float = 0.0
    pose_std: float = 0.5
    control_cost_weight: float = 0.0
    alive_bonus: float = 0.0

    def __post_init__(self):
        if isinstance(self.site_names, str):
            raise TypeError("site_names must be a sequence of site names, not a str.")
        object.__setattr__(self, "site_names", tuple(self.site_names))
        if not self.pose_std > 0:
            raise ValueError(f"pose_std must be positive, got {self.pose_std}.")


class MujocoModelEnv(MujocoEnv):
    r"""Any MuJoCo model as an env: the bare simulator, with a small task config.

    The action is the model's actuator control vector (position targets for
    most arms and hands, torques for most legged robots, in the units and
    ranges of the MJCF). The observation is the raw state: ``qpos``, ``qvel``,
    the model's ``sensordata`` when it defines sensors, and the world
    positions of the sites named in the task under ``site_positions``. As
    after MuJoCo's own ``mj_step``, ``sensordata``, ``site_positions`` and
    contacts are computed before the last physics substep of the env step, so
    they trail ``qpos`` and ``qvel`` by one substep; a reset returns them
    consistent. A reset starts from the model's ``home`` keyframe plus
    ``reset_noise_scale`` uniform noise. The episode ends at ``max_episode_steps``, when the state
    stops being finite or, if the task asks for it, when a floating base
    drops below a height. The reward is the weighted sum of the
    :class:`MujocoModelTask` terms, all off by default; a task of your own is
    a :class:`~torchrl.envs.Transform` that writes ``("next", "reward")``
    from the observation.

    The model is a path to an XML, or a
    :class:`~torchrl.envs.custom.mujoco.sources.ModelSource` that resolves to
    one, such as :class:`~torchrl.envs.GitHubModelSource` for a repository
    pinned to a revision or :class:`~torchrl.envs.MenagerieModelSource` for
    a MuJoCo Menagerie robot (:class:`~torchrl.envs.MenagerieEnv` wraps the
    latter). ``download=True`` is the explicit permission for a source to
    fetch files. The XML is loaded unpatched, so relative includes, meshes and
    textures resolve from its directory. The resolved XML is
    :attr:`model_path`; :attr:`reset_state` is the keyframe the reset draws
    around, which position-controlled robots also use as their home targets.

    Args:
        source (ModelSource, str or Path): the model: the path or ``http(s)``
            URL of the XML itself, or a source that resolves to it.

    Keyword Args:
        download (bool, optional): whether the source may download files.
            Defaults to ``False``, in which case a source that would have to
            raises ``FileNotFoundError`` describing what to do.
        task (MujocoModelTask, optional): the reset keyframe, the observed
            sites, the termination height and the reward weights. Defaults to
            ``MujocoModelTask()``: the ``home`` keyframe, no sites, no
            termination on height and a zero reward. See
            :meth:`hold_pose_task`.
        backend (str, optional): ``"mujoco"`` (default) runs the official C
            bindings, one simulator per worker process with
            :class:`~torchrl.envs.ParallelEnv` when ``num_envs > 1`` (or in
            one process with :class:`~torchrl.envs.SerialEnv` when
            ``parallel=False``). ``"mujoco-torch"`` and ``"mjx"`` vectorize
            the ``num_envs`` simulators inside the engine and need a model
            those engines support (primitive collision geoms, no collision
            pair they do not implement).
        max_episode_steps (int, optional): truncation horizon. Defaults to
            ``1000``.
        \*\*kwargs: forwarded to :class:`~torchrl.envs.MujocoEnv`:
            ``num_envs``, ``device``, ``seed``, ``frame_skip``,
            ``reset_noise_scale``, ``dtype``, ``compile_step``,
            ``from_pixels``, ``render_width``, ``render_height``,
            ``camera_id`` and so on. ``xml_path`` and ``patch_xml`` are not
            accepted. ``camera_id`` indexes the cameras the model defines;
            pass ``camera_id=-1`` for MuJoCo's free camera when it defines
            none.

    Examples:
        A model on disk, then the same env batched over worker processes:

        >>> from torchrl.envs import MujocoModelEnv, MujocoModelTask
        >>> env = MujocoModelEnv("~/robots/go2/scene.xml", seed=0)  # doctest: +SKIP
        >>> env.reset()["qpos"].shape  # doctest: +SKIP
        torch.Size([1, 19])
        >>> env = MujocoModelEnv("~/robots/go2/scene.xml", num_envs=8, parallel=True)  # doctest: +SKIP

        A third-party robot from GitHub, pinned to a commit, with the flange
        site in the observation:

        >>> from torchrl.envs import GitHubModelSource
        >>> env = MujocoModelEnv(  # doctest: +SKIP
        ...     GitHubModelSource(
        ...         "SouthColumn76/universal_robots_ur3e",
        ...         revision="5f042ffca6b5885fd18f5448e17b71ab46274fa3",
        ...         entry="ur3e.xml",
        ...     ),
        ...     download=True,
        ...     task=MujocoModelTask(site_names=("attachment_site",)),
        ... )
        >>> env.rollout(10)["next", "site_positions"].shape  # doctest: +SKIP
        torch.Size([1, 10, 1, 3])

        A hold-pose task that ends the episode when the base falls:

        >>> env = MujocoModelEnv(  # doctest: +SKIP
        ...     "~/robots/go2/scene.xml",
        ...     task=MujocoModelEnv.hold_pose_task(
        ...         control_cost_weight=0.05, terminate_below_height=0.15, alive_bonus=0.5
        ...     ),
        ... )
    """

    DEFAULT_BACKEND: ClassVar[BackendName] = "mujoco"

    def __init__(
        self,
        source: ModelSource | str | Path,
        *,
        download: bool = False,
        task: MujocoModelTask | None = None,
        backend: BackendName = "mujoco",
        max_episode_steps: int = 1000,
        **kwargs: Any,
    ):
        self.task = MujocoModelTask() if task is None else task
        self.model_path = _resolve_model_source(source, download=download)
        super().__init__(
            xml_path=self.model_path,
            patch_xml=False,
            backend=backend,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

    @classmethod
    def _resolve_before_batching(
        cls,
        source: ModelSource | str | Path,
        *,
        download: bool = False,
        **kwargs: Any,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Resolve the source to a local XML so batched workers never download."""
        cls._reject_xml_kwargs(kwargs)
        path = _resolve_model_source(source, download=download)
        return (path,), {**kwargs, "download": False}

    @classmethod
    def _reject_xml_kwargs(cls, kwargs: dict[str, Any]) -> None:
        for forbidden in ("xml_path", "patch_xml"):
            if forbidden in kwargs:
                raise ValueError(
                    f"{cls.__name__} loads the model itself; {forbidden}=... is "
                    "not accepted."
                )

    @property
    def _model_name(self) -> str:
        return Path(str(self.model_path)).name

    # ------------------------------------------------------------------
    # Task presets
    # ------------------------------------------------------------------

    @classmethod
    def hold_pose_task(
        cls,
        *,
        pose_weight: float = 1.0,
        pose_std: float = 0.5,
        control_cost_weight: float = 0.01,
        **overrides: Any,
    ) -> MujocoModelTask:
        r"""Hold the reset keyframe pose under a control cost.

        The usual first task for a new robot: the reward is the pose term
        times ``pose_weight`` plus the control cost times
        ``control_cost_weight``.

        Keyword Args:
            pose_weight (float, optional): weight of the pose term. Defaults
                to ``1.0``.
            pose_std (float, optional): scale of the pose term. Defaults to
                ``0.5``.
            control_cost_weight (float, optional): weight of the control cost.
                Defaults to ``0.01``.
            \*\*overrides: the other :class:`MujocoModelTask` fields, typically
                ``terminate_below_height`` and ``alive_bonus`` for a
                floating-base robot.
        """
        return MujocoModelTask(
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
                "terminate_below_height needs a floating base, but "
                f"{self._model_name} has no free joint."
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
            f"{self._model_name} has no keyframe {keyframe!r}; it defines "
            f"{keyframes}. Pass keyframe=None to reset to qpos0."
        )

    @property
    def reset_state(self) -> TensorDict:
        """The ``qpos`` and ``qvel`` the reset draws around, without a batch dimension.

        The task keyframe when the model defines it, otherwise ``qpos0`` at
        rest. Both entries are cast to the env's ``dtype``.
        """
        return TensorDict(
            qpos=self._reset_qpos.to(self.dtype),
            qvel=self._reset_qvel.to(self.dtype),
            device=self.device,
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
