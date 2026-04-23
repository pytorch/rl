# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib

import torch
from tensordict import TensorDict

from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import _EnvPostInit, EnvBase
from torchrl.envs.utils import _classproperty

_has_genesis = importlib.util.find_spec("genesis") is not None

__all__ = ["GenesisEnv", "GenesisWrapper"]


def _genesis_cleanup():
    """Clean up Genesis resources to free memory.

    Call this function to force cleanup of Genesis cached kernels and free up memory.
    This is useful when creating multiple Genesis environments in a long-running process.
    """
    if not _has_genesis:
        return
    import genesis as gs

    gs.destroy()


def _as_tensor(x, *, device, dtype=None) -> torch.Tensor:
    """Coerce a Genesis output to a ``torch.Tensor`` on ``device``."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if dtype is not None and x.dtype != dtype:
        x = x.to(dtype)
    if x.device != torch.device(device):
        x = x.to(device)
    return x


class GenesisWrapper(EnvBase):
    """TorchRL wrapper around a Genesis physics scene.

    Genesis is a torch-native physics engine for general-purpose robotics
    and embodied AI. This wrapper keeps tensors on-device end-to-end: no
    numpy round-trips, no gym-style shims.

    Customization is done by subclassing and overriding the hooks below;
    ``GenesisWrapper(scene)`` works out of the box on any built scene with
    sensible defaults, so it's still useful for one-off experiments.

    Subclass hooks (all optional):

    * :meth:`_make_obs` -- read observations off ``self._scene`` and return a
      ``dict[str, Tensor]`` (or a single ``Tensor``). Default: for each entity
      with DoFs, emit ``{entity.name}_qpos`` and ``{entity.name}_qvel`` via
      :meth:`get_dofs_position` / :meth:`get_dofs_velocity`.
    * :meth:`_apply_action` -- push the agent's action into the scene before
      ``scene.step()`` is called. Default: feed ``action[..., :n_dofs]`` of
      the first actuated entity as a position target via
      :meth:`control_dofs_position`.
    * :meth:`_compute_reward` -- compute the per-step reward. Called once per
      physics substep when ``frame_skip > 1`` and the results are summed.
      Default: ``0``.
    * :meth:`_compute_done` -- compute the truncation flag. Default:
      ``self._current_step >= self._max_steps``.

    Args:
        scene (gs.Scene): a pre-built Genesis scene.
        max_steps (int, optional): truncation horizon consulted by the default
            :meth:`_compute_done`. Defaults to ``1000``.
        frame_skip (int, optional): physics steps per env step. Defaults to ``1``.
        device (torch.device, optional): torch device for returned tensors.
            Defaults to ``"cpu"``.
        batch_size (torch.Size, optional): batch size for the env. Should match
            the ``n_envs`` passed to :meth:`scene.build`. Defaults to ``()``.
        allow_done_after_reset (bool, optional): passed through to
            :class:`~torchrl.envs.EnvBase`. Defaults to ``False``.

    Examples:
        Out-of-the-box use on an arbitrary scene::

            >>> import genesis as gs
            >>> from torchrl.envs import GenesisWrapper
            >>> gs.init(backend=gs.cpu)
            >>> scene = gs.Scene(show_viewer=False)
            >>> scene.add_entity(gs.morphs.Plane())
            >>> scene.add_entity(
            ...     gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
            ... )
            >>> scene.build()
            >>> env = GenesisWrapper(scene)
            >>> td = env.rollout(10)

        Subclassing for a custom task::

            >>> class ReachEnv(GenesisWrapper):
            ...     def __init__(self, scene, target, **kwargs):
            ...         self._franka = scene.entities[1]  # set before super().__init__
            ...         self._target = target
            ...         super().__init__(scene, **kwargs)
            ...     def _make_obs(self):
            ...         return {"qpos": self._franka.get_dofs_position()}
            ...     def _apply_action(self, action):
            ...         self._franka.control_dofs_position(action)
            ...     def _compute_reward(self, action):
            ...         ee = self._franka.get_link("hand").get_pos()
            ...         return -(ee - self._target).norm()
    """

    git_url = "https://github.com/Genesis-Embodied-AI/Genesis"
    libname = "genesis"

    _lib = None

    @_classproperty
    def lib(cls):
        if cls._lib is not None:
            return cls._lib
        import genesis

        cls._lib = genesis
        return genesis

    def __init__(
        self,
        scene=None,
        max_steps: int = 1000,
        frame_skip: int = 1,
        device: DEVICE_TYPING = "cpu",
        batch_size: torch.Size | None = None,
        allow_done_after_reset: bool = False,
    ):
        if scene is None:
            raise TypeError("GenesisWrapper requires a 'scene' argument.")
        if not hasattr(scene, "step"):
            raise TypeError("scene does not have a 'step' method.")

        if batch_size is None:
            batch_size = torch.Size([])
        else:
            batch_size = torch.Size(batch_size)

        super().__init__(
            device=device,
            batch_size=batch_size,
            allow_done_after_reset=allow_done_after_reset,
        )

        self._scene = scene
        self._max_steps = max_steps
        self._frame_skip = frame_skip
        self._current_step = 0

        self._make_specs()

    @_classproperty
    def available_envs(cls):
        if not _has_genesis:
            return []
        return ["custom_scene"]

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _make_obs(self) -> dict[str, torch.Tensor] | torch.Tensor:
        """Read observations from ``self._scene``.

        Override in a subclass for task-specific observations. The default
        emits per-entity DoF position and velocity.
        """
        obs = {}
        for entity in self._scene.entities:
            if getattr(entity, "n_dofs", 0) > 0:
                obs[f"{entity.name}_qpos"] = entity.get_dofs_position()
                obs[f"{entity.name}_qvel"] = entity.get_dofs_velocity()
        if not obs:
            obs["empty_obs"] = torch.zeros(1)
        return obs

    def _apply_action(self, action: torch.Tensor) -> None:
        """Push ``action`` into ``self._scene`` before ``scene.step()``.

        The default applies ``action[..., :n_dofs]`` of the first entity with
        DoFs as a joint-position target. Override for velocity / force control
        or to address multiple entities.
        """
        for entity in self._scene.entities:
            n = getattr(entity, "n_dofs", 0)
            if n > 0:
                entity.control_dofs_position(action[..., :n])
                return

    def _compute_reward(
        self, action: torch.Tensor
    ) -> torch.Tensor | float | None:
        """Per-substep reward. Override for task-specific reward.

        Called once per physics substep (``frame_skip`` times per env step);
        the wrapper sums the returned values. Return ``None`` to skip the
        contribution for this substep. The default returns ``0``.
        """
        return 0.0

    def _compute_done(self) -> torch.Tensor | bool:
        """Per-env truncation flag. Override for early termination.

        Returned as ``truncated``; ``terminated`` is always ``False`` unless
        you override this to return a tensor that you've separately stashed
        under ``self`` (in which case, override :meth:`_step` too for full
        control). The default truncates at ``self._max_steps``.
        """
        return self._current_step >= self._max_steps

    # ------------------------------------------------------------------
    # Spec / TD machinery
    # ------------------------------------------------------------------

    def _make_specs(self) -> None:
        dummy_obs = self._make_obs()
        if not isinstance(dummy_obs, dict):
            dummy_obs = {"observation": dummy_obs}

        obs_entries = {}
        for k, v in dummy_obs.items():
            t = _as_tensor(v, device=self.device)
            event_shape = t.shape[len(self.batch_size) :]
            obs_entries[k] = Unbounded(
                shape=(*self.batch_size, *event_shape),
                dtype=t.dtype if t.is_floating_point() else torch.float32,
                device=self.device,
            )
        self.observation_spec = Composite(
            **obs_entries, shape=self.batch_size, device=self.device
        )

        self.reward_spec = Unbounded(
            shape=(*self.batch_size, 1), dtype=torch.float32, device=self.device
        )

        done_spec = Categorical(
            n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device
        )
        self.done_spec = Composite(
            done=done_spec.clone(),
            truncated=done_spec.clone(),
            terminated=done_spec.clone(),
            shape=self.batch_size,
            device=self.device,
        )

        action_dim = 0
        for entity in self._scene.entities:
            n = getattr(entity, "n_dofs", 0)
            if n > action_dim:
                action_dim = n
        if action_dim == 0:
            action_dim = 1
        self.action_spec = Unbounded(
            shape=(*self.batch_size, action_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def _obs_as_tensordict(self) -> dict[str, torch.Tensor]:
        obs = self._make_obs()
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        return {k: _as_tensor(v, device=self.device) for k, v in obs.items()}

    def _reset(self, tensordict=None, **kwargs):
        self._current_step = 0
        return TensorDict(
            self._obs_as_tensordict(),
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        action = tensordict.get(self.action_key)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device)

        reward = torch.zeros(
            *self.batch_size, 1, dtype=torch.float32, device=self.device
        )
        for _ in range(self._frame_skip):
            self._apply_action(action)
            self._scene.step()
            self._current_step += 1
            r = self._compute_reward(action)
            if r is not None:
                reward = reward + _as_tensor(
                    r, device=self.device, dtype=torch.float32
                ).reshape(*self.batch_size, 1)

        terminated = torch.zeros(
            *self.batch_size, 1, dtype=torch.bool, device=self.device
        )
        truncated = _as_tensor(
            self._compute_done(), device=self.device, dtype=torch.bool
        ).reshape(*self.batch_size, 1)
        done = terminated | truncated

        return TensorDict(
            {
                **self._obs_as_tensordict(),
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        torch.manual_seed(seed)
        self._current_step = 0

    def close(self, *, raise_if_closed: bool = True) -> None:
        scene = getattr(self, "_scene", None)
        if scene is not None and hasattr(scene, "destroy"):
            scene.destroy()
        super().close(raise_if_closed=raise_if_closed)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scene={type(self._scene).__name__}, batch_size={self.batch_size})"
        )


class _GenesisEnvMeta(_EnvPostInit):
    """Return a lazy ParallelEnv when ``num_workers > 1``."""

    def __call__(cls, *args, num_workers: int | None = None, **kwargs):
        if num_workers is None:
            num_workers = kwargs.pop("num_workers", 1)
        else:
            kwargs.pop("num_workers", None)

        num_workers = int(num_workers) if num_workers is not None else 1
        if cls.__name__ == "GenesisEnv" and num_workers > 1:
            from torchrl.envs import ParallelEnv

            env_name = args[0] if len(args) >= 1 else kwargs.get("env_name")
            task_name = args[1] if len(args) >= 2 else kwargs.get("task_name")

            env_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("env_name", "task_name")
            }

            def make_env(_env_name=env_name, _task_name=task_name, _kwargs=env_kwargs):
                return cls(_env_name, _task_name, num_workers=1, **_kwargs)

            return ParallelEnv(num_workers, make_env)

        return super().__call__(*args, **kwargs)


class GenesisEnv(GenesisWrapper, metaclass=_GenesisEnvMeta):
    """Genesis environment built from a named configuration.

    Args:
        env_name (str): registered environment name. Currently one of
            ``'franka_reach'`` or ``'franka_grab'``.
        task_name (str, optional): task name; unused by the built-in configs.
        num_workers (int, optional): when ``> 1``, returns a lazy
            :class:`~torchrl.envs.ParallelEnv` wrapping per-worker Genesis envs.
            Defaults to ``1``.
        max_steps (int, optional): truncation horizon. Defaults to ``1000``.
        frame_skip (int, optional): physics steps per env step. Defaults to ``1``.
        device (torch.device, optional): torch device. Defaults to ``"cpu"``.
        batch_size (torch.Size, optional): env batch size. Defaults to ``()``.
        allow_done_after_reset (bool, optional): Defaults to ``False``.

    For task-specific obs / reward / action, subclass :class:`GenesisWrapper`
    directly -- see its docstring for the hook list.

    Examples:
        >>> from torchrl.envs import GenesisEnv
        >>> env = GenesisEnv(env_name="franka_reach")
        >>> td = env.rollout(10)
    """

    _ENV_CONFIGS = {
        "franka_reach": {
            "description": "Franka robot reaching task",
            "default_obs": "joints",
        },
        "franka_grab": {
            "description": "Franka robot grasping task",
            "default_obs": "joints",
        },
    }

    def __init__(
        self,
        env_name: str,
        task_name: str | None = None,
        num_workers: int | None = None,
        max_steps: int = 1000,
        frame_skip: int = 1,
        device: DEVICE_TYPING = "cpu",
        batch_size: torch.Size | None = None,
        allow_done_after_reset: bool = False,
        **scene_kwargs,
    ):
        if not _has_genesis:
            raise ImportError(
                "genesis python package was not found. "
                "Please install it with: pip install genesis-world"
            )

        self._env_name = env_name
        self._task_name = task_name

        scene = self._create_scene(env_name, task_name, **scene_kwargs)

        super().__init__(
            scene=scene,
            max_steps=max_steps,
            frame_skip=frame_skip,
            device=device,
            batch_size=batch_size,
            allow_done_after_reset=allow_done_after_reset,
        )

    def _create_scene(self, env_name: str, task_name: str | None, **kwargs):
        gs = self.lib

        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.cpu)

        scene = gs.Scene(show_viewer=False)

        if env_name == "franka_reach":
            scene.add_entity(gs.morphs.Plane())
            scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
            scene.build()
        elif env_name == "franka_grab":
            scene.add_entity(gs.morphs.Plane())
            scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
            scene.add_entity(
                gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.5, 0, 0.05))
            )
            scene.build()
        else:
            raise ValueError(
                f"Unknown environment: {env_name}. "
                f"Available environments: {list(self._ENV_CONFIGS.keys())}"
            )

        return scene

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"env={self._env_name}, task={self._task_name}, "
            f"batch_size={self.batch_size})"
        )
