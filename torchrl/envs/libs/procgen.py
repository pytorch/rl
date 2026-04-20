from __future__ import annotations

import importlib
import warnings

import torch
from tensordict import TensorDict

from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform, set_gym_backend
from torchrl.envs.utils import _classproperty

__all__ = ["ProcgenWrapper", "ProcgenEnv"]

_has_procgen = importlib.util.find_spec("procgen") is not None

if _has_procgen:
    import procgen  # type: ignore
else:
    procgen = None  # type: ignore


def _get_procgen_envs() -> list[str]:
    if not _has_procgen:
        raise ImportError("procgen is not installed.")
    env_names = getattr(procgen, "ENV_NAMES", None)
    if env_names:
        return list(env_names)
    try:
        env_mod = importlib.import_module("procgen.env")
        return list(getattr(env_mod, "ENV_NAMES", []))
    except Exception:
        return list(getattr(procgen, "ENV_NAMES", []))


def _get_num_envs(env) -> int | None:
    """Get the number of parallel environments from a procgen env."""
    # procgen.ProcgenEnv returns a ToGymEnv wrapper; the num attribute
    # may be on the wrapper, the inner env (.env), or as num_envs
    return (
        getattr(env, "num", None)
        or getattr(env, "nenvs", None)
        or getattr(env, "num_envs", None)
        or getattr(getattr(env, "env", None), "num", None)
    )


class ProcgenWrapper(_EnvWrapper):
    """OpenAI Procgen environment wrapper.

    Wraps an existing :class:`procgen.ProcgenEnv` instance and exposes it
    under the TorchRL environment API.

    This wrapper is responsible for:
    - Converting Procgen observations (``{"rgb": np.ndarray}``) to Torch tensors
    - Handling vectorized Procgen semantics
    - Producing TorchRL-compliant ``TensorDict`` outputs

    Args:
        env (procgen.ProcgenEnv): an already constructed Procgen environment.

    Keyword Args:
        device (torch.device | str, optional): device on which tensors are placed.
        batch_size (torch.Size, optional): expected batch size.
        allow_done_after_reset (bool, optional): tolerate done right after reset.

    Attributes:
        available_envs (List[str]): list of Procgen environment ids.

    Examples:
        >>> import procgen
        >>> from torchrl.envs.libs.procgen import ProcgenWrapper
        >>> env = procgen.ProcgenEnv(4, "coinrun")
        >>> env = ProcgenWrapper(env=env)
        >>> td = env.reset()
        >>> print(td)
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([4, 3, 64, 64]), device=cpu, dtype=torch.uint8, is_shared=False),
                reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False
        )
        >>> print(td["observation"].shape)
        torch.Size([4, 3, 64, 64])
    """

    git_url = "https://github.com/openai/procgen"
    lib = procgen

    @_classproperty
    def available_envs(cls) -> list[str]:
        if not _has_procgen:
            return []
        return _get_procgen_envs()

    def __init__(self, env, **kwargs):
        # Detect num_envs before calling parent __init__ so batch_size is set
        # before _make_specs() is called
        n = _get_num_envs(env)
        if n is not None and "batch_size" not in kwargs:
            kwargs["batch_size"] = torch.Size([n])
        super().__init__(env=env, **kwargs)

    def _check_kwargs(self, kwargs: dict) -> None:
        if "env" not in kwargs:
            raise TypeError("ProcgenWrapper requires an 'env' argument.")

    def _build_env(self, env, **_) -> procgen.ProcgenEnv:
        return env

    @property
    def observation_space(self):
        # gym3 uses ob_space instead of observation_space
        return getattr(self._env, "observation_space", None) or self._env.ob_space

    @property
    def action_space(self):
        # gym3 uses ac_space instead of action_space
        return getattr(self._env, "action_space", None) or self._env.ac_space

    def _make_specs(self, env) -> None:
        from torchrl.data.tensor_specs import Bounded

        batch_size = self.batch_size

        # Procgen observation is rgb with shape (64, 64, 3) per env
        # After permuting in _reset/_step it becomes (3, 64, 64) per env
        # With batch_size, full shape is (*batch_size, 3, 64, 64)
        self.observation_spec = Composite(
            observation=Bounded(
                low=0,
                high=255,
                shape=(*batch_size, 3, 64, 64),
                dtype=torch.uint8,
                device=self.device,
            ),
            shape=batch_size,
        )

        # Procgen has Discrete(15) action space
        with set_gym_backend("gym"):
            action_spec = _gym_to_torchrl_spec_transform(
                self.action_space,
                categorical_action_encoding=True,
                device=self.device,
            )
        # Expand action spec to include batch dimension
        if len(batch_size) > 0 and action_spec.shape[: len(batch_size)] != batch_size:
            action_spec = action_spec.expand(*batch_size, *action_spec.shape)
        self.action_spec = action_spec

        self.reward_spec = Composite(
            reward=Unbounded(
                shape=(*batch_size, 1), dtype=torch.float32, device=self.device
            ),
            shape=batch_size,
        )

        done_leaf = Categorical(
            n=2, shape=(*batch_size, 1), dtype=torch.bool, device=self.device
        )
        self.done_spec = Composite(
            done=done_leaf.clone(),
            terminated=done_leaf.clone(),
            truncated=done_leaf.clone(),
            shape=batch_size,
        )

    def _init_env(self) -> None:
        # batch_size is set in __init__ before _make_specs() is called
        try:
            self._env.reset()
        except Exception:
            pass

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        try:
            if hasattr(self._env, "seed"):
                self._env.seed(seed)
            elif hasattr(self._env, "set_seed"):
                self._env.set_seed(seed)
            elif hasattr(self._env, "rand_seed"):
                self._env.rand_seed = seed
        except Exception:
            warnings.warn("ProcgenWrapper: seeding failed (best-effort).")

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        obs = self._env.reset()
        if isinstance(obs, (tuple, list)):
            obs = obs[0]

        rgb = torch.from_numpy(obs["rgb"]).to(self.device).permute(0, 3, 1, 2)

        td = TensorDict(
            {"observation": rgb},
            batch_size=self.batch_size,
            device=self.device,
        )

        # Set done flags (required by TorchRL)
        zeros = torch.zeros((*self.batch_size, 1), device=self.device, dtype=torch.bool)
        td.set("done", zeros)
        td.set("terminated", zeros.clone())
        td.set("truncated", zeros.clone())

        return td

    def _step(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        action = tensordict.get("action")
        # Procgen expects numpy arrays with shape (num_envs,)
        action_np = action.cpu().numpy().flatten()
        obs, reward, done, info = self._env.step(action_np)

        rgb = torch.from_numpy(obs["rgb"]).to(self.device).permute(0, 3, 1, 2)
        reward = torch.as_tensor(reward, device=self.device).view(-1, 1)
        done = torch.as_tensor(done, device=self.device).view(-1, 1).bool()

        td = TensorDict(
            {
                "observation": rgb,
                "reward": reward,
                "done": done,
                "terminated": done.clone(),
                "truncated": torch.zeros_like(done),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        # Expose info dict fields (e.g., level_seed, prev_level_complete)
        # Note: procgen may return info as a list of dicts or a single dict
        if info and isinstance(info, dict):
            for key, val in info.items():
                td.set(key, torch.as_tensor(val, device=self.device))

        return td


class ProcgenEnv(ProcgenWrapper):
    """OpenAI Procgen environment.

    Convenience class that constructs a Procgen environment by name.

    See https://github.com/openai/procgen for more details on Procgen.

    Args:
        env_name (str): name of the Procgen game (e.g. ``"coinrun"``).
            Available games: bigfish, bossfight, caveflyer, chaser, climber,
            coinrun, dodgeball, fruitbot, heist, jumper, leaper, maze, miner,
            ninja, plunder, starpilot.

    Keyword Args:
        num_envs (int, optional): number of parallel environments. Defaults to 1.
        distribution_mode (str, optional): Procgen distribution mode. One of
            ``"easy"``, ``"hard"``, ``"extreme"``, ``"memory"``, ``"exploration"``.
            Defaults to ``"hard"``.
        start_level (int, optional): the level id to start from. Defaults to 0.
        num_levels (int, optional): the number of unique levels that can be
            generated. Set to 0 for unlimited levels. Defaults to 0.
        use_sequential_levels (bool, optional): if ``True``, levels are played
            sequentially rather than randomly. Defaults to ``False``.
        center_agent (bool, optional): if ``True``, observations are centered
            on the agent. Defaults to ``True``.
        use_backgrounds (bool, optional): if ``True``, include background
            assets. Defaults to ``True``.
        use_monochrome_assets (bool, optional): if ``True``, use monochrome
            assets for simpler visuals. Defaults to ``False``.
        restrict_themes (bool, optional): if ``True``, restrict visual themes.
            Defaults to ``False``.
        use_generated_assets (bool, optional): if ``True``, use procedurally
            generated assets. Defaults to ``False``.
        paint_vel_info (bool, optional): if ``True``, paint velocity info on
            observations. Defaults to ``False``.
        seed (int, optional): random seed for the environment. Note that procgen
            environments must be seeded at construction time; calling ``set_seed()``
            after construction will not work reliably.
        render_mode (str, optional): render mode for the environment.
        device (torch.device | str, optional): device for tensors.
        allow_done_after_reset (bool, optional): tolerate done after reset.

    Examples:
        >>> from torchrl.envs.libs.procgen import ProcgenEnv
        >>> env = ProcgenEnv("coinrun", num_envs=8)
        >>> td = env.reset()
        >>> print(td)
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([8, 3, 64, 64]), device=cpu, dtype=torch.uint8, is_shared=False),
                reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False
        )
        >>> print(td["observation"].shape)
        torch.Size([8, 3, 64, 64])
        >>> print(env.available_envs)
        ['bigfish', 'bossfight', 'caveflyer', 'chaser', 'climber', 'coinrun', 'dodgeball', 'fruitbot', 'heist', 'jumper', 'leaper', 'maze', 'miner', 'ninja', 'plunder', 'starpilot']
    """

    def __init__(self, env_name: str, **kwargs):
        if not _has_procgen:
            raise ImportError(
                "procgen python package was not found. "
                "Install it from https://github.com/openai/procgen."
            )

        if env_name not in self.available_envs:
            raise ValueError(
                f"Unknown Procgen environment '{env_name}'. "
                f"Available envs: {self.available_envs}"
            )

        num_envs = kwargs.pop("num_envs", 1)
        # Procgen uses rand_seed for seeding at construction time
        seed = kwargs.pop("seed", None)
        if seed is not None:
            kwargs["rand_seed"] = seed
        # Extract procgen-specific kwargs before passing to parent
        procgen_kwargs = {}
        for key in list(kwargs.keys()):
            if key in (
                "distribution_mode",
                "start_level",
                "num_levels",
                "use_sequential_levels",
                "center_agent",
                "use_backgrounds",
                "use_monochrome_assets",
                "restrict_themes",
                "use_generated_assets",
                "paint_vel_info",
                "render_mode",
                "rand_seed",
            ):
                procgen_kwargs[key] = kwargs.pop(key)
        env = procgen.ProcgenEnv(num_envs, env_name, **procgen_kwargs)
        # Pass batch_size to parent; it will be set before _make_specs()
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = torch.Size([num_envs])
        super().__init__(env=env, **kwargs)
