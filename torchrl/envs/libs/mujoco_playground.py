# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs.batched_envs import ParallelEnv
from torchrl.envs.common import _EnvPostInit, _EnvWrapper
from torchrl.envs.libs.jax_utils import (
    _ndarray_to_tensor,
    _tensor_to_ndarray,
    _tree_flatten,
    _tree_reshape,
)
from torchrl.envs.utils import _classproperty

_has_mujoco_playground = importlib.util.find_spec("mujoco_playground") is not None


def _listerize(ranges: list) -> list[int]:
    result = []
    for r in ranges:
        if isinstance(r, tuple):
            result.extend(range(r[0], r[1] + 1))
        else:
            result.append(r)
    return result


@dataclass
class MujocoPlaygroundAgentSpec:
    """Observation/action slice definition for one agent in a cooperative task.

    Args:
        name (str): group key used in output TensorDicts (e.g. ``"agent_0"``).
        action_indices (list of int): indices into the global action vector that
            this agent controls. Must be non-overlapping across all agents and
            together must cover ``range(env.action_size)``.
        observation_indices (list of int or dict of str to list of int): for
            flat-obs environments, a list of ints selecting from the global
            observation vector. For dict-obs environments, a ``dict`` mapping
            each observation key to a list of ints selecting from that key's
            sub-vector.
    """

    name: str
    action_indices: list[int]
    observation_indices: list[int] | dict[str, list[int]]


@dataclass
class MujocoPlaygroundAgentMapping:
    """Agent mapping for :class:`MujocoPlaygroundWrapper`.

    Defines how to split a single-agent MuJoCo Playground environment into a
    cooperative multi-agent task by partitioning the observation and action
    vectors among named agents.

    Args:
        agents (list of MujocoPlaygroundAgentSpec): one entry per agent,
            defining each agent's observation slice and the action indices it
            controls.
        homogenization_mode (str, optional): strategy for unifying
            heterogeneous observation/action shapes across agents so that a
            single shared policy can be used.

            - ``"none"`` (default): each agent receives exactly its own
              observation/action slice; shapes may differ across agents.
            - ``"max"``: observations are padded to
              ``max_obs_size + n_agents`` (a one-hot agent-ID prefix is
              prepended), and actions are padded to ``max_action_size``.
              All agents share the same input/output shape.
            - ``"concat"``: each agent receives the full global
              observation/action vector with zeros at positions it does not
              own. All agents share the same input/output shape equal to the
              full environment dimensions.

    Examples:
        >>> mapping = MujocoPlaygroundAgentMapping(
        ...     agents=[
        ...         MujocoPlaygroundAgentSpec(
        ...             name="agent_0",
        ...             action_indices=[0, 1, 2],
        ...             observation_indices=[0, 1, 2, 3],
        ...         ),
        ...         MujocoPlaygroundAgentSpec(
        ...             name="agent_1",
        ...             action_indices=[3, 4, 5],
        ...             observation_indices=[4, 5, 6, 7],
        ...         ),
        ...     ],
        ...     homogenization_mode="none",
        ... )
    """

    agents: list[MujocoPlaygroundAgentSpec]
    homogenization_mode: Literal["none", "max", "concat"] = "none"


def _validate_agent_mapping(
    mapping: MujocoPlaygroundAgentMapping,
    action_size: int,
    observation_size: int | dict,
) -> None:
    """Validate a :class:`MujocoPlaygroundAgentMapping` against environment dims.

    Args:
        mapping (MujocoPlaygroundAgentMapping): the mapping to validate.
        action_size (int): total action dimension of the environment.
        observation_size (int or dict): flat obs size or dict obs size map.

    Raises:
        ValueError: if action_indices overlap, are out of range, do not cover
            ``range(action_size)``, or if observation_indices are invalid.
    """
    if not mapping.agents:
        raise ValueError("MujocoPlaygroundAgentMapping.agents must not be empty.")

    # Validate action indices
    seen: set[int] = set()
    for agent in mapping.agents:
        for idx in agent.action_indices:
            if idx < 0 or idx >= action_size:
                raise ValueError(
                    f"Agent '{agent.name}' has action index {idx} out of range "
                    f"[0, {action_size})."
                )
            if idx in seen:
                raise ValueError(
                    f"Action index {idx} appears more than once across agents "
                    f"(duplicate found in agent '{agent.name}'). "
                    "Action indices must be non-overlapping."
                )
            seen.add(idx)
    if seen != set(range(action_size)):
        missing = sorted(set(range(action_size)) - seen)
        raise ValueError(
            f"Action indices do not cover range({action_size}). "
            f"Missing indices: {missing}."
        )

    # Validate observation indices
    obs_is_dict = isinstance(observation_size, dict)
    for agent in mapping.agents:
        if obs_is_dict:
            if not isinstance(agent.observation_indices, dict):
                raise ValueError(
                    f"Agent '{agent.name}': environment has dict observations, "
                    "so observation_indices must be a dict mapping obs keys to "
                    "lists of ints."
                )
            for key, indices in agent.observation_indices.items():
                if key not in observation_size:
                    raise ValueError(
                        f"Agent '{agent.name}': observation key '{key}' not "
                        f"found in environment observation_size keys: "
                        f"{list(observation_size.keys())}."
                    )
                key_size = (
                    int(observation_size[key][0])
                    if hasattr(observation_size[key], "__len__")
                    else int(observation_size[key])
                )
                for idx in indices:
                    if idx < 0 or idx >= key_size:
                        raise ValueError(
                            f"Agent '{agent.name}': observation index {idx} "
                            f"for key '{key}' out of range [0, {key_size})."
                        )
        else:
            if not isinstance(agent.observation_indices, list):
                raise ValueError(
                    f"Agent '{agent.name}': environment has flat observations, "
                    "so observation_indices must be a list of ints."
                )
            for idx in agent.observation_indices:
                if idx < 0 or idx >= observation_size:
                    raise ValueError(
                        f"Agent '{agent.name}' has observation index {idx} "
                        f"out of range [0, {observation_size})."
                    )


# Predefined multi-agent partitionings for common MuJoCo locomotion tasks.
# These mappings mirror the decompositions used in JaxMARL's MABrax suite.
KNOWN_MARL_MAPPINGS: dict[str, MujocoPlaygroundAgentMapping] = {
    "ant_4x2": MujocoPlaygroundAgentMapping(
        agents=[
            MujocoPlaygroundAgentSpec(
                "agent_0",
                [0, 1],
                _listerize([(0, 5), 6, 7, 9, 11, (13, 18), 19, 20]),
            ),
            MujocoPlaygroundAgentSpec(
                "agent_1",
                [2, 3],
                _listerize([(0, 5), 7, 8, 9, 11, (13, 18), 21, 22]),
            ),
            MujocoPlaygroundAgentSpec(
                "agent_2",
                [4, 5],
                _listerize([(0, 5), 7, 9, 10, 11, (13, 18), 23, 24]),
            ),
            MujocoPlaygroundAgentSpec(
                "agent_3",
                [6, 7],
                _listerize([(0, 5), 7, 9, 11, 12, (13, 18), 25, 26]),
            ),
        ]
    ),
    "halfcheetah_6x1": MujocoPlaygroundAgentMapping(
        agents=[
            MujocoPlaygroundAgentSpec(
                "agent_0", [0], _listerize([(1, 2), 3, 4, 6, (9, 11), 12])
            ),
            MujocoPlaygroundAgentSpec(
                "agent_1", [1], _listerize([(1, 2), 3, 4, 5, (9, 11), 13])
            ),
            MujocoPlaygroundAgentSpec(
                "agent_2", [2], _listerize([(1, 2), 4, 5, (9, 11), 14])
            ),
            MujocoPlaygroundAgentSpec(
                "agent_3", [3], _listerize([(1, 2), 3, 6, 7, (9, 11), 15])
            ),
            MujocoPlaygroundAgentSpec(
                "agent_4", [4], _listerize([(1, 2), 6, 7, 8, (9, 11), 16])
            ),
            MujocoPlaygroundAgentSpec(
                "agent_5", [5], _listerize([(1, 2), 7, 8, (9, 11)])
            ),
        ]
    ),
    "hopper_3x1": MujocoPlaygroundAgentMapping(
        agents=[
            MujocoPlaygroundAgentSpec(
                "agent_0", [0], _listerize([(0, 1), 2, 3, (5, 7), 8])
            ),
            MujocoPlaygroundAgentSpec(
                "agent_1", [1], _listerize([(0, 1), 2, 3, 4, (5, 7), 9])
            ),
            MujocoPlaygroundAgentSpec(
                "agent_2", [2], _listerize([(0, 1), 3, 4, (5, 7), 10])
            ),
        ]
    ),
    "humanoid_9|8": MujocoPlaygroundAgentMapping(
        agents=[
            MujocoPlaygroundAgentSpec(
                "agent_0",
                [0, 1, 2, 11, 12, 13, 14, 15, 16],
                _listerize(
                    [
                        (0, 10),
                        (12, 14),
                        (16, 30),
                        (39, 44),
                        (55, 94),
                        (115, 124),
                        (145, 184),
                        (191, 214),
                        (227, 232),
                        (245, 277),
                        (286, 291),
                        (298, 321),
                        (334, 339),
                        (352, 375),
                    ]
                ),
            ),
            MujocoPlaygroundAgentSpec(
                "agent_1",
                [3, 4, 5, 6, 7, 8, 9, 10],
                _listerize(
                    [
                        (0, 15),
                        (22, 27),
                        (31, 38),
                        (85, 144),
                        (209, 244),
                        (269, 274),
                        (278, 285),
                        (316, 351),
                    ]
                ),
            ),
        ]
    ),
    "walker2d_2x3": MujocoPlaygroundAgentMapping(
        agents=[
            MujocoPlaygroundAgentSpec(
                "agent_0", [0, 1, 2], _listerize([0, (2, 5), (8, 9), (11, 13)])
            ),
            MujocoPlaygroundAgentSpec(
                "agent_1", [3, 4, 5], _listerize([0, 2, (5, 9), (14, 16)])
            ),
        ]
    ),
}


def _get_envs():
    if not _has_mujoco_playground:
        raise ImportError(
            "mujoco_playground is not installed in your virtual environment."
        )

    from mujoco_playground import dm_control_suite, locomotion, manipulation

    return (
        list(dm_control_suite.ALL_ENVS)
        + list(locomotion.ALL_ENVS)
        + list(manipulation.ALL_ENVS)
    )


class _MujocoPlaygroundMeta(_EnvPostInit):
    """Metaclass for MujocoPlaygroundEnv that returns a lazy ParallelEnv when num_workers > 1."""

    def __call__(cls, *args, num_workers: int | None = None, **kwargs):
        num_workers = 1 if num_workers is None else int(num_workers)
        if cls.__name__ == "MujocoPlaygroundEnv" and num_workers > 1:
            env_name = args[0] if len(args) >= 1 else kwargs.get("env_name")
            env_kwargs = {k: v for k, v in kwargs.items() if k != "env_name"}

            def make_env(_env_name=env_name, _kwargs=env_kwargs):
                return cls(_env_name, num_workers=1, **_kwargs)

            return ParallelEnv(num_workers, make_env)

        return super().__call__(*args, **kwargs)


class MujocoPlaygroundWrapper(_EnvWrapper):
    """Google DeepMind MuJoCo Playground environment wrapper.

    MuJoCo Playground is a collection of JAX-based MJX environments spanning
    locomotion, manipulation, and dm_control suite tasks.

    GitHub: https://github.com/google-deepmind/mujoco_playground

    Args:
        env (mujoco_playground._src.mjx_env.MjxEnv): the environment to wrap.
        agent_mapping (:class:`MujocoPlaygroundAgentMapping` or str, optional):
            if provided, the environment is decomposed into a cooperative
            multi-agent task. Can be either a :class:`MujocoPlaygroundAgentMapping`
            instance or a string key into :data:`KNOWN_MARL_MAPPINGS`.
            Known string values: ``"ant_4x2"``, ``"halfcheetah_6x1"``,
            ``"hopper_3x1"``, ``"humanoid_9|8"``, ``"walker2d_2x3"``.
            Defaults to ``None`` (single-agent mode).

    Keyword Args:
        from_pixels (bool, optional): Not yet supported.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            In ``mujoco_playground``, this controls the number of environments
            simulated in parallel via JAX's ``vmap`` on a single device (GPU/TPU).
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs: environments available to build

    Examples:
        >>> from mujoco_playground import dm_control_suite
        >>> from torchrl.envs import MujocoPlaygroundWrapper
        >>> import torch
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> base_env = dm_control_suite.load("CartpoleBalance")
        >>> env = MujocoPlaygroundWrapper(base_env, device=device)
        >>> env.set_seed(0)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([1]), dtype=torch.float32),
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                next: TensorDict(
                    fields={
                        observation: Tensor(torch.Size([5]), dtype=torch.float32)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(torch.Size([5]), dtype=torch.float32),
                reward: Tensor(torch.Size([1]), dtype=torch.float32),
                state: TensorDict(...)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['AcrobotSwingup', 'AcrobotSwingupSparse', 'BallInCupCatch', ...]

    """

    git_url = "https://github.com/google-deepmind/mujoco_playground"

    @_classproperty
    def available_envs(cls):
        if not _has_mujoco_playground:
            return []
        return list(_get_envs())

    libname = "mujoco_playground"

    _lib = None
    _jax = None

    @_classproperty
    def lib(cls):
        if cls._lib is not None:
            return cls._lib

        import mujoco_playground

        cls._lib = mujoco_playground
        return mujoco_playground

    @_classproperty
    def jax(cls):
        if cls._jax is not None:
            return cls._jax

        import jax

        cls._jax = jax
        return jax

    def __init__(
        self,
        env=None,
        agent_mapping: MujocoPlaygroundAgentMapping | str | None = None,
        **kwargs,
    ):
        if isinstance(agent_mapping, str):
            if agent_mapping not in KNOWN_MARL_MAPPINGS:
                raise ValueError(
                    f"Unknown agent_mapping '{agent_mapping}'. "
                    f"Known mappings: {sorted(KNOWN_MARL_MAPPINGS)}."
                )
            agent_mapping = KNOWN_MARL_MAPPINGS[agent_mapping]
        if env is not None:
            kwargs["env"] = env
        self._seed_calls_reset = None
        self._agent_mapping = agent_mapping
        super().__init__(**kwargs)
        if not self.device:
            warnings.warn(
                f"No device is set for env {self}. "
                "Setting a device in MujocoPlayground wrapped environments is strongly recommended."
            )

    def _check_kwargs(self, kwargs: dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        for attr in ("reset", "step", "action_size", "observation_size"):
            if not hasattr(env, attr):
                raise TypeError(
                    f"env is missing required attribute '{attr}'. "
                    "Expected a mujoco_playground MjxEnv instance."
                )

    def _build_env(
        self,
        env,
        _seed: int | None = None,
        from_pixels: bool = False,
        render_kwargs: dict | None = None,
        pixels_only: bool = False,
        camera_id: int | str = 0,
        **kwargs,
    ):
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only

        if from_pixels:
            raise NotImplementedError(
                "from_pixels=True is not yet supported within MujocoPlaygroundWrapper"
            )
        return env

    def _obs_is_dict(self) -> bool:
        """Returns True if the wrapped environment uses dict observations."""
        return isinstance(self._env.observation_size, dict)

    def _make_specs(self, env) -> None:
        if self._agent_mapping is not None:
            self._make_marl_specs(env)
            return

        obs_is_dict = self._obs_is_dict()

        self.action_spec = Bounded(
            low=-1,
            high=1,
            shape=(*self.batch_size, env.action_size),
            device=self.device,
        )
        self.reward_spec = Unbounded(shape=[*self.batch_size, 1], device=self.device)

        if not obs_is_dict:
            self.observation_spec = Composite(
                observation=Unbounded(
                    shape=(*self.batch_size, env.observation_size),
                    device=self.device,
                ),
                shape=self.batch_size,
            )
        else:
            obs_specs = {
                key: Unbounded(shape=(*self.batch_size, *shape), device=self.device)
                for key, shape in env.observation_size.items()
            }
            self.observation_spec = Composite(**obs_specs, shape=self.batch_size)

    def _make_marl_specs(self, env) -> None:
        """Build nested per-agent Composite specs when agent_mapping is set."""
        mapping = self._agent_mapping
        _validate_agent_mapping(mapping, env.action_size, env.observation_size)

        agents = mapping.agents
        n_agents = len(agents)
        mode = mapping.homogenization_mode
        obs_is_dict = self._obs_is_dict()

        # Compute per-agent action dimension
        if mode == "none":
            action_dims = [len(a.action_indices) for a in agents]
        elif mode == "max":
            max_act = max(len(a.action_indices) for a in agents)
            action_dims = [max_act] * n_agents
        else:  # concat
            action_dims = [env.action_size] * n_agents

        # Compute per-agent observation dimension
        if obs_is_dict:
            # For dict obs, per-agent obs size = sum of all selected key dims
            raw_obs_sizes = []
            for a in agents:
                total = sum(len(idxs) for idxs in a.observation_indices.values())
                raw_obs_sizes.append(total)
        else:
            raw_obs_sizes = [len(a.observation_indices) for a in agents]

        if obs_is_dict and mode != "none":
            raise NotImplementedError(
                f"homogenization_mode='{mode}' is not yet supported for "
                "dict-observation environments."
            )

        if mode == "none":
            obs_dims = raw_obs_sizes
        elif mode == "max":
            max_obs = max(raw_obs_sizes)
            obs_dims = [max_obs + n_agents] * n_agents  # +n_agents for one-hot
        else:  # concat
            obs_dims = [env.observation_size] * n_agents

        # Build per-agent specs
        action_spec_dict = {}
        obs_spec_dict = {}
        reward_spec_dict = {}
        for i, agent in enumerate(agents):
            action_spec_dict[agent.name] = Composite(
                action=Bounded(
                    low=-1,
                    high=1,
                    shape=(*self.batch_size, action_dims[i]),
                    device=self.device,
                ),
                shape=self.batch_size,
            )
            obs_spec_dict[agent.name] = Composite(
                observation=Unbounded(
                    shape=(*self.batch_size, obs_dims[i]),
                    device=self.device,
                ),
                shape=self.batch_size,
            )
            reward_spec_dict[agent.name] = Composite(
                reward=Unbounded(
                    shape=(*self.batch_size, 1),
                    device=self.device,
                ),
                shape=self.batch_size,
            )

        self.action_spec = Composite(
            **action_spec_dict,
            shape=self.batch_size,
        )
        self.reward_spec = Composite(
            **reward_spec_dict,
            shape=self.batch_size,
        )
        self.observation_spec = Composite(
            **obs_spec_dict,
            shape=self.batch_size,
        )

    def _init_env(self) -> int | None:
        jax = self.jax
        self._key = None
        self._current_state = None
        # jit inside vmap (not outside) avoids retracing when batch_size changes
        # and lets XLA fuse the per-env kernel before stacking across the batch.
        self._vmap_jit_env_reset = jax.vmap(jax.jit(self._env.reset))
        self._vmap_jit_env_step = jax.vmap(jax.jit(self._env.step))

    def _set_seed(self, seed: int | None) -> None:
        jax = self.jax
        if seed is None:
            raise Exception("MujocoPlayground requires an integer seed.")
        self._key = jax.random.PRNGKey(seed)

    def _extract_obs(self, state) -> dict:
        """Extract observation tensors directly from raw JAX state.

        For flat obs: returns ``{"observation": tensor}``.
        For dict obs: returns ``{key: tensor, ...}`` spread directly.
        """
        if not self._obs_is_dict():
            return {"observation": _ndarray_to_tensor(state.obs).to(self.device)}
        else:
            return {
                key: _ndarray_to_tensor(getattr(state.obs, key)).to(self.device)
                for key in self._env.observation_size
            }

    def _split_obs_for_agents(self, state) -> dict:
        """Build per-agent observation TensorDicts from a raw JAX state.

        Args:
            state: raw JAX env state with ``.obs`` attribute.

        Returns:
            dict mapping each agent name to a :class:`~tensordict.TensorDict`
            with an ``"observation"`` key whose shape matches the agent's spec.
        """
        mapping = self._agent_mapping
        agents = mapping.agents
        n_agents = len(agents)
        mode = mapping.homogenization_mode
        obs_is_dict = self._obs_is_dict()

        if obs_is_dict:
            obs_raw = {
                key: _ndarray_to_tensor(getattr(state.obs, key)).to(self.device)
                for key in self._env.observation_size
            }
        else:
            obs_raw = _ndarray_to_tensor(state.obs).to(self.device)

        result = {}
        for i, agent in enumerate(agents):
            if obs_is_dict:
                # Concatenate selected indices from each obs key
                # (only "none" mode is supported for dict obs, validated at spec build time)
                parts = []
                for key, idxs in agent.observation_indices.items():
                    parts.append(obs_raw[key][..., idxs])
                obs_tensor = torch.cat(parts, dim=-1)
            else:
                # Flat obs
                raw = obs_raw[..., agent.observation_indices]
                if mode == "none":
                    obs_tensor = raw
                elif mode == "max":
                    raw_sizes = [len(a.observation_indices) for a in agents]
                    max_obs = max(raw_sizes)
                    padded_size = max_obs + n_agents
                    obs_tensor = torch.zeros(
                        *self.batch_size,
                        padded_size,
                        dtype=obs_raw.dtype,
                        device=self.device,
                    )
                    # One-hot agent ID in first n_agents positions
                    obs_tensor[..., i] = 1.0
                    # Raw obs in next len(obs_indices) positions
                    obs_tensor[
                        ..., n_agents : n_agents + len(agent.observation_indices)
                    ] = raw
                else:  # concat
                    total_obs = self._env.observation_size
                    obs_tensor = torch.zeros(
                        *self.batch_size,
                        total_obs,
                        dtype=obs_raw.dtype,
                        device=self.device,
                    )
                    obs_tensor[..., agent.observation_indices] = raw

            result[agent.name] = TensorDict(
                {"observation": obs_tensor},
                batch_size=self.batch_size,
                device=self.device,
            )
        return result

    def _reconstruct_global_action(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Reassemble the global action tensor from per-agent action slices.

        Args:
            tensordict (TensorDictBase): input TensorDict containing per-agent
                action tensors at ``(agent_name, "action")``.

        Returns:
            :class:`torch.Tensor` of shape ``(*batch_size, action_size)``
            suitable for passing to the underlying JAX environment.
        """
        mapping = self._agent_mapping
        mode = mapping.homogenization_mode
        global_action = torch.zeros(
            *self.batch_size,
            self._env.action_size,
            dtype=torch.float32,
            device=self.device,
        )
        for agent in mapping.agents:
            a = tensordict.get((agent.name, "action"))
            n = len(agent.action_indices)
            if mode == "none":
                global_action[..., agent.action_indices] = a
            elif mode == "max":
                # Policy outputs up to max_action_size; only first n are real
                global_action[..., agent.action_indices] = a[..., :n]
            else:  # concat
                # Policy outputs full global action; take only this agent's slice
                global_action[..., agent.action_indices] = a[..., agent.action_indices]
        return global_action

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        jax = self.jax

        if getattr(self, "_key", None) is None:
            seed = getattr(self, "_seed", None)
            if seed is None:
                seed = 0
            self._key = jax.random.PRNGKey(int(seed))

        self._key, *keys = jax.random.split(self._key, 1 + self.numel())

        state = self._vmap_jit_env_reset(jax.numpy.stack(keys))
        # vmap output has leading dim = batch_size.numel() (flat).
        # _tree_reshape restores the original batch shape (e.g. [4, 8]).
        state = _tree_reshape(state, self.batch_size)
        # Store JAX state directly — avoids converting MJX/pytree state to
        # TensorDict and back, which breaks MJX's metadata pytree registration.
        self._current_state = state

        done_shape = (*self.batch_size, 1)
        done = _ndarray_to_tensor(state.done).to(self.device).bool().view(*done_shape)

        if self._agent_mapping is not None:
            source = {
                **self._split_obs_for_agents(state),
                "done": done,
                "terminated": done.clone(),
            }
        else:
            source = {
                **self._extract_obs(state),
                "done": done,
                "terminated": done.clone(),
            }

        return TensorDict._new_unsafe(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        state = self._current_state

        if self._agent_mapping is not None:
            action_tensor = self._reconstruct_global_action(tensordict)
        else:
            action_tensor = tensordict.get("action")

        action = _tensor_to_ndarray(action_tensor)

        # vmap expects a flat leading batch dim, so collapse [d0, d1, ...] → [d0*d1*...].
        state = _tree_flatten(state, self.batch_size)
        action = _tree_flatten(action, self.batch_size)

        next_state = self._vmap_jit_env_step(state, action)

        # Restore the original batch shape after vmap.
        next_state = _tree_reshape(next_state, self.batch_size)
        self._current_state = next_state

        done_shape = (*self.batch_size, 1)
        reward = _ndarray_to_tensor(next_state.reward).to(self.device).view(*done_shape)
        done = (
            _ndarray_to_tensor(next_state.done).to(self.device).bool().view(*done_shape)
        )

        if self._agent_mapping is not None:
            agent_tds = self._split_obs_for_agents(next_state)
            for agent in self._agent_mapping.agents:
                agent_tds[agent.name].set("reward", reward.clone())
            source = {
                **agent_tds,
                "done": done,
                "terminated": done.clone(),
            }
        else:
            source = {
                **self._extract_obs(next_state),
                "reward": reward,
                "done": done,
                "terminated": done.clone(),
            }

        return TensorDict._new_unsafe(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )


class MujocoPlaygroundEnv(MujocoPlaygroundWrapper, metaclass=_MujocoPlaygroundMeta):
    """Google DeepMind MuJoCo Playground environment wrapper built with the environment name.

    MuJoCo Playground is a collection of JAX-based MJX environments spanning
    locomotion, manipulation, and dm_control suite tasks. All environments from
    all suites are accessible by name via the unified registry.

    GitHub: https://github.com/google-deepmind/mujoco_playground

    Args:
        env_name (str): the environment name of the env to wrap. Must be part of
            :attr:`~.available_envs`.
        config (ml_collections.ConfigDict, optional): configuration for the environment.
            If ``None``, the default configuration is used. Defaults to ``None``.
        config_overrides (dict, optional): overrides to apply on top of ``config``.
            Defaults to ``None``.
        agent_mapping (:class:`MujocoPlaygroundAgentMapping` or str, optional):
            if provided, the environment is decomposed into a cooperative
            multi-agent task. Can be either a :class:`MujocoPlaygroundAgentMapping`
            instance or a string key into :data:`KNOWN_MARL_MAPPINGS`.
            Known string values: ``"ant_4x2"``, ``"halfcheetah_6x1"``,
            ``"hopper_3x1"``, ``"humanoid_9|8"``, ``"walker2d_2x3"``.
            The mapping and the environment name are validated against each other
            at construction time. Defaults to ``None`` (single-agent mode).

    Keyword Args:
        from_pixels (bool, optional): Not yet supported.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            In ``mujoco_playground``, this controls the number of environments
            simulated in parallel via JAX's ``vmap`` on a single device (GPU/TPU).
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.
        num_workers (int, optional): if greater than 1, a lazy :class:`~torchrl.envs.ParallelEnv`
            will be returned instead, with each worker instantiating its own
            :class:`~torchrl.envs.MujocoPlaygroundEnv` instance. Defaults to ``None``.

    .. note::
        There are two orthogonal ways to scale environment throughput:

        - **batch_size**: Uses MuJoCo Playground's native JAX-based vectorization
          (``vmap``) to run multiple environments in parallel on a single GPU/TPU.
        - **num_workers**: Uses TorchRL's :class:`~torchrl.envs.ParallelEnv` to
          spawn multiple Python processes, each running its own
          ``MujocoPlaygroundEnv``.

        These can be combined: ``MujocoPlaygroundEnv("CartpoleBalance", batch_size=[128], num_workers=4)``
        creates 4 worker processes each running 128 vectorized environments.

    Attributes:
        available_envs: environments available to build (all suites combined)

    Examples:
        >>> from torchrl.envs import MujocoPlaygroundEnv
        >>> import torch
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> env = MujocoPlaygroundEnv("CartpoleBalance", device=device)
        >>> env.set_seed(0)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([1]), dtype=torch.float32),
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                next: TensorDict(
                    fields={
                        observation: Tensor(torch.Size([5]), dtype=torch.float32)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(torch.Size([5]), dtype=torch.float32),
                reward: Tensor(torch.Size([1]), dtype=torch.float32),
                state: TensorDict(...)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['AcrobotSwingup', 'AcrobotSwingupSparse', 'BallInCupCatch', ...]

    To take advantage of MuJoCo Playground's JAX-based parallelism, pass a
    ``batch_size`` to run multiple environments in parallel on a single device:

    Examples:
        >>> from torchrl.envs import MujocoPlaygroundEnv
        >>> import torch
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> env = MujocoPlaygroundEnv("CartpoleBalance", batch_size=[128], device=device)
        >>> env.set_seed(0)
        >>> td = env.rollout(100)
        >>> print(td.shape)
        torch.Size([128, 100])

    """

    def __init__(self, env_name: str, config=None, config_overrides=None, **kwargs):
        kwargs["env_name"] = env_name
        if config is not None:
            kwargs["config"] = config
        if config_overrides is not None:
            kwargs["config_overrides"] = config_overrides
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        config=None,
        config_overrides=None,
        **kwargs,
    ):
        if not _has_mujoco_playground:
            raise ImportError(
                f"mujoco_playground not found, unable to create {env_name}. "
                f"Consider downloading and installing mujoco_playground from"
                f" {self.git_url}"
            )
        from mujoco_playground import registry

        from_pixels = kwargs.pop("from_pixels", False)
        pixels_only = kwargs.pop("pixels_only", True)
        camera_id = kwargs.pop("camera_id", 0)
        render_kwargs = kwargs.pop("render_kwargs", None)
        if kwargs:
            raise ValueError(f"Unsupported kwargs: {sorted(kwargs)}")

        self.wrapper_frame_skip = 1
        env = registry.load(env_name, config=config, config_overrides=config_overrides)
        return super()._build_env(
            env,
            pixels_only=pixels_only,
            from_pixels=from_pixels,
            camera_id=camera_id,
            render_kwargs=render_kwargs,
        )

    @property
    def env_name(self) -> str:
        return self._constructor_kwargs["env_name"]

    def _check_kwargs(self, kwargs: dict):
        if "env_name" not in kwargs:
            raise TypeError("Expected 'env_name' to be part of kwargs")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self.env_name}, "
            f"batch_size={self.batch_size}, device={self.device})"
        )
