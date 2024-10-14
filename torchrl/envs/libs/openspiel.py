# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
from typing import Dict, List

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    Categorical,
    Composite,
    NonTensor,
    OneHot,
    Unbounded,
)
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import _classproperty, check_marl_grouping, MarlGroupMapType

_has_pyspiel = importlib.util.find_spec("pyspiel") is not None


def _get_envs():
    if not _has_pyspiel:
        raise ImportError(
            "open_spiel not found. Consider downloading and installing "
            f"open_spiel from {OpenSpielWrapper.git_url}."
        )

    import pyspiel

    return [game.short_name for game in pyspiel.registered_games()]


class OpenSpielWrapper(_EnvWrapper):
    """Google DeepMind OpenSpiel environment wrapper.

    GitHub: https://github.com/google-deepmind/open_spiel

    Documentation: https://openspiel.readthedocs.io/en/latest/index.html

    Args:
        env (pyspiel.State): the game to wrap.

    Keyword Args:
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``None``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.
        group_map (MarlGroupMapType or Dict[str, List[str]]], optional): how to
            group agents in tensordicts for input/output. See
            :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.
            Defaults to
            :class:`~torchrl.envs.utils.MarlGroupMapType.ALL_IN_ONE_GROUP`.
        categorical_actions (bool, optional): if ``True``, categorical specs
            will be converted to the TorchRL equivalent
            (:class:`torchrl.data.Categorical`), otherwise a one-hot encoding
            will be used (:class:`torchrl.data.OneHot`).  Defaults to ``False``.
        return_state (bool, optional): if ``True``, "state" is included in the
            output of :meth:`~.reset` and :meth:`~step`. The state can be given
            to :meth:`~.reset` to reset to that state, rather than resetting to
            the initial state.
            Defaults to ``False``.

    Attributes:
        available_envs: environments available to build

    Examples:
        >>> import pyspiel
        >>> from torchrl.envs import OpenSpielWrapper
        >>> from tensordict import TensorDict
        >>> base_env = pyspiel.load_game('chess').new_initial_state()
        >>> env = OpenSpielWrapper(base_env, return_state=True)
        >>> td = env.reset()
        >>> td = env.step(env.full_action_spec.rand())
        >>> print(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([2, 4672]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                observation: Tensor(shape=torch.Size([2, 20, 8, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                                reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([2]),
                            device=None,
                            is_shared=False),
                        current_player: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        state: NonTensorData(data=FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
                        3009
                        , batch_size=torch.Size([]), device=None),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(env.available_envs)
        ['2048', 'add_noise', 'amazons', 'backgammon', ...]

    :meth:`~.reset` can restore a specific state, rather than the initial
    state, as long as ``return_state=True``.

        >>> import pyspiel
        >>> from torchrl.envs import OpenSpielWrapper
        >>> from tensordict import TensorDict
        >>> base_env = pyspiel.load_game('chess').new_initial_state()
        >>> env = OpenSpielWrapper(base_env, return_state=True)
        >>> td = env.reset()
        >>> td = env.step(env.full_action_spec.rand())
        >>> td_restore = td["next"]
        >>> td = env.step(env.full_action_spec.rand())
        >>> # Current state is not equal `td_restore`
        >>> (td["next"] == td_restore).all()
        False
        >>> td = env.reset(td_restore)
        >>> # After resetting, now the current state is equal to `td_restore`
        >>> (td == td_restore).all()
        True
    """

    git_url = "https://github.com/google-deepmind/open_spiel"
    libname = "pyspiel"
    _lib = None

    @_classproperty
    def lib(cls):
        if cls._lib is not None:
            return cls._lib

        import pyspiel

        cls._lib = pyspiel
        return pyspiel

    @_classproperty
    def available_envs(cls):
        if not _has_pyspiel:
            return []
        return _get_envs()

    def __init__(
        self,
        env=None,
        *,
        group_map: MarlGroupMapType
        | Dict[str, List[str]] = MarlGroupMapType.ALL_IN_ONE_GROUP,
        categorical_actions: bool = False,
        return_state: bool = False,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env

        self.group_map = group_map
        self.categorical_actions = categorical_actions
        self.return_state = return_state
        self._cached_game = None
        super().__init__(**kwargs)

        # `reset` allows resetting to any state, including a terminal state
        self._allow_done_after_reset = True

    def _check_kwargs(self, kwargs: Dict):
        pyspiel = self.lib
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, pyspiel.State):
            raise TypeError("env is not of type 'pyspiel.State'.")

    def _build_env(self, env, requires_grad: bool = False, **kwargs):
        game = env.get_game()
        game_type = game.get_type()

        if game.max_chance_outcomes() != 0:
            raise NotImplementedError(
                f"The game '{game_type.short_name}' has chance nodes, which are not yet supported."
            )
        if game_type.dynamics == self.lib.GameType.Dynamics.MEAN_FIELD:
            # NOTE: It is unclear from the OpenSpiel documentation what exactly
            # "mean field" means exactly, and there is no documentation on the
            # several games which have it.
            raise RuntimeError(
                f"Mean field games like '{game_type.name}' are not yet " "supported."
            )
        self.parallel = game_type.dynamics == self.lib.GameType.Dynamics.SIMULTANEOUS
        self.requires_grad = requires_grad
        return env

    def _init_env(self):
        self._update_action_mask()

    def _get_game(self):
        if self._cached_game is None:
            self._cached_game = self._env.get_game()
        return self._cached_game

    def _make_group_map(self, group_map, agent_names):
        if group_map is None:
            group_map = MarlGroupMapType.ONE_GROUP_PER_AGENT.get_group_map(agent_names)
        elif isinstance(group_map, MarlGroupMapType):
            group_map = group_map.get_group_map(agent_names)
        check_marl_grouping(group_map, agent_names)
        return group_map

    def _make_group_specs(
        self,
        env,
        group: str,
    ):
        observation_specs = []
        action_specs = []
        reward_specs = []
        game = env.get_game()

        for _ in self.group_map[group]:
            observation_spec = Composite()

            if self.has_observation:
                observation_spec["observation"] = Unbounded(
                    shape=(*game.observation_tensor_shape(),),
                    device=self.device,
                    domain="continuous",
                )

            if self.has_information_state:
                observation_spec["information_state"] = Unbounded(
                    shape=(*game.information_state_tensor_shape(),),
                    device=self.device,
                    domain="continuous",
                )

            observation_specs.append(observation_spec)

            action_spec_cls = Categorical if self.categorical_actions else OneHot
            action_specs.append(
                Composite(
                    action=action_spec_cls(
                        env.num_distinct_actions(),
                        dtype=torch.int64,
                        device=self.device,
                    )
                )
            )

            reward_specs.append(
                Composite(
                    reward=Unbounded(
                        shape=(1,),
                        device=self.device,
                        domain="continuous",
                    )
                )
            )

        group_observation_spec = torch.stack(
            observation_specs, dim=0
        )  # shape = (n_agents, n_obser_per_agent)
        group_action_spec = torch.stack(
            action_specs, dim=0
        )  # shape = (n_agents, n_actions_per_agent)
        group_reward_spec = torch.stack(reward_specs, dim=0)  # shape = (n_agents, 1)

        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
        )

    def _make_specs(self, env: "pyspiel.State") -> None:  # noqa: F821
        self.agent_names = [f"player_{index}" for index in range(env.num_players())]
        self.agent_names_to_indices_map = {
            agent_name: i for i, agent_name in enumerate(self.agent_names)
        }
        self.group_map = self._make_group_map(self.group_map, self.agent_names)
        self.done_spec = Categorical(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
            device=self.device,
        )
        game = env.get_game()
        game_type = game.get_type()
        # In OpenSpiel, a game's state may have either an "observation" tensor,
        # an "information state" tensor, or both. If the OpenSpiel game does not
        # have one of these, then its corresponding accessor functions raise an
        # error, so we must avoid calling them.
        self.has_observation = game_type.provides_observation_tensor
        self.has_information_state = game_type.provides_information_state_tensor

        observation_spec = {}
        action_spec = {}
        reward_spec = {}

        for group in self.group_map.keys():
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
            ) = self._make_group_specs(
                env,
                group,
            )
            observation_spec[group] = group_observation_spec
            action_spec[group] = group_action_spec
            reward_spec[group] = group_reward_spec

        if self.return_state:
            observation_spec["state"] = NonTensor([])

        observation_spec["current_player"] = Unbounded(
            shape=(),
            dtype=torch.int,
            device=self.device,
            domain="discrete",
        )

        self.observation_spec = Composite(observation_spec)
        self.action_spec = Composite(action_spec)
        self.reward_spec = Composite(reward_spec)

    def _set_seed(self, seed):
        if seed is not None:
            raise NotImplementedError("This environment has no seed.")

    def current_player(self):
        return self._env.current_player()

    def _update_action_mask(self):
        if self._env.is_terminal():
            agents_acting = []
        else:
            agents_acting = [
                self.agent_names
                if self.parallel
                else self.agent_names[self._env.current_player()]
            ]
        for group, agents in self.group_map.items():
            action_masks = []
            for agent in agents:
                agent_index = self.agent_names_to_indices_map[agent]
                if agent in agents_acting:
                    action_mask = torch.zeros(
                        self._env.num_distinct_actions(),
                        device=self.device,
                        dtype=torch.bool,
                    )
                    action_mask[self._env.legal_actions(agent_index)] = True
                else:
                    action_mask = torch.zeros(
                        self._env.num_distinct_actions(),
                        device=self.device,
                        dtype=torch.bool,
                    )
                    # In OpenSpiel parallel games, non-acting players are
                    # expected to take action 0.
                    # https://openspiel.readthedocs.io/en/latest/api_reference/state_apply_action.html
                    action_mask[0] = True
                action_masks.append(action_mask)
            self.full_action_spec[group, "action"].update_mask(
                torch.stack(action_masks, dim=0)
            )

    def _make_td_out(self, exclude_reward=False):
        done = torch.tensor(
            self._env.is_terminal(), device=self.device, dtype=torch.bool
        )
        current_player = torch.tensor(
            self.current_player(), device=self.device, dtype=torch.int
        )

        source = {
            "done": done,
            "terminated": done.clone(),
            "current_player": current_player,
        }

        if self.return_state:
            source["state"] = self._env.serialize()

        reward = self._env.returns()

        for group, agent_names in self.group_map.items():
            agent_tds = []

            for agent in agent_names:
                agent_index = self.agent_names_to_indices_map[agent]
                agent_source = {}
                if self.has_observation:
                    observation_shape = self._get_game().observation_tensor_shape()
                    agent_source["observation"] = self._to_tensor(
                        self._env.observation_tensor(agent_index)
                    ).reshape(observation_shape)

                if self.has_information_state:
                    information_state_shape = (
                        self._get_game().information_state_tensor_shape()
                    )
                    agent_source["information_state"] = self._to_tensor(
                        self._env.information_state_tensor(agent_index)
                    ).reshape(information_state_shape)

                if not exclude_reward:
                    agent_source["reward"] = self._to_tensor(reward[agent_index])

                agent_td = TensorDict(
                    source=agent_source,
                    batch_size=self.batch_size,
                    device=self.device,
                )
                agent_tds.append(agent_td)

            source[group] = torch.stack(agent_tds, dim=0)

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )

        return tensordict_out

    def _get_action_from_tensor(self, tensor):
        if not self.categorical_actions:
            action = torch.argmax(tensor, dim=-1)
        else:
            action = tensor
        return action

    def _step_parallel(self, tensordict: TensorDictBase):
        actions = [0] * self._env.num_players()
        for group, agents in self.group_map.items():
            for index_in_group, agent in enumerate(agents):
                agent_index = self.agent_names_to_indices_map[agent]
                action_tensor = tensordict[group, "action"][index_in_group]
                action = self._get_action_from_tensor(action_tensor)
                actions[agent_index] = action

        self._env.apply_actions(actions)

    def _step_sequential(self, tensordict: TensorDictBase):
        agent_index = self._env.current_player()

        # If the game has ended, do nothing
        if agent_index == self.lib.PlayerId.TERMINAL:
            return

        agent = self.agent_names[agent_index]
        agent_group = None
        agent_index_in_group = None

        for group, agents in self.group_map.items():
            if agent in agents:
                agent_group = group
                agent_index_in_group = agents.index(agent)
                break

        assert agent_group is not None

        action_tensor = tensordict[agent_group, "action"][agent_index_in_group]
        action = self._get_action_from_tensor(action_tensor)
        self._env.apply_action(action)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.parallel:
            self._step_parallel(tensordict)
        else:
            self._step_sequential(tensordict)

        self._update_action_mask()
        return self._make_td_out()

    def _to_tensor(self, value):
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        game = self._get_game()

        if tensordict is not None and "state" in tensordict:
            new_env = game.deserialize_state(tensordict["state"])
        else:
            new_env = game.new_initial_state()

        self._env = new_env
        self._update_action_mask()
        return self._make_td_out(exclude_reward=True)


class OpenSpielEnv(OpenSpielWrapper):
    """Google DeepMind OpenSpiel environment wrapper built with the game string.

    GitHub: https://github.com/google-deepmind/open_spiel

    Documentation: https://openspiel.readthedocs.io/en/latest/index.html

    Args:
        game_string (str): the name of the game to wrap. Must be part of
            :attr:`~.available_envs`.

    Keyword Args:
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``None``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.
        group_map (MarlGroupMapType or Dict[str, List[str]]], optional): how to
            group agents in tensordicts for input/output. See
            :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.
            Defaults to
            :class:`~torchrl.envs.utils.MarlGroupMapType.ALL_IN_ONE_GROUP`.
        categorical_actions (bool, optional): if ``True``, categorical specs
            will be converted to the TorchRL equivalent
            (:class:`torchrl.data.Categorical`), otherwise a one-hot encoding
            will be used (:class:`torchrl.data.OneHot`).  Defaults to ``False``.
        return_state (bool, optional): if ``True``, "state" is included in the
            output of :meth:`~.reset` and :meth:`~step`. The state can be given
            to :meth:`~.reset` to reset to that state, rather than resetting to
            the initial state.
            Defaults to ``False``.

    Attributes:
        available_envs: environments available to build

    Examples:
        >>> from torchrl.envs import OpenSpielEnv
        >>> from tensordict import TensorDict
        >>> env = OpenSpielEnv("chess", return_state=True)
        >>> td = env.reset()
        >>> td = env.step(env.full_action_spec.rand())
        >>> print(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([2, 4672]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                observation: Tensor(shape=torch.Size([2, 20, 8, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                                reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([2]),
                            device=None,
                            is_shared=False),
                        current_player: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        state: NonTensorData(data=FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
                        674
                        , batch_size=torch.Size([]), device=None),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(env.available_envs)
        ['2048', 'add_noise', 'amazons', 'backgammon', ...]

    :meth:`~.reset` can restore a specific state, rather than the initial state,
    as long as ``return_state=True``.

        >>> from torchrl.envs import OpenSpielEnv
        >>> from tensordict import TensorDict
        >>> env = OpenSpielEnv("chess", return_state=True)
        >>> td = env.reset()
        >>> td = env.step(env.full_action_spec.rand())
        >>> td_restore = td["next"]
        >>> td = env.step(env.full_action_spec.rand())
        >>> # Current state is not equal `td_restore`
        >>> (td["next"] == td_restore).all()
        False
        >>> td = env.reset(td_restore)
        >>> # After resetting, now the current state is equal to `td_restore`
        >>> (td == td_restore).all()
        True
    """

    def __init__(
        self,
        game_string,
        *,
        group_map: MarlGroupMapType
        | Dict[str, List[str]] = MarlGroupMapType.ALL_IN_ONE_GROUP,
        categorical_actions=False,
        return_state: bool = False,
        **kwargs,
    ):
        kwargs["game_string"] = game_string
        super().__init__(
            group_map=group_map,
            categorical_actions=categorical_actions,
            return_state=return_state,
            **kwargs,
        )

    def _build_env(
        self,
        game_string: str,
        **kwargs,
    ) -> "pyspiel.State":  # noqa: F821
        if not _has_pyspiel:
            raise ImportError(
                f"open_spiel not found, unable to create {game_string}. Consider "
                f"downloading and installing open_spiel from {self.git_url}"
            )
        requires_grad = kwargs.pop("requires_grad", False)
        parameters = kwargs.pop("parameters", None)
        if kwargs:
            raise ValueError("kwargs not supported.")

        if parameters:
            game = self.lib.load_game(game_string, parameters=parameters)
        else:
            game = self.lib.load_game(game_string)

        env = game.new_initial_state()
        return super()._build_env(
            env,
            requires_grad=requires_grad,
        )

    @property
    def game_string(self):
        return self._constructor_kwargs["game_string"]

    def _check_kwargs(self, kwargs: Dict):
        if "game_string" not in kwargs:
            raise TypeError("Expected 'game_string' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.game_string}, batch_size={self.batch_size}, device={self.device})"
