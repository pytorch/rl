import importlib.util
from typing import Dict, Optional

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    NonTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import _classproperty

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

    Args:
        env (pyspiel.State): the game to wrap.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.DiscreteTensorSpec`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHotTensorSpec`).
            Defaults to ``False``.

    Keyword Args:
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs: environments available to build

    Examples:
        >>> import pyspiel
        >>> from torchrl.envs import OpenSpielWrapper
        >>> base_env = pyspiel.load_game('chess').new_initial_state()
        >>> env = OpenSpielWrapper(base_env)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                current_player: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        current_player: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([2, 20, 8, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                        state: NonTensorData(data=FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
                        3572
                        , batch_size=torch.Size([]), device=None),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([2, 20, 8, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                state: NonTensorData(data=FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

                , batch_size=torch.Size([]), device=None),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(env.available_envs)
        ['2048', 'add_noise', 'amazons', 'backgammon', ...]
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

    def __init__(self, env=None, categorical_action_encoding=True, **kwargs):
        if env is not None:
            kwargs["env"] = env
        self._seed_calls_reset = None
        if not categorical_action_encoding:
            raise NotImplementedError
        self._categorical_action_encoding = categorical_action_encoding
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        pyspiel = self.lib
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, pyspiel.State):
            raise TypeError("env is not of type 'pyspiel.State'.")

    def _build_env(self, env, requires_grad: bool = False, **kwargs):
        self.requires_grad = requires_grad
        return env

    def _init_env(self):
        self._update_action_mask()

    def _make_specs(self, env: "pyspiel.State") -> None:  # noqa: F821
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=[*self.batch_size, env.num_players()],
            device=self.device,
        )
        self.done_spec = DiscreteTensorSpec(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
            device=self.device,
        )
        action_spec_cls = (
            DiscreteTensorSpec
            if self._categorical_action_encoding
            else OneHotDiscreteTensorSpec
        )
        self.action_spec = action_spec_cls(
            env.num_distinct_actions(),
            dtype=torch.int64,
            device=self.device,
        )
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(
                    *self.batch_size,
                    env.num_players(),
                    *env.get_game().observation_tensor_shape(),
                ),
                device=self.device,
            ),
            current_player=DiscreteTensorSpec(
                env.num_players(),
                dtype=torch.int,
                device=self.device,
            ),
            state=NonTensorSpec([]),
            shape=self.batch_size,
        )

    def _set_seed(self, seed):
        if seed is not None:
            raise NotImplementedError("This environment has no seed.")

    def current_player(self):
        return self._env.current_player()

    def _update_action_mask(self):
        action_mask = torch.zeros(
            self._env.num_distinct_actions(), device=self.device, dtype=torch.bool
        )
        action_mask[self._env.legal_actions()] = True
        self.action_spec.update_mask(action_mask)

    def _make_td_out(self, exclude_reward=False):
        observation_shape = self._env.get_game().observation_tensor_shape()
        observation = torch.stack(
            [
                self._to_tensor(self._env.observation_tensor(idx)).reshape(
                    observation_shape
                )
                for idx in range(self._env.num_players())
            ]
        )
        reward = self._to_tensor(self._env.returns())
        done = torch.tensor(
            self._env.is_terminal(), device=self.device, dtype=torch.bool
        )
        state = self._env.serialize()
        current_player = torch.tensor(
            self.current_player(), device=self.device, dtype=torch.int
        )

        tensordict_out = TensorDict(
            source={
                "observation": observation,
                "done": done,
                "terminated": done.clone(),
                "state": state,
                "current_player": current_player,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        if not exclude_reward:
            tensordict_out["reward"] = reward

        return tensordict_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get("action")

        if not self._categorical_action_encoding:
            # TODO: To support this, need to convert the one-hot to an index value
            raise NotImplementedError

        self._env.apply_action(action)
        self._update_action_mask()
        return self._make_td_out()

    def _to_tensor(self, value):
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        game = self._env.get_game()

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

    Args:
        game_string (str): the name of the game to wrap. Must be part of
            :attr:`~.available_envs`.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.DiscreteTensorSpec`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHotTensorSpec`).
            Defaults to ``False``.

    Keyword Args:
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs: environments available to build

    Examples:
        >>> from torchrl.envs import OpenSpielEnv
        >>> env = OpenSpielEnv("chess")
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                current_player: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        current_player: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, is_shared=False),
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([2, 20, 8, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                        state: NonTensorData(data=FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
                        673
                        , batch_size=torch.Size([]), device=None),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([2, 20, 8, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                state: NonTensorData(data=FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

                , batch_size=torch.Size([]), device=None),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(env.available_envs)
        ['2048', 'add_noise', 'amazons', 'backgammon', ...]
    """

    def __init__(self, game_string, **kwargs):
        kwargs["game_string"] = game_string
        super().__init__(**kwargs)

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
