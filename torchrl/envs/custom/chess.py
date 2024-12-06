# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, NonTensor, Unbounded

from torchrl.envs import EnvBase

from torchrl.envs.utils import _classproperty


class ChessEnv(EnvBase):
    """A chess environment that follows the TorchRL API.

    Requires: the `chess` library. More info `here <https://python-chess.readthedocs.io/en/latest/>`__.

    Args:
        stateful (bool): Whether to keep track of the internal state of the board.
            If False, the state will be stored in the observation and passed back
            to the environment on each call. Default: ``False``.

    .. note:: the action spec is a :class:`~torchrl.data.Categorical` spec with a ``-1`` shape.
        Unless :meth:`~torchrl.data.Categorical.set_provisional_n` is called with the cardinality of the legal moves,
        valid random actions cannot be taken. :meth:`~torchrl.envs.EnvBase.rand_action` has been adapted to account for
        this behavior.

    Examples:
        >>> env = ChessEnv()
        >>> r = env.reset()
        >>> env.rand_step(r)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                fen: NonTensorData(data=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1, batch_size=torch.Size([]), device=None),
                hashing: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        fen: NonTensorData(data=rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1, batch_size=torch.Size([]), device=None),
                        hashing: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        turn: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                turn: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> env.rollout(1000)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([322]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([322, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                fen: NonTensorStack(
                    ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ...,
                    batch_size=torch.Size([322]),
                    device=None),
                hashing: Tensor(shape=torch.Size([322]), device=cpu, dtype=torch.int64, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([322, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        fen: NonTensorStack(
                            ['rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b ...,
                            batch_size=torch.Size([322]),
                            device=None),
                        hashing: Tensor(shape=torch.Size([322]), device=cpu, dtype=torch.int64, is_shared=False),
                        reward: Tensor(shape=torch.Size([322, 1]), device=cpu, dtype=torch.int32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([322, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        turn: Tensor(shape=torch.Size([322]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([322]),
                    device=None,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([322, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                turn: Tensor(shape=torch.Size([322]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([322]),
            device=None,
            is_shared=False)


    """

    _hash_table = {}

    @_classproperty
    def lib(cls):
        try:
            import chess
        except ImportError:
            raise ImportError(
                "The `chess` library could not be found. Make sure you installed it through `pip install chess`."
            )
        return chess

    def __init__(self, stateful: bool = False):
        chess = self.lib
        super().__init__()
        self.full_observation_spec = Composite(
            hashing=Unbounded(shape=(), dtype=torch.int64),
            fen=NonTensor(shape=()),
            turn=Categorical(n=2, dtype=torch.bool, shape=()),
        )
        self.stateful = stateful
        if not self.stateful:
            self.full_state_spec = self.full_observation_spec.clone()
        self.full_action_spec = Composite(
            action=Categorical(n=-1, shape=(), dtype=torch.int64)
        )
        self.full_reward_spec = Composite(
            reward=Unbounded(shape=(1,), dtype=torch.int32)
        )
        # done spec generated automatically
        self.board = chess.Board()
        if self.stateful:
            self.action_spec.set_provisional_n(len(list(self.board.legal_moves)))

    def rand_action(self, tensordict: Optional[TensorDictBase] = None):
        self._set_action_space(tensordict)
        return super().rand_action(tensordict)

    def _reset(self, tensordict=None):
        fen = None
        if tensordict is not None:
            fen = self._get_fen(tensordict)
            dest = tensordict.empty()
        else:
            dest = TensorDict()

        if fen is None:
            self.board.reset()
            fen = self.board.fen()
        else:
            self.board.set_fen(fen.data)

        hashing = hash(fen)

        self._set_action_space()
        turn = self.board.turn
        return dest.set("fen", fen).set("hashing", hashing).set("turn", turn)

    def _set_action_space(self, tensordict: TensorDict | None = None):
        if not self.stateful and tensordict is not None:
            fen = self._get_fen(tensordict).data
            self.board.set_fen(fen)
        self.action_spec.set_provisional_n(self.board.legal_moves.count())

    @classmethod
    def _get_fen(cls, tensordict):
        fen = tensordict.get("fen", None)
        if fen is None:
            hashing = tensordict.get("hashing", None)
            if hashing is not None:
                fen = cls._hash_table.get(hashing.item())
        return fen

    def _step(self, tensordict):
        # action
        action = tensordict.get("action")
        board = self.board
        if not self.stateful:
            fen = self._get_fen(tensordict).data
            board.set_fen(fen)
        action = str(list(board.legal_moves)[action])
        # assert chess.Move.from_uci(action) in board.legal_moves
        board.push_san(action)
        self._set_action_space()

        # Collect data
        fen = self.board.fen()
        dest = tensordict.empty()
        hashing = hash(fen)
        dest.set("fen", fen)
        dest.set("hashing", hashing)

        done = board.is_checkmate()
        turn = torch.tensor(board.turn)
        reward = torch.tensor([done]).int() * (turn.int() * 2 - 1)
        done = done | board.is_stalemate() | board.is_game_over()
        dest.set("reward", reward)
        dest.set("turn", turn)
        dest.set("done", [done])
        dest.set("terminated", [done])
        return dest

    def _set_seed(self, *args, **kwargs):
        ...

    def cardinality(self, tensordict: TensorDictBase|None=None) -> int:
        self._set_action_space(tensordict)
        return self.action_spec.cardinality()
