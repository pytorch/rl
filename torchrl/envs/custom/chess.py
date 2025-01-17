# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import io
from typing import Dict, Optional

import torch
from PIL import Image
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, NonTensor, Unbounded

from torchrl.envs import EnvBase
from torchrl.envs.common import _EnvPostInit

from torchrl.envs.utils import _classproperty


class _HashMeta(_EnvPostInit):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if kwargs.get("include_hash"):
            from torchrl.envs import Hash

            in_keys = []
            out_keys = []
            if instance.include_san:
                in_keys.append("san")
                out_keys.append("san_hash")
            if instance.include_fen:
                in_keys.append("fen")
                out_keys.append("fen_hash")
            if instance.include_pgn:
                in_keys.append("pgn")
                out_keys.append("pgn_hash")
            return instance.append_transform(Hash(in_keys, out_keys))
        return instance


class ChessEnv(EnvBase, metaclass=_HashMeta):
    """A chess environment that follows the TorchRL API.

    Requires: the `chess` library. More info `here <https://python-chess.readthedocs.io/en/latest/>`__.

    Args:
        stateful (bool): Whether to keep track of the internal state of the board.
            If False, the state will be stored in the observation and passed back
            to the environment on each call. Default: ``True``.

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

    _hash_table: Dict[int, str] = {}
    _PNG_RESTART = """[Event "?"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "*"]

*"""

    @_classproperty
    def lib(cls):
        try:
            import chess
            import chess.pgn
        except ImportError:
            raise ImportError(
                "The `chess` library could not be found. Make sure you installed it through `pip install chess`."
            )
        return chess

    def __init__(
        self,
        *,
        stateful: bool = True,
        include_san: bool = False,
        include_fen: bool = False,
        include_pgn: bool = False,
        include_hash: bool = False,
        pixels: bool = False,
    ):
        chess = self.lib
        super().__init__()
        self.full_observation_spec = Composite(
            turn=Categorical(n=2, dtype=torch.bool, shape=()),
        )
        self.include_san = include_san
        self.include_fen = include_fen
        self.include_pgn = include_pgn
        if include_san:
            self.full_observation_spec["san"] = NonTensor(shape=(), example_data="Nc6")
        if include_pgn:
            self.full_observation_spec["pgn"] = NonTensor(
                shape=(), example_data=self._PNG_RESTART
            )
        if include_fen:
            self.full_observation_spec["fen"] = NonTensor(shape=(), example_data="any")
        if not stateful and not (include_pgn or include_fen):
            raise RuntimeError(
                "At least one state representation (pgn or fen) must be enabled when stateful "
                f"is {stateful}."
            )

        self.stateful = stateful

        if not self.stateful:
            self.full_state_spec = self.full_observation_spec.clone()

        self.pixels = pixels
        if pixels:
            if importlib.util.find_spec("cairosvg") is None:
                raise ImportError(
                    "Please install cairosvg to use this environment with pixel rendering."
                )
            if importlib.util.find_spec("torchvision") is None:
                raise ImportError(
                    "Please install torchvision to use this environment with pixel rendering."
                )
            self.full_observation_spec["pixels"] = Unbounded(shape=())

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

    def _is_done(self, board):
        return board.is_game_over() | board.is_fifty_moves()

    def _reset(self, tensordict=None):
        fen = None
        pgn = None
        if tensordict is not None:
            if self.include_fen:
                fen = self._get_fen(tensordict).data
                dest = tensordict.empty()
            if self.include_pgn:
                fen = self._get_pgn(tensordict).data
                dest = tensordict.empty()
        else:
            dest = TensorDict()

        if fen is None and pgn is None:
            self.board.reset()
            if self.include_fen and fen is None:
                fen = self.board.fen()
            if self.include_pgn and pgn is None:
                pgn = self._PNG_RESTART
        else:
            if fen is not None:
                self.board.set_fen(fen)
                if self._is_done(self.board):
                    raise ValueError(
                        "Cannot reset to a fen that is a gameover state." f" fen: {fen}"
                    )
            elif pgn is not None:
                self.board = self._pgn_to_board(pgn)

        self._set_action_space()
        turn = self.board.turn
        if self.include_san:
            dest.set("san", "[SAN][START]")
        if self.include_fen:
            if fen is None:
                fen = self.board.fen()
            dest.set("fen", fen)
        if self.include_pgn:
            if pgn is None:
                pgn = self._board_to_pgn(self.board)
            dest.set("pgn", pgn)
        dest.set("turn", turn)
        if self.pixels:
            dest.set("pixels", self._get_tensor_image(board=self.board))
        return dest

    _cairosvg_lib = None

    @_classproperty
    def _cairosvg(cls):
        csvg = cls._cairosvg_lib
        if csvg is None:
            import cairosvg

            csvg = cls._cairosvg_lib = cairosvg
        return csvg

    _torchvision_lib = None

    @_classproperty
    def _torchvision(cls):
        tv = cls._torchvision_lib
        if tv is None:
            import torchvision

            tv = cls._torchvision_lib = torchvision
        return tv

    @classmethod
    def _get_tensor_image(cls, board):
        try:
            svg = board._repr_svg_()
            # Convert SVG to PNG using cairosvg
            png_data = io.BytesIO()
            cls._cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=png_data)
            png_data.seek(0)
            # Open the PNG image using Pillow
            img = Image.open(png_data)
            img = cls._torchvision.transforms.functional.pil_to_tensor(img)
        except ImportError:
            raise ImportError(
                "Chess rendering requires cairosvg and torchvision to be installed."
            )
        return img

    def _set_action_space(self, tensordict: TensorDict | None = None):
        if not self.stateful and tensordict is not None:
            fen = self._get_fen(tensordict).data
            self.board.set_fen(fen)
        self.action_spec.set_provisional_n(self.board.legal_moves.count())

    @classmethod
    def _pgn_to_board(
        cls, pgn_string: str, board: "chess.Board" | None = None
    ) -> "chess.Board":
        pgn_io = io.StringIO(pgn_string)
        game = cls.lib.pgn.read_game(pgn_io)
        if board is None:
            board = cls.Board()
        else:
            board.reset()
        for move in game.mainline_moves():
            board.push(move)
        return board

    @classmethod
    def _board_to_pgn(cls, board: "chess.Board") -> str:
        # Create a new Game object
        game = cls.lib.pgn.Game()

        # Add the moves to the game
        node = game
        for move in board.move_stack:
            node = node.add_variation(move)

        # Generate the PGN string
        pgn_string = str(game)
        return pgn_string

    @classmethod
    def _get_fen(cls, tensordict):
        fen = tensordict.get("fen", None)
        return fen

    def get_legal_moves(self, tensordict=None, uci=False):
        """List the legal moves in a position.

        To choose one of the actions, the "action" key can be set to the index
        of the move in this list.

        Args:
            tensordict (TensorDict, optional): Tensordict containing the fen
                string of a position. Required if not stateful. If stateful,
                this argument is ignored and the current state of the env is
                used instead.

            uci (bool, optional): If ``False``, moves are given in SAN format.
                If ``True``, moves are given in UCI format. Default is
                ``False``.

        """
        board = self.board
        if not self.stateful:
            if tensordict is None:
                raise ValueError(
                    "tensordict must be given since this env is not stateful"
                )
            fen = self._get_fen(tensordict).data
            board.set_fen(fen)
        moves = board.legal_moves

        if uci:
            return [board.uci(move) for move in moves]
        else:
            return [board.san(move) for move in moves]

    def _step(self, tensordict):
        # action
        action = tensordict.get("action")
        board = self.board

        if not self.stateful:
            if self.include_fen:
                fen = self._get_fen(tensordict).data
                board.set_fen(fen)
            elif self.include_pgn:
                pgn = self._get_pgn(tensordict).data
                self._pgn_to_board(pgn, board)
            else:
                raise RuntimeError(
                    "Not enough information to deduce the board. If stateful=False, include_pgn or include_fen must be True."
                )

        action = list(board.legal_moves)[action]
        san = None
        if self.include_san:
            san = board.san(action)
        board.push(action)

        self._set_action_space()

        dest = tensordict.empty()

        # Collect data
        if self.include_fen:
            fen = board.fen()
            dest.set("fen", fen)

        if self.include_pgn:
            pgn = self._board_to_pgn(board)
            dest.set("pgn", pgn)

        if san is not None:
            dest.set("san", san)

        turn = torch.tensor(board.turn)
        if board.is_checkmate():
            # turn flips after every move, even if the game is over
            winner = not turn
            reward_val = 1 if winner == self.lib.WHITE else -1
        else:
            reward_val = 0

        reward = torch.tensor([reward_val], dtype=torch.int32)
        done = self._is_done(board)
        dest.set("reward", reward)
        dest.set("turn", turn)
        dest.set("done", [done])
        dest.set("terminated", [done])
        if self.pixels:
            dest.set("pixels", self._get_tensor_image(board=self.board))
        return dest

    def _set_seed(self, *args, **kwargs):
        ...

    def cardinality(self, tensordict: TensorDictBase | None = None) -> int:
        self._set_action_space(tensordict)
        return self.action_spec.cardinality()
