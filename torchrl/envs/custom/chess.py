# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import io
import pathlib
from typing import Dict, Optional

import torch
from PIL import Image
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, NonTensor, Unbounded

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

    This environment simulates a chess game using the `chess` library. It supports various state representations
    and can be configured to include different types of observations such as SAN, FEN, PGN, and legal moves.

    Requires: the `chess` library. More info `here <https://python-chess.readthedocs.io/en/latest/>`__.

    Args:
        stateful (bool): Whether to keep track of the internal state of the board.
            If False, the state will be stored in the observation and passed back
            to the environment on each call. Default: ``True``.
        include_san (bool): Whether to include SAN (Standard Algebraic Notation) in the observations. Default: ``False``.
        include_fen (bool): Whether to include FEN (Forsyth-Edwards Notation) in the observations. Default: ``False``.
        include_pgn (bool): Whether to include PGN (Portable Game Notation) in the observations. Default: ``False``.
        include_legal_moves (bool): Whether to include legal moves in the observations. Default: ``False``.
        include_hash (bool): Whether to include hash transformations in the environment. Default: ``False``.
        pixels (bool): Whether to include pixel-based observations of the board. Default: ``False``.

    .. note:: The action spec is a :class:`~torchrl.data.Categorical` with a number of actions equal to the number of possible SAN moves.
        The action space is structured as a categorical distribution over all possible SAN moves, with the legal moves
        being a subset of this space. The environment uses a mask to ensure only legal moves are selected.

    Examples:
        >>> env = ChessEnv(include_fen=True, include_san=True, include_pgn=True, include_legal_moves=True)
        >>> r = env.reset()
        >>> env.rand_step(r)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                fen: NonTensorData(data=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1, batch_size=torch.Size([]), device=None),
                legal_moves: Tensor(shape=torch.Size([219]), device=cpu, dtype=torch.int64, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        fen: NonTensorData(data=rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1, batch_size=torch.Size([]), device=None),
                        legal_moves: Tensor(shape=torch.Size([219]), device=cpu, dtype=torch.int64, is_shared=False),
                        pgn: NonTensorData(data=[Event "?"]
                        [Site "?"]
                        [Date "????.??.??"]
                        [Round "?"]
                        [White "?"]
                        [Black "?"]
                        [Result "*"]
                        1. b3 *, batch_size=torch.Size([]), device=None),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                        san: NonTensorData(data=b3, batch_size=torch.Size([]), device=None),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        turn: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                pgn: NonTensorData(data=[Event "?"]
                [Site "?"]
                [Date "????.??.??"]
                [Round "?"]
                [White "?"]
                [Black "?"]
                [Result "*"]
                *, batch_size=torch.Size([]), device=None),
                san: NonTensorData(data=[SAN][START], batch_size=torch.Size([]), device=None),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                turn: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> env.rollout(1000)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([352]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([352, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                fen: NonTensorStack(
                    ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ...,
                    batch_size=torch.Size([352]),
                    device=None),
                legal_moves: Tensor(shape=torch.Size([352, 219]), device=cpu, dtype=torch.int64, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([352, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        fen: NonTensorStack(
                            ['rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR b K...,
                            batch_size=torch.Size([352]),
                            device=None),
                        legal_moves: Tensor(shape=torch.Size([352, 219]), device=cpu, dtype=torch.int64, is_shared=False),
                        pgn: NonTensorStack(
                            ['[Event "?"]\n[Site "?"]\n[Date "????.??.??"]\n[R...,
                            batch_size=torch.Size([352]),
                            device=None),
                        reward: Tensor(shape=torch.Size([352, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        san: NonTensorStack(
                            ['Na3', 'a5', 'Nb1', 'Nc6', 'a3', 'g6', 'd4', 'd6'...,
                            batch_size=torch.Size([352]),
                            device=None),
                        terminated: Tensor(shape=torch.Size([352, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        turn: Tensor(shape=torch.Size([352]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([352]),
                    device=None,
                    is_shared=False),
                pgn: NonTensorStack(
                    ['[Event "?"]\n[Site "?"]\n[Date "????.??.??"]\n[R...,
                    batch_size=torch.Size([352]),
                    device=None),
                san: NonTensorStack(
                    ['[SAN][START]', 'Na3', 'a5', 'Nb1', 'Nc6', 'a3', ...,
                    batch_size=torch.Size([352]),
                    device=None),
                terminated: Tensor(shape=torch.Size([352, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                turn: Tensor(shape=torch.Size([352]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([352]),
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

    _san_moves = []

    @_classproperty
    def san_moves(cls):
        if not cls._san_moves:
            with open(pathlib.Path(__file__).parent / "san_moves.txt", "r+") as f:
                cls._san_moves.extend(f.read().split("\n"))
        return cls._san_moves

    def _legal_moves_to_index(
        self,
        tensordict: TensorDictBase | None = None,
        board: "chess.Board" | None = None,  # noqa: F821
        return_mask: bool = False,
        pad: bool = False,
    ) -> torch.Tensor:
        if not self.stateful and tensordict is not None:
            fen = self._get_fen(tensordict).data
            self.board.set_fen(fen)
            board = self.board
        elif board is None:
            board = self.board
        indices = torch.tensor(
            [self._san_moves.index(board.san(m)) for m in board.legal_moves],
            dtype=torch.int64,
        )
        if return_mask:
            return torch.zeros(len(self.san_moves), dtype=torch.bool).index_fill_(
                0, indices, True
            )
        if pad:
            indices = torch.nn.functional.pad(
                indices, [0, 218 - indices.numel() + 1], value=len(self.san_moves)
            )
        return indices

    def __init__(
        self,
        *,
        stateful: bool = True,
        include_san: bool = False,
        include_fen: bool = False,
        include_pgn: bool = False,
        include_legal_moves: bool = False,
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
        self.include_legal_moves = include_legal_moves
        if include_legal_moves:
            # 218 max possible legal moves per chess board position
            # https://www.stmintz.com/ccc/index.php?id=424966
            # len(self.san_moves)+1 is the padding value
            self.full_observation_spec["legal_moves"] = Bounded(
                0, 1 + len(self.san_moves), shape=(218,), dtype=torch.int64
            )
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
            action=Categorical(n=len(self.san_moves), shape=(), dtype=torch.int64)
        )
        self.full_reward_spec = Composite(
            reward=Unbounded(shape=(1,), dtype=torch.float32)
        )
        # done spec generated automatically
        self.board = chess.Board()
        if self.stateful:
            self.action_spec.set_provisional_n(len(list(self.board.legal_moves)))

    def rand_action(self, tensordict: Optional[TensorDictBase] = None):
        mask = self._legal_moves_to_index(tensordict, return_mask=True)
        self.action_spec.update_mask(mask)
        return super().rand_action(tensordict)

    def _is_done(self, board):
        return board.is_game_over() | board.is_fifty_moves()

    def _reset(self, tensordict=None):
        fen = None
        pgn = None
        if tensordict is not None:
            if self.include_fen:
                fen = self._get_fen(tensordict)
                if fen is not None:
                    fen = fen.data
                dest = tensordict.empty()
            if self.include_pgn:
                pgn = self._get_pgn(tensordict)
                if pgn is not None:
                    pgn = pgn.data
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
        if self.include_legal_moves:
            moves_idx = self._legal_moves_to_index(board=self.board, pad=True)
            dest.set("legal_moves", moves_idx)
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
        cls, pgn_string: str, board: "chess.Board" | None = None  # noqa: F821
    ) -> "chess.Board":  # noqa: F821
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
    def _board_to_pgn(cls, board: "chess.Board") -> str:  # noqa: F821
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

        san = self.san_moves[action]
        board.push_san(san)

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

        if self.include_legal_moves:
            moves_idx = self._legal_moves_to_index(board=board, pad=True)
            dest.set("legal_moves", moves_idx)

        turn = torch.tensor(board.turn)
        done = self._is_done(board)
        if board.is_checkmate():
            # turn flips after every move, even if the game is over
            # winner = not turn
            reward_val = 1  # if winner == self.lib.WHITE else 0
        elif done:
            reward_val = 0.5
        else:
            reward_val = 0.0

        reward = torch.tensor([reward_val], dtype=torch.float32)
        dest.set("reward", reward)
        dest.set("turn", turn)
        dest.set("done", [done])
        dest.set("terminated", [done])
        if self.pixels:
            dest.set("pixels", self._get_tensor_image(board=self.board))

        if self.stateful:
            # Make sure that rand_action will work next iteration
            self._set_action_space()

        return dest

    def _set_seed(self, *args, **kwargs):
        ...

    def cardinality(self, tensordict: TensorDictBase | None = None) -> int:
        self._set_action_space(tensordict)
        return self.action_spec.cardinality()
