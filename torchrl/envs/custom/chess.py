# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import io
import pathlib

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    Binary,
    Bounded,
    Categorical,
    Composite,
    NonTensor,
    Unbounded,
)
from torchrl.envs import EnvBase
from torchrl.envs.common import _EnvPostInit
from torchrl.envs.utils import _classproperty


class _ChessMeta(_EnvPostInit):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        include_hash = kwargs.get("include_hash")
        include_hash_inv = kwargs.get("include_hash_inv")
        if include_hash:
            from torchrl.envs import Hash

            in_keys = []
            out_keys = []
            in_keys_inv = [] if include_hash_inv else None
            out_keys_inv = [] if include_hash_inv else None

            def maybe_add_keys(condition, in_key, out_key):
                if condition:
                    in_keys.append(in_key)
                    out_keys.append(out_key)
                    if include_hash_inv:
                        in_keys_inv.append(in_key)
                        out_keys_inv.append(out_key)

            maybe_add_keys(instance.include_san, "san", "san_hash")
            maybe_add_keys(instance.include_fen, "fen", "fen_hash")
            maybe_add_keys(instance.include_pgn, "pgn", "pgn_hash")

            instance = instance.append_transform(
                Hash(in_keys, out_keys, in_keys_inv, out_keys_inv)
            )
        elif include_hash_inv:
            raise ValueError(
                "'include_hash_inv=True' can only be set if"
                f"'include_hash=True', but got 'include_hash={include_hash}'."
            )
        if kwargs.get("mask_actions", True):
            from torchrl.envs import ActionMask

            instance = instance.append_transform(ActionMask())
        return instance


class ChessEnv(EnvBase, metaclass=_ChessMeta):
    r"""A chess environment that follows the TorchRL API.

    This environment simulates a chess game using the `chess` library. It supports various state representations
    and can be configured to include different types of observations such as SAN, FEN, PGN, and legal moves.

    Requires: the `chess` library. More info `here <https://python-chess.readthedocs.io/en/latest/>`__.

    Args:
        stateful (bool): Whether to keep track of the internal state of the board.
            If False, the state will be stored in the observation and passed back
            to the environment on each call. Default: ``True``.
        include_san (bool): Whether to include SAN (Standard Algebraic Notation) in the observations. Default: ``False``.

            .. note:: The `"san"` entry corresponding to `rollout["action"]` will be found in `rollout["next", "san"]`,
                whereas the value at the root `rollout["san"]` will correspond to the value of the san preceding the
                same index action.

        include_fen (bool): Whether to include FEN (Forsyth-Edwards Notation) in the observations. Default: ``False``.
        include_pgn (bool): Whether to include PGN (Portable Game Notation) in the observations. Default: ``False``.
        include_legal_moves (bool): Whether to include legal moves in the observations. Default: ``False``.
        include_hash (bool): Whether to include hash transformations in the environment. Default: ``False``.
        mask_actions (bool): if ``True``, a :class:`~torchrl.envs.ActionMask` transform will be appended
            to the env to make sure that the actions are properly masked. Default: ``True``.
        pixels (bool): Whether to include pixel-based observations of the board. Default: ``False``.

    .. note:: The action spec is a :class:`~torchrl.data.Categorical` with a number of actions equal to the number of possible SAN moves.
        The action space is structured as a categorical distribution over all possible SAN moves, with the legal moves
        being a subset of this space. The environment uses a mask to ensure only legal moves are selected.

    Examples:
        >>> import torch
        >>> from torchrl.envs import ChessEnv
        >>> _ = torch.manual_seed(0)
        >>> env = ChessEnv(include_fen=True, include_san=True, include_pgn=True, include_legal_moves=True)
        >>> print(env)
        TransformedEnv(
            env=ChessEnv(),
            transform=ActionMask(keys=['action', 'action_mask']))
        >>> r = env.reset()
        >>> print(env.rand_step(r))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                action_mask: Tensor(shape=torch.Size([29275]), device=cpu, dtype=torch.bool, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                fen: NonTensorData(data=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1, batch_size=torch.Size([]), device=None),
                legal_moves: Tensor(shape=torch.Size([219]), device=cpu, dtype=torch.int64, is_shared=False),
                next: TensorDict(
                    fields={
                        action_mask: Tensor(shape=torch.Size([29275]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        fen: NonTensorData(data=rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 0 1, batch_size=torch.Size([]), device=None),
                        legal_moves: Tensor(shape=torch.Size([219]), device=cpu, dtype=torch.int64, is_shared=False),
                        pgn: NonTensorData(data=[Event "?"]
                        [Site "?"]
                        [Date "????.??.??"]
                        [Round "?"]
                        [White "?"]
                        [Black "?"]
                        [Result "*"]

                        1. f4 *, batch_size=torch.Size([]), device=None),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                        san: NonTensorData(data=f4, batch_size=torch.Size([]), device=None),
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
                san: NonTensorData(data=<start>, batch_size=torch.Size([]), device=None),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                turn: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(env.rollout(1000))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([96]), device=cpu, dtype=torch.int64, is_shared=False),
                action_mask: Tensor(shape=torch.Size([96, 29275]), device=cpu, dtype=torch.bool, is_shared=False),
                done: Tensor(shape=torch.Size([96, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                fen: NonTensorStack(
                    ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ...,
                    batch_size=torch.Size([96]),
                    device=None),
                legal_moves: Tensor(shape=torch.Size([96, 219]), device=cpu, dtype=torch.int64, is_shared=False),
                next: TensorDict(
                    fields={
                        action_mask: Tensor(shape=torch.Size([96, 29275]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([96, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        fen: NonTensorStack(
                            ['rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b ...,
                            batch_size=torch.Size([96]),
                            device=None),
                        legal_moves: Tensor(shape=torch.Size([96, 219]), device=cpu, dtype=torch.int64, is_shared=False),
                        pgn: NonTensorStack(
                            ['[Event "?"]\n[Site "?"]\n[Date "????.??.??"]\n[R...,
                            batch_size=torch.Size([96]),
                            device=None),
                        reward: Tensor(shape=torch.Size([96, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        san: NonTensorStack(
                            ['Nf3', 'Na6', 'c4', 'f6', 'h4', 'Rb8', 'Na3', 'Ra...,
                            batch_size=torch.Size([96]),
                            device=None),
                        terminated: Tensor(shape=torch.Size([96, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        turn: Tensor(shape=torch.Size([96]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([96]),
                    device=None,
                    is_shared=False),
                pgn: NonTensorStack(
                    ['[Event "?"]\n[Site "?"]\n[Date "????.??.??"]\n[R...,
                    batch_size=torch.Size([96]),
                    device=None),
                san: NonTensorStack(
                    ['<start>', 'Nf3', 'Na6', 'c4', 'f6', 'h4', 'Rb8',...,
                    batch_size=torch.Size([96]),
                    device=None),
                terminated: Tensor(shape=torch.Size([96, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                turn: Tensor(shape=torch.Size([96]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([96]),
            device=None,
            is_shared=False)


    """

    _hash_table: dict[int, str] = {}
    _PGN_RESTART = """[Event "?"]
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
        board: chess.Board | None = None,  # noqa: F821
        return_mask: bool = False,
        pad: bool = False,
    ) -> torch.Tensor:
        if not self.stateful:
            if tensordict is None:
                # trust the board
                pass
            elif self.include_fen:
                fen = tensordict.get("fen", None)
                fen = fen.data
                self.board.set_fen(fen)
                board = self.board
            elif self.include_pgn:
                pgn = tensordict.get("pgn")
                pgn = pgn.data
                board = self._pgn_to_board(pgn, self.board)

        if board is None:
            board = self.board

        indices = torch.tensor(
            [self._san_moves.index(board.san(m)) for m in board.legal_moves],
            dtype=torch.int64,
        )
        mask = None
        if return_mask:
            mask = self._move_index_to_mask(indices)
        if pad:
            indices = torch.nn.functional.pad(
                indices, [0, 218 - indices.numel() + 1], value=len(self.san_moves)
            )
        if return_mask:
            return indices, mask
        return indices

    @classmethod
    def _move_index_to_mask(cls, indices: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(cls.san_moves), dtype=torch.bool).index_fill_(
            0, indices, True
        )

    def __init__(
        self,
        *,
        stateful: bool = True,
        include_san: bool = False,
        include_fen: bool = False,
        include_pgn: bool = False,
        include_legal_moves: bool = False,
        include_hash: bool = False,
        include_hash_inv: bool = False,
        mask_actions: bool = True,
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
        self.mask_actions = mask_actions
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
                shape=(), example_data=self._PGN_RESTART
            )
        if include_fen:
            self.full_observation_spec["fen"] = NonTensor(shape=(), example_data="any")
        if not stateful and not (include_pgn or include_fen):
            raise RuntimeError(
                "At least one state representation (pgn or fen) must be enabled when stateful "
                f"is {stateful}."
            )

        self.stateful = stateful

        # state_spec is loosely defined as such - it's not really an issue that extra keys
        # can go missing but it allows us to reset the env using fen passed to the reset
        # method.
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
            self.full_observation_spec["pixels"] = Unbounded(
                shape=(3, 390, 390), dtype=torch.uint8
            )

        self.full_action_spec = Composite(
            action=Categorical(n=len(self.san_moves), shape=(), dtype=torch.int64)
        )
        self.full_reward_spec = Composite(
            reward=Unbounded(shape=(1,), dtype=torch.float32)
        )
        if self.mask_actions:
            self.full_observation_spec["action_mask"] = Binary(
                n=len(self.san_moves), dtype=torch.bool
            )

        # done spec generated automatically
        self.board = chess.Board()
        if self.stateful:
            self.action_spec.set_provisional_n(len(list(self.board.legal_moves)))

    def _is_done(self, board):
        return board.is_game_over() | board.is_fifty_moves()

    def all_actions(self, tensordict: TensorDictBase | None = None) -> TensorDictBase:
        if not self.mask_actions:
            raise RuntimeError(
                "Cannot generate legal actions since 'mask_actions=False' was "
                "set. If you really want to generate all actions, not just "
                "legal ones, call 'env.full_action_spec.enumerate()'."
            )
        return super().all_actions(tensordict)

    def _reset(self, tensordict=None):
        fen = None
        pgn = None
        if tensordict is not None:
            dest = tensordict.empty()
            if self.include_fen:
                fen = tensordict.get("fen", None)
                if fen is not None:
                    fen = fen.data
            elif self.include_pgn:
                pgn = tensordict.get("pgn", None)
                if pgn is not None:
                    pgn = pgn.data
        else:
            dest = TensorDict()

        if fen is None and pgn is None:
            self.board.reset()
        elif fen is not None:
            self.board.set_fen(fen)
            if self._is_done(self.board):
                raise ValueError(
                    "Cannot reset to a fen that is a gameover state." f" fen: {fen}"
                )
        elif pgn is not None:
            self.board = self._pgn_to_board(pgn)

        if self.include_fen and fen is None:
            fen = self.board.fen()
        if self.include_pgn and pgn is None:
            pgn = self._board_to_pgn(self.board)

        turn = self.board.turn
        if self.include_san:
            if self.board.move_stack:
                move = self.board.peek()
            else:
                move = None
            if move is None:
                dest.set("san", "<start>")
            else:
                dest.set("san", self.board.san(move))
        if self.include_fen:
            dest.set("fen", fen)
        if self.include_pgn:
            dest.set("pgn", pgn)
        dest.set("turn", turn)
        if self.include_legal_moves:
            moves_idx = self._legal_moves_to_index(
                board=self.board, pad=True, return_mask=self.mask_actions
            )
            if self.mask_actions:
                moves_idx, mask = moves_idx
                dest.set("action_mask", mask)
            dest.set("legal_moves", moves_idx)
        elif self.mask_actions:
            dest.set(
                "action_mask",
                self._legal_moves_to_index(
                    board=self.board, pad=True, return_mask=True
                )[1],
            )

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
            from PIL import Image

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
                "Chess rendering requires cairosvg, PIL and torchvision to be installed."
            )
        return img

    @classmethod
    def _pgn_to_board(
        cls, pgn_string: str, board: chess.Board | None = None  # noqa: F821
    ) -> chess.Board:  # noqa: F821
        pgn_io = io.StringIO(pgn_string)
        game = cls.lib.pgn.read_game(pgn_io)
        if board is None:
            board = cls.lib.Board()
        else:
            board.reset()
        for move in game.mainline_moves():
            board.push(move)
        return board

    @classmethod
    def _add_move_to_pgn(cls, pgn_string: str, move: chess.Move) -> str:  # noqa: F821
        pgn_io = io.StringIO(pgn_string)
        game = cls.lib.pgn.read_game(pgn_io)
        if game is None:
            raise ValueError("Invalid PGN string")
        game.end().add_variation(move)
        return str(game)

    @classmethod
    def _board_to_pgn(cls, board: chess.Board) -> str:  # noqa: F821
        game = cls.lib.pgn.Game.from_board(board)
        pgn_string = str(game)
        return pgn_string

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
            fen = tensordict.get("fen").data
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

        pgn = None
        fen = None
        if not self.stateful:
            if self.include_fen:
                fen = tensordict.get("fen").data
                board.set_fen(fen)
            elif self.include_pgn:
                pgn = tensordict.get("pgn").data
                board = self._pgn_to_board(pgn, board)
            else:
                raise RuntimeError(
                    "Not enough information to deduce the board. If stateful=False, include_pgn or include_fen must be True."
                )

        san = self.san_moves[action]
        board.push_san(san)

        dest = tensordict.empty()

        # Collect data
        if self.include_fen:
            fen = board.fen()
            dest.set("fen", fen)

        if self.include_pgn:
            if pgn is not None:
                pgn = self._add_move_to_pgn(pgn, board.move_stack[-1])
            else:
                pgn = self._board_to_pgn(board)
            dest.set("pgn", pgn)

        if self.include_san:
            dest.set("san", san)

        if self.include_legal_moves:
            moves_idx = self._legal_moves_to_index(
                board=board, pad=True, return_mask=self.mask_actions
            )
            if self.mask_actions:
                moves_idx, mask = moves_idx
                dest.set("action_mask", mask)
            dest.set("legal_moves", moves_idx)
        elif self.mask_actions:
            dest.set(
                "action_mask",
                self._legal_moves_to_index(
                    board=self.board, pad=True, return_mask=True
                )[1],
            )

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
        dest.set("done", torch.tensor([done]))
        dest.set("terminated", torch.tensor([done]))
        if self.pixels:
            dest.set("pixels", self._get_tensor_image(board=self.board))
        return dest

    def _set_seed(self, *args, **kwargs) -> None:
        ...

    def cardinality(self, tensordict: TensorDictBase | None = None) -> int:
        self._set_action_space(tensordict)
        return self.action_spec.cardinality()
