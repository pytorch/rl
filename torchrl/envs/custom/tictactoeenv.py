# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs.common import EnvBase


class TicTacToeEnv(EnvBase):
    """A Tic-Tac-Toe implementation.

    Keyword Args:
        single_player (bool, optional): whether one or two players have to be
            accounted for. ``single_player=True`` means that ``"player1"`` is
            playing randomly. If ``False`` (default), at each turn,
            one of the two players has to play.
        device (torch.device, optional): the device where to put the tensors.
            Defaults to ``None`` (default device).

    The environment is stateless. To run it across multiple batches, call

        >>> env.reset(TensorDict(batch_size=desired_batch_size))

    If the ``"mask"`` entry is present, ``rand_action`` takes it into account to
    generate the next action. Any policy executed on this env should take this
    mask into account, as well as the turn of the player (stored in the ``"turn"``
    output entry).

    Specs:
        >>> print(env.specs)
        Composite(
            output_spec: Composite(
                full_observation_spec: Composite(
                    board: Categorical(
                        shape=torch.Size([3, 3]),
                        space=DiscreteBox(n=2),
                        dtype=torch.int32,
                        domain=discrete),
                    turn: Categorical(
                        shape=torch.Size([1]),
                        space=DiscreteBox(n=2),
                        dtype=torch.int32,
                        domain=discrete),
                    mask: Categorical(
                        shape=torch.Size([9]),
                        space=DiscreteBox(n=2),
                        dtype=torch.bool,
                        domain=discrete),
                    shape=torch.Size([])),
                full_reward_spec: Composite(
                    player0: Composite(
                        reward: UnboundedContinuous(
                            shape=torch.Size([1]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                            dtype=torch.float32,
                            domain=continuous),
                        shape=torch.Size([])),
                    player1: Composite(
                        reward: UnboundedContinuous(
                            shape=torch.Size([1]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                            dtype=torch.float32,
                            domain=continuous),
                        shape=torch.Size([])),
                    shape=torch.Size([])),
                full_done_spec: Composite(
                    done: Categorical(
                        shape=torch.Size([1]),
                        space=DiscreteBox(n=2),
                        dtype=torch.bool,
                        domain=discrete),
                    terminated: Categorical(
                        shape=torch.Size([1]),
                        space=DiscreteBox(n=2),
                        dtype=torch.bool,
                        domain=discrete),
                    truncated: Categorical(
                        shape=torch.Size([1]),
                        space=DiscreteBox(n=2),
                        dtype=torch.bool,
                        domain=discrete),
                    shape=torch.Size([])),
                shape=torch.Size([])),
            input_spec: Composite(
                full_state_spec: Composite(
                    board: Categorical(
                        shape=torch.Size([3, 3]),
                        space=DiscreteBox(n=2),
                        dtype=torch.int32,
                        domain=discrete),
                    turn: Categorical(
                        shape=torch.Size([1]),
                        space=DiscreteBox(n=2),
                        dtype=torch.int32,
                        domain=discrete),
                    mask: Categorical(
                        shape=torch.Size([9]),
                        space=DiscreteBox(n=2),
                        dtype=torch.bool,
                        domain=discrete), shape=torch.Size([])),
                full_action_spec: Composite(
                    action: Categorical(
                        shape=torch.Size([1]),
                        space=DiscreteBox(n=9),
                        dtype=torch.int64,
                        domain=discrete),
                    shape=torch.Size([])),
                shape=torch.Size([])),
            shape=torch.Size([]))

    To run a dummy rollout, execute the following command:

    Examples:
        >>> env = TicTacToeEnv()
        >>> env.rollout(10)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                board: Tensor(shape=torch.Size([9, 3, 3]), device=cpu, dtype=torch.int32, is_shared=False),
                done: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                mask: Tensor(shape=torch.Size([9, 9]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        board: Tensor(shape=torch.Size([9, 3, 3]), device=cpu, dtype=torch.int32, is_shared=False),
                        done: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        mask: Tensor(shape=torch.Size([9, 9]), device=cpu, dtype=torch.bool, is_shared=False),
                        player0: TensorDict(
                            fields={
                                reward: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([9]),
                            device=None,
                            is_shared=False),
                        player1: TensorDict(
                            fields={
                                reward: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([9]),
                            device=None,
                            is_shared=False),
                        terminated: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        turn: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.int32, is_shared=False)},
                    batch_size=torch.Size([9]),
                    device=None,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                turn: Tensor(shape=torch.Size([9, 1]), device=cpu, dtype=torch.int32, is_shared=False)},
            batch_size=torch.Size([9]),
            device=None,
            is_shared=False)

    """

    # batch_locked is set to False since various batch sizes can be provided to the env
    batch_locked: bool = False

    def __init__(self, *, single_player: bool = False, device=None):
        super().__init__(device=device)
        self.single_player = single_player
        self.action_spec: Unbounded = Categorical(
            n=9,
            shape=(),
            device=device,
        )

        self.full_observation_spec: Composite = Composite(
            board=Unbounded(shape=(3, 3), dtype=torch.int, device=device),
            turn=Categorical(
                2,
                shape=(1,),
                dtype=torch.int,
                device=device,
            ),
            mask=Categorical(
                2,
                shape=(9,),
                dtype=torch.bool,
                device=device,
            ),
            device=device,
        )
        self.state_spec: Composite = self.observation_spec.clone()

        self.reward_spec: Unbounded = Composite(
            {
                ("player0", "reward"): Unbounded(shape=(1,), device=device),
                ("player1", "reward"): Unbounded(shape=(1,), device=device),
            },
            device=device,
        )

        self.full_done_spec: Categorical = Composite(
            done=Categorical(2, shape=(1,), dtype=torch.bool, device=device),
            device=device,
        )
        self.full_done_spec["terminated"] = self.full_done_spec["done"].clone()
        self.full_done_spec["truncated"] = self.full_done_spec["done"].clone()

    def _reset(self, reset_td: TensorDict) -> TensorDict:
        shape = reset_td.shape if reset_td is not None else ()
        state = self.state_spec.zero(shape)
        state["board"] -= 1
        state["mask"].fill_(True)
        return state.update(self.full_done_spec.zero(shape))

    def _step(self, state: TensorDict) -> TensorDict:
        board = state["board"].clone()
        turn = state["turn"].clone()
        action = state["action"]
        board.flatten(-2, -1).scatter_(index=action.unsqueeze(-1), dim=-1, value=1)
        wins = self.win(board, action)

        mask = board.flatten(-2, -1) == -1
        done = wins | ~mask.any(-1, keepdim=True)
        terminated = done.clone()

        reward_0 = wins & (turn == 0)
        reward_1 = wins & (turn == 1)

        state = TensorDict(
            {
                "done": done,
                "terminated": terminated,
                ("player0", "reward"): reward_0.float(),
                ("player1", "reward"): reward_1.float(),
                "board": torch.where(board == -1, board, 1 - board),
                "turn": 1 - turn,
                "mask": mask,
            },
            batch_size=state.batch_size,
        )
        if self.single_player:
            select = (~done & (turn == 0)).squeeze(-1)
            if select.all():
                state_select = state
            elif select.any():
                state_select = state[select]
            else:
                return state
            state_select = self._step(self.rand_action(state_select))
            if select.all():
                return state_select
            return torch.where(done, state, state_select)
        return state

    def _set_seed(self, seed: int | None):
        ...

    @staticmethod
    def win(board: torch.Tensor, action: torch.Tensor):
        row = action // 3  # type: ignore
        col = action % 3  # type: ignore
        if board[..., row, :].sum() == 3:
            return True
        if board[..., col].sum() == 3:
            return True
        if board.diagonal(0, -2, -1).sum() == 3:
            return True
        if board.flip(-1).diagonal(0, -2, -1).sum() == 3:
            return True
        return False

    @staticmethod
    def full(board: torch.Tensor) -> bool:
        return torch.sym_int(board.abs().sum()) == 9

    @staticmethod
    def get_action_mask():
        pass

    def rand_action(self, tensordict: Optional[TensorDictBase] = None):
        mask = tensordict.get("mask")
        action_spec = self.action_spec
        if tensordict.ndim:
            action_spec = action_spec.expand(tensordict.shape)
        else:
            action_spec = action_spec.clone()
        action_spec.update_mask(mask)
        tensordict.set(self.action_key, action_spec.rand())
        return tensordict
