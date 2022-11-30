# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Tuple

import torch
from tensordict.tensordict import TensorDictBase
from tensordict.utils import expand_as_right
from torch import nn
from torch.nn import functional as F


def _conv1d_reward(
    reward: torch.Tensor, gammas: torch.Tensor, n_steps_max: int
) -> torch.Tensor:
    if not (reward.ndimension() == 3 and reward.shape[-1] == 1):
        raise RuntimeError(
            f"Expected a B x T x 1 reward tensor, got reward.shape = {reward.shape}"
        )
    reward_pad = F.pad(reward, [0, 0, 0, n_steps_max])
    reward_pad_transpose = reward_pad.transpose(-1, -2)
    partial_return_transpose = torch.conv1d(reward_pad_transpose, gammas)
    partial_return = partial_return_transpose.transpose(-1, -2)
    return partial_return


def _get_terminal(
    done: torch.Tensor, n_steps_max: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # terminal states (done or last)
    terminal = done.clone()
    terminal[:, -1] = done[:, -1] | (done.sum(1) != 1)
    if not (terminal.sum(1) == 1).all():
        raise RuntimeError("Got more or less than one terminal state per episode.")
    post_terminal = terminal.cumsum(1).cumsum(1) >= 2
    post_terminal = torch.cat(
        [
            post_terminal,
            torch.ones(
                post_terminal.shape[0],
                n_steps_max,
                *post_terminal.shape[2:],
                device=post_terminal.device,
                dtype=torch.bool,
            ),
        ],
        1,
    )
    return terminal, post_terminal


def _get_gamma(
    gamma: float, reward: torch.Tensor, mask: torch.Tensor, n_steps_max: int
) -> torch.Tensor:
    # Compute gamma for n-step value function
    gamma_masked = gamma * torch.ones_like(reward)
    gamma_masked = gamma_masked.masked_fill_(~mask, 1.0)
    gamma_masked = torch.nn.functional.pad(
        gamma_masked, [0, 0, 0, n_steps_max], value=1.0
    )
    gamma_masked = gamma_masked.unfold(1, n_steps_max + 1, 1)
    gamma_masked = gamma_masked.flip(1).cumprod(-1).flip(1)
    return gamma_masked[..., -1]


def _get_steps_to_next_obs(nonterminal: torch.Tensor, n_steps_max: int) -> torch.Tensor:
    steps_to_next_obs = nonterminal.flip(1).cumsum(1).flip(1)
    steps_to_next_obs.clamp_max_(n_steps_max + 1)
    return steps_to_next_obs


def _select_and_repeat(
    tensor: torch.Tensor,
    terminal: torch.Tensor,
    post_terminal: torch.Tensor,
    mask: torch.Tensor,
    n_steps_max: int,
) -> torch.Tensor:
    T = tensor.shape[1]
    terminal = expand_as_right(terminal.squeeze(-1), tensor)
    last_tensor = (terminal * tensor).sum(1, True)

    last_tensor = last_tensor.expand(
        last_tensor.shape[0], post_terminal.shape[1], *last_tensor.shape[2:]
    )
    post_terminal = expand_as_right(post_terminal.squeeze(-1), last_tensor)
    post_terminal_tensor = last_tensor * post_terminal

    tensor_repeat = torch.zeros(
        tensor.shape[0],
        n_steps_max,
        *tensor.shape[2:],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    tensor_cat = torch.cat([tensor, tensor_repeat], 1) + post_terminal_tensor
    tensor_cat = tensor_cat[:, -T:]
    mask = expand_as_right(mask.squeeze(-1), tensor_cat)
    return tensor_cat.masked_fill(~mask, 0.0)


class MultiStep(nn.Module):
    """Multistep reward transform.

    Presented in 'Sutton, R. S. 1988. Learning to
    predict by the methods of temporal differences. Machine learning 3(
    1):9–44.'

    Args:
        gamma (float): Discount factor for return computation
        n_steps_max (integer): maximum look-ahead steps.

    """

    def __init__(
        self,
        gamma: float,
        n_steps_max: int,
    ):
        super().__init__()
        if n_steps_max < 0:
            raise ValueError("n_steps_max must be a null or positive integer")
        if not (gamma > 0 and gamma <= 1):
            raise ValueError(f"got out-of-bounds gamma decay: gamma={gamma}")

        self.gamma = gamma
        self.n_steps_max = n_steps_max
        self.register_buffer(
            "gammas",
            torch.tensor(
                [gamma**i for i in range(n_steps_max + 1)],
                dtype=torch.float,
            ).reshape(1, 1, -1),
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Re-writes a tensordict following the multi-step transform.

        Args:
            tensordict: TennsorDict instance with Batch x Time-steps x ...
                dimensions.
                The TensorDict must contain a "reward" and "done" key. All
                keys that are contained within the "next" nested tensordict
                will be shifted by (at most) :obj:`MultiStep.n_steps_max` frames.
                The TensorDict will also be updated with new key-value pairs:

                - gamma: indicating the discount to be used for the next
                reward;

                - nonterminal: boolean value indicating whether a step is
                non-terminal (not done or not last of trajectory);

                - original_reward: previous reward collected in the
                environment (i.e. before multi-step);

                - The "reward" values will be replaced by the newly computed
                rewards.


        Returns:
            in-place transformation of the input tensordict.

        """
        if tensordict.batch_dims != 2:
            raise RuntimeError("Expected a tensordict with B x T x ... dimensions")

        done = tensordict.get("done")
        if "mask" in tensordict.keys():
            mask = tensordict.get("mask")
        else:
            mask = done.clone().flip(1).cumsum(1).flip(1).to(torch.bool)
        reward = tensordict.get("reward")

        b, T, *_ = mask.shape

        terminal, post_terminal = _get_terminal(done, self.n_steps_max)

        # Compute gamma for n-step value function
        gamma_masked = _get_gamma(self.gamma, reward, mask, self.n_steps_max)

        # Discounted summed reward
        partial_return = _conv1d_reward(reward, self.gammas, self.n_steps_max)

        # step_to_next_state
        nonterminal = ~post_terminal[:, :T]
        steps_to_next_obs = _get_steps_to_next_obs(nonterminal, self.n_steps_max)

        selected_td = tensordict.select("next", "done")

        def _select_and_repeat_local(item):
            return _select_and_repeat(
                item,
                terminal,
                post_terminal,
                mask,
                self.n_steps_max,
            )

        selected_td.apply_(_select_and_repeat_local)

        tensordict.set("gamma", gamma_masked)
        tensordict.set("steps_to_next_obs", steps_to_next_obs)
        tensordict.set("nonterminal", nonterminal)
        tensordict.rename_key("reward", "original_reward")
        tensordict.set("reward", partial_return)

        tensordict.set_("done", done)
        return tensordict
