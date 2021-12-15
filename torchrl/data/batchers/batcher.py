from __future__ import annotations

from torch import nn

from torchrl.data.batchers.utils import expand_as_right
from torchrl.data.tensordict.tensordict import _TensorDict

from typing import Union
from numbers import Number

import torch

__all__ = ["MultiStep"]


def _conv1d(reward: torch.Tensor, gammas: torch.Tensor, n_steps_max: int):
    assert (
            reward.ndimension() == 3 and reward.shape[-1] == 1
    ), "Expected a B x T x 1 reward tensor"
    reward_pad = torch.nn.functional.pad(
        reward, [0, 0, 0, n_steps_max]
    ).transpose(-1, -2)
    reward_pad = torch.conv1d(reward_pad, gammas).transpose(-1, -2)
    return reward_pad

def _get_terminal(done, n_steps_max):
    # terminal states (done or last)
    terminal = done.clone()
    terminal[:, -1] = done[:, -1] | (done.sum(1) != 1)
    assert (terminal.sum(1) == 1).all()
    post_terminal = terminal.cumsum(1).cumsum(1) >= 2
    post_terminal = torch.cat(
        [
            post_terminal,
            torch.ones(
                post_terminal.shape[0], n_steps_max, *post_terminal.shape[2:],
                device=post_terminal.device,
                dtype=torch.bool
            ),
        ],
        1,
    )
    return terminal, post_terminal

def _get_gamma(gamma: Number, reward: torch.Tensor, mask: torch.Tensor, n_steps_max: int):
    # Compute gamma for n-step value function
    gamma_masked = gamma * torch.ones_like(reward)
    gamma_masked = gamma_masked.masked_fill_(~mask, 1.0)
    gamma_masked = torch.nn.functional.pad(
        gamma_masked, [0, 0, 0, n_steps_max], value=1.0
    )
    gamma_masked = gamma_masked.unfold(1, n_steps_max + 1, 1)
    gamma_masked = gamma_masked.flip(1).cumprod(-1).flip(1)
    return gamma_masked[..., -1]


def _get_steps_to_next_obs(nonterminal: torch.Tensor, n_steps_max: int):
    steps_to_next_obs = nonterminal.flip(1).cumsum(1).flip(1)
    steps_to_next_obs.clamp_max_(n_steps_max + 1)
    return steps_to_next_obs

def select_and_repeat(tensor: torch.Tensor, terminal: torch.Tensor, post_terminal: torch.Tensor, mask: torch.Tensor,
                      n_steps_max: int):
    T = tensor.shape[1]
    terminal = expand_as_right(terminal.squeeze(-1), tensor)
    last_tensor = (terminal * tensor).sum(1, True)

    last_tensor = last_tensor.expand(last_tensor.shape[0], post_terminal.shape[1], *last_tensor.shape[2:])
    post_terminal = expand_as_right(post_terminal.squeeze(-1), last_tensor)
    post_terminal_tensor = last_tensor * post_terminal

    tensor_repeat = torch.zeros(
        tensor.shape[0],
        n_steps_max,
        *tensor.shape[2:],
        device=tensor.device,
        dtype=tensor.dtype
    )
    tensor_cat = torch.cat([tensor, tensor_repeat], 1) + post_terminal_tensor
    tensor_cat = tensor_cat[:, -T:]
    mask = expand_as_right(mask.squeeze(-1), tensor_cat)
    return tensor_cat.masked_fill(~mask, 0.0)


class MultiStep(nn.Module):
    def __init__(self, gamma: Number, n_steps_max: int, device: Union[int, torch.device, str] = "cpu"):
        super().__init__()
        assert n_steps_max >= 0
        assert 0 < gamma
        assert gamma <= 1.0

        self.gamma = gamma
        self.n_steps_max = n_steps_max
        self.register_buffer('gammas', torch.tensor(
            [gamma ** i for i in range(n_steps_max + 1)], dtype=torch.float,
        ).reshape(1, 1, -1))
        self.device = torch.device(device)
        if self.device != torch.device("cpu"):
            self.to(self.device)

    def forward(self, tensor_dict: _TensorDict):
        assert (
                tensor_dict.batch_dims == 2
        ), "Expected a tensordict with B x T x ... dimensions"

        done = tensor_dict.get("done")
        try:
            mask = tensor_dict.get("mask")
        except KeyError:
            mask = done.clone().flip(1).cumsum(1).flip(1).to(torch.bool)
        reward = tensor_dict.get("reward")
        b, T, *_ = mask.shape

        terminal, post_terminal = _get_terminal(done, self.n_steps_max)

        # Compute gamma for n-step value function
        gamma_masked = _get_gamma(self.gamma, reward, mask, self.n_steps_max)

        # step_to_next_state
        nonterminal = ~post_terminal[:, :T]
        steps_to_next_obs = _get_steps_to_next_obs(nonterminal, self.n_steps_max)

        # Discounted summed reward
        partial_return = _conv1d(reward, self.gammas, self.n_steps_max)

        selected_td = tensor_dict.select(*[key for key in tensor_dict.keys() if (key.startswith("next_") or key=="done")])

        for key, item in selected_td.items():
            tensor_dict.set_(
                key, select_and_repeat(
                    item,
                    terminal,
                    post_terminal,
                    mask,
                    self.n_steps_max,
                ))

        tensor_dict.set("gamma", gamma_masked)
        tensor_dict.set("steps_to_next_obs", steps_to_next_obs)
        tensor_dict.set("nonterminal", nonterminal)
        tensor_dict.set("partial_return", partial_return)

        tensor_dict.set_("done", done)
        return tensor_dict


