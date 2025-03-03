# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase


# TODO: code small architecture ref in Impala paper


class _ResNetBlock(nn.Module):
    def __init__(
        self,
        num_ch,
    ):
        super().__init__()
        resnet_block = []
        resnet_block.append(nn.ReLU(inplace=True))
        resnet_block.append(
            nn.LazyConv2d(
                out_channels=num_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        resnet_block.append(nn.ReLU(inplace=True))
        resnet_block.append(
            nn.Conv2d(
                in_channels=num_ch,
                out_channels=num_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.seq = nn.Sequential(*resnet_block)

    def forward(self, x):
        x += self.seq(x)
        return x


class _ConvNetBlock(nn.Module):
    def __init__(self, num_ch):
        super().__init__()

        conv = nn.LazyConv2d(
            out_channels=num_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.feats_conv = nn.Sequential(conv, mp)
        self.resnet1 = _ResNetBlock(num_ch=num_ch)
        self.resnet2 = _ResNetBlock(num_ch=num_ch)

    def forward(self, x):
        x = self.feats_conv(x)
        x = self.resnet1(x)
        x = self.resnet1(x)
        return x


class ImpalaNet(nn.Module):  # noqa: D101
    def __init__(
        self,
        num_actions,
        channels=(16, 32, 32),
        out_features=256,
        use_lstm=False,
        batch_first=True,
        clamp_reward=True,
        one_hot=False,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.use_lstm = use_lstm
        self.clamp_reward = clamp_reward
        self.one_hot = one_hot
        self.num_actions = num_actions

        layers = [_ConvNetBlock(num_ch) for num_ch in channels]
        layers += [nn.ReLU(inplace=True)]
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.LazyLinear(out_features), nn.ReLU(inplace=True))

        # FC output size + last reward.
        core_output_size = out_features + 1

        if use_lstm:
            self.core = nn.LSTM(
                core_output_size,
                out_features,
                num_layers=1,
                batch_first=batch_first,
            )
            core_output_size = out_features

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def forward(self, x, reward, done, core_state=None, mask=None):  # noqa: D102
        if self.batch_first:
            B, T, *x_shape = x.shape
            batch_shape = torch.Size([B, T])
        else:
            T, B, *x_shape = x.shape
            batch_shape = torch.Size([T, B])
        if mask is None:
            x = x.view(-1, *x.shape[-3:])
        else:
            x = x[mask]
            if x.ndimension() != 4:
                raise RuntimeError(
                    f"masked input should have 4 dimensions but got {x.ndimension()} instead"
                )
        x = self.convs(x)
        x = x.view(B * T, -1)
        x = self.fc(x)

        if mask is None:
            if self.batch_first:
                x = x.view(B, T, -1)
            else:
                x = x.view(T, B, -1)
        else:
            x = self._allocate_masked_x(x, mask)

        if self.clamp_reward:
            reward = torch.clamp(reward, -1, 1)
        reward = reward.unsqueeze(-1)

        core_input = torch.cat([x, reward], dim=-1)

        if self.use_lstm:
            core_output, _ = self.core(core_input, core_state)
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        softmax_vals = F.softmax(policy_logits, dim=-1)
        action = torch.multinomial(
            softmax_vals.view(-1, softmax_vals.shape[-1]), num_samples=1
        ).view(softmax_vals.shape[:-1])
        if self.one_hot:
            action = F.one_hot(action, policy_logits.shape[-1])

        if policy_logits.shape[:2] != batch_shape:
            raise RuntimeError("policy_logits and batch-shape mismatch")
        if baseline.shape[:2] != batch_shape:
            raise RuntimeError("baseline and batch-shape mismatch")
        if action.shape[:2] != batch_shape:
            raise RuntimeError("action and batch-shape mismatch")

        return (action, policy_logits, baseline), core_state

    def _allocate_masked_x(self, x, mask):
        x_empty = torch.zeros(
            *mask.shape[:2], x.shape[-1], device=x.device, dtype=x.dtype
        )
        x_empty[mask] = x
        return x_empty


class ImpalaNetTensorDict(ImpalaNet):  # noqa: D101
    observation_key = "pixels"

    def forward(self, tensordict: TensorDictBase):  # noqa: D102
        x = tensordict.get(self.observation_key)
        done = tensordict.get("done").squeeze(-1)
        reward = tensordict.get("reward").squeeze(-1)
        mask = tensordict.get(("collector", "mask"))
        core_state = (
            tensordict.get("core_state") if "core_state" in tensordict.keys() else None
        )
        return super().forward(x, reward, done, core_state=core_state, mask=mask)
