# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Private GTrXL candidate for the recurrent-state container comparison.

The public state class is deliberately undecided. This reference uses a
sequential window implementation with gradients through the current window.
See Parisotto et al., https://proceedings.mlr.press/v119/parisotto20a.html.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from torchrl.data import Binary, Composite, Unbounded


class _GRUGate(nn.Module):
    def __init__(self, width, bias, *, device=None, dtype=None):
        super().__init__()
        self.x_rz = nn.Linear(width, 2 * width, bias=False, device=device, dtype=dtype)
        self.y_rzh = nn.Linear(width, 3 * width, bias=False, device=device, dtype=dtype)
        self.x_h = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.bias = nn.Parameter(torch.full((width,), bias, device=device, dtype=dtype))

    def forward(self, x, y):
        xr, xz = self.x_rz(x).chunk(2, -1)
        yr, yz, yh = self.y_rzh(y).chunk(3, -1)
        reset = (xr + yr).sigmoid()
        update = (xz + yz - self.bias).sigmoid()
        candidate = (yh + self.x_h(reset * x)).tanh()
        return (1 - update) * x + update * candidate


class _GTrXLBlock(nn.Module):
    def __init__(
        self, width, heads, memory_len, gate_bias, dropout, *, device=None, dtype=None
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.q = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.kv = nn.Linear(width, 2 * width, bias=False, device=device, dtype=dtype)
        self.relative = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.content_bias = nn.Parameter(
            torch.zeros(heads, self.head_dim, device=device, dtype=dtype)
        )
        self.position_bias = nn.Parameter(
            torch.zeros(heads, self.head_dim, device=device, dtype=dtype)
        )
        self.out = nn.Linear(width, width, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(4 * width, width, device=device, dtype=dtype),
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_gate = _GRUGate(width, gate_bias, device=device, dtype=dtype)
        self.mlp_gate = _GRUGate(width, gate_bias, device=device, dtype=dtype)
        distance = torch.arange(
            memory_len, -1, -1, device=device, dtype=dtype or torch.float32
        )
        frequency = torch.exp(
            -math.log(10000)
            * torch.arange(0, width, 2, device=device, dtype=dtype or torch.float32)
            / width
        )
        angle = distance[:, None] * frequency[None]
        self.register_buffer("positions", torch.cat((angle.sin(), angle.cos()), -1))

    def forward(self, current, memory, valid):
        batch, width = current.shape
        context = self.norm1(torch.cat((memory, current.unsqueeze(1)), 1))
        q = self.q(context[:, -1]).reshape(batch, self.heads, self.head_dim)
        k, v = self.kv(context).chunk(2, -1)
        k = k.reshape(batch, -1, self.heads, self.head_dim)
        v = v.reshape(batch, -1, self.heads, self.head_dim)
        relative = self.relative(self.positions).reshape(-1, self.heads, self.head_dim)
        scores = torch.einsum("bhd,bmhd->bhm", q + self.content_bias, k)
        scores = scores + torch.einsum("bhd,mhd->bhm", q + self.position_bias, relative)
        scores = scores / math.sqrt(self.head_dim)
        valid = torch.cat((valid, torch.ones_like(valid[:, :1])), 1)
        weights = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        attention = torch.einsum("bhm,bmhd->bhd", self.dropout(weights), v)
        current = self.attention_gate(
            current, self.dropout(F.relu(self.out(attention.reshape(batch, width))))
        )
        return self.mlp_gate(
            current, self.dropout(F.relu(self.mlp(self.norm2(current))))
        )


class _GTrXL(nn.Module):
    """Explicit layer-memory backbone; not exported until the comparison is reviewed."""

    def __init__(
        self,
        input_size,
        hidden_size=32,
        num_layers=2,
        *,
        num_heads=4,
        memory_len=16,
        state_cls=TensorDict,
        gate_bias=2.0,
        dropout=0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if hidden_size % num_heads or hidden_size % 2:
            raise ValueError("hidden_size must be even and divisible by num_heads")
        if memory_len < 1 or num_layers < 1:
            raise ValueError("memory_len and num_layers must be positive")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.memory_len = memory_len
        self.state_cls = state_cls
        self.embedding = nn.Linear(input_size, hidden_size, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            _GTrXLBlock(
                hidden_size,
                num_heads,
                memory_len,
                gate_bias,
                dropout,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )

    @property
    def state_spec(self):
        weight = self.embedding.weight
        return Composite(
            memory=Unbounded(
                shape=(self.num_layers, self.memory_len, self.hidden_size),
                dtype=weight.dtype,
                device=weight.device,
            ),
            valid=Binary(
                shape=(self.memory_len,), dtype=torch.bool, device=weight.device
            ),
            data_cls=self.state_cls,
            device=weight.device,
        )

    def forward_state(self, features, state, is_init, *, recurrent):
        # The wrapper flattens environment batch dimensions, keeping time last.
        stored_memory = state.get("memory").detach()
        stored_valid = state.get("valid")
        memory = stored_memory[:, 0]
        valid = stored_valid[:, 0]
        outputs, memories, validity = [], [], []
        for step in range(features.shape[1]):
            init = is_init[:, step]
            reset_memory = (
                stored_memory[:, step] if recurrent else torch.zeros_like(memory)
            )
            reset_valid = (
                stored_valid[:, step] if recurrent else torch.zeros_like(valid)
            )
            memory = torch.where(init[:, None, None, None], reset_memory, memory)
            valid = torch.where(init[:, None], reset_valid, valid)
            value = self.embedding(features[:, step])
            layer_inputs = []
            for layer, block in enumerate(self.blocks):
                layer_inputs.append(value)
                value = block(value, memory[:, layer], valid)
            # Only layer inputs are needed by the next step's attention.
            memory = torch.cat(
                (memory[:, :, 1:], torch.stack(layer_inputs, 1).unsqueeze(2)), 2
            )
            valid = torch.cat((valid[:, 1:], torch.ones_like(valid[:, :1])), 1)
            outputs.append(value)
            memories.append(memory.detach())
            validity.append(valid)
        next_state = self.state_cls.from_dict(
            {"memory": torch.stack(memories, 1), "valid": torch.stack(validity, 1)},
            batch_size=features.shape[:2],
            device=features.device,
        )
        return torch.stack(outputs, 1), next_state
