# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Gated Transformer-XL with caller-owned recurrent layer memory."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tensordict import TensorClass, TensorDict, TensorDictBase
from torch import nn

from torchrl.data.tensor_specs import Binary, Composite, Unbounded


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

    def forward_window(self, current, memory, mask, relative_index):
        # Project the M historical inputs once, then all T queries in parallel.
        batch, length, width = current.shape
        context = self.norm1(torch.cat((memory, current), 1))
        q = self.q(context[:, -length:]).reshape(
            batch, length, self.heads, self.head_dim
        )
        k, v = self.kv(context).chunk(2, -1)
        k = k.reshape(batch, -1, self.heads, self.head_dim)
        v = v.reshape(batch, -1, self.heads, self.head_dim)
        relative = self.relative(self.positions).reshape(-1, self.heads, self.head_dim)
        scores = torch.einsum("bthd,bkhd->bhtk", q + self.content_bias, k)
        position_scores = torch.einsum(
            "bthd,mhd->bhtm", q + self.position_bias, relative
        )
        scores = scores + position_scores.gather(
            -1, relative_index[None, None].expand(batch, self.heads, -1, -1)
        )
        weights = (
            (scores / math.sqrt(self.head_dim))
            .masked_fill(~mask[:, None], -torch.inf)
            .softmax(-1)
        )
        attention = torch.einsum("bhtk,bkhd->bthd", self.dropout(weights), v)
        current = self.attention_gate(
            current,
            self.dropout(F.relu(self.out(attention.reshape(batch, length, width)))),
        )
        return self.mlp_gate(
            current, self.dropout(F.relu(self.mlp(self.norm2(current))))
        )


class GTrXL(nn.Module):
    """Gated Transformer-XL backbone with explicit, detached recurrent memory.

    Implements relative attention, reordered layer normalization and GRU-style
    residual gates from Parisotto et al., *Stabilizing Transformers for
    Reinforcement Learning* (2020), https://arxiv.org/abs/1910.06764.
    Use with :class:`~torchrl.modules.TransformerModule` and its environment
    primer. Each query sees at most ``memory_len`` preceding inputs plus itself.

    Memory contains the **inputs** to each layer, ordered oldest to newest,
    with feature shape ``[num_layers, memory_len, hidden_size]``. A boolean
    ``valid`` tensor of shape ``[memory_len]`` marks usable slots. Environment
    batch dimensions precede these feature dimensions. Invalid slots may
    contain arbitrary values; attention never reads them.

    Args:
        input_size (int): number of observation or encoder features.
        hidden_size (int, optional): even residual width, divisible by
            ``num_heads``. Defaults to ``32``.
        num_layers (int, optional): number of gated blocks. Defaults to ``2``.

    Keyword Args:
        num_heads (int, optional): attention heads per block. Defaults to ``4``.
        memory_len (int, optional): positive capacity of each layer's rolling
            memory. Defaults to ``16``.
        state_cls (type, optional): TensorDict-compatible state container with
            ``memory`` and ``valid`` fields, constructed by ``Composite(data_cls=...)``.
            Defaults to :class:`~tensordict.TensorDict`. TensorClass and
            TypedTensorDict schemas are also supported; this choice does not
            change existing recurrent module defaults.
        gate_bias (float, optional): positive initial update-gate bias favoring
            the residual identity path. Defaults to ``2.0``.
        dropout (float, optional): attention and residual dropout probability.
            Defaults to ``0.0``. Disable dropout for collection/training parity.
        device (torch.device or str, optional): initial parameter device.
            Defaults to PyTorch's default device.
        dtype (torch.dtype, optional): initial parameter and memory dtype.
            Defaults to PyTorch's default floating dtype.

    .. note::
        Supplied memory is caller-owned and detached at sequence boundaries.
        Gradients flow through recomputed activations within a training window.
        Stored activations can become stale after weight updates, as with RNN
        replay; they are never silently discarded after an optimizer step.

    Examples:
        >>> from tensordict import TensorDict
        >>> from torchrl.modules import TransformerModule, set_recurrent_mode
        >>> backbone = GTrXL(3, 16, memory_len=8)
        >>> module = TransformerModule(
        ...     transformer=backbone, in_keys=["observation", "state"],
        ...     out_keys=["features", ("next", "state")])
        >>> step = TensorDict(dict(observation=torch.randn(2, 3),
        ...     state=backbone.state_spec.zero([2]),
        ...     is_init=torch.ones(2, 1, dtype=torch.bool)), [2])
        >>> module(step)["features"].shape
        torch.Size([2, 16])
        >>> # A compact record has batch [B], observations [B, T, input_size].
        >>> window = TensorDict(dict(observation=torch.randn(2, 5, 3),
        ...     state=step["next", "state"],
        ...     is_init=torch.zeros(2, 5, 1, dtype=torch.bool)), [2])
        >>> with set_recurrent_mode(True):
        ...     result = module(window)
        >>> result["features"].shape, result["next", "state"].batch_size
        (torch.Size([2, 5, 16]), torch.Size([2]))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        *,
        num_heads: int = 4,
        memory_len: int = 16,
        state_cls: type[TensorDictBase] | type[TensorClass] = TensorDict,
        gate_bias: float = 2.0,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if (
            num_heads < 1
            or hidden_size < 1
            or hidden_size % num_heads
            or hidden_size % 2
        ):
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
    def state_spec(self) -> Composite:
        """Composite with explicit memory/valid leaf specs and the chosen container."""
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

    def forward_state(
        self,
        features: torch.Tensor,
        state: TensorDictBase | TensorClass,
        is_init: torch.Tensor,
        *,
        recurrent: bool,
    ) -> tuple[torch.Tensor, TensorDictBase | TensorClass]:
        """Process flattened ``[B, T]`` inputs for ``TransformerModule``.

        Args:
            features (torch.Tensor): input features with shape ``[B, T, F]``.
            state (TensorDictBase or TensorClass): batch ``[B, T]`` for stored
                per-step states, or ``[B]`` for one initial state per window.
            is_init (torch.Tensor): boolean markers of shape ``[B, T]``.

        Keyword Args:
            recurrent (bool): enable training-window execution. With per-step
                states, boundaries load their stored carry, including synthetic
                slice starts. With one initial state, markers denote **real
                episode resets**; arbitrary slice starts cannot be reconstructed.

        Returns:
            Features ``[B, T, hidden_size]`` and detached next state with the
            same batch shape as ``state``. Compact windows return only their
            final carry and use parallel attention. Per-step states return
            every carry, retaining the arbitrary-slice training contract.
        """
        if state.ndim == 1:
            if not recurrent:
                raise ValueError("A compact initial state requires recurrent mode")
            return self._forward_window(features, state, is_init)
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

    def _forward_window(self, features, state, is_init):
        memory = state.get("memory").detach()
        valid = state.get("valid")
        length = features.shape[1]
        positions = torch.arange(length, device=features.device)
        keys = torch.arange(-self.memory_len, length, device=features.device)
        distance = positions[:, None] - keys[None]
        horizon = (distance >= 0) & (distance <= self.memory_len)
        segments = is_init.long().cumsum(-1)
        memory_mask = valid[:, None] & (segments[:, :, None] == 0)
        window_mask = segments[:, :, None] == segments[:, None, :]
        mask = torch.cat((memory_mask, window_mask), -1) & horizon
        relative_index = self.memory_len - distance.clamp(0, self.memory_len)
        value = self.embedding(features)
        memories = []
        for layer, block in enumerate(self.blocks):
            layer_memory = memory[:, layer]
            memories.append(torch.cat((layer_memory, value), 1)[:, -self.memory_len :])
            value = block.forward_window(value, layer_memory, mask, relative_index)
        # Only history belonging to the last episode can enter the next window.
        valid = torch.cat(
            (valid & (segments[:, -1:] == 0), segments == segments[:, -1:]), -1
        )[:, -self.memory_len :]
        memory = torch.stack(memories, 1).detach()
        memory = memory.masked_fill(~valid[:, None, :, None], 0)
        return value, self.state_cls.from_dict(
            {"memory": memory, "valid": valid},
            batch_size=features.shape[:1],
            device=features.device,
        )
