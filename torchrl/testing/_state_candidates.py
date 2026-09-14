# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared, private fixtures for the recurrent-state comparison and benchmarks."""
from __future__ import annotations

import torch
from tensordict import TensorClass, TensorDict, TypedTensorDict
from tensordict.nn import TensorDictModuleBase

from torchrl.data import Composite, Unbounded
from torchrl.envs import TensorDictPrimer
from torchrl.modules import GRUModule, GTrXL, LSTMModule, TransformerModule


class _GRUTC(TensorClass):
    carry: torch.Tensor


class _GRUTTD(TypedTensorDict):
    carry: torch.Tensor


class _LSTMTC(TensorClass):
    hidden: torch.Tensor
    cell: torch.Tensor


class _LSTMTTD(TypedTensorDict):
    hidden: torch.Tensor
    cell: torch.Tensor


class _GTrXLTC(TensorClass):
    memory: torch.Tensor
    valid: torch.Tensor


class _GTrXLTTD(TypedTensorDict):
    memory: torch.Tensor
    valid: torch.Tensor


_STATE_CLASSES = {
    "gru": {"td": TensorDict, "tc": _GRUTC, "ttd": _GRUTTD},
    "lstm": {"td": TensorDict, "tc": _LSTMTC, "ttd": _LSTMTTD},
    "gtrxl": {"td": TensorDict, "tc": _GTrXLTC, "ttd": _GTrXLTTD},
}


class _StateRNN(TensorDictModuleBase):
    def __init__(
        self, kind, state_cls, input_size, hidden_size, num_layers, device, state_key
    ):
        super().__init__()
        self.state_cls = state_cls
        self.state_key = state_key
        fields = ("carry",) if kind == "gru" else ("hidden", "cell")
        in_keys = ["observation", *((*state_key, field) for field in fields)]
        out_keys = ["embed", *(("next", *state_key, field) for field in fields)]
        rnn_cls = GRUModule if kind == "gru" else LSTMModule
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            num_layers=num_layers,
            in_keys=in_keys,
            out_keys=out_keys,
            device=device,
        )
        self.in_keys = self.rnn.in_keys
        self.out_keys = ["embed", ("next", *state_key)]
        self.state_spec = Composite(
            {
                field: Unbounded((num_layers, hidden_size), device=device)
                for field in fields
            },
            data_cls=state_cls,
            device=device,
        )

    def make_tensordict_primer(self):
        return TensorDictPrimer(
            Composite({self.state_key: self.state_spec}), expand_specs=True
        )

    def forward(self, td):
        td = self.rnn(td)
        key = ("next", *self.state_key)
        value = td.get(key)
        if type(value) is not self.state_cls:
            td.set(key, self.state_cls.from_tensordict(value))
        return td


def _make_candidate(
    kind="gtrxl",
    container="td",
    *,
    input_size=7,
    hidden_size=16,
    num_layers=2,
    memory_len=8,
    device="cpu",
    state_key=("agent", "state"),
):
    if container == "flat":
        rnn_cls = GRUModule if kind == "gru" else LSTMModule
        return rnn_cls(
            input_size,
            hidden_size,
            num_layers=num_layers,
            in_key="observation",
            out_key="embed",
            device=device,
        )
    state_cls = _STATE_CLASSES[kind][container]
    if kind == "gtrxl":
        return TransformerModule(
            transformer=GTrXL(
                input_size,
                hidden_size,
                num_layers,
                num_heads=2,
                memory_len=memory_len,
                state_cls=state_cls,
                device=device,
            ),
            in_keys=["observation", state_key, "is_init"],
            out_keys=["embed", ("next", *state_key)],
        )
    return _StateRNN(
        kind, state_cls, input_size, hidden_size, num_layers, device, state_key
    )
