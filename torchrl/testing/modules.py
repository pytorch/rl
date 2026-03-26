from __future__ import annotations

import torch
from tensordict import NestedKey, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torch import nn, vmap

from torchrl._utils import logger, RL_WARNINGS
from torchrl.modules import MLP
from torchrl.objectives.value.advantages import _vmap_func

__all__ = [
    "BiasModule",
    "LSTMNet",
    "NonSerializableBiasModule",
    "call_value_nets",
]


class BiasModule(nn.Module):
    """Simple bias module to check weight synchronization correctness."""

    def __init__(self, value: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(value, dtype=torch.float))

    def forward(self, x):
        return x + self.bias


class NonSerializableBiasModule(BiasModule):
    """Bias module that intentionally fails to serialize.

    This is used in tests to simulate a policy that cannot be pickled.
    """

    def __getstate__(self):
        # Simulate a non-serializable policy by raising on pickling
        raise RuntimeError("NonSerializableBiasModule cannot be pickled")


class LSTMNet(nn.Module):
    """An embedder for an LSTM preceded by an MLP.

    The forward method returns the hidden states of the current state
    (input hidden states) and the output, as
    the environment returns the 'observation' and 'next_observation'.

    Because the LSTM kernel only returns the last hidden state, hidden states
    are padded with zeros such that they have the right size to be stored in a
    TensorDict of size [batch x time_steps].

    If a 2D tensor is provided as input, it is assumed that it is a batch of data
    with only one time step. This means that we explicitly assume that users will
    unsqueeze inputs of a single batch with multiple time steps.

    Args:
        out_features (int): number of output features.
        lstm_kwargs (dict): the keyword arguments for the
            :class:`~torch.nn.LSTM` layer.
        mlp_kwargs (dict): the keyword arguments for the
            :class:`~torchrl.modules.MLP` layer.
        device (torch.device, optional): the device where the module should
            be instantiated.

    Keyword Args:
        lstm_backend (str, optional): one of ``"torchrl"`` or ``"torch"`` that
            indicates where the LSTM class is to be retrieved. The ``"torchrl"``
            backend (:class:`~torchrl.modules.LSTM`) is slower but works with
            :func:`~torch.vmap` and should work with :func:`~torch.compile`.
            Defaults to ``"torch"``.

    Examples:
        >>> batch = 7
        >>> time_steps = 6
        >>> in_features = 4
        >>> out_features = 10
        >>> hidden_size = 5
        >>> net = LSTMNet(
        ...     out_features,
        ...     {"input_size": hidden_size, "hidden_size": hidden_size},
        ...     {"out_features": hidden_size},
        ... )
        >>> # test single step vs multi-step
        >>> x = torch.randn(batch, time_steps, in_features)  # >3 dims = multi-step
        >>> y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(x)
        >>> x = torch.randn(batch, in_features)  # 2 dims = single step
        >>> y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(x)

    """

    def __init__(
        self,
        out_features: int,
        lstm_kwargs,
        mlp_kwargs,
        device=None,
        *,
        lstm_backend: str | None = None,
    ) -> None:
        super().__init__()
        lstm_kwargs.update({"batch_first": True})
        self.mlp = MLP(device=device, **mlp_kwargs)
        if lstm_backend is None:
            lstm_backend = "torch"
        self.lstm_backend = lstm_backend
        if self.lstm_backend == "torch":
            LSTM = nn.LSTM
        else:
            from torchrl.modules.tensordict_module.rnn import LSTM
        self.lstm = LSTM(device=device, **lstm_kwargs)
        self.linear = nn.LazyLinear(out_features, device=device)

    def _lstm(
        self,
        input: torch.Tensor,
        hidden0_in: torch.Tensor | None = None,
        hidden1_in: torch.Tensor | None = None,
    ):
        squeeze0 = False
        squeeze1 = False
        if input.ndimension() == 1:
            squeeze0 = True
            input = input.unsqueeze(0).contiguous()

        if input.ndimension() == 2:
            squeeze1 = True
            input = input.unsqueeze(1).contiguous()
        batch, steps = input.shape[:2]

        if hidden1_in is None and hidden0_in is None:
            shape = (batch, steps) if not squeeze1 else (batch,)
            hidden0_in, hidden1_in = (
                torch.zeros(
                    *shape,
                    self.lstm.num_layers,
                    self.lstm.hidden_size,
                    device=input.device,
                    dtype=input.dtype,
                )
                for _ in range(2)
            )
        elif hidden1_in is None or hidden0_in is None:
            raise RuntimeError(
                f"got type(hidden0)={type(hidden0_in)} and type(hidden1)={type(hidden1_in)}"
            )
        elif squeeze0:
            hidden0_in = hidden0_in.unsqueeze(0)
            hidden1_in = hidden1_in.unsqueeze(0)

        # we only need the first hidden state
        if not squeeze1:
            _hidden0_in = hidden0_in[:, 0]
            _hidden1_in = hidden1_in[:, 0]
        else:
            _hidden0_in = hidden0_in
            _hidden1_in = hidden1_in
        hidden = (
            _hidden0_in.transpose(-3, -2).contiguous(),
            _hidden1_in.transpose(-3, -2).contiguous(),
        )

        y0, hidden = self.lstm(input, hidden)
        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = tuple(_h.transpose(0, 1) for _h in hidden)
        y = self.linear(y0)

        out = [y, hidden0_in, hidden1_in, *hidden]
        if squeeze1:
            # squeezes time
            out[0] = out[0].squeeze(1)
        if not squeeze1:
            # we pad the hidden states with zero to make tensordict happy
            for i in range(3, 5):
                out[i] = torch.stack(
                    [torch.zeros_like(out[i]) for _ in range(input.shape[1] - 1)]
                    + [out[i]],
                    1,
                )
        if squeeze0:
            out = [_out.squeeze(0) for _out in out]
        return tuple(out)

    def forward(
        self,
        input: torch.Tensor,
        hidden0_in: torch.Tensor | None = None,
        hidden1_in: torch.Tensor | None = None,
    ):
        input = self.mlp(input)
        return self._lstm(input, hidden0_in, hidden1_in)


def call_value_nets(
    value_net: TensorDictModuleBase,
    data: TensorDictBase,
    params: TensorDictBase,
    next_params: TensorDictBase,
    single_call: bool,
    value_key: NestedKey,
    detach_next: bool,
    vmap_randomness: str = "error",
):
    """Call value networks to compute values at t and t+1.

    This is a testing utility for computing value estimates in advantage
    calculations.

    Args:
        value_net: The value network module.
        data: Input tensordict with observations.
        params: Parameters for the value network at time t.
        next_params: Parameters for the value network at time t+1.
        single_call: Whether to use a single forward pass for both t and t+1.
        value_key: The key where values are stored.
        detach_next: Whether to detach the next value from the computation graph.
        vmap_randomness: Randomness mode for vmap.

    Returns:
        Tuple of (value, value_next).
    """
    in_keys = value_net.in_keys
    if single_call:
        for i, name in enumerate(data.names):
            if name == "time":
                ndim = i + 1
                break
        else:
            ndim = None
        if ndim is not None:
            # get data at t and last of t+1
            idx0 = (slice(None),) * (ndim - 1) + (slice(-1, None),)
            idx = (slice(None),) * (ndim - 1) + (slice(None, -1),)
            idx_ = (slice(None),) * (ndim - 1) + (slice(1, None),)
            data_in = torch.cat(
                [
                    data.select(*in_keys, value_key, strict=False),
                    data.get("next").select(*in_keys, value_key, strict=False)[idx0],
                ],
                ndim - 1,
            )
        else:
            if RL_WARNINGS:
                logger.warning(
                    "Got a tensordict without a time-marked dimension, assuming time is along the last dimension. "
                    "This warning can be turned off by setting the environment variable RL_WARNINGS to False."
                )
            ndim = data.ndim
            idx = (slice(None),) * (ndim - 1) + (slice(None, data.shape[ndim - 1]),)
            idx_ = (slice(None),) * (ndim - 1) + (slice(data.shape[ndim - 1], None),)
            data_in = torch.cat(
                [
                    data.select(*in_keys, value_key, strict=False),
                    data.get("next").select(*in_keys, value_key, strict=False),
                ],
                ndim - 1,
            )

        # next_params should be None or be identical to params
        if next_params is not None and next_params is not params:
            raise ValueError(
                "the value at t and t+1 cannot be retrieved in a single call without recurring to vmap when both params and next params are passed."
            )
        if params is not None:
            with params.to_module(value_net):
                value_est = value_net(data_in).get(value_key)
        else:
            value_est = value_net(data_in).get(value_key)
        value, value_ = value_est[idx], value_est[idx_]
    else:
        data_in = torch.stack(
            [
                data.select(*in_keys, value_key, strict=False),
                data.get("next").select(*in_keys, value_key, strict=False),
            ],
            0,
        )
        if (params is not None) ^ (next_params is not None):
            raise ValueError(
                "params and next_params must be either both provided or not."
            )
        elif params is not None:
            params_stack = torch.stack([params, next_params], 0).contiguous()
            data_out = _vmap_func(value_net, (0, 0), randomness=vmap_randomness)(
                data_in, params_stack
            )
        else:
            data_out = vmap(value_net, (0,), randomness=vmap_randomness)(data_in)
        value_est = data_out.get(value_key)
        value, value_ = value_est[0], value_est[1]
    data.set(value_key, value)
    data.set(("next", value_key), value_)
    if detach_next:
        value_ = value_.detach()
    return value, value_
