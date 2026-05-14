# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.metadata
import typing
from typing import Any

import torch
import torch.nn.functional as F
from packaging import version
from tensordict import TensorDict, TensorDictBase, unravel_key_list
from tensordict.base import NO_DEFAULT
from tensordict.nn import dispatch, TensorDictModuleBase as ModuleBase
from tensordict.utils import expand_as_right, prod, set_lazy_legacy
from torch import nn, Tensor
from torch.nn.modules.rnn import RNNCellBase

from torchrl._utils import (
    _ContextManager,
    _DecoratorContextManager,
    implement_for,
    is_compiling,
)
from torchrl.data.tensor_specs import Unbounded

# ``torch._higher_order_ops.scan`` was introduced in PyTorch 2.6. Gate the
# import on the runtime torch version: probing via ``importlib.util.find_spec``
# would eagerly import the (missing) ``torch._higher_order_ops`` parent on
# older builds and crash this module at load time.
_has_torch_scan = version.parse(torch.__version__) >= version.parse("2.6.0")
if _has_torch_scan:
    from torch._higher_order_ops import scan as _torch_scan
else:
    _torch_scan = None


def _check_triton_available() -> bool:
    """True if Triton is installed and exposes the API the kernels need.

    Mirrors the probe in :mod:`torchrl.modules.tensordict_module._rnn_triton`.
    The backend requires ``triton.language.extra.libdevice`` which is only
    available from Triton 2.2 onwards. Older Triton builds fall back to the
    scan / pad backends. The version is read from package metadata to avoid
    eagerly importing Triton (or its missing ``triton.language.extra`` parent)
    at torchrl import time.
    """
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        return False
    return version.parse(triton_version) >= version.parse("2.2")


_has_triton = _check_triton_available()


@implement_for("torch", None, "2.6.0", compilable=True)
def _scan(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "torch._higher_order_ops.scan is required for the scan recurrent backend "
        "and is available in PyTorch >= 2.6.0."
    )


@implement_for("torch", "2.6.0", compilable=True)
def _scan(*args: Any, **kwargs: Any) -> Any:  # noqa: F811
    if _torch_scan is None:
        raise NotImplementedError(
            "torch._higher_order_ops.scan is required for the scan recurrent "
            "backend but is not available in this PyTorch build."
        )
    return _torch_scan(*args, **kwargs)


def _place_at_traj_end(
    h: torch.Tensor, splits: torch.Tensor, steps: int
) -> torch.Tensor:
    """Scatter per-trajectory hidden states onto a zero-padded time grid.

    Given ``h`` of shape ``[N, *F]``, returns a tensor of shape
    ``[N, steps, *F]`` whose row ``i`` is zero everywhere except at index
    ``splits[i] - 1`` along dim 1, where it holds ``h[i]``. Out-of-place
    ``scatter`` is used so the call is compatible with :func:`torch.vmap`.
    """
    h_padded = torch.zeros(
        h.shape[0], steps, *h.shape[1:], device=h.device, dtype=h.dtype
    )
    idx = (
        (splits - 1)
        .long()
        .view(-1, 1, *([1] * (h.dim() - 1)))
        .expand_as(h.unsqueeze(1))
    )
    return h_padded.scatter(1, idx, h.unsqueeze(1))


def _num_directions(rnn: nn.RNNBase) -> int:
    return 2 if rnn.bidirectional else 1


def _end_mask_from_is_init(is_init: torch.Tensor) -> torch.Tensor:
    end_mask = torch.empty_like(is_init)
    end_mask[:, :-1] = is_init[:, 1:]
    end_mask[:, -1] = True
    return end_mask


class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell that performs the same operation as nn.LSTMCell but is fully coded in Python.

    .. note::
        This class is implemented without relying on CuDNN, which makes it compatible with :func:`torch.vmap` and :func:`torch.compile`.

    Examples:
        >>> import torch
        >>> from torchrl.modules.tensordict_module.rnn import LSTMCell
        >>> device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
        >>> B = 2
        >>> N_IN = 10
        >>> N_OUT = 20
        >>> V = 4  # vector size
        >>> lstm_cell = LSTMCell(input_size=N_IN, hidden_size=N_OUT, device=device)

        # single call
        >>> x = torch.randn(B, 10, device=device)
        >>> h0 = torch.zeros(B, 20, device=device)
        >>> c0 = torch.zeros(B, 20, device=device)
        >>> with torch.no_grad():
        ...     (h1, c1) = lstm_cell(x, (h0, c0))

        # vectorised call - not possible with nn.LSTMCell
        >>> def call_lstm(x, h, c):
        ...     h_out, c_out = lstm_cell(x, (h, c))
        ...     return h_out, c_out
        >>> batched_call = torch.vmap(call_lstm)
        >>> x = torch.randn(V, B, 10, device=device)
        >>> h0 = torch.zeros(V, B, 20, device=device)
        >>> c0 = torch.zeros(V, B, 20, device=device)
        >>> with torch.no_grad():
        ...     (h1, c1) = batched_call(x, h0, c0)
    """

    __doc__ += nn.LSTMCell.__doc__

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"LSTMCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None:
            for idx, value in enumerate(hx):
                if value.dim() not in (1, 2):
                    raise ValueError(
                        f"LSTMCell: Expected hx[{idx}] to be 1D or 2D, got {value.dim()}D instead"
                    )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        ret = self.lstm_cell(input, hx[0], hx[1])

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret

    def lstm_cell(self, x, hx, cx):
        x = x.view(-1, x.size(1))

        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(
            hx, self.weight_hh, self.bias_hh
        )

        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        i_gate = i_gate.sigmoid()
        f_gate = f_gate.sigmoid()
        g_gate = g_gate.tanh()
        o_gate = o_gate.sigmoid()

        cy = cx * f_gate + i_gate * g_gate

        hy = o_gate * cy.tanh()

        return hy, cy


# copy LSTM
class LSTMBase(nn.RNNBase):
    """A Base module for LSTM. Inheriting from LSTMBase enables compatibility with torch.compile."""

    def __init__(self, *args, **kwargs):
        return super().__init__("LSTM", *args, **kwargs)


for attr in nn.LSTM.__dict__:
    if attr != "__init__":
        setattr(LSTMBase, attr, getattr(nn.LSTM, attr))


class LSTM(LSTMBase):
    """A PyTorch module for executing multiple steps of a multi-layer LSTM. The module behaves exactly like :class:`torch.nn.LSTM`, but this implementation is exclusively coded in Python.

    .. note::
        This class is implemented without relying on CuDNN, which makes it compatible with :func:`torch.vmap` and :func:`torch.compile`.

    Examples:
        >>> import torch
        >>> from torchrl.modules.tensordict_module.rnn import LSTM

        >>> device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
        >>> B = 2
        >>> T = 4
        >>> N_IN = 10
        >>> N_OUT = 20
        >>> N_LAYERS = 2
        >>> V = 4  # vector size
        >>> lstm = LSTM(
        ...     input_size=N_IN,
        ...     hidden_size=N_OUT,
        ...     device=device,
        ...     num_layers=N_LAYERS,
        ... )

        # single call
        >>> x = torch.randn(B, T, N_IN, device=device)
        >>> h0 = torch.zeros(N_LAYERS, B, N_OUT, device=device)
        >>> c0 = torch.zeros(N_LAYERS, B, N_OUT, device=device)
        >>> with torch.no_grad():
        ...     h1, c1 = lstm(x, (h0, c0))

        # vectorised call - not possible with nn.LSTM
        >>> def call_lstm(x, h, c):
        ...     h_out, c_out = lstm(x, (h, c))
        ...     return h_out, c_out
        >>> batched_call = torch.vmap(call_lstm)
        >>> x = torch.randn(V, B, T, 10, device=device)
        >>> h0 = torch.zeros(V, N_LAYERS, B, N_OUT, device=device)
        >>> c0 = torch.zeros(V, N_LAYERS, B, N_OUT, device=device)
        >>> with torch.no_grad():
        ...     h1, c1 = batched_call(x, h0, c0)
    """

    __doc__ += nn.LSTM.__doc__

    use_scan: bool = False

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: float = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
        use_scan: bool = False,
    ) -> None:

        if bidirectional is True:
            raise NotImplementedError(
                "Bidirectional LSTMs are not supported yet in this implementation."
            )

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            device=device,
            dtype=dtype,
        )
        # Opt-in prototype: replace the python time loop with
        # ``torch._higher_order_ops.scan``. Requires :func:`torch.compile`
        # to capture the scan; eager use will fail. Dropout is not supported
        # on this path. See :meth:`_lstm_scan`.
        self.use_scan = use_scan

    @staticmethod
    def _lstm_cell(x, hx, cx, weight_ih, bias_ih, weight_hh, bias_hh):

        gates = F.linear(x, weight_ih, bias_ih) + F.linear(hx, weight_hh, bias_hh)

        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        i_gate = i_gate.sigmoid()
        f_gate = f_gate.sigmoid()
        g_gate = g_gate.tanh()
        o_gate = o_gate.sigmoid()

        cy = cx * f_gate + i_gate * g_gate

        hy = o_gate * cy.tanh()

        return hy, cy

    def _lstm(self, x, hx, mask=None):

        if self.use_scan:
            return self._lstm_scan(x, hx, mask)

        h_t, c_t = hx
        h_t, c_t = h_t.unbind(0), c_t.unbind(0)

        outputs = []

        weight_ihs = []
        weight_hhs = []
        bias_ihs = []
        bias_hhs = []
        for weights in self._all_weights:
            # Retrieve weights
            weight_ihs.append(getattr(self, weights[0]))
            weight_hhs.append(getattr(self, weights[1]))
            if self.bias:
                bias_ihs.append(getattr(self, weights[2]))
                bias_hhs.append(getattr(self, weights[3]))
            else:
                bias_ihs.append(None)
                bias_hhs.append(None)

        time_dim = int(self.batch_first)
        x_unbound = x.unbind(time_dim)
        if mask is not None:
            mask_unbound = mask.unbind(time_dim)

        for t, x_t in enumerate(x_unbound):
            h_t_out = []
            c_t_out = []
            m_t = mask_unbound[t].unsqueeze(-1) if mask is not None else None

            for layer, (
                weight_ih,
                bias_ih,
                weight_hh,
                bias_hh,
                _h_t,
                _c_t,
            ) in enumerate(zip(weight_ihs, bias_ihs, weight_hhs, bias_hhs, h_t, c_t)):
                # Run cell
                h_new, c_new = self._lstm_cell(
                    x_t, _h_t, _c_t, weight_ih, bias_ih, weight_hh, bias_hh
                )
                if m_t is not None:
                    # Freeze hidden/cell state for batch entries whose
                    # trajectory has already ended at this time step. The cell
                    # is still evaluated for the full batch (wasteful but
                    # vmap/compile-friendly); only the carry is masked.
                    h_new = torch.where(m_t, h_new, _h_t)
                    c_new = torch.where(m_t, c_new, _c_t)
                h_t_out.append(h_new)
                c_t_out.append(c_new)

                # Apply dropout if in training mode
                if layer < self.num_layers - 1 and self.dropout:
                    x_t = F.dropout(h_new, p=self.dropout, training=self.training)
                else:  # No dropout after the last layer
                    x_t = h_new
            h_t = h_t_out
            c_t = c_t_out
            outputs.append(x_t)

        outputs = torch.stack(outputs, dim=time_dim)

        return outputs, (torch.stack(h_t_out, 0), torch.stack(c_t_out, 0))

    def _lstm_scan(self, x, hx, mask=None):
        """Prototype scan-based time loop. Must be called inside ``torch.compile``.

        ``torch._higher_order_ops.scan`` is a prototype feature; this path is
        opt-in via ``LSTM(..., use_scan=True)`` and replaces the python
        ``for`` loop over time. Dropout is not supported here.
        """
        if self.dropout:
            raise NotImplementedError(
                "LSTM(use_scan=True) does not support dropout yet."
            )

        weight_ihs, weight_hhs, bias_ihs, bias_hhs = [], [], [], []
        for weights in self._all_weights:
            weight_ihs.append(getattr(self, weights[0]).clone())
            weight_hhs.append(getattr(self, weights[1]).clone())
            bias_ihs.append(getattr(self, weights[2]).clone() if self.bias else None)
            bias_hhs.append(getattr(self, weights[3]).clone() if self.bias else None)

        # scan iterates along dim 0; permute to time-first if needed.
        if self.batch_first:
            x = x.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

        num_layers = self.num_layers

        def step(carry, inputs):
            h_layers, c_layers = carry  # each [num_layers, B, H]
            x_t, m_t = inputs
            m_t = m_t.unsqueeze(-1)
            new_h, new_c = [], []
            h_unbound = h_layers.unbind(0)
            c_unbound = c_layers.unbind(0)
            for layer in range(num_layers):
                h_prev = h_unbound[layer]
                c_prev = c_unbound[layer]
                h_new, c_new = self._lstm_cell(
                    x_t,
                    h_prev,
                    c_prev,
                    weight_ihs[layer],
                    bias_ihs[layer],
                    weight_hhs[layer],
                    bias_hhs[layer],
                )
                h_new = torch.where(m_t, h_new, h_prev)
                c_new = torch.where(m_t, c_new, c_prev)
                new_h.append(h_new)
                new_c.append(c_new)
                x_t = h_new
            new_h = torch.stack(new_h, 0).clone()
            new_c = torch.stack(new_c, 0).clone()
            return (new_h, new_c), x_t.clone()

        h0, c0 = hx
        (h_final, c_final), outputs = _scan(step, (h0, c0), (x, mask), dim=0)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, (h_final, c_final)

    def forward(self, input, hx=None, mask=None):  # noqa: F811
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        if input.dim() != 3:
            raise ValueError(
                f"LSTM: Expected input to be 3D, got {input.dim()}D instead"
            )
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            h_zeros = torch.zeros(
                self.num_layers,
                max_batch_size,
                real_hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            c_zeros = torch.zeros(
                self.num_layers,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (h_zeros, c_zeros)
        return self._lstm(input, hx, mask)


class LSTMModule(ModuleBase):
    """An embedder for an LSTM module.

    This class adds the following functionality to :class:`torch.nn.LSTM`:

    - Compatibility with TensorDict: the hidden states are reshaped to match
      the tensordict batch size.
    - Optional multi-step execution: with torch.nn, one has to choose between
      :class:`torch.nn.LSTMCell` and :class:`torch.nn.LSTM`, the former being
      compatible with single step inputs and the latter being compatible with
      multi-step. This class enables both usages.


    After construction, the module is *not* set in recurrent mode, ie. it will
    expect single steps inputs.

    If in recurrent mode, it is expected that the last dimension of the tensordict
    marks the number of steps. There is no constrain on the dimensionality of the
    tensordict (except that it must be greater than one for temporal inputs).

    .. note::
      This class can handle multiple consecutive trajectories along the time dimension
      *but* the final hidden values should not be trusted in those cases (ie. they
      should not be reused for a consecutive trajectory).
      The reason is that LSTM returns only the last hidden value, which for the
      padded inputs we provide can correspond to a 0-filled input.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        python_based: If ``True``, will use a full Python implementation of the LSTM cell. Default: ``False``
        recurrent_backend: backend used in recurrent mode when trajectories reset
            in the middle of a batch. ``"pad"`` keeps the existing split/pad
            strategy. ``"scan"`` uses a scan loop over the time dimension and
            avoids materializing padded trajectory chunks. ``"triton"``
            (prototype, CUDA only) uses Triton kernels where available and
            otherwise preserves pad-backend recurrent semantics for dropout,
            projections and bidirectional layers. ``"auto"`` uses ``"pad"``
            in eager mode and ``"scan"`` when called under
            :func:`torch.compile`. Default: ``"pad"``.
        recurrent_compute_dtype: dtype used for the recurrent matmul inside the
            ``"triton"`` backend (``torch.float32`` -> TF32 on H100, default;
            ``torch.bfloat16`` -> bigger SMEM margin, lower precision).
            Ignored by the other backends. Default: ``torch.float32``.

    Keyword Args:
        in_key (str or tuple of str): the input key of the module. Exclusive use
            with ``in_keys``. If provided, the recurrent keys are assumed to be
            ["recurrent_state_h", "recurrent_state_c"] and the ``in_key`` will be
            appended before these.
        in_keys (list of str): a triplet of strings corresponding to the input value,
            first and second hidden key. Exclusive with ``in_key``.
        out_key (str or tuple of str): the output key of the module. Exclusive use
            with ``out_keys``. If provided, the recurrent keys are assumed to be
            [("next", "recurrent_state_h"), ("next", "recurrent_state_c")]
            and the ``out_key`` will be
            appended before these.
        out_keys (list of str): a triplet of strings corresponding to the output value,
            first and second hidden key.

            .. note::
               For a better integration with TorchRL's environments, the best naming
               for the output hidden key is ``("next", <custom_key>)``, such
               that the hidden values are passed from step to step during a rollout.

        device (torch.device or compatible): the device of the module.
        lstm (torch.nn.LSTM, optional): an LSTM instance to be wrapped.
            Exclusive with other nn.LSTM arguments.
        default_recurrent_mode (bool, optional): if provided, the recurrent mode if it hasn't been overridden
            by the :class:`~torchrl.modules.set_recurrent_mode` context manager / decorator.
            Defaults to ``False``.

    Attributes:
        recurrent_mode: Returns the recurrent mode of the module.

    Methods:
        set_recurrent_mode: controls whether the module should be executed in
            recurrent mode.
        make_tensordict_primer: creates the TensorDictPrimer transforms for the environment to be aware of the
            recurrent states of the RNN.

    .. note:: This module relies on specific ``recurrent_state`` keys being present in the input
        TensorDicts. To generate a :class:`~torchrl.envs.transforms.TensorDictPrimer` transform that will automatically
        add hidden states to the environment TensorDicts, use the method :func:`~torchrl.modules.rnn.LSTMModule.make_tensordict_primer`.
        If this class is a submodule in a larger module, the method :func:`~torchrl.modules.utils.get_primers_from_module` can be called
        on the parent module to automatically generate the primer transforms required for all submodules, including this one.


    Examples:
        >>> from torchrl.envs import TransformedEnv, InitTracker
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.modules import MLP, LSTMModule
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
        >>> lstm_module = LSTMModule(
        ...     input_size=env.observation_spec["observation"].shape[-1],
        ...     hidden_size=64,
        ...     in_keys=["observation", "rs_h", "rs_c"],
        ...     out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")])
        >>> mlp = MLP(num_cells=[64], out_features=1)
        >>> policy = Seq(lstm_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
        >>> policy(env.reset())
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                intermediate: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
                is_init: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        rs_c: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                        rs_h: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)

    """

    DEFAULT_IN_KEYS = ["recurrent_state_h", "recurrent_state_c"]
    DEFAULT_OUT_KEYS = [("next", "recurrent_state_h"), ("next", "recurrent_state_c")]

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int | None = None,
        num_layers: int = 1,
        bias: bool = True,
        batch_first=True,
        dropout=0,
        proj_size=0,
        bidirectional=False,
        python_based=False,
        recurrent_backend: typing.Literal["auto", "pad", "scan", "triton"] = "pad",
        recurrent_compute_dtype: torch.dtype = torch.float32,
        *,
        in_key=None,
        in_keys=None,
        out_key=None,
        out_keys=None,
        device=None,
        lstm=None,
        default_recurrent_mode: bool | None = None,
    ):
        super().__init__()
        if recurrent_backend not in {"auto", "pad", "scan", "triton"}:
            raise ValueError(
                "recurrent_backend must be one of 'auto', 'pad', 'scan' or 'triton'. "
                f"Got {recurrent_backend}."
            )
        if recurrent_backend == "triton" and not _has_triton:
            raise RuntimeError(
                "recurrent_backend='triton' requires the triton package. "
                "Install it with `pip install triton`."
            )
        if lstm is not None:
            if not lstm.batch_first:
                raise ValueError("The input lstm must have batch_first=True.")
            if input_size is not None or hidden_size is not None:
                raise ValueError(
                    "An LSTM instance cannot be passed along with class argument."
                )
        else:
            if not batch_first:
                raise ValueError("The input lstm must have batch_first=True.")
            if not hidden_size:
                raise ValueError("hidden_size must be passed.")
            if python_based and bidirectional:
                raise ValueError(
                    "python_based=True does not support bidirectional LSTMs."
                )
            if python_based:
                lstm = LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bias=bias,
                    dropout=dropout,
                    proj_size=proj_size,
                    device=device,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            else:
                lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bias=bias,
                    dropout=dropout,
                    proj_size=proj_size,
                    device=device,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
        if not ((in_key is None) ^ (in_keys is None)):
            raise ValueError(
                f"Either in_keys or in_key must be specified but not both or none. Got {in_keys} and {in_key} respectively."
            )
        elif in_key:
            in_keys = [in_key, *self.DEFAULT_IN_KEYS]

        if not ((out_key is None) ^ (out_keys is None)):
            raise ValueError(
                f"Either out_keys or out_key must be specified but not both or none. Got {out_keys} and {out_key} respectively."
            )
        elif out_key:
            out_keys = [out_key, *self.DEFAULT_OUT_KEYS]

        in_keys = unravel_key_list(in_keys)
        out_keys = unravel_key_list(out_keys)
        if not isinstance(in_keys, (tuple, list)) or (
            len(in_keys) != 3 and not (len(in_keys) == 4 and in_keys[-1] == "is_init")
        ):
            raise ValueError(
                f"LSTMModule expects 3 inputs: a value, and two hidden states (and potentially an 'is_init' marker). Got in_keys {in_keys} instead."
            )
        if not isinstance(out_keys, (tuple, list)) or len(out_keys) != 3:
            raise ValueError(
                f"LSTMModule expects 3 outputs: a value, and two hidden states. Got out_keys {out_keys} instead."
            )
        self.lstm = lstm
        if "is_init" not in in_keys:
            in_keys = in_keys + ["is_init"]
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._recurrent_mode = default_recurrent_mode
        self.recurrent_backend = recurrent_backend
        self.recurrent_compute_dtype = recurrent_compute_dtype

    def make_python_based(self) -> LSTMModule:
        """Transforms the LSTM layer in its python-based version.

        Returns:
            self

        """
        if isinstance(self.lstm, LSTM):
            return self
        lstm = LSTM(
            input_size=self.lstm.input_size,
            hidden_size=self.lstm.hidden_size,
            num_layers=self.lstm.num_layers,
            bias=self.lstm.bias,
            dropout=self.lstm.dropout,
            proj_size=self.lstm.proj_size,
            device="meta",
            batch_first=self.lstm.batch_first,
            bidirectional=self.lstm.bidirectional,
        )
        from tensordict import from_module

        from_module(self.lstm).to_module(lstm)
        self.lstm = lstm
        return self

    def make_cudnn_based(self) -> LSTMModule:
        """Transforms the LSTM layer in its CuDNN-based version.

        Returns:
            self

        """
        if isinstance(self.lstm, nn.LSTM):
            return self
        lstm = nn.LSTM(
            input_size=self.lstm.input_size,
            hidden_size=self.lstm.hidden_size,
            num_layers=self.lstm.num_layers,
            bias=self.lstm.bias,
            dropout=self.lstm.dropout,
            proj_size=self.lstm.proj_size,
            device="meta",
            batch_first=self.lstm.batch_first,
            bidirectional=self.lstm.bidirectional,
        )
        from tensordict import from_module

        from_module(self.lstm).to_module(lstm)
        self.lstm = lstm
        return self

    def make_tensordict_primer(self):
        """Makes a tensordict primer for the environment.

        A :class:`~torchrl.envs.TensorDictPrimer` object will ensure that the policy is aware of the supplementary
        inputs and outputs (recurrent states) during rollout execution. That way, the data can be shared across
        processes and dealt with properly.

        When using batched environments such as :class:`~torchrl.envs.ParallelEnv`, the transform can be used at the
        single env instance level (i.e., a batch of transformed envs with tensordict primers set within) or at the
        batched env instance level (i.e., a transformed batch of regular envs).

        Not including a ``TensorDictPrimer`` in the environment may result in poorly defined behaviors, for instance
        in parallel settings where a step involves copying the new recurrent state from ``"next"`` to the root
        tensordict, which the meth:`~torchrl.EnvBase.step_mdp` method will not be able to do as the recurrent states
        are not registered within the environment specs.

        See :func:`torchrl.modules.utils.get_primers_from_module` for a method to generate all primers for a given
        module.

        Examples:
            >>> from torchrl.collectors import Collector
            >>> from torchrl.envs import TransformedEnv, InitTracker
            >>> from torchrl.envs import GymEnv
            >>> from torchrl.modules import MLP, LSTMModule
            >>> from torch import nn
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>>
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
            >>> lstm_module = LSTMModule(
            ...     input_size=env.observation_spec["observation"].shape[-1],
            ...     hidden_size=64,
            ...     in_keys=["observation", "rs_h", "rs_c"],
            ...     out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")])
            >>> mlp = MLP(num_cells=[64], out_features=1)
            >>> policy = Seq(lstm_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> policy(env.reset())
            >>> env = env.append_transform(lstm_module.make_tensordict_primer())
            >>> data_collector = Collector(
            ...     env,
            ...     policy,
            ...     frames_per_batch=10
            ... )
            >>> for data in data_collector:
            ...     print(data)
            ...     break

        """
        from torchrl.envs.transforms.transforms import TensorDictPrimer

        def make_tuple(key):
            if isinstance(key, tuple):
                return key
            return (key,)

        out_key1 = make_tuple(self.out_keys[1])
        in_key1 = make_tuple(self.in_keys[1])
        out_key2 = make_tuple(self.out_keys[2])
        in_key2 = make_tuple(self.in_keys[2])
        if out_key1 != ("next", *in_key1) or out_key2 != ("next", *in_key2):
            raise RuntimeError(
                "make_tensordict_primer is supposed to work with in_keys/out_keys that "
                "have compatible names, ie. the out_keys should be named after ('next', <in_key>). Got "
                f"in_keys={self.in_keys} and out_keys={self.out_keys} instead."
            )
        num_states = self.lstm.num_layers * _num_directions(self.lstm)
        real_hidden_size = (
            self.lstm.proj_size if self.lstm.proj_size > 0 else self.lstm.hidden_size
        )
        return TensorDictPrimer(
            {
                in_key1: Unbounded(shape=(num_states, real_hidden_size)),
                in_key2: Unbounded(shape=(num_states, self.lstm.hidden_size)),
            },
            expand_specs=True,
        )

    @property
    def recurrent_mode(self):
        rm = recurrent_mode()
        if rm is None:
            return bool(self._recurrent_mode)
        return rm

    @recurrent_mode.setter
    def recurrent_mode(self, value):
        raise RuntimeError(
            "recurrent_mode cannot be changed in-place. Please use the set_recurrent_mode context manager."
        )

    @property
    def temporal_mode(self):
        raise RuntimeError(
            "temporal_mode is deprecated, use recurrent_mode instead.",
        )

    def set_recurrent_mode(self, mode: bool = True):
        raise RuntimeError(
            "The lstm.set_recurrent_mode() API has been removed in v0.8. "
            "To set the recurrent mode, use the :class:`~torchrl.modules.set_recurrent_mode` context manager or "
            "the `default_recurrent_mode` keyword argument in the constructor."
        )

    @dispatch
    def forward(self, tensordict: TensorDictBase):
        from torchrl.objectives.value.functional import (
            _inv_pad_sequence,
            _split_and_pad_sequence,
        )

        # we want to get an error if the value input is missing, but not the hidden states
        defaults = [NO_DEFAULT, None, None]
        shape = tensordict.shape
        tensordict_shaped = tensordict
        if self.recurrent_mode:
            # if less than 2 dims, unsqueeze
            ndim = tensordict_shaped.get(self.in_keys[0]).ndim
            while ndim < 3:
                tensordict_shaped = tensordict_shaped.unsqueeze(0)
                ndim += 1
            if ndim > 3:
                dims_to_flatten = ndim - 3
                # we assume that the tensordict can be flattened like this
                nelts = prod(tensordict_shaped.shape[: dims_to_flatten + 1])
                tensordict_shaped = tensordict_shaped.apply(
                    lambda value: value.flatten(0, dims_to_flatten),
                    batch_size=[nelts, tensordict_shaped.shape[-1]],
                )
        else:
            tensordict_shaped = tensordict.reshape(-1).unsqueeze(-1)

        is_init = tensordict_shaped["is_init"].squeeze(-1)
        splits = None
        backend = self.recurrent_backend
        if backend == "auto":
            backend = "scan" if is_compiling() else "pad"
        use_scan = self.recurrent_mode and backend == "scan"
        use_triton = self.recurrent_mode and backend == "triton"
        if (
            self.recurrent_mode
            and not use_scan
            and not use_triton
            and is_init[..., 1:].any()
        ):
            from torchrl.objectives.value.utils import _get_num_per_traj_init

            # if we have consecutive trajectories, things get a little more complicated
            # we have a tensordict of shape [B, T]
            # we will split / pad things such that we get a tensordict of shape
            # [N, T'] where T' <= T and N >= B is the new batch size, such that
            # each index of N is an independent trajectory. We'll need to keep
            # track of the indices though, as we want to put things back together in the end.
            splits = _get_num_per_traj_init(is_init)
            tensordict_shaped_shape = tensordict_shaped.shape
            tensordict_shaped = _split_and_pad_sequence(
                tensordict_shaped.select(*self.in_keys, strict=False), splits
            )
            is_init = tensordict_shaped["is_init"].squeeze(-1)

        value, hidden0, hidden1 = (
            tensordict_shaped.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        # packed sequences do not help to get the accurate last hidden values
        # if splits is not None:
        #     value = torch.nn.utils.rnn.pack_padded_sequence(value, splits, batch_first=True)

        if not self.recurrent_mode and hidden0 is not None:
            # We zero the hidden states if we're calling the lstm recursively
            #  as we assume the hidden state comes from the previous trajectory.
            #  When using the recurrent_mode=True option, the lstm can be called from
            #  any intermediate state, hence zeroing should not be done.
            is_init_expand = expand_as_right(is_init, hidden0)
            zeros = torch.zeros_like(hidden0)
            hidden0 = torch.where(is_init_expand, zeros, hidden0)
            hidden1 = torch.where(is_init_expand, zeros, hidden1)

        batch, steps = value.shape[:2]
        device = value.device
        dtype = value.dtype

        val, hidden0, hidden1 = self._lstm(
            value,
            batch,
            steps,
            device,
            dtype,
            hidden0,
            hidden1,
            splits,
            is_init=is_init if (use_scan or use_triton) else None,
            backend=backend if self.recurrent_mode else "pad",
        )
        tensordict_shaped.set(self.out_keys[0], val)
        tensordict_shaped.set(self.out_keys[1], hidden0)
        tensordict_shaped.set(self.out_keys[2], hidden1)
        if splits is not None:
            # let's recover our original shape
            tensordict_shaped = _inv_pad_sequence(tensordict_shaped, splits).reshape(
                tensordict_shaped_shape
            )

        if shape != tensordict_shaped.shape or tensordict_shaped is not tensordict:
            tensordict.update(tensordict_shaped.reshape(shape))
        return tensordict

    def _lstm(
        self,
        input: torch.Tensor,
        batch,
        steps,
        device,
        dtype,
        hidden0_in: torch.Tensor | None = None,
        hidden1_in: torch.Tensor | None = None,
        splits: torch.Tensor | None = None,
        is_init: torch.Tensor | None = None,
        backend: str = "pad",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.recurrent_mode and steps != 1:
            raise ValueError("Expected a single step")

        if hidden1_in is None and hidden0_in is None:
            shape = (batch, steps)
            num_states = self.lstm.num_layers * _num_directions(self.lstm)
            real_hidden_size = (
                self.lstm.proj_size
                if self.lstm.proj_size > 0
                else self.lstm.hidden_size
            )
            hidden0_in, hidden1_in = (
                torch.zeros(
                    *shape,
                    num_states,
                    hidden_size,
                    device=device,
                    dtype=dtype,
                )
                for hidden_size in (real_hidden_size, self.lstm.hidden_size)
            )
        elif hidden1_in is None or hidden0_in is None:
            raise RuntimeError(
                f"got type(hidden0)={type(hidden0_in)} and type(hidden1)={type(hidden1_in)}"
            )

        # we only need the first hidden state
        _hidden0_in = hidden0_in[..., 0, :, :]
        _hidden1_in = hidden1_in[..., 0, :, :]
        hidden = (
            _hidden0_in.transpose(-3, -2).contiguous(),
            _hidden1_in.transpose(-3, -2).contiguous(),
        )

        if is_init is not None and backend == "triton":
            return self._lstm_triton_with_resets(input, hidden0_in, hidden1_in, is_init)
        if is_init is not None:
            return self._lstm_scan_with_resets(
                input, hidden0_in, hidden1_in, hidden, is_init
            )
        if splits is None:
            y, hidden = self.lstm(input, hidden)
        elif isinstance(self.lstm, nn.LSTM):
            # Variable-length trajectories: pack so the LSTM does not consume
            # padding zeros, and h_n/c_n reflect the state after the last real
            # step of each trajectory rather than after the padded tail.
            lengths = splits.detach().to(device="cpu", dtype=torch.long)
            packed = nn.utils.rnn.pack_padded_sequence(
                input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_y, hidden = self.lstm(packed, hidden)
            y, _ = nn.utils.rnn.pad_packed_sequence(
                packed_y, batch_first=True, total_length=steps
            )
        else:
            # python-based custom LSTM does not accept PackedSequence. Run the
            # full padded batch through it but pass a per-step active mask so
            # the cell freezes h/c for batch entries whose trajectory has
            # already ended (wasteful compute on the padded tail, but the
            # batch dimension stays vectorised -- vmap/compile-friendly).
            mask = torch.arange(steps, device=device).unsqueeze(0) < splits.unsqueeze(1)
            y, hidden = self.lstm(input, hidden, mask=mask)

        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = tuple(_h.transpose(0, 1) for _h in hidden)

        out = [y, *hidden]
        # Place hidden states so that _inv_pad_sequence (which keeps the first
        # splits[i] positions of each row) retains them.
        for i in range(1, 3):
            if splits is not None:
                out[i] = _place_at_traj_end(out[i], splits, steps)
            else:
                out[i] = torch.stack(
                    [torch.zeros_like(out[i]) for _ in range(steps - 1)] + [out[i]],
                    1,
                )
        return tuple(out)

    def _lstm_triton_with_resets(
        self,
        input: torch.Tensor,
        hidden0_in: torch.Tensor,
        hidden1_in: torch.Tensor,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.lstm.bidirectional or self.lstm.proj_size:
            return self._lstm_pad_with_resets(input, hidden0_in, hidden1_in, is_init)
        from torchrl.modules.tensordict_module._rnn_triton import lstm_triton

        if self.lstm.bidirectional:
            raise RuntimeError(
                "Triton LSTM layer composition expects unidirectional weights."
            )

        layer_input = input
        hidden0_layers = []
        hidden1_layers = []
        for layer in range(self.lstm.num_layers):
            weights = self.lstm._all_weights[layer]
            w_ih = getattr(self.lstm, weights[0])
            w_hh = getattr(self.lstm, weights[1])
            b_ih = getattr(self.lstm, weights[2]) if self.lstm.bias else None
            b_hh = getattr(self.lstm, weights[3]) if self.lstm.bias else None
            if b_ih is None or b_hh is None:
                zeros = torch.zeros(
                    4 * self.lstm.hidden_size, device=input.device, dtype=input.dtype
                )
                b_ih = zeros if b_ih is None else b_ih
                b_hh = zeros if b_hh is None else b_hh

            hidden_per_step = hidden0_in[..., layer, :]
            cell_per_step = hidden1_in[..., layer, :]

            h_steps, c_steps, _, _ = lstm_triton(
                layer_input,
                hidden_per_step,
                cell_per_step,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                is_init,
                compute_dtype=self.recurrent_compute_dtype,
            )
            hidden0_layers.append(h_steps)
            hidden1_layers.append(c_steps)
            if layer < self.lstm.num_layers - 1 and self.lstm.dropout:
                layer_input = F.dropout(
                    h_steps, p=self.lstm.dropout, training=self.lstm.training
                )
            else:
                layer_input = h_steps

        # Match the per-step "next hidden" semantics used by the scan backend:
        # the [b, t] hidden slot is populated only at trajectory ends.
        end_mask = _end_mask_from_is_init(is_init)
        hidden0_steps = torch.stack(hidden0_layers, -2)
        hidden1_steps = torch.stack(hidden1_layers, -2)
        hidden0_steps = torch.where(
            end_mask.unsqueeze(-1).unsqueeze(-1),
            hidden0_steps,
            torch.zeros_like(hidden0_steps),
        )
        hidden1_steps = torch.where(
            end_mask.unsqueeze(-1).unsqueeze(-1),
            hidden1_steps,
            torch.zeros_like(hidden1_steps),
        )
        return layer_input, hidden0_steps, hidden1_steps

    def _lstm_pad_with_resets(
        self,
        input: torch.Tensor,
        hidden0_in: torch.Tensor,
        hidden1_in: torch.Tensor,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchrl.objectives.value.functional import (
            _inv_pad_sequence,
            _split_and_pad_sequence,
        )
        from torchrl.objectives.value.utils import _get_num_per_traj_init

        # The outer forward path intentionally skips split/pad when
        # ``backend='triton'``. Configurations handled here need pad semantics,
        # so the split/pad work is redone locally before re-entering ``_lstm``.
        splits = _get_num_per_traj_init(is_init)
        batch, steps = input.shape[:2]
        # Private synthetic keys avoid collisions with user-provided in/out keys
        # while this helper reshapes the data through TensorDict utilities.
        source = TensorDict(
            {
                "_input": input,
                "_hidden0": hidden0_in,
                "_hidden1": hidden1_in,
                "is_init": is_init.unsqueeze(-1),
            },
            [batch, steps],
        )
        padded = _split_and_pad_sequence(source, splits)
        val, hidden0, hidden1 = self._lstm(
            padded["_input"],
            padded.shape[0],
            padded.shape[1],
            input.device,
            input.dtype,
            padded["_hidden0"],
            padded["_hidden1"],
            splits=splits,
            is_init=None,
            backend="pad",
        )
        padded.set("_value_out", val)
        padded.set("_hidden0_out", hidden0)
        padded.set("_hidden1_out", hidden1)
        restored = _inv_pad_sequence(
            padded.select("_value_out", "_hidden0_out", "_hidden1_out"), splits
        ).reshape(batch, steps)
        return (
            restored["_value_out"],
            restored["_hidden0_out"],
            restored["_hidden1_out"],
        )

    def _lstm_scan_with_resets(
        self,
        input: torch.Tensor,
        hidden0_in: torch.Tensor,
        hidden1_in: torch.Tensor,
        initial_hidden: tuple[torch.Tensor, torch.Tensor],
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.lstm.dropout:
            raise NotImplementedError(
                "LSTMModule(recurrent_backend='scan') does not support dropout yet."
            )
        if self.lstm.proj_size:
            raise NotImplementedError(
                "LSTMModule(recurrent_backend='scan') does not support proj_size yet."
            )
        if self.lstm.bidirectional:
            raise ValueError(
                "LSTMModule(recurrent_backend='scan') does not support bidirectional LSTMs yet."
            )

        weight_ihs, weight_hhs, bias_ihs, bias_hhs = [], [], [], []
        for layer in range(self.lstm.num_layers):
            weights = self.lstm._all_weights[layer]
            weight_ihs.append(getattr(self.lstm, weights[0]).clone())
            weight_hhs.append(getattr(self.lstm, weights[1]).clone())
            bias_ihs.append(
                getattr(self.lstm, weights[2]).clone() if self.lstm.bias else None
            )
            bias_hhs.append(
                getattr(self.lstm, weights[3]).clone() if self.lstm.bias else None
            )

        input = input.transpose(0, 1)
        is_init = is_init.transpose(0, 1)
        reset_hidden0 = hidden0_in.transpose(0, 1).transpose(-3, -2).contiguous()
        reset_hidden1 = hidden1_in.transpose(0, 1).transpose(-3, -2).contiguous()
        num_layers = self.lstm.num_layers

        def step(carry, inputs):
            h_layers, c_layers = carry
            x_t, init_t, reset_hidden0_t, reset_hidden1_t = inputs
            init_t = init_t.unsqueeze(0).unsqueeze(-1)
            h_layers = torch.where(init_t, reset_hidden0_t, h_layers)
            c_layers = torch.where(init_t, reset_hidden1_t, c_layers)
            h_unbound = h_layers.unbind(0)
            c_unbound = c_layers.unbind(0)
            new_h = []
            new_c = []
            for layer in range(num_layers):
                h_new, c_new = LSTM._lstm_cell(
                    x_t,
                    h_unbound[layer],
                    c_unbound[layer],
                    weight_ihs[layer],
                    bias_ihs[layer],
                    weight_hhs[layer],
                    bias_hhs[layer],
                )
                new_h.append(h_new)
                new_c.append(c_new)
                x_t = h_new
            new_h = torch.stack(new_h, 0).clone()
            new_c = torch.stack(new_c, 0).clone()
            hidden0_out = new_h.transpose(0, 1).flatten(1).clone()
            hidden1_out = new_c.transpose(0, 1).flatten(1).clone()
            return (new_h, new_c), (x_t.clone(), hidden0_out, hidden1_out)

        _, (outputs, hidden0_steps, hidden1_steps) = _scan(
            step,
            initial_hidden,
            (input, is_init, reset_hidden0, reset_hidden1),
            dim=0,
        )
        outputs = outputs.transpose(0, 1)
        hidden0_steps = hidden0_steps.unflatten(
            -1, (self.lstm.num_layers, self.lstm.hidden_size)
        ).transpose(0, 1)
        hidden1_steps = hidden1_steps.unflatten(
            -1, (self.lstm.num_layers, self.lstm.hidden_size)
        ).transpose(0, 1)
        end_mask = torch.empty_like(is_init.transpose(0, 1))
        end_mask[:, :-1] = is_init.transpose(0, 1)[:, 1:]
        end_mask[:, -1] = True
        hidden0_steps = torch.where(
            end_mask.unsqueeze(-1).unsqueeze(-1),
            hidden0_steps,
            torch.zeros_like(hidden0_steps),
        )
        hidden1_steps = torch.where(
            end_mask.unsqueeze(-1).unsqueeze(-1),
            hidden1_steps,
            torch.zeros_like(hidden1_steps),
        )
        return outputs, hidden0_steps, hidden1_steps


class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell that performs the same operation as nn.LSTMCell but is fully coded in Python.

    .. note::
        This class is implemented without relying on CuDNN, which makes it compatible with :func:`torch.vmap` and :func:`torch.compile`.

    Examples:
        >>> import torch
        >>> from torchrl.modules.tensordict_module.rnn import GRUCell
        >>> device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
        >>> B = 2
        >>> N_IN = 10
        >>> N_OUT = 20
        >>> V = 4  # vector size
        >>> gru_cell = GRUCell(input_size=N_IN, hidden_size=N_OUT, device=device)

        # single call
        >>> x = torch.randn(B, 10, device=device)
        >>> h0 = torch.zeros(B, 20, device=device)
        >>> with torch.no_grad():
        ...     h1 = gru_cell(x, h0)

        # vectorised call - not possible with nn.GRUCell
        >>> def call_gru(x, h):
        ...     h_out = gru_cell(x, h)
        ...     return h_out
        >>> batched_call = torch.vmap(call_gru)
        >>> x = torch.randn(V, B, 10, device=device)
        >>> h0 = torch.zeros(V, B, 20, device=device)
        >>> with torch.no_grad():
        ...     h1 = batched_call(x, h0)
    """

    __doc__ += nn.GRUCell.__doc__

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        ret = self.gru_cell(input, hx)

        if not is_batched:
            ret = ret.squeeze(0)

        return ret

    def gru_cell(self, x, hx):

        x = x.view(-1, x.size(1))

        gate_x = F.linear(x, self.weight_ih, self.bias_ih)
        gate_h = F.linear(hx, self.weight_hh, self.bias_hh)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hx - newgate)

        return hy


# copy GRU
class GRUBase(nn.RNNBase):
    """A Base module for GRU. Inheriting from GRUBase enables compatibility with torch.compile."""

    def __init__(self, *args, **kwargs):
        return super().__init__("GRU", *args, **kwargs)


for attr in nn.GRU.__dict__:
    if attr != "__init__":
        setattr(GRUBase, attr, getattr(nn.GRU, attr))


class GRU(GRUBase):
    """A PyTorch module for executing multiple steps of a multi-layer GRU. The module behaves exactly like :class:`torch.nn.GRU`, but this implementation is exclusively coded in Python.

    .. note::
        This class is implemented without relying on CuDNN, which makes it
        compatible with :func:`torch.vmap` and :func:`torch.compile`.

    Examples:
        >>> import torch
        >>> from torchrl.modules.tensordict_module.rnn import GRU

        >>> device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
        >>> B = 2
        >>> T = 4
        >>> N_IN = 10
        >>> N_OUT = 20
        >>> N_LAYERS = 2
        >>> V = 4  # vector size
        >>> gru = GRU(
        ...     input_size=N_IN,
        ...     hidden_size=N_OUT,
        ...     device=device,
        ...     num_layers=N_LAYERS,
        ... )

        # single call
        >>> x = torch.randn(B, T, N_IN, device=device)
        >>> h0 = torch.zeros(N_LAYERS, B, N_OUT, device=device)
        >>> with torch.no_grad():
        ...     h1 = gru(x, h0)

        # vectorised call - not possible with nn.GRU
        >>> def call_gru(x, h):
        ...     h_out = gru(x, h)
        ...     return h_out
        >>> batched_call = torch.vmap(call_gru)
        >>> x = torch.randn(V, B, T, 10, device=device)
        >>> h0 = torch.zeros(V, N_LAYERS, B, N_OUT, device=device)
        >>> with torch.no_grad():
        ...     h1 = batched_call(x, h0)
    """

    __doc__ += nn.GRU.__doc__

    use_scan: bool = False

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
        use_scan: bool = False,
    ) -> None:

        if bidirectional:
            raise NotImplementedError(
                "Bidirectional LSTMs are not supported yet in this implementation."
            )

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=False,
            device=device,
            dtype=dtype,
        )
        # Opt-in prototype: see :meth:`_gru_scan` and :class:`LSTM`.
        self.use_scan = use_scan

    @staticmethod
    def _gru_cell(x, hx, weight_ih, bias_ih, weight_hh, bias_hh):
        x = x.view(-1, x.size(1))

        gate_x = F.linear(x, weight_ih, bias_ih)
        gate_h = F.linear(hx, weight_hh, bias_hh)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = (i_r + h_r).sigmoid()
        inputgate = (i_i + h_i).sigmoid()
        newgate = (i_n + (resetgate * h_n)).tanh()

        hy = newgate + inputgate * (hx - newgate)

        return hy

    def _gru(self, x, hx, mask=None):

        if self.use_scan:
            return self._gru_scan(x, hx, mask)

        if not self.batch_first:
            x = x.permute(
                1, 0, 2
            )  # Change (seq_len, batch, features) to (batch, seq_len, features)
            if mask is not None:
                mask = mask.permute(1, 0)

        bs, seq_len, input_size = x.size()
        h_t = list(hx.unbind(0))

        weight_ih = []
        weight_hh = []
        bias_ih = []
        bias_hh = []
        for layer in range(self.num_layers):

            # Retrieve weights
            weights = self._all_weights[layer]
            weight_ih.append(getattr(self, weights[0]))
            weight_hh.append(getattr(self, weights[1]))
            if self.bias:
                bias_ih.append(getattr(self, weights[2]))
                bias_hh.append(getattr(self, weights[3]))
            else:
                bias_ih.append(None)
                bias_hh.append(None)

        outputs = []
        mask_unbound = mask.unbind(1) if mask is not None else None

        for t, x_t in enumerate(x.unbind(1)):
            m_t = mask_unbound[t].unsqueeze(-1) if mask_unbound is not None else None
            for layer in range(self.num_layers):
                h_prev = h_t[layer]
                h_new = self._gru_cell(
                    x_t,
                    h_prev,
                    weight_ih[layer],
                    bias_ih[layer],
                    weight_hh[layer],
                    bias_hh[layer],
                )
                if m_t is not None:
                    # Freeze hidden state for batch entries whose trajectory
                    # has already ended (see _lstm for rationale).
                    h_new = torch.where(m_t, h_new, h_prev)
                h_t[layer] = h_new

                # Apply dropout if in training mode and not the last layer
                if layer < self.num_layers - 1 and self.dropout:
                    x_t = F.dropout(h_t[layer], p=self.dropout, training=self.training)
                else:
                    x_t = h_t[layer]

            outputs.append(x_t)

        outputs = torch.stack(outputs, dim=1)
        if not self.batch_first:
            outputs = outputs.permute(
                1, 0, 2
            )  # Change back (batch, seq_len, features) to (seq_len, batch, features)

        return outputs, torch.stack(h_t, 0)

    def _gru_scan(self, x, hx, mask=None):
        """Prototype scan-based time loop. See :meth:`LSTM._lstm_scan`."""
        if self.dropout:
            raise NotImplementedError(
                "GRU(use_scan=True) does not support dropout yet."
            )

        weight_ihs, weight_hhs, bias_ihs, bias_hhs = [], [], [], []
        for layer in range(self.num_layers):
            weights = self._all_weights[layer]
            weight_ihs.append(getattr(self, weights[0]).clone())
            weight_hhs.append(getattr(self, weights[1]).clone())
            bias_ihs.append(getattr(self, weights[2]).clone() if self.bias else None)
            bias_hhs.append(getattr(self, weights[3]).clone() if self.bias else None)

        if self.batch_first:
            x = x.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

        num_layers = self.num_layers

        def step(carry, inputs):
            h_layers = carry  # [num_layers, B, H]
            x_t, m_t = inputs
            m_t = m_t.unsqueeze(-1)
            new_h = []
            h_unbound = h_layers.unbind(0)
            for layer in range(num_layers):
                h_prev = h_unbound[layer]
                h_new = self._gru_cell(
                    x_t,
                    h_prev,
                    weight_ihs[layer],
                    bias_ihs[layer],
                    weight_hhs[layer],
                    bias_hhs[layer],
                )
                h_new = torch.where(m_t, h_new, h_prev)
                new_h.append(h_new)
                x_t = h_new
            new_h = torch.stack(new_h, 0).clone()
            return new_h, x_t.clone()

        h_final, outputs = _scan(step, hx, (x, mask), dim=0)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, h_final

    def forward(self, input, hx=None, mask=None):  # noqa: F811
        if input.dim() != 3:
            raise ValueError(
                f"GRU: Expected input to be 3D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() != 3:
            raise RuntimeError(
                f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
            )
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            hx = torch.zeros(
                self.num_layers,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )

        self.check_forward_args(input, hx, batch_sizes=None)
        result = self._gru(input, hx, mask)

        output = result[0]
        hidden = result[1]

        return output, hidden


class GRUModule(ModuleBase):
    """An embedder for an GRU module.

    This class adds the following functionality to :class:`torch.nn.GRU`:

    - Compatibility with TensorDict: the hidden states are reshaped to match
      the tensordict batch size.
    - Optional multi-step execution: with torch.nn, one has to choose between
      :class:`torch.nn.GRUCell` and :class:`torch.nn.GRU`, the former being
      compatible with single step inputs and the latter being compatible with
      multi-step. This class enables both usages.


    After construction, the module is *not* set in recurrent mode, ie. it will
    expect single steps inputs.

    If in recurrent mode, it is expected that the last dimension of the tensordict
    marks the number of steps. There is no constrain on the dimensionality of the
    tensordict (except that it must be greater than one for temporal inputs).

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        python_based: If ``True``, will use a full Python implementation of the GRU cell. Default: ``False``
        recurrent_backend: backend used in recurrent mode when trajectories reset
            in the middle of a batch. ``"pad"`` keeps the existing split/pad
            strategy. ``"scan"`` uses a scan loop over the time dimension and
            avoids materializing padded trajectory chunks. ``"triton"``
            (prototype, CUDA only) uses Triton kernels where available and
            otherwise preserves pad-backend recurrent semantics for dropout
            and bidirectional layers.
            ``"auto"`` uses ``"pad"`` in eager mode and ``"scan"`` when called
            under :func:`torch.compile`. Default: ``"pad"``.
        recurrent_compute_dtype: dtype used for the recurrent matmul inside the
            ``"triton"`` backend (``torch.float32`` -> TF32 on H100, default;
            ``torch.bfloat16`` -> bigger SMEM margin, lower precision).
            Ignored by the other backends. Default: ``torch.float32``.

    Keyword Args:
        in_key (str or tuple of str): the input key of the module. Exclusive use
            with ``in_keys``. If provided, the recurrent keys are assumed to be
            ["recurrent_state"] and the ``in_key`` will be
            appended before this.
        in_keys (list of str): a pair of strings corresponding to the input value and recurrent entry.
            Exclusive with ``in_key``.
        out_key (str or tuple of str): the output key of the module. Exclusive use
            with ``out_keys``. If provided, the recurrent keys are assumed to be
            [("recurrent_state")] and the ``out_key`` will be
            appended before these.
        out_keys (list of str): a pair of strings corresponding to the output value,
            first and second hidden key.

            .. note::
               For a better integration with TorchRL's environments, the best naming
               for the output hidden key is ``("next", <custom_key>)``, such
               that the hidden values are passed from step to step during a rollout.

        device (torch.device or compatible): the device of the module.
        gru (torch.nn.GRU, optional): a GRU instance to be wrapped.
            Exclusive with other nn.GRU arguments.
        default_recurrent_mode (bool, optional): if provided, the recurrent mode if it hasn't been overridden
            by the :class:`~torchrl.modules.set_recurrent_mode` context manager / decorator.
            Defaults to ``False``.

    Attributes:
        recurrent_mode: Returns the recurrent mode of the module.

    Methods:
        set_recurrent_mode: controls whether the module should be executed in
            recurrent mode.
        make_tensordict_primer: creates the TensorDictPrimer transforms for the environment to be aware of the
            recurrent states of the RNN.

    .. note:: This module relies on specific ``recurrent_state`` keys being present in the input
        TensorDicts. To generate a :class:`~torchrl.envs.transforms.TensorDictPrimer` transform that will automatically
        add hidden states to the environment TensorDicts, use the method :func:`~torchrl.modules.rnn.GRUModule.make_tensordict_primer`.
        If this class is a submodule in a larger module, the method :func:`~torchrl.modules.utils.get_primers_from_module` can be called
        on the parent module to automatically generate the primer transforms required for all submodules, including this one.

    Examples:
        >>> from torchrl.envs import TransformedEnv, InitTracker
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.modules import MLP
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
        >>> gru_module = GRUModule(
        ...     input_size=env.observation_spec["observation"].shape[-1],
        ...     hidden_size=64,
        ...     in_keys=["observation", "rs"],
        ...     out_keys=["intermediate", ("next", "rs")])
        >>> mlp = MLP(num_cells=[64], out_features=1)
        >>> policy = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
        >>> policy(env.reset())
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                intermediate: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
                is_init: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        rs: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> gru_module_training = gru_module.set_recurrent_mode()
        >>> policy_training = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
        >>> traj_td = env.rollout(3) # some random temporal data
        >>> traj_td = policy_training(traj_td)
        >>> print(traj_td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                intermediate: Tensor(shape=torch.Size([3, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                is_init: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        is_init: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        rs: Tensor(shape=torch.Size([3, 1, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

    """

    DEFAULT_IN_KEYS = ["recurrent_state"]
    DEFAULT_OUT_KEYS = [("next", "recurrent_state")]

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int | None = None,
        num_layers: int = 1,
        bias: bool = True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
        python_based=False,
        recurrent_backend: typing.Literal["auto", "pad", "scan", "triton"] = "pad",
        recurrent_compute_dtype: torch.dtype = torch.float32,
        *,
        in_key=None,
        in_keys=None,
        out_key=None,
        out_keys=None,
        device=None,
        gru=None,
        default_recurrent_mode: bool | None = None,
    ):
        super().__init__()
        if recurrent_backend not in {"auto", "pad", "scan", "triton"}:
            raise ValueError(
                "recurrent_backend must be one of 'auto', 'pad', 'scan' or 'triton'. "
                f"Got {recurrent_backend}."
            )
        if recurrent_backend == "triton" and not _has_triton:
            raise RuntimeError(
                "recurrent_backend='triton' requires the triton package. "
                "Install it with `pip install triton`."
            )
        if gru is not None:
            if not gru.batch_first:
                raise ValueError("The input gru must have batch_first=True.")
            if input_size is not None or hidden_size is not None:
                raise ValueError(
                    "An GRU instance cannot be passed along with class argument."
                )
        else:
            if not batch_first:
                raise ValueError("The input gru must have batch_first=True.")
            if python_based and bidirectional:
                raise ValueError(
                    "python_based=True does not support bidirectional GRUs."
                )

            if python_based:
                gru = GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bias=bias,
                    dropout=dropout,
                    device=device,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            else:
                gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bias=bias,
                    dropout=dropout,
                    device=device,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
        if not ((in_key is None) ^ (in_keys is None)):
            raise ValueError(
                f"Either in_keys or in_key must be specified but not both or none. Got {in_keys} and {in_key} respectively."
            )
        elif in_key:
            in_keys = [in_key, *self.DEFAULT_IN_KEYS]

        if not ((out_key is None) ^ (out_keys is None)):
            raise ValueError(
                f"Either out_keys or out_key must be specified but not both or none. Got {out_keys} and {out_key} respectively."
            )
        elif out_key:
            out_keys = [out_key, *self.DEFAULT_OUT_KEYS]

        in_keys = unravel_key_list(in_keys)
        out_keys = unravel_key_list(out_keys)
        if not isinstance(in_keys, (tuple, list)) or (
            len(in_keys) != 2 and not (len(in_keys) == 3 and in_keys[-1] == "is_init")
        ):
            raise ValueError(
                f"GRUModule expects 3 inputs: a value, and two hidden states (and potentially an 'is_init' marker). Got in_keys {in_keys} instead."
            )
        if not isinstance(out_keys, (tuple, list)) or len(out_keys) != 2:
            raise ValueError(
                f"GRUModule expects 3 outputs: a value, and two hidden states. Got out_keys {out_keys} instead."
            )
        self.gru = gru
        if "is_init" not in in_keys:
            in_keys = in_keys + ["is_init"]
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._recurrent_mode = default_recurrent_mode
        self.recurrent_backend = recurrent_backend
        self.recurrent_compute_dtype = recurrent_compute_dtype

    def make_python_based(self) -> GRUModule:
        """Transforms the GRU layer in its python-based version.

        Returns:
            self

        """
        if isinstance(self.gru, GRU):
            return self
        gru = GRU(
            input_size=self.gru.input_size,
            hidden_size=self.gru.hidden_size,
            num_layers=self.gru.num_layers,
            bias=self.gru.bias,
            dropout=self.gru.dropout,
            device="meta",
            batch_first=self.gru.batch_first,
            bidirectional=self.gru.bidirectional,
        )
        from tensordict import from_module

        from_module(self.gru).to_module(gru)
        self.gru = gru
        return self

    def make_cudnn_based(self) -> GRUModule:
        """Transforms the GRU layer in its CuDNN-based version.

        Returns:
            self

        """
        if isinstance(self.gru, nn.GRU):
            return self
        gru = nn.GRU(
            input_size=self.gru.input_size,
            hidden_size=self.gru.hidden_size,
            num_layers=self.gru.num_layers,
            bias=self.gru.bias,
            dropout=self.gru.dropout,
            device="meta",
            batch_first=self.gru.batch_first,
            bidirectional=self.gru.bidirectional,
        )
        from tensordict import from_module

        from_module(self.gru).to_module(gru)
        self.gru = gru
        return self

    def make_tensordict_primer(self):
        """Makes a tensordict primer for the environment.

        A :class:`~torchrl.envs.TensorDictPrimer` object will ensure that the policy is aware of the supplementary
        inputs and outputs (recurrent states) during rollout execution. That way, the data can be shared across
        processes and dealt with properly.

        Not including a ``TensorDictPrimer`` in the environment may result in poorly defined behaviors, for instance
        in parallel settings where a step involves copying the new recurrent state from ``"next"`` to the root
        tensordict, which the meth:`~torchrl.EnvBase.step_mdp` method will not be able to do as the recurrent states
        are not registered within the environment specs.

        When using batched environments such as :class:`~torchrl.envs.ParallelEnv`, the transform can be used at the
        single env instance level (i.e., a batch of transformed envs with tensordict primers set within) or at the
        batched env instance level (i.e., a transformed batch of regular envs).

        See :func:`torchrl.modules.utils.get_primers_from_module` for a method to generate all primers for a given
        module.

        Examples:
            >>> from torchrl.collectors import Collector
            >>> from torchrl.envs import TransformedEnv, InitTracker
            >>> from torchrl.envs import GymEnv
            >>> from torchrl.modules import MLP, LSTMModule
            >>> from torch import nn
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>>
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
            >>> gru_module = GRUModule(
            ...     input_size=env.observation_spec["observation"].shape[-1],
            ...     hidden_size=64,
            ...     in_keys=["observation", "rs"],
            ...     out_keys=["intermediate", ("next", "rs")])
            >>> mlp = MLP(num_cells=[64], out_features=1)
            >>> policy = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> policy(env.reset())
            >>> env = env.append_transform(gru_module.make_tensordict_primer())
            >>> data_collector = Collector(
            ...     env,
            ...     policy,
            ...     frames_per_batch=10
            ... )
            >>> for data in data_collector:
            ...     print(data)
            ...     break

        """
        from torchrl.envs import TensorDictPrimer

        def make_tuple(key):
            if isinstance(key, tuple):
                return key
            return (key,)

        out_key1 = make_tuple(self.out_keys[1])
        in_key1 = make_tuple(self.in_keys[1])
        if out_key1 != ("next", *in_key1):
            raise RuntimeError(
                "make_tensordict_primer is supposed to work with in_keys/out_keys that "
                "have compatible names, ie. the out_keys should be named after ('next', <in_key>). Got "
                f"in_keys={self.in_keys} and out_keys={self.out_keys} instead."
            )
        return TensorDictPrimer(
            {
                in_key1: Unbounded(
                    shape=(
                        self.gru.num_layers * _num_directions(self.gru),
                        self.gru.hidden_size,
                    )
                ),
            },
            expand_specs=True,
        )

    @property
    def recurrent_mode(self):
        rm = recurrent_mode()
        if rm is None:
            return bool(self._recurrent_mode)
        return rm

    @recurrent_mode.setter
    def recurrent_mode(self, value):
        raise RuntimeError(
            "recurrent_mode cannot be changed in-place. Please use the set_recurrent_mode context manager."
        )

    @property
    def temporal_mode(self):
        raise RuntimeError(
            "temporal_mode is deprecated, use recurrent_mode instead.",
        )

    def set_recurrent_mode(self, mode: bool = True):
        raise RuntimeError(
            "The gru.set_recurrent_mode() API has been removed in v0.8. "
            "To set the recurrent mode, use the :class:`~torchrl.modules.set_recurrent_mode` context manager or "
            "the `default_recurrent_mode` keyword argument in the constructor.",
        )

    @dispatch
    @set_lazy_legacy(False)
    def forward(self, tensordict: TensorDictBase):
        from torchrl.objectives.value.functional import (
            _inv_pad_sequence,
            _split_and_pad_sequence,
        )

        # we want to get an error if the value input is missing, but not the hidden states
        defaults = [NO_DEFAULT, None]
        shape = tensordict.shape
        tensordict_shaped = tensordict
        if self.recurrent_mode:
            # if less than 2 dims, unsqueeze
            ndim = tensordict_shaped.get(self.in_keys[0]).ndim
            while ndim < 3:
                tensordict_shaped = tensordict_shaped.unsqueeze(0)
                ndim += 1
            if ndim > 3:
                dims_to_flatten = ndim - 3
                # we assume that the tensordict can be flattened like this
                nelts = prod(tensordict_shaped.shape[: dims_to_flatten + 1])
                tensordict_shaped = tensordict_shaped.apply(
                    lambda value: value.flatten(0, dims_to_flatten),
                    batch_size=[nelts, tensordict_shaped.shape[-1]],
                )
        else:
            tensordict_shaped = tensordict.reshape(-1).unsqueeze(-1)

        is_init = tensordict_shaped["is_init"].squeeze(-1)
        splits = None
        backend = self.recurrent_backend
        if backend == "auto":
            backend = "scan" if is_compiling() else "pad"
        use_scan = self.recurrent_mode and backend == "scan"
        use_triton = self.recurrent_mode and backend == "triton"
        if (
            self.recurrent_mode
            and not use_scan
            and not use_triton
            and is_init[..., 1:].any()
        ):
            from torchrl.objectives.value.utils import _get_num_per_traj_init

            # if we have consecutive trajectories, things get a little more complicated
            # we have a tensordict of shape [B, T]
            # we will split / pad things such that we get a tensordict of shape
            # [N, T'] where T' <= T and N >= B is the new batch size, such that
            # each index of N is an independent trajectory. We'll need to keep
            # track of the indices though, as we want to put things back together in the end.
            splits = _get_num_per_traj_init(is_init)
            tensordict_shaped_shape = tensordict_shaped.shape
            tensordict_shaped = _split_and_pad_sequence(
                tensordict_shaped.select(*self.in_keys, strict=False), splits
            )
            is_init = tensordict_shaped["is_init"].squeeze(-1)

        value, hidden = (
            tensordict_shaped.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        batch, steps = value.shape[:2]
        device = value.device
        dtype = value.dtype
        if not self.recurrent_mode and is_init.any() and hidden is not None:
            is_init_expand = expand_as_right(is_init, hidden)
            hidden = torch.where(is_init_expand, 0, hidden)
        val, hidden = self._gru(
            value,
            batch,
            steps,
            device,
            dtype,
            hidden,
            splits,
            is_init=is_init if (use_scan or use_triton) else None,
            backend=backend if self.recurrent_mode else "pad",
        )
        tensordict_shaped.set(self.out_keys[0], val)
        tensordict_shaped.set(self.out_keys[1], hidden)
        if splits is not None:
            # let's recover our original shape
            tensordict_shaped = _inv_pad_sequence(tensordict_shaped, splits).reshape(
                tensordict_shaped_shape
            )

        if shape != tensordict_shaped.shape or tensordict_shaped is not tensordict:
            tensordict.update(tensordict_shaped.reshape(shape))
        return tensordict

    def _gru(
        self,
        input: torch.Tensor,
        batch,
        steps,
        device,
        dtype,
        hidden_in: torch.Tensor | None = None,
        splits: torch.Tensor | None = None,
        is_init: torch.Tensor | None = None,
        backend: str = "pad",
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not self.recurrent_mode and steps != 1:
            raise ValueError("Expected a single step")

        if hidden_in is None:
            shape = (batch, steps)
            hidden_in = torch.zeros(
                *shape,
                self.gru.num_layers * _num_directions(self.gru),
                self.gru.hidden_size,
                device=device,
                dtype=dtype,
            )

        # we only need the first hidden state
        _hidden_in = hidden_in[:, 0]
        hidden = _hidden_in.transpose(-3, -2).contiguous()

        if is_init is not None and backend == "triton":
            return self._gru_triton_with_resets(input, hidden_in, is_init)
        if is_init is not None:
            return self._gru_scan_with_resets(input, hidden_in, hidden, is_init)
        if splits is None:
            y, hidden = self.gru(input, hidden)
        elif isinstance(self.gru, nn.GRU):
            # See LSTMModule._lstm for rationale.
            lengths = splits.detach().to(device="cpu", dtype=torch.long)
            packed = nn.utils.rnn.pack_padded_sequence(
                input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_y, hidden = self.gru(packed, hidden)
            y, _ = nn.utils.rnn.pad_packed_sequence(
                packed_y, batch_first=True, total_length=steps
            )
        else:
            mask = torch.arange(steps, device=device).unsqueeze(0) < splits.unsqueeze(1)
            y, hidden = self.gru(input, hidden, mask=mask)

        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = hidden.transpose(0, 1)

        if splits is not None:
            hidden = _place_at_traj_end(hidden, splits, steps)
        else:
            # we pad the hidden states with zero to make tensordict happy
            hidden = torch.stack(
                [torch.zeros_like(hidden) for _ in range(steps - 1)] + [hidden],
                1,
            )
        out = [y, hidden]
        return tuple(out)

    def _gru_triton_with_resets(
        self,
        input: torch.Tensor,
        hidden_in: torch.Tensor,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gru.bidirectional:
            return self._gru_pad_with_resets(input, hidden_in, is_init)
        from torchrl.modules.tensordict_module._rnn_triton import gru_triton

        if self.gru.bidirectional:
            raise RuntimeError(
                "Triton GRU layer composition expects unidirectional weights."
            )

        layer_input = input
        hidden_layers = []
        for layer in range(self.gru.num_layers):
            weights = self.gru._all_weights[layer]
            w_ih = getattr(self.gru, weights[0])
            w_hh = getattr(self.gru, weights[1])
            b_ih = getattr(self.gru, weights[2]) if self.gru.bias else None
            b_hh = getattr(self.gru, weights[3]) if self.gru.bias else None
            if b_ih is None or b_hh is None:
                zeros = torch.zeros(
                    3 * self.gru.hidden_size, device=input.device, dtype=input.dtype
                )
                b_ih = zeros if b_ih is None else b_ih
                b_hh = zeros if b_hh is None else b_hh

            hidden_per_step = hidden_in[..., layer, :]
            h_steps, _ = gru_triton(
                layer_input,
                hidden_per_step,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                is_init,
                compute_dtype=self.recurrent_compute_dtype,
            )
            hidden_layers.append(h_steps)
            if layer < self.gru.num_layers - 1 and self.gru.dropout:
                layer_input = F.dropout(
                    h_steps, p=self.gru.dropout, training=self.gru.training
                )
            else:
                layer_input = h_steps

        # Match the scan backend's per-step hidden output semantics.
        end_mask = _end_mask_from_is_init(is_init)
        hidden_steps = torch.stack(hidden_layers, -2)
        hidden_steps = torch.where(
            end_mask.unsqueeze(-1).unsqueeze(-1),
            hidden_steps,
            torch.zeros_like(hidden_steps),
        )
        return layer_input, hidden_steps

    def _gru_pad_with_resets(
        self,
        input: torch.Tensor,
        hidden_in: torch.Tensor,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torchrl.objectives.value.functional import (
            _inv_pad_sequence,
            _split_and_pad_sequence,
        )
        from torchrl.objectives.value.utils import _get_num_per_traj_init

        # See ``_lstm_pad_with_resets``: this helper owns split/pad because the
        # outer recurrent path bypasses it for ``backend='triton'``.
        splits = _get_num_per_traj_init(is_init)
        batch, steps = input.shape[:2]
        # Private synthetic keys avoid collisions with user-provided in/out keys.
        source = TensorDict(
            {
                "_input": input,
                "_hidden": hidden_in,
                "is_init": is_init.unsqueeze(-1),
            },
            [batch, steps],
        )
        padded = _split_and_pad_sequence(source, splits)
        val, hidden = self._gru(
            padded["_input"],
            padded.shape[0],
            padded.shape[1],
            input.device,
            input.dtype,
            padded["_hidden"],
            splits=splits,
            is_init=None,
            backend="pad",
        )
        padded.set("_value_out", val)
        padded.set("_hidden_out", hidden)
        restored = _inv_pad_sequence(
            padded.select("_value_out", "_hidden_out"), splits
        ).reshape(batch, steps)
        return restored["_value_out"], restored["_hidden_out"]

    def _gru_scan_with_resets(
        self,
        input: torch.Tensor,
        hidden_in: torch.Tensor,
        initial_hidden: torch.Tensor,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gru.dropout:
            raise NotImplementedError(
                "GRUModule(recurrent_backend='scan') does not support dropout yet."
            )
        if self.gru.bidirectional:
            raise ValueError(
                "GRUModule(recurrent_backend='scan') does not support bidirectional GRUs yet."
            )

        weight_ihs, weight_hhs, bias_ihs, bias_hhs = [], [], [], []
        for layer in range(self.gru.num_layers):
            weights = self.gru._all_weights[layer]
            weight_ihs.append(getattr(self.gru, weights[0]).clone())
            weight_hhs.append(getattr(self.gru, weights[1]).clone())
            bias_ihs.append(
                getattr(self.gru, weights[2]).clone() if self.gru.bias else None
            )
            bias_hhs.append(
                getattr(self.gru, weights[3]).clone() if self.gru.bias else None
            )

        input = input.transpose(0, 1)
        is_init = is_init.transpose(0, 1)
        reset_hidden = hidden_in.permute(1, 2, 0, 3).contiguous()
        num_layers = self.gru.num_layers

        def step(carry, inputs):
            h_layers = carry
            x_t, init_t, reset_hidden_t = inputs
            init_t = init_t.unsqueeze(0).unsqueeze(-1)
            h_layers = torch.where(init_t, reset_hidden_t, h_layers)
            h_unbound = h_layers.unbind(0)
            new_h = []
            for layer in range(num_layers):
                h_new = GRU._gru_cell(
                    x_t,
                    h_unbound[layer],
                    weight_ihs[layer],
                    bias_ihs[layer],
                    weight_hhs[layer],
                    bias_hhs[layer],
                )
                new_h.append(h_new)
                x_t = h_new
            # scan returns both carry and per-step outputs; clone to avoid
            # aliasing between those two pytrees under torch.compile.
            new_h = torch.stack(new_h, 0).clone()
            hidden_out = new_h.transpose(0, 1).flatten(1).clone()
            return new_h, (x_t.clone(), hidden_out)

        _, (outputs, hidden_steps) = _scan(
            step, initial_hidden, (input, is_init, reset_hidden), dim=0
        )
        outputs = outputs.transpose(0, 1)
        hidden_steps = hidden_steps.unflatten(
            -1, (self.gru.num_layers, self.gru.hidden_size)
        ).transpose(0, 1)
        end_mask = torch.empty_like(is_init.transpose(0, 1))
        end_mask[:, :-1] = is_init.transpose(0, 1)[:, 1:]
        end_mask[:, -1] = True
        hidden_steps = torch.where(
            end_mask.unsqueeze(-1).unsqueeze(-1),
            hidden_steps,
            torch.zeros_like(hidden_steps),
        )
        return outputs, hidden_steps


# Recurrent mode manager
recurrent_mode_state_manager = _ContextManager()


def recurrent_mode() -> bool | None:
    """Returns the current sampling type."""
    return recurrent_mode_state_manager.get_mode()


class set_recurrent_mode(_DecoratorContextManager):
    """Context manager for setting RNNs recurrent mode.

    Args:
        mode (bool, "recurrent" or "sequential"): the recurrent mode to be used within the context manager.
            `"recurrent"` leads to `mode=True` and `"sequential"` leads to `mode=False`.
            An RNN executed with recurrent_mode "on" assumes that the data comes in time batches, otherwise
            it is assumed that each data element in a tensordict is independent of the others.
            The default value of this context manager is ``True``.
            The default recurrent mode is ``None``, i.e., the default recurrent mode of the RNN is used
            (see :class:`~torchrl.modules.LSTMModule` and :class:`~torchrl.modules.GRUModule` constructors).

    .. seealso:: :class:`~torchrl.modules.recurrent_mode``.

    .. note:: All of TorchRL methods are decorated with ``set_recurrent_mode(True)`` by default.

    """

    def __init__(
        self, mode: bool | typing.Literal["recurrent", "sequential"] | None = True
    ) -> None:
        super().__init__()
        if isinstance(mode, str):
            if mode.lower() in ("recurrent",):
                mode = True
            elif mode.lower() in ("sequential",):
                mode = False
            else:
                raise ValueError(
                    f"Unsupported recurrent mode. Must be a bool, or one of {('recurrent', 'sequential')}"
                )
        self.mode = mode

    def clone(self) -> set_recurrent_mode:
        # override this method if your children class takes __init__ parameters
        return type(self)(self.mode)

    def __enter__(self) -> None:
        self.prev = recurrent_mode_state_manager.get_mode()
        recurrent_mode_state_manager.set_mode(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        recurrent_mode_state_manager.set_mode(self.prev)
