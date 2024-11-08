# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase, unravel_key_list

from tensordict.base import NO_DEFAULT

from tensordict.nn import TensorDictModuleBase as ModuleBase
from tensordict.utils import expand_as_right, prod, set_lazy_legacy

from torch import nn, Tensor
from torch.nn.modules.rnn import RNNCellBase

from torchrl.data.tensor_specs import Unbounded
from torchrl.objectives.value.functional import (
    _inv_pad_sequence,
    _split_and_pad_sequence,
)
from torchrl.objectives.value.utils import _get_num_per_traj_init


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
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
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

    def _lstm(self, x, hx):

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

        for x_t in x.unbind(int(self.batch_first)):
            h_t_out = []
            c_t_out = []

            for layer, (
                weight_ih,
                bias_ih,
                weight_hh,
                bias_hh,
                _h_t,
                _c_t,
            ) in enumerate(zip(weight_ihs, bias_ihs, weight_hhs, bias_hhs, h_t, c_t)):
                # Run cell
                _h_t, _c_t = self._lstm_cell(
                    x_t, _h_t, _c_t, weight_ih, bias_ih, weight_hh, bias_hh
                )
                h_t_out.append(_h_t)
                c_t_out.append(_c_t)

                # Apply dropout if in training mode
                if layer < self.num_layers - 1 and self.dropout:
                    x_t = F.dropout(_h_t, p=self.dropout, training=self.training)
                else:  # No dropout after the last layer
                    x_t = _h_t
            h_t = h_t_out
            c_t = c_t_out
            outputs.append(x_t)

        outputs = torch.stack(outputs, dim=int(self.batch_first))

        return outputs, (torch.stack(h_t_out, 0), torch.stack(c_t_out, 0))

    def forward(self, input, hx=None):  # noqa: F811
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
        return self._lstm(input, hx)


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
      should not be re-used for a consecutive trajectory).
      The reason is that LSTM returns only the last hidden value, which for the
      padded inputs we provide can correspont to a 0-filled input.

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
        >>> from torchrl.modules import MLP
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
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = 1,
        bias: bool = True,
        batch_first=True,
        dropout=0,
        proj_size=0,
        bidirectional=False,
        python_based=False,
        *,
        in_key=None,
        in_keys=None,
        out_key=None,
        out_keys=None,
        device=None,
        lstm=None,
    ):
        super().__init__()
        if lstm is not None:
            if not lstm.batch_first:
                raise ValueError("The input lstm must have batch_first=True.")
            if lstm.bidirectional:
                raise ValueError("The input lstm cannot be bidirectional.")
            if input_size is not None or hidden_size is not None:
                raise ValueError(
                    "An LSTM instance cannot be passed along with class argument."
                )
        else:
            if not batch_first:
                raise ValueError("The input lstm must have batch_first=True.")
            if bidirectional:
                raise ValueError("The input lstm cannot be bidirectional.")
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
                    bidirectional=False,
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
                    bidirectional=False,
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
        self._recurrent_mode = False

    def make_tensordict_primer(self):
        """Makes a tensordict primer for the environment.

        A :class:`~torchrl.envs.TensorDictPrimer` object will ensure that the policy is aware of the supplementary
        inputs and outputs (recurrent states) during rollout execution. That way, the data can be shared across
        processes and dealt with properly.

        Not including a ``TensorDictPrimer`` in the environment may result in poorly defined behaviors, for instance
        in parallel settings where a step involves copying the new recurrent state from ``"next"`` to the root
        tensordict, which the meth:`~torchrl.EnvBase.step_mdp` method will not be able to do as the recurrent states
        are not registered within the environment specs.

        See :func:`torchrl.modules.utils.get_primers_from_module` for a method to generate all primers for a given
        module.

        Examples:
            >>> from torchrl.collectors import SyncDataCollector
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
            >>> data_collector = SyncDataCollector(
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
        return TensorDictPrimer(
            {
                in_key1: Unbounded(shape=(self.lstm.num_layers, self.lstm.hidden_size)),
                in_key2: Unbounded(shape=(self.lstm.num_layers, self.lstm.hidden_size)),
            }
        )

    @property
    def recurrent_mode(self):
        return self._recurrent_mode

    @recurrent_mode.setter
    def recurrent_mode(self, value):
        raise RuntimeError(
            "recurrent_mode cannot be changed in-place. Call `module.set"
        )

    @property
    def temporal_mode(self):
        raise RuntimeError(
            "temporal_mode is deprecated, use recurrent_mode instead.",
        )

    def set_recurrent_mode(self, mode: bool = True):
        """Returns a new copy of the module that shares the same lstm model but with a different ``recurrent_mode`` attribute (if it differs).

        A copy is created such that the module can be used with divergent behavior
        in various parts of the code (inference vs training):

        Examples:
            >>> from torchrl.envs import TransformedEnv, InitTracker, step_mdp
            >>> from torchrl.envs import GymEnv
            >>> from torchrl.modules import MLP
            >>> from tensordict import TensorDict
            >>> from torch import nn
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
            >>> lstm = nn.LSTM(input_size=env.observation_spec["observation"].shape[-1], hidden_size=64, batch_first=True)
            >>> lstm_module = LSTMModule(lstm=lstm, in_keys=["observation", "hidden0", "hidden1"], out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")])
            >>> mlp = MLP(num_cells=[64], out_features=1)
            >>> # building two policies with different behaviors:
            >>> policy_inference = Seq(lstm_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> policy_training = Seq(lstm_module.set_recurrent_mode(True), Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> traj_td = env.rollout(3) # some random temporal data
            >>> traj_td = policy_training(traj_td)
            >>> # let's check that both return the same results
            >>> td_inf = TensorDict({}, traj_td.shape[:-1])
            >>> for td in traj_td.unbind(-1):
            ...     td_inf = td_inf.update(td.select("is_init", "observation", ("next", "observation")))
            ...     td_inf = policy_inference(td_inf)
            ...     td_inf = step_mdp(td_inf)
            ...
            >>> torch.testing.assert_close(td_inf["hidden0"], traj_td[..., -1]["next", "hidden0"])
        """
        if mode is self._recurrent_mode:
            return self
        out = LSTMModule(lstm=self.lstm, in_keys=self.in_keys, out_keys=self.out_keys)
        out._recurrent_mode = mode
        return out

    def forward(self, tensordict: TensorDictBase):
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
        if self.recurrent_mode and is_init[..., 1:].any():
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
        batch, steps = value.shape[:2]
        device = value.device
        dtype = value.dtype
        # packed sequences do not help to get the accurate last hidden values
        # if splits is not None:
        #     value = torch.nn.utils.rnn.pack_padded_sequence(value, splits, batch_first=True)
        if hidden0 is not None:
            is_init_expand = expand_as_right(is_init, hidden0)
            hidden0 = torch.where(is_init_expand, 0, hidden0)
            hidden1 = torch.where(is_init_expand, 0, hidden1)
        val, hidden0, hidden1 = self._lstm(
            value, batch, steps, device, dtype, hidden0, hidden1
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
        hidden0_in: Optional[torch.Tensor] = None,
        hidden1_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.recurrent_mode and steps != 1:
            raise ValueError("Expected a single step")

        if hidden1_in is None and hidden0_in is None:
            shape = (batch, steps)
            hidden0_in, hidden1_in = [
                torch.zeros(
                    *shape,
                    self.lstm.num_layers,
                    self.lstm.hidden_size,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(2)
            ]
        elif hidden1_in is None or hidden0_in is None:
            raise RuntimeError(
                f"got type(hidden0)={type(hidden0_in)} and type(hidden1)={type(hidden1_in)}"
            )

        # we only need the first hidden state
        _hidden0_in = hidden0_in[:, 0]
        _hidden1_in = hidden1_in[:, 0]
        hidden = (
            _hidden0_in.transpose(-3, -2).contiguous(),
            _hidden1_in.transpose(-3, -2).contiguous(),
        )

        y, hidden = self.lstm(input, hidden)
        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = tuple(_h.transpose(0, 1) for _h in hidden)

        out = [y, *hidden]
        # we pad the hidden states with zero to make tensordict happy
        for i in range(1, 3):
            out[i] = torch.stack(
                [torch.zeros_like(out[i]) for _ in range(steps - 1)] + [out[i]],
                1,
            )
        return tuple(out)


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

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
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

    def _gru(self, x, hx):

        if not self.batch_first:
            x = x.permute(
                1, 0, 2
            )  # Change (seq_len, batch, features) to (batch, seq_len, features)

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

        for x_t in x.unbind(1):
            for layer in range(self.num_layers):
                h_t[layer] = self._gru_cell(
                    x_t,
                    h_t[layer],
                    weight_ih[layer],
                    bias_ih[layer],
                    weight_hh[layer],
                    bias_hh[layer],
                )

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

    def forward(self, input, hx=None):  # noqa: F811
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
        result = self._gru(input, hx)

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
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = 1,
        bias: bool = True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
        python_based=False,
        *,
        in_key=None,
        in_keys=None,
        out_key=None,
        out_keys=None,
        device=None,
        gru=None,
    ):
        super().__init__()
        if gru is not None:
            if not gru.batch_first:
                raise ValueError("The input gru must have batch_first=True.")
            if gru.bidirectional:
                raise ValueError("The input gru cannot be bidirectional.")
            if input_size is not None or hidden_size is not None:
                raise ValueError(
                    "An GRU instance cannot be passed along with class argument."
                )
        else:
            if not batch_first:
                raise ValueError("The input gru must have batch_first=True.")
            if bidirectional:
                raise ValueError("The input gru cannot be bidirectional.")

            if python_based:
                gru = GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bias=bias,
                    dropout=dropout,
                    device=device,
                    batch_first=True,
                    bidirectional=False,
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
                    bidirectional=False,
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
        self._recurrent_mode = False

    def make_tensordict_primer(self):
        """Makes a tensordict primer for the environment.

        A :class:`~torchrl.envs.TensorDictPrimer` object will ensure that the policy is aware of the supplementary
        inputs and outputs (recurrent states) during rollout execution. That way, the data can be shared across
        processes and dealt with properly.

        Not including a ``TensorDictPrimer`` in the environment may result in poorly defined behaviors, for instance
        in parallel settings where a step involves copying the new recurrent state from ``"next"`` to the root
        tensordict, which the meth:`~torchrl.EnvBase.step_mdp` method will not be able to do as the recurrent states
        are not registered within the environment specs.

        See :func:`torchrl.modules.utils.get_primers_from_module` for a method to generate all primers for a given
        module.

        Examples:
            >>> from torchrl.collectors import SyncDataCollector
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
            >>> data_collector = SyncDataCollector(
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
                in_key1: Unbounded(shape=(self.gru.num_layers, self.gru.hidden_size)),
            }
        )

    @property
    def recurrent_mode(self):
        return self._recurrent_mode

    @recurrent_mode.setter
    def recurrent_mode(self, value):
        raise RuntimeError(
            "recurrent_mode cannot be changed in-place. Call `module.set"
        )

    @property
    def temporal_mode(self):
        raise RuntimeError(
            "temporal_mode is deprecated, use recurrent_mode instead.",
        )

    def set_recurrent_mode(self, mode: bool = True):
        """Returns a new copy of the module that shares the same gru model but with a different ``recurrent_mode`` attribute (if it differs).

        A copy is created such that the module can be used with divergent behavior
        in various parts of the code (inference vs training):

        Examples:
            >>> from torchrl.envs import GymEnv, TransformedEnv, InitTracker, step_mdp
            >>> from torchrl.modules import MLP
            >>> from tensordict import TensorDict
            >>> from torch import nn
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
            >>> gru = nn.GRU(input_size=env.observation_spec["observation"].shape[-1], hidden_size=64, batch_first=True)
            >>> gru_module = GRUModule(gru=gru, in_keys=["observation", "hidden"], out_keys=["intermediate", ("next", "hidden")])
            >>> mlp = MLP(num_cells=[64], out_features=1)
            >>> # building two policies with different behaviors:
            >>> policy_inference = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> policy_training = Seq(gru_module.set_recurrent_mode(True), Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> traj_td = env.rollout(3) # some random temporal data
            >>> traj_td = policy_training(traj_td)
            >>> # let's check that both return the same results
            >>> td_inf = TensorDict({}, traj_td.shape[:-1])
            >>> for td in traj_td.unbind(-1):
            ...     td_inf = td_inf.update(td.select("is_init", "observation", ("next", "observation")))
            ...     td_inf = policy_inference(td_inf)
            ...     td_inf = step_mdp(td_inf)
            ...
            >>> torch.testing.assert_close(td_inf["hidden"], traj_td[..., -1]["next", "hidden"])
        """
        if mode is self._recurrent_mode:
            return self
        out = GRUModule(gru=self.gru, in_keys=self.in_keys, out_keys=self.out_keys)
        out._recurrent_mode = mode
        return out

    @set_lazy_legacy(False)
    def forward(self, tensordict: TensorDictBase):
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
        if self.recurrent_mode and is_init[..., 1:].any():
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
        # packed sequences do not help to get the accurate last hidden values
        # if splits is not None:
        #     value = torch.nn.utils.rnn.pack_padded_sequence(value, splits, batch_first=True)
        if is_init.any() and hidden is not None:
            is_init_expand = expand_as_right(is_init, hidden)
            hidden = torch.where(is_init_expand, 0, hidden)
        val, hidden = self._gru(value, batch, steps, device, dtype, hidden)
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
        hidden_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.recurrent_mode and steps != 1:
            raise ValueError("Expected a single step")

        if hidden_in is None:
            shape = (batch, steps)
            hidden_in = torch.zeros(
                *shape,
                self.gru.num_layers,
                self.gru.hidden_size,
                device=device,
                dtype=dtype,
            )

        # we only need the first hidden state
        _hidden_in = hidden_in[:, 0]
        hidden = _hidden_in.transpose(-3, -2).contiguous()

        y, hidden = self.gru(input, hidden)
        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = hidden.transpose(0, 1)

        # we pad the hidden states with zero to make tensordict happy
        hidden = torch.stack(
            [torch.zeros_like(hidden) for _ in range(steps - 1)] + [hidden],
            1,
        )
        out = [y, hidden]
        return tuple(out)
