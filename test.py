# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase
from tensordict import TensorDictBase, unravel_key_list

from tensordict.nn import TensorDictModuleBase as ModuleBase

from tensordict.tensordict import NO_DEFAULT
from tensordict.utils import prod

from torch import nn

from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.objectives.value.functional import (
    _inv_pad_sequence,
    _split_and_pad_sequence,
)
from torchrl.objectives.value.utils import _get_num_per_traj_init


class PythonLSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell that performs the same operation as nn.LSTMCell but is
    fully coded in Python.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)` or `(input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial hidden state
        - **c_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial cell state

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next hidden state
        - **c_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next cell state

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)  # (input_size, hidden_size)
        >>> input = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
        >>> hx = torch.randn(3, 20)  # (batch, hidden_size)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(input.size()[0]):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
        >>> output = torch.stack(output, dim=0)
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        if input.dim() not in (1, 2):
            raise ValueError(f"LSTMCell: Expected input to be 1D or 2D, got {input.dim()}D instead")
        if hx is not None:
            for idx, value in enumerate(hx):
                if value.dim() not in (1, 2):
                    raise ValueError(f"LSTMCell: Expected hx[{idx}] to be 1D or 2D, got {value.dim()}D instead")
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        ret = self.lstm_cell(input, hx[0], hx[1])

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret

    def lstm_cell(self, x, hx, cx):
        x = x.view(-1, x.size(1))

        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = ingate.sigmoid()
        forgetgate = forgetgate.sigmoid()
        cellgate = cellgate.tanh()
        outgate = outgate.sigmoid()

        cy = cx * forgetgate + ingate * cellgate

        hy = outgate * cy.tanh()

        return (hy, cy)

if __name__ == "__main__":

    lstm_cell1 = PythonLSTMCell(10, 20)
    lstm_cell2 = nn.LSTMCell(10, 20)

    for k,v in lstm_cell1.named_parameters():
        print(k)

    print()

    for k, v in lstm_cell2.named_parameters():
        print(k)

    input = torch.randn(2, 3, 10)
    hx = torch.randn(3, 20)
    cx = torch.randn(3, 20)
    output1 = []
    output2 = []
    with torch.no_grad():
        for i in range(input.size()[0]):
            hx, cx = lstm_cell1(input[i], (hx, cx))
            output1.append(hx)
            hx, cx = lstm_cell2(input[i], (hx, cx))
            output2.append(hx)
    output1 = torch.stack(output1, dim=0)
    import ipdb; ipdb.set_trace()