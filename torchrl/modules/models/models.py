# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import dataclasses

from copy import deepcopy
from numbers import Number
from typing import Callable, Dict, List, Sequence, Tuple, Type, Union

import torch
from torch import nn

from torchrl._utils import prod
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules.models.decision_transformer import DecisionTransformer
from torchrl.modules.models.utils import (
    _find_depth,
    create_on_device,
    LazyMapping,
    SquashDims,
    Squeeze2dLayer,
    SqueezeLayer,
)
from torchrl.modules.tensordict_module.common import DistributionalDQNnet  # noqa


class MLP(nn.Sequential):
    """A multi-layer perceptron.

    If MLP receives more than one input, it concatenates them all along the last dimension before passing the
    resulting tensor through the network. This is aimed at allowing for a seamless interface with calls of the type of

        >>> model(state, action)  # compute state-action value

    In the future, this feature may be moved to the ProbabilisticTDModule, though it would require it to handle
    different cases (vectors, images, ...)

    Args:
        in_features (int, optional): number of input features;
        out_features (int, torch.Size or equivalent): number of output
            features. If iterable of integers, the output is reshaped to the
            desired shape.
        depth (int, optional): depth of the network. A depth of 0 will produce
            a single linear layer network with the desired input and output size.
            A length of 1 will create 2 linear layers etc. If no depth is indicated,
            the depth information should be contained in the ``num_cells``
            argument (see below). If ``num_cells`` is an iterable and depth is
            indicated, both should match: ``len(num_cells)`` must be equal to
            ``depth``.
        num_cells (int or sequence of int, optional): number of cells of every
            layer in between the input and output. If an integer is provided,
            every layer will have the same number of cells. If an iterable is provided,
            the linear layers ``out_features`` will match the content of
            ``num_cells``. Defaults to ``32``;
        activation_class (Type[nn.Module] or callable, optional): activation
            class or constructor to be used.
            Defaults to :class:`~torch.nn.Tanh`.
        activation_kwargs (dict or list of dicts, optional): kwargs to be used
            with the activation class. Aslo accepts a list of kwargs of length
            ``depth + int(activate_last_layer)``.
        norm_class (Type or callable, optional): normalization class or
            constructor, if any.
        norm_kwargs (dict or list of dicts, optional): kwargs to be used with
            the normalization layers. Aslo accepts a list of kwargs of length
            ``depth + int(activate_last_layer)``.
        dropout (float, optional): dropout probability. Defaults to ``None`` (no
            dropout);
        bias_last_layer (bool): if ``True``, the last Linear layer will have a bias parameter.
            default: True;
        single_bias_last_layer (bool): if ``True``, the last dimension of the bias of the last layer will be a singleton
            dimension.
            default: True;
        layer_class (Type[nn.Module] or callable, optional): class to be used
            for the linear layers;
        layer_kwargs (dict or list of dicts, optional): kwargs for the linear
            layers. Aslo accepts a list of kwargs of length ``depth + 1``.
        activate_last_layer (bool): whether the MLP output should be activated. This is useful when the MLP output
            is used as the input for another module.
            default: False.
        device (torch.device, optional): device to create the module on.

    Examples:
        >>> # All of the following examples provide valid, working MLPs
        >>> mlp = MLP(in_features=3, out_features=6, depth=0) # MLP consisting of a single 3 x 6 linear layer
        >>> print(mlp)
        MLP(
          (0): Linear(in_features=3, out_features=6, bias=True)
        )
        >>> mlp = MLP(in_features=3, out_features=6, depth=4, num_cells=32)
        >>> print(mlp)
        MLP(
          (0): Linear(in_features=3, out_features=32, bias=True)
          (1): Tanh()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): Tanh()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): Tanh()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): Tanh()
          (8): Linear(in_features=32, out_features=6, bias=True)
        )
        >>> mlp = MLP(out_features=6, depth=4, num_cells=32)  # LazyLinear for the first layer
        >>> print(mlp)
        MLP(
          (0): LazyLinear(in_features=0, out_features=32, bias=True)
          (1): Tanh()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): Tanh()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): Tanh()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): Tanh()
          (8): Linear(in_features=32, out_features=6, bias=True)
        )
        >>> mlp = MLP(out_features=6, num_cells=[32, 33, 34, 35])  # defines the depth by the num_cells arg
        >>> print(mlp)
        MLP(
          (0): LazyLinear(in_features=0, out_features=32, bias=True)
          (1): Tanh()
          (2): Linear(in_features=32, out_features=33, bias=True)
          (3): Tanh()
          (4): Linear(in_features=33, out_features=34, bias=True)
          (5): Tanh()
          (6): Linear(in_features=34, out_features=35, bias=True)
          (7): Tanh()
          (8): Linear(in_features=35, out_features=6, bias=True)
        )
        >>> mlp = MLP(out_features=(6, 7), num_cells=[32, 33, 34, 35])  # returns a view of the output tensor with shape [*, 6, 7]
        >>> print(mlp)
        MLP(
          (0): LazyLinear(in_features=0, out_features=32, bias=True)
          (1): Tanh()
          (2): Linear(in_features=32, out_features=33, bias=True)
          (3): Tanh()
          (4): Linear(in_features=33, out_features=34, bias=True)
          (5): Tanh()
          (6): Linear(in_features=34, out_features=35, bias=True)
          (7): Tanh()
          (8): Linear(in_features=35, out_features=42, bias=True)
        )
        >>> from torchrl.modules import NoisyLinear
        >>> mlp = MLP(out_features=(6, 7), num_cells=[32, 33, 34, 35], layer_class=NoisyLinear)  # uses NoisyLinear layers
        >>> print(mlp)
        MLP(
          (0): NoisyLazyLinear(in_features=0, out_features=32, bias=False)
          (1): Tanh()
          (2): NoisyLinear(in_features=32, out_features=33, bias=True)
          (3): Tanh()
          (4): NoisyLinear(in_features=33, out_features=34, bias=True)
          (5): Tanh()
          (6): NoisyLinear(in_features=34, out_features=35, bias=True)
          (7): Tanh()
          (8): NoisyLinear(in_features=35, out_features=42, bias=True)
        )

    """

    def __init__(
        self,
        in_features: int | None = None,
        out_features: int | torch.Size = None,
        depth: int | None = None,
        num_cells: Sequence[int] | int | None = None,
        activation_class: Type[nn.Module] | Callable = nn.Tanh,
        activation_kwargs: dict | List[dict] | None = None,
        norm_class: Type[nn.Module] | Callable | None = None,
        norm_kwargs: dict | List[dict] | None = None,
        dropout: float | None = None,
        bias_last_layer: bool = True,
        single_bias_last_layer: bool = False,
        layer_class: Type[nn.Module] | Callable = nn.Linear,
        layer_kwargs: dict | None = None,
        activate_last_layer: bool = False,
        device: DEVICE_TYPING | None = None,
    ):
        if out_features is None:
            raise ValueError("out_features must be specified for MLP.")

        default_num_cells = 32
        if num_cells is None:
            if depth is None:
                num_cells = [default_num_cells] * 3
                depth = 3
            else:
                num_cells = [default_num_cells] * depth

        self.in_features = in_features

        _out_features_num = out_features
        if not isinstance(out_features, Number):
            _out_features_num = prod(out_features)
        self.out_features = out_features
        self._out_features_num = _out_features_num
        self.activation_class = activation_class
        self.norm_class = norm_class
        self.dropout = dropout
        self.bias_last_layer = bias_last_layer
        self.single_bias_last_layer = single_bias_last_layer
        self.layer_class = layer_class

        self.activation_kwargs = activation_kwargs
        self.norm_kwargs = norm_kwargs
        self.layer_kwargs = layer_kwargs

        self.activate_last_layer = activate_last_layer
        if single_bias_last_layer:
            raise NotImplementedError

        if not (isinstance(num_cells, Sequence) or depth is not None):
            raise RuntimeError(
                "If num_cells is provided as an integer, \
            depth must be provided too."
            )
        self.num_cells = (
            list(num_cells) if isinstance(num_cells, Sequence) else [num_cells] * depth
        )
        self.depth = depth if depth is not None else len(self.num_cells)
        if not (len(self.num_cells) == depth or depth is None):
            raise RuntimeError(
                "depth and num_cells length conflict, \
            consider matching or specifying a constant num_cells argument together with a a desired depth"
            )

        self._activation_kwargs_iter = _iter_maybe_over_single(
            activation_kwargs, n=self.depth + self.activate_last_layer
        )
        self._norm_kwargs_iter = _iter_maybe_over_single(
            norm_kwargs, n=self.depth + self.activate_last_layer
        )
        self._layer_kwargs_iter = _iter_maybe_over_single(
            layer_kwargs, n=self.depth + 1
        )
        layers = self._make_net(device)
        layers = [
            layer if isinstance(layer, nn.Module) else _ExecutableLayer(layer)
            for layer in layers
        ]
        super().__init__(*layers)

    def _make_net(self, device: DEVICE_TYPING | None) -> List[nn.Module]:
        layers = []
        in_features = [self.in_features] + self.num_cells
        out_features = self.num_cells + [self._out_features_num]
        for i, (_in, _out) in enumerate(zip(in_features, out_features)):
            layer_kwargs = next(self._layer_kwargs_iter)
            _bias = layer_kwargs.pop(
                "bias", self.bias_last_layer if i == self.depth else True
            )
            if _in is not None:
                layers.append(
                    create_on_device(
                        self.layer_class,
                        device,
                        _in,
                        _out,
                        bias=_bias,
                        **layer_kwargs,
                    )
                )
            else:
                try:
                    lazy_version = LazyMapping[self.layer_class]
                except KeyError:
                    raise KeyError(
                        f"The lazy version of {self.layer_class.__name__} is not implemented yet. "
                        "Consider providing the input feature dimensions explicitely when creating an MLP module"
                    )
                layers.append(
                    create_on_device(
                        lazy_version, device, _out, bias=_bias, **layer_kwargs
                    )
                )

            if i < self.depth or self.activate_last_layer:
                norm_kwargs = next(self._norm_kwargs_iter)
                activation_kwargs = next(self._activation_kwargs_iter)
                if self.dropout is not None:
                    layers.append(create_on_device(nn.Dropout, device, p=self.dropout))
                if self.norm_class is not None:
                    layers.append(
                        create_on_device(self.norm_class, device, **norm_kwargs)
                    )
                layers.append(
                    create_on_device(self.activation_class, device, **activation_kwargs)
                )

        return layers

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)

        out = super().forward(*inputs)
        if not isinstance(self.out_features, Number):
            out = out.view(*out.shape[:-1], *self.out_features)
        return out


class ConvNet(nn.Sequential):
    """A convolutional neural network.

    Args:
        in_features (int, optional): number of input features. If ``None``, a
            :class:`~torch.nn.LazyConv2d` module is used for the first layer.;
        depth (int, optional): depth of the network. A depth of 1 will produce
            a single linear layer network with the desired input size, and
            with an output size equal to the last element of the num_cells
            argument.
            If no depth is indicated, the depth information should be contained
            in the ``num_cells`` argument (see below).
            If ``num_cells`` is an iterable and ``depth`` is indicated, both
            should match: ``len(num_cells)`` must be equal to the ``depth``.
        num_cells (int or Sequence of int, optional): number of cells of
            every layer in between the input and output. If an integer is
            provided, every layer will have the same number of cells. If an
            iterable is provided, the linear layers ``out_features`` will match
            the content of num_cells. Defaults to ``[32, 32, 32]``.
        kernel_sizes (int, sequence of int, optional): Kernel size(s) of the
            conv network. If iterable, the length must match the depth,
            defined by the ``num_cells`` or depth arguments.
            Defaults to ``3``.
        strides (int or sequence of int, optional): Stride(s) of the conv network. If
            iterable, the length must match the depth, defined by the
            ``num_cells`` or depth arguments. Defaults to ``1``.
        activation_class (Type[nn.Module] or callable, optional): activation
            class or constructor to be used.
            Defaults to :class:`~torch.nn.Tanh`.
        activation_kwargs (dict or list of dicts, optional): kwargs to be used
            with the activation class. A list of kwargs of length ``depth``
            can also be passed, with one element per layer.
        norm_class (Type or callable, optional): normalization class or
            constructor, if any.
        norm_kwargs (dict or list of dicts, optional): kwargs to be used with
            the normalization layers. A list of kwargs of length ``depth`` can
            also be passed, with one element per layer.
        bias_last_layer (bool): if ``True``, the last Linear layer will have a
            bias parameter. Defaults to ``True``.
        aggregator_class (Type[nn.Module] or callable): aggregator class or
            constructor to use at the end of the chain.
            Defaults to :class:`torchrl.modules.utils.models.SquashDims`;
        aggregator_kwargs (dict, optional): kwargs for the
            ``aggregator_class``.
        squeeze_output (bool): whether the output should be squeezed of its
            singleton dimensions.
            Defaults to ``False``.
        device (torch.device, optional): device to create the module on.

    Examples:
        >>> # All of the following examples provide valid, working MLPs
        >>> cnet = ConvNet(in_features=3, depth=1, num_cells=[32,]) # MLP consisting of a single 3 x 6 linear layer
        >>> print(cnet)
        ConvNet(
          (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
          (1): ELU(alpha=1.0)
          (2): SquashDims()
        )
        >>> cnet = ConvNet(in_features=3, depth=4, num_cells=32)
        >>> print(cnet)
        ConvNet(
          (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
          (1): ELU(alpha=1.0)
          (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
          (3): ELU(alpha=1.0)
          (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
          (5): ELU(alpha=1.0)
          (6): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
          (7): ELU(alpha=1.0)
          (8): SquashDims()
        )
        >>> cnet = ConvNet(in_features=3, num_cells=[32, 33, 34, 35])  # defines the depth by the num_cells arg
        >>> print(cnet)
        ConvNet(
          (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
          (1): ELU(alpha=1.0)
          (2): Conv2d(32, 33, kernel_size=(3, 3), stride=(1, 1))
          (3): ELU(alpha=1.0)
          (4): Conv2d(33, 34, kernel_size=(3, 3), stride=(1, 1))
          (5): ELU(alpha=1.0)
          (6): Conv2d(34, 35, kernel_size=(3, 3), stride=(1, 1))
          (7): ELU(alpha=1.0)
          (8): SquashDims()
        )
        >>> cnet = ConvNet(in_features=3, num_cells=[32, 33, 34, 35], kernel_sizes=[3, 4, 5, (2, 3)])  # defines kernels, possibly rectangular
        >>> print(cnet)
        ConvNet(
          (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
          (1): ELU(alpha=1.0)
          (2): Conv2d(32, 33, kernel_size=(4, 4), stride=(1, 1))
          (3): ELU(alpha=1.0)
          (4): Conv2d(33, 34, kernel_size=(5, 5), stride=(1, 1))
          (5): ELU(alpha=1.0)
          (6): Conv2d(34, 35, kernel_size=(2, 3), stride=(1, 1))
          (7): ELU(alpha=1.0)
          (8): SquashDims()
        )

    """

    def __init__(
        self,
        in_features: int | None = None,
        depth: int | None = None,
        num_cells: Sequence[int] | int = None,
        kernel_sizes: Union[Sequence[int], int] = 3,
        strides: Sequence[int] | int = 1,
        paddings: Sequence[int] | int = 0,
        activation_class: Type[nn.Module] | Callable = nn.ELU,
        activation_kwargs: dict | List[dict] | None = None,
        norm_class: Type[nn.Module] | Callable | None = None,
        norm_kwargs: dict | List[dict] | None = None,
        bias_last_layer: bool = True,
        aggregator_class: Type[nn.Module] | Callable | None = SquashDims,
        aggregator_kwargs: dict | None = None,
        squeeze_output: bool = False,
        device: DEVICE_TYPING | None = None,
    ):
        if num_cells is None:
            num_cells = [32, 32, 32]

        self.in_features = in_features
        self.activation_class = activation_class
        self.norm_class = norm_class
        self.bias_last_layer = bias_last_layer
        self.aggregator_class = aggregator_class
        self.aggregator_kwargs = (
            aggregator_kwargs if aggregator_kwargs is not None else {"ndims_in": 3}
        )
        self.squeeze_output = squeeze_output
        # self.single_bias_last_layer = single_bias_last_layer

        self.activation_kwargs = (
            activation_kwargs if activation_kwargs is not None else {}
        )
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}

        depth = _find_depth(depth, num_cells, kernel_sizes, strides, paddings)
        self.depth = depth
        if depth == 0:
            raise ValueError("Null depth is not permitted with ConvNet.")

        for _field, _value in zip(
            ["num_cells", "kernel_sizes", "strides", "paddings"],
            [num_cells, kernel_sizes, strides, paddings],
        ):
            _depth = depth
            setattr(
                self,
                _field,
                (_value if isinstance(_value, Sequence) else [_value] * _depth),
            )
            if not (isinstance(_value, Sequence) or _depth is not None):
                raise RuntimeError(
                    f"If {_field} is provided as an integer, "
                    "depth must be provided too."
                )
            if not (len(getattr(self, _field)) == _depth or _depth is None):
                raise RuntimeError(
                    f"depth={depth} and {_field}={len(getattr(self, _field))} length conflict, "
                    + f"consider matching or specifying a constant {_field} argument together with a a desired depth"
                )

        self.out_features = self.num_cells[-1]

        self.depth = len(self.kernel_sizes)

        self._activation_kwargs_iter = _iter_maybe_over_single(
            activation_kwargs, n=self.depth
        )
        self._norm_kwargs_iter = _iter_maybe_over_single(norm_kwargs, n=self.depth)

        layers = self._make_net(device)
        layers = [
            layer if isinstance(layer, nn.Module) else _ExecutableLayer(layer)
            for layer in layers
        ]
        super().__init__(*layers)

    def _make_net(self, device: DEVICE_TYPING | None) -> nn.Module:
        layers = []
        in_features = [self.in_features] + list(self.num_cells[: self.depth])
        out_features = list(self.num_cells) + [self.out_features]
        kernel_sizes = self.kernel_sizes
        strides = self.strides
        paddings = self.paddings
        for i, (_in, _out, _kernel, _stride, _padding) in enumerate(
            zip(in_features, out_features, kernel_sizes, strides, paddings)
        ):
            _bias = (i < len(in_features) - 1) or self.bias_last_layer
            if _in is not None:
                layers.append(
                    nn.Conv2d(
                        _in,
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        bias=_bias,
                        padding=_padding,
                        device=device,
                    )
                )
            else:
                layers.append(
                    nn.LazyConv2d(
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        bias=_bias,
                        padding=_padding,
                        device=device,
                    )
                )

            activation_kwargs = next(self._activation_kwargs_iter)
            layers.append(
                create_on_device(self.activation_class, device, **activation_kwargs)
            )
            if self.norm_class is not None:
                norm_kwargs = next(self._norm_kwargs_iter)
                layers.append(create_on_device(self.norm_class, device, **norm_kwargs))

        if self.aggregator_class is not None:
            layers.append(
                create_on_device(
                    self.aggregator_class, device, **self.aggregator_kwargs
                )
            )

        if self.squeeze_output:
            layers.append(Squeeze2dLayer())
        return layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, C, L, W = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = super(ConvNet, self).forward(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out


Conv2dNet = ConvNet


class Conv3dNet(nn.Sequential):
    """A 3D-convolutional neural network.

    Args:
        in_features (int, optional): number of input features. A lazy
            implementation that automatically retrieves the input size will be
            used if none is provided.
        depth (int, optional): depth of the network. A depth of ``1`` will
            produce a single linear layer network with the desired input size,
            and with an output size equal to the last element of the
            ``num_cells`` argument. If no ``depth`` is indicated, the ``depth``
            information should be contained in the ``num_cells`` argument
            (see below).
            If ``num_cells`` is an iterable and ``depth`` is indicated,
            both should match: ``len(num_cells)`` must be equal to
            the ``depth``.
        num_cells (int or sequence of int, optional): number of cells of every
            layer in between the input and output. If an integer is provided,
            every layer will have the same number of cells and the depth will
            be retrieved from ``depth``. If an iterable is
            provided, the linear layers ``out_features`` will match the content
            of num_cells. Defaults to ``[32, 32, 32]`` or ``[32] * depth` is
            depth is not ``None``.
        kernel_sizes (int, sequence of int, optional): Kernel size(s) of the
            conv network. If iterable, the length must match the depth,
            defined by the ``num_cells`` or depth arguments. Defaults to ``3``.
        strides (int or sequence of int): Stride(s) of the conv network.
            If iterable, the length must match the depth, defined by the
            ``num_cells`` or depth arguments. Defaults to ``1``.
        activation_class (Type[nn.Module] or callable): activation class or
            constructor to be used. Defaults to :class:`~torch.nn.Tanh`.
        activation_kwargs (dict or list of dicts, optional): kwargs to be used
            with the activation class. A list of kwargs of length ``depth``
            with one element per layer can also be provided.
        norm_class (Type or callable, optional): normalization class, if any.
        norm_kwargs (dict or list of dicts, optional): kwargs to be used with
            the normalization layers. A list of kwargs of length ``depth``
            with one element per layer can also be provided.
        bias_last_layer (bool): if ``True``, the last Linear layer will have a
            bias parameter. Defaults to ``True``.
        aggregator_class (Type[nn.Module] or callable): aggregator class or
            constructor to use at the end of the chain. Defaults to
            :class:`~torchrl.modules.models.utils.SquashDims`.
        aggregator_kwargs (dict, optional): kwargs for the ``aggregator_class``
            constructor.
        squeeze_output (bool): whether the output should be squeezed of its
            singleton dimensions. Defaults to ``False``.
        device (torch.device, optional): device to create the module on.

    Examples:
        >>> # All of the following examples provide valid, working MLPs
        >>> cnet = Conv3dNet(in_features=3, depth=1, num_cells=[32,])
        >>> print(cnet)
        Conv3dNet(
            (0): Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (1): ELU(alpha=1.0)
            (2): SquashDims()
        )
        >>> cnet = Conv3dNet(in_features=3, depth=4, num_cells=32)
        >>> print(cnet)
        Conv3dNet(
            (0): Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (1): ELU(alpha=1.0)
            (2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (3): ELU(alpha=1.0)
            (4): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (5): ELU(alpha=1.0)
            (6): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (7): ELU(alpha=1.0)
            (8): SquashDims()
        )
        >>> cnet = Conv3dNet(in_features=3, num_cells=[32, 33, 34, 35])  # defines the depth by the num_cells arg
        >>> print(cnet)
        Conv3dNet(
            (0): Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (1): ELU(alpha=1.0)
            (2): Conv3d(32, 33, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (3): ELU(alpha=1.0)
            (4): Conv3d(33, 34, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (5): ELU(alpha=1.0)
            (6): Conv3d(34, 35, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (7): ELU(alpha=1.0)
            (8): SquashDims()
        )
        >>> cnet = Conv3dNet(in_features=3, num_cells=[32, 33, 34, 35], kernel_sizes=[3, 4, 5, (2, 3, 4)])  # defines kernels, possibly rectangular
        >>> print(cnet)
        Conv3dNet(
            (0): Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            (1): ELU(alpha=1.0)
            (2): Conv3d(32, 33, kernel_size=(4, 4, 4), stride=(1, 1, 1))
            (3): ELU(alpha=1.0)
            (4): Conv3d(33, 34, kernel_size=(5, 5, 5), stride=(1, 1, 1))
            (5): ELU(alpha=1.0)
            (6): Conv3d(34, 35, kernel_size=(2, 3, 4), stride=(1, 1, 1))
            (7): ELU(alpha=1.0)
            (8): SquashDims()
        )

    """

    def __init__(
        self,
        in_features: int | None = None,
        depth: int | None = None,
        num_cells: Sequence[int] | int = None,
        kernel_sizes: Sequence[int] | int = 3,
        strides: Sequence[int] | int = 1,
        paddings: Sequence[int] | int = 0,
        activation_class: Type[nn.Module] | Callable = nn.ELU,
        activation_kwargs: dict | List[dict] | None = None,
        norm_class: Type[nn.Module] | Callable | None = None,
        norm_kwargs: dict | List[dict] | None = None,
        bias_last_layer: bool = True,
        aggregator_class: Type[nn.Module] | Callable | None = SquashDims,
        aggregator_kwargs: dict | None = None,
        squeeze_output: bool = False,
        device: DEVICE_TYPING | None = None,
    ):
        if num_cells is None:
            if depth is None:
                num_cells = [32, 32, 32]
            else:
                num_cells = [32] * depth

        self.in_features = in_features
        self.activation_class = activation_class
        self.norm_class = norm_class

        self.activation_kwargs = (
            activation_kwargs if activation_kwargs is not None else {}
        )
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}

        self.bias_last_layer = bias_last_layer
        self.aggregator_class = aggregator_class
        self.aggregator_kwargs = (
            aggregator_kwargs if aggregator_kwargs is not None else {"ndims_in": 4}
        )
        self.squeeze_output = squeeze_output
        # self.single_bias_last_layer = single_bias_last_layer

        depth = _find_depth(depth, num_cells, kernel_sizes, strides, paddings)
        self.depth = depth
        if depth == 0:
            raise ValueError("Null depth is not permitted with Conv3dNet.")

        for _field, _value in zip(
            ["num_cells", "kernel_sizes", "strides", "paddings"],
            [num_cells, kernel_sizes, strides, paddings],
        ):
            _depth = depth
            setattr(
                self,
                _field,
                (_value if isinstance(_value, Sequence) else [_value] * _depth),
            )
            if not (len(getattr(self, _field)) == _depth or _depth is None):
                raise ValueError(
                    f"depth={depth} and {_field}={len(getattr(self, _field))} length conflict, "
                    + f"consider matching or specifying a constant {_field} argument together with a a desired depth"
                )

        self.out_features = self.num_cells[-1]

        self.depth = len(self.kernel_sizes)

        self._activation_kwargs_iter = _iter_maybe_over_single(
            activation_kwargs, n=self.depth
        )
        self._norm_kwargs_iter = _iter_maybe_over_single(norm_kwargs, n=self.depth)

        layers = self._make_net(device)
        layers = [
            layer if isinstance(layer, nn.Module) else _ExecutableLayer(layer)
            for layer in layers
        ]
        super().__init__(*layers)

    def _make_net(self, device: DEVICE_TYPING | None) -> nn.Module:
        layers = []
        in_features = [self.in_features] + self.num_cells[: self.depth]
        out_features = self.num_cells + [self.out_features]
        kernel_sizes = self.kernel_sizes
        strides = self.strides
        paddings = self.paddings
        for i, (_in, _out, _kernel, _stride, _padding) in enumerate(
            zip(in_features, out_features, kernel_sizes, strides, paddings)
        ):
            _bias = (i < len(in_features) - 1) or self.bias_last_layer
            if _in is not None:
                layers.append(
                    nn.Conv3d(
                        _in,
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        bias=_bias,
                        padding=_padding,
                        device=device,
                    )
                )
            else:
                layers.append(
                    nn.LazyConv3d(
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        bias=_bias,
                        padding=_padding,
                        device=device,
                    )
                )

            activation_kwargs = next(self._activation_kwargs_iter)
            layers.append(
                create_on_device(self.activation_class, device, **activation_kwargs)
            )
            if self.norm_class is not None:
                norm_kwargs = next(self._norm_kwargs_iter)
                layers.append(create_on_device(self.norm_class, device, **norm_kwargs))

        if self.aggregator_class is not None:
            layers.append(
                create_on_device(
                    self.aggregator_class, device, **self.aggregator_kwargs
                )
            )

        if self.squeeze_output:
            layers.append(SqueezeLayer((-3, -2, -1)))
        return layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        try:
            *batch, C, D, L, W = inputs.shape
        except ValueError as err:
            raise ValueError(
                f"The input value of {self.__class__.__name__} must have at least 4 dimensions, got {inputs.ndim} instead."
            ) from err
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = super().forward(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out


class DuelingMlpDQNet(nn.Module):
    """Creates a Dueling MLP Q-network.

    Presented in https://arxiv.org/abs/1511.06581

    Args:
        out_features (int, torch.Size or equivalent): number of features for the advantage network
        out_features_value (int): number of features for the value network.
            Defaults to ``1``.
        mlp_kwargs_feature (dict, optional): kwargs for the feature network.
            Default is

            >>> mlp_kwargs_feature = {
            ...     'num_cells': [256, 256],
            ...     'activation_class': nn.ELU,
            ...     'out_features': 256,
            ...     'activate_last_layer': True,
            ... }

        mlp_kwargs_output (dict, optional): kwargs for the advantage and
            value networks. Default is

            >>> mlp_kwargs_output = {
            ...     "depth": 1,
            ...     "activation_class": nn.ELU,
            ...     "num_cells": 512,
            ...     "bias_last_layer": True,
            ... }

        device (torch.device, optional): device to create the module on.

    Examples:
        >>> import torch
        >>> from torchrl.modules import DuelingMlpDQNet
        >>> # we can ask for a specific output shape
        >>> net = DuelingMlpDQNet(out_features=(3, 2))
        >>> print(net)
        DuelingMlpDQNet(
          (features): MLP(
            (0): LazyLinear(in_features=0, out_features=256, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=256, out_features=256, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=256, out_features=256, bias=True)
            (5): ELU(alpha=1.0)
          )
          (advantage): MLP(
            (0): LazyLinear(in_features=0, out_features=512, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=512, out_features=6, bias=True)
          )
          (value): MLP(
            (0): LazyLinear(in_features=0, out_features=512, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=512, out_features=1, bias=True)
          )
        )
        >>> x = torch.zeros(1, 5)
        >>> y = net(x)
        >>> print(y)
        tensor([[[ 0.0232, -0.0477],
                 [-0.0226, -0.0019],
                 [-0.0314,  0.0069]]], grad_fn=<SubBackward0>)

    """

    def __init__(
        self,
        out_features: int | torch.Size,
        out_features_value: int = 1,
        mlp_kwargs_feature: dict | None = None,
        mlp_kwargs_output: dict | None = None,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()

        mlp_kwargs_feature = (
            mlp_kwargs_feature if mlp_kwargs_feature is not None else {}
        )
        _mlp_kwargs_feature = {
            "num_cells": [256, 256],
            "out_features": 256,
            "activation_class": nn.ELU,
            "activate_last_layer": True,
        }
        _mlp_kwargs_feature.update(mlp_kwargs_feature)
        self.features = MLP(device=device, **_mlp_kwargs_feature)

        _mlp_kwargs_output = {
            "depth": 1,
            "activation_class": nn.ELU,
            "num_cells": 512,
            "bias_last_layer": True,
        }
        mlp_kwargs_output = mlp_kwargs_output if mlp_kwargs_output is not None else {}
        _mlp_kwargs_output.update(mlp_kwargs_output)
        self.out_features = out_features
        self.out_features_value = out_features_value
        self.advantage = MLP(
            out_features=out_features, device=device, **_mlp_kwargs_output
        )
        self.value = MLP(
            out_features=out_features_value, device=device, **_mlp_kwargs_output
        )
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)) and isinstance(
                layer.bias, torch.Tensor
            ):
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DuelingCnnDQNet(nn.Module):
    """Dueling CNN Q-network.

    Presented in https://arxiv.org/abs/1511.06581

    Args:
        out_features (int): number of features for the advantage network.
        out_features_value (int): number of features for the value network.
        cnn_kwargs (dict or list of dicts, optional): kwargs for the feature
            network. Default is

            >>> cnn_kwargs = {
            ...     'num_cells': [32, 64, 64],
            ...     'strides': [4, 2, 1],
            ...     'kernels': [8, 4, 3],
            ... }

        mlp_kwargs (dict or list of dicts, optional): kwargs for the advantage
            and value network. Default is

            >>> mlp_kwargs = {
            ...     "depth": 1,
            ...     "activation_class": nn.ELU,
            ...     "num_cells": 512,
            ...     "bias_last_layer": True,
            ... }

        device (torch.device, optional): device to create the module on.

    Examples:
        >>> import torch
        >>> from torchrl.modules import DuelingCnnDQNet
        >>> net = DuelingCnnDQNet(out_features=20)
        >>> print(net)
        DuelingCnnDQNet(
          (features): ConvNet(
            (0): LazyConv2d(0, 32, kernel_size=(8, 8), stride=(4, 4))
            (1): ELU(alpha=1.0)
            (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
            (3): ELU(alpha=1.0)
            (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
            (5): ELU(alpha=1.0)
            (6): SquashDims()
          )
          (advantage): MLP(
            (0): LazyLinear(in_features=0, out_features=512, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=512, out_features=20, bias=True)
          )
          (value): MLP(
            (0): LazyLinear(in_features=0, out_features=512, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=512, out_features=1, bias=True)
          )
        )
        >>> x = torch.zeros(1, 3, 64, 64)
        >>> y = net(x)
        >>> print(y.shape)
        torch.Size([1, 20])

    """

    def __init__(
        self,
        out_features: int,
        out_features_value: int = 1,
        cnn_kwargs: dict | None = None,
        mlp_kwargs: dict | None = None,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        _cnn_kwargs = {
            "num_cells": [32, 64, 64],
            "strides": [4, 2, 1],
            "kernel_sizes": [8, 4, 3],
        }
        _cnn_kwargs.update(cnn_kwargs)
        self.features = ConvNet(device=device, **_cnn_kwargs)

        _mlp_kwargs = {
            "depth": 1,
            "activation_class": nn.ELU,
            "num_cells": 512,
            "bias_last_layer": True,
        }
        mlp_kwargs = mlp_kwargs if mlp_kwargs is not None else {}
        _mlp_kwargs.update(mlp_kwargs)
        self.out_features = out_features
        self.out_features_value = out_features_value
        self.advantage = MLP(out_features=out_features, device=device, **_mlp_kwargs)
        self.value = MLP(out_features=out_features_value, device=device, **_mlp_kwargs)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)) and isinstance(
                layer.bias, torch.Tensor
            ):
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def ddpg_init_last_layer(
    module: nn.Sequential,
    scale: float = 6e-4,
    device: DEVICE_TYPING | None = None,
) -> None:
    """Initializer for the last layer of DDPG modules.

    Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
    https://arxiv.org/pdf/1509.02971.pdf

    Args:
        module (nn.Module): an actor or critic to be initialized.
        scale (float, optional): the noise scale. Defaults to ``6e-4``.
        device (torch.device, optional): the device where the noise should be
            created. Defaults to the device of the last layer's weight
            parameter.

    Examples:
        >>> from torchrl.modules.models.models import MLP, ddpg_init_last_layer
        >>> mlp = MLP(in_features=4, out_features=5, num_cells=(10, 10))
        >>> # init the last layer of the MLP
        >>> ddpg_init_last_layer(mlp)

    """
    for last_layer in reversed(module):
        if isinstance(last_layer, (nn.Linear, nn.Conv2d)):
            break
    else:
        raise RuntimeError("Could not find a nn.Linear / nn.Conv2d to initialize.")

    last_layer.weight.data.copy_(
        torch.rand_like(last_layer.weight.data, device=device) * scale - scale / 2
    )
    if last_layer.bias is not None:
        last_layer.bias.data.copy_(
            torch.rand_like(last_layer.bias.data, device=device) * scale - scale / 2
        )


class DdpgCnnActor(nn.Module):
    """DDPG Convolutional Actor class.

    Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
    https://arxiv.org/pdf/1509.02971.pdf

    The DDPG Convolutional Actor takes as input an observation (some simple
    transformation of the observed pixels) and returns an action vector from
    it, as well as an observation embedding that can be reused for a value
    estimation. It should be trained to maximise the value returned by the
    DDPG Q Value network.

    Args:
        action_dim (int): length of the action vector.
        conv_net_kwargs (dict or list of dicts, optional): kwargs for the ConvNet.
            Defaults to

            >>> {
            ...     'in_features': None,
            ...     "num_cells": [32, 64, 64],
            ...     "kernel_sizes": [8, 4, 3],
            ...     "strides": [4, 2, 1],
            ...     "paddings": [0, 0, 1],
            ...     'activation_class': torch.nn.ELU,
            ...     'norm_class': None,
            ...     'aggregator_class': SquashDims,
            ...     'aggregator_kwargs': {"ndims_in": 3},
            ...     'squeeze_output': True,
            ... }  #

        mlp_net_kwargs: kwargs for MLP.
            Defaults to:

            >>> {
            ...     'in_features': None,
            ...     'out_features': action_dim,
            ...     'depth': 2,
            ...     'num_cells': 200,
            ...     'activation_class': nn.ELU,
            ...     'bias_last_layer': True,
            ... }

        use_avg_pooling (bool, optional): if ``True``, a
            :class:`~torch.nn.AvgPooling` layer is used to aggregate the
            output. Defaults to ``False``.
        device (torch.device, optional): device to create the module on.

    Examples:
        >>> import torch
        >>> from torchrl.modules import DdpgCnnActor
        >>> actor = DdpgCnnActor(action_dim=4)
        >>> print(actor)
        DdpgCnnActor(
          (convnet): ConvNet(
            (0): LazyConv2d(0, 32, kernel_size=(8, 8), stride=(4, 4))
            (1): ELU(alpha=1.0)
            (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
            (3): ELU(alpha=1.0)
            (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (5): ELU(alpha=1.0)
            (6): SquashDims()
          )
          (mlp): MLP(
            (0): LazyLinear(in_features=0, out_features=200, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=200, out_features=200, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=200, out_features=4, bias=True)
          )
        )
        >>> obs = torch.randn(10, 3, 64, 64)
        >>> action, hidden = actor(obs)
        >>> print(action.shape)
        torch.Size([10, 4])
        >>> print(hidden.shape)
        torch.Size([10, 2304])

    """

    def __init__(
        self,
        action_dim: int,
        conv_net_kwargs: dict | None = None,
        mlp_net_kwargs: dict | None = None,
        use_avg_pooling: bool = False,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        conv_net_default_kwargs = {
            "in_features": None,
            "num_cells": [32, 64, 64],
            "kernel_sizes": [8, 4, 3],
            "strides": [4, 2, 1],
            "paddings": [0, 0, 1],
            "activation_class": nn.ELU,
            "norm_class": None,
            "aggregator_class": SquashDims
            if not use_avg_pooling
            else nn.AdaptiveAvgPool2d,
            "aggregator_kwargs": {"ndims_in": 3}
            if not use_avg_pooling
            else {"output_size": (1, 1)},
            "squeeze_output": use_avg_pooling,
        }
        conv_net_kwargs = conv_net_kwargs if conv_net_kwargs is not None else {}
        conv_net_default_kwargs.update(conv_net_kwargs)
        mlp_net_default_kwargs = {
            "in_features": None,
            "out_features": action_dim,
            "depth": 2,
            "num_cells": 200,
            "activation_class": nn.ELU,
            "bias_last_layer": True,
        }
        mlp_net_kwargs = mlp_net_kwargs if mlp_net_kwargs is not None else {}
        mlp_net_default_kwargs.update(mlp_net_kwargs)
        self.convnet = ConvNet(device=device, **conv_net_default_kwargs)
        self.mlp = MLP(device=device, **mlp_net_default_kwargs)
        ddpg_init_last_layer(self.mlp, 6e-4, device=device)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.convnet(observation)
        action = self.mlp(hidden)
        return action, hidden


class DdpgMlpActor(nn.Module):
    """DDPG Actor class.

    Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
    https://arxiv.org/pdf/1509.02971.pdf

    The DDPG Actor takes as input an observation vector and returns an action from it.
    It is trained to maximise the value returned by the DDPG Q Value network.

    Args:
        action_dim (int): length of the action vector
        mlp_net_kwargs (dict, optional): kwargs for MLP.
            Defaults to

            >>> {
            ...     'in_features': None,
            ...     'out_features': action_dim,
            ...     'depth': 2,
            ...     'num_cells': [400, 300],
            ...     'activation_class': nn.ELU,
            ...     'bias_last_layer': True,
            ... }

        device (torch.device, optional): device to create the module on.

    Examples:
        >>> import torch
        >>> from torchrl.modules import DdpgMlpActor
        >>> actor = DdpgMlpActor(action_dim=4)
        >>> print(actor)
        DdpgMlpActor(
          (mlp): MLP(
            (0): LazyLinear(in_features=0, out_features=400, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=400, out_features=300, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=300, out_features=4, bias=True)
          )
        )
        >>> obs = torch.zeros(10, 6)
        >>> action = actor(obs)
        >>> print(action.shape)
        torch.Size([10, 4])

    """

    def __init__(
        self,
        action_dim: int,
        mlp_net_kwargs: dict | None = None,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        mlp_net_default_kwargs = {
            "in_features": None,
            "out_features": action_dim,
            "depth": 2,
            "num_cells": [400, 300],
            "activation_class": nn.ELU,
            "bias_last_layer": True,
        }
        mlp_net_kwargs = mlp_net_kwargs if mlp_net_kwargs is not None else {}
        mlp_net_default_kwargs.update(mlp_net_kwargs)
        self.mlp = MLP(device=device, **mlp_net_default_kwargs)
        ddpg_init_last_layer(self.mlp, 6e-3, device=device)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.mlp(observation)
        return action


class DdpgCnnQNet(nn.Module):
    """DDPG Convolutional Q-value class.

    Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
    https://arxiv.org/pdf/1509.02971.pdf

    The DDPG Q-value network takes as input an observation and an action, and
    returns a scalar from it.

    Args:
        conv_net_kwargs (dict, optional): kwargs for the
            convolutional network.
            Defaults to

            >>> {
            ...     'in_features': None,
            ...     "num_cells": [32, 64, 128],
            ...     "kernel_sizes": [8, 4, 3],
            ...     "strides": [4, 2, 1],
            ...     "paddings": [0, 0, 1],
            ...     'activation_class': nn.ELU,
            ...     'norm_class': None,
            ...     'aggregator_class': nn.AdaptiveAvgPool2d,
            ...     'aggregator_kwargs': {},
            ...     'squeeze_output': True,
            ... }

        mlp_net_kwargs (dict, optional): kwargs for MLP.
            Defaults to

            >>> {
            ...     'in_features': None,
            ...     'out_features': 1,
            ...     'depth': 2,
            ...     'num_cells': 200,
            ...     'activation_class': nn.ELU,
            ...     'bias_last_layer': True,
            ... }

        use_avg_pooling (bool, optional): if ``True``, a
            :class:`~torch.nn.AvgPooling` layer is used to aggregate the
            output. Default is ``True``.
        device (torch.device, optional): device to create the module on.

    Examples:
        >>> from torchrl.modules import DdpgCnnQNet
        >>> import torch
        >>> net = DdpgCnnQNet()
        >>> print(net)
        DdpgCnnQNet(
          (convnet): ConvNet(
            (0): LazyConv2d(0, 32, kernel_size=(8, 8), stride=(4, 4))
            (1): ELU(alpha=1.0)
            (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
            (3): ELU(alpha=1.0)
            (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (5): ELU(alpha=1.0)
            (6): AdaptiveAvgPool2d(output_size=(1, 1))
            (7): Squeeze2dLayer()
          )
          (mlp): MLP(
            (0): LazyLinear(in_features=0, out_features=200, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=200, out_features=200, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=200, out_features=1, bias=True)
          )
        )
        >>> obs = torch.zeros(1, 3, 64, 64)
        >>> action = torch.zeros(1, 4)
        >>> value = net(obs, action)
        >>> print(value.shape)
        torch.Size([1, 1])


    """

    def __init__(
        self,
        conv_net_kwargs: dict | None = None,
        mlp_net_kwargs: dict | None = None,
        use_avg_pooling: bool = True,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        conv_net_default_kwargs = {
            "in_features": None,
            "num_cells": [32, 64, 128],
            "kernel_sizes": [8, 4, 3],
            "strides": [4, 2, 1],
            "paddings": [0, 0, 1],
            "activation_class": nn.ELU,
            "norm_class": None,
            "aggregator_class": SquashDims
            if not use_avg_pooling
            else nn.AdaptiveAvgPool2d,
            "aggregator_kwargs": {"ndims_in": 3}
            if not use_avg_pooling
            else {"output_size": (1, 1)},
            "squeeze_output": use_avg_pooling,
        }
        conv_net_kwargs = conv_net_kwargs if conv_net_kwargs is not None else {}
        conv_net_default_kwargs.update(conv_net_kwargs)
        mlp_net_default_kwargs = {
            "in_features": None,
            "out_features": 1,
            "depth": 2,
            "num_cells": 200,
            "activation_class": nn.ELU,
            "bias_last_layer": True,
        }
        mlp_net_kwargs = mlp_net_kwargs if mlp_net_kwargs is not None else {}
        mlp_net_default_kwargs.update(mlp_net_kwargs)
        self.convnet = ConvNet(device=device, **conv_net_default_kwargs)
        self.mlp = MLP(device=device, **mlp_net_default_kwargs)
        ddpg_init_last_layer(self.mlp, 6e-4, device=device)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([self.convnet(observation), action], -1)
        value = self.mlp(hidden)
        return value


class DdpgMlpQNet(nn.Module):
    """DDPG Q-value MLP class.

    Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
    https://arxiv.org/pdf/1509.02971.pdf

    The DDPG Q-value network takes as input an observation and an action,
    and returns a scalar from it.
    Because actions are integrated later than observations, two networks are
    created.

    Args:
        mlp_net_kwargs_net1 (dict, optional): kwargs for MLP.
            Defaults to

            >>> {
            ...     'in_features': None,
            ...     'out_features': 400,
            ...     'depth': 0,
            ...     'num_cells': [],
            ...     'activation_class': nn.ELU,
            ...     'bias_last_layer': True,
            ...     'activate_last_layer': True,
            ...     }

        mlp_net_kwargs_net2
            Defaults to

            >>> {
            ...     'in_features': None,
            ...     'out_features': 1,
            ...     'depth': 1,
            ...     'num_cells': [300, ],
            ...     'activation_class': nn.ELU,
            ...     'bias_last_layer': True,
            ... }

        device (torch.device, optional): device to create the module on.

    Examples:
        >>> import torch
        >>> from torchrl.modules import DdpgMlpQNet
        >>> net = DdpgMlpQNet()
        >>> print(net)
        DdpgMlpQNet(
          (mlp1): MLP(
            (0): LazyLinear(in_features=0, out_features=400, bias=True)
            (1): ELU(alpha=1.0)
          )
          (mlp2): MLP(
            (0): LazyLinear(in_features=0, out_features=300, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=300, out_features=1, bias=True)
          )
        )
        >>> obs = torch.zeros(1, 32)
        >>> action = torch.zeros(1, 4)
        >>> value = net(obs, action)
        >>> print(value.shape)
        torch.Size([1, 1])

    """

    def __init__(
        self,
        mlp_net_kwargs_net1: dict | None = None,
        mlp_net_kwargs_net2: dict | None = None,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        mlp1_net_default_kwargs = {
            "in_features": None,
            "out_features": 400,
            "depth": 0,
            "num_cells": [],
            "activation_class": nn.ELU,
            "bias_last_layer": True,
            "activate_last_layer": True,
        }
        mlp_net_kwargs_net1: Dict = (
            mlp_net_kwargs_net1 if mlp_net_kwargs_net1 is not None else {}
        )
        mlp1_net_default_kwargs.update(mlp_net_kwargs_net1)
        self.mlp1 = MLP(device=device, **mlp1_net_default_kwargs)

        mlp2_net_default_kwargs = {
            "in_features": None,
            "out_features": 1,
            "num_cells": [
                300,
            ],
            "activation_class": nn.ELU,
            "bias_last_layer": True,
        }
        mlp_net_kwargs_net2 = (
            mlp_net_kwargs_net2 if mlp_net_kwargs_net2 is not None else {}
        )
        mlp2_net_default_kwargs.update(mlp_net_kwargs_net2)
        self.mlp2 = MLP(device=device, **mlp2_net_default_kwargs)
        ddpg_init_last_layer(self.mlp2, 6e-3, device=device)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = self.mlp2(torch.cat([self.mlp1(observation), action], -1))
        return value


class OnlineDTActor(nn.Module):
    """Online Decision Transformer Actor class.

    Actor class for the Online Decision Transformer to sample actions from
    gaussian distribution as presented inresented in
    `"Online Decision Transformer" <https://arxiv.org/abs/2202.05607.pdf>`_.

    Returns the mean and standard deviation for the gaussian distribution to sample actions from.

    Args:
        state_dim (int): state dimension.
        action_dim (int): action dimension.
        transformer_config (Dict or :class:`DecisionTransformer.DTConfig`):
            config for the GPT2 transformer.
            Defaults to :meth:`~.default_config`.
        device (torch.device, optional): device to use. Defaults to None.

    Examples:
        >>> model = OnlineDTActor(state_dim=4, action_dim=2,
        ...     transformer_config=OnlineDTActor.default_config())
        >>> observation = torch.randn(32, 10, 4)
        >>> action = torch.randn(32, 10, 2)
        >>> return_to_go = torch.randn(32, 10, 1)
        >>> mu, std = model(observation, action, return_to_go)
        >>> mu.shape
        torch.Size([32, 10, 2])
        >>> std.shape
        torch.Size([32, 10, 2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        transformer_config: Dict | DecisionTransformer.DTConfig = None,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        if transformer_config is None:
            transformer_config = self.default_config()
        if isinstance(transformer_config, DecisionTransformer.DTConfig):
            transformer_config = dataclasses.asdict(transformer_config)
        self.transformer = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            config=transformer_config,
        )
        self.action_layer_mean = nn.Linear(
            transformer_config["n_embd"], action_dim, device=device
        )
        self.action_layer_logstd = nn.Linear(
            transformer_config["n_embd"], action_dim, device=device
        )

        self.log_std_min, self.log_std_max = -5.0, 2.0

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.action_layer_mean.apply(weight_init)
        self.action_layer_logstd.apply(weight_init)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = self.transformer(observation, action, return_to_go)
        mu = self.action_layer_mean(hidden_state)
        log_std = self.action_layer_logstd(hidden_state)

        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1.0
        )
        std = log_std.exp()

        return mu, std

    @classmethod
    def default_config(cls):
        """Default configuration for :class:`~OnlineDTActor`."""
        return DecisionTransformer.DTConfig(
            n_embd=512,
            n_layer=4,
            n_head=4,
            n_inner=2048,
            activation="relu",
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )


class DTActor(nn.Module):
    """Decision Transformer Actor class.

    Actor class for the Decision Transformer to output deterministic action as
    presented in `"Decision Transformer" <https://arxiv.org/abs/2202.05607.pdf>`.
    Returns the deterministic actions.

    Args:
        state_dim (int): state dimension.
        action_dim (int): action dimension.
        transformer_config (Dict or :class:`DecisionTransformer.DTConfig`, optional):
            config for the GPT2 transformer.
            Defaults to :meth:`~.default_config`.
        device (torch.device, optional): device to use. Defaults to None.

    Examples:
        >>> model = DTActor(state_dim=4, action_dim=2,
        ...     transformer_config=DTActor.default_config())
        >>> observation = torch.randn(32, 10, 4)
        >>> action = torch.randn(32, 10, 2)
        >>> return_to_go = torch.randn(32, 10, 1)
        >>> output = model(observation, action, return_to_go)
        >>> output.shape
        torch.Size([32, 10, 2])

    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        transformer_config: Dict | DecisionTransformer.DTConfig = None,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        if transformer_config is None:
            transformer_config = self.default_config()
        if isinstance(transformer_config, DecisionTransformer.DTConfig):
            transformer_config = dataclasses.asdict(transformer_config)
        self.transformer = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            config=transformer_config,
        )
        self.action_layer = nn.Linear(
            transformer_config["n_embd"], action_dim, device=device
        )

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.action_layer.apply(weight_init)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
    ) -> torch.Tensor:
        hidden_state = self.transformer(observation, action, return_to_go)
        out = self.action_layer(hidden_state)
        return out

    @classmethod
    def default_config(cls):
        """Default configuration for :class:`~DTActor`."""
        return DecisionTransformer.DTConfig(
            n_embd=512,
            n_layer=4,
            n_head=4,
            n_inner=2048,
            activation="relu",
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )


def _iter_maybe_over_single(item: dict | List[dict] | None, n):
    if item is None:
        return iter([{} for _ in range(n)])
    elif isinstance(item, dict):
        return iter([deepcopy(item) for _ in range(n)])
    else:
        return iter([deepcopy(_item) for _item in item])


class _ExecutableLayer(nn.Module):
    """A thin wrapper around a function to be exectued as a module."""

    def __init__(self, func):
        super(_ExecutableLayer, self).__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(func={self.func})"
