from numbers import Number
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchrl.modules.models.utils import Squeeze2dLayer, LazyMapping, SquashDims, _find_depth

__all__ = ["MLP", "ConvNet", "DuelingCnnDQNet", "DistributionalDQNnet", "DdpgCnnActor", "DdpgCnnQNet", "DdpgMlpActor", "DdpgMlpQNet"]


class MLP(nn.Sequential):
    def __init__(
            self,
            in_features=None,
            out_features=None,
            depth=None,
            num_cells=None,
            activation_class=nn.Tanh,
            activation_kwargs={},
            norm_class=None,
            norm_kwargs={},
            bias_last_layer=True,
            single_bias_last_layer=False,
            layer_class=nn.Linear,
            layer_kwargs={},
            activate_last_layer=False,
    ):
        assert out_features is not None, "out_feature must be specified for MLP."

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
            _out_features_num = np.prod(out_features)
        self.out_features = out_features
        self._out_features_num = _out_features_num
        self.activation_class = activation_class
        self.activation_kwargs = activation_kwargs
        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs
        self.bias_last_layer = bias_last_layer
        self.single_bias_last_layer = single_bias_last_layer
        self.layer_class = layer_class
        self.layer_kwargs = layer_kwargs
        self.activate_last_layer = activate_last_layer
        if single_bias_last_layer:
            raise NotImplementedError

        assert (
                isinstance(num_cells, Iterable) or depth is not None
        ), "If num_cells is provided as an integer, \
            depth must be provided too."
        self.num_cells = (
            list(num_cells) if isinstance(num_cells, Iterable) else [num_cells] * depth
        )
        self.depth = depth if depth is not None else len(self.num_cells)
        assert (
                len(self.num_cells) == depth or depth is None
        ), "depth and num_cells length conflict, \
            consider matching or specifying a constan num_cells argument together with a a desired depth"

        layers = self._make_net()
        super().__init__(*layers)

    def _make_net(self):
        layers = []
        in_features = [self.in_features] + self.num_cells
        out_features = self.num_cells + [self._out_features_num]
        for i, (_in, _out) in enumerate(zip(in_features, out_features)):
            _bias = self.bias_last_layer if i == self.depth else True
            if _in is not None:
                layers.append(self.layer_class(_in, _out, bias=_bias, **self.layer_kwargs))
            else:
                try:
                    lazy_version = LazyMapping[self.layer_class]
                except KeyError:
                    raise KeyError(
                        f"The lazy version of {self.layer_class.__name__} is not implemented yet. " "Consider providing the input feature dimensions explicitely when creating an MLP module")
                layers.append(lazy_version(_out, bias=_bias, **self.layer_kwargs))

            if i < self.depth or self.activate_last_layer:
                layers.append(self.activation_class(**self.activation_kwargs))
                if self.norm_class is not None:
                    layers.append(self.norm_class(**self.norm_kwargs))
        return layers

    def forward(self, *inputs):
        out = super().forward(*inputs)
        if not isinstance(self.out_features, Number):
            out = out.view(*out.shape[:-1], *self.out_features)
        if not torch.isfinite(out).all():
            print(out)
        return out


class ConvNet(nn.Sequential):
    def __init__(
            self,
            in_features=None,
            depth=None,
            num_cells=[32, 32, 32],
            kernel_sizes=3,
            strides=1,
            actionvation_class=nn.ReLU,
            activation_kwargs={},
            norm_class=None,
            norm_kwargs={},
            bias_last_layer=True,
            aggregator_class=SquashDims,
            aggregator_kwargs={"ndims_in": 3},
            squeeze_output=True,
    ):

        self.in_features = in_features
        self.activation_class = actionvation_class
        self.activation_kwargs = activation_kwargs
        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs
        self.bias_last_layer = bias_last_layer
        self.aggregator_class = aggregator_class
        self.aggregator_kwargs = aggregator_kwargs
        self.squeeze_output = squeeze_output
        # self.single_bias_last_layer = single_bias_last_layer

        self.depth = depth
        depth = _find_depth(depth, num_cells, kernel_sizes, strides)

        for _field, _value in zip(
                ["num_cells", "kernel_sizes", "strides"],
                [num_cells, kernel_sizes, strides],
        ):
            _depth = depth
            setattr(
                self,
                _field,
                (_value if isinstance(_value, Iterable) else [_value] * _depth),
            )
            assert (
                    isinstance(_value, Iterable) or _depth is not None
            ), f"If {_field} is provided as an integer, \
                depth must be provided too."
            assert len(getattr(self, _field)) == _depth or _depth is None, (
                    f"depth={depth} and {_field}={len(getattr(self, _field))} length conflict, "
                    + f"consider matching or specifying a constan {_field} argument together with a a desired depth"
            )

        self.out_features = self.num_cells[-1]

        self.depth = len(self.kernel_sizes)
        layers = self._make_net()
        super().__init__(*layers)

    def _make_net(self):
        layers = []
        in_features = [self.in_features] + self.num_cells[: self.depth]
        out_features = self.num_cells + [self.out_features]
        kernel_sizes = self.kernel_sizes
        strides = self.strides
        for i, (_in, _out, _kernel, _stride) in enumerate(
                zip(in_features, out_features, kernel_sizes, strides)
        ):
            _bias = (i < len(in_features) - 1) or self.bias_last_layer
            if _in is not None:
                layers.append(
                    nn.Conv2d(
                        _in, _out, kernel_size=_kernel, stride=_stride, bias=_bias
                    )
                )
            else:
                layers.append(
                    nn.LazyConv2d(_out, kernel_size=_kernel, stride=_stride, bias=_bias)
                )

            layers.append(self.activation_class(**self.activation_kwargs))
            if self.norm_class is not None:
                layers.append(self.norm_class(**self.norm_kwargs))

        if self.aggregator_class is not None:
            layers.append(
                self.aggregator_class(**self.aggregator_kwargs))

        if self.squeeze_output:
            layers.append(Squeeze2dLayer())
        return layers


class DuelingCnnDQNet(nn.Module):
    def __init__(
            self, out_features, out_features_value=1, cnn_kwargs={}, mlp_kwargs={},
    ):
        super(DuelingCnnDQNet, self).__init__()

        _cnn_kwargs = {}
        _cnn_kwargs.update(cnn_kwargs)

        self.features = ConvNet(**_cnn_kwargs)

        _mlp_kwargs = {
            "depth": 1,
            "activation_class": nn.ReLU,
            "num_cells": 512,
            "bias_last_layer": True,
        }
        _mlp_kwargs.update(mlp_kwargs)
        self.out_features = out_features
        self.out_features_value = out_features_value
        len_out_features = 1
        if not isinstance(out_features, Number):
            len_out_features = len(out_features)
        self.advantage = MLP(out_features=out_features, **_mlp_kwargs)
        self.value = MLP(out_features=out_features_value, **_mlp_kwargs)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)) and isinstance(layer.bias, torch.Tensor):
                layer.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DistributionalDQNnet(nn.Module):
    _wrong_out_feature_dims_error = "DistributionalDQNnet requires dqn output to be at least 3-dimensional, with dimensions Batch x #Atoms x #Actions"

    def __init__(self, DQNet):
        super().__init__()
        assert (
                not isinstance(DQNet.out_features, Number) and len(DQNet.out_features) > 1
        ), self._wrong_out_feature_dims_error
        self.dqn = DQNet

    def forward(self, x):
        q_values = self.dqn(x)
        assert q_values.ndimension() >= 3, self._wrong_out_feature_dims_error
        return F.log_softmax(q_values, dim=-2)


def ddpg_init_last_layer(last_layer, scale=6e-4):
    last_layer.weight.data.copy_(torch.rand_like(last_layer.weight.data) * scale - scale/2)
    if last_layer.bias is not None:
        last_layer.bias.data.copy_(torch.rand_like(last_layer.bias.data) * scale - scale/2)


class DdpgCnnActor(nn.Module):
    def __init__(self, action_dim, conv_net_kwargs={}, mlp_net_kwargs={}):
        super().__init__()
        conv_net_default_kwargs = {
            'in_features': None,
            'num_cells': [32, 32, 32],
            'kernel_sizes': 3,
            'strides': 1,
            'actionvation_class': nn.ReLU,
            'activation_kwargs': {'inplace': True},
            'norm_class': None,
            'aggregator_class': SquashDims,
            'aggregator_kwargs': {"ndims_in": 3},
            'squeeze_output': True,
        }
        conv_net_default_kwargs.update(conv_net_kwargs)
        mlp_net_default_kwargs = {
            'in_features': None,
            'out_features': action_dim,
            'depth': 2,
            'num_cells': 200,
            'activation_class': nn.ReLU,
            'activation_kwargs': {'inplace': True},
            'bias_last_layer': True,
        }
        mlp_net_default_kwargs.update(mlp_net_kwargs)
        self.convnet = ConvNet(**conv_net_default_kwargs)
        self.mlp = MLP(**mlp_net_default_kwargs)
        ddpg_init_last_layer(self.mlp[-1], 6e-4)

    def forward(self, observation):
        action = self.mlp(self.convnet(observation))
        return action


class DdpgMlpActor(nn.Module):
    def __init__(self, action_dim, mlp_net_kwargs={}):
        super().__init__()
        mlp_net_default_kwargs = {
            'in_features': None,
            'out_features': action_dim,
            'depth': 2,
            'num_cells': [400, 300],
            'activation_class': nn.ReLU,
            'activation_kwargs': {'inplace': True},
            'bias_last_layer': True,
        }
        mlp_net_default_kwargs.update(mlp_net_kwargs)
        self.mlp = MLP(**mlp_net_default_kwargs)
        ddpg_init_last_layer(self.mlp[-1], 6e-3)

    def forward(self, observation):
        action = self.mlp(observation)
        return action

class DdpgCnnQNet(nn.Module):
    def __init__(self, conv_net_kwargs={}, mlp_net_kwargs={}):
        super().__init__()
        conv_net_default_kwargs = {
            'in_features': None,
            'num_cells': [32, 32, 32],
            'kernel_sizes': 3,
            'strides': 1,
            'actionvation_class': nn.ReLU,
            'activation_kwargs': {'inplace': True},
            'norm_class': None,
            'aggregator_class': SquashDims,
            'aggregator_kwargs': {"ndims_in": 3},
            'squeeze_output': True,
        }
        conv_net_default_kwargs.update(conv_net_kwargs)
        mlp_net_default_kwargs = {
            'in_features': None,
            'out_features': 1,
            'depth': 2,
            'num_cells': 200,
            'activation_class': nn.ReLU,
            'activation_kwargs': {'inplace': True},
            'bias_last_layer': True,
        }
        mlp_net_default_kwargs.update(mlp_net_kwargs)
        self.convnet = ConvNet(**conv_net_default_kwargs)
        self.mlp = MLP(**mlp_net_default_kwargs)
        ddpg_init_last_layer(self.mlp[-1], 6e-4)

    def forward(self, observation, action):
        value = self.mlp(torch.cat([self.convnet(observation), action], -1))
        return value


class DdpgMlpQNet(nn.Module):
    def __init__(self, mlp_net_kwargs={}):
        super().__init__()
        mlp1_net_default_kwargs = {
            'in_features': None,
            'out_features': 400,
            'depth': 0,
            'num_cells': [],
            'activation_class': nn.ReLU,
            'activation_kwargs': {'inplace': True},
            'bias_last_layer': True,
            'activate_last_layer': True,
        }
        mlp1_net_default_kwargs.update(mlp_net_kwargs)
        self.mlp1 = MLP(**mlp1_net_default_kwargs)

        mlp2_net_default_kwargs = {
            'in_features': None,
            'out_features': 1,
            'depth': 1,
            'num_cells': [300, ],
            'activation_class': nn.ReLU,
            'activation_kwargs': {'inplace': True},
            'bias_last_layer': True,
        }
        mlp1_net_default_kwargs.update(mlp_net_kwargs)
        self.mlp2 = MLP(**mlp2_net_default_kwargs)
        ddpg_init_last_layer(self.mlp2[-1], 6e-3)

    def forward(self, observation, action):
        value = self.mlp2(torch.cat([self.mlp1(observation), action], -1))
        return value

