# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
from numbers import Number

import pytest
import torch
from torch import nn
from torchrl.modules.models import BatchRenorm1d, Conv3dNet, ConvNet, MLP, NoisyLinear
from torchrl.modules.models.recipes.impala import _ConvNetBlock
from torchrl.modules.models.utils import SquashDims

from torchrl.testing import get_default_devices


class TestMLP:
    @pytest.mark.parametrize("in_features", [3, 10, None])
    @pytest.mark.parametrize("out_features", [3, (3, 10)])
    @pytest.mark.parametrize("depth, num_cells", [(3, 32), (None, (32, 32, 32))])
    @pytest.mark.parametrize(
        "activation_class, activation_kwargs",
        [(nn.ReLU, {"inplace": True}), (nn.ReLU, {}), (nn.PReLU, {})],
    )
    @pytest.mark.parametrize(
        "norm_class, norm_kwargs",
        [
            (nn.LazyBatchNorm1d, {}),
            (nn.BatchNorm1d, {"num_features": 32}),
            (nn.LayerNorm, {"normalized_shape": 32}),
        ],
    )
    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    @pytest.mark.parametrize("bias_last_layer", [True, False])
    @pytest.mark.parametrize("single_bias_last_layer", [True, False])
    @pytest.mark.parametrize("layer_class", [nn.Linear, NoisyLinear])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_mlp(
        self,
        in_features,
        out_features,
        depth,
        num_cells,
        activation_class,
        activation_kwargs,
        dropout,
        bias_last_layer,
        norm_class,
        norm_kwargs,
        single_bias_last_layer,
        layer_class,
        device,
        seed=0,
    ):
        torch.manual_seed(seed)
        batch = 2
        mlp = MLP(
            in_features=in_features,
            out_features=out_features,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
            activation_kwargs=activation_kwargs,
            norm_class=norm_class,
            norm_kwargs=norm_kwargs,
            dropout=dropout,
            bias_last_layer=bias_last_layer,
            single_bias_last_layer=False,
            layer_class=layer_class,
            device=device,
        )
        if in_features is None:
            in_features = 5
        x = torch.randn(batch, in_features, device=device)
        y = mlp(x)
        out_features = (
            [out_features] if isinstance(out_features, Number) else out_features
        )
        assert y.shape == torch.Size([batch, *out_features])

    def test_kwargs(self):
        def make_activation(shift):
            return lambda x: x + shift

        def layer(*args, **kwargs):
            linear = nn.Linear(*args, **kwargs)
            linear.weight.data.copy_(torch.eye(4))
            return linear

        in_features = 4
        out_features = 4
        num_cells = [4, 4, 4]
        mlp = MLP(
            in_features=in_features,
            out_features=out_features,
            num_cells=num_cells,
            activation_class=make_activation,
            activation_kwargs=[{"shift": 0}, {"shift": 1}, {"shift": 2}],
            layer_class=layer,
            layer_kwargs=[{"bias": False}] * 4,
            bias_last_layer=False,
        )
        x = torch.zeros(4)
        y = mlp(x)
        for i, module in enumerate(mlp.modules()):
            if isinstance(module, nn.Linear):
                assert (module.weight == torch.eye(4)).all(), i
                assert module.bias is None, i
        assert (y == 3).all()


@pytest.mark.parametrize("in_features", [3, 10, None])
@pytest.mark.parametrize(
    "input_size, depth, num_cells, kernel_sizes, strides, paddings, expected_features",
    [(100, None, None, 3, 1, 0, 32 * 94 * 94), (100, 3, 32, 3, 1, 1, 32 * 100 * 100)],
)
@pytest.mark.parametrize(
    "activation_class, activation_kwargs",
    [(nn.ReLU, {"inplace": True}), (nn.ReLU, {}), (nn.PReLU, {})],
)
@pytest.mark.parametrize(
    "norm_class, norm_kwargs",
    [(None, None), (nn.LazyBatchNorm2d, {}), (nn.BatchNorm2d, {"num_features": 32})],
)
@pytest.mark.parametrize("bias_last_layer", [True, False])
@pytest.mark.parametrize(
    "aggregator_class, aggregator_kwargs",
    [(SquashDims, {})],
)
@pytest.mark.parametrize("squeeze_output", [False])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("batch", [(2,), (2, 2)])
def test_convnet(
    batch,
    in_features,
    depth,
    num_cells,
    kernel_sizes,
    strides,
    paddings,
    activation_class,
    activation_kwargs,
    norm_class,
    norm_kwargs,
    bias_last_layer,
    aggregator_class,
    aggregator_kwargs,
    squeeze_output,
    device,
    input_size,
    expected_features,
    seed=0,
):
    torch.manual_seed(seed)
    convnet = ConvNet(
        in_features=in_features,
        depth=depth,
        num_cells=num_cells,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        activation_class=activation_class,
        activation_kwargs=activation_kwargs,
        norm_class=norm_class,
        norm_kwargs=norm_kwargs,
        bias_last_layer=bias_last_layer,
        aggregator_class=aggregator_class,
        aggregator_kwargs=aggregator_kwargs,
        squeeze_output=squeeze_output,
        device=device,
    )
    if in_features is None:
        in_features = 5
    x = torch.randn(*batch, in_features, input_size, input_size, device=device)
    y = convnet(x)
    assert y.shape == torch.Size([*batch, expected_features])


class TestConv3d:
    @pytest.mark.parametrize("in_features", [3, 10, None])
    @pytest.mark.parametrize(
        "input_size, depth, num_cells, kernel_sizes, strides, paddings, expected_features",
        [
            (10, None, None, 3, 1, 0, 32 * 4 * 4 * 4),
            (10, 3, 32, 3, 1, 1, 32 * 10 * 10 * 10),
        ],
    )
    @pytest.mark.parametrize(
        "activation_class, activation_kwargs",
        [(nn.ReLU, {"inplace": True}), (nn.ReLU, {}), (nn.PReLU, {})],
    )
    @pytest.mark.parametrize(
        "norm_class, norm_kwargs",
        [
            (None, None),
            (nn.LazyBatchNorm3d, {}),
            (nn.BatchNorm3d, {"num_features": 32}),
        ],
    )
    @pytest.mark.parametrize("bias_last_layer", [True, False])
    @pytest.mark.parametrize(
        "aggregator_class, aggregator_kwargs",
        [(SquashDims, None)],
    )
    @pytest.mark.parametrize("squeeze_output", [False])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch", [(2,), (2, 2)])
    def test_conv3dnet(
        self,
        batch,
        in_features,
        depth,
        num_cells,
        kernel_sizes,
        strides,
        paddings,
        activation_class,
        activation_kwargs,
        norm_class,
        norm_kwargs,
        bias_last_layer,
        aggregator_class,
        aggregator_kwargs,
        squeeze_output,
        device,
        input_size,
        expected_features,
        seed=0,
    ):
        torch.manual_seed(seed)
        conv3dnet = Conv3dNet(
            in_features=in_features,
            depth=depth,
            num_cells=num_cells,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            activation_class=activation_class,
            activation_kwargs=activation_kwargs,
            norm_class=norm_class,
            norm_kwargs=norm_kwargs,
            bias_last_layer=bias_last_layer,
            aggregator_class=aggregator_class,
            aggregator_kwargs=aggregator_kwargs,
            squeeze_output=squeeze_output,
            device=device,
        )
        if in_features is None:
            in_features = 5
        x = torch.randn(
            *batch, in_features, input_size, input_size, input_size, device=device
        )
        y = conv3dnet(x)
        assert y.shape == torch.Size([*batch, expected_features])
        with pytest.raises(ValueError, match="must have at least 4 dimensions"):
            conv3dnet(torch.randn(3, 16, 16))

    def test_errors(self):
        with pytest.raises(
            ValueError, match="Null depth is not permitted with Conv3dNet"
        ):
            conv3dnet = Conv3dNet(
                in_features=5,
                num_cells=32,
                depth=0,
            )
        with pytest.raises(
            ValueError, match="depth=None requires one of the input args"
        ):
            conv3dnet = Conv3dNet(
                in_features=5,
                num_cells=32,
                depth=None,
            )
        with pytest.raises(
            ValueError, match="consider matching or specifying a constant num_cells"
        ):
            conv3dnet = Conv3dNet(
                in_features=5,
                num_cells=[32],
                depth=None,
                kernel_sizes=[3, 3],
            )


class TestBatchRenorm:
    @pytest.mark.parametrize("num_steps", [0, 5])
    @pytest.mark.parametrize("smooth", [False, True])
    def test_batchrenorm(self, num_steps, smooth):
        torch.manual_seed(0)
        bn = torch.nn.BatchNorm1d(5, momentum=0.1, eps=1e-5)
        brn = BatchRenorm1d(
            5,
            momentum=0.1,
            eps=1e-5,
            warmup_steps=num_steps,
            max_d=10000,
            max_r=10000,
            smooth=smooth,
        )
        bn.train()
        brn.train()
        data_train = torch.randn(100, 5).split(25)
        data_test = torch.randn(100, 5)
        for i, d in enumerate(data_train):
            b = bn(d)
            a = brn(d)
            if num_steps > 0 and (
                (i < num_steps and not smooth) or (i == 0 and smooth)
            ):
                torch.testing.assert_close(a, b)
            else:
                assert not torch.isclose(a, b).all(), i

        bn.eval()
        brn.eval()
        torch.testing.assert_close(bn(data_test), brn(data_test))


def test_convnetblock_uses_both_resnets():
    """Regression test for https://github.com/pytorch/rl/issues/3519."""
    block = _ConvNetBlock(num_ch=16)
    x = torch.randn(2, 3, 8, 8)
    out = block(x).mean()
    out.backward()

    resnet1_grad = sum(p.grad.abs().sum() for p in block.resnet1.parameters())
    resnet2_grad = sum(p.grad.abs().sum() for p in block.resnet2.parameters())
    assert resnet1_grad > 0, "resnet1 parameters received no gradients"
    assert resnet2_grad > 0, "resnet2 parameters received no gradients"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
