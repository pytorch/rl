# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
from functorch import make_functional, make_functional_with_buffers
from torch import nn
from torchrl.data import TensorDict
from torchrl.data.tensor_specs import (
    CompositeSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
)
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import NormalParamWrapper, TanhNormal, TensorDictModule
from torchrl.modules.tensordict_module.probabilistic import (
    ProbabilisticTensorDictModule,
)
from torchrl.modules.tensordict_module.sequence import TensorDictSequential


class TestTDModule:
    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful(self, safe, spec_type, lazy):
        torch.manual_seed(0)
        param_multiplier = 1
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tensordict_module = TensorDictModule(
                    module=net,
                    spec=spec,
                    in_keys=["in"],
                    out_keys=["out"],
                    safe=safe,
                )
            return
        else:
            tensordict_module = TensorDictModule(
                module=net,
                spec=spec,
                in_keys=["in"],
                out_keys=["out"],
                safe=safe,
            )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("out_keys", [["loc", "scale"], ["loc_1", "scale_1"]])
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("exp_mode", ["mode", "random", None])
    def test_stateful_probabilistic(self, safe, spec_type, lazy, exp_mode, out_keys):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        in_keys = ["in"]
        net = TensorDictModule(
            module=NormalParamWrapper(net),
            spec=None,
            in_keys=in_keys,
            out_keys=out_keys,
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, **{out_key: None for out_key in out_keys})
            if spec is not None
            else None
        )

        kwargs = {"distribution_class": TanhNormal}
        if out_keys == ["loc", "scale"]:
            dist_param_keys = ["loc", "scale"]
        elif out_keys == ["loc_1", "scale_1"]:
            dist_param_keys = {"loc": "loc_1", "scale": "scale_1"}
        else:
            raise NotImplementedError

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tensordict_module = ProbabilisticTensorDictModule(
                    module=net,
                    spec=spec,
                    dist_param_keys=dist_param_keys,
                    out_key_sample=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tensordict_module = ProbabilisticTensorDictModule(
                module=net,
                spec=spec,
                dist_param_keys=dist_param_keys,
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_exploration_mode(exp_mode):
            tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.Linear(3, 4 * param_multiplier)

        fnet, params = make_functional(net)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tensordict_module = TensorDictModule(
                    spec=spec,
                    module=fnet,
                    in_keys=["in"],
                    out_keys=["out"],
                    safe=safe,
                )
            return
        else:
            tensordict_module = TensorDictModule(
                spec=spec,
                module=fnet,
                in_keys=["in"],
                out_keys=["out"],
                safe=safe,
            )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional_probabilistic(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.Linear(3, 4 * param_multiplier)
        in_keys = ["in"]
        net = NormalParamWrapper(net)
        fnet, params = make_functional(net)
        tdnet = TensorDictModule(
            module=fnet, spec=None, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tensordict_module = ProbabilisticTensorDictModule(
                    module=tdnet,
                    spec=spec,
                    dist_param_keys=["loc", "scale"],
                    out_key_sample=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tensordict_module = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional_probabilistic_laterconstruct(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.Linear(3, 4 * param_multiplier)
        in_keys = ["in"]
        net = NormalParamWrapper(net)
        tdnet = TensorDictModule(
            module=net, spec=None, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tensordict_module = ProbabilisticTensorDictModule(
                    module=tdnet,
                    spec=spec,
                    dist_param_keys=["loc", "scale"],
                    out_key_sample=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tensordict_module = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )
        tensordict_module, (
            params,
            buffers,
        ) = tensordict_module.make_functional_with_buffers()

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td = tensordict_module(td, params=params, buffers=buffers)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional_with_buffer(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.BatchNorm1d(32 * param_multiplier)

        fnet, params, buffers = make_functional_with_buffers(net)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 32)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(32)

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = TensorDictModule(
                    spec=spec,
                    module=fnet,
                    in_keys=["in"],
                    out_keys=["out"],
                    safe=safe,
                )
            return
        else:
            tdmodule = TensorDictModule(
                spec=spec,
                module=fnet,
                in_keys=["in"],
                out_keys=["out"],
                safe=safe,
            )

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params, buffers=buffers)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 32])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional_with_buffer_probabilistic(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.BatchNorm1d(32 * param_multiplier)
        in_keys = ["in"]
        net = NormalParamWrapper(net)
        fnet, params, buffers = make_functional_with_buffers(net)
        tdnet = TensorDictModule(
            module=fnet, spec=None, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 32)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(32)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = ProbabilisticTensorDictModule(
                    module=tdnet,
                    spec=spec,
                    dist_param_keys=["loc", "scale"],
                    out_key_sample=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tdmodule = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params, buffers=buffers)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 32])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional_with_buffer_probabilistic_laterconstruct(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.BatchNorm1d(32 * param_multiplier)
        in_keys = ["in"]
        net = NormalParamWrapper(net)
        tdnet = TensorDictModule(
            module=net, spec=None, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 32)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(32)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = ProbabilisticTensorDictModule(
                    module=tdnet,
                    spec=spec,
                    dist_param_keys=["loc", "scale"],
                    out_key_sample=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tdmodule = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )
        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers()

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params, buffers=buffers)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 32])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_vmap(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.Linear(3, 4 * param_multiplier)

        fnet, params = make_functional(net)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = TensorDictModule(
                    spec=spec,
                    module=fnet,
                    in_keys=["in"],
                    out_keys=["out"],
                    safe=safe,
                )
            return
        else:
            tdmodule = TensorDictModule(
                spec=spec,
                module=fnet,
                in_keys=["in"],
                out_keys=["out"],
                safe=safe,
            )

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, None)
        td_out = tdmodule(td, params=params, vmap=(0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, vmap=(0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_vmap_probabilistic(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.Linear(3, 4 * param_multiplier)
        net = NormalParamWrapper(net)
        in_keys = ["in"]
        fnet, params = make_functional(net)
        tdnet = TensorDictModule(
            module=fnet, spec=None, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = ProbabilisticTensorDictModule(
                    module=tdnet,
                    spec=spec,
                    dist_param_keys=["loc", "scale"],
                    out_key_sample=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tdmodule = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, None)
        td_out = tdmodule(td, params=params, vmap=(0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, vmap=(0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_vmap_probabilistic_laterconstruct(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.Linear(3, 4 * param_multiplier)
        net = NormalParamWrapper(net)
        in_keys = ["in"]
        tdnet = TensorDictModule(
            module=net, spec=None, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = ProbabilisticTensorDictModule(
                    module=tdnet,
                    spec=spec,
                    dist_param_keys=["loc", "scale"],
                    out_key_sample=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tdmodule = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )
        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers()

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, buffers=buffers, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0, None)
        td_out = tdmodule(td, params=params, buffers=buffers, vmap=(0, 0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, buffers=buffers, vmap=(0, 0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()


class TestTDSequence:
    def test_key_exclusion(self):
        module1 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["key1", "key2"], out_keys=["foo1"]
        )
        module2 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["key1", "key3"], out_keys=["key1"]
        )
        module3 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["foo1", "key3"], out_keys=["key2"]
        )
        seq = TensorDictSequential(module1, module2, module3)
        assert set(seq.in_keys) == {"key1", "key2", "key3"}
        assert set(seq.out_keys) == {"foo1", "key1", "key2"}

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful(self, safe, spec_type, lazy):
        torch.manual_seed(0)
        param_multiplier = 1
        if lazy:
            net1 = nn.LazyLinear(4)
            dummy_net = nn.LazyLinear(4)
            net2 = nn.LazyLinear(4 * param_multiplier)
        else:
            net1 = nn.Linear(3, 4)
            dummy_net = nn.Linear(4, 4)
            net2 = nn.Linear(4, 4 * param_multiplier)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)

        kwargs = {}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = TensorDictModule(
                net1,
                spec=None,
                in_keys=["in"],
                out_keys=["hidden"],
                safe=False,
            )
            dummy_tdmodule = TensorDictModule(
                dummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = TensorDictModule(
                spec=spec,
                module=net2,
                in_keys=["hidden"],
                out_keys=["out"],
                safe=False,
                **kwargs,
            )
            tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        with pytest.raises(RuntimeError, match="Cannot call get_dist on a sequence"):
            dist, *_ = tdmodule.get_dist(td)

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful_probabilistic(self, safe, spec_type, lazy):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net1 = nn.LazyLinear(4)
            dummy_net = nn.LazyLinear(4)
            net2 = nn.LazyLinear(4 * param_multiplier)
        else:
            net1 = nn.Linear(3, 4)
            dummy_net = nn.Linear(4, 4)
            net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)
        net2 = TensorDictModule(
            module=net2, in_keys=["hidden"], out_keys=["loc", "scale"]
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = TensorDictModule(
                net1,
                spec=None,
                in_keys=["in"],
                out_keys=["hidden"],
                safe=False,
            )
            dummy_tdmodule = TensorDictModule(
                dummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = ProbabilisticTensorDictModule(
                spec=spec,
                module=net2,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=False,
                **kwargs,
            )
            tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist, *_ = tdmodule.get_dist(td)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)

        fnet1, params1 = make_functional(net1)
        fdummy_net, _ = make_functional(dummy_net)
        fnet2, params2 = make_functional(net2)
        params = list(params1) + list(params2)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = TensorDictModule(
                fnet1, spec=None, in_keys=["in"], out_keys=["hidden"], safe=False
            )
            dummy_tdmodule = TensorDictModule(
                fdummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = TensorDictModule(
                fnet2,
                spec=spec,
                in_keys=["hidden"],
                out_keys=["out"],
                safe=safe,
            )
            tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        with pytest.raises(RuntimeError, match="Cannot call get_dist on a sequence"):
            dist, *_ = tdmodule.get_dist(td, params=params)

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional_probabilistic(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)

        fnet1, params1 = make_functional(net1)
        fdummy_net, _ = make_functional(dummy_net)
        fnet2, params2 = make_functional(net2)
        fnet2 = TensorDictModule(
            module=fnet2, in_keys=["hidden"], out_keys=["loc", "scale"]
        )
        params = list(params1) + list(params2)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = TensorDictModule(
                fnet1, spec=None, in_keys=["in"], out_keys=["hidden"], safe=False
            )
            dummy_tdmodule = TensorDictModule(
                fdummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = ProbabilisticTensorDictModule(
                fnet2,
                spec=spec,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )
            tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist, *_ = tdmodule.get_dist(td, params=params)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional_with_buffer(
        self,
        safe,
        spec_type,
    ):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        dummy_net = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )

        fnet1, params1, buffers1 = make_functional_with_buffers(net1)
        fdummy_net, _, _ = make_functional_with_buffers(dummy_net)
        fnet2, params2, buffers2 = make_functional_with_buffers(net2)

        params = list(params1) + list(params2)
        buffers = list(buffers1) + list(buffers2)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 7)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(7)

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = TensorDictModule(
                fnet1, spec=None, in_keys=["in"], out_keys=["hidden"], safe=False
            )
            dummy_tdmodule = TensorDictModule(
                fdummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = TensorDictModule(
                fnet2,
                spec=spec,
                in_keys=["hidden"],
                out_keys=["out"],
                safe=safe,
            )
            tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params, buffers=buffers)

        with pytest.raises(RuntimeError, match="Cannot call get_dist on a sequence"):
            dist, *_ = tdmodule.get_dist(td, params=params, buffers=buffers)

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional_with_buffer_probabilistic(
        self,
        safe,
        spec_type,
    ):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        dummy_net = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )
        net2 = NormalParamWrapper(net2)

        fnet1, params1, buffers1 = make_functional_with_buffers(net1)
        fdummy_net, _, _ = make_functional_with_buffers(dummy_net)
        # fnet2, params2, buffers2 = make_functional_with_buffers(net2)
        # fnet2 = TensorDictModule(fnet2, in_keys=["hidden"], out_keys=["loc", "scale"])
        net2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["loc", "scale"])
        fnet2, (params2, buffers2) = net2.make_functional_with_buffers()

        params = list(params1) + list(params2)
        buffers = list(buffers1) + list(buffers2)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 7)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(7)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = TensorDictModule(
                fnet1, spec=None, in_keys=["in"], out_keys=["hidden"], safe=False
            )
            dummy_tdmodule = TensorDictModule(
                fdummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = ProbabilisticTensorDictModule(
                fnet2,
                spec=spec,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )
            tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params, buffers=buffers)

        dist, *_ = tdmodule.get_dist(td, params=params, buffers=buffers)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_functional_with_buffer_probabilistic_laterconstruct(
        self,
        safe,
        spec_type,
    ):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )
        net2 = NormalParamWrapper(net2)
        net2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["loc", "scale"])

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 7)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(7)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = TensorDictModule(
                net1, spec=None, in_keys=["in"], out_keys=["hidden"], safe=False
            )
            tdmodule2 = ProbabilisticTensorDictModule(
                net2,
                spec=spec,
                dist_param_keys=["loc", "scale"],
                out_key_sample=["out"],
                safe=safe,
                **kwargs,
            )
            tdmodule = TensorDictSequential(tdmodule1, tdmodule2)

        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers()

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params, buffers=buffers)

        dist, *_ = tdmodule.get_dist(td, params=params, buffers=buffers)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_vmap(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)

        fnet1, params1 = make_functional(net1)
        fdummy_net, _ = make_functional(dummy_net)
        fnet2, params2 = make_functional(net2)
        params = params1 + params2

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = TensorDictModule(
                fnet1,
                spec=None,
                in_keys=["in"],
                out_keys=["hidden"],
                safe=False,
            )
            dummy_tdmodule = TensorDictModule(
                fdummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = TensorDictModule(
                fnet2,
                spec=spec,
                in_keys=["hidden"],
                out_keys=["out"],
                safe=safe,
            )
            tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, None)
        td_out = tdmodule(td, params=params, vmap=(0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, vmap=(0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    def test_vmap_probabilistic(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)
        fnet1, params1 = make_functional(net1)

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)
        fnet2, params2 = make_functional(net2)
        fnet2 = TensorDictModule(fnet2, in_keys=["hidden"], out_keys=["loc", "scale"])

        params = params1 + params2

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError
        spec = (
            CompositeSpec(out=spec, loc=None, scale=None) if spec is not None else None
        )

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = TensorDictModule(
                fnet1,
                spec=None,
                in_keys=["in"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = ProbabilisticTensorDictModule(
                fnet2,
                spec=spec,
                out_key_sample=["out"],
                dist_param_keys=["loc", "scale"],
                safe=safe,
                **kwargs,
            )
            tdmodule = TensorDictSequential(tdmodule1, tdmodule2)

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, None)
        td_out = tdmodule(td, params=params, vmap=(0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, vmap=(0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

    @pytest.mark.parametrize("functional", [True, False])
    def test_submodule_sequence(self, functional):
        td_module_1 = TensorDictModule(
            nn.Linear(3, 2),
            in_keys=["in"],
            out_keys=["hidden"],
        )
        td_module_2 = TensorDictModule(
            nn.Linear(2, 4),
            in_keys=["hidden"],
            out_keys=["out"],
        )
        td_module = TensorDictSequential(td_module_1, td_module_2)

        if functional:
            td_1 = TensorDict({"in": torch.randn(5, 3)}, [5])
            sub_seq_1 = td_module.select_subsequence(out_keys=["hidden"])
            sub_seq_1, (params, buffers) = sub_seq_1.make_functional_with_buffers()
            sub_seq_1(
                td_1,
                params=params,
                buffers=buffers,
            )
            assert "hidden" in td_1.keys()
            assert "out" not in td_1.keys()
            td_2 = TensorDict({"hidden": torch.randn(5, 2)}, [5])
            sub_seq_2 = td_module.select_subsequence(in_keys=["hidden"])
            sub_seq_2, (params, buffers) = sub_seq_2.make_functional_with_buffers()
            sub_seq_2(
                td_2,
                params=params,
                buffers=buffers,
            )
            assert "out" in td_2.keys()
            assert td_2.get("out").shape == torch.Size([5, 4])
        else:
            td_1 = TensorDict({"in": torch.randn(5, 3)}, [5])
            sub_seq_1 = td_module.select_subsequence(out_keys=["hidden"])
            sub_seq_1(td_1)
            assert "hidden" in td_1.keys()
            assert "out" not in td_1.keys()
            td_2 = TensorDict({"hidden": torch.randn(5, 2)}, [5])
            sub_seq_2 = td_module.select_subsequence(in_keys=["hidden"])
            sub_seq_2(td_2)
            assert "out" in td_2.keys()
            assert td_2.get("out").shape == torch.Size([5, 4])

    @pytest.mark.parametrize("stack", [True, False])
    @pytest.mark.parametrize("functional", [True, False])
    def test_sequential_partial(self, stack, functional):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)
        if functional:
            fnet1, params1 = make_functional(net1)
        else:
            params1 = None
            fnet1 = net1

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)
        if functional:
            fnet2, params2 = make_functional(net2)
        else:
            fnet2 = net2
            params2 = None
        fnet2 = TensorDictModule(fnet2, in_keys=["b"], out_keys=["loc", "scale"])

        net3 = nn.Linear(4, 4 * param_multiplier)
        net3 = NormalParamWrapper(net3)
        if functional:
            fnet3, params3 = make_functional(net3)
        else:
            fnet3 = net3
            params3 = None
        fnet3 = TensorDictModule(fnet3, in_keys=["c"], out_keys=["loc", "scale"])

        spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        spec = CompositeSpec(out=spec, loc=None, scale=None)

        kwargs = {"distribution_class": TanhNormal}

        tdmodule1 = TensorDictModule(
            fnet1,
            spec=None,
            in_keys=["a"],
            out_keys=["hidden"],
            safe=False,
        )
        tdmodule2 = ProbabilisticTensorDictModule(
            fnet2,
            spec=spec,
            out_key_sample=["out"],
            dist_param_keys=["loc", "scale"],
            safe=True,
            **kwargs,
        )
        tdmodule3 = ProbabilisticTensorDictModule(
            fnet3,
            spec=spec,
            out_key_sample=["out"],
            dist_param_keys=["loc", "scale"],
            safe=True,
            **kwargs,
        )
        tdmodule = TensorDictSequential(
            tdmodule1, tdmodule2, tdmodule3, partial_tolerant=True
        )

        if stack:
            td = torch.stack(
                [
                    TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, []),
                    TensorDict({"a": torch.randn(3), "c": torch.randn(4)}, []),
                ],
                0,
            )
            if functional:
                tdmodule(td, params=params1 + params2 + params3)
            else:
                tdmodule(td)
            assert "loc" in td.keys()
            assert "scale" in td.keys()
            assert "out" in td.keys()
            assert td["out"].shape[0] == 2
            assert td["loc"].shape[0] == 2
            assert td["scale"].shape[0] == 2
            assert "b" not in td.keys()
            assert "b" in td[0].keys()
        else:
            td = TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, [])
            if functional:
                tdmodule(td, params=params1 + params2 + params3)
            else:
                tdmodule(td)
            assert "loc" in td.keys()
            assert "scale" in td.keys()
            assert "out" in td.keys()
            assert "b" in td.keys()

    def test_subsequence_weight_update(self):
        td_module_1 = TensorDictModule(
            nn.Linear(3, 2),
            in_keys=["in"],
            out_keys=["hidden"],
        )
        td_module_2 = TensorDictModule(
            nn.Linear(2, 4),
            in_keys=["hidden"],
            out_keys=["out"],
        )
        td_module = TensorDictSequential(td_module_1, td_module_2)

        td_1 = TensorDict({"in": torch.randn(5, 3)}, [5])
        sub_seq_1 = td_module.select_subsequence(out_keys=["hidden"])
        copy = sub_seq_1[0].module.weight.clone()

        opt = torch.optim.SGD(td_module.parameters(), lr=0.1)
        opt.zero_grad()
        td_1 = td_module(td_1)
        td_1["out"].mean().backward()
        opt.step()

        assert not torch.allclose(copy, sub_seq_1[0].module.weight)
        assert torch.allclose(td_module[0].module.weight, sub_seq_1[0].module.weight)

    if __name__ == "__main__":
        args, unknown = argparse.ArgumentParser().parse_known_args()
        pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
