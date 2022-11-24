# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
from tensordict.tensordict import TensorDictBase

_has_functorch = False
try:
    from functorch import make_functional, make_functional_with_buffers

    _has_functorch = True
except ImportError:
    from tensordict.nn.functional_modules import (
        FunctionalModule,
        FunctionalModuleWithBuffers,
    )

    make_functional = FunctionalModule._create_from
    make_functional_with_buffers = FunctionalModuleWithBuffers._create_from
from tensordict import TensorDict
from torch import nn
from torchrl.data.tensor_specs import (
    CompositeSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
)
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import NormalParamWrapper, TanhNormal, TensorDictModule
from torchrl.modules.tensordict_module.common import (
    ensure_tensordict_compatible,
    is_tensordict_compatible,
)
from torchrl.modules.tensordict_module.probabilistic import (
    ProbabilisticTensorDictModule,
)
from torchrl.modules.tensordict_module.sequence import TensorDictSequential


class TestTDModule:
    def test_multiple_output(self):
        class MultiHeadLinear(nn.Module):
            def __init__(self, in_1, out_1, out_2, out_3):
                super().__init__()
                self.linear_1 = nn.Linear(in_1, out_1)
                self.linear_2 = nn.Linear(in_1, out_2)
                self.linear_3 = nn.Linear(in_1, out_3)

            def forward(self, x):
                return self.linear_1(x), self.linear_2(x), self.linear_3(x)

        tensordict_module = TensorDictModule(
            MultiHeadLinear(5, 4, 3, 2),
            in_keys=["input"],
            out_keys=["out_1", "out_2", "out_3"],
        )
        td = TensorDict({"input": torch.randn(3, 5)}, batch_size=[3])
        td = tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert "input" in td.keys()
        assert "out_1" in td.keys()
        assert "out_2" in td.keys()
        assert "out_3" in td.keys()
        assert td.get("out_3").shape == torch.Size([3, 2])

        # Using "_" key to ignore some output
        tensordict_module = TensorDictModule(
            MultiHeadLinear(5, 4, 3, 2),
            in_keys=["input"],
            out_keys=["_", "_", "out_3"],
        )
        td = TensorDict({"input": torch.randn(3, 5)}, batch_size=[3])
        td = tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert "input" in td.keys()
        assert "out_3" in td.keys()
        assert "_" not in td.keys()
        assert td.get("out_3").shape == torch.Size([3, 2])

    def test_spec_key_warning(self):
        class MultiHeadLinear(nn.Module):
            def __init__(self, in_1, out_1, out_2):
                super().__init__()
                self.linear_1 = nn.Linear(in_1, out_1)
                self.linear_2 = nn.Linear(in_1, out_2)

            def forward(self, x):
                return self.linear_1(x), self.linear_2(x)

        spec_dict = {
            "_": NdUnboundedContinuousTensorSpec((4,)),
            "out_2": NdUnboundedContinuousTensorSpec((3,)),
        }

        # warning due to "_" in spec keys
        with pytest.warns(UserWarning, match='got a spec with key "_"'):
            tensordict_module = TensorDictModule(
                MultiHeadLinear(5, 4, 3),
                in_keys=["input"],
                out_keys=["_", "out_2"],
                spec=CompositeSpec(**spec_dict),
            )

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
            dist_in_keys = ["loc", "scale"]
        elif out_keys == ["loc_1", "scale_1"]:
            dist_in_keys = {"loc": "loc_1", "scale": "scale_1"}
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
                    dist_in_keys=dist_in_keys,
                    sample_out_key=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tensordict_module = ProbabilisticTensorDictModule(
                module=net,
                spec=spec,
                dist_in_keys=dist_in_keys,
                sample_out_key=["out"],
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
                    dist_in_keys=["loc", "scale"],
                    sample_out_key=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tensordict_module = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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
                    dist_in_keys=["loc", "scale"],
                    sample_out_key=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tensordict_module = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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
                    dist_in_keys=["loc", "scale"],
                    sample_out_key=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tdmodule = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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
                ProbabilisticTensorDictModule(
                    module=tdnet,
                    spec=spec,
                    dist_in_keys=["loc", "scale"],
                    sample_out_key=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tdmodule = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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

    @pytest.mark.skipif(
        not _has_functorch, reason="vmap can only be used with functorch"
    )
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

    @pytest.mark.skipif(
        not _has_functorch, reason="vmap can only be used with functorch"
    )
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
                    dist_in_keys=["loc", "scale"],
                    sample_out_key=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tdmodule = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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

    @pytest.mark.skipif(
        not _has_functorch, reason="vmap can only be used with functorch"
    )
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
                    dist_in_keys=["loc", "scale"],
                    sample_out_key=["out"],
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            tdmodule = ProbabilisticTensorDictModule(
                module=tdnet,
                spec=spec,
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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
    def test_in_key_warning(self):
        with pytest.warns(UserWarning, match='key "_" is for ignoring output'):
            tensordict_module = TensorDictModule(
                nn.Linear(3, 4), in_keys=["_"], out_keys=["out1"]
            )
        with pytest.warns(UserWarning, match='key "_" is for ignoring output'):
            tensordict_module = TensorDictModule(
                nn.Linear(3, 4), in_keys=["_", "key2"], out_keys=["out1"]
            )

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
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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
        if isinstance(params1, TensorDictBase):
            params = TensorDict({"0": params1, "1": params2}, [])
        else:
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
        if isinstance(params1, TensorDictBase):
            params = TensorDict({"0": params1, "1": params2}, [])
        else:
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
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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

        if isinstance(params1, TensorDictBase):
            params = TensorDict({"0": params1, "1": params2}, [])
        else:
            params = list(params1) + list(params2)
        if isinstance(buffers1, TensorDictBase):
            buffers = TensorDict({"0": buffers1, "1": buffers2}, [])
        else:
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

        if isinstance(params1, TensorDictBase):
            params = TensorDict({"0": params1, "1": params2}, [])
        else:
            params = list(params1) + list(params2)
        if isinstance(buffers1, TensorDictBase):
            buffers = TensorDict({"0": buffers1, "1": buffers2}, [])
        else:
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
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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
                dist_in_keys=["loc", "scale"],
                sample_out_key=["out"],
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

    @pytest.mark.skipif(
        not _has_functorch, reason="vmap can only be used with functorch"
    )
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

    @pytest.mark.skipif(
        not _has_functorch, reason="vmap can only be used with functorch"
    )
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
                sample_out_key=["out"],
                dist_in_keys=["loc", "scale"],
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
            sample_out_key=["out"],
            dist_in_keys=["loc", "scale"],
            safe=True,
            **kwargs,
        )
        tdmodule3 = ProbabilisticTensorDictModule(
            fnet3,
            spec=spec,
            sample_out_key=["out"],
            dist_in_keys=["loc", "scale"],
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
                if _has_functorch:
                    params = params1 + params2 + params3
                else:
                    params = TensorDict(
                        {
                            str(i): params
                            for i, params in enumerate((params1, params2, params3))
                        },
                        [],
                    )
                tdmodule(td, params=params)
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
                if _has_functorch:
                    params = params1 + params2 + params3
                else:
                    params = TensorDict(
                        {
                            str(i): params
                            for i, params in enumerate((params1, params2, params3))
                        },
                        [],
                    )
                tdmodule(td, params=params)
            else:
                tdmodule(td)
            assert "loc" in td.keys()
            assert "scale" in td.keys()
            assert "out" in td.keys()
            assert "b" in td.keys()


def test_is_tensordict_compatible():
    class MultiHeadLinear(nn.Module):
        def __init__(self, in_1, out_1, out_2, out_3):
            super().__init__()
            self.linear_1 = nn.Linear(in_1, out_1)
            self.linear_2 = nn.Linear(in_1, out_2)
            self.linear_3 = nn.Linear(in_1, out_3)

        def forward(self, x):
            return self.linear_1(x), self.linear_2(x), self.linear_3(x)

    td_module = TensorDictModule(
        MultiHeadLinear(5, 4, 3, 2),
        in_keys=["in_1", "in_2"],
        out_keys=["out_1", "out_2"],
    )
    assert is_tensordict_compatible(td_module)

    class MockCompatibleModule(nn.Module):
        def __init__(self, in_keys, out_keys):
            self.in_keys = in_keys
            self.out_keys = out_keys

        def forward(self, tensordict):
            pass

    compatible_nn_module = MockCompatibleModule(
        in_keys=["in_1", "in_2"],
        out_keys=["out_1", "out_2"],
    )
    assert is_tensordict_compatible(compatible_nn_module)

    class MockIncompatibleModuleNoKeys(nn.Module):
        def forward(self, input):
            pass

    incompatible_nn_module_no_keys = MockIncompatibleModuleNoKeys()
    assert not is_tensordict_compatible(incompatible_nn_module_no_keys)

    class MockIncompatibleModuleMultipleArgs(nn.Module):
        def __init__(self, in_keys, out_keys):
            self.in_keys = in_keys
            self.out_keys = out_keys

        def forward(self, input_1, input_2):
            pass

    incompatible_nn_module_multi_args = MockIncompatibleModuleMultipleArgs(
        in_keys=["in_1", "in_2"],
        out_keys=["out_1", "out_2"],
    )
    with pytest.raises(TypeError):
        is_tensordict_compatible(incompatible_nn_module_multi_args)


def test_ensure_tensordict_compatible():
    class MultiHeadLinear(nn.Module):
        def __init__(self, in_1, out_1, out_2, out_3):
            super().__init__()
            self.linear_1 = nn.Linear(in_1, out_1)
            self.linear_2 = nn.Linear(in_1, out_2)
            self.linear_3 = nn.Linear(in_1, out_3)

        def forward(self, x):
            return self.linear_1(x), self.linear_2(x), self.linear_3(x)

    td_module = TensorDictModule(
        MultiHeadLinear(5, 4, 3, 2),
        in_keys=["in_1", "in_2"],
        out_keys=["out_1", "out_2"],
    )
    ensured_module = ensure_tensordict_compatible(td_module)
    assert ensured_module is td_module
    with pytest.raises(TypeError):
        ensure_tensordict_compatible(td_module, in_keys=["input"])
    with pytest.raises(TypeError):
        ensure_tensordict_compatible(td_module, out_keys=["output"])

    class NonNNModule:
        def __init__(self):
            pass

        def forward(self, x):
            pass

    non_nn_module = NonNNModule()
    with pytest.raises(TypeError):
        ensure_tensordict_compatible(non_nn_module)

    class ErrorNNModule(nn.Module):
        def forward(self, in_1, in_2):
            pass

    error_nn_module = ErrorNNModule()
    with pytest.raises(TypeError):
        ensure_tensordict_compatible(error_nn_module, in_keys=["input"])

    nn_module = MultiHeadLinear(5, 4, 3, 2)
    ensured_module = ensure_tensordict_compatible(
        nn_module,
        in_keys=["x"],
        out_keys=["out_1", "out_2", "out_3"],
    )
    assert set(ensured_module.in_keys) == {"x"}
    assert isinstance(ensured_module, TensorDictModule)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
