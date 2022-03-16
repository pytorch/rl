import argparse

import pytest
import torch
from functorch import make_functional, make_functional_with_buffers
from torch import nn
from torchrl.data import TensorDict
from torchrl.data.tensor_specs import (
    NdUnboundedContinuousTensorSpec,
    NdBoundedTensorSpec,
)
from torchrl.modules import (
    TDModule,
    ProbabilisticTDModule,
    TanhNormal,
)


class TestTDModule:
    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("probabilistic", [True, False])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful(self, safe, spec_type, probabilistic, lazy):
        torch.manual_seed(0)
        param_multiplier = 2 if probabilistic else 1
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

        if probabilistic:
            tdclass = ProbabilisticTDModule
            kwargs = {"distribution_class": TanhNormal}
        else:
            tdclass = TDModule
            kwargs = {}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = tdclass(
                    spec, net, in_keys=["in"], out_keys=["out"], safe=safe, **kwargs
                )
            return
        else:
            tdmodule = tdclass(
                spec, net, in_keys=["in"], out_keys=["out"], safe=safe, **kwargs
            )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_functional(self, safe, spec_type, probabilistic):
        torch.manual_seed(0)
        param_multiplier = 2 if probabilistic else 1

        net = nn.Linear(3, 4 * param_multiplier)
        fnet, params = make_functional(net)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)

        if probabilistic:
            tdclass = ProbabilisticTDModule
            kwargs = {"distribution_class": TanhNormal}
        else:
            tdclass = TDModule
            kwargs = {}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = tdclass(
                    spec, fnet, in_keys=["in"], out_keys=["out"], safe=safe, **kwargs
                )
            return
        else:
            tdmodule = tdclass(
                spec, fnet, in_keys=["in"], out_keys=["out"], safe=safe, **kwargs
            )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_functional_with_buffer(self, safe, spec_type, probabilistic):
        torch.manual_seed(0)
        param_multiplier = 2 if probabilistic else 1

        net = nn.BatchNorm1d(32 * param_multiplier)
        fnet, params, buffers = make_functional_with_buffers(net)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 32)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(32)

        if probabilistic:
            tdclass = ProbabilisticTDModule
            kwargs = {"distribution_class": TanhNormal}
        else:
            tdclass = TDModule
            kwargs = {}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = tdclass(
                    spec, fnet, in_keys=["in"], out_keys=["out"], safe=safe, **kwargs
                )
            return
        else:
            tdmodule = tdclass(
                spec, fnet, in_keys=["in"], out_keys=["out"], safe=safe, **kwargs
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
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_vmap(self, safe, spec_type, probabilistic):
        torch.manual_seed(0)
        param_multiplier = 2 if probabilistic else 1

        net = nn.Linear(3, 4 * param_multiplier)
        fnet, params = make_functional(net)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = NdBoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = NdUnboundedContinuousTensorSpec(4)

        if probabilistic:
            tdclass = ProbabilisticTDModule
            kwargs = {"distribution_class": TanhNormal}
        else:
            tdclass = TDModule
            kwargs = {}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = tdclass(
                    spec, fnet, in_keys=["in"], out_keys=["out"], safe=safe, **kwargs
                )
            return
        else:
            tdmodule = tdclass(
                spec, fnet, in_keys=["in"], out_keys=["out"], safe=safe, **kwargs
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
        td_repeat = td.expand(10).clone()
        td_out = tdmodule(td_repeat, params=params, vmap=(0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()


class TestTDSequence:
    def test_stateful(self):
        raise NotImplementedError

    def test_functional(self):
        raise NotImplementedError

    def test_vmap(self):
        raise NotImplementedError


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
