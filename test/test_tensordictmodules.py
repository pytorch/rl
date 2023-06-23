# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import InteractionType, make_functional, TensorDictModule
from torch import nn
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.utils import set_exploration_type, step_mdp
from torchrl.modules import LSTMModule, NormalParamWrapper, SafeModule, TanhNormal
from torchrl.modules.tensordict_module.common import (
    ensure_tensordict_compatible,
    is_tensordict_compatible,
)
from torchrl.modules.tensordict_module.probabilistic import (
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)
from torchrl.modules.tensordict_module.sequence import SafeSequential

_has_functorch = False
try:
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    _has_functorch = True
except ImportError:
    pass


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

        tensordict_module = SafeModule(
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
        tensordict_module = SafeModule(
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
            "_": UnboundedContinuousTensorSpec((4,)),
            "out_2": UnboundedContinuousTensorSpec((3,)),
        }

        # warning due to "_" in spec keys
        with pytest.warns(UserWarning, match='got a spec with key "_"'):
            tensordict_module = SafeModule(
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
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tensordict_module = SafeModule(
                    module=net,
                    spec=spec,
                    in_keys=["in"],
                    out_keys=["out"],
                    safe=safe,
                )
            return
        else:
            tensordict_module = SafeModule(
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
    @pytest.mark.parametrize(
        "exp_mode", [InteractionType.MODE, InteractionType.RANDOM, None]
    )
    def test_stateful_probabilistic(self, safe, spec_type, lazy, exp_mode, out_keys):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        in_keys = ["in"]
        net = SafeModule(
            module=NormalParamWrapper(net),
            spec=None,
            in_keys=in_keys,
            out_keys=out_keys,
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError

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
                prob_module = SafeProbabilisticModule(
                    in_keys=dist_in_keys,
                    out_keys=["out"],
                    spec=spec,
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            prob_module = SafeProbabilisticModule(
                in_keys=dist_in_keys,
                out_keys=["out"],
                spec=spec,
                safe=safe,
                **kwargs,
            )

        tensordict_module = SafeProbabilisticTensorDictSequential(net, prob_module)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_exploration_type(exp_mode):
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

        params = make_functional(net)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tensordict_module = SafeModule(
                    spec=spec,
                    module=net,
                    in_keys=["in"],
                    out_keys=["out"],
                    safe=safe,
                )
            return
        else:
            tensordict_module = SafeModule(
                spec=spec,
                module=net,
                in_keys=["in"],
                out_keys=["out"],
                safe=safe,
            )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td, params=TensorDict({"module": params}, []))
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

        tdnet = SafeModule(
            module=NormalParamWrapper(nn.Linear(3, 4 * param_multiplier)),
            spec=None,
            in_keys=["in"],
            out_keys=["loc", "scale"],
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                prob_module = SafeProbabilisticModule(
                    in_keys=["loc", "scale"],
                    out_keys=["out"],
                    spec=spec,
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            prob_module = SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=safe,
                **kwargs,
            )

        tensordict_module = SafeProbabilisticTensorDictSequential(tdnet, prob_module)
        params = make_functional(tensordict_module)

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
    def test_functional_with_buffer(self, safe, spec_type):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.BatchNorm1d(32 * param_multiplier)
        params = make_functional(net)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 32)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(32)

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = SafeModule(
                    spec=spec,
                    module=net,
                    in_keys=["in"],
                    out_keys=["out"],
                    safe=safe,
                )
            return
        else:
            tdmodule = SafeModule(
                spec=spec,
                module=net,
                in_keys=["in"],
                out_keys=["out"],
                safe=safe,
            )

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=TensorDict({"module": params}, []))
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

        tdnet = SafeModule(
            module=NormalParamWrapper(nn.BatchNorm1d(32 * param_multiplier)),
            spec=None,
            in_keys=["in"],
            out_keys=["loc", "scale"],
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 32)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(32)
        else:
            raise NotImplementedError

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                prob_module = SafeProbabilisticModule(
                    in_keys=["loc", "scale"],
                    out_keys=["out"],
                    spec=spec,
                    safe=safe,
                    **kwargs,
                )

            return
        else:
            prob_module = SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=safe,
                **kwargs,
            )

        tdmodule = SafeProbabilisticTensorDictSequential(tdnet, prob_module)
        params = make_functional(tdmodule)

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params)
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

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                tdmodule = SafeModule(
                    spec=spec,
                    module=net,
                    in_keys=["in"],
                    out_keys=["out"],
                    safe=safe,
                )
            return
        else:
            tdmodule = SafeModule(
                spec=spec,
                module=net,
                in_keys=["in"],
                out_keys=["out"],
                safe=safe,
            )

        params = make_functional(tdmodule)

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        if safe and spec_type == "bounded":
            with pytest.raises(
                RuntimeError, match="vmap cannot be used with safe=True"
            ):
                td_out = vmap(tdmodule, (None, 0))(td, params)
            return
        else:
            td_out = vmap(tdmodule, (None, 0))(td, params)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
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

        net = NormalParamWrapper(nn.Linear(3, 4 * param_multiplier))
        tdnet = SafeModule(
            module=net, in_keys=["in"], out_keys=["loc", "scale"], spec=None
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            with pytest.raises(
                RuntimeError,
                match="is not a valid configuration as the tensor specs are not "
                "specified",
            ):
                prob_module = SafeProbabilisticModule(
                    in_keys=["loc", "scale"],
                    out_keys=["out"],
                    spec=spec,
                    safe=safe,
                    **kwargs,
                )
            return
        else:
            prob_module = SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=safe,
                **kwargs,
            )

        tdmodule = SafeProbabilisticTensorDictSequential(tdnet, prob_module)
        params = make_functional(tdmodule)

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        if safe and spec_type == "bounded":
            with pytest.raises(
                RuntimeError, match="vmap cannot be used with safe=True"
            ):
                td_out = vmap(tdmodule, (None, 0))(td, params)
            return
        else:
            td_out = vmap(tdmodule, (None, 0))(td, params)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
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
            tensordict_module = SafeModule(
                nn.Linear(3, 4), in_keys=["_"], out_keys=["out1"]
            )
        with pytest.warns(UserWarning, match='key "_" is for ignoring output'):
            tensordict_module = SafeModule(
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
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)

        kwargs = {}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1,
                spec=None,
                in_keys=["in"],
                out_keys=["hidden"],
                safe=False,
            )
            dummy_tdmodule = SafeModule(
                dummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = SafeModule(
                spec=spec,
                module=net2,
                in_keys=["hidden"],
                out_keys=["out"],
                safe=False,
                **kwargs,
            )
            tdmodule = SafeSequential(tdmodule1, dummy_tdmodule, tdmodule2)

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

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1,
                in_keys=["in"],
                out_keys=["hidden"],
                spec=None,
                safe=False,
            )
            dummy_tdmodule = SafeModule(
                dummy_net,
                in_keys=["hidden"],
                out_keys=["hidden"],
                spec=None,
                safe=False,
            )
            tdmodule2 = SafeModule(
                module=net2,
                in_keys=["hidden"],
                out_keys=["loc", "scale"],
                spec=None,
                safe=False,
            )

            prob_module = SafeProbabilisticModule(
                spec=spec,
                in_keys=["loc", "scale"],
                out_keys=["out"],
                safe=False,
                **kwargs,
            )
            tdmodule = SafeProbabilisticTensorDictSequential(
                tdmodule1, dummy_tdmodule, tdmodule2, prob_module
            )

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 4
        tdmodule[1] = tdmodule2
        tdmodule[2] = prob_module
        assert len(tdmodule) == 4

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 4
        del tdmodule[3]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is prob_module

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist = tdmodule.get_dist(td)
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

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1, spec=None, in_keys=["in"], out_keys=["hidden"], safe=False
            )
            dummy_tdmodule = SafeModule(
                dummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = SafeModule(
                net2,
                spec=spec,
                in_keys=["hidden"],
                out_keys=["out"],
                safe=safe,
            )
            tdmodule = SafeSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        params = make_functional(tdmodule)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        with params.unlock_():
            params["module", "1"] = params["module", "2"]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        del params["module", "2"]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td, params)
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

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1, spec=None, in_keys=["in"], out_keys=["hidden"], safe=False
            )
            dummy_tdmodule = SafeModule(
                dummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = SafeModule(
                module=net2, in_keys=["hidden"], out_keys=["loc", "scale"]
            )

            prob_module = SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=safe,
                **kwargs,
            )
            tdmodule = SafeProbabilisticTensorDictSequential(
                tdmodule1, dummy_tdmodule, tdmodule2, prob_module
            )

        params = make_functional(tdmodule, funs_to_decorate=["forward", "get_dist"])

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 4
        tdmodule[1] = tdmodule2
        tdmodule[2] = prob_module
        with params.unlock_():
            params["module", "1"] = params["module", "2"]
            params["module", "2"] = params["module", "3"]
        assert len(tdmodule) == 4

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 4
        del tdmodule[3]
        del params["module", "3"]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is prob_module

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist = tdmodule.get_dist(td, params=params)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

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

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        dummy_net = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 7)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(7)

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1, spec=None, in_keys=["in"], out_keys=["hidden"], safe=False
            )
            dummy_tdmodule = SafeModule(
                dummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = SafeModule(
                net2,
                spec=spec,
                in_keys=["hidden"],
                out_keys=["out"],
                safe=safe,
            )
            tdmodule = SafeSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        params = make_functional(tdmodule)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        with params.unlock_():
            params["module", "1"] = params["module", "2"]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        del params["module", "2"]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params)

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

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

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        dummy_net = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )
        net2 = NormalParamWrapper(net2)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 7)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(7)
        else:
            raise NotImplementedError

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1, in_keys=["in"], out_keys=["hidden"], spec=None, safe=False
            )
            dummy_tdmodule = SafeModule(
                dummy_net,
                in_keys=["hidden"],
                out_keys=["hidden"],
                spec=None,
                safe=False,
            )
            tdmodule2 = SafeModule(
                net2,
                in_keys=["hidden"],
                out_keys=["loc", "scale"],
                spec=None,
                safe=False,
            )

            prob_module = SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=safe,
                **kwargs,
            )
            tdmodule = SafeProbabilisticTensorDictSequential(
                tdmodule1, dummy_tdmodule, tdmodule2, prob_module
            )

        params = make_functional(tdmodule, ["forward", "get_dist"])

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 4
        tdmodule[1] = tdmodule2
        tdmodule[2] = prob_module
        with params.unlock_():
            params["module", "1"] = params["module", "2"]
            params["module", "2"] = params["module", "3"]
        assert len(tdmodule) == 4

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 4
        del tdmodule[3]
        del params["module", "3"]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is prob_module

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params)

        dist = tdmodule.get_dist(td, params=params)
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

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1,
                spec=None,
                in_keys=["in"],
                out_keys=["hidden"],
                safe=False,
            )
            dummy_tdmodule = SafeModule(
                dummy_net,
                spec=None,
                in_keys=["hidden"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = SafeModule(
                net2,
                spec=spec,
                in_keys=["hidden"],
                out_keys=["out"],
                safe=safe,
            )
            tdmodule = SafeSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        params = make_functional(tdmodule)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        with params.unlock_():
            params["module", "1"] = params["module", "2"]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        del params["module", "2"]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        if safe and spec_type == "bounded":
            with pytest.raises(
                RuntimeError, match="vmap cannot be used with safe=True"
            ):
                td_out = vmap(tdmodule, (None, 0))(td, params)
            return
        else:
            td_out = vmap(tdmodule, (None, 0))(td, params)

        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
        assert td_out is not td_repeat
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

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = BoundedTensorSpec(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = UnboundedContinuousTensorSpec(4)
        else:
            raise NotImplementedError

        kwargs = {"distribution_class": TanhNormal}

        if safe and spec is None:
            pytest.skip("safe and spec is None is checked elsewhere")
        else:
            tdmodule1 = SafeModule(
                net1,
                spec=None,
                in_keys=["in"],
                out_keys=["hidden"],
                safe=False,
            )
            tdmodule2 = SafeModule(net2, in_keys=["hidden"], out_keys=["loc", "scale"])
            prob_module = SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=safe,
                **kwargs,
            )
            tdmodule = SafeProbabilisticTensorDictSequential(
                tdmodule1, tdmodule2, prob_module
            )

        params = make_functional(tdmodule)

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        if safe and spec_type == "bounded":
            with pytest.raises(
                RuntimeError, match="vmap cannot be used with safe=True"
            ):
                td_out = vmap(tdmodule, (None, 0))(td, params)
            return
        else:
            td_out = vmap(tdmodule, (None, 0))(td, params)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
        assert td_out is not td_repeat
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])
        # test bounds
        if not safe and spec_type == "bounded":
            assert ((td_out.get("out") > 0.1) | (td_out.get("out") < -0.1)).any()
        elif safe and spec_type == "bounded":
            assert ((td_out.get("out") < 0.1) | (td_out.get("out") > -0.1)).all()

    @pytest.mark.parametrize("functional", [True, False])
    def test_submodule_sequence(self, functional):
        td_module_1 = SafeModule(
            nn.Linear(3, 2),
            in_keys=["in"],
            out_keys=["hidden"],
        )
        td_module_2 = SafeModule(
            nn.Linear(2, 4),
            in_keys=["hidden"],
            out_keys=["out"],
        )
        td_module = SafeSequential(td_module_1, td_module_2)

        if functional:
            td_1 = TensorDict({"in": torch.randn(5, 3)}, [5])
            sub_seq_1 = td_module.select_subsequence(out_keys=["hidden"])
            params = make_functional(sub_seq_1)
            sub_seq_1(td_1, params=params)
            assert "hidden" in td_1.keys()
            assert "out" not in td_1.keys()
            td_2 = TensorDict({"hidden": torch.randn(5, 2)}, [5])
            sub_seq_2 = td_module.select_subsequence(in_keys=["hidden"])
            params = make_functional(sub_seq_2)
            sub_seq_2(td_2, params=params)
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

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)
        net2 = SafeModule(net2, in_keys=["b"], out_keys=["loc", "scale"])

        net3 = nn.Linear(4, 4 * param_multiplier)
        net3 = NormalParamWrapper(net3)
        net3 = SafeModule(net3, in_keys=["c"], out_keys=["loc", "scale"])

        spec = BoundedTensorSpec(-0.1, 0.1, 4)

        kwargs = {"distribution_class": TanhNormal}

        tdmodule1 = SafeModule(
            net1,
            in_keys=["a"],
            out_keys=["hidden"],
            spec=None,
            safe=False,
        )
        tdmodule2 = SafeProbabilisticTensorDictSequential(
            net2,
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=True,
                **kwargs,
            ),
        )
        tdmodule3 = SafeProbabilisticTensorDictSequential(
            net3,
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=["out"],
                spec=spec,
                safe=True,
                **kwargs,
            ),
        )
        tdmodule = SafeSequential(
            tdmodule1, tdmodule2, tdmodule3, partial_tolerant=True
        )

        if functional:
            params = make_functional(tdmodule)
        else:
            params = None

        if stack:
            td = torch.stack(
                [
                    TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, []),
                    TensorDict({"a": torch.randn(3), "c": torch.randn(4)}, []),
                ],
                0,
            )
            if functional:
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

    td_module = SafeModule(
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

    td_module = SafeModule(
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


class TestLSTMModule:
    def test_errs(self):
        with pytest.raises(ValueError, match="batch_first"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=False,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=[
                    "observation",
                    "hidden0",
                ],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys="abc",
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="in_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_key="smth",
                in_keys=[
                    "observation",
                    "hidden0",
                ],
                out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
            )
        with pytest.raises(ValueError, match="out_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys=["intermediate", ("next", "hidden0")],
            )
        with pytest.raises(ValueError, match="out_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_keys="abc",
            )
        with pytest.raises(ValueError, match="out_keys"):
            lstm_module = LSTMModule(
                input_size=3,
                hidden_size=12,
                batch_first=True,
                in_keys=["observation", "hidden0", "hidden1"],
                out_key="smth",
                out_keys=["intermediate", ("next", "hidden0")],
            )
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
        )
        td = TensorDict({"observation": torch.randn(3)}, [])
        with pytest.raises(KeyError, match="is_init"):
            lstm_module(td)

    def test_set_temporal_mode(self):
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
        )
        assert lstm_module.set_recurrent_mode(False) is lstm_module
        assert not lstm_module.set_recurrent_mode(False).temporal_mode
        assert lstm_module.set_recurrent_mode(True) is not lstm_module
        assert lstm_module.set_recurrent_mode(True).temporal_mode
        assert set(lstm_module.set_recurrent_mode(True).parameters()) == set(
            lstm_module.parameters()
        )

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    def test_singel_step(self, shape):
        td = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        lstm_module = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
        )
        td = lstm_module(td)
        td_next = step_mdp(td, keep_other=True)
        td_next = lstm_module(td_next)

        assert not torch.isclose(
            td_next["next", "hidden0"], td["next", "hidden0"]
        ).any()

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.parametrize("t", [1, 10])
    def test_single_step_vs_multi(self, shape, t):
        td = TensorDict(
            {
                "observation": torch.arange(t, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(*shape, t, 3),
                "is_init": torch.zeros(*shape, t, 1, dtype=torch.bool),
            },
            [*shape, t],
        )
        lstm_module_ss = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
        )
        lstm_module_ms = lstm_module_ss.set_recurrent_mode()
        lstm_module_ms(td)
        td_ss = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        for _t in range(t):
            lstm_module_ss(td_ss)
            td_ss = step_mdp(td_ss, keep_other=True)
            td_ss["observation"][:] = _t + 1
        torch.testing.assert_close(
            td_ss["hidden0"], td["next", "hidden0"][..., -1, :, :]
        )

    @pytest.mark.parametrize("shape", [[], [2], [2, 3], [2, 3, 4]])
    def test_multi_consecutive(self, shape):
        t = 20
        td = TensorDict(
            {
                "observation": torch.arange(t, dtype=torch.float32)
                .unsqueeze(-1)
                .expand(*shape, t, 3),
                "is_init": torch.zeros(*shape, t, 1, dtype=torch.bool),
            },
            [*shape, t],
        )
        if shape:
            td["is_init"][0, ..., 13, :] = True
        else:
            td["is_init"][13, :] = True

        lstm_module_ss = LSTMModule(
            input_size=3,
            hidden_size=12,
            batch_first=True,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")],
        )
        lstm_module_ms = lstm_module_ss.set_recurrent_mode()
        lstm_module_ms(td)
        td_ss = TensorDict(
            {
                "observation": torch.zeros(*shape, 3),
                "is_init": torch.zeros(*shape, 1, dtype=torch.bool),
            },
            shape,
        )
        for _t in range(t):
            td_ss["is_init"][:] = td["is_init"][..., _t, :]
            lstm_module_ss(td_ss)
            td_ss = step_mdp(td_ss, keep_other=True)
            td_ss["observation"][:] = _t + 1
        torch.testing.assert_close(
            td_ss["intermediate"], td["intermediate"][..., -1, :]
        )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
