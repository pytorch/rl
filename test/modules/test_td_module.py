# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
from _modules_common import _has_functorch
from tensordict import LazyStackedTensorDict, TensorDict, unravel_key_list
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import (
    AdditiveGaussianModule,
    NormalParamExtractor,
    SafeModule,
    TanhNormal,
    ValueOperator,
)
from torchrl.modules.tensordict_module.common import (
    ensure_tensordict_compatible,
    is_tensordict_compatible,
    VmapModule,
)
from torchrl.modules.tensordict_module.probabilistic import (
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)
from torchrl.modules.tensordict_module.sequence import SafeSequential
from torchrl.objectives import DDPGLoss


_vmap = None


def _get_vmap():
    global _vmap
    if _vmap is None:
        if hasattr(torch, "vmap"):
            _vmap = torch.vmap
        else:
            from functorch import vmap

            _vmap = vmap
    return _vmap


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
            "_": Unbounded((4,)),
            "out_2": Unbounded((3,)),
        }

        # warning due to "_" in spec keys
        with pytest.warns(UserWarning, match='got a spec with key "_"'):
            tensordict_module = SafeModule(
                MultiHeadLinear(5, 4, 3),
                in_keys=["input"],
                out_keys=["_", "out_2"],
                spec=Composite(**spec_dict),
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
            spec = Bounded(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = Unbounded(4)

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
            assert ((td.get("out") > 0.1) | (td.get("out") < -0.1)).any(), td.get("out")
        elif safe and spec_type == "bounded":
            assert ((td.get("out") < 0.1) | (td.get("out") > -0.1)).all()

    @pytest.mark.parametrize("safe", [True, False])
    @pytest.mark.parametrize("spec_type", [None, "bounded", "unbounded"])
    @pytest.mark.parametrize("out_keys", [["loc", "scale"], ["loc_1", "scale_1"]])
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "exp_mode", [InteractionType.DETERMINISTIC, InteractionType.RANDOM, None]
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
            module=nn.Sequential(net, NormalParamExtractor()),
            spec=None,
            in_keys=in_keys,
            out_keys=out_keys,
        )

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = Bounded(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = Unbounded(4)
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


class TestTDSequence:
    # Temporarily disabling this test until 473 is merged in tensordict
    # def test_in_key_warning(self):
    #     with pytest.warns(UserWarning, match='key "_" is for ignoring output'):
    #         tensordict_module = SafeModule(
    #             nn.Linear(3, 4), in_keys=["_"], out_keys=["out1"]
    #         )
    #     with pytest.warns(UserWarning, match='key "_" is for ignoring output'):
    #         tensordict_module = SafeModule(
    #             nn.Linear(3, 4), in_keys=["_", "key2"], out_keys=["out1"]
    #         )

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
            spec = Bounded(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = Unbounded(4)

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
        net2 = nn.Sequential(net2, NormalParamExtractor())

        if spec_type is None:
            spec = None
        elif spec_type == "bounded":
            spec = Bounded(-0.1, 0.1, 4)
        elif spec_type == "unbounded":
            spec = Unbounded(4)
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

    def test_submodule_sequence(self):
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
    def test_sequential_partial(self, stack):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = nn.Sequential(net2, NormalParamExtractor())
        net2 = SafeModule(net2, in_keys=["b"], out_keys=["loc", "scale"])

        net3 = nn.Linear(4, 4 * param_multiplier)
        net3 = nn.Sequential(net3, NormalParamExtractor())
        net3 = SafeModule(net3, in_keys=["c"], out_keys=["loc", "scale"])

        spec = Bounded(-0.1, 0.1, 4)

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

        if stack:
            td = LazyStackedTensorDict.maybe_dense_stack(
                [
                    TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, []),
                    TensorDict({"a": torch.randn(3), "c": torch.randn(4)}, []),
                ],
                0,
            )
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
    assert set(unravel_key_list(ensured_module.in_keys)) == {"x"}
    assert isinstance(ensured_module, TensorDictModule)


def test_safe_specs():
    out_key = ("a", "b")
    spec = Composite(Composite({out_key: Unbounded()}))
    original_spec = spec.clone()
    mod = SafeModule(
        module=nn.Linear(3, 1),
        spec=spec,
        out_keys=[out_key, ("other", "key")],
        in_keys=[],
    )
    assert original_spec == spec
    assert original_spec[out_key] == mod.spec[out_key]


def test_actor_critic_specs():
    action_key = ("agents", "action")
    spec = Composite(Composite({action_key: Unbounded(shape=(3,))}))
    policy_module = TensorDictModule(
        nn.Linear(3, 1),
        in_keys=[("agents", "observation")],
        out_keys=[action_key],
    )
    original_spec = spec.clone()
    module = TensorDictSequential(
        policy_module, AdditiveGaussianModule(spec=spec, action_key=action_key)
    )
    value_module = ValueOperator(
        module=module,
        in_keys=[("agents", "observation"), action_key],
        out_keys=[("agents", "state_action_value")],
    )
    assert original_spec == spec
    assert module[1].spec == spec
    DDPGLoss(actor_network=module, value_network=value_module)
    assert original_spec == spec
    assert module[1].spec == spec


def test_vmapmodule():
    lam = TensorDictModule(lambda x: x[0], in_keys=["x"], out_keys=["y"])
    sample_in = torch.ones((10, 3, 2))
    sample_in_td = TensorDict({"x": sample_in}, batch_size=[10])
    lam(sample_in)
    vm = VmapModule(lam, 0)
    vm(sample_in_td)
    assert (sample_in_td["x"][:, 0] == sample_in_td["y"]).all()


class TestFunctorchIntegration:
    """Test suite for functorch integration with TensorDictModule."""

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_tdmodule_functional_params(self):
        """Test that TDModule functional params can be extracted and used."""
        torch.manual_seed(0)

        net = nn.Linear(3, 4)
        td_module = SafeModule(
            module=net,
            in_keys=["input"],
            out_keys=["output"],
        )

        params = TensorDict.from_module(td_module)

        assert len(params.keys()) > 0
        for key in params.keys(True):
            assert params.get(key) is not None

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_vmap_on_tdmodule_with_params(self):
        """Test vmap on TensorDictModule with functional params."""
        torch.manual_seed(0)

        net = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        td_module = TensorDictModule(
            module=net,
            in_keys=["input"],
            out_keys=["output"],
        )

        params = TensorDict.from_module(td_module)
        params_expanded = params.expand(4, *params.shape)

        td = TensorDict({"input": torch.randn(2, 3)}, [2])

        def call(td, params):
            with params.to_module(td_module):
                return td_module(td.clone())

        vmap = _get_vmap()
        result = vmap(call, (None, 0))(td, params_expanded)

        assert result.shape == torch.Size([4, 2])
        assert "output" in result.keys()
        assert result["output"].shape == torch.Size([4, 2, 2])

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_nested_tdmodule_param_length(self):
        """Test nested TDModules (ProbabilisticTensorDictModule) param length."""
        torch.manual_seed(0)

        net = nn.Sequential(nn.Linear(3, 4), NormalParamExtractor())
        base_module = TensorDictModule(
            module=net,
            in_keys=["input"],
            out_keys=["loc", "scale"],
        )

        prob_module = SafeProbabilisticModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
        )

        td_sequence = SafeProbabilisticTensorDictSequential(base_module, prob_module)

        params = TensorDict.from_module(td_sequence)

        assert len(params.keys()) > 0

        for key in params.keys(True):
            assert params.get(key) is not None

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_nested_tdmodule_param_casting(self):
        """Test nested TDModules param casting."""
        torch.manual_seed(0)

        net = nn.Sequential(nn.Linear(3, 4), NormalParamExtractor())
        base_module = TensorDictModule(
            module=net,
            in_keys=["input"],
            out_keys=["loc", "scale"],
        )

        prob_module = SafeProbabilisticModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
        )

        td_sequence = SafeProbabilisticTensorDictSequential(base_module, prob_module)

        params = TensorDict.from_module(td_sequence)

        params_float64 = params.to(torch.float64)

        for key in params_float64.keys(True):
            assert params_float64.get(key).dtype == torch.float64

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_tdsequence_vmap_params(self):
        """Test TDSequence param handling with vmap."""
        torch.manual_seed(0)

        td_module1 = SafeModule(
            module=nn.Linear(3, 4),
            in_keys=["input"],
            out_keys=["hidden"],
        )

        td_module2 = SafeModule(
            module=nn.Linear(4, 2),
            in_keys=["hidden"],
            out_keys=["output"],
        )

        td_sequence = SafeSequential(td_module1, td_module2)

        params = TensorDict.from_module(td_sequence)

        td = TensorDict({"input": torch.randn(2, 3)}, [2])
        with params.to_module(td_sequence):
            result = td_sequence(td.clone())

        assert "output" in result.keys()
        assert result["output"].shape == torch.Size([2, 2])

        params_expanded = params.expand(4, *params.shape)

        def call(td, params):
            with params.to_module(td_sequence):
                return td_sequence(td.clone())

        vmap = _get_vmap()
        result_vmap = vmap(call, (None, 0))(td, params_expanded)

        assert result_vmap.shape == torch.Size([4, 2])
        assert "output" in result_vmap.keys()
        assert result_vmap["output"].shape == torch.Size([4, 2, 2])

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_vmap_multiple_inputs(self):
        """Test vmap with multiple inputs to module."""
        torch.manual_seed(0)

        class MultiInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 4)
                self.fc2 = nn.Linear(4, 4)

            def forward(self, x, y):
                return self.fc1(x) + self.fc2(y)

        net = MultiInputModule()
        td_module = TensorDictModule(
            module=net,
            in_keys=["x", "y"],
            out_keys=["output"],
        )

        params = TensorDict.from_module(td_module)
        params_expanded = params.expand(3, *params.shape)

        td = TensorDict(
            {
                "x": torch.randn(2, 3),
                "y": torch.randn(2, 4),
            },
            [2],
        )

        def call(td, params):
            with params.to_module(td_module):
                return td_module(td.clone())

        vmap = _get_vmap()
        result = vmap(call, (None, 0))(td, params_expanded)

        assert result.shape == torch.Size([3, 2])
        assert "output" in result.keys()
        assert result["output"].shape == torch.Size([3, 2, 4])

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_vmap_module_class(self):
        """Test VmapModule class."""
        torch.manual_seed(0)

        td_module = TensorDictModule(
            module=nn.Linear(3, 4),
            in_keys=["input"],
            out_keys=["output"],
        )

        vmapped_module = VmapModule(td_module, vmap_dim=0)

        td = TensorDict({"input": torch.randn(3, 3)}, [3])
        result = vmapped_module(td)

        assert result["output"].shape == torch.Size([3, 4])

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_tdsequence_vmap(self):
        """Test vmap on TDSequence."""
        torch.manual_seed(0)

        td_module1 = SafeModule(
            module=nn.Linear(3, 4),
            in_keys=["input"],
            out_keys=["hidden"],
        )

        td_module2 = SafeModule(
            module=nn.Linear(4, 2),
            in_keys=["hidden"],
            out_keys=["output"],
        )

        td_sequence = SafeSequential(td_module1, td_module2)

        params = TensorDict.from_module(td_sequence)
        params_expanded = params.expand(4, *params.shape)

        td = TensorDict({"input": torch.randn(2, 3)}, [2])

        def call(td, params):
            with params.to_module(td_sequence):
                return td_sequence(td.clone())

        vmap = _get_vmap()
        result = vmap(call, (None, 0))(td, params_expanded)

        assert result.shape == torch.Size([4, 2])
        assert "output" in result.keys()
        assert result["output"].shape == torch.Size([4, 2, 2])

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_tdmodule_functional_to_module(self):
        """Test TDModule functional params with to_module context manager."""
        torch.manual_seed(0)

        net = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        td_module = TensorDictModule(
            module=net,
            in_keys=["input"],
            out_keys=["output"],
        )

        params = TensorDict.from_module(td_module)

        td = TensorDict({"input": torch.randn(3, 3)}, [3])
        with params.to_module(td_module):
            result = td_module(td.clone())

        assert "output" in result.keys()
        assert result["output"].shape == torch.Size([3, 2])

    @pytest.mark.skipif(not _has_functorch, reason="functorch is required")
    def test_nested_sequential_vmap(self):
        """Test vmap on nested sequential modules."""
        torch.manual_seed(0)

        net1 = nn.Linear(3, 4)
        net2 = nn.Linear(4, 2)
        net3 = nn.Linear(2, 1)

        td_seq1 = TensorDictSequential(
            TensorDictModule(net1, in_keys=["a"], out_keys=["b"]),
            TensorDictModule(net2, in_keys=["b"], out_keys=["c"]),
        )

        td_seq2 = TensorDictSequential(
            TensorDictModule(net3, in_keys=["c"], out_keys=["d"]),
        )

        td_sequence = TensorDictSequential(td_seq1, td_seq2)

        params = TensorDict.from_module(td_sequence)
        params_expanded = params.expand(2, *params.shape)

        td = TensorDict({"a": torch.randn(3, 3)}, [3])

        def call(td, params):
            with params.to_module(td_sequence):
                return td_sequence(td.clone())

        vmap = _get_vmap()
        result = vmap(call, (None, 0))(td, params_expanded)

        assert result.shape == torch.Size([2, 3])
        assert "d" in result.keys()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
