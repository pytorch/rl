# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import functools
import operator
import warnings
from dataclasses import dataclass

import pytest
import torch
from _objectives_common import _has_functorch, FUNCTORCH_ERR

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    NormalParamExtractor,
    set_composite_lp_aggregate,
    TensorDictModule,
    TensorDictSequential,
)
from tensordict.nn.utils import Buffer
from torch import nn

from torchrl._utils import _standardize, rl_warnings
from torchrl.data import Bounded, Composite
from torchrl.envs.utils import exploration_type, ExplorationType, set_exploration_type
from torchrl.modules import recurrent_mode
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.tensordict_module.actors import (
    ActorValueOperator,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss, DQNLoss, PPOLoss, SACLoss
from torchrl.objectives.common import add_random_module, LossModule
from torchrl.objectives.utils import _vmap_func, HardUpdate, hold_out_net, SoftUpdate
from torchrl.objectives.value.advantages import GAE, TD0Estimator
from torchrl.objectives.value.functional import _transpose_time, reward2go
from torchrl.objectives.value.utils import (
    _get_num_per_traj,
    _get_num_per_traj_init,
    _inv_pad_sequence,
    _split_and_pad_sequence,
)

from torchrl.testing import (  # noqa
    call_value_nets as _call_value_nets,
    dtype_fixture,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
)


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("vmap_randomness", (None, "different", "same", "error"))
@pytest.mark.parametrize("dropout", (0.0, 0.1))
def test_loss_vmap_random(device, vmap_randomness, dropout):
    class VmapTestLoss(LossModule):
        model: TensorDictModule
        model_params: TensorDict
        target_model_params: TensorDict

        def __init__(self):
            super().__init__()
            layers = [nn.Linear(4, 4), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(4, 4))
            net = nn.Sequential(*layers).to(device)
            model = TensorDictModule(net, in_keys=["obs"], out_keys=["action"])
            self.convert_to_functional(model, "model", expand_dim=4)
            self._make_vmap()

        def _make_vmap(self):
            self.vmap_model = _vmap_func(
                self.model,
                (None, 0),
                randomness=(
                    "error" if vmap_randomness == "error" else self.vmap_randomness
                ),
            )

        def forward(self, td):
            out = self.vmap_model(td, self.model_params)
            return {"loss": out["action"].mean()}

    loss_module = VmapTestLoss()
    td = TensorDict({"obs": torch.randn(3, 4).to(device)}, [3])

    # If user sets vmap randomness to a specific value
    if vmap_randomness in ("different", "same") and dropout > 0.0:
        loss_module.set_vmap_randomness(vmap_randomness)
    # Fail case
    elif vmap_randomness == "error" and dropout > 0.0:
        with pytest.raises(
            RuntimeError,
            match="vmap: called random operation while in randomness error mode",
        ):
            loss_module(td)["loss"]
        return
    loss_module(td)["loss"]


def test_hold_out():
    net = torch.nn.Linear(3, 4)
    x = torch.randn(1, 3)
    x_rg = torch.randn(1, 3, requires_grad=True)
    y = net(x)
    assert y.requires_grad
    with hold_out_net(net):
        y = net(x)
        assert not y.requires_grad
        y = net(x_rg)
        assert y.requires_grad

    y = net(x)
    assert y.requires_grad

    # nested case
    with hold_out_net(net):
        y = net(x)
        assert not y.requires_grad
        with hold_out_net(net):
            y = net(x)
            assert not y.requires_grad
            y = net(x_rg)
            assert y.requires_grad

    y = net(x)
    assert y.requires_grad

    # exception
    net = torch.nn.Sequential()
    with hold_out_net(net):
        pass


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("value_network_update_interval", [100, 1000])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.float32,
    ],
)
def test_updater(mode, value_network_update_interval, device, dtype):
    torch.manual_seed(100)

    class custom_module_error(nn.Module):
        def __init__(self):
            super().__init__()
            self.target_params = [torch.randn(3, 4)]
            self.target_error_params = [torch.randn(3, 4)]
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(3, 4, requires_grad=True))]
            )

    module = custom_module_error().to(device)
    with pytest.raises(
        ValueError, match="The loss_module must be a LossModule instance"
    ):
        if mode == "hard":
            upd = HardUpdate(
                module, value_network_update_interval=value_network_update_interval
            )
        elif mode == "soft":
            upd = SoftUpdate(module, eps=1 - 1 / value_network_update_interval)

    class custom_module(LossModule):
        module1: TensorDictModule
        module1_params: TensorDict
        target_module1_params: TensorDict

        def __init__(self, delay_module=True):
            super().__init__()
            module1 = torch.nn.BatchNorm2d(10).eval()
            self.convert_to_functional(
                module1, "module1", create_target_params=delay_module
            )

            module2 = torch.nn.BatchNorm2d(10).eval()
            self.module2 = module2
            tparam = self._modules.get("target_module1_params", None)
            if tparam is None:
                tparam = self._modules.get("module1_params").data
            iterator_params = tparam.values(include_nested=True, leaves_only=True)
            for target in iterator_params:
                if target.dtype is not torch.int64:
                    target.data.normal_()
                else:
                    target.data += 10

        def _forward_value_estimator_keys(self, **kwargs) -> None:
            pass

    module = custom_module(delay_module=False)
    with pytest.raises(
        RuntimeError,
        match="Did not find any target parameters or buffers in the loss module",
    ):
        if mode == "hard":
            upd = HardUpdate(
                module, value_network_update_interval=value_network_update_interval
            )
        elif mode == "soft":
            upd = SoftUpdate(
                module,
                eps=1 - 1 / value_network_update_interval,
            )
        else:
            raise NotImplementedError

    # this is now allowed
    # with pytest.warns(UserWarning, match="No target network updater has been"):
    #     module = custom_module().to(device).to(dtype)

    if mode == "soft":
        with pytest.raises(ValueError, match="One and only one argument"):
            upd = SoftUpdate(
                module,
                eps=1 - 1 / value_network_update_interval,
                tau=0.1,
            )

    module = custom_module(delay_module=True)
    _ = module.module1_params
    with pytest.warns(
        UserWarning, match="No target network updater has been"
    ) if rl_warnings() else contextlib.nullcontext():
        _ = module.target_module1_params
    if mode == "hard":
        upd = HardUpdate(
            module, value_network_update_interval=value_network_update_interval
        )
    elif mode == "soft":
        upd = SoftUpdate(module, eps=1 - 1 / value_network_update_interval)
    for _, _v in upd._targets.items(True, True):
        if _v.dtype is not torch.int64:
            _v.copy_(torch.randn_like(_v))
        else:
            _v += 10

    # total dist
    d0 = 0.0
    for key, source_val in upd._sources.items(True, True):
        if not isinstance(key, tuple):
            key = (key,)
        key = ("target_" + key[0], *key[1:])
        target_val = upd._targets[key]
        assert target_val.dtype is source_val.dtype, key
        assert target_val.device == source_val.device, key
        if target_val.dtype == torch.long:
            continue
        with torch.no_grad():
            d0 += (target_val - source_val).norm().item()

    assert d0 > 0
    if mode == "hard":
        for i in range(value_network_update_interval + 1):
            # test that no update is occurring until value_network_update_interval
            d1 = 0.0
            for key, source_val in upd._sources.items(True, True):
                if not isinstance(key, tuple):
                    key = (key,)
                key = ("target_" + key[0], *key[1:])
                target_val = upd._targets[key]
                if target_val.dtype == torch.long:
                    continue
                with torch.no_grad():
                    d1 += (target_val - source_val).norm().item()

            assert d1 == d0, i
            assert upd.counter == i
            upd.step()
        assert upd.counter == 0
        # test that a new update has occurred
        d1 = 0.0
        for key, source_val in upd._sources.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            key = ("target_" + key[0], *key[1:])
            target_val = upd._targets[key]
            if target_val.dtype == torch.long:
                continue
            with torch.no_grad():
                d1 += (target_val - source_val).norm().item()
        assert d1 < d0

    elif mode == "soft":
        upd.step()
        d1 = 0.0
        for key, source_val in upd._sources.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            key = ("target_" + key[0], *key[1:])
            target_val = upd._targets[key]
            if target_val.dtype == torch.long:
                continue
            with torch.no_grad():
                d1 += (target_val - source_val).norm().item()
        assert d1 < d0
    with pytest.warns(UserWarning, match="already"):
        upd.init_()
    upd.step()
    d2 = 0.0
    for key, source_val in upd._sources.items(True, True):
        if not isinstance(key, tuple):
            key = (key,)
        key = ("target_" + key[0], *key[1:])
        target_val = upd._targets[key]
        if target_val.dtype == torch.long:
            continue
        with torch.no_grad():
            d2 += (target_val - source_val).norm().item()
    assert d2 < 1e-6


@pytest.mark.skipif(
    not _has_functorch,
    reason=f"no vmap allowed without functorch, error: {FUNCTORCH_ERR}",
)
@pytest.mark.parametrize(
    "dest,expected_dtype,expected_device",
    list(
        zip(
            get_available_devices(),
            [torch.float] * len(get_available_devices()),
            get_available_devices(),
        )
    )
    + [
        ["cuda", torch.float, "cuda:0"],
        ["double", torch.double, "cpu"],
        [torch.double, torch.double, "cpu"],
        [torch.half, torch.half, "cpu"],
        ["half", torch.half, "cpu"],
    ],
)
@set_composite_lp_aggregate(False)
def test_shared_params(dest, expected_dtype, expected_device):
    if torch.cuda.device_count() == 0 and dest == "cuda":
        pytest.skip("no cuda device available")
    module_hidden = torch.nn.Linear(4, 4)
    td_module_hidden = TensorDictModule(
        module=module_hidden,
        in_keys=["observation"],
        out_keys=["hidden"],
    )
    module_action = TensorDictModule(
        nn.Sequential(nn.Linear(4, 8), NormalParamExtractor()),
        in_keys=["hidden"],
        out_keys=["loc", "scale"],
    )
    td_module_action = ProbabilisticActor(
        module=module_action,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        spec=None,
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    module_value = torch.nn.Linear(4, 1)
    td_module_value = ValueOperator(module=module_value, in_keys=["hidden"])
    td_module = ActorValueOperator(td_module_hidden, td_module_action, td_module_value)

    class MyLoss(LossModule):
        actor_network: TensorDictModule
        actor_network_params: TensorDict
        target_actor_network_params: TensorDict
        qvalue_network: TensorDictModule
        qvalue_network_params: TensorDict
        target_qvalue_network_params: TensorDict

        def __init__(self, actor_network, qvalue_network):
            super().__init__()
            self.convert_to_functional(
                actor_network,
                "actor_network",
                create_target_params=True,
            )
            self.convert_to_functional(
                qvalue_network,
                "qvalue_network",
                3,
                create_target_params=True,
                compare_against=list(actor_network.parameters()),
            )

        def _forward_value_estimator_keys(self, **kwargs) -> None:
            pass

    actor_network = td_module.get_policy_operator()
    value_network = td_module.get_value_operator()

    loss = MyLoss(actor_network, value_network)
    # modify params
    for p in loss.parameters():
        if p.requires_grad:
            p.data += torch.randn_like(p)

    assert len([p for p in loss.parameters() if p.requires_grad]) == 6
    assert (
        len(loss.actor_network_params.keys(include_nested=True, leaves_only=True)) == 4
    )
    assert (
        len(loss.qvalue_network_params.keys(include_nested=True, leaves_only=True)) == 4
    )
    for p in loss.actor_network_params.values(include_nested=True, leaves_only=True):
        assert isinstance(p, nn.Parameter) or isinstance(p, Buffer)
    for i, (key, value) in enumerate(
        loss.qvalue_network_params.items(include_nested=True, leaves_only=True)
    ):
        p1 = value
        p2 = loss.actor_network_params[key]
        assert (p1 == p2).all()
        if i == 1:
            break

    # map module
    if dest == "double":
        loss = loss.double()
    elif dest == "cuda":
        loss = loss.cuda()
    elif dest == "half":
        loss = loss.half()
    else:
        loss = loss.to(dest)

    for p in loss.actor_network_params.values(include_nested=True, leaves_only=True):
        assert isinstance(p, nn.Parameter)
        assert p.dtype is expected_dtype
        assert p.device == torch.device(expected_device)
    for i, (key, qvalparam) in enumerate(
        loss.qvalue_network_params.items(include_nested=True, leaves_only=True)
    ):
        assert qvalparam.dtype is expected_dtype, (key, qvalparam)
        assert qvalparam.device == torch.device(expected_device), key
        assert (qvalparam == loss.actor_network_params[key]).all(), key
        if i == 1:
            break


class TestBase:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    def test_decorators(self):
        class MyLoss(LossModule):
            def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
                assert recurrent_mode()
                assert exploration_type() is ExplorationType.DETERMINISTIC
                return TensorDict()

            def actor_loss(self, tensordict: TensorDictBase) -> TensorDictBase:
                assert recurrent_mode()
                assert exploration_type() is ExplorationType.DETERMINISTIC
                return TensorDict()

            def something_loss(self, tensordict: TensorDictBase) -> TensorDictBase:
                assert recurrent_mode()
                assert exploration_type() is ExplorationType.DETERMINISTIC
                return TensorDict()

        loss = MyLoss()
        loss.forward(None)
        loss.actor_loss(None)
        loss.something_loss(None)
        assert not recurrent_mode()

    @pytest.mark.parametrize("expand_dim", [None, 2])
    @pytest.mark.parametrize("compare_against", [True, False])
    @pytest.mark.skipif(not _has_functorch, reason="functorch is needed for expansion")
    def test_convert_to_func(self, compare_against, expand_dim):
        class MyLoss(LossModule):
            module_a: TensorDictModule
            module_b: TensorDictModule
            module_a_params: TensorDict
            module_b_params: TensorDict
            target_module_a_params: TensorDict
            target_module_b_params: TensorDict

            def __init__(self, compare_against, expand_dim):
                super().__init__()
                module1 = nn.Linear(3, 4)
                module2 = nn.Linear(3, 4)
                module3 = nn.Linear(3, 4)
                module_a = TensorDictModule(
                    nn.Sequential(module1, module2), in_keys=["a"], out_keys=["c"]
                )
                module_b = TensorDictModule(
                    nn.Sequential(module1, module3), in_keys=["b"], out_keys=["c"]
                )
                self.convert_to_functional(module_a, "module_a")
                self.convert_to_functional(
                    module_b,
                    "module_b",
                    compare_against=module_a.parameters() if compare_against else [],
                    expand_dim=expand_dim,
                )

        loss_module = MyLoss(compare_against=compare_against, expand_dim=expand_dim)

        for key in ["module.0.bias", "module.0.weight"]:
            if compare_against:
                assert not loss_module.module_b_params.flatten_keys()[key].requires_grad
            else:
                assert loss_module.module_b_params.flatten_keys()[key].requires_grad
            if expand_dim:
                assert (
                    loss_module.module_b_params.flatten_keys()[key].shape[0]
                    == expand_dim
                )
            else:
                assert (
                    loss_module.module_b_params.flatten_keys()[key].shape[0]
                    != expand_dim
                )

        for key in ["module.1.bias", "module.1.weight"]:
            loss_module.module_b_params.flatten_keys()[key].requires_grad

    def test_init_params(self):
        class MyLoss(LossModule):
            module_a: TensorDictModule
            module_b: TensorDictModule
            module_a_params: TensorDict
            module_b_params: TensorDict
            target_module_a_params: TensorDict
            target_module_b_params: TensorDict

            def __init__(self, expand_dim=2):
                super().__init__()
                module1 = nn.Linear(3, 4)
                module2 = nn.Linear(3, 4)
                module3 = nn.Linear(3, 4)
                module_a = TensorDictModule(
                    nn.Sequential(module1, module2), in_keys=["a"], out_keys=["c"]
                )
                module_b = TensorDictModule(
                    nn.Sequential(module1, module3), in_keys=["b"], out_keys=["c"]
                )
                self.convert_to_functional(module_a, "module_a")
                self.convert_to_functional(
                    module_b,
                    "module_b",
                    compare_against=module_a.parameters(),
                    expand_dim=expand_dim,
                )

        loss = MyLoss()

        module_a = loss.get_stateful_net("module_a", copy=False)
        assert module_a is loss.module_a

        module_a = loss.get_stateful_net("module_a")
        assert module_a is not loss.module_a

        def init(mod):
            if hasattr(mod, "weight"):
                mod.weight.data.zero_()
            if hasattr(mod, "bias"):
                mod.bias.data.zero_()

        module_a.apply(init)
        assert (loss.module_a_params == 0).all()

        def init(mod):
            if hasattr(mod, "weight"):
                mod.weight = torch.nn.Parameter(mod.weight.data + 1)
            if hasattr(mod, "bias"):
                mod.bias = torch.nn.Parameter(mod.bias.data + 1)

        module_a.apply(init)
        assert (loss.module_a_params == 0).all()
        loss.from_stateful_net("module_a", module_a)
        assert (loss.module_a_params == 1).all()

    def test_from_module_list(self):
        class MyLoss(LossModule):
            module_a: TensorDictModule
            module_b: TensorDictModule

            module_a_params: TensorDict
            module_b_params: TensorDict

            target_module_a_params: TensorDict
            target_module_b_params: TensorDict

            def __init__(self, module_a, module_b0, module_b1, expand_dim=2):
                super().__init__()
                self.convert_to_functional(module_a, "module_a")
                self.convert_to_functional(
                    [module_b0, module_b1],
                    "module_b",
                    # This will be ignored
                    compare_against=module_a.parameters(),
                    expand_dim=expand_dim,
                )

        module1 = nn.Linear(3, 4)
        module2 = nn.Linear(3, 4)
        module3a = nn.Linear(3, 4)
        module3b = nn.Linear(3, 4)

        module_a = TensorDictModule(
            nn.Sequential(module1, module2), in_keys=["a"], out_keys=["c"]
        )

        module_b0 = TensorDictModule(
            nn.Sequential(module1, module3a), in_keys=["b"], out_keys=["c"]
        )
        module_b1 = TensorDictModule(
            nn.Sequential(module1, module3b), in_keys=["b"], out_keys=["c"]
        )

        loss = MyLoss(module_a, module_b0, module_b1)

        # This should be extended
        assert not isinstance(
            loss.module_b_params["module", "0", "weight"], nn.Parameter
        )
        assert loss.module_b_params["module", "0", "weight"].shape[0] == 2
        assert (
            loss.module_b_params["module", "0", "weight"].data.data_ptr()
            == loss.module_a_params["module", "0", "weight"].data.data_ptr()
        )
        assert isinstance(loss.module_b_params["module", "1", "weight"], nn.Parameter)
        assert loss.module_b_params["module", "1", "weight"].shape[0] == 2
        assert (
            loss.module_b_params["module", "1", "weight"].data.data_ptr()
            != loss.module_a_params["module", "1", "weight"].data.data_ptr()
        )

    def test_tensordict_keys(self):
        """Test configurable tensordict key behavior with derived classes."""

        class MyLoss(LossModule):
            def __init__(self):
                super().__init__()

        loss_module = MyLoss()
        with pytest.raises(AttributeError):
            loss_module.set_keys()

        class MyLoss2(MyLoss):
            def _forward_value_estimator_keys(self, **kwargs) -> None:
                pass

        loss_module = MyLoss2()
        assert loss_module.set_keys() is None
        with pytest.raises(ValueError):
            loss_module.set_keys(some_key="test")

        class MyLoss3(MyLoss2):
            @dataclass
            class _AcceptedKeys:
                some_key: str = "some_value"

        loss_module = MyLoss3()
        assert loss_module.tensor_keys.some_key == "some_value"
        loss_module.set_keys(some_key="test")
        assert loss_module.tensor_keys.some_key == "test"


class TestUtils:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    def test_add_random_module(self):
        class MyMod(nn.Module):
            ...

        add_random_module(MyMod)
        import torchrl.objectives.utils

        assert MyMod in torchrl.objectives.utils.RANDOM_MODULE_LIST

    def test_standardization(self):
        t = torch.arange(3 * 4 * 5 * 6, dtype=torch.float32).view(3, 4, 5, 6)
        std_t0 = _standardize(t, exclude_dims=(1, 3))
        std_t1 = (t - t.mean((0, 2), keepdim=True)) / t.std((0, 2), keepdim=True).clamp(
            1 - 6
        )
        torch.testing.assert_close(std_t0, std_t1)
        std_t = _standardize(t, (), -1, 2)
        torch.testing.assert_close(std_t, (t + 1) / 2)
        std_t = _standardize(t, ())
        torch.testing.assert_close(std_t, (t - t.mean()) / t.std())

    @pytest.mark.parametrize("B", [None, (1, ), (4, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [1, 10])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_get_num_per_traj_no_stops(self, B, T, device):
        """check _get_num_per_traj when input contains no stops"""
        size = (*B, T) if B else (T,)

        done = torch.zeros(*size, dtype=torch.bool, device=device)
        splits = _get_num_per_traj(done)

        count = functools.reduce(operator.mul, B, 1) if B else 1
        res = torch.full((count,), T, device=device)

        torch.testing.assert_close(splits, res)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_get_num_per_traj(self, B, T, device):
        """check _get_num_per_traj where input contains a stop at half of each trace"""
        size = (*B, T)

        done = torch.zeros(*size, dtype=torch.bool, device=device)
        done[..., T // 2] = True
        splits = _get_num_per_traj(done)

        count = functools.reduce(operator.mul, B, 1)
        res = [T - (T + 1) // 2 + 1, (T + 1) // 2 - 1] * count
        res = torch.as_tensor(res, device=device)

        torch.testing.assert_close(splits, res)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_split_pad_reverse(self, B, T, device):
        """calls _split_and_pad_sequence and reverts it"""
        torch.manual_seed(42)

        size = (*B, T)
        traj = torch.rand(*size, device=device)
        done = torch.zeros(*size, dtype=torch.bool, device=device).bernoulli(0.2)
        splits = _get_num_per_traj(done)

        splitted = _split_and_pad_sequence(traj, splits)
        reversed = _inv_pad_sequence(splitted, splits).reshape(traj.shape)

        torch.testing.assert_close(traj, reversed)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_split_pad_no_stops(self, B, T, device):
        """_split_and_pad_sequence on trajectories without stops should not change input but flatten it along batch dimension"""
        size = (*B, T)
        count = functools.reduce(operator.mul, size, 1)

        traj = torch.arange(0, count, device=device).reshape(size)
        done = torch.zeros(*size, dtype=torch.bool, device=device)

        splits = _get_num_per_traj(done)
        splitted = _split_and_pad_sequence(traj, splits)

        traj_flat = traj.flatten(0, -2)
        torch.testing.assert_close(traj_flat, splitted)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_split_pad_manual(self, device):
        """handcrafted example to test _split_and_pad_seqeunce"""

        traj = torch.as_tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], device=device)
        splits = torch.as_tensor([3, 2, 1, 4], device=device)
        res = torch.as_tensor(
            [[0, 1, 2, 0], [3, 4, 0, 0], [5, 0, 0, 0], [6, 7, 8, 9]], device=device
        )

        splitted = _split_and_pad_sequence(traj, splits)
        torch.testing.assert_close(res, splitted)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_split_pad_reverse_tensordict(self, B, T, device):
        """calls _split_and_pad_sequence and reverts it on tensordict input"""
        torch.manual_seed(42)

        td = TensorDict(
            {
                "observation": torch.arange(T, dtype=torch.float32, device=device)
                .unsqueeze(-1)
                .expand(*B, T, 3),
                "is_init": torch.zeros(
                    *B, T, 1, dtype=torch.bool, device=device
                ).bernoulli(0.3),
            },
            [*B, T],
            device=device,
        )

        is_init = td.get("is_init").squeeze(-1)
        splits = _get_num_per_traj_init(is_init)
        splitted = _split_and_pad_sequence(
            td.select("observation", strict=False), splits
        )

        reversed = _inv_pad_sequence(splitted, splits)
        reversed = reversed.reshape(td.shape)
        torch.testing.assert_close(td["observation"], reversed["observation"])

    def test_reward2go(self):
        reward = torch.zeros(4, 2)
        reward[3, 0] = 1
        reward[3, 1] = -1
        done = torch.zeros(4, 2, dtype=bool)
        done[3, :] = True
        r = torch.ones(4)
        r[1:] = 0.9
        r = torch.cumprod(r, 0).flip(0)
        r = torch.stack([r, -r], -1)
        torch.testing.assert_close(reward2go(reward, done, 0.9), r)

        reward = torch.zeros(4, 1)
        reward[3, 0] = 1
        done = torch.zeros(4, 1, dtype=bool)
        done[3, :] = True
        r = torch.ones(4)
        r[1:] = 0.9
        reward = reward.expand(2, 4, 1)
        done = done.expand(2, 4, 1)
        r = torch.cumprod(r, 0).flip(0).unsqueeze(-1).expand(2, 4, 1)
        r2go = reward2go(reward, done, 0.9)
        torch.testing.assert_close(r2go, r)

    def test_timedimtranspose_single(self):
        @_transpose_time
        def fun(a, b, time_dim=-2):
            return a + 1

        x = torch.zeros(10)
        y = torch.ones(10)
        with pytest.raises(RuntimeError):
            z = fun(x, y, time_dim=-3)
        with pytest.raises(RuntimeError):
            z = fun(x, y, time_dim=-2)
        z = fun(x, y, time_dim=-1)
        assert z.shape == torch.Size([10])
        assert (z == 1).all()

        @_transpose_time
        def fun(a, b, time_dim=-2):
            return a + 1, b + 1

        with pytest.raises(RuntimeError):
            z1, z2 = fun(x, y, time_dim=-3)
        with pytest.raises(RuntimeError):
            z1, z2 = fun(x, y, time_dim=-2)
        z1, z2 = fun(x, y, time_dim=-1)
        assert z1.shape == torch.Size([10])
        assert (z1 == 1).all()
        assert z2.shape == torch.Size([10])
        assert (z2 == 2).all()


@pytest.mark.parametrize(
    "updater,kwarg",
    [
        (HardUpdate, {"value_network_update_interval": 1000}),
        (SoftUpdate, {"eps": 0.99}),
    ],
)
@set_composite_lp_aggregate(False)
def test_updater_warning(updater, kwarg):
    with warnings.catch_warnings():
        dqn = DQNLoss(torch.nn.Linear(3, 4), delay_value=True, action_space="one_hot")
    with pytest.warns(UserWarning) if rl_warnings() else contextlib.nullcontext():
        dqn.target_value_network_params
    with warnings.catch_warnings():
        updater(dqn, **kwarg)
    with warnings.catch_warnings():
        dqn.target_value_network_params


class TestSingleCall:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    def _mock_value_net(self, has_target, value_key):
        model = nn.Linear(3, 1)
        module = TensorDictModule(model, in_keys=["obs"], out_keys=[value_key])
        params = TensorDict(dict(module.named_parameters()), []).unflatten_keys(".")
        if has_target:
            return (
                module,
                params,
                params.apply(lambda x: x.detach() + torch.randn_like(x)),
            )
        return module, params, params

    def _mock_data(self):
        return TensorDict(
            {
                "obs": torch.randn(10, 3),
                ("next", "obs"): torch.randn(10, 3),
                ("next", "reward"): torch.randn(10, 1),
                ("next", "done"): torch.zeros(10, 1, dtype=torch.bool),
            },
            [10],
            names=["time"],
        )

    @pytest.mark.parametrize("has_target", [True, False])
    @pytest.mark.parametrize("single_call", [True, False])
    @pytest.mark.parametrize("value_key", ["value", ("some", "other", "key")])
    def test_single_call(self, has_target, value_key, single_call, detach_next=True):
        torch.manual_seed(0)
        value_net, params, next_params = self._mock_value_net(has_target, value_key)
        data = self._mock_data()
        if single_call and has_target:
            with pytest.raises(
                ValueError,
                match=r"without recurring to vmap when both params and next params are passed",
            ):
                _call_value_nets(
                    value_net,
                    data,
                    params,
                    next_params,
                    single_call,
                    value_key,
                    detach_next,
                )
            return
        value, value_ = _call_value_nets(
            value_net, data, params, next_params, single_call, value_key, detach_next
        )
        assert (value != value_).all()


@set_composite_lp_aggregate(False)
def test_instantiate_with_different_keys():
    loss_1 = DQNLoss(
        value_network=nn.Linear(3, 3), action_space="one_hot", delay_value=True
    )
    loss_1.set_keys(reward="a")
    assert loss_1.tensor_keys.reward == "a"
    loss_2 = DQNLoss(
        value_network=nn.Linear(3, 3), action_space="one_hot", delay_value=True
    )
    loss_2.set_keys(reward="b")
    assert loss_1.tensor_keys.reward == "a"


class TestBuffer:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    # @pytest.mark.parametrize('dtype', (torch.double, torch.float, torch.half))
    # def test_param_cast(self, dtype):
    #     param = nn.Parameter(torch.zeros(3))
    #     idb = param.data_ptr()
    #     param = param.to(dtype)
    #     assert param.data_ptr() == idb
    #     assert param.dtype == dtype
    #     assert param.data.dtype == dtype
    # @pytest.mark.parametrize('dtype', (torch.double, torch.float, torch.half))
    # def test_buffer_cast(self, dtype):
    #     buffer = Buffer(torch.zeros(3))
    #     idb = buffer.data_ptr()
    #     buffer = buffer.to(dtype)
    #     assert isinstance(buffer, Buffer)
    #     assert buffer.data_ptr() == idb
    #     assert buffer.dtype == dtype
    #     assert buffer.data.dtype == dtype

    @pytest.mark.parametrize("create_target_params", [True, False])
    @pytest.mark.parametrize(
        "dest", [torch.float, torch.double, torch.half, *get_default_devices()]
    )
    def test_module_cast(self, create_target_params, dest):
        # test that when casting a loss module, all the tensors (params and buffers)
        # are properly cast
        class DummyModule(LossModule):
            actor: TensorDictModule
            value: TensorDictModule
            actor_params: TensorDict
            value_params: TensorDict
            target_actor_params: TensorDict
            target_value_params: TensorDict

            def __init__(self):
                common = nn.Linear(3, 4)
                actor = nn.Linear(4, 4)
                value = nn.Linear(4, 1)
                common = TensorDictModule(common, in_keys=["obs"], out_keys=["hidden"])
                actor = TensorDictSequential(
                    common,
                    TensorDictModule(actor, in_keys=["hidden"], out_keys=["action"]),
                )
                value = TensorDictSequential(
                    common,
                    TensorDictModule(value, in_keys=["hidden"], out_keys=["value"]),
                )
                super().__init__()
                self.convert_to_functional(
                    actor,
                    "actor",
                    expand_dim=None,
                    create_target_params=False,
                    compare_against=None,
                )
                self.convert_to_functional(
                    value,
                    "value",
                    expand_dim=2,
                    create_target_params=create_target_params,
                    compare_against=actor.parameters(),
                )

        mod = DummyModule()
        v_p1 = set(mod.value_params.values(True, True)).union(
            set(mod.actor_params.values(True, True))
        )
        v_params1 = set(mod.parameters())
        v_buffers1 = set(mod.buffers())
        mod.to(dest)
        v_p2 = set(mod.value_params.values(True, True)).union(
            set(mod.actor_params.values(True, True))
        )
        v_params2 = set(mod.parameters())
        v_buffers2 = set(mod.buffers())
        assert v_p1 == v_p2
        assert v_params1 == v_params2
        assert v_buffers1 == v_buffers2
        for k, p in mod.named_parameters():
            assert isinstance(p, nn.Parameter), k
        for k, p in mod.named_buffers():
            assert isinstance(p, Buffer), k
        for p in mod.actor_params.values(True, True):
            assert isinstance(p, (nn.Parameter, Buffer))
        for p in mod.value_params.values(True, True):
            assert isinstance(p, (nn.Parameter, Buffer))
        if isinstance(dest, torch.dtype):
            for p in mod.parameters():
                assert p.dtype == dest
            for p in mod.buffers():
                assert p.dtype == dest
            for p in mod.actor_params.values(True, True):
                assert p.dtype == dest
            for p in mod.value_params.values(True, True):
                assert p.dtype == dest
        else:
            for p in mod.parameters():
                assert p.device == dest
            for p in mod.buffers():
                assert p.device == dest
            for p in mod.actor_params.values(True, True):
                assert p.device == dest
            for p in mod.value_params.values(True, True):
                assert p.device == dest


@set_composite_lp_aggregate(False)
def test_loss_exploration():
    class DummyLoss(LossModule):
        def forward(self, td, mode):
            if mode is None:
                mode = self.deterministic_sampling_mode
            assert exploration_type() == mode
            with set_exploration_type(ExplorationType.RANDOM):
                assert exploration_type() == ExplorationType.RANDOM
            assert exploration_type() == mode
            return td

    loss_fn = DummyLoss()
    with set_exploration_type(ExplorationType.RANDOM):
        assert exploration_type() == ExplorationType.RANDOM
        loss_fn(None, None)
        assert exploration_type() == ExplorationType.RANDOM

    with set_exploration_type(ExplorationType.RANDOM):
        assert exploration_type() == ExplorationType.RANDOM
        loss_fn(None, ExplorationType.DETERMINISTIC)
        assert exploration_type() == ExplorationType.RANDOM

    loss_fn.deterministic_sampling_mode = ExplorationType.MODE
    with set_exploration_type(ExplorationType.RANDOM):
        assert exploration_type() == ExplorationType.RANDOM
        loss_fn(None, ExplorationType.MODE)
        assert exploration_type() == ExplorationType.RANDOM

    loss_fn.deterministic_sampling_mode = ExplorationType.MEAN
    with set_exploration_type(ExplorationType.RANDOM):
        assert exploration_type() == ExplorationType.RANDOM
        loss_fn(None, ExplorationType.MEAN)
        assert exploration_type() == ExplorationType.RANDOM


@pytest.mark.parametrize("device", get_default_devices())
class TestMakeValueEstimator:
    """Tests for make_value_estimator accepting ValueEstimatorBase instances and subclasses."""

    def _create_mock_value_net(self, obs_dim=4, device="cpu"):
        """Create a simple value network for testing."""
        return TensorDictModule(
            nn.Linear(obs_dim, 1),
            in_keys=["observation"],
            out_keys=["state_value"],
        ).to(device)

    def _create_mock_actor(self, obs_dim=4, action_dim=2, device="cpu"):
        """Create a simple actor network for testing."""
        return ProbabilisticActor(
            module=TensorDictModule(
                nn.Linear(obs_dim, 2 * action_dim),
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
            spec=Composite(action=Bounded(-1, 1, (action_dim,))),
        ).to(device)

    def _create_mock_qvalue(self, obs_dim=4, action_dim=2, device="cpu"):
        """Create a simple Q-value network for testing."""
        return TensorDictModule(
            nn.Linear(obs_dim + action_dim, 1),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        ).to(device)

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_instance(self, device):
        """Test that make_value_estimator accepts a ValueEstimatorBase instance."""
        value_net = self._create_mock_value_net(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        # Create a value estimator instance
        value_estimator = TD0Estimator(
            gamma=0.99,
            value_network=value_net,
        )

        # Create a loss module that supports value estimation
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )

        # Pass the instance to make_value_estimator
        result = loss_fn.make_value_estimator(value_estimator)

        # Verify the value estimator was set correctly
        assert loss_fn._value_estimator is value_estimator
        assert loss_fn.value_type is TD0Estimator
        # Verify chaining works
        assert result is loss_fn

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_class(self, device):
        """Test that make_value_estimator accepts a ValueEstimatorBase subclass."""
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        # Create a loss module
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )

        # Pass a class with hyperparameters
        result = loss_fn.make_value_estimator(
            TD0Estimator,
            gamma=0.95,
            value_network=None,  # SAC losses don't need a separate value network
        )

        # Verify the value estimator was instantiated correctly
        assert isinstance(loss_fn._value_estimator, TD0Estimator)
        assert loss_fn.value_type is TD0Estimator
        assert loss_fn._value_estimator.gamma == 0.95
        # Verify chaining works
        assert result is loss_fn

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_class_inherits_device(self, device):
        """Test that make_value_estimator with a class inherits device from loss module."""
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        # Create a loss module
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )

        # Pass a class without explicit device
        loss_fn.make_value_estimator(
            TD0Estimator,
            gamma=0.99,
            value_network=None,
        )

        # The value estimator should have inherited the device
        assert loss_fn._value_estimator.gamma.device == device

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_gae_class(self, device):
        """Test that make_value_estimator works with GAE class."""
        value_net = self._create_mock_value_net(device=device)
        actor = self._create_mock_actor(device=device)

        # Create a PPO loss which supports GAE
        loss_fn = PPOLoss(
            actor_network=actor,
            critic_network=value_net,
        )

        # Pass GAE class with hyperparameters
        loss_fn.make_value_estimator(
            GAE,
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
        )

        # Verify the value estimator was instantiated correctly
        assert isinstance(loss_fn._value_estimator, GAE)
        assert loss_fn.value_type is GAE

    @set_composite_lp_aggregate(False)
    def test_make_value_estimator_with_gae_instance(self, device):
        """Test that make_value_estimator works with GAE instance."""
        value_net = self._create_mock_value_net(device=device)
        actor = self._create_mock_actor(device=device)

        # Create a GAE instance
        gae = GAE(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
        )

        # Create a PPO loss
        loss_fn = PPOLoss(
            actor_network=actor,
            critic_network=value_net,
        )

        # Pass the GAE instance
        loss_fn.make_value_estimator(gae)

        # Verify it was set directly
        assert loss_fn._value_estimator is gae
        assert loss_fn.value_type is GAE


@pytest.mark.parametrize("device", get_default_devices())
class TestSchedulableBuffers:
    """Tests for the _schedulable_buffers / __setattr__ mechanism."""

    def _create_mock_actor(self, obs_dim=4, action_dim=2, device="cpu"):
        return ProbabilisticActor(
            module=TensorDictModule(
                nn.Linear(obs_dim, 2 * action_dim),
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
            spec=Composite(action=Bounded(-1, 1, (action_dim,))),
        ).to(device)

    def _create_mock_value_net(self, obs_dim=4, device="cpu"):
        return TensorDictModule(
            nn.Linear(obs_dim, 1),
            in_keys=["observation"],
            out_keys=["state_value"],
        ).to(device)

    def _create_mock_qvalue(self, obs_dim=4, action_dim=2, device="cpu"):
        return TensorDictModule(
            nn.Linear(obs_dim + action_dim, 1),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        ).to(device)

    @set_composite_lp_aggregate(False)
    def test_float_assignment(self, device):
        actor = self._create_mock_actor(device=device)
        critic = self._create_mock_value_net(device=device)
        loss = ClipPPOLoss(actor, critic)
        loss = loss.to(device)

        # Assign float to entropy_coeff
        loss.entropy_coeff = 0.003
        assert loss.entropy_coeff.item() == pytest.approx(0.003)
        # Buffer should still be in _buffers dict
        assert "entropy_coeff" in loss._buffers
        # Buffer should still be in state_dict
        assert "entropy_coeff" in loss.state_dict()

    @set_composite_lp_aggregate(False)
    def test_int_assignment(self, device):
        actor = self._create_mock_actor(device=device)
        critic = self._create_mock_value_net(device=device)
        loss = ClipPPOLoss(actor, critic)
        loss = loss.to(device)

        loss.entropy_coeff = 0
        assert loss.entropy_coeff.item() == 0.0
        assert "entropy_coeff" in loss._buffers

    @set_composite_lp_aggregate(False)
    def test_device_preservation(self, device):
        actor = self._create_mock_actor(device=device)
        critic = self._create_mock_value_net(device=device)
        loss = ClipPPOLoss(actor, critic)
        loss = loss.to(device)

        loss.entropy_coeff = 0.05
        assert loss.entropy_coeff.device == torch.device(device)

    @set_composite_lp_aggregate(False)
    def test_tensor_assignment_still_works(self, device):
        actor = self._create_mock_actor(device=device)
        critic = self._create_mock_value_net(device=device)
        loss = ClipPPOLoss(actor, critic)
        loss = loss.to(device)

        # Tensor assignment goes through normal nn.Module path
        loss.entropy_coeff = torch.tensor(0.05, device=device)
        assert loss.entropy_coeff.item() == pytest.approx(0.05)

    @set_composite_lp_aggregate(False)
    def test_inheritance_merging(self, device):
        # ClipPPOLoss should have both PPOLoss buffers and its own
        merged = ClipPPOLoss._all_schedulable_buffers
        assert "entropy_coeff" in merged  # from PPOLoss
        assert "critic_coeff" in merged  # from PPOLoss
        assert "clip_value" in merged  # from PPOLoss
        assert "clip_epsilon" in merged  # from ClipPPOLoss

        # Actually set clip_epsilon
        actor = self._create_mock_actor(device=device)
        critic = self._create_mock_value_net(device=device)
        loss = ClipPPOLoss(actor, critic)
        loss = loss.to(device)

        loss.clip_epsilon = 0.1
        assert loss.clip_epsilon.item() == pytest.approx(0.1)
        assert "clip_epsilon" in loss._buffers

    @set_composite_lp_aggregate(False)
    def test_sac_log_alpha_as_buffer(self, device):
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        loss = SACLoss(actor, qvalue, fixed_alpha=True)
        loss = loss.to(device)

        # log_alpha is a buffer when fixed_alpha=True
        assert "log_alpha" in loss._buffers
        loss.log_alpha = 0.5
        assert loss.log_alpha.item() == pytest.approx(0.5)

    @set_composite_lp_aggregate(False)
    def test_sac_log_alpha_as_parameter(self, device):
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        loss = SACLoss(actor, qvalue, fixed_alpha=False)
        loss = loss.to(device)

        # log_alpha is a Parameter when fixed_alpha=False
        assert isinstance(loss.log_alpha, nn.Parameter)
        # Float assignment should NOT be intercepted (not in _buffers)
        assert "log_alpha" not in loss._buffers

    @set_composite_lp_aggregate(False)
    def test_optional_none_buffer(self, device):
        actor = self._create_mock_actor(device=device)
        critic = self._create_mock_value_net(device=device)
        # clip_value defaults to None
        loss = ClipPPOLoss(actor, critic, clip_value=None)
        loss = loss.to(device)

        assert loss.clip_value is None
        # Assigning float when buffer is None should NOT raise
        # (it just sets a plain attribute via nn.Module)
        loss.clip_value = 0.5
        # Now it's a plain float, not a buffer
        assert loss.clip_value == 0.5

    @set_composite_lp_aggregate(False)
    def test_state_dict_roundtrip(self, device):
        actor = self._create_mock_actor(device=device)
        critic = self._create_mock_value_net(device=device)
        loss = ClipPPOLoss(actor, critic)
        loss = loss.to(device)

        loss.entropy_coeff = 0.042
        loss.clip_epsilon = 0.15
        sd = loss.state_dict()

        loss2 = ClipPPOLoss(actor, critic)
        loss2 = loss2.to(device)
        loss2.load_state_dict(sd)
        assert loss2.entropy_coeff.item() == pytest.approx(0.042)
        assert loss2.clip_epsilon.item() == pytest.approx(0.15)
