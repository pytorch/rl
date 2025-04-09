# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import os

import pytest
import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    NonTensorStack,
    set_list_to_stack,
    TensorDict,
)
from tensordict.nn import CompositeDistribution, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torch import distributions as dist, nn

from torchrl.collectors import SyncDataCollector
from torchrl.data import Binary, Bounded, Categorical, Composite, MultiOneHot, OneHot
from torchrl.data.llm import LLMData
from torchrl.data.llm.dataset import _has_transformers
from torchrl.envs import LLMEnv
from torchrl.modules import (
    MLP,
    SafeModule,
    TanhDelta,
    TanhNormal,
    TransformersWrapper,
    vLLMWrapper,
)
from torchrl.modules.tensordict_module.actors import (
    _process_action_space_spec,
    ActorValueOperator,
    DistributionalQValueActor,
    DistributionalQValueHook,
    DistributionalQValueModule,
    LMHeadActorValueOperator,
    ProbabilisticActor,
    QValueActor,
    QValueHook,
    QValueModule,
    ValueOperator,
)

if os.getenv("PYTORCH_TEST_FBCODE"):
    from pytorch.rl.test._utils_internal import get_default_devices
    from pytorch.rl.test.mocking_classes import DummyStrDataLoader, NestedCountingEnv
else:
    from _utils_internal import get_default_devices
    from mocking_classes import DummyStrDataLoader, NestedCountingEnv

_has_vllm = importlib.util.find_spec("vllm") is not None


@pytest.mark.parametrize(
    "log_prob_key",
    [
        None,
        "sample_log_prob",
        ("nested", "sample_log_prob"),
        ("data", "sample_log_prob"),
    ],
)
def test_probabilistic_actor_nested_delta(log_prob_key, nested_dim=5, n_actions=1):
    env = NestedCountingEnv(nested_dim=nested_dim)
    action_spec = Bounded(shape=torch.Size((nested_dim, n_actions)), high=1, low=-1)
    policy_module = TensorDictModule(
        nn.Linear(1, 1), in_keys=[("data", "states")], out_keys=[("data", "param")]
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=[("data", "param")],
        out_keys=[("data", "action")],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        log_prob_key=log_prob_key,
        return_log_prob=True,
    )

    td = env.reset()
    td["data", "states"] = td["data", "states"].to(torch.float)
    td_out = policy(td)
    assert td_out["data", "action"].shape == (5, 1)
    if log_prob_key:
        assert td_out[log_prob_key].shape == (5,)
    else:
        assert td_out["sample_log_prob"].shape == (5,)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys={"param": ("data", "param")},
        out_keys=[("data", "action")],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        log_prob_key=log_prob_key,
        return_log_prob=True,
    )
    td_out = policy(td)
    assert td_out["data", "action"].shape == (5, 1)
    if log_prob_key:
        assert td_out[log_prob_key].shape == (5,)
    else:
        assert td_out["sample_log_prob"].shape == (5,)


@pytest.mark.parametrize(
    "log_prob_key",
    [
        None,
        "sample_log_prob",
        ("nested", "sample_log_prob"),
        ("data", "sample_log_prob"),
    ],
)
def test_probabilistic_actor_nested_normal(log_prob_key, nested_dim=5, n_actions=3):
    env = NestedCountingEnv(nested_dim=nested_dim)
    action_spec = Bounded(shape=torch.Size((nested_dim, n_actions)), high=1, low=-1)
    actor_net = nn.Sequential(
        nn.Linear(1, 2),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("data", "states")],
        out_keys=[("data", "loc"), ("data", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=[("data", "loc"), ("data", "scale")],
        out_keys=[("data", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        log_prob_key=log_prob_key,
        return_log_prob=True,
    )

    td = env.reset()
    td["data", "states"] = td["data", "states"].to(torch.float)
    td_out = policy(td)
    assert td_out["data", "action"].shape == (5, 1)
    if log_prob_key:
        assert td_out[log_prob_key].shape == (5,)
    else:
        assert td_out["sample_log_prob"].shape == (5,)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys={"loc": ("data", "loc"), "scale": ("data", "scale")},
        out_keys=[("data", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        log_prob_key=log_prob_key,
        return_log_prob=True,
    )
    td_out = policy(td)
    assert td_out["data", "action"].shape == (5, 1)
    if log_prob_key:
        assert td_out[log_prob_key].shape == (5,)
    else:
        assert td_out["sample_log_prob"].shape == (5,)


class TestQValue:
    def test_qvalue_hook_wrong_action_space(self):
        with pytest.raises(
            ValueError, match="action_space was not specified/not compatible"
        ):
            QValueHook(action_space="wrong_value")

    def test_distributional_qvalue_hook_wrong_action_space(self):
        with pytest.raises(
            ValueError, match="action_space was not specified/not compatible"
        ):
            DistributionalQValueHook(action_space="wrong_value", support=None)

    def test_distributional_qvalue_hook_conflicting_spec(self):
        spec = OneHot(3)
        _process_action_space_spec("one-hot", spec)
        _process_action_space_spec("one_hot", spec)
        _process_action_space_spec("one_hot", None)
        _process_action_space_spec(None, spec)
        with pytest.raises(
            ValueError, match="The action spec and the action space do not match"
        ):
            _process_action_space_spec("multi-one-hot", spec)
        spec = MultiOneHot([3, 3])
        _process_action_space_spec("multi-one-hot", spec)
        _process_action_space_spec(spec, spec)
        with pytest.raises(
            ValueError, match="Passing an action_space as a TensorSpec and a spec"
        ):
            _process_action_space_spec(OneHot(3), spec)
        with pytest.raises(
            ValueError, match="action_space cannot be of type Composite"
        ):
            _process_action_space_spec(Composite(), spec)
        with pytest.raises(KeyError, match="action could not be found in the spec"):
            _process_action_space_spec(None, Composite())
        with pytest.raises(
            ValueError, match="Neither action_space nor spec was defined"
        ):
            _process_action_space_spec(None, None)

    @pytest.mark.parametrize("nested_action", [True, False])
    @pytest.mark.parametrize("batch_size", [(), (32,), (32, 1)])
    def test_nested_keys(self, nested_action, batch_size, nested_dim=5):
        # _process_action_space_spec can take
        # an action_space argument (which can be string or non-composite spec)
        # and a action_spec, which can be a spec
        env = NestedCountingEnv(
            nest_obs_action=nested_action, batch_size=batch_size, nested_dim=nested_dim
        )
        action_spec = env._input_spec["full_action_spec"]
        leaf_action_spec = env.action_spec

        space_str, spec = _process_action_space_spec(None, action_spec)
        assert spec == action_spec
        assert space_str == "binary"

        space_str, spec = _process_action_space_spec(None, leaf_action_spec)
        assert spec == leaf_action_spec
        assert space_str == "binary"

        space_str, spec = _process_action_space_spec(leaf_action_spec, None)
        assert spec == leaf_action_spec
        assert space_str == "binary"

        space_str, spec = _process_action_space_spec(leaf_action_spec, action_spec)
        assert spec == action_spec  # Spec wins
        assert space_str == "binary"

        space_str, spec = _process_action_space_spec("binary", action_spec)
        assert spec == action_spec
        assert space_str == "binary"

        space_str, spec = _process_action_space_spec("binary", leaf_action_spec)
        assert spec == leaf_action_spec
        assert space_str == "binary"

        with pytest.raises(
            ValueError,
            match="Passing an action_space as a TensorSpec and a spec isn't allowed, unless they match.",
        ):
            _process_action_space_spec(Binary(n=1), action_spec)
            _process_action_space_spec(Binary(n=1), leaf_action_spec)
        with pytest.raises(
            ValueError, match="action_space cannot be of type Composite"
        ):
            _process_action_space_spec(action_spec, None)

        mod = QValueModule(
            action_value_key=("data", "action_value"),
            out_keys=[
                env.action_key,
                ("data", "action_value"),
                ("data", "chosen_action_value"),
            ],
            action_space=None,
            spec=action_spec,
        )

    @pytest.mark.parametrize(
        "action_space, var_nums, expected_action",
        (
            ("multi_one_hot", [2, 2, 2], [1, 0, 1, 0, 1, 0]),
            ("multi_one_hot", [2, 4], [1, 0, 1, 0, 0, 0]),
        ),
    )
    def test_qvalue_module_multi_one_hot(self, action_space, var_nums, expected_action):
        module = QValueModule(action_space=action_space, var_nums=var_nums)
        in_values = torch.tensor([1.0, 0, 2, 0, 1, 0])
        action, values, chosen_action_value = module(in_values)
        assert (torch.tensor(expected_action, dtype=torch.long) == action).all()
        assert (values == in_values).all()

    @pytest.mark.parametrize(
        "action_space, expected_action",
        (
            ("one_hot", [0, 0, 1, 0, 0]),
            ("categorical", 2),
        ),
    )
    @pytest.mark.parametrize("key", ["somekey", None])
    def test_qvalue_module_0_dim_batch(self, action_space, expected_action, key):
        if key is not None:
            module = QValueModule(action_space=action_space, action_value_key=key)
        else:
            module = QValueModule(action_space=action_space)
            key = "action_value"

        in_values = torch.tensor([1.0, -1.0, 100.0, -2.0, -3.0])
        # test tensor
        action, values, chosen_action_value = module(in_values)
        assert (torch.tensor(expected_action, dtype=torch.long) == action).all()
        assert (values == in_values).all()
        assert (torch.tensor([100.0]) == chosen_action_value).all()

        # test tensor, keyword
        action, values, chosen_action_value = module(**{key: in_values})
        assert (torch.tensor(expected_action, dtype=torch.long) == action).all()
        assert (values == in_values).all()
        assert (torch.tensor([100.0]) == chosen_action_value).all()

        # test tensor, tensordict
        td = module(TensorDict({key: in_values}, []))
        action = td["action"]
        values = td[key]
        if key != "action_value_keys":
            assert "action_value_keys" not in td.keys()
        chosen_action_value = td["chosen_action_value"]
        assert (torch.tensor(expected_action, dtype=torch.long) == action).all()
        assert (values == in_values).all()
        assert (torch.tensor([100.0]) == chosen_action_value).all()

    @pytest.mark.parametrize(
        "action_space, expected_action",
        (
            ("one_hot", [0, 0, 1, 0, 0]),
            ("categorical", 2),
        ),
    )
    @pytest.mark.parametrize("model_type", ["td", "nn"])
    @pytest.mark.parametrize("key", ["somekey", None])
    def test_qvalue_actor_0_dim_batch(
        self, action_space, expected_action, key, model_type
    ):
        if model_type == "nn":
            model = nn.Identity()
        else:
            out_keys = ["action_value"] if key is None else [key]
            model = TensorDictModule(
                nn.Identity(),
                in_keys=["observation"],
                out_keys=out_keys,
            )
        if key is not None:
            module = QValueActor(model, action_space=action_space, action_value_key=key)
        else:
            module = QValueActor(model, action_space=action_space)
            key = "action_value"

        in_values = torch.tensor([1.0, -1.0, 100.0, -2.0, -3.0])
        # test tensor
        action, values, chosen_action_value = module(in_values)
        assert (torch.tensor(expected_action, dtype=torch.long) == action).all()
        assert (values == in_values).all()
        assert (torch.tensor([100.0]) == chosen_action_value).all()

        # test tensor, keyword
        action, values, chosen_action_value = module(**{"observation": in_values})
        assert (torch.tensor(expected_action, dtype=torch.long) == action).all()
        assert (values == in_values).all()
        assert (torch.tensor([100.0]) == chosen_action_value).all()

        # test tensor, tensordict
        td = module(TensorDict({"observation": in_values}, []))
        action = td["action"]
        values = td[key]
        if key != "action_value_keys":
            assert "action_value_keys" not in td.keys()
        chosen_action_value = td["chosen_action_value"]
        assert (torch.tensor(expected_action, dtype=torch.long) == action).all()
        assert (values == in_values).all()
        assert (torch.tensor([100.0]) == chosen_action_value).all()

    @pytest.mark.parametrize(
        "action_space, expected_action",
        (
            ("one_hot", [0, 0, 1, 0, 0]),
            ("categorical", 2),
        ),
    )
    def test_qvalue_hook_0_dim_batch(self, action_space, expected_action):
        hook = QValueHook(action_space=action_space)

        in_values = torch.tensor([1.0, -1.0, 100.0, -2.0, -3.0])
        action, values, chosen_action_value = hook(
            net=None, observation=None, values=in_values
        )

        assert (torch.tensor(expected_action, dtype=torch.long) == action).all()
        assert (values == in_values).all()
        assert (torch.tensor([100.0]) == chosen_action_value).all()

    @pytest.mark.parametrize(
        "action_space, expected_action",
        (
            ("one_hot", [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]),
            ("categorical", [2, 0]),
        ),
    )
    def test_qvalue_hook_1_dim_batch(self, action_space, expected_action):
        hook = QValueHook(action_space=action_space)

        in_values = torch.tensor(
            [
                [1.0, -1.0, 100.0, -2.0, -3.0],
                [5.0, 4.0, 3.0, 2.0, -5.0],
            ]
        )
        action, values, chosen_action_value = hook(
            net=None, observation=None, values=in_values
        )

        assert (torch.tensor(expected_action, dtype=torch.long) == action).all()
        assert (values == in_values).all()
        assert (torch.tensor([[100.0], [5.0]]) == chosen_action_value).all()

    @pytest.mark.parametrize(
        "action_space, expected_action",
        (
            ("one_hot", [0, 0, 1, 0, 0]),
            ("categorical", 2),
        ),
    )
    @pytest.mark.parametrize("key", ["somekey", None])
    def test_distributional_qvalue_module_0_dim_batch(
        self, action_space, expected_action, key
    ):
        support = torch.tensor([-2.0, 0.0, 2.0])
        if key is not None:
            module = DistributionalQValueModule(
                action_space=action_space, support=support, action_value_key=key
            )
        else:
            key = "action_value"
            module = DistributionalQValueModule(
                action_space=action_space, support=support
            )

        in_values = torch.nn.LogSoftmax(dim=-1)(
            torch.tensor(
                [
                    [1.0, -1.0, 11.0, -2.0, 30.0],
                    [1.0, -1.0, 1.0, -2.0, -3.0],
                    [1.0, -1.0, 10.0, -2.0, -3.0],
                ]
            )
        )
        # tensor
        action, values = module(in_values)
        expected_action = torch.tensor(expected_action, dtype=torch.long)

        assert action.shape == expected_action.shape
        assert (action == expected_action).all()
        assert values.shape == in_values.shape
        assert (values == in_values).all()

        # tensor, keyword
        action, values = module(**{key: in_values})
        expected_action = torch.tensor(expected_action, dtype=torch.long)

        assert action.shape == expected_action.shape
        assert (action == expected_action).all()
        assert values.shape == in_values.shape
        assert (values == in_values).all()

        # tensor, tensordict
        td = module(TensorDict({key: in_values}, []))
        action = td["action"]
        values = td[key]
        if key != "action_value":
            assert "action_value" not in td.keys()
        expected_action = torch.tensor(expected_action, dtype=torch.long)

        assert action.shape == expected_action.shape
        assert (action == expected_action).all()
        assert values.shape == in_values.shape
        assert (values == in_values).all()

    @pytest.mark.parametrize(
        "action_space, expected_action",
        (
            ("one_hot", [0, 0, 1, 0, 0]),
            ("categorical", 2),
        ),
    )
    @pytest.mark.parametrize("model_type", ["td", "nn"])
    @pytest.mark.parametrize("key", ["somekey", None])
    def test_distributional_qvalue_actor_0_dim_batch(
        self, action_space, expected_action, key, model_type
    ):
        support = torch.tensor([-2.0, 0.0, 2.0])
        if model_type == "nn":
            model = nn.Identity()
        else:
            if key is not None:
                model = TensorDictModule(
                    nn.Identity(), in_keys=["observation"], out_keys=[key]
                )
            else:
                model = TensorDictModule(
                    nn.Identity(), in_keys=["observation"], out_keys=["action_value"]
                )

        if key is not None:
            module = DistributionalQValueActor(
                model, action_space=action_space, support=support, action_value_key=key
            )
        else:
            key = "action_value"
            module = DistributionalQValueActor(
                model, action_space=action_space, support=support
            )

        in_values = torch.nn.LogSoftmax(dim=-1)(
            torch.tensor(
                [
                    [1.0, -1.0, 11.0, -2.0, 30.0],
                    [1.0, -1.0, 1.0, -2.0, -3.0],
                    [1.0, -1.0, 10.0, -2.0, -3.0],
                ]
            )
        )
        # tensor
        action, values = module(in_values)
        expected_action = torch.tensor(expected_action, dtype=torch.long)

        assert action.shape == expected_action.shape
        assert (action == expected_action).all()
        assert values.shape == in_values.shape
        assert (values == in_values.log_softmax(-2)).all()

        # tensor, keyword
        action, values = module(observation=in_values)
        expected_action = torch.tensor(expected_action, dtype=torch.long)

        assert action.shape == expected_action.shape
        assert (action == expected_action).all()
        assert values.shape == in_values.shape
        assert (values == in_values.log_softmax(-2)).all()

        # tensor, tensordict
        td = module(TensorDict({"observation": in_values}, []))
        action = td["action"]
        values = td[key]
        expected_action = torch.tensor(expected_action, dtype=torch.long)

        assert action.shape == expected_action.shape
        assert (action == expected_action).all()
        assert values.shape == in_values.shape
        assert (values == in_values.log_softmax(-2)).all()

    @pytest.mark.parametrize(
        "action_space, expected_action",
        (
            ("one_hot", [0, 0, 1, 0, 0]),
            ("categorical", 2),
        ),
    )
    def test_distributional_qvalue_hook_0_dim_batch(
        self, action_space, expected_action
    ):
        support = torch.tensor([-2.0, 0.0, 2.0])
        hook = DistributionalQValueHook(action_space=action_space, support=support)

        in_values = torch.nn.LogSoftmax(dim=-1)(
            torch.tensor(
                [
                    [1.0, -1.0, 11.0, -2.0, 30.0],
                    [1.0, -1.0, 1.0, -2.0, -3.0],
                    [1.0, -1.0, 10.0, -2.0, -3.0],
                ]
            )
        )
        action, values = hook(net=None, observation=None, values=in_values)
        expected_action = torch.tensor(expected_action, dtype=torch.long)

        assert action.shape == expected_action.shape
        assert (action == expected_action).all()
        assert values.shape == in_values.shape
        assert (values == in_values).all()

    @pytest.mark.parametrize(
        "action_space, expected_action",
        (
            ("one_hot", [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]),
            ("categorical", [2, 0]),
        ),
    )
    def test_qvalue_hook_categorical_1_dim_batch(self, action_space, expected_action):
        support = torch.tensor([-2.0, 0.0, 2.0])
        hook = DistributionalQValueHook(action_space=action_space, support=support)

        in_values = torch.nn.LogSoftmax(dim=-1)(
            torch.tensor(
                [
                    [
                        [1.0, -1.0, 11.0, -2.0, 30.0],
                        [1.0, -1.0, 1.0, -2.0, -3.0],
                        [1.0, -1.0, 10.0, -2.0, -3.0],
                    ],
                    [
                        [11.0, -1.0, 7.0, -1.0, 20.0],
                        [10.0, 19.0, 1.0, -2.0, -3.0],
                        [1.0, -1.0, 0.0, -2.0, -3.0],
                    ],
                ]
            )
        )
        action, values = hook(net=None, observation=None, values=in_values)
        expected_action = torch.tensor(expected_action, dtype=torch.long)

        assert action.shape == expected_action.shape
        assert (action == expected_action).all()
        assert values.shape == in_values.shape
        assert (values == in_values).all()

    @pytest.mark.parametrize("action_space", ["categorical", "one-hot"])
    @pytest.mark.parametrize("action_n", [2, 3, 4, 5])
    def test_qvalue_mask(self, action_space, action_n):
        torch.manual_seed(0)
        shape = (3, 4, 3, action_n)
        action_values = torch.randn(size=shape)
        td = TensorDict({"action_value": action_values}, [3])
        module = QValueModule(
            action_space=action_space,
            action_value_key="action_value",
            action_mask_key="action_mask",
        )
        with pytest.raises(KeyError, match="Action mask key "):
            module(td)

        action_mask = torch.randint(high=2, size=shape).to(torch.bool)
        while not action_mask.any(dim=-1).all() or action_mask.all():
            action_mask = torch.randint(high=2, size=shape).to(torch.bool)

        td.set("action_mask", action_mask)
        module(td)
        new_action_values = td.get("action_value")

        assert (new_action_values[~action_mask] != action_values[~action_mask]).all()
        assert (new_action_values[action_mask] == action_values[action_mask]).all()
        assert (td.get("chosen_action_value") > torch.finfo(torch.float).min).all()

        if action_space == "one-hot":
            assert (td.get("action")[action_mask]).any()
            assert not (td.get("action")[~action_mask]).any()
        else:
            assert action_mask.gather(-1, td.get("action").unsqueeze(-1)).all()


@pytest.mark.parametrize("device", get_default_devices())
def test_value_based_policy(device):
    torch.manual_seed(0)
    obs_dim = 4
    action_dim = 5
    action_spec = OneHot(action_dim)

    def make_net():
        net = MLP(in_features=obs_dim, out_features=action_dim, depth=2, device=device)
        for mod in net.modules():
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias.data.zero_()
        return net

    actor = QValueActor(spec=action_spec, module=make_net(), safe=True)
    obs = torch.zeros(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (action.sum(-1) == 1).all()

    actor = QValueActor(spec=action_spec, module=make_net(), safe=False)
    obs = torch.randn(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (action.sum(-1) == 1).all()

    actor = QValueActor(spec=action_spec, module=make_net(), safe=False)
    obs = torch.zeros(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    with pytest.raises(AssertionError):
        assert (action.sum(-1) == 1).all()


@pytest.mark.parametrize("spec", [None, OneHot(3), MultiOneHot([3, 2])])
@pytest.mark.parametrize(
    "action_space", [None, "one-hot", "one_hot", "mult-one-hot", "mult_one_hot"]
)
def test_qvalactor_construct(
    spec,
    action_space,
):
    kwargs = {}
    if spec is not None:
        kwargs["spec"] = spec
    if action_space is not None:
        kwargs["action_space"] = action_space
    kwargs["module"] = TensorDictModule(
        lambda x: x, in_keys=["x"], out_keys=["action_value"]
    )
    if spec is None and action_space is None:
        with pytest.raises(
            ValueError, match="Neither action_space nor spec was defined"
        ):
            QValueActor(**kwargs)
        return
    if (
        type(spec) is MultiOneHot
        and action_space not in ("mult-one-hot", "mult_one_hot", None)
    ) or (type(spec) is OneHot and action_space not in ("one-hot", "one_hot", None)):
        with pytest.raises(
            ValueError, match="The action spec and the action space do not match"
        ):
            QValueActor(**kwargs)
        return
    QValueActor(**kwargs)


@pytest.mark.parametrize("device", get_default_devices())
def test_value_based_policy_categorical(device):
    torch.manual_seed(0)
    obs_dim = 4
    action_dim = 5
    action_spec = Categorical(action_dim)

    def make_net():
        net = MLP(in_features=obs_dim, out_features=action_dim, depth=2, device=device)
        for mod in net.modules():
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias.data.zero_()
        return net

    actor = QValueActor(
        spec=action_spec, module=make_net(), safe=True, action_space="categorical"
    )
    obs = torch.zeros(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (0 <= action).all() and (action < action_dim).all()

    actor = QValueActor(
        spec=action_spec, module=make_net(), safe=False, action_space="categorical"
    )
    obs = torch.randn(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (0 <= action).all() and (action < action_dim).all()


@pytest.mark.parametrize("device", get_default_devices())
def test_actorcritic(device):
    common_module = SafeModule(
        module=nn.Linear(3, 4), in_keys=["obs"], out_keys=["hidden"], spec=None
    ).to(device)
    module = SafeModule(nn.Linear(4, 5), in_keys=["hidden"], out_keys=["param"])
    policy_operator = ProbabilisticActor(
        module=module, in_keys=["param"], spec=None, return_log_prob=True
    ).to(device)
    value_operator = ValueOperator(nn.Linear(4, 1), in_keys=["hidden"]).to(device)
    op = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_operator,
        value_operator=value_operator,
    ).to(device)
    td = TensorDict(
        source={"obs": torch.randn(4, 3)},
        batch_size=[
            4,
        ],
    ).to(device)
    td_total = op(td.clone())
    policy_op = op.get_policy_operator()
    td_policy = policy_op(td.clone())
    value_op = op.get_value_operator()
    td_value = value_op(td)
    torch.testing.assert_close(td_total.get("action"), td_policy.get("action"))
    torch.testing.assert_close(
        td_total.get("sample_log_prob"), td_policy.get("sample_log_prob")
    )
    torch.testing.assert_close(td_total.get("state_value"), td_value.get("state_value"))

    value_params = set(
        list(op.get_value_operator().parameters()) + list(op.module[0].parameters())
    )
    value_params2 = set(value_op.parameters())
    assert len(value_params.difference(value_params2)) == 0 and len(
        value_params.intersection(value_params2)
    ) == len(value_params)

    policy_params = set(
        list(op.get_policy_operator().parameters()) + list(op.module[0].parameters())
    )
    policy_params2 = set(policy_op.parameters())
    assert len(policy_params.difference(policy_params2)) == 0 and len(
        policy_params.intersection(policy_params2)
    ) == len(policy_params)


@pytest.mark.parametrize("name_map", [True, False])
def test_compound_actor(name_map):
    class Module(nn.Module):
        def forward(self, x):
            return x[..., :3], x[..., 3:6], x[..., 6:]

    module = TensorDictModule(
        Module(),
        in_keys=["x"],
        out_keys=[
            ("params", "normal", "loc"),
            ("params", "normal", "scale"),
            ("params", "categ", "logits"),
        ],
    )
    distribution_kwargs = {
        "distribution_map": {"normal": dist.Normal, "categ": dist.Categorical}
    }
    if name_map:
        distribution_kwargs.update(
            {
                "name_map": {
                    "normal": ("action", "normal"),
                    "categ": ("action", "categ"),
                },
            }
        )
    actor = ProbabilisticActor(
        module,
        in_keys=["params"],
        distribution_class=CompositeDistribution,
        distribution_kwargs=distribution_kwargs,
    )
    if not name_map:
        assert actor.out_keys == module.out_keys + ["normal", "categ"]
    else:
        assert actor.out_keys == module.out_keys + [
            ("action", "normal"),
            ("action", "categ"),
        ]

    data = TensorDict({"x": torch.rand(10)}, [])
    actor(data)
    assert set(data.keys(True, True)) == {
        "categ" if not name_map else ("action", "categ"),
        "normal" if not name_map else ("action", "normal"),
        ("params", "categ", "logits"),
        ("params", "normal", "loc"),
        ("params", "normal", "scale"),
        "x",
    }


@pytest.mark.skipif(not _has_transformers, reason="missing dependencies")
@pytest.mark.parametrize("device", get_default_devices())
def test_lmhead_actorvalueoperator(device):
    from transformers import AutoModelForCausalLM, GPT2Config

    config = GPT2Config(return_dict=False)
    base_model = AutoModelForCausalLM.from_config(config).eval()
    aco = LMHeadActorValueOperator(base_model).to(device)

    # check common
    assert aco.module[0][0].module is base_model.transformer
    assert aco.module[0][1].in_keys == ["x"]
    assert aco.module[0][1].out_keys == ["x"]

    # check actor
    assert aco.module[1].in_keys == ["x"]
    assert aco.module[1].out_keys == ["logits", "action", "sample_log_prob"]
    assert aco.module[1][0].module is base_model.lm_head

    # check critic
    assert aco.module[2].in_keys == ["x"]
    assert aco.module[2].out_keys == ["state_value"]
    assert isinstance(aco.module[2].module, nn.Linear)
    assert aco.module[2].module.in_features == base_model.transformer.embed_dim
    assert aco.module[2].module.out_features == 1

    td = TensorDict(
        source={
            "input_ids": torch.randint(50257, (4, 3)),
            "attention_mask": torch.ones((4, 3)),
        },
        batch_size=[
            4,
        ],
        device=device,
    )
    td_total = aco(td.clone())
    policy_op = aco.get_policy_operator()
    td_policy = policy_op(td.clone())
    value_op = aco.get_value_operator()
    td_value = value_op(td)
    torch.testing.assert_close(td_total.get("action"), td_policy.get("action"))
    torch.testing.assert_close(
        td_total.get("sample_log_prob"), td_policy.get("sample_log_prob")
    )
    torch.testing.assert_close(td_total.get("state_value"), td_value.get("state_value"))

    value_params = set(
        list(aco.get_value_operator().parameters()) + list(aco.module[0].parameters())
    )
    value_params2 = set(value_op.parameters())
    assert len(value_params.difference(value_params2)) == 0 and len(
        value_params.intersection(value_params2)
    ) == len(value_params)

    policy_params = set(
        list(aco.get_policy_operator().parameters()) + list(aco.module[0].parameters())
    )
    policy_params2 = set(policy_op.parameters())
    assert len(policy_params.difference(policy_params2)) == 0 and len(
        policy_params.intersection(policy_params2)
    ) == len(policy_params)


@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
class TestLLMActor:
    @pytest.fixture(scope="module")
    def vllm_instance(self):
        try:
            import vllm
        except ImportError:
            pytest.skip(reason="missing vllm")

        llm_model = vllm.LLM("gpt2")
        tokenizer = llm_model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        return llm_model

    @pytest.fixture(scope="module")
    def transformers_instance(self):
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel(GPT2Config()).eval()
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # model = OPTModel(OPTConfig("facebook/opt-125m"))
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # model = OPTForCausalLM(OPTConfig())

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    @pytest.fixture(scope="module")
    def transformers_instance_pretrained(self):
        from transformers import AutoTokenizer, OPTForCausalLM

        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # model = GPT2LMHeadModel(GPT2Config())
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # model = OPTModel(OPTConfig("facebook/opt-125m"))
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        model = OPTForCausalLM.from_pretrained("facebook/opt-125m")

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    @pytest.mark.parametrize(
        "from_text, generate, return_log_probs, tokens, attention_mask",
        [
            (True, True, True, None, None),
            (True, True, False, None, None),
            (True, False, None, None, None),
            (
                False,
                True,
                True,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, True, True, torch.randint(1024, (1, 10)), None),
            (
                False,
                True,
                False,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, True, False, torch.randint(1024, (1, 10)), None),
        ],
    )
    def test_transformers_wrapper(
        self,
        from_text,
        generate,
        return_log_probs,
        tokens,
        attention_mask,
        transformers_instance,
    ):
        torch.manual_seed(0)

        model, tokenizer = transformers_instance

        m = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            from_text=from_text,
            generate=generate,
            return_log_probs=return_log_probs,
        )
        self._run_check(
            m,
            tokens,
            attention_mask,
            generate,
            return_log_probs,
            from_text,
            has_logits=True,
        )

    @pytest.mark.parametrize(
        "from_text, generate, return_log_probs, tokens, attention_mask",
        [
            (True, True, True, None, None),
            (True, True, False, None, None),
            (True, False, None, None, None),
            (
                False,
                True,
                True,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, True, True, torch.randint(1024, (1, 10)), None),
            (
                False,
                True,
                False,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, True, False, torch.randint(1024, (1, 10)), None),
        ],
    )
    def test_vllm_wrapper(
        self,
        from_text,
        generate,
        return_log_probs,
        tokens,
        attention_mask,
        vllm_instance,
    ):
        torch.manual_seed(0)

        model = vllm_instance
        m = vLLMWrapper(
            model,
            from_text=from_text,
            generate=generate,
            return_log_probs=return_log_probs,
        )
        self._run_check(
            m,
            tokens,
            attention_mask,
            generate,
            return_log_probs,
            from_text,
            has_logits=False,
        )

    def _make_data(
        self,
        m,
        tokens,
        attention_mask,
        generate,
        from_text,
        has_logits,
        batch_size=1,
        text_response=None,
        tokens_response=None,
    ):
        lp_kwargs = {}
        if from_text:
            if not generate:
                text_response = (
                    NonTensorStack(" and another text that follows")
                    if text_response is None
                    else text_response
                )
                if not isinstance(text_response, NonTensorStack):
                    if isinstance(text_response, list):
                        text_response = NonTensorStack(*text_response)
                    else:
                        text_response = NonTensorStack(text_response)
                lp_kwargs.update({"text_response": text_response})
            tdin = LLMData(
                text=NonTensorStack("a text"), **lp_kwargs, batch_size=batch_size
            )
        else:
            if not generate:
                if tokens_response is None:
                    shape_response = tokens.shape
                    shape_response = shape_response[:-1] + (shape_response[-1] * 2,)
                    tokens_response = torch.randint(1024, shape_response)
                lp_kwargs.update({"tokens_response": tokens_response})
            tdin = LLMData(
                tokens=tokens,
                attention_mask=attention_mask,
                **lp_kwargs,
                batch_size=batch_size,
            )
        return tdin

    def _run_check(
        self,
        m,
        tokens,
        attention_mask,
        generate,
        return_log_probs,
        from_text,
        has_logits,
    ):
        tdin = self._make_data(
            m, tokens, attention_mask, generate, from_text, has_logits
        )
        if from_text and generate:
            assert tdin.text_response is None
        elif from_text and not generate:
            assert tdin.text_response is not None

        tdin.copy()
        td = m(tdin)
        assert td is tdin
        assert isinstance(td, LLMData)
        if from_text and generate:
            assert td.text_response is not None

        # TODO: vLLM may produce an attention mask when hf does not - explore consistency!
        # if generate and (from_text or tdincopy.attention_mask is not None):
        #     assert td.attention_mask is not None, (generate, from_text, tdincopy.attention_mask is not None)
        #     if isinstance(td.attention_mask, torch.Tensor):
        #         assert td.attention_mask.shape == td.tokens.shape
        # else:
        #     assert td.attention_mask is None, (generate, from_text)

        if not generate:
            # logprobs are computed on text response of tokens_response
            assert td.text_response is not None or td.tokens_response is not None
            assert td.log_probs is not None
            if has_logits:
                assert td.logits is not None
        if generate:
            if return_log_probs:
                assert td.log_probs is not None
                assert td.log_probs.shape[-1] == td.tokens_response.shape[-1]
            else:
                assert td.log_probs is None

        # Test the shapes
        assert td.tokens_response is not None, (generate, has_logits, from_text)

        # If from text and not generating, the tokens are not returned for now
        if not (from_text and not generate):
            assert td.tokens_response is not None
            assert td.tokens is not None
            assert td.tokens_response.shape[:-1] == td.tokens.shape[:-1]
            # The convention is that the response only has new tokens
            assert (
                td.tokens_response[..., : td.tokens.shape[-1]]
                != td.tokens[..., : td.tokens_response.shape[-1]]
            ).any(), (generate, from_text)

    @pytest.mark.parametrize(
        "from_text, tokens, attention_mask",
        [
            (
                False,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (False, torch.randint(1024, (1, 10)), None),
            (True, None, None),
        ],
    )
    def test_transformers_logprobs(
        self, from_text, tokens, attention_mask, transformers_instance
    ):
        torch.manual_seed(0)
        model, tokenizer = transformers_instance

        m_generate = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            from_text=from_text,
            generate=True,
            return_log_probs=True,
        )
        m_logprobs = TransformersWrapper(
            model, tokenizer=tokenizer, from_text=from_text, generate=False
        )
        self._check_lps(
            m_generate, m_logprobs, tokens, attention_mask, from_text, has_logits=False
        )

    @pytest.mark.parametrize(
        "pad_output, from_text, tokens, attention_mask",
        [
            (True, True, None, None),
            (False, True, None, None),
            (
                True,
                False,
                torch.randint(1024, (1, 10)),
                torch.ones(1, 10, dtype=torch.int64),
            ),
            (True, False, torch.randint(1024, (1, 10)), None),
        ],
    )
    def test_vllm_logprobs(
        self, from_text, tokens, attention_mask, pad_output, vllm_instance
    ):
        torch.manual_seed(0)

        model = vllm_instance
        m_generate = vLLMWrapper(
            model,
            from_text=from_text,
            generate=True,
            return_log_probs=True,
            pad_output=pad_output,
        )
        m_logprobs = vLLMWrapper(
            model, from_text=from_text, generate=False, pad_output=pad_output
        )
        self._check_lps(
            m_generate,
            m_logprobs,
            tokens,
            attention_mask,
            from_text,
            has_logits=False,
            tol=1e-1,
        )

    def _check_lps(
        self,
        model_generate,
        model_logprobs,
        tokens,
        attention_mask,
        from_text,
        has_logits,
        tol=1e-2,
    ):
        # Checks that the log-probs gathered with generate=False equate those with generate=True
        tdin_genetate = self._make_data(
            model_generate, tokens, attention_mask, True, from_text, has_logits
        )
        td_generate = model_generate(tdin_genetate)
        tdin_logprobs = self._make_data(
            model_logprobs,
            tokens,
            attention_mask,
            False,
            from_text,
            has_logits,
            tokens_response=td_generate.tokens_response,
            text_response=td_generate.text_response,
        )
        td_logprobs = model_logprobs(tdin_logprobs)
        assert td_generate.log_probs.shape == td_generate.tokens_response.shape
        assert td_logprobs.log_probs.shape == td_logprobs.tokens_response.shape
        assert td_logprobs.log_probs.shape == td_generate.tokens_response.shape
        torch.testing.assert_close(
            td_generate.log_probs, td_logprobs.log_probs, rtol=tol, atol=tol
        )

    @pytest.mark.parametrize("pad", [True, False])
    @pytest.mark.parametrize("generate", [True, False])
    @pytest.mark.parametrize("use_tensorclass", [True, False])
    def test_vllm_batch_run(self, pad, generate, use_tensorclass, vllm_instance):
        # Test generate - padding combinations
        policy = vLLMWrapper(
            vllm_instance,
            from_text=True,
            generate=generate,
            return_log_probs=True,
            pad_output=pad,
            generate_kwargs={"max_tokens": 10000},
        )
        if generate:
            data = LazyStackedTensorDict(
                *TensorDict(
                    text=NonTensorStack("a string", "another very long string"),
                    batch_size=[2],
                ).unbind(0)
            )
        else:
            data = LazyStackedTensorDict(
                *TensorDict(
                    text=NonTensorStack("a string", "another very long string"),
                    text_response=NonTensorStack(
                        " is a string", " is still a very long string"
                    ),
                    batch_size=[2],
                ).unbind(0)
            )
        if use_tensorclass:
            data = LLMData.from_tensordict(data)
        output = policy(data)
        try:
            log_probs = output.get("log_probs")
        except Exception:
            log_probs = output.get("log_probs", as_list=True)
        if pad:
            assert isinstance(log_probs, torch.Tensor)
        else:
            assert isinstance(log_probs, list)
        text = output.get("text", as_list=True)
        # TODO: this is not ideal...
        if use_tensorclass:
            assert isinstance(text, list)
        else:
            assert isinstance(text, NonTensorStack)
        text_response = output.get("text_response", as_list=True)
        if use_tensorclass:
            assert isinstance(text_response, list)
        else:
            assert isinstance(text_response, NonTensorStack)
        try:
            tokens_response = output.get("tokens_response")
        except Exception:
            tokens_response = output.get("tokens_response", as_list=True)
        if pad:
            assert isinstance(tokens_response, torch.Tensor)
        else:
            assert isinstance(tokens_response, list)
        try:
            tokens = output.get("tokens")
        except Exception:
            tokens = output.get("tokens", as_list=True)
        if not generate:
            assert tokens is None
        elif pad:
            assert isinstance(tokens, torch.Tensor), tokens
        else:
            assert isinstance(tokens, list)

    @pytest.mark.parametrize("from_text", [True])
    def test_vllm_collection(self, vllm_instance, from_text):
        policy = vLLMWrapper(
            vllm_instance,
            return_log_probs=True,
            generate_kwargs={"max_tokens": 32},
            from_text=from_text in (True, None),
        )
        tokenizer = vllm_instance.get_tokenizer()
        self._run_check_collector(policy, from_text=from_text, tokenizer=tokenizer)

    def test_transformers_collection(self):
        ...

    @classmethod
    def env_constructor(cls, **kwargs):
        def make():
            # if kwargs.get("from_text", True):
            dl = DummyStrDataLoader(batch_size=32)
            # else:
            #     dl = DummyTensorDataLoader(batch_size=32)
            env = LLMEnv.from_dataloader(
                dl,
                batch_size=4,
                repeats=4,
                **kwargs,
            )
            assert env.batch_size == (16,)
            return env

        return make

    def _run_check_collector(self, policy, from_text, tokenizer):
        if from_text is None:
            kwargs = {"eos_token_id": tokenizer.eos_token_id}
        else:
            kwargs = {
                "from_text": from_text,
                "tokenizer": tokenizer,
                "eos_token_id": tokenizer.eos_token_id,
            }
        collector = SyncDataCollector(
            self.env_constructor(**kwargs),
            policy=policy,
            frames_per_batch=32,
            total_frames=128,
            use_buffers=False,
        )
        t = 0
        for data in collector:
            assert isinstance(data, LazyStackedTensorDict)
            assert isinstance(data.reshape(-1).get("text_response"), NonTensorStack)
            # action
            assert "text_response" in data
            assert "tokens_response" in data
            # obs
            assert "text" in data
            assert ("next", "text") in data
            # tokens
            assert "tokens" in data

            t += data.numel()
            assert collector._frames == t
            assert t < 512, t
            # assert ("next", "tokens") in data

    def test_vllm_generate_multiple_trajs(self, vllm_instance):
        policy = vLLMWrapper(
            vllm_instance,
            return_log_probs=True,
            generate_kwargs={"n": 10, "max_tokens": 1024},
            inplace=False,
        )
        data = TensorDict(
            text=NonTensorStack("a string", "another very long string"), batch_size=2
        )
        data = policy(data)

    @set_list_to_stack(True)
    @pytest.mark.parametrize("from_text", [True, False])
    @pytest.mark.parametrize("generate", [True, False])
    def test_transformers_long_sequences(
        self, from_text, generate, transformers_instance_pretrained
    ):
        torch.manual_seed(42)
        model, tokenizer = transformers_instance_pretrained
        prompts = [
            "The quick brown fox jumps over the lazy dog.",  # Likely to finish soon
            "Once upon a time in a land far, far away, there was a",  # Likely to continue longer
            "In the beginning, the universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
        ]
        data = lazy_stack([TensorDict() for _ in range(len(prompts))])
        data["text"] = prompts
        eos_token_id = tokenizer.convert_tokens_to_ids(",")
        if not from_text:
            data["tokens"] = tokenizer(data["text"])["input_ids"]
            data["attention_mask"] = (
                0 * data.get("tokens", as_nested_tensor=True, layout=torch.strided) + 1
            )
        if not generate:
            # we need responses
            responses = prompts[1:] + [" et dolore magna aliqua."]
            data["text_response"] = responses
            if not from_text:
                data["tokens_response"] = tokenizer(data["text_response"])["input_ids"]
        # make sure dimensions are ragged for tokens entries
        if "tokens" in data:
            assert data.get_item_shape("tokens")[-1] == -1
        if "tokens_response" in data:
            assert data.get_item_shape("tokens_response")[-1] == -1
        generate_kwargs = {}
        if generate:
            generate_kwargs = {
                "max_new_tokens": 128,  # Set a reasonable number of new tokens to generate
                "min_length": 20,  # Ensure a minimum length for the generated sequence
                "pad_token_id": tokenizer.pad_token_id,  # Use the tokenizer's pad token
                "forced_eos_token_id": eos_token_id,  # Use comma as an EOS token
            }
        policy = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            from_text=from_text,
            generate=generate,
            return_log_probs=True,
            # TODO: use n trajs
            generate_kwargs=generate_kwargs,
        )
        data_policy = policy(data)
        if "tokens" in data_policy:
            assert data_policy.get_item_shape("tokens")[-1] == -1
        if "tokens_response" in data_policy:
            assert (
                data_policy.get_item_shape("tokens_response")[-1] == -1
            )  # TODO: this fails


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
