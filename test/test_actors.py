# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch

from _utils_internal import get_default_devices
from mocking_classes import NestedCountingEnv
from tensordict import TensorDict
from tensordict.nn import CompositeDistribution, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torch import distributions as dist, nn
from torchrl.data import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
)
from torchrl.data.rlhf.dataset import _has_transformers
from torchrl.modules import MLP, SafeModule, TanhDelta, TanhNormal
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


@pytest.mark.parametrize(
    "log_prob_key",
    [
        None,
        "sample_log_prob",
        ("nested", "sample_log_prob"),
        ("data", "sample_log_prob"),
    ],
)
def test_probabilistic_actor_nested_delta(log_prob_key, nested_dim=5, n_actions=3):
    env = NestedCountingEnv(nested_dim=nested_dim)
    action_spec = BoundedTensorSpec(
        shape=torch.Size((nested_dim, n_actions)), high=1, low=-1
    )
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
            "min": action_spec.space.minimum,
            "max": action_spec.space.maximum,
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
            "min": action_spec.space.minimum,
            "max": action_spec.space.maximum,
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
    action_spec = BoundedTensorSpec(
        shape=torch.Size((nested_dim, n_actions)), high=1, low=-1
    )
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
            "min": action_spec.space.minimum,
            "max": action_spec.space.maximum,
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
            "min": action_spec.space.minimum,
            "max": action_spec.space.maximum,
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
        spec = OneHotDiscreteTensorSpec(3)
        _process_action_space_spec("one-hot", spec)
        _process_action_space_spec("one_hot", spec)
        _process_action_space_spec("one_hot", None)
        _process_action_space_spec(None, spec)
        with pytest.raises(
            ValueError, match="The action spec and the action space do not match"
        ):
            _process_action_space_spec("multi-one-hot", spec)
        spec = MultiOneHotDiscreteTensorSpec([3, 3])
        _process_action_space_spec("multi-one-hot", spec)
        _process_action_space_spec(spec, spec)
        with pytest.raises(
            ValueError, match="Passing an action_space as a TensorSpec and a spec"
        ):
            _process_action_space_spec(OneHotDiscreteTensorSpec(3), spec)
        with pytest.raises(
            ValueError, match="action_space cannot be of type CompositeSpec"
        ):
            _process_action_space_spec(CompositeSpec(), spec)
        with pytest.raises(KeyError, match="action could not be found in the spec"):
            _process_action_space_spec(None, CompositeSpec())
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
            _process_action_space_spec(BinaryDiscreteTensorSpec(n=1), action_spec)
            _process_action_space_spec(BinaryDiscreteTensorSpec(n=1), leaf_action_spec)
        with pytest.raises(
            ValueError, match="action_space cannot be of type CompositeSpec"
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
    action_spec = OneHotDiscreteTensorSpec(action_dim)

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


@pytest.mark.parametrize(
    "spec", [None, OneHotDiscreteTensorSpec(3), MultiOneHotDiscreteTensorSpec([3, 2])]
)
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
        type(spec) is MultiOneHotDiscreteTensorSpec
        and action_space not in ("mult-one-hot", "mult_one_hot", None)
    ) or (
        type(spec) is OneHotDiscreteTensorSpec
        and action_space not in ("one-hot", "one_hot", None)
    ):
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
    action_spec = DiscreteTensorSpec(action_dim)

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


def test_compound_actor():
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
    actor = ProbabilisticActor(
        module,
        in_keys=["params"],
        distribution_class=CompositeDistribution,
        distribution_kwargs={
            "distribution_map": {"normal": dist.Normal, "categ": dist.Categorical}
        },
    )
    data = TensorDict({"x": torch.rand(10)}, [])
    actor(data)
    assert set(data.keys(True, True)) == {
        "categ",
        "normal",
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
