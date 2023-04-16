# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch

from _utils_internal import get_available_devices
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import DiscreteTensorSpec, OneHotDiscreteTensorSpec
from torchrl.modules import MLP, SafeModule
from torchrl.modules.tensordict_module.actors import (
    ActorValueOperator,
    DistributionalQValueActor,
    DistributionalQValueHook,
    DistributionalQValueModule,
    ProbabilisticActor,
    QValueActor,
    QValueHook,
    QValueModule,
    ValueOperator,
)


class TestQValue:
    def test_qvalue_hook_wrong_action_space(self):
        with pytest.raises(ValueError) as exc:
            QValueHook(action_space="wrong_value")
        assert "action_space must be one of" in str(exc.value)

    def test_distributional_qvalue_hook_wrong_action_space(self):
        with pytest.raises(ValueError) as exc:
            DistributionalQValueHook(action_space="wrong_value", support=None)
        assert "action_space must be one of" in str(exc.value)

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


@pytest.mark.parametrize("device", get_available_devices())
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


@pytest.mark.parametrize("device", get_available_devices())
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


@pytest.mark.parametrize("device", get_available_devices())
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
