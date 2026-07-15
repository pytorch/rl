# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import warnings

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torch import nn

from torchrl.data import Binary, Categorical, Composite, MultiOneHot, OneHot
from torchrl.modules import MLP
from torchrl.modules.tensordict_module.actors import (
    _process_action_space_spec,
    DistributionalQValueActor,
    DistributionalQValueHook,
    DistributionalQValueModule,
    QValueActor,
    QValueHook,
    QValueModule,
)

from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import NestedCountingEnv


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
        if nested_action:
            leaf_action_spec = env.full_action_spec[env.action_keys[0]]
        else:
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

    def test_qvalue_actor_strict_shape_auto(self):
        """Test that strict_shape='auto' reshapes action to match spec (issue #3059)."""
        action_spec = Categorical(4, shape=torch.Size((1, 1)), dtype=torch.int64)
        module = TensorDictModule(
            module=nn.Linear(3, 1), in_keys=("observation",), out_keys=("action_value",)
        )
        qvalue_actor = QValueActor(
            module=module,
            in_keys=["observation"],
            spec=action_spec,
            strict_shape="auto",
        )
        td = TensorDict({"observation": torch.randn(12, 3)})
        qvalue_actor(td)
        assert td["action"].shape == torch.Size([12, 1])

    def test_qvalue_actor_strict_shape_true_raises(self):
        """Test that strict_shape=True raises on shape mismatch."""
        action_spec = Categorical(4, shape=torch.Size((1, 1)), dtype=torch.int64)
        module = TensorDictModule(
            module=nn.Linear(3, 1), in_keys=("observation",), out_keys=("action_value",)
        )
        qvalue_actor = QValueActor(
            module=module, in_keys=["observation"], spec=action_spec, strict_shape=True
        )
        td = TensorDict({"observation": torch.randn(12, 3)})
        with pytest.raises(RuntimeError, match="does not match expected shape"):
            qvalue_actor(td)

    def test_qvalue_actor_strict_shape_none_warns(self):
        """Test that strict_shape=None (default) issues FutureWarning."""
        action_spec = Categorical(4, shape=torch.Size((1, 1)), dtype=torch.int64)
        module = TensorDictModule(
            module=nn.Linear(3, 1), in_keys=("observation",), out_keys=("action_value",)
        )
        qvalue_actor = QValueActor(
            module=module, in_keys=["observation"], spec=action_spec
        )
        td = TensorDict({"observation": torch.randn(12, 3)})
        with pytest.warns(FutureWarning, match="does not match expected shape"):
            qvalue_actor(td)

    def test_qvalue_actor_strict_shape_normal_no_warning(self):
        """Test that matching shapes produce no warning even with strict_shape='auto'."""
        action_spec = OneHot(4)
        module = TensorDictModule(
            module=nn.Linear(3, 4), in_keys=("observation",), out_keys=("action_value",)
        )
        qvalue_actor = QValueActor(
            module=module,
            in_keys=["observation"],
            spec=action_spec,
            strict_shape="auto",
        )
        td = TensorDict({"observation": torch.randn(5, 3)})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            qvalue_actor(td)
            future_warns = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(future_warns) == 0
        assert td["action"].shape == torch.Size([5, 4])


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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
