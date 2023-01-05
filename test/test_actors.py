# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchrl.modules.tensordict_module.actors import (
    DistributionalQValueHook,
    QValueHook,
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
