# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for torch.compile compatibility of value estimation functions."""
from __future__ import annotations

import sys

import pytest
import torch

from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    td_lambda_return_estimate,
    vec_generalized_advantage_estimate,
    vec_td_lambda_return_estimate,
)

IS_WINDOWS = sys.platform == "win32"


@pytest.mark.skipif(IS_WINDOWS, reason="windows tests do not support compile")
class TestValueFunctionCompile:
    """Test compilation of value estimation functions."""

    @pytest.fixture
    def value_data(self):
        """Create test data for value functions."""
        batch_size = 32
        time_steps = 15
        feature_dim = 1

        return {
            "gamma": 0.99,
            "lmbda": 0.95,
            "state_value": torch.randn(batch_size, time_steps, feature_dim),
            "next_state_value": torch.randn(batch_size, time_steps, feature_dim),
            "reward": torch.randn(batch_size, time_steps, feature_dim),
            "done": torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool),
            "terminated": torch.zeros(
                batch_size, time_steps, feature_dim, dtype=torch.bool
            ),
        }

    def test_td_lambda_return_estimate_compiles_fullgraph(self, value_data):
        """Test that td_lambda_return_estimate (non-vectorized) compiles with fullgraph=True."""
        result_eager = td_lambda_return_estimate(
            gamma=value_data["gamma"],
            lmbda=value_data["lmbda"],
            next_state_value=value_data["next_state_value"],
            reward=value_data["reward"],
            done=value_data["done"],
            terminated=value_data["terminated"],
        )

        compiled_fn = torch.compile(
            td_lambda_return_estimate,
            fullgraph=True,
            backend="inductor",
        )

        result_compiled = compiled_fn(
            gamma=value_data["gamma"],
            lmbda=value_data["lmbda"],
            next_state_value=value_data["next_state_value"],
            reward=value_data["reward"],
            done=value_data["done"],
            terminated=value_data["terminated"],
        )

        torch.testing.assert_close(result_eager, result_compiled, rtol=1e-4, atol=1e-4)

    def test_generalized_advantage_estimate_compiles_fullgraph(self, value_data):
        """Test that generalized_advantage_estimate (non-vectorized) compiles with fullgraph=True."""
        advantage_eager, value_target_eager = generalized_advantage_estimate(
            gamma=value_data["gamma"],
            lmbda=value_data["lmbda"],
            state_value=value_data["state_value"],
            next_state_value=value_data["next_state_value"],
            reward=value_data["reward"],
            done=value_data["done"],
            terminated=value_data["terminated"],
        )

        compiled_fn = torch.compile(
            generalized_advantage_estimate,
            fullgraph=True,
            backend="inductor",
        )

        advantage_compiled, value_target_compiled = compiled_fn(
            gamma=value_data["gamma"],
            lmbda=value_data["lmbda"],
            state_value=value_data["state_value"],
            next_state_value=value_data["next_state_value"],
            reward=value_data["reward"],
            done=value_data["done"],
            terminated=value_data["terminated"],
        )

        torch.testing.assert_close(
            advantage_eager, advantage_compiled, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            value_target_eager, value_target_compiled, rtol=1e-4, atol=1e-4
        )

    def test_vec_td_lambda_return_estimate_fails_fullgraph(self, value_data):
        """Test that vec_td_lambda_return_estimate fails with fullgraph=True due to data-dependent shapes."""
        compiled_fn = torch.compile(
            vec_td_lambda_return_estimate,
            fullgraph=True,
            backend="inductor",
        )

        # This should fail because of data-dependent shapes in _get_num_per_traj
        with pytest.raises(Exception):
            compiled_fn(
                gamma=value_data["gamma"],
                lmbda=value_data["lmbda"],
                next_state_value=value_data["next_state_value"],
                reward=value_data["reward"],
                done=value_data["done"],
                terminated=value_data["terminated"],
            )

    def test_vec_generalized_advantage_estimate_fails_fullgraph(self, value_data):
        """Test that vec_generalized_advantage_estimate fails with fullgraph=True due to data-dependent shapes."""
        compiled_fn = torch.compile(
            vec_generalized_advantage_estimate,
            fullgraph=True,
            backend="inductor",
        )

        # This should fail because of data-dependent shapes in _get_num_per_traj
        with pytest.raises(Exception):
            compiled_fn(
                gamma=value_data["gamma"],
                lmbda=value_data["lmbda"],
                state_value=value_data["state_value"],
                next_state_value=value_data["next_state_value"],
                reward=value_data["reward"],
                done=value_data["done"],
                terminated=value_data["terminated"],
            )

    def test_td_lambda_with_tensor_gamma_compiles_fullgraph(self, value_data):
        """Test that td_lambda_return_estimate compiles with 0-d tensor gamma (fullgraph=True).

        This tests the fix for PendingUnbackedSymbolNotFound error that occurred when
        torch.full_like received a 0-d tensor and internally called .item().
        """
        # Use 0-d tensor gamma/lmbda - this was the problematic case
        gamma_tensor = torch.tensor(value_data["gamma"])
        lmbda_tensor = torch.tensor(value_data["lmbda"])

        result_eager = td_lambda_return_estimate(
            gamma=gamma_tensor,
            lmbda=lmbda_tensor,
            next_state_value=value_data["next_state_value"],
            reward=value_data["reward"],
            done=value_data["done"],
            terminated=value_data["terminated"],
        )

        compiled_fn = torch.compile(
            td_lambda_return_estimate,
            fullgraph=True,
            backend="inductor",
        )

        result_compiled = compiled_fn(
            gamma=gamma_tensor,
            lmbda=lmbda_tensor,
            next_state_value=value_data["next_state_value"],
            reward=value_data["reward"],
            done=value_data["done"],
            terminated=value_data["terminated"],
        )

        torch.testing.assert_close(result_eager, result_compiled, rtol=1e-4, atol=1e-4)

    def test_gae_with_tensor_gamma_compiles_fullgraph(self, value_data):
        """Test that generalized_advantage_estimate compiles with 0-d tensor gamma (fullgraph=True).

        This tests the fix for PendingUnbackedSymbolNotFound error that occurred when
        torch.full_like received a 0-d tensor and internally called .item().
        """
        # Use 0-d tensor gamma/lmbda - this was the problematic case
        gamma_tensor = torch.tensor(value_data["gamma"])
        lmbda_tensor = torch.tensor(value_data["lmbda"])

        advantage_eager, value_target_eager = generalized_advantage_estimate(
            gamma=gamma_tensor,
            lmbda=lmbda_tensor,
            state_value=value_data["state_value"],
            next_state_value=value_data["next_state_value"],
            reward=value_data["reward"],
            done=value_data["done"],
            terminated=value_data["terminated"],
        )

        compiled_fn = torch.compile(
            generalized_advantage_estimate,
            fullgraph=True,
            backend="inductor",
        )

        advantage_compiled, value_target_compiled = compiled_fn(
            gamma=gamma_tensor,
            lmbda=lmbda_tensor,
            state_value=value_data["state_value"],
            next_state_value=value_data["next_state_value"],
            reward=value_data["reward"],
            done=value_data["done"],
            terminated=value_data["terminated"],
        )

        torch.testing.assert_close(
            advantage_eager, advantage_compiled, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            value_target_eager, value_target_compiled, rtol=1e-4, atol=1e-4
        )


class TestTDLambdaEstimatorCompile:
    """Test TDLambdaEstimator compile-friendly vectorized property."""

    def test_vectorized_property_returns_true_in_eager_mode(self):
        """Test that TDLambdaEstimator.vectorized returns True in eager mode when set to True."""
        from tensordict.nn import TensorDictModule
        from torch import nn

        from torchrl.objectives.value.advantages import TDLambdaEstimator

        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        estimator = TDLambdaEstimator(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
            vectorized=True,
        )

        assert estimator.vectorized is True
        assert estimator._vectorized is True

    def test_vectorized_property_returns_false_in_eager_mode_when_set_false(self):
        """Test that TDLambdaEstimator.vectorized returns False in eager mode when set to False."""
        from tensordict.nn import TensorDictModule
        from torch import nn

        from torchrl.objectives.value.advantages import TDLambdaEstimator

        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        estimator = TDLambdaEstimator(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
            vectorized=False,
        )

        assert estimator.vectorized is False
        assert estimator._vectorized is False

    def test_vectorized_setter_works(self):
        """Test that TDLambdaEstimator.vectorized setter works correctly."""
        from tensordict.nn import TensorDictModule
        from torch import nn

        from torchrl.objectives.value.advantages import TDLambdaEstimator

        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        estimator = TDLambdaEstimator(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
            vectorized=True,
        )

        assert estimator.vectorized is True

        estimator.vectorized = False
        assert estimator.vectorized is False
        assert estimator._vectorized is False

        estimator.vectorized = True
        assert estimator.vectorized is True
        assert estimator._vectorized is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
