"""Test script to reproduce DreamerActorLoss compilation issue - focused on value estimation."""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.objectives.value.advantages import TDLambdaEstimator
from torchrl.objectives.value.functional import (
    td_lambda_return_estimate,
    vec_td_lambda_return_estimate,
)


def test_value_estimate_during_compile():
    """Test the value_estimate path that DreamerActorLoss.lambda_target uses."""
    print("=" * 60)
    print("Testing TDLambdaEstimator.value_estimate during compile")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create a simple value network (not used in value_estimate but required)
    value_net = TensorDictModule(
        nn.Linear(10, 1),
        in_keys=["obs"],
        out_keys=["state_value"],
    )
    
    # Create TDLambdaEstimator
    estimator = TDLambdaEstimator(
        gamma=0.99,
        lmbda=0.95,
        value_network=value_net,
        vectorized=True,  # This is what Dreamer uses by default
    )
    
    print(f"estimator.vectorized: {estimator.vectorized}")
    print(f"estimator._vectorized: {estimator._vectorized}")
    
    # Simulate what DreamerActorLoss.lambda_target does
    batch_size = 10000
    imagination_horizon = 15
    feature_dim = 1
    
    # Create data like lambda_target does
    reward = torch.randn(batch_size, imagination_horizon, feature_dim, device=device)
    next_value = torch.randn(batch_size, imagination_horizon, feature_dim, device=device)
    done = torch.zeros(batch_size, imagination_horizon, feature_dim, dtype=torch.bool, device=device)
    terminated = torch.zeros(batch_size, imagination_horizon, feature_dim, dtype=torch.bool, device=device)
    
    input_tensordict = TensorDict(
        {
            ("next", "reward"): reward,
            ("next", "state_value"): next_value,
            ("next", "done"): done,
            ("next", "terminated"): terminated,
        },
        batch_size=[],
    )
    
    # Test eager execution
    print("\nTesting eager execution of value_estimate...")
    try:
        result = estimator.value_estimate(input_tensordict, next_value=next_value)
        print(f"Eager result shape: {result.shape}")
    except Exception as e:
        print(f"Eager failed: {e}")
        return False
    
    # Now test what happens during compile
    # We'll wrap value_estimate in a function and compile it
    def value_estimate_wrapper(reward, next_value, done, terminated):
        input_td = TensorDict(
            {
                ("next", "reward"): reward,
                ("next", "state_value"): next_value,
                ("next", "done"): done,
                ("next", "terminated"): terminated,
            },
            batch_size=[],
        )
        return estimator.value_estimate(input_td, next_value=next_value)
    
    print("\nCompiling value_estimate_wrapper...")
    try:
        compiled_fn = torch.compile(
            value_estimate_wrapper,
            fullgraph=False,
            backend="inductor",
        )
        
        print("Running compiled value_estimate_wrapper...")
        result_compiled = compiled_fn(reward, next_value, done, terminated)
        print(f"Compiled result shape: {result_compiled.shape}")
        
    except Exception as e:
        print(f"Compilation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_direct_td_lambda_compile():
    """Test compiling td_lambda_return_estimate directly."""
    print("\n" + "=" * 60)
    print("Testing direct td_lambda_return_estimate compilation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 10000
    time_steps = 15
    feature_dim = 1
    
    gamma = 0.99
    lmbda = 0.95
    
    next_state_value = torch.randn(batch_size, time_steps, feature_dim, device=device)
    reward = torch.randn(batch_size, time_steps, feature_dim, device=device)
    done = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool, device=device)
    terminated = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool, device=device)
    
    # Test non-vectorized (should work)
    print("\nTesting td_lambda_return_estimate (non-vectorized)...")
    try:
        compiled_fn = torch.compile(
            td_lambda_return_estimate,
            fullgraph=True,
            backend="inductor",
        )
        result = compiled_fn(
            gamma=gamma, lmbda=lmbda, next_state_value=next_state_value,
            reward=reward, done=done, terminated=terminated,
        )
        print(f"Non-vectorized compiled result shape: {result.shape}")
    except Exception as e:
        print(f"Non-vectorized compilation failed: {type(e).__name__}: {e}")
    
    # Test vectorized (should fail with fullgraph=True)
    print("\nTesting vec_td_lambda_return_estimate (vectorized, fullgraph=False)...")
    try:
        compiled_fn = torch.compile(
            vec_td_lambda_return_estimate,
            fullgraph=False,  # Allow graph breaks
            backend="inductor",
        )
        result = compiled_fn(
            gamma=gamma, lmbda=lmbda, next_state_value=next_state_value,
            reward=reward, done=done, terminated=terminated,
        )
        print(f"Vectorized compiled result shape: {result.shape}")
    except Exception as e:
        print(f"Vectorized compilation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_value_estimate_during_compile()
    test_direct_td_lambda_compile()
