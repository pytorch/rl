"""Test TDLambdaEstimator compilation with vectorized=False."""

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives.value.advantages import TDLambdaEstimator


def test_tdlambda_estimator_vectorized_false():
    """Test TDLambdaEstimator compilation with vectorized=False."""
    print("\n" + "=" * 60)
    print("Testing TDLambdaEstimator (vectorized=False) compilation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    time_steps = 15
    feature_dim = 3

    # Create a simple value network
    value_net = TensorDictModule(
        nn.Linear(feature_dim, 1), in_keys=["obs"], out_keys=["state_value"]
    )
    value_net.to(device)

    # Create TDLambdaEstimator with vectorized=False
    estimator = TDLambdaEstimator(
        gamma=0.99,
        lmbda=0.95,
        value_network=value_net,
        vectorized=False,  # Use non-vectorized path
    )

    # Create test data
    obs = torch.randn(batch_size, time_steps, feature_dim, device=device)
    next_obs = torch.randn(batch_size, time_steps, feature_dim, device=device)
    reward = torch.randn(batch_size, time_steps, 1, device=device)
    done = torch.zeros(batch_size, time_steps, 1, dtype=torch.bool, device=device)
    terminated = torch.zeros(batch_size, time_steps, 1, dtype=torch.bool, device=device)

    tensordict = TensorDict(
        {
            "obs": obs,
            "next": {
                "obs": next_obs,
                "reward": reward,
                "done": done,
                "terminated": terminated,
            },
        },
        batch_size=[batch_size, time_steps],
    )

    # Run eagerly first
    print("\nRunning eagerly first...")
    result_eager = estimator(tensordict.clone())
    print(f"Eager advantage shape: {result_eager['advantage'].shape}")

    # Try to compile with fullgraph=True
    print("\nAttempting to compile with fullgraph=True...")
    try:
        compiled_estimator = torch.compile(
            estimator,
            fullgraph=True,
            backend="inductor",
        )

        result_compiled = compiled_estimator(tensordict.clone())
        print(f"Compiled advantage shape: {result_compiled['advantage'].shape}")
        print("SUCCESS: TDLambdaEstimator (vectorized=False) compiled with fullgraph=True!")
        return True

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


def test_tdlambda_estimator_vectorized_true():
    """Test TDLambdaEstimator compilation with vectorized=True (default)."""
    print("\n" + "=" * 60)
    print("Testing TDLambdaEstimator (vectorized=True) compilation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    time_steps = 15
    feature_dim = 3

    # Create a simple value network
    value_net = TensorDictModule(
        nn.Linear(feature_dim, 1), in_keys=["obs"], out_keys=["state_value"]
    )
    value_net.to(device)

    # Create TDLambdaEstimator with vectorized=True (default)
    estimator = TDLambdaEstimator(
        gamma=0.99,
        lmbda=0.95,
        value_network=value_net,
        vectorized=True,  # Use vectorized path (default)
    )

    # Create test data
    obs = torch.randn(batch_size, time_steps, feature_dim, device=device)
    next_obs = torch.randn(batch_size, time_steps, feature_dim, device=device)
    reward = torch.randn(batch_size, time_steps, 1, device=device)
    done = torch.zeros(batch_size, time_steps, 1, dtype=torch.bool, device=device)
    terminated = torch.zeros(batch_size, time_steps, 1, dtype=torch.bool, device=device)

    tensordict = TensorDict(
        {
            "obs": obs,
            "next": {
                "obs": next_obs,
                "reward": reward,
                "done": done,
                "terminated": terminated,
            },
        },
        batch_size=[batch_size, time_steps],
    )

    # Run eagerly first
    print("\nRunning eagerly first...")
    result_eager = estimator(tensordict.clone())
    print(f"Eager advantage shape: {result_eager['advantage'].shape}")

    # Try to compile with fullgraph=True
    print("\nAttempting to compile with fullgraph=True...")
    try:
        compiled_estimator = torch.compile(
            estimator,
            fullgraph=True,
            backend="inductor",
        )

        result_compiled = compiled_estimator(tensordict.clone())
        print(f"Compiled advantage shape: {result_compiled['advantage'].shape}")
        print("SUCCESS: TDLambdaEstimator (vectorized=True) compiled with fullgraph=True!")
        return True

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("Testing TDLambdaEstimator compilation")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results = {}
    results["TDLambdaEstimator (vectorized=False)"] = test_tdlambda_estimator_vectorized_false()
    results["TDLambdaEstimator (vectorized=True)"] = test_tdlambda_estimator_vectorized_true()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")



