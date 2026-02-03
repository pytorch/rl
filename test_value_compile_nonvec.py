"""Test script to verify if non-vectorized value functions can be compiled with fullgraph=True."""

import torch
from torchrl.objectives.value.functional import (
    td_lambda_return_estimate,
    generalized_advantage_estimate,
)


def test_td_lambda_return_estimate_nonvec_compile():
    """Test compiling td_lambda_return_estimate (non-vectorized) with fullgraph=True."""
    print("\n" + "=" * 60)
    print("Testing td_lambda_return_estimate (non-vectorized) compilation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    time_steps = 15
    feature_dim = 1

    gamma = 0.99
    lmbda = 0.95

    next_state_value = torch.randn(batch_size, time_steps, feature_dim, device=device)
    reward = torch.randn(batch_size, time_steps, feature_dim, device=device)
    done = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool, device=device)
    terminated = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool, device=device)

    # Run eagerly first
    print("\nRunning eagerly first...")
    result_eager = td_lambda_return_estimate(
        gamma=gamma,
        lmbda=lmbda,
        next_state_value=next_state_value,
        reward=reward,
        done=done,
        terminated=terminated,
    )
    print(f"Eager result shape: {result_eager.shape}")

    # Try to compile with fullgraph=True
    print("\nAttempting to compile with fullgraph=True...")
    try:
        compiled_fn = torch.compile(
            td_lambda_return_estimate,
            fullgraph=True,
            backend="inductor",
        )

        result_compiled = compiled_fn(
            gamma=gamma,
            lmbda=lmbda,
            next_state_value=next_state_value,
            reward=reward,
            done=done,
            terminated=terminated,
        )
        print(f"Compiled result shape: {result_compiled.shape}")
        print("SUCCESS: td_lambda_return_estimate compiled with fullgraph=True!")

        if torch.allclose(result_eager, result_compiled, rtol=1e-4, atol=1e-4):
            print("Results match between eager and compiled versions.")
        else:
            print("WARNING: Results differ between eager and compiled versions!")

        return True

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


def test_generalized_advantage_estimate_nonvec_compile():
    """Test compiling generalized_advantage_estimate (non-vectorized) with fullgraph=True."""
    print("\n" + "=" * 60)
    print("Testing generalized_advantage_estimate (non-vectorized) compilation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    time_steps = 15
    feature_dim = 1

    gamma = 0.99
    lmbda = 0.95

    state_value = torch.randn(batch_size, time_steps, feature_dim, device=device)
    next_state_value = torch.randn(batch_size, time_steps, feature_dim, device=device)
    reward = torch.randn(batch_size, time_steps, feature_dim, device=device)
    done = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool, device=device)
    terminated = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool, device=device)

    # Run eagerly first
    print("\nRunning eagerly first...")
    advantage_eager, value_target_eager = generalized_advantage_estimate(
        gamma=gamma,
        lmbda=lmbda,
        state_value=state_value,
        next_state_value=next_state_value,
        reward=reward,
        done=done,
        terminated=terminated,
    )
    print(f"Eager advantage shape: {advantage_eager.shape}")

    # Try to compile with fullgraph=True
    print("\nAttempting to compile with fullgraph=True...")
    try:
        compiled_fn = torch.compile(
            generalized_advantage_estimate,
            fullgraph=True,
            backend="inductor",
        )

        advantage_compiled, value_target_compiled = compiled_fn(
            gamma=gamma,
            lmbda=lmbda,
            state_value=state_value,
            next_state_value=next_state_value,
            reward=reward,
            done=done,
            terminated=terminated,
        )
        print(f"Compiled advantage shape: {advantage_compiled.shape}")
        print("SUCCESS: generalized_advantage_estimate compiled with fullgraph=True!")

        if torch.allclose(advantage_eager, advantage_compiled, rtol=1e-4, atol=1e-4):
            print("Advantages match between eager and compiled versions.")
        else:
            print("WARNING: Advantages differ between eager and compiled versions!")

        return True

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("Testing non-vectorized value function compilation")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results = {}
    results["td_lambda_return_estimate (non-vec, fullgraph=True)"] = test_td_lambda_return_estimate_nonvec_compile()
    results["generalized_advantage_estimate (non-vec, fullgraph=True)"] = test_generalized_advantage_estimate_nonvec_compile()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")



