"""Test script to verify if value functions can be compiled with fullgraph=True."""

import torch
from torchrl.objectives.value.functional import (
    vec_td_lambda_return_estimate,
    vec_generalized_advantage_estimate,
    reward2go,
)


def test_vec_td_lambda_return_estimate_compile():
    """Test compiling vec_td_lambda_return_estimate with fullgraph=True."""
    print("\n" + "=" * 60)
    print("Testing vec_td_lambda_return_estimate compilation")
    print("=" * 60)

    # Setup: typical dreamer-like scenario with scalar gamma/lmbda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    time_steps = 15
    feature_dim = 1

    gamma = 0.99
    lmbda = 0.95

    # Create test tensors (no mid-trajectory dones - fixed horizon scenario)
    next_state_value = torch.randn(batch_size, time_steps, feature_dim, device=device)
    reward = torch.randn(batch_size, time_steps, feature_dim, device=device)
    done = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool, device=device)
    terminated = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool, device=device)

    # First, run eagerly to ensure correctness
    print("\nRunning eagerly first...")
    result_eager = vec_td_lambda_return_estimate(
        gamma=gamma,
        lmbda=lmbda,
        next_state_value=next_state_value,
        reward=reward,
        done=done,
        terminated=terminated,
    )
    print(f"Eager result shape: {result_eager.shape}")

    # Now try to compile with fullgraph=True
    print("\nAttempting to compile with fullgraph=True...")
    try:
        compiled_fn = torch.compile(
            vec_td_lambda_return_estimate,
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
        print("SUCCESS: vec_td_lambda_return_estimate compiled with fullgraph=True!")

        # Verify correctness
        if torch.allclose(result_eager, result_compiled, rtol=1e-4, atol=1e-4):
            print("Results match between eager and compiled versions.")
        else:
            print("WARNING: Results differ between eager and compiled versions!")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False

    return True


def test_vec_gae_compile():
    """Test compiling vec_generalized_advantage_estimate with fullgraph=True."""
    print("\n" + "=" * 60)
    print("Testing vec_generalized_advantage_estimate compilation")
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
    advantage_eager, value_target_eager = vec_generalized_advantage_estimate(
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
            vec_generalized_advantage_estimate,
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
        print("SUCCESS: vec_generalized_advantage_estimate compiled with fullgraph=True!")

        # Verify correctness
        if torch.allclose(advantage_eager, advantage_compiled, rtol=1e-4, atol=1e-4):
            print("Advantages match between eager and compiled versions.")
        else:
            print("WARNING: Advantages differ between eager and compiled versions!")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False

    return True


def test_reward2go_compile():
    """Test compiling reward2go with fullgraph=True."""
    print("\n" + "=" * 60)
    print("Testing reward2go compilation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    time_steps = 15
    feature_dim = 1

    gamma = 0.99

    reward = torch.randn(batch_size, time_steps, feature_dim, device=device)
    done = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool, device=device)

    # Run eagerly first
    print("\nRunning eagerly first...")
    result_eager = reward2go(reward, done, gamma)
    print(f"Eager result shape: {result_eager.shape}")

    # Try to compile with fullgraph=True
    print("\nAttempting to compile with fullgraph=True...")
    try:
        compiled_fn = torch.compile(
            reward2go,
            fullgraph=True,
            backend="inductor",
        )

        result_compiled = compiled_fn(reward, done, gamma)
        print(f"Compiled result shape: {result_compiled.shape}")
        print("SUCCESS: reward2go compiled with fullgraph=True!")

        # Verify correctness
        if torch.allclose(result_eager, result_compiled, rtol=1e-4, atol=1e-4):
            print("Results match between eager and compiled versions.")
        else:
            print("WARNING: Results differ between eager and compiled versions!")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False

    return True


def test_with_graph_breaks():
    """Test compilation with fullgraph=False (allowing graph breaks)."""
    print("\n" + "=" * 60)
    print("Testing compilation with fullgraph=False (graph breaks allowed)")
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

    print("\nCompiling vec_td_lambda_return_estimate with fullgraph=False...")
    try:
        compiled_fn = torch.compile(
            vec_td_lambda_return_estimate,
            fullgraph=False,
            backend="inductor",
        )

        result = compiled_fn(
            gamma=gamma,
            lmbda=lmbda,
            next_state_value=next_state_value,
            reward=reward,
            done=done,
            terminated=terminated,
        )
        print(f"Result shape: {result.shape}")
        print("SUCCESS: Compiled with graph breaks allowed.")
        return True

    except Exception as e:
        print(f"FAILED even with graph breaks: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("Testing value function compilation for torch.compile compatibility")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Run tests
    results = {}
    results["vec_td_lambda_return_estimate (fullgraph=True)"] = test_vec_td_lambda_return_estimate_compile()
    results["vec_generalized_advantage_estimate (fullgraph=True)"] = test_vec_gae_compile()
    results["reward2go (fullgraph=True)"] = test_reward2go_compile()
    results["vec_td_lambda_return_estimate (fullgraph=False)"] = test_with_graph_breaks()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")



