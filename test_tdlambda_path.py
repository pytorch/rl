"""Test which path TDLambdaEstimator.value_estimate takes during compile."""
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

# Import the module first
from torchrl.objectives.value import advantages

# Patch the functions at the module level where they're used
original_vec = advantages.vec_td_lambda_return_estimate
original_nonvec = advantages.td_lambda_return_estimate

vec_call_count = [0]
nonvec_call_count = [0]

def patched_vec(*args, **kwargs):
    vec_call_count[0] += 1
    print(f"[TRACE] vec_td_lambda_return_estimate called (count: {vec_call_count[0]})")
    return original_vec(*args, **kwargs)

def patched_nonvec(*args, **kwargs):
    nonvec_call_count[0] += 1
    print(f"[TRACE] td_lambda_return_estimate called (count: {nonvec_call_count[0]})")
    return original_nonvec(*args, **kwargs)

advantages.vec_td_lambda_return_estimate = patched_vec
advantages.td_lambda_return_estimate = patched_nonvec

from torchrl.objectives.value.advantages import TDLambdaEstimator

# Create estimator
value_net = TensorDictModule(
    nn.Linear(10, 1),
    in_keys=["obs"],
    out_keys=["state_value"],
)

estimator = TDLambdaEstimator(
    gamma=0.99,
    lmbda=0.95,
    value_network=value_net,
    vectorized=True,
)

print(f"estimator._vectorized: {estimator._vectorized}")

# Create test data
batch_size = 100
time_steps = 15
feature_dim = 1

reward = torch.randn(batch_size, time_steps, feature_dim)
next_value = torch.randn(batch_size, time_steps, feature_dim)
done = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool)
terminated = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.bool)

input_tensordict = TensorDict(
    {
        ("next", "reward"): reward,
        ("next", "state_value"): next_value,
        ("next", "done"): done,
        ("next", "terminated"): terminated,
    },
    batch_size=[],
)

print("\n=== Eager execution ===")
vec_call_count[0] = 0
nonvec_call_count[0] = 0
result = estimator.value_estimate(input_tensordict.clone(), next_value=next_value)
print(f"vec calls: {vec_call_count[0]}, nonvec calls: {nonvec_call_count[0]}")

print("\n=== Compiled execution ===")
torch._dynamo.reset()  # Clear any cached compilations
vec_call_count[0] = 0
nonvec_call_count[0] = 0

def wrapper(reward, next_value, done, terminated):
    td = TensorDict(
        {
            ("next", "reward"): reward,
            ("next", "state_value"): next_value,
            ("next", "done"): done,
            ("next", "terminated"): terminated,
        },
        batch_size=[],
    )
    return estimator.value_estimate(td, next_value=next_value)

compiled = torch.compile(wrapper, fullgraph=False, backend="eager")
result = compiled(reward, next_value, done, terminated)
print(f"vec calls: {vec_call_count[0]}, nonvec calls: {nonvec_call_count[0]}")
