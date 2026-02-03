"""Test if is_dynamo_compiling works correctly."""
import torch

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling


def check_compile_status(x):
    """Function that returns different values based on compile status."""
    if is_dynamo_compiling():
        # During compile, return x + 100
        return x + 100.0
    else:
        # During eager, return x + 1
        return x + 1.0


# Test eager
print("=== Eager mode ===")
x = torch.tensor([0.0])
result = check_compile_status(x)
print(f"Result: {result}")  # Should be 1.0

# Test compiled
print("\n=== Compiled mode ===")
compiled_fn = torch.compile(check_compile_status, fullgraph=True, backend="eager")
result = compiled_fn(x)
print(f"Result: {result}")  # Should be 100.0 if is_dynamo_compiling() works during tracing
