"""Test if property with is_dynamo_compiling works correctly in a class."""
import torch

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling


class MyEstimator:
    def __init__(self, vectorized=True):
        self._vectorized = vectorized
    
    @property
    def vectorized(self):
        if is_dynamo_compiling():
            return False
        return self._vectorized
    
    def estimate(self, x):
        if self.vectorized:
            # Vectorized path - add 1000
            return x + 1000.0
        else:
            # Non-vectorized path - add 1
            return x + 1.0


# Test eager
print("=== Eager mode ===")
estimator = MyEstimator(vectorized=True)
print(f"estimator.vectorized (outside): {estimator.vectorized}")
x = torch.tensor([0.0])
result = estimator.estimate(x)
print(f"Result: {result}")  # Should be 1000.0 (vectorized path)

# Test compiled - compile just the estimate method
print("\n=== Compiled mode (method only) ===")
compiled_estimate = torch.compile(estimator.estimate, fullgraph=True, backend="eager")
result = compiled_estimate(x)
print(f"Result: {result}")  # Should be 1.0 (non-vectorized path) because is_dynamo_compiling() returns True


# Test with the whole class compiled
print("\n=== Compiled mode (whole forward) ===")
def forward_wrapper(x):
    return estimator.estimate(x)

compiled_wrapper = torch.compile(forward_wrapper, fullgraph=True, backend="eager")
result = compiled_wrapper(x)
print(f"Result: {result}")  # Should be 1.0 (non-vectorized path)


