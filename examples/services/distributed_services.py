# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Example: Distributed Service Registry with Ray

This example demonstrates how to use TorchRL's service registry to share
actors (tokenizers, replay buffers, etc.) across distributed workers.

Key features:
- Services registered by one worker are immediately visible to all workers
- Supports Ray's full options API for resource management
- Clean dict-like interface for service access

Run this example:
    python examples/services/distributed_services.py
"""

import ray
from torchrl.services import get_services


# Example 1: Simple service class
class TokenizerService:
    """A simple tokenizer service that can be shared across workers."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        print(f"TokenizerService initialized with vocab_size={vocab_size}")

    def encode(self, text: str) -> list[int]:
        """Simple character-based encoding."""
        return [hash(c) % self.vocab_size for c in text]

    def decode(self, tokens: list[int]) -> str:
        """Simple decoding."""
        return "".join([chr(65 + (t % 26)) for t in tokens])


# Example 2: Stateful service
class CounterService:
    """A stateful counter service."""

    def __init__(self, initial_value: int = 0):
        self.count = initial_value

    def increment(self) -> int:
        self.count += 1
        return self.count

    def get_count(self) -> int:
        return self.count

    def reset(self):
        self.count = 0


# Example 3: Resource-intensive service
class ModelService:
    """Simulates a model service that needs GPU resources."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"ModelService '{model_name}' initialized")

    def predict(self, data: list) -> list:
        """Simulate model inference."""
        return [x * 2 for x in data]  # Simple transformation


@ray.remote
class Worker:
    """Worker class that uses services.

    Separates initialization from execution for better control.
    """

    def __init__(self, worker_id: int, namespace: str):
        """Initialize worker and get service registry."""
        self.worker_id = worker_id
        self.namespace = namespace
        self.services = get_services(backend="ray", namespace=namespace)
        print(f"\n[Worker {worker_id}] Initialized")

    def setup_services(self):
        """Register shared services (typically done by worker 0)."""
        print(f"[Worker {self.worker_id}] Registering services...")

        # Register tokenizer with specific resources
        self.services.register(
            "tokenizer",
            TokenizerService,
            vocab_size=5000,
            num_cpus=1,
            num_gpus=0,
        )

        # Register counter
        self.services.register("counter", CounterService, initial_value=0, num_cpus=0.5)

        # Register model with GPU (if available)
        self.services.register(
            "model",
            ModelService,
            model_name="my_model",
            num_cpus=2,
            num_gpus=0,  # Set to 1 if GPU available
        )

        print(f"[Worker {self.worker_id}] Services registered: {self.services.list()}")
        return "setup_complete"

    def run(self):
        """Execute worker's main task using services."""
        print(f"[Worker {self.worker_id}] Starting execution...")
        print(f"[Worker {self.worker_id}] Available services: {self.services.list()}")

        # Use tokenizer
        if "tokenizer" in self.services:
            tokenizer = self.services["tokenizer"]
            text = f"Hello from worker {self.worker_id}"
            encoded = ray.get(tokenizer.encode.remote(text))
            decoded = ray.get(tokenizer.decode.remote(encoded))
            print(f"[Worker {self.worker_id}] Tokenizer - Encoded/decoded: '{decoded}'")

        # Use counter (demonstrates statefulness)
        if "counter" in self.services:
            counter = self.services["counter"]
            count = ray.get(counter.increment.remote())
            print(f"[Worker {self.worker_id}] Counter incremented to: {count}")

        # Use model
        if "model" in self.services:
            model = self.services["model"]
            result = ray.get(model.predict.remote([1, 2, 3]))
            print(f"[Worker {self.worker_id}] Model prediction: {result}")

        print(f"[Worker {self.worker_id}] Finished!")
        return f"Worker {self.worker_id} completed"


def main():
    """Main function demonstrating service registry usage."""
    print("=== TorchRL Distributed Service Registry Example ===\n")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    namespace = "example_services"

    # Example 1: Basic usage
    print("--- Example 1: Basic Usage ---")
    services = get_services(backend="ray", namespace=namespace)

    # Register a simple service
    services.register("shared_tokenizer", TokenizerService, vocab_size=1000)

    # Access it
    tokenizer = services["shared_tokenizer"]
    result = ray.get(tokenizer.encode.remote("Hello"))
    print(f"Encoded 'Hello': {result}\n")

    # Example 2: Conditional registration pattern
    print("--- Example 2: Conditional Registration ---")
    assert "shared_tokenizer" in services
    try:
        services.register("shared_tokenizer", TokenizerService, vocab_size=1000)
        raise RuntimeError("Registed twice! Should not happen!")
    except ValueError:
        print("shared_tokenizer already registered")

    # Example 3: Multiple workers using same services
    print("--- Example 3: Multiple Workers Sharing Services ---")

    # Create worker actors
    num_workers = 3
    workers = [
        Worker.remote(worker_id, namespace) for worker_id in range(num_workers)  # type: ignore[attr-defined]
    ]

    # Worker 0 sets up services, others wait for completion
    # The registered services are: (1) tokenizer, (2) counter, (3) model
    print("Worker 0 setting up services...")
    setup_complete = ray.get(workers[0].setup_services.remote())
    print(f"Setup complete: {setup_complete}")

    # note that in `main` the services are updated too!
    assert "tokenizer" in services
    assert "counter" in services
    assert "model" in services

    # Now all workers can run in parallel
    print("\nAll workers executing...")
    run_futures = [worker.run.remote() for worker in workers]

    # Wait for all workers to complete
    results = ray.get(run_futures)
    print("\nAll workers completed:", results)

    # Example 4: Using register_with_options for clarity
    print("\n--- Example 4: Register with Explicit Options ---")
    services.reset()
    # More explicit separation of Ray options vs constructor args
    services.register_with_options(  # type: ignore[attr-defined]
        "tokenizer",
        TokenizerService,
        actor_options={
            "num_cpus": 2,
            "num_gpus": 0,
            "max_concurrency": 10,
            "memory": 5 * 1024**3,
        },
        vocab_size=50000,  # Constructor argument
    )

    tokenizer = services["tokenizer"]
    result = ray.get(tokenizer.encode.remote("test"))
    print(f"Used tokenizer with explicit options: {result}")

    # Example 5: Listing all services
    print("\n--- Example 5: Service Discovery ---")
    all_services = services.list()
    print(f"All registered services ({len(all_services)}): {all_services}")

    # Example 6: Resetting services
    print("\n--- Example 6: Resetting Services ---")
    print(f"Services before reset: {services.list()}")

    # Reset will terminate all actors and clear the registry
    services.reset()

    print(f"Services after reset: {services.list()}")
    assert len(services.list()) == 0, "Registry should be empty after reset"

    print("\n=== Example Complete ===")

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()
