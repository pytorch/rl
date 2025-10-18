# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Weight Synchronization Schemes - Standalone Usage
==================================================

This example demonstrates how to use weight synchronization schemes independently
of collectors for custom synchronization scenarios.

The weight synchronization infrastructure provides flexible sender/receiver patterns
that can be used for various multiprocessing scenarios.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import multiprocessing as mp
from torchrl.weight_update import (
    MultiProcessWeightSyncScheme,
    SharedMemWeightSyncScheme,
)


def worker_process_mp(child_pipe, model_state):
    """Worker process that receives weights via multiprocessing pipe."""
    print("Worker: Starting...")

    # Create a policy on the worker side
    policy = nn.Linear(4, 2)
    with torch.no_grad():
        policy.weight.fill_(0.0)
        policy.bias.fill_(0.0)

    # Create receiver and register the policy
    scheme = MultiProcessWeightSyncScheme(strategy="state_dict")
    receiver = scheme.create_receiver()
    receiver.register_model(policy)
    receiver.register_worker_transport(child_pipe)

    print(f"Worker: Before update - weight sum: {policy.weight.sum().item():.4f}")

    # Receive and apply weights
    result = receiver._transport.receive_weights(timeout=5.0)
    if result is not None:
        model_id, weights = result
        receiver.apply_weights(weights)
        print(f"Worker: After update - weight sum: {policy.weight.sum().item():.4f}")
    else:
        print("Worker: No weights received")

    # Store final state for verification
    model_state["weight_sum"] = policy.weight.sum().item()
    model_state["bias_sum"] = policy.bias.sum().item()


def worker_process_shared_mem(child_pipe, model_state):
    """Worker process that receives shared memory buffer reference."""
    print("SharedMem Worker: Starting...")

    # Create a policy on the worker side
    policy = nn.Linear(4, 2)

    # Wait for shared memory buffer registration
    if child_pipe.poll(timeout=10.0):
        data, msg = child_pipe.recv()
        if msg == "register_shared_weights":
            model_id, shared_weights = data
            print(f"SharedMem Worker: Received shared buffer for model '{model_id}'")
            # Apply shared weights to policy
            shared_weights.to_module(policy)
            # Send acknowledgment
            child_pipe.send((None, "registered"))

    # Small delay to ensure main process updates shared memory
    import time

    time.sleep(0.5)

    print(f"SharedMem Worker: weight sum: {policy.weight.sum().item():.4f}")

    # Store final state for verification
    model_state["weight_sum"] = policy.weight.sum().item()
    model_state["bias_sum"] = policy.bias.sum().item()


def example_multiprocess_sync():
    """Example 1: Multiprocess weight synchronization with state_dict."""
    print("\n" + "=" * 70)
    print("Example 1: Multiprocess Weight Synchronization")
    print("=" * 70)

    # Create a simple policy on main process
    policy = nn.Linear(4, 2)
    with torch.no_grad():
        policy.weight.fill_(1.0)
        policy.bias.fill_(0.5)

    print(f"Main: Policy weight sum: {policy.weight.sum().item():.4f}")

    # Create scheme and sender
    scheme = MultiProcessWeightSyncScheme(strategy="state_dict")
    sender = scheme.create_sender()

    # Create pipe for communication
    parent_pipe, child_pipe = mp.Pipe()
    sender.register_worker(worker_idx=0, pipe_or_context=parent_pipe)

    # Start worker process
    manager = mp.Manager()
    model_state = manager.dict()
    process = mp.Process(target=worker_process_mp, args=(child_pipe, model_state))
    process.start()

    # Send weights to worker
    weights = policy.state_dict()
    print("Main: Sending weights to worker...")
    sender.update_weights(weights)

    # Wait for worker to complete
    process.join(timeout=10.0)

    if process.is_alive():
        print("Warning: Worker process did not terminate in time")
        process.terminate()
    else:
        print(
            f"Main: Worker completed. Worker's weight sum: {model_state['weight_sum']:.4f}"
        )
        print("✓ Weight synchronization successful!")


def example_shared_memory_sync():
    """Example 2: Shared memory weight synchronization."""
    print("\n" + "=" * 70)
    print("Example 2: Shared Memory Weight Synchronization")
    print("=" * 70)

    # Create a simple policy
    policy = nn.Linear(4, 2)

    # Create shared memory scheme with auto-registration
    scheme = SharedMemWeightSyncScheme(strategy="tensordict", auto_register=True)
    sender = scheme.create_sender()

    # Create pipe for lazy registration
    parent_pipe, child_pipe = mp.Pipe()
    sender.register_worker(worker_idx=0, pipe_or_context=parent_pipe)

    # Start worker process
    manager = mp.Manager()
    model_state = manager.dict()
    process = mp.Process(
        target=worker_process_shared_mem, args=(child_pipe, model_state)
    )
    process.start()

    # Send weights (automatically creates shared buffer on first send)
    weights_td = TensorDict.from_module(policy)
    with torch.no_grad():
        weights_td["weight"].fill_(2.0)
        weights_td["bias"].fill_(1.0)

    print("Main: Sending weights via shared memory...")
    sender.update_weights(weights_td)

    # Workers automatically see updates via shared memory!
    print("Main: Weights are now in shared memory, workers can access them")

    # Wait for worker to complete
    process.join(timeout=10.0)

    if process.is_alive():
        print("Warning: Worker process did not terminate in time")
        process.terminate()
    else:
        print(
            f"Main: Worker completed. Worker's weight sum: {model_state['weight_sum']:.4f}"
        )
        print("✓ Shared memory synchronization successful!")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Weight Synchronization Schemes - Standalone Usage Examples")
    print("=" * 70)

    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    # Run examples
    example_multiprocess_sync()
    example_shared_memory_sync()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
