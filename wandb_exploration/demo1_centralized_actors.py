#!/usr/bin/env python3
"""
Demo 1: Centralized Logging via Ray Actors

This demo shows how to structure W&B logging where:
- One Ray actor owns the training W&B run
- Another Ray actor owns the inference W&B run
- Sub-workers send metrics back to their respective actor for centralized logging

Key patterns demonstrated:
- Each actor calls wandb.init() in its constructor
- Workers generate metrics and send to actors via ray.get()
- Actors aggregate and log metrics to W&B
- Proper wandb.finish() cleanup in each actor

Usage:
    uv run python personal/vincent/wandb_exploration/demo1_centralized_actors.py
"""

import time

import ray

from common import (
    DemoConfig,
    InferencePipelineActor,
    InferenceWorker,
    TrainingPipelineActor,
    TrainingWorker,
    init_ray_if_needed,
    timestamp,
)


def run_demo(config: DemoConfig) -> None:
    """Run the centralized logging demo."""
    init_ray_if_needed()

    run_ts = timestamp()
    experiment_name = f"demo1-centralized-{run_ts}"

    print("=" * 70)
    print("Demo 1: Centralized Logging via Ray Actors")
    print("=" * 70)
    print(f"\nExperiment: {experiment_name}")
    print(f"Steps: {config.num_steps}")
    print(f"Training workers: {config.num_training_workers}")
    print(f"Inference workers: {config.num_inference_workers}")
    print()

    # Create training and inference actors with their own W&B runs
    # Note: Each actor gets its own project to demonstrate complete separation
    training_actor = TrainingPipelineActor.remote(
        project="rl-training-demo",
        run_name=f"train-{experiment_name}",
        config={
            "experiment": experiment_name,
            "pipeline": "training",
            "num_workers": config.num_training_workers,
            "num_steps": config.num_steps,
        },
        tags=["demo1", "centralized", "training"],
    )

    inference_actor = InferencePipelineActor.remote(
        project="rl-inference-demo",
        run_name=f"inference-{experiment_name}",
        config={
            "experiment": experiment_name,
            "pipeline": "inference",
            "num_workers": config.num_inference_workers,
            "num_steps": config.num_steps,
        },
        tags=["demo1", "centralized", "inference"],
    )

    # Print run URLs
    train_url = ray.get(training_actor.get_url.remote())
    inference_url = ray.get(inference_actor.get_url.remote())
    print(f"Training run:  {train_url}")
    print(f"Inference run: {inference_url}")
    print()

    # Create worker actors
    training_workers = [
        TrainingWorker.remote(worker_id=i) for i in range(config.num_training_workers)
    ]
    inference_workers = [
        InferenceWorker.remote(worker_id=i) for i in range(config.num_inference_workers)
    ]

    # Run training and inference in parallel
    print("Starting training and inference loops...")
    print("-" * 70)

    for step in range(config.num_steps):
        step_start = time.time()

        # Training: Workers do work and send metrics to training actor
        training_futures = [
            worker.do_training_step.remote(step) for worker in training_workers
        ]
        training_results = ray.get(training_futures)

        # Aggregate training metrics in the actor
        for worker_id, metrics in enumerate(training_results):
            ray.get(training_actor.aggregate_worker_metrics.remote(worker_id, metrics))
        ray.get(training_actor.flush_aggregated_metrics.remote(step))

        # Inference: Workers do work and send metrics to inference actor
        inference_futures = [
            worker.do_inference_step.remote(step) for worker in inference_workers
        ]
        inference_results = ray.get(inference_futures)

        # Aggregate inference metrics in the actor
        for worker_id, metrics in enumerate(inference_results):
            ray.get(inference_actor.aggregate_worker_metrics.remote(worker_id, metrics))
        ray.get(inference_actor.flush_aggregated_metrics.remote(step))

        step_time = time.time() - step_start
        print(f"Step {step:3d}: train_loss={training_results[0]['loss']:.4f}, "
              f"reward={inference_results[0]['episode_reward']:.2f} "
              f"({step_time:.2f}s)")

        # Delay for visualization in W&B dashboard
        if config.step_delay > 0:
            time.sleep(config.step_delay)

    print("-" * 70)
    print("Demo complete! Cleaning up...")

    # Proper cleanup: finish both runs
    ray.get(training_actor.finish.remote())
    ray.get(inference_actor.finish.remote())

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Training run:  {train_url}")
    print(f"Inference run: {inference_url}")
    print()
    print("Key observations:")
    print("  - Two separate W&B projects: rl-training-demo and rl-inference-demo")
    print("  - Each project contains runs only for its pipeline type")
    print("  - Metrics are aggregated from workers before logging")
    print("  - Each actor manages its own W&B lifecycle (init/log/finish)")


if __name__ == "__main__":
    config = DemoConfig(
        num_steps=20,
        num_training_workers=2,
        num_inference_workers=2,
        step_delay=0.3,  # Faster for demo
    )
    run_demo(config)

