#!/usr/bin/env python3
"""
Demo 3: Project Indicates the Pipeline Within a Job (Hierarchical Metrics)

This demo shows how to use a single W&B project with hierarchical metric paths
to separate training and inference metrics:
- project="rl-agent" for all runs
- Training metrics: "training/loss", "training/grads/grad_norm_0", etc.
- Inference metrics: "inference/e2e/speed", "inference/reward/mean", etc.

Key patterns demonstrated:
- Single project contains all related runs
- Metric path hierarchy (using /) creates automatic panel grouping in W&B
- Training and inference can be in the same run or separate runs
- W&B dashboard auto-organizes panels by metric path prefix

Usage:
    uv run python personal/vincent/wandb_exploration/demo3_hierarchical_metrics.py
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
    """Run the hierarchical metrics demo."""
    init_ray_if_needed()

    run_ts = timestamp()
    experiment_id = f"exp-{run_ts}"

    print("=" * 70)
    print("Demo 3: Hierarchical Metrics in a Single Project")
    print("=" * 70)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"Project: rl-agent (single project for both pipelines)")
    print(f"Steps: {config.num_steps}")
    print()
    print("Metric hierarchy:")
    print("  training/")
    print("    ├── loss")
    print("    ├── learning_rate")
    print("    └── grads/")
    print("        ├── grad_norm_0")
    print("        ├── grad_norm_1")
    print("        └── grad_norm_total")
    print("  inference/")
    print("    ├── reward/")
    print("    │   └── mean")
    print("    └── e2e/")
    print("        ├── speed")
    print("        └── latency_ms")
    print()

    # Both actors log to the SAME project but with different metric prefixes
    training_actor = TrainingPipelineActor.remote(
        project="rl-agent",
        run_name=f"train-{experiment_id}",
        config={
            "experiment_id": experiment_id,
            "pipeline": "training",
            "model": "gpt-rl-v1",
            "learning_rate": 1e-4,
        },
        tags=["demo3", "hierarchical", "training"],
        metric_prefix="training",  # All metrics will be prefixed with "training/"
    )

    inference_actor = InferencePipelineActor.remote(
        project="rl-agent",  # Same project!
        run_name=f"inference-{experiment_id}",
        config={
            "experiment_id": experiment_id,
            "pipeline": "inference",
            "model": "gpt-rl-v1",
            "temperature": 0.7,
        },
        tags=["demo3", "hierarchical", "inference"],
        metric_prefix="inference",  # All metrics will be prefixed with "inference/"
    )

    train_url = ray.get(training_actor.get_url.remote())
    inference_url = ray.get(inference_actor.get_url.remote())
    print(f"Training run:  {train_url}")
    print(f"Inference run: {inference_url}")
    print()

    # Create workers
    training_workers = [
        TrainingWorker.remote(worker_id=i) for i in range(config.num_training_workers)
    ]
    inference_workers = [
        InferenceWorker.remote(worker_id=i) for i in range(config.num_inference_workers)
    ]

    print("Running with hierarchical metric paths...")
    print("-" * 70)

    for step in range(config.num_steps):
        # Training step - metrics will be logged as "training/..."
        training_futures = [w.do_training_step.remote(step) for w in training_workers]
        training_results = ray.get(training_futures)

        avg_loss = sum(r["loss"] for r in training_results) / len(training_results)
        avg_grad_0 = sum(r["grad_norm_0"] for r in training_results) / len(training_results)
        avg_grad_1 = sum(r["grad_norm_1"] for r in training_results) / len(training_results)
        avg_grad_total = sum(r["grad_norm_total"] for r in training_results) / len(
            training_results
        )

        # Log with hierarchical structure (prefix "training/" added by actor)
        # Final keys: training/loss, training/grads/grad_norm_0, etc.
        ray.get(
            training_actor.log_metrics.remote(
                {
                    "loss": avg_loss,
                    "learning_rate": training_results[0]["learning_rate"],
                    "grads/grad_norm_0": avg_grad_0,
                    "grads/grad_norm_1": avg_grad_1,
                    "grads/grad_norm_total": avg_grad_total,
                    "tokens_processed": training_results[0]["tokens_processed"],
                },
                step=step,
            )
        )

        # Inference step - metrics will be logged as "inference/..."
        inference_futures = [w.do_inference_step.remote(step) for w in inference_workers]
        inference_results = ray.get(inference_futures)

        avg_reward = sum(r["episode_reward"] for r in inference_results) / len(
            inference_results
        )
        avg_speed = sum(r["e2e_speed"] for r in inference_results) / len(inference_results)
        avg_latency = sum(r["e2e_latency_ms"] for r in inference_results) / len(
            inference_results
        )

        # Log with hierarchical structure (prefix "inference/" added by actor)
        # Final keys: inference/reward/mean, inference/e2e/speed, etc.
        ray.get(
            inference_actor.log_metrics.remote(
                {
                    "reward/mean": avg_reward,
                    "reward/max": max(r["episode_reward"] for r in inference_results),
                    "reward/min": min(r["episode_reward"] for r in inference_results),
                    "e2e/speed": avg_speed,
                    "e2e/latency_ms": avg_latency,
                    "success_rate": inference_results[0]["success_rate"],
                    "rollouts_completed": inference_results[0]["rollouts_completed"],
                },
                step=step,
            )
        )

        print(
            f"Step {step:3d}: training/loss={avg_loss:.4f}, "
            f"inference/reward/mean={avg_reward:.2f}"
        )

        if config.step_delay > 0:
            time.sleep(config.step_delay)

    print("-" * 70)
    print("Finishing runs...")

    ray.get(training_actor.finish.remote())
    ray.get(inference_actor.finish.remote())

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nProject: rl-agent")
    print(f"Training run:  {train_url}")
    print(f"Inference run: {inference_url}")
    print()
    print("Key observations:")
    print("  - Both runs are in the same 'rl-agent' project")
    print("  - Metric paths use '/' to create hierarchy")
    print("  - W&B dashboard auto-groups panels by prefix (training/, inference/)")
    print()
    print("Dashboard organization:")
    print("  - 'training' section: loss, learning_rate, grads/* panels")
    print("  - 'inference' section: reward/*, e2e/* panels")
    print("  - Use W&B's 'Group by' to organize panels by path prefix")
    print()
    print("Comparison benefits:")
    print("  - Easy to compare training vs inference in same workspace")
    print("  - Clear visual separation of pipeline metrics")
    print("  - Can create dashboards that show both pipelines side-by-side")


if __name__ == "__main__":
    config = DemoConfig(
        num_steps=20,
        num_training_workers=2,
        num_inference_workers=2,
        step_delay=0.3,
    )
    run_demo(config)

