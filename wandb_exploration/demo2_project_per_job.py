#!/usr/bin/env python3
"""
Demo 2: Project Indicates the Type of Job

This demo shows how to use W&B projects to separate different job types:
- project="rl-agent" for RL training runs
- project="inference-only" for inference-only runs

Key patterns demonstrated:
- Same experiment name prefix links related runs across projects
- SHARED METRIC NAMES for cross-project comparison (e2e/speed, throughput)
- Pipeline-specific metrics with prefixes (training/loss, inference/reward)
- W&B Reports can compare shared metrics across projects

Usage:
    uv run python personal/vincent/wandb_exploration/demo2_project_per_job.py
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
    """Run the project-per-job-type demo."""
    init_ray_if_needed()

    run_ts = timestamp()
    experiment_id = f"exp-{run_ts}"

    print("=" * 70)
    print("Demo 2: Project Indicates the Type of Job")
    print("=" * 70)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"Steps: {config.num_steps}")
    print()
    print("Project structure:")
    print("  - project='rl-agent'        -> Training runs")
    print("  - project='inference-only'  -> Inference runs")
    print()

    # Create training actor in the "rl-agent" project
    training_actor = TrainingPipelineActor.remote(
        project="rl-agent",
        run_name=f"training-{experiment_id}",
        config={
            "experiment_id": experiment_id,
            "job_type": "training",
            "model": "gpt-rl-v1",
            "learning_rate": 1e-4,
            "batch_size": 32,
            "num_workers": config.num_training_workers,
        },
        tags=["demo2", "training", experiment_id],
    )

    # Create inference actor in the "inference-only" project
    inference_actor = InferencePipelineActor.remote(
        project="inference-only",
        run_name=f"inference-{experiment_id}",
        config={
            "experiment_id": experiment_id,
            "job_type": "inference",
            "model": "gpt-rl-v1",
            "max_tokens": 2048,
            "temperature": 0.7,
            "num_workers": config.num_inference_workers,
            # Cross-reference to training run
            "related_training_run": f"training-{experiment_id}",
        },
        tags=["demo2", "inference", experiment_id],
    )

    # Get run URLs
    train_url = ray.get(training_actor.get_url.remote())
    inference_url = ray.get(inference_actor.get_url.remote())
    print(f"Training run (rl-agent):      {train_url}")
    print(f"Inference run (inference-only): {inference_url}")
    print()

    # Create workers
    training_workers = [
        TrainingWorker.remote(worker_id=i) for i in range(config.num_training_workers)
    ]
    inference_workers = [
        InferenceWorker.remote(worker_id=i) for i in range(config.num_inference_workers)
    ]

    print("Running training and inference...")
    print("-" * 70)
    print()
    print("Shared metrics (for cross-project comparison):")
    print("  - e2e/speed        (tokens/sec)")
    print("  - e2e/latency_ms   (milliseconds)")
    print("  - throughput       (items/step)")
    print()

    for step in range(config.num_steps):
        # Training step
        training_futures = [w.do_training_step.remote(step) for w in training_workers]
        training_results = ray.get(training_futures)

        avg_loss = sum(r["loss"] for r in training_results) / len(training_results)
        avg_grad_norm = sum(r["grad_norm_total"] for r in training_results) / len(
            training_results
        )
        # Compute training throughput (tokens per second)
        training_speed = training_results[0]["tokens_processed"] / max(1, step + 1) * 10

        ray.get(
            training_actor.log_metrics.remote(
                {
                    # === SHARED METRICS (same names as inference project) ===
                    "e2e/speed": training_speed,  # tokens/sec during training
                    "e2e/latency_ms": 1000.0 / max(1, training_speed) * 100,  # simulated
                    "throughput": config.num_training_workers * 32,  # batches * batch_size
                    # === TRAINING-SPECIFIC METRICS ===
                    "training/loss": avg_loss,
                    "training/grad_norm": avg_grad_norm,
                    "training/learning_rate": training_results[0]["learning_rate"],
                    "training/tokens_processed": training_results[0]["tokens_processed"],
                },
                step=step,
            )
        )

        # Inference step
        inference_futures = [w.do_inference_step.remote(step) for w in inference_workers]
        inference_results = ray.get(inference_futures)

        avg_reward = sum(r["episode_reward"] for r in inference_results) / len(
            inference_results
        )
        avg_speed = sum(r["e2e_speed"] for r in inference_results) / len(inference_results)

        ray.get(
            inference_actor.log_metrics.remote(
                {
                    # === SHARED METRICS (same names as training project) ===
                    "e2e/speed": avg_speed,  # tokens/sec during inference
                    "e2e/latency_ms": inference_results[0]["e2e_latency_ms"],
                    "throughput": config.num_inference_workers * inference_results[0]["rollouts_completed"],
                    # === INFERENCE-SPECIFIC METRICS ===
                    "inference/reward": avg_reward,
                    "inference/success_rate": inference_results[0]["success_rate"],
                    "inference/episode_length": inference_results[0]["episode_length"],
                },
                step=step,
            )
        )

        print(f"Step {step:3d}: loss={avg_loss:.4f}, reward={avg_reward:.2f}, e2e/speed: train={training_speed:.0f}, infer={avg_speed:.0f}")

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
    print(f"\nExperiment ID: {experiment_id}")
    print()
    print("W&B Projects:")
    print(f"  rl-agent:        {train_url}")
    print(f"  inference-only:  {inference_url}")
    print()
    print("Metric structure in BOTH projects:")
    print("  ├── e2e/           <- SHARED (for cross-project comparison)")
    print("  │   ├── speed")
    print("  │   └── latency_ms")
    print("  ├── throughput     <- SHARED")
    print("  ├── training/      <- Only in rl-agent")
    print("  │   ├── loss")
    print("  │   └── grad_norm")
    print("  └── inference/     <- Only in inference-only")
    print("      ├── reward")
    print("      └── success_rate")
    print()
    print("Key observations:")
    print("  - SHARED metrics (e2e/speed, throughput) appear in BOTH projects")
    print("  - Pipeline-specific metrics use prefixes (training/, inference/)")
    print("  - W&B auto-creates panel sections for e2e/, training/, inference/")
    print()
    print("Cross-project comparison:")
    print("  1. Create a W&B Report")
    print("  2. Add panels from both 'rl-agent' and 'inference-only' projects")
    print("  3. Compare 'e2e/speed' side-by-side since names match")
    print("  4. Filter by experiment_id tag to find related runs")


if __name__ == "__main__":
    config = DemoConfig(
        num_steps=20,
        num_training_workers=2,
        num_inference_workers=2,
        step_delay=0.3,
    )
    run_demo(config)

