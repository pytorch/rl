#!/usr/bin/env python3
"""
Demo 4: Advanced Extensions

This demo combines all patterns with advanced W&B features:
- job_type metadata for filtering runs (job_type="training" or "inference")
- group parameter to link training and inference runs by experiment ID
- Artifact logging (model checkpoints, rollout trajectories)
- Cross-project comparison with consistent metric naming
- Tags for additional filtering

Key patterns demonstrated:
- Using job_type for run categorization within a project
- Using group to create experiment "families" of runs
- Centralized artifact logging through pipeline actors
- Consistent metric naming (e2e/speed) for cross-project reports

Usage:
    uv run python personal/vincent/wandb_exploration/demo4_extensions.py
"""

import json
import random
import tempfile
import time

import ray

from common import (
    DemoConfig,
    InferencePipelineActor,
    InferenceWorker,
    TrainingPipelineActor,
    TrainingWorker,
    generate_training_metrics,
    init_ray_if_needed,
    timestamp,
)


def generate_dummy_checkpoint(step: int) -> dict:
    """Generate dummy model checkpoint data."""
    return {
        "step": step,
        "model_state": {
            "layer_0_weight_mean": random.gauss(0, 0.1),
            "layer_0_weight_std": abs(random.gauss(0.5, 0.1)),
            "layer_1_weight_mean": random.gauss(0, 0.1),
            "layer_1_weight_std": abs(random.gauss(0.5, 0.1)),
        },
        "optimizer_state": {
            "learning_rate": 1e-4 * (0.95**step),
            "beta1": 0.9,
            "beta2": 0.999,
        },
        "training_metrics": generate_training_metrics(step),
    }


def generate_dummy_trajectory(step: int, num_rollouts: int = 5) -> dict:
    """Generate dummy rollout trajectory data."""
    return {
        "step": step,
        "num_rollouts": num_rollouts,
        "rollouts": [
            {
                "episode_id": f"ep_{step}_{i}",
                "reward": random.gauss(step * 0.5, 2.0),
                "length": random.randint(10, 100),
                "actions": [random.randint(0, 3) for _ in range(10)],
                "success": random.random() > 0.5,
            }
            for i in range(num_rollouts)
        ],
        "summary": {
            "mean_reward": step * 0.5,
            "success_rate": 0.3 + step * 0.03,
        },
    }


def run_demo(config: DemoConfig) -> None:
    """Run the advanced extensions demo."""
    init_ray_if_needed()

    run_ts = timestamp()
    experiment_id = f"exp-{run_ts}"
    seed = 42

    print("=" * 70)
    print("Demo 4: Advanced Extensions")
    print("=" * 70)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"Seed: {seed}")
    print(f"Steps: {config.num_steps}")
    print()
    print("Features demonstrated:")
    print("  - job_type metadata for run filtering")
    print("  - group parameter to link related runs")
    print("  - Artifact logging (checkpoints, trajectories)")
    print("  - Consistent metric naming for cross-project comparison")
    print("  - Tags for additional filtering")
    print()

    # Training actor with job_type and group
    training_actor = TrainingPipelineActor.remote(
        project="rl-agent-extended",
        run_name=f"train-{experiment_id}",
        config={
            "experiment_id": experiment_id,
            "seed": seed,
            "pipeline": "training",
            "model": "gpt-rl-v1",
            "learning_rate": 1e-4,
            "batch_size": 32,
        },
        tags=["demo4", "extended", f"seed-{seed}", experiment_id],
        group=experiment_id,  # Links this run to other runs with same experiment_id
        job_type="training",  # Allows filtering by job type in W&B UI
    )

    # Inference actor with job_type and group
    inference_actor = InferencePipelineActor.remote(
        project="rl-agent-extended",  # Same project
        run_name=f"inference-{experiment_id}",
        config={
            "experiment_id": experiment_id,
            "seed": seed,
            "pipeline": "inference",
            "model": "gpt-rl-v1",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        tags=["demo4", "extended", f"seed-{seed}", experiment_id],
        group=experiment_id,  # Same group links to training run
        job_type="inference",  # Different job_type for filtering
    )

    train_url = ray.get(training_actor.get_url.remote())
    inference_url = ray.get(inference_actor.get_url.remote())
    print(f"Training run:  {train_url}")
    print(f"Inference run: {inference_url}")
    print(f"\nGroup: {experiment_id}")
    print("  -> Both runs will appear together in W&B's grouped view")
    print()

    # Create workers
    training_workers = [
        TrainingWorker.remote(worker_id=i) for i in range(config.num_training_workers)
    ]
    inference_workers = [
        InferenceWorker.remote(worker_id=i) for i in range(config.num_inference_workers)
    ]

    print("Running with artifact logging...")
    print("-" * 70)

    checkpoint_interval = 5  # Log checkpoint every 5 steps
    trajectory_interval = 3  # Log trajectory every 3 steps

    for step in range(config.num_steps):
        # Training step
        training_futures = [w.do_training_step.remote(step) for w in training_workers]
        training_results = ray.get(training_futures)

        avg_loss = sum(r["loss"] for r in training_results) / len(training_results)
        avg_grad_norm = sum(r["grad_norm_total"] for r in training_results) / len(
            training_results
        )

        # Log training metrics
        ray.get(
            training_actor.log_metrics.remote(
                {
                    "loss": avg_loss,
                    "grad_norm": avg_grad_norm,
                    "learning_rate": training_results[0]["learning_rate"],
                    "tokens_processed": training_results[0]["tokens_processed"],
                    # Cross-project comparable metric with consistent naming
                    "e2e/tokens_per_step": training_results[0]["tokens_processed"]
                    / (step + 1),
                },
                step=step,
            )
        )

        # Log checkpoint artifact periodically
        if step > 0 and step % checkpoint_interval == 0:
            checkpoint_data = generate_dummy_checkpoint(step)
            ray.get(
                training_actor.log_artifact.remote(
                    name=f"checkpoint-step-{step}",
                    artifact_type="model-checkpoint",
                    data=checkpoint_data,
                    filename=f"checkpoint_{step}.json",
                )
            )
            print(f"  [artifact] Logged checkpoint at step {step}")

        # Inference step
        inference_futures = [w.do_inference_step.remote(step) for w in inference_workers]
        inference_results = ray.get(inference_futures)

        avg_reward = sum(r["episode_reward"] for r in inference_results) / len(
            inference_results
        )
        avg_speed = sum(r["e2e_speed"] for r in inference_results) / len(inference_results)

        # Log inference metrics with consistent naming for cross-project comparison
        ray.get(
            inference_actor.log_metrics.remote(
                {
                    "reward/mean": avg_reward,
                    "reward/max": max(r["episode_reward"] for r in inference_results),
                    "success_rate": inference_results[0]["success_rate"],
                    # Cross-project comparable metrics (same name as training)
                    "e2e/speed": avg_speed,
                    "e2e/latency_ms": inference_results[0]["e2e_latency_ms"],
                },
                step=step,
            )
        )

        # Log trajectory artifact periodically
        if step > 0 and step % trajectory_interval == 0:
            trajectory_data = generate_dummy_trajectory(step)
            ray.get(
                inference_actor.log_artifact.remote(
                    name=f"trajectories-step-{step}",
                    artifact_type="rollout-trajectories",
                    data=trajectory_data,
                    filename=f"trajectories_{step}.json",
                )
            )
            print(f"  [artifact] Logged trajectories at step {step}")

        print(f"Step {step:3d}: loss={avg_loss:.4f}, reward={avg_reward:.2f}")

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
    print(f"\nProject: rl-agent-extended")
    print(f"Group: {experiment_id}")
    print(f"Training run:  {train_url}")
    print(f"Inference run: {inference_url}")
    print()
    print("Advanced features used:")
    print()
    print("1. job_type metadata:")
    print("   - Training run: job_type='training'")
    print("   - Inference run: job_type='inference'")
    print("   -> Filter in W&B UI: 'Job Type' dropdown or query 'job_type: training'")
    print()
    print("2. Group parameter:")
    print(f"   - Both runs share group='{experiment_id}'")
    print("   -> In W&B UI: View -> Group by -> Group")
    print("   -> Shows training and inference runs together as an experiment family")
    print()
    print("3. Artifact logging:")
    print(f"   - Model checkpoints (every {checkpoint_interval} steps)")
    print(f"   - Rollout trajectories (every {trajectory_interval} steps)")
    print("   -> View in run's 'Artifacts' tab")
    print("   -> Track lineage across training iterations")
    print()
    print("4. Consistent metric naming:")
    print("   - Both runs log 'e2e/speed' and similar metrics")
    print("   -> Create W&B Report comparing across runs")
    print("   -> Query: e2e/speed for side-by-side comparison")
    print()
    print("5. Tags:")
    print(f"   - Common tags: ['demo4', 'extended', 'seed-{seed}', '{experiment_id}']")
    print("   -> Filter runs by tag in W&B UI sidebar")
    print()
    print("Cross-project Report tips:")
    print("  - Create a Report and add panels from multiple projects")
    print("  - Use consistent metric names (e2e/speed) for comparison")
    print("  - Filter by tags or group to find related runs")


if __name__ == "__main__":
    config = DemoConfig(
        num_steps=20,
        num_training_workers=2,
        num_inference_workers=2,
        step_delay=0.3,
    )
    run_demo(config)

