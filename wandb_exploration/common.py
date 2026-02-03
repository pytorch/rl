"""
Common backbone for W&B logging exploration demos.

Provides:
- WandbRun: Thread-safe wrapper around a Ray actor that owns a W&B run
- TrainingPipelineActor / InferencePipelineActor: Centralized logging actors
- Dummy metric generators for realistic simulation
"""

import random
import time
from dataclasses import dataclass

import ray
import wandb


# -----------------------------------------------------------------------------
# WandbRun: Proxy pattern for thread-safe remote W&B logging
# -----------------------------------------------------------------------------


@ray.remote(num_cpus=0, num_gpus=0)
class _WandbActor:
    """Ray actor that owns a single W&B run.

    This allows multiple W&B runs to coexist across different Ray actors,
    each in their own process.
    """

    def __init__(
        self,
        project: str,
        name: str,
        config: dict | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
        entity: str | None = None,
    ) -> None:
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            group=group,
            job_type=job_type,
            entity=entity,
            reinit=True,
        )
        self._step_counter = 0

    def log(self, metrics: dict, step: int | None = None) -> None:
        if step is None:
            step = self._step_counter
            self._step_counter += 1
        else:
            self._step_counter = max(self._step_counter, step + 1)
        self.run.log(metrics, step=step)

    def log_artifact(self, name: str, artifact_type: str, file_path: str) -> None:
        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    def log_artifact_from_dict(
        self, name: str, artifact_type: str, data: dict, filename: str
    ) -> None:
        """Log artifact from dict data (serialized as JSON)."""
        import json
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(temp_path, name=filename)
        self.run.log_artifact(artifact)
        os.remove(temp_path)

    def get_url(self) -> str:
        return self.run.get_url() or ""

    def get_run_id(self) -> str:
        return self.run.id

    def finish(self) -> None:
        self.run.finish()


class WandbRun:
    """Thread-safe wrapper for a W&B run via a Ray actor.

    Usage:
        run = WandbRun(project="my-project", name="my-run")
        run.log({"loss": 0.5}, step=0)
        run.finish()
    """

    def __init__(
        self,
        project: str,
        name: str,
        config: dict | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
        entity: str | None = None,
    ) -> None:
        self._actor = _WandbActor.remote(
            project=project,
            name=name,
            config=config,
            tags=tags,
            group=group,
            job_type=job_type,
            entity=entity,
        )
        # Wait for init to complete
        ray.get(self._actor.get_run_id.remote())

    def log(self, metrics: dict, step: int | None = None) -> None:
        ray.get(self._actor.log.remote(metrics, step))

    def log_async(self, metrics: dict, step: int | None = None) -> ray.ObjectRef:
        """Non-blocking log that returns an ObjectRef."""
        return self._actor.log.remote(metrics, step)

    def log_artifact(self, name: str, artifact_type: str, file_path: str) -> None:
        ray.get(self._actor.log_artifact.remote(name, artifact_type, file_path))

    def log_artifact_from_dict(
        self, name: str, artifact_type: str, data: dict, filename: str
    ) -> None:
        ray.get(
            self._actor.log_artifact_from_dict.remote(name, artifact_type, data, filename)
        )

    def get_url(self) -> str:
        return ray.get(self._actor.get_url.remote())

    def get_run_id(self) -> str:
        return ray.get(self._actor.get_run_id.remote())

    def finish(self) -> None:
        ray.get(self._actor.finish.remote())


# -----------------------------------------------------------------------------
# Dummy metric generators
# -----------------------------------------------------------------------------


def generate_training_metrics(step: int) -> dict:
    """Generate realistic dummy training metrics."""
    # Loss decays over time with some noise
    base_loss = 2.0 / (step + 1) + 0.1
    noise = random.gauss(0, 0.05)
    loss = max(0.01, base_loss + noise)

    # Gradient norms vary randomly
    grad_norm_0 = random.uniform(0.1, 2.0)
    grad_norm_1 = random.uniform(0.05, 1.5)
    grad_norm_total = (grad_norm_0**2 + grad_norm_1**2) ** 0.5

    # Learning rate with warmup then decay
    warmup_steps = 5
    if step < warmup_steps:
        lr = 1e-4 * (step + 1) / warmup_steps
    else:
        lr = 1e-4 * (0.95 ** (step - warmup_steps))

    return {
        "loss": loss,
        "grad_norm_0": grad_norm_0,
        "grad_norm_1": grad_norm_1,
        "grad_norm_total": grad_norm_total,
        "learning_rate": lr,
        "batch_size": 32,
        "tokens_processed": (step + 1) * 32 * 512,
    }


def generate_inference_metrics(step: int) -> dict:
    """Generate realistic dummy inference metrics."""
    # Episode reward increases over time (learning)
    base_reward = step * 0.5 + random.gauss(0, 2.0)
    episode_reward = max(-10, min(100, base_reward))

    # Speed metrics with some variance
    tokens_per_sec = random.uniform(1000, 3000)
    latency_ms = random.uniform(50, 200)

    # Rollout statistics
    episode_length = random.randint(10, 100)
    success_rate = min(1.0, 0.3 + step * 0.03 + random.gauss(0, 0.1))

    return {
        "episode_reward": episode_reward,
        "e2e_speed": tokens_per_sec,
        "e2e_latency_ms": latency_ms,
        "episode_length": episode_length,
        "success_rate": max(0, min(1, success_rate)),
        "rollouts_completed": step + 1,
    }


# -----------------------------------------------------------------------------
# Pipeline Actors with centralized logging
# -----------------------------------------------------------------------------


@ray.remote(num_cpus=0, num_gpus=0)
class TrainingPipelineActor:
    """Actor that owns the training W&B run and aggregates metrics from workers."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
        metric_prefix: str = "",
    ) -> None:
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=tags,
            group=group,
            job_type=job_type,
            reinit=True,
        )
        self._step_counter = 0
        self._metric_prefix = metric_prefix
        self._aggregated_metrics: dict[str, list[float]] = {}
        print(f"[TrainingPipelineActor] W&B run: {self.run.get_url()}")

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics with optional prefix."""
        if self._metric_prefix:
            prefixed = {f"{self._metric_prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed = metrics

        if step is None:
            step = self._step_counter
            self._step_counter += 1
        else:
            self._step_counter = max(self._step_counter, step + 1)

        self.run.log(prefixed, step=step)
        print(f"[TrainingPipelineActor] Logged step {step}: {list(prefixed.keys())}")

    def aggregate_worker_metrics(self, worker_id: int, metrics: dict) -> None:
        """Aggregate metrics from a worker for batch logging."""
        for key, value in metrics.items():
            if key not in self._aggregated_metrics:
                self._aggregated_metrics[key] = []
            self._aggregated_metrics[key].append(value)

    def flush_aggregated_metrics(self, step: int) -> None:
        """Compute averages of aggregated metrics and log them."""
        if not self._aggregated_metrics:
            return

        averaged = {}
        for key, values in self._aggregated_metrics.items():
            averaged[f"avg_{key}"] = sum(values) / len(values)
            averaged[f"max_{key}"] = max(values)
            averaged[f"min_{key}"] = min(values)

        self.log_metrics(averaged, step=step)
        self._aggregated_metrics.clear()

    def log_artifact(self, name: str, artifact_type: str, data: dict, filename: str) -> None:
        """Log artifact from dict data."""
        import json
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(temp_path, name=filename)
        self.run.log_artifact(artifact)
        os.remove(temp_path)
        print(f"[TrainingPipelineActor] Logged artifact: {name}")

    def get_url(self) -> str:
        return self.run.get_url() or ""

    def finish(self) -> None:
        self.run.finish()
        print("[TrainingPipelineActor] W&B run finished")


@ray.remote(num_cpus=0, num_gpus=0)
class InferencePipelineActor:
    """Actor that owns the inference W&B run and aggregates metrics from workers."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
        metric_prefix: str = "",
    ) -> None:
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=tags,
            group=group,
            job_type=job_type,
            reinit=True,
        )
        self._step_counter = 0
        self._metric_prefix = metric_prefix
        self._aggregated_metrics: dict[str, list[float]] = {}
        print(f"[InferencePipelineActor] W&B run: {self.run.get_url()}")

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics with optional prefix."""
        if self._metric_prefix:
            prefixed = {f"{self._metric_prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed = metrics

        if step is None:
            step = self._step_counter
            self._step_counter += 1
        else:
            self._step_counter = max(self._step_counter, step + 1)

        self.run.log(prefixed, step=step)
        print(f"[InferencePipelineActor] Logged step {step}: {list(prefixed.keys())}")

    def aggregate_worker_metrics(self, worker_id: int, metrics: dict) -> None:
        """Aggregate metrics from a worker for batch logging."""
        for key, value in metrics.items():
            if key not in self._aggregated_metrics:
                self._aggregated_metrics[key] = []
            self._aggregated_metrics[key].append(value)

    def flush_aggregated_metrics(self, step: int) -> None:
        """Compute averages of aggregated metrics and log them."""
        if not self._aggregated_metrics:
            return

        averaged = {}
        for key, values in self._aggregated_metrics.items():
            averaged[f"avg_{key}"] = sum(values) / len(values)
            averaged[f"max_{key}"] = max(values)
            averaged[f"min_{key}"] = min(values)

        self.log_metrics(averaged, step=step)
        self._aggregated_metrics.clear()

    def log_artifact(self, name: str, artifact_type: str, data: dict, filename: str) -> None:
        """Log artifact from dict data."""
        import json
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(temp_path, name=filename)
        self.run.log_artifact(artifact)
        os.remove(temp_path)
        print(f"[InferencePipelineActor] Logged artifact: {name}")

    def get_url(self) -> str:
        return self.run.get_url() or ""

    def finish(self) -> None:
        self.run.finish()
        print("[InferencePipelineActor] W&B run finished")


# -----------------------------------------------------------------------------
# Worker actors that send metrics to centralized loggers
# -----------------------------------------------------------------------------


@ray.remote(num_cpus=0, num_gpus=0)
class TrainingWorker:
    """Simulates a training worker that generates metrics and sends to aggregator."""

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id

    def do_training_step(self, step: int) -> dict:
        """Simulate a training step and return metrics."""
        # Simulate some work
        time.sleep(random.uniform(0.05, 0.15))
        metrics = generate_training_metrics(step)
        metrics["worker_id"] = self.worker_id
        return metrics


@ray.remote(num_cpus=0, num_gpus=0)
class InferenceWorker:
    """Simulates an inference worker that generates metrics and sends to aggregator."""

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id

    def do_inference_step(self, step: int) -> dict:
        """Simulate an inference step and return metrics."""
        # Simulate some work
        time.sleep(random.uniform(0.05, 0.15))
        metrics = generate_inference_metrics(step)
        metrics["worker_id"] = self.worker_id
        return metrics


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def timestamp() -> str:
    """Generate a timestamp string for run naming."""
    return time.strftime("%Y%m%d_%H%M%S")


def init_ray_if_needed() -> None:
    """Initialize Ray if not already initialized."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        print("[common] Ray initialized")


@dataclass
class DemoConfig:
    """Configuration for demo runs."""

    num_steps: int = 20
    num_training_workers: int = 2
    num_inference_workers: int = 2
    step_delay: float = 0.5  # Seconds between steps for visualization
    entity: str | None = None  # W&B entity (team/user)

