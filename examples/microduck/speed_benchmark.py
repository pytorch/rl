# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compare synchronous and asynchronous recurrent PPO on nine MicroDuck skills."""

from __future__ import annotations

import argparse
import functools as ft
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import tensordict
import torch

from examples.microduck.ppo_mujoco import (
    evaluation_metrics,
    make_env,
    make_evaluator,
    make_models,
    make_tasks,
    make_video_env,
    record_task_grid,
    save_checkpoint,
)
from tensordict import TensorDictBase
from torchrl.checkpoint import Checkpoint, GlobalRNGState
from torchrl.collectors import MultiAsyncCollector, MultiCollector
from torchrl.data import LazyTensorStorage, RateLimitedReplayBuffer, SliceSampler
from torchrl.envs import MicroDuckEnv, StepCounter, TrajCounter, Transform
from torchrl.modules import get_primers_from_module, set_recurrent_mode
from torchrl.objectives import ClipPPOLoss, SoftUpdate
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers import WandbLogger
from torchrl.trainers import BatchSubSampler, TargetNetUpdaterHook
from torchrl.trainers.algorithms import PPOTrainer


Mode = Literal["ppo", "ppo-ewma", "semi-async", "full-async"]

# This ordering is part of the learned task embedding and matches the selected
# Hugging Face nine-skill checkpoint exactly.
SKILL_PRESETS = [
    {"preset": "standing_task", "reward_weights": {"head_level": 4.0}},
    {
        "preset": "tracking_task",
        "speed": 0.2,
        "reward_weights": {
            "tracking": 3.0,
            "yaw_rate": 2.0,
            "head_level": 2.0,
            "termination": -20.0,
        },
        "head_level_std": 0.7,
    },
    {
        "preset": "tracking_task",
        "speed": -0.2,
        "reward_weights": {"tracking": 3.0, "yaw_rate": 2.0, "head_level": 2.0},
        "head_level_std": 0.7,
    },
    {
        "preset": "sidestep_task",
        "speed": 0.15,
        "reward_weights": {"yaw_rate": 2.0, "head_level": 2.0},
        "head_level_std": 0.7,
    },
    {
        "preset": "sidestep_task",
        "speed": -0.15,
        "reward_weights": {"yaw_rate": 2.0, "head_level": 2.0},
        "head_level_std": 0.7,
    },
    {
        "preset": "jump_task",
        "speed": 0.3,
        "weight": 3.0,
        "reward_weights": {"tracking": 3.0, "yaw_rate": 2.0, "head_level": 4.0},
        "head_level_std": 0.7,
    },
    {
        "preset": "turning_task",
        "rate": 1.0,
        "turn_rate_std": 1.0,
        "reward_weights": {"turn": 4.0, "head_level": 2.0},
        "head_level_std": 0.7,
    },
    {
        "preset": "turning_task",
        "rate": -1.0,
        "turn_rate_std": 1.0,
        "reward_weights": {"turn": 4.0, "head_level": 2.0},
        "head_level_std": 0.7,
    },
    {
        "preset": "jump_task",
        "weight": 3.0,
        "reward_weights": {"drift": -15.0, "head_level": 4.0},
        "drift_speed_scale": 1.0,
        "head_level_std": 0.7,
    },
]


class NamespacedWandbLogger(WandbLogger):
    """Keep application metrics out of W&B's ungrouped root chart namespace."""

    @staticmethod
    def _name(name: str) -> str:
        if name.startswith(("training/", "evaluation/", "system/")):
            return name
        return f"training/{name.lstrip('/')}"

    def log_scalar(self, name: str, value: float, step: int | None = None, **kwargs):
        return super().log_scalar(self._name(name), value, step=step, **kwargs)

    def log_video(self, name: str, video: torch.Tensor, **kwargs):
        return super().log_video(self._name(name), video, **kwargs)

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        return super().log_histogram(self._name(name), data, **kwargs)

    def log_str(self, name: str, value: str, step: int | None = None, **kwargs):
        return super().log_str(self._name(name), value, step=step, **kwargs)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        *,
        keys_sep: str = "/",
        **kwargs,
    ) -> dict[str, Any]:
        metrics = {self._name(name): value for name, value in metrics.items()}
        if any(name.startswith("training/rewards/") for name in metrics):
            metrics["training/wall_clock_seconds"] = time.monotonic() - self.started_at
        return super().log_metrics(
            metrics,
            step=step,
            keys_sep=keys_sep,
            **kwargs,
        )

    def _define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        if name.startswith("training/rewards/") and not name.endswith("/step"):
            step_metric = "training/wall_clock_seconds"
        elif name.startswith("evaluation/") and name.endswith("/reward"):
            step_metric = "evaluation/wall_clock_seconds"
        super()._define_metric(name, step_metric=step_metric)


class BenchmarkMonitor:
    """Evaluate, record videos and save the highest-reward policy during training."""

    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        env_config: Mapping[str, Any],
        policy_kwargs: Mapping[str, Any],
        logger: NamespacedWandbLogger,
        output_dir: Path,
        evaluation_interval: int,
        video_interval: int,
        evaluation_episodes: int,
        evaluation_steps: int,
    ):
        self.actor = actor
        self.critic = critic
        self.env_config = dict(env_config)
        self.policy_kwargs = dict(policy_kwargs)
        self.logger = logger
        self.output_dir = output_dir
        self.evaluation_interval = evaluation_interval
        self.video_interval = video_interval
        self.evaluation_steps = evaluation_steps
        self.started_at = logger.started_at
        self.next_evaluation = 0
        self.next_video = 0
        self.last_evaluation: int | None = None
        self.best_score: float | None = None
        self.trainer: PPOTrainer | None = None

        tasks = make_tasks(SKILL_PRESETS)
        jump_index = list(MicroDuckEnv.REWARD_TERMS).index("jump")
        self.evaluators = []
        for task_id, task in enumerate(tasks):
            eval_config = {
                **self.env_config,
                "task_id": task_id,
                "diagnostics": True,
                "num_envs": 1,
                "parallel": False,
                "seed": 20260914,
            }
            self.evaluators.append(
                make_evaluator(
                    make_env(eval_config),
                    actor,
                    label=f"skill_{task_id:02d}",
                    jumping=bool(task.reward_weights[jump_index] > 0),
                    num_episodes=evaluation_episodes,
                    steps=evaluation_steps,
                )
            )

        self.video_recorder = VideoRecorder(
            logger,
            tag="evaluation/video/nine_skills",
            make_grid=True,
            fps=50,
            skip=1,
            max_frames=evaluation_steps,
        )
        self.video_env = make_video_env(
            self.env_config,
            list(range(len(SKILL_PRESETS))),
            recorder=self.video_recorder,
            width=160,
            height=120,
        )
        self.video_env.append_transform(get_primers_from_module(actor))

    def register(self, trainer: PPOTrainer) -> None:
        self.trainer = trainer
        trainer.register_op("setup", self.setup)
        trainer.register_op("post_steps", self.post_steps)
        trainer.register_op("shutdown", self.shutdown)

    def _checkpoint(self, name: str, frame: int, metrics: Mapping[str, Any]) -> None:
        save_checkpoint(
            self.output_dir / name,
            self.actor,
            self.critic,
            transitions=frame,
            policy_kwargs=self.policy_kwargs,
            metrics=metrics,
            config={"env": self.env_config},
        )

    def evaluate(self, frame: int, *, final: bool = False) -> None:
        results = [
            evaluator.evaluate(weights=self.actor, step=frame)
            for evaluator in self.evaluators
        ]
        metrics = evaluation_metrics(results)
        metrics.update(
            {
                "evaluation/frames": frame,
                "evaluation/wall_clock_seconds": time.monotonic() - self.started_at,
                "evaluation/final": float(final),
            }
        )
        score = metrics["evaluation/reward"]
        is_best = self.best_score is None or score > self.best_score
        metrics["evaluation/is_best"] = float(is_best)
        self._checkpoint("latest.ckpt", frame, metrics)
        if is_best:
            self.best_score = score
            self._checkpoint("best.ckpt", frame, metrics)
        self.logger.log_metrics(metrics, step=frame)
        self.last_evaluation = frame
        while self.next_evaluation <= frame:
            self.next_evaluation += self.evaluation_interval

        if frame >= self.next_video or final:
            record_task_grid(
                self.video_env,
                self.video_recorder,
                self.actor,
                steps=self.evaluation_steps,
                step=frame,
            )
            while self.next_video <= frame:
                self.next_video += self.video_interval

    def setup(self) -> None:
        if self.last_evaluation is None:
            self.evaluate(0)

    def post_steps(self) -> None:
        frame = int(self.trainer.collected_frames)
        self.logger.log_metrics(
            {
                "training/frames": frame,
                "training/wall_clock_seconds": time.monotonic() - self.started_at,
            },
            step=frame,
        )
        if frame >= self.next_evaluation:
            self.evaluate(frame)

    def shutdown(self) -> None:
        try:
            frame = int(self.trainer.collected_frames)
            if self.trainer.async_collection:
                frame = max(
                    frame,
                    int(self.trainer.collector.getattr_rb("write_count")),
                )
                self.trainer.collected_frames = frame
            if self.last_evaluation != frame:
                self.evaluate(frame, final=True)
        finally:
            for evaluator in self.evaluators:
                evaluator.shutdown()
            if not self.video_env.is_closed:
                self.video_env.close()


class FirstBatchRecorder:
    """Persist the first collected batch layout to verify resolved collection."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "first_batch.json"

    def __call__(self, batch):
        if self.path.exists():
            return batch
        mask = batch.get(("collector", "mask"), None)
        payload = {
            "batch_size": list(batch.batch_size),
            "numel": batch.numel(),
            "has_mask": mask is not None,
            "valid_per_row": None if mask is None else mask.sum(-1).tolist(),
            "trajectory_ids": int(
                batch.get(("collector", "traj_ids")).unique().numel()
            ),
        }
        self.path.write_text(json.dumps(payload, indent=2))
        return batch


class FlattenEnvironmentBatch:
    """Collapse native worker and lane dimensions while preserving time."""

    def __call__(self, batch):
        if batch.ndim > 2:
            return batch.flatten(0, batch.ndim - 2)
        return batch


class WorkerTrajectoryCounter(TrajCounter):
    """Give every worker a disjoint arithmetic progression of trajectory ids."""

    def __init__(self, worker_id: int, num_workers: int):
        super().__init__(out_key="global_traj_id")
        self.worker_id = worker_id
        self.num_workers = num_workers

    def _reset(self, tensordict, tensordict_reset):
        tensordict_reset = super()._reset(tensordict, tensordict_reset)
        trajectory_id = tensordict_reset.get("global_traj_id")
        tensordict_reset.set(
            "global_traj_id",
            trajectory_id * self.num_workers + self.worker_id,
        )
        return tensordict_reset


class AccumulatingCollector:
    """Yield global PPO batches while an asynchronous collector keeps running."""

    def __init__(self, collector: MultiAsyncCollector, batch_size: int):
        self.collector = collector
        self.batch_size = batch_size

    def __getattr__(self, name: str):
        return getattr(self.collector, name)

    def __iter__(self):
        chunks = []
        transitions = 0
        for batch in self.collector:
            if batch.ndim == 1:
                batch = batch.unsqueeze(0)
            elif batch.ndim > 2:
                batch = batch.flatten(0, batch.ndim - 2)
            chunks.append(batch)
            transitions += batch.numel()
            if transitions < self.batch_size:
                continue
            if transitions != self.batch_size:
                raise RuntimeError(
                    "Semi-async worker chunks must divide frames_per_batch exactly: "
                    f"collected {transitions} for target {self.batch_size}."
                )
            yield torch.cat(chunks, dim=0)
            chunks = []
            transitions = 0


class AsyncReplayBatch:
    """Draw a fresh recurrent replay sample and compute boundary-safe GAE."""

    def __init__(
        self,
        replay: RateLimitedReplayBuffer,
        gae: GAE,
        output_dir: Path,
    ):
        self.replay = replay
        self.gae = gae
        self.first_sample_path = output_dir / "first_batch.json"
        self.sample_min_version = 0
        self.sample_max_version = 0

    def __call__(self, batch: TensorDictBase | None) -> TensorDictBase:
        del batch
        try:
            sample, info = self.replay.sample(return_info=True, wait=True, timeout=1.0)
        except TimeoutError as error:
            raise StopIteration from error
        # Generic replay returns SliceSampler's padding mask and boundaries as
        # separate metadata; GAE and PPO consume them from the sampled TensorDict.
        info["index"] = torch.stack(info["index"], dim=-1)
        sample.update(info)
        slice_end = sample.get(("collector", "slice_end"))
        sample.set(("next", "done"), sample.get(("next", "done")) | slice_end)
        versions = sample.get(("next", "policy_version"))
        self.sample_min_version = int(versions.min())
        self.sample_max_version = int(versions.max())
        with torch.no_grad():
            self.gae(sample)
        if not self.first_sample_path.exists():
            mask = sample.get(("collector", "mask"))
            self.first_sample_path.write_text(
                json.dumps(
                    {
                        "batch_size": list(sample.batch_size),
                        "numel": sample.numel(),
                        "valid_transitions": int(mask.sum()),
                        "trajectory_ids": int(
                            sample.get("global_traj_id").unique().numel()
                        ),
                        "sample_min_policy_version": self.sample_min_version,
                        "sample_max_policy_version": self.sample_max_version,
                    },
                    indent=2,
                )
            )
        return sample


class AsyncReplayDiagnostics:
    """Log replay pacing and policy-sample age after each asynchronous update."""

    def __init__(
        self,
        sampler: AsyncReplayBatch,
        replay: RateLimitedReplayBuffer,
        logger: NamespacedWandbLogger,
    ):
        self.sampler = sampler
        self.replay = replay
        self.logger = logger
        self.trainer: PPOTrainer | None = None

    def register(self, trainer: PPOTrainer) -> None:
        self.trainer = trainer
        trainer.register_op("post_steps", self)

    def __call__(self) -> None:
        flow = self.replay.stats()
        metrics = {
            f"training/replay_flow/{name}": value
            for name, value in flow.items()
            if isinstance(value, (int, float, bool)) and value is not None
        }
        current_version = int(self.trainer._optim_count)
        metrics.update(
            {
                "training/policy/sample_min_version": self.sampler.sample_min_version,
                "training/policy/sample_max_version": self.sampler.sample_max_version,
                "training/policy/sample_version_span": (
                    self.sampler.sample_max_version - self.sampler.sample_min_version
                ),
                "training/policy/oldest_sample_age_updates": max(
                    0, current_version - self.sampler.sample_min_version
                ),
            }
        )
        self.logger.log_metrics(metrics, step=int(self.trainer.collected_frames))


def make_loss_gae_optimizer(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    *,
    learning_rate: float,
    ewma: bool,
    max_importance_ratio: float,
    valid_key=None,
):
    """Build the shared recurrent PPO learner components for overlap modes."""
    loss_kwargs = {
        "clip_epsilon": 0.2,
        "entropy_coeff": 0.01,
        "critic_coeff": 1.0,
        "loss_critic_type": "smooth_l1",
        "normalize_advantage": False,
    }
    if ewma:
        loss_kwargs.update(
            delay_actor=True,
            max_importance_ratio=max_importance_ratio,
        )
    loss = ClipPPOLoss(actor, critic, **loss_kwargs)
    loss.set_keys(
        action="action",
        reward="reward",
        done="done",
        terminated="terminated",
        value="state_value",
    )
    gae = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
        average_gae=True,
        group_key="task_id",
        shifted=False,
        deactivate_vmap=True,
    )
    gae.set_keys(
        reward="reward",
        done="done",
        terminated="terminated",
        value="state_value",
        **({"valid": valid_key} if valid_key is not None else {}),
    )
    gae = set_recurrent_mode(True)(gae)
    optimizer = torch.optim.Adam(loss.parameters(), lr=learning_rate)
    return loss, gae, optimizer


def make_worker_factories(
    env_config: Mapping[str, Any], num_workers: int, *, policy: torch.nn.Module
) -> list[Any]:
    """Split the environment batch evenly among process collectors."""
    if num_workers < 1 or env_config["num_envs"] % num_workers:
        raise ValueError("num_envs must be divisible by the positive collector count.")
    envs_per_worker = env_config["num_envs"] // num_workers
    worker_config = {
        **env_config,
        "num_envs": envs_per_worker,
        "parallel": envs_per_worker > 1,
    }
    return [
        ft.partial(
            make_worker_env,
            worker_config,
            worker_id=worker_id,
            num_workers=num_workers,
            policy_transform=get_primers_from_module(policy),
        )
        for worker_id in range(num_workers)
    ]


def make_worker_env(
    env_config: Mapping[str, Any],
    *,
    worker_id: int,
    num_workers: int,
    policy_transform: Transform,
):
    """Build one process-collector env with explicit trajectory step metadata."""
    env = make_env(env_config)
    # MultiCollector allocates shared replay storage from the factory's specs
    # before worker-side automatic policy transforms are attached.
    env.append_transform(policy_transform.clone())
    env.append_transform(StepCounter(max_steps=env_config["max_episode_steps"]))
    env.append_transform(WorkerTrajectoryCounter(worker_id, num_workers))
    return env


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the complete experiment configuration recorded in W&B and on disk."""
    config = {
        "campaign": args.campaign,
        "mode": args.mode,
        "source_revision": args.source_revision,
        "tensordict_version": tensordict.__version__,
        "tensordict_source": tensordict.__file__,
        "from_scratch": True,
        "env": {
            "backend": "mujoco",
            "device": "cpu",
            "download": True,
            "num_envs": args.num_envs,
            "parallel": args.num_envs > 1,
            "max_episode_steps": args.episode_steps,
            "action_scale": 1.0,
            "tasks": SKILL_PRESETS,
            "seed": args.seed,
        },
        "policy": {
            "architecture": "shared GRU trunk, specialized actor/critic heads",
            "hidden_size": 128,
            "policy_head": "gaussian",
            "initial_policy_scale": 1.0,
        },
        "ppo": {
            "total_frames": args.frames,
            "frames_per_batch": args.frames_per_batch,
            "minibatch_size": args.minibatch_size,
            "sub_traj_len": args.sub_traj_len,
            "epochs": 1 if args.mode == "full-async" else args.epochs,
            "learning_rate": args.learning_rate,
            "clip_epsilon": 0.2,
            "entropy_coeff": 0.01,
            "critic_coeff": 1.0,
            "loss_critic_type": "smooth_l1",
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "gae_normalization": "per task_id",
            "gradient_clip_norm": 1.0,
            "ewma": args.mode != "ppo",
            "ewma_decay": args.ewma_decay if args.mode != "ppo" else None,
            "max_importance_ratio": (
                args.max_importance_ratio if args.mode != "ppo" else None
            ),
        },
        "evaluation": {
            "seed": 20260914,
            "interval_frames": args.evaluation_interval,
            "episodes_per_skill": args.evaluation_episodes,
            "steps": args.episode_steps,
            "video_interval_frames": args.video_interval,
            "video_fps": 50,
            "video_frame_skip": 1,
        },
        "wandb": {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "base_url": args.wandb_base_url,
            "offline": args.wandb_offline,
        },
    }
    if args.mode == "semi-async":
        config["collection"] = {
            "collector": "MultiAsyncCollector",
            "workers": args.collector_workers or args.num_envs,
            "environments_per_worker": args.num_envs
            // (args.collector_workers or args.num_envs),
            "worker_chunk": args.collector_chunk,
            "accumulated_batch": args.frames_per_batch,
            "overlap": "collect next global batch during PPO-EWMA epochs",
        }
    elif args.mode == "full-async":
        config["collection"] = {
            "collector": "MultiCollector(sync=False).start",
            "workers": args.collector_workers or args.num_envs,
            "environments_per_worker": args.num_envs
            // (args.collector_workers or args.num_envs),
            "worker_chunk": args.collector_chunk,
            "replay_write_mode": "trajectory",
            "completed_trajectories_only": True,
            "replay_capacity": args.replay_capacity,
            "replay_buffer": "RateLimitedReplayBuffer",
            "replay_storage_ndim": 1,
            "sampler": "SliceSampler(fragmented=True)",
            "sample_without_replacement": False,
            "replacement_reason": (
                "without-replacement trajectory caches are unsafe while worker "
                "processes mutate shared circular storage"
            ),
            "num_slices": args.minibatch_size // args.sub_traj_len,
            "slice_len": args.sub_traj_len,
            "structured_batch_time": True,
            "samples_per_insert": args.samples_per_insert,
            "learner_epochs": 1,
            "learner_updates_per_loop": 1,
        }
    return config


def run(args: argparse.Namespace) -> None:
    """Run one benchmark variant from a fresh initialization."""
    started_at = time.monotonic()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = resolved_config(args)
    (output_dir / "run.json").write_text(json.dumps(config, indent=2))

    logger = NamespacedWandbLogger(
        exp_name=f"{args.campaign}-{args.mode}",
        offline=args.wandb_offline,
        save_dir=output_dir,
        project=args.wandb_project,
        base_url=args.wandb_base_url,
        entity=args.wandb_entity,
        group=args.campaign,
        job_type=args.mode,
        config=config,
        tags=["microduck", "speed-benchmark", args.mode],
    )
    logger.started_at = started_at
    run_info = {
        **config,
        "wandb_run_id": logger.experiment.id,
        "wandb_run_url": logger.experiment.url,
    }
    (output_dir / "run.json").write_text(json.dumps(run_info, indent=2))

    env_config = config["env"]
    policy_kwargs = {
        "hidden_size": 128,
        "policy_head": "gaussian",
        "initial_policy_scale": 1.0,
    }
    trainer_kwargs = {
        "total_frames": args.frames,
        "frame_skip": 1,
        "progress_bar": False,
        "logger": logger,
        "log_interval": max(1, args.frames_per_batch - 1),
        "log_timings": True,
        "telemetry": "standard",
        "checkpoint": Checkpoint(rng=GlobalRNGState()),
        "save_trainer_file": output_dir / "trainer.ckpt",
        "save_trainer_interval": args.evaluation_interval,
        "seed": args.seed,
        "clip_norm": 1.0,
    }
    if args.mode in ("ppo", "ppo-ewma"):
        env = make_env(env_config)
        actor, critic = make_models(env, **policy_kwargs)
        loss_kwargs = {
            "clip_epsilon": 0.2,
            "entropy_coeff": 0.01,
            "critic_coeff": 1.0,
            "loss_critic_type": "smooth_l1",
        }
        if args.mode == "ppo-ewma":
            loss_kwargs.update(
                delay_actor=True,
                max_importance_ratio=args.max_importance_ratio,
            )
        from_env_kwargs = dict(trainer_kwargs)
        from_env_kwargs.pop("frame_skip")
        trainer = PPOTrainer.from_env(
            env,
            actor=actor,
            critic=critic,
            frames_per_batch=args.frames_per_batch,
            minibatch_size=args.minibatch_size,
            sub_traj_len=args.sub_traj_len,
            learning_rate=args.learning_rate,
            gae_kwargs={"average_gae": True, "group_key": "task_id"},
            loss_kwargs=loss_kwargs,
            num_epochs=args.epochs,
            gamma=0.99,
            lmbda=0.95,
            **from_env_kwargs,
        )
        if args.mode == "ppo-ewma":
            updater = SoftUpdate(trainer.loss_module, eps=args.ewma_decay)
            trainer.target_net_updater = updater
            trainer.register_module("target_updater", updater)
            trainer.register_op("post_optim", TargetNetUpdaterHook(updater))
        trainer.register_op("batch_process", FlattenEnvironmentBatch())
        trainer.register_op("batch_process", FirstBatchRecorder(output_dir))
    else:
        model_env = make_env({**env_config, "num_envs": 1, "parallel": False})
        actor, critic = make_models(model_env, **policy_kwargs)
        factories = make_worker_factories(
            env_config, args.collector_workers or args.num_envs, policy=actor
        )
        if args.mode == "semi-async":
            if args.frames_per_batch % args.collector_chunk:
                raise ValueError(
                    "collector_chunk must divide frames_per_batch in semi-async mode."
                )
            collector = MultiAsyncCollector(
                create_env_fn=factories,
                policy=actor,
                frames_per_batch=args.collector_chunk,
                total_frames=-1,
                storing_device="cpu",
                auto_register_policy_transforms=True,
                track_policy_version=True,
            )
            collector = AccumulatingCollector(collector, args.frames_per_batch)
            loss, gae, optimizer = make_loss_gae_optimizer(
                actor,
                critic,
                learning_rate=args.learning_rate,
                ewma=True,
                max_importance_ratio=args.max_importance_ratio,
            )
            updater = SoftUpdate(loss, eps=args.ewma_decay)
            trainer = PPOTrainer(
                collector=collector,
                loss_module=loss,
                optimizer=optimizer,
                gae=gae,
                replay_buffer=None,
                target_net_updater=updater,
                optim_steps_per_batch=args.frames_per_batch // args.minibatch_size,
                num_epochs=args.epochs,
                weight_update_map={"policy": "loss_module.actor_network"},
                **trainer_kwargs,
            )
            BatchSubSampler(
                args.minibatch_size, sub_traj_len=args.sub_traj_len
            ).register(trainer)
            trainer.register_op("batch_process", FirstBatchRecorder(output_dir))
        else:
            # MultiCollector cannot currently serve state_dict requests while
            # paused. The evaluator still writes renderable latest/best policy
            # checkpoints, which are the artifacts needed by this benchmark.
            trainer_kwargs["save_trainer_file"] = None
            replay = RateLimitedReplayBuffer(
                storage=LazyTensorStorage(args.replay_capacity, ndim=1),
                sampler=SliceSampler(
                    num_slices=args.minibatch_size // args.sub_traj_len,
                    traj_key="global_traj_id",
                    step_key="step_count",
                    fragmented=True,
                    strict_length=False,
                    output_layout="batch_time",
                ),
                batch_size=args.minibatch_size,
                samples_per_insert=args.samples_per_insert,
            )
            collector = MultiCollector(
                sync=False,
                create_env_fn=factories,
                policy=actor,
                frames_per_batch=args.collector_chunk,
                total_frames=-1,
                storing_device="cpu",
                replay_buffer=replay,
                replay_write_mode="trajectory",
                trajs_per_write=1,
                auto_register_policy_transforms=True,
                track_policy_version=True,
            )
            loss, gae, optimizer = make_loss_gae_optimizer(
                actor,
                critic,
                learning_rate=args.learning_rate,
                ewma=True,
                max_importance_ratio=args.max_importance_ratio,
                valid_key=("collector", "mask"),
            )
            updater = SoftUpdate(loss, eps=args.ewma_decay)
            trainer = PPOTrainer(
                collector=collector,
                loss_module=loss,
                optimizer=optimizer,
                gae=None,
                add_gae=False,
                replay_buffer=None,
                target_net_updater=updater,
                optim_steps_per_batch=1,
                num_epochs=1,
                async_collection=True,
                weight_update_map={"policy": "loss_module.actor_network"},
                **trainer_kwargs,
            )
            trainer.replay_buffer = replay
            replay_batch = AsyncReplayBatch(replay, gae, output_dir)
            trainer.register_op("process_optim_batch", replay_batch)
            AsyncReplayDiagnostics(replay_batch, replay, logger).register(trainer)
        model_env.close()

    monitor = BenchmarkMonitor(
        actor=actor,
        critic=critic,
        env_config=env_config,
        policy_kwargs=policy_kwargs,
        logger=logger,
        output_dir=output_dir,
        evaluation_interval=args.evaluation_interval,
        video_interval=args.video_interval,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_steps=args.episode_steps,
    )
    if args.mode == "full-async":
        monitor.evaluate(0)
    monitor.register(trainer)
    exit_code = 0
    try:
        with trainer.stop_on_signal():
            trainer.train()
    except BaseException:
        exit_code = 1
        raise
    finally:
        try:
            logger.experiment.summary["training/final_frames"] = int(
                trainer.collected_frames
            )
            logger.experiment.summary["training/stop_reason"] = (
                trainer._stop_reason or "completed"
            )
        finally:
            logger.experiment.finish(exit_code=exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("ppo", "ppo-ewma", "semi-async", "full-async"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign", default="microduck-speed")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--frames", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--collector-workers", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--frames-per-batch", type=int, default=16_384)
    parser.add_argument("--minibatch-size", type=int, default=2_048)
    parser.add_argument("--sub-traj-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ewma-decay", type=float, default=8 / 9)
    parser.add_argument("--max-importance-ratio", type=float, default=10.0)
    parser.add_argument("--collector-chunk", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=18_432)
    parser.add_argument("--samples-per-insert", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--evaluation-interval", type=int, default=1_000_000)
    parser.add_argument("--video-interval", type=int, default=5_000_000)
    parser.add_argument("--evaluation-episodes", type=int, default=1)
    parser.add_argument("--wandb-project", default="torchrl-microduck-speed-bench")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-base-url", default="https://api.wandb.ai")
    parser.add_argument("--wandb-offline", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
