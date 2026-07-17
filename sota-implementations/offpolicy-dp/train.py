# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run an off-policy Trainer with Ray collection, replay, and DDP learners."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import ray
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector, Evaluator
from torchrl.collectors.distributed import RayCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    GymEnv,
    HopperEnv,
    HumanoidEnv,
    InitTracker,
    RewardSum,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    AdditiveGaussianModule,
    EGreedyModule,
    MLP,
    ProbabilisticActor,
    QValueActor,
    TanhModule,
    ValueOperator,
)
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import (
    DDPGLoss,
    DQNLoss,
    HardUpdate,
    SACLoss,
    SoftUpdate,
    TD3Loss,
)
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.record import VideoRecorder
from torchrl.record.loggers import get_logger, Logger
from torchrl.trainers import Trainer
from torchrl.trainers.algorithms import DDPGTrainer, DQNTrainer, SACTrainer, TD3Trainer
from torchrl.trainers.algorithms.td3 import TD3OptimizationStepper
from torchrl.trainers.trainers import OptimizationStepper


@dataclass
class AlgorithmObjects:
    """Objects whose connected parameter graph is copied to learner ranks."""

    policy: nn.Module
    evaluation_policy: nn.Module
    loss_module: LossModule
    optimizer: optim.Optimizer | None
    target_updater: TargetNetUpdater
    trainer_cls: type[Trainer]
    optimization_stepper: OptimizationStepper | None = None
    trainer_kwargs: dict[str, Any] = field(default_factory=dict)


class _ContinueVideoAfterTermination:
    """Keep a diagnostic rollout alive after the task termination condition."""

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        return torch.zeros_like(super()._compute_done(state, next_state))


class _HopperVideoEnv(_ContinueVideoAfterTermination, HopperEnv):
    """Hopper environment used only for diagnostic video recording."""


class _HumanoidVideoEnv(_ContinueVideoAfterTermination, HumanoidEnv):
    """Humanoid environment used only for diagnostic video recording."""


_CONTINUOUS_ENV_CLASSES = {
    "hopper": HopperEnv,
    "humanoid": HumanoidEnv,
}
_CONTINUOUS_VIDEO_ENV_CLASSES = {
    "hopper": _HopperVideoEnv,
    "humanoid": _HumanoidVideoEnv,
}
_EVALUATION_ENV_LOCK = threading.Lock()


def _continuous_env_class(
    environment_name: str, *, record_video: bool = False
) -> type[HopperEnv] | type[HumanoidEnv]:
    classes = _CONTINUOUS_VIDEO_ENV_CLASSES if record_video else _CONTINUOUS_ENV_CLASSES
    try:
        return classes[environment_name]
    except KeyError as err:
        raise ValueError(
            f"Unsupported continuous environment {environment_name!r}; "
            f"expected one of {sorted(classes)}."
        ) from err


def _wandb_metric_name(name: str) -> str:
    """Place scalar histories in stable W&B workspace sections."""
    if "/" in name:
        return name
    if "." in name:
        return name.replace(".", "/")
    exact_names = {
        "alpha": "policy/alpha",
        "entropy": "policy/entropy",
        "grad_norm": "optimization/grad_norm",
        "optim_steps": "optimization/steps",
        "td_error": "value/td_error",
    }
    mapped = exact_names.get(name)
    if mapped is not None:
        return mapped
    prefixes = (
        ("loss_", "loss/"),
        ("target_value_", "value/target_"),
        ("pred_value_", "value/predicted_"),
        ("value_", "value/"),
        ("action_value_", "value/action_"),
    )
    for source, destination in prefixes:
        if name.startswith(source):
            return destination + name.removeprefix(source)
    if name == "loss":
        return "loss/total"
    return f"training/{name}"


class NamespacedLogger:
    """Serialize logger calls and group unscoped learner metrics in W&B."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._lock = threading.RLock()

    @property
    def experiment(self) -> Any:
        return self._logger.experiment

    def client(self) -> NamespacedLogger:
        return self

    def log_metrics(
        self,
        metrics: dict[str, Any] | TensorDictBase,
        step: int | None = None,
        *,
        keys_sep: str = "/",
    ) -> dict[str, Any]:
        if isinstance(metrics, TensorDictBase):
            metrics = metrics.flatten_keys(keys_sep).to_dict()
        grouped = {_wandb_metric_name(key): value for key, value in metrics.items()}
        with self._lock:
            return self._logger.log_metrics(grouped, step=step, keys_sep=keys_sep)

    def log_scalar(
        self, name: str, value: float, step: int | None = None, **kwargs: Any
    ) -> None:
        with self._lock:
            self._logger.log_scalar(
                _wandb_metric_name(name), value, step=step, **kwargs
            )

    def log_video(
        self, name: str, video: torch.Tensor, step: int | None = None, **kwargs: Any
    ) -> None:
        with self._lock:
            self._logger.log_video(_wandb_metric_name(name), video, step=step, **kwargs)

    def log_histogram(self, name: str, data: Any, **kwargs: Any) -> None:
        with self._lock:
            self._logger.log_histogram(_wandb_metric_name(name), data, **kwargs)

    def log_hparams(self, cfg: DictConfig | dict[str, Any]) -> None:
        with self._lock:
            self._logger.log_hparams(cfg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)


def flatten_replay_postproc(data: TensorDictBase) -> TensorDictBase:
    """Materialize a flat transition batch for direct Ray replay insertion."""
    num_transitions = data.numel()
    data = data.exclude(
        "action_value",
        "chosen_action_value",
        "loc",
        "param",
        "scale",
    ).reshape(-1)
    # Collector rollouts reuse preallocated storage. Materialization keeps Ray from
    # serializing the complete backing allocation through a flattened tensor view.
    data = data.clone()
    if data.ndim != 1 or data.numel() != num_transitions:
        raise RuntimeError(
            "Off-policy replay postprocessing must flatten [B, T] rollouts to "
            f"[B * T]; got batch_size={data.batch_size}."
        )
    return data


def make_continuous_env(
    *,
    environment_name: str,
    backend: str,
    num_envs: int,
    device: str,
    compile_step: bool,
    compile_mode: str | None,
    frame_skip: int | None,
    max_episode_steps: int,
    seed: int,
) -> TransformedEnv:
    """Create one natively batched MuJoCo environment inside a Ray actor."""
    compile_kwargs = None if compile_mode is None else {"mode": compile_mode}
    env_cls = _continuous_env_class(environment_name)
    env = env_cls(
        backend=backend,
        num_envs=num_envs,
        device=torch.device(device),
        seed=seed,
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        compile_step=compile_step,
        compile_kwargs=compile_kwargs,
        dtype=torch.float32,
    )
    return TransformedEnv(env, Compose(InitTracker(), RewardSum()))


def make_cartpole_env(*, env_name: str, seed: int) -> TransformedEnv:
    """Create one CPU CartPole environment inside a Ray actor."""
    env = TransformedEnv(
        GymEnv(env_name, device="cpu"),
        Compose(InitTracker(), RewardSum()),
    )
    env.set_seed(seed)
    return env


def _materialize(module: nn.Module, proof_env: TransformedEnv) -> None:
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        module(proof_env.fake_tensordict())


def _make_dqn(cfg: DictConfig, proof_env: TransformedEnv) -> AlgorithmObjects:
    action_spec = proof_env.action_spec_unbatched
    observation_dim = proof_env.observation_spec["observation"].shape[-1]
    action_dim = action_spec.shape[-1]
    qvalue_module = TensorDictModule(
        MLP(
            in_features=observation_dim,
            out_features=action_dim,
            num_cells=list(cfg.algorithm.hidden_sizes),
            activation_class=nn.ReLU,
        ),
        in_keys=["observation"],
        out_keys=["action_value"],
    )
    value_network = QValueActor(
        qvalue_module,
        in_keys=["observation"],
        spec=action_spec,
    )
    _materialize(value_network, proof_env)
    greedy_module = EGreedyModule(
        spec=action_spec,
        eps_init=float(cfg.algorithm.eps_init),
        eps_end=float(cfg.algorithm.eps_end),
        annealing_num_steps=int(cfg.algorithm.annealing_num_steps),
    )
    policy = TensorDictSequential(value_network, greedy_module)
    loss_module = DQNLoss(
        value_network,
        action_space=action_spec,
        delay_value=True,
        loss_function="smooth_l1",
    )
    loss_module.make_value_estimator(gamma=float(cfg.algorithm.gamma))
    optimizer = optim.Adam(
        loss_module.parameters(), lr=float(cfg.algorithm.learning_rate)
    )
    target_updater = HardUpdate(
        loss_module,
        value_network_update_interval=int(cfg.algorithm.target_update_interval),
    )
    return AlgorithmObjects(
        policy=policy,
        evaluation_policy=value_network,
        loss_module=loss_module,
        optimizer=optimizer,
        target_updater=target_updater,
        trainer_cls=DQNTrainer,
        trainer_kwargs={"greedy_module": greedy_module},
    )


def _continuous_specs(proof_env: TransformedEnv) -> tuple[int, int, Any]:
    observation_dim = proof_env.observation_spec["observation"].shape[-1]
    action_spec = proof_env.action_spec_unbatched
    action_dim = action_spec.shape[-1]
    return observation_dim, action_dim, action_spec


def _make_stochastic_actor(
    observation_dim: int,
    action_dim: int,
    action_spec: Any,
    hidden_sizes: list[int],
) -> ProbabilisticActor:
    actor_net = nn.Sequential(
        MLP(
            in_features=observation_dim,
            out_features=2 * action_dim,
            num_cells=hidden_sizes,
            activation_class=nn.ReLU,
        ),
        NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4),
    )
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    return ProbabilisticActor(
        module=actor_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )


def _make_deterministic_actor(
    observation_dim: int,
    action_dim: int,
    action_spec: Any,
    hidden_sizes: list[int],
) -> TensorDictSequential:
    return TensorDictSequential(
        TensorDictModule(
            MLP(
                in_features=observation_dim,
                out_features=action_dim,
                num_cells=hidden_sizes,
                activation_class=nn.ReLU,
            ),
            in_keys=["observation"],
            out_keys=["param"],
        ),
        TanhModule(
            in_keys=["param"],
            out_keys=["action"],
            spec=action_spec,
        ),
    )


def _make_qvalue(
    observation_dim: int,
    action_dim: int,
    hidden_sizes: list[int],
) -> ValueOperator:
    return ValueOperator(
        module=MLP(
            in_features=observation_dim + action_dim,
            out_features=1,
            num_cells=hidden_sizes,
            activation_class=nn.ReLU,
        ),
        in_keys=["action", "observation"],
    )


def _make_sac(cfg: DictConfig, proof_env: TransformedEnv) -> AlgorithmObjects:
    observation_dim, action_dim, action_spec = _continuous_specs(proof_env)
    hidden_sizes = list(cfg.algorithm.hidden_sizes)
    actor = _make_stochastic_actor(
        observation_dim, action_dim, action_spec, hidden_sizes
    )
    qvalue = _make_qvalue(observation_dim, action_dim, hidden_sizes)
    _materialize(actor, proof_env)
    _materialize(qvalue, proof_env)
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        loss_function="smooth_l1",
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=float(cfg.algorithm.alpha_init),
        action_spec=action_spec,
    )
    loss_module.make_value_estimator(gamma=float(cfg.algorithm.gamma))
    optimizer = optim.Adam(
        loss_module.parameters(), lr=float(cfg.algorithm.learning_rate)
    )
    target_updater = SoftUpdate(
        loss_module, eps=float(cfg.algorithm.target_update_polyak)
    )
    return AlgorithmObjects(
        policy=actor,
        evaluation_policy=actor,
        loss_module=loss_module,
        optimizer=optimizer,
        target_updater=target_updater,
        trainer_cls=SACTrainer,
    )


def _make_ddpg(cfg: DictConfig, proof_env: TransformedEnv) -> AlgorithmObjects:
    observation_dim, action_dim, action_spec = _continuous_specs(proof_env)
    hidden_sizes = list(cfg.algorithm.hidden_sizes)
    actor = _make_deterministic_actor(
        observation_dim, action_dim, action_spec, hidden_sizes
    )
    qvalue = _make_qvalue(observation_dim, action_dim, hidden_sizes)
    _materialize(actor, proof_env)
    _materialize(qvalue, proof_env)
    exploration_module = AdditiveGaussianModule(
        spec=action_spec,
        sigma_init=1.0,
        sigma_end=1.0,
        mean=0.0,
        std=float(cfg.algorithm.exploration_std),
        safe=False,
    )
    policy = TensorDictSequential(actor, exploration_module)
    loss_module = DDPGLoss(
        actor_network=actor,
        value_network=qvalue,
        loss_function="smooth_l1",
        delay_actor=True,
        delay_value=True,
    )
    loss_module.make_value_estimator(gamma=float(cfg.algorithm.gamma))
    optimizer = optim.Adam(
        loss_module.parameters(), lr=float(cfg.algorithm.learning_rate)
    )
    target_updater = SoftUpdate(
        loss_module, eps=float(cfg.algorithm.target_update_polyak)
    )
    return AlgorithmObjects(
        policy=policy,
        evaluation_policy=actor,
        loss_module=loss_module,
        optimizer=optimizer,
        target_updater=target_updater,
        trainer_cls=DDPGTrainer,
        trainer_kwargs={"exploration_module": exploration_module},
    )


def _make_td3(cfg: DictConfig, proof_env: TransformedEnv) -> AlgorithmObjects:
    observation_dim, action_dim, action_spec = _continuous_specs(proof_env)
    hidden_sizes = list(cfg.algorithm.hidden_sizes)
    actor = _make_deterministic_actor(
        observation_dim, action_dim, action_spec, hidden_sizes
    )
    qvalue = _make_qvalue(observation_dim, action_dim, hidden_sizes)
    _materialize(actor, proof_env)
    _materialize(qvalue, proof_env)
    exploration_module = AdditiveGaussianModule(
        spec=action_spec,
        sigma_init=1.0,
        sigma_end=1.0,
        mean=0.0,
        std=float(cfg.algorithm.exploration_std),
        safe=False,
    )
    policy = TensorDictSequential(actor, exploration_module)
    loss_module = TD3Loss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        loss_function="smooth_l1",
        delay_actor=True,
        delay_qvalue=True,
        action_spec=action_spec,
        policy_noise=float(cfg.algorithm.policy_noise),
        noise_clip=float(cfg.algorithm.noise_clip),
    )
    loss_module.make_value_estimator(gamma=float(cfg.algorithm.gamma))
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    optimizer_actor = optim.Adam(actor_params, lr=float(cfg.algorithm.learning_rate))
    optimizer_critic = optim.Adam(critic_params, lr=float(cfg.algorithm.learning_rate))
    optimization_stepper = TD3OptimizationStepper(
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        policy_update_delay=int(cfg.algorithm.policy_update_delay),
    )
    target_updater = SoftUpdate(
        loss_module, eps=float(cfg.algorithm.target_update_polyak)
    )
    return AlgorithmObjects(
        policy=policy,
        evaluation_policy=actor,
        loss_module=loss_module,
        optimizer=None,
        target_updater=target_updater,
        trainer_cls=TD3Trainer,
        optimization_stepper=optimization_stepper,
        trainer_kwargs={"exploration_module": exploration_module},
    )


def _make_algorithm(cfg: DictConfig, proof_env: TransformedEnv) -> AlgorithmObjects:
    builders = {
        "dqn": _make_dqn,
        "sac": _make_sac,
        "ddpg": _make_ddpg,
        "td3": _make_td3,
    }
    try:
        builder = builders[str(cfg.algorithm.name)]
    except KeyError as err:
        raise ValueError(f"Unsupported algorithm {cfg.algorithm.name!r}.") from err
    return builder(cfg, proof_env)


def _make_env_factories(cfg: DictConfig) -> list[partial]:
    num_collectors = int(cfg.environment.num_collectors)
    total_num_envs = int(cfg.environment.total_num_envs)
    if total_num_envs % num_collectors:
        raise ValueError("total_num_envs must be divisible by num_collectors.")
    envs_per_collector = total_num_envs // num_collectors
    if cfg.algorithm.environment_kind == "gym":
        if envs_per_collector != 1:
            raise ValueError("The Gym DQN profile requires one env per Ray collector.")
        return [
            partial(
                make_cartpole_env,
                env_name=str(cfg.algorithm.environment_name),
                seed=int(cfg.seed) + worker,
            )
            for worker in range(num_collectors)
        ]
    return [
        partial(
            make_continuous_env,
            environment_name=str(cfg.algorithm.environment_name),
            backend=str(cfg.environment.backend),
            num_envs=envs_per_collector,
            device="cuda",
            compile_step=bool(cfg.environment.compile_step),
            compile_mode=cfg.environment.compile_mode,
            frame_skip=(
                None
                if cfg.environment.frame_skip is None
                else int(cfg.environment.frame_skip)
            ),
            max_episode_steps=int(cfg.environment.max_episode_steps),
            seed=int(cfg.seed) + worker,
        )
        for worker in range(num_collectors)
    ]


def _make_proof_env(cfg: DictConfig) -> TransformedEnv:
    if cfg.algorithm.environment_kind == "gym":
        return make_cartpole_env(
            env_name=str(cfg.algorithm.environment_name), seed=int(cfg.seed)
        )
    return make_continuous_env(
        environment_name=str(cfg.algorithm.environment_name),
        backend=str(cfg.environment.backend),
        num_envs=1,
        device="cpu",
        compile_step=False,
        compile_mode=None,
        frame_skip=(
            None
            if cfg.environment.frame_skip is None
            else int(cfg.environment.frame_skip)
        ),
        max_episode_steps=int(cfg.environment.max_episode_steps),
        seed=int(cfg.seed),
    )


def _make_evaluation_env_unlocked(
    cfg: DictConfig,
    logger: NamespacedLogger,
    *,
    record_video: bool = False,
) -> TransformedEnv:
    """Create a metric or video evaluation environment."""
    transforms = [InitTracker(), RewardSum()]
    if record_video:
        transforms.append(
            VideoRecorder(
                logger,
                tag="evaluation/video",
                in_keys=["pixels"],
                skip=int(cfg.evaluation.video_skip),
                fps=int(cfg.evaluation.video_fps),
                max_frames=int(cfg.evaluation.video_max_frames),
                make_grid=True,
            )
        )
    if cfg.algorithm.environment_kind == "gym":
        env = GymEnv(
            str(cfg.algorithm.environment_name),
            device="cpu",
            from_pixels=record_video,
            pixels_only=False,
        )
    else:
        env_cls = _continuous_env_class(
            str(cfg.algorithm.environment_name), record_video=record_video
        )
        num_envs = (
            int(cfg.evaluation.video_num_envs)
            if record_video
            else int(cfg.evaluation.num_envs)
        )
        max_steps = (
            int(cfg.evaluation.video_max_steps)
            if record_video
            else int(cfg.evaluation.max_steps)
        )
        env = env_cls(
            backend=str(cfg.evaluation.backend),
            num_envs=num_envs,
            device=torch.device(str(cfg.evaluation.device)),
            seed=int(cfg.seed) + 10_000,
            frame_skip=(
                None
                if cfg.evaluation.frame_skip is None
                else int(cfg.evaluation.frame_skip)
            ),
            max_episode_steps=max_steps,
            compile_step=bool(cfg.evaluation.compile_step),
            dtype=torch.float32,
            from_pixels=record_video,
            pixels_only=False,
            render_width=int(cfg.evaluation.render_width),
            render_height=int(cfg.evaluation.render_height),
            render_every=int(cfg.evaluation.video_skip),
        )
    eval_env = TransformedEnv(env, Compose(*transforms))
    eval_env.set_seed(int(cfg.seed) + 10_000)
    return eval_env


def _make_evaluation_env(
    cfg: DictConfig,
    logger: NamespacedLogger,
    *,
    record_video: bool = False,
) -> TransformedEnv:
    """Create one metric or video environment at a time."""
    # mujoco-torch environment construction invokes lazy wrappers that cannot
    # be initialised concurrently by the metric and video evaluator threads.
    with _EVALUATION_ENV_LOCK:
        return _make_evaluation_env_unlocked(cfg, logger, record_video=record_video)


def _make_evaluation_policy(
    env: TransformedEnv,
    *,
    policy: nn.Module,
    device: str,
) -> nn.Module:
    """Build the evaluation policy lazily beside its run environment."""
    del env
    return copy.deepcopy(policy).to(device)


def _initialize_evaluator(evaluator: Evaluator) -> None:
    """Materialize a thread evaluator before asynchronous training starts."""
    # Environment construction alone is insufficient: Collector construction
    # resets the environment and also touches lazy environment specifications.
    evaluator._backend._ensure_collector()


def _ray_init_config(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg.ray, resolve=True)  # type: ignore[return-value]


def _make_replay_transport_specs(
    cfg: DictConfig,
    policy: nn.Module,
    proof_env: TransformedEnv,
) -> tuple[TensorDictBase | None, TensorDictBase | None]:
    if str(cfg.replay.transport) != "distributed":
        return None, None
    num_collectors = int(cfg.environment.num_collectors)
    envs_per_collector = int(cfg.environment.total_num_envs) // num_collectors
    frames_per_batch = envs_per_collector * int(cfg.collection.frames_per_env)
    fake_collector = Collector(
        proof_env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch,
        init_random_frames=0,
        device="cpu",
        storing_device="cpu",
        env_device="cpu",
        policy_device="cpu",
        exploration_type=ExplorationType.RANDOM,
        postproc=flatten_replay_postproc,
        track_policy_version=True,
        return_same_td=True,
    )
    try:
        extend_spec = fake_collector.fake_tensordict()
    finally:
        fake_collector.shutdown(close_env=False)

    world_size = int(cfg.learner.world_size)
    global_batch_size = int(cfg.replay.batch_size)
    if global_batch_size % world_size:
        raise ValueError("replay.batch_size must be divisible by learner.world_size.")
    local_batch_size = global_batch_size // world_size
    sample_spec = extend_spec[:local_batch_size].clone()
    sample_spec.set("index", torch.zeros(local_batch_size, dtype=torch.int64))
    # Consolidating the schemas makes their one-time Ray bootstrap a single
    # compact storage rather than one torch.save payload per TensorDict leaf.
    return extend_spec.consolidate(), sample_spec.consolidate()


def _make_replay(
    cfg: DictConfig,
    extend_spec: TensorDictBase | None = None,
    sample_spec: TensorDictBase | None = None,
) -> TensorDictReplayBuffer:
    if str(cfg.replay.storage_mode) != "transitions":
        raise ValueError(
            "DQN, SAC, DDPG, and TD3 require replay.storage_mode=transitions. "
            "Grouped sequence replay must be configured by a workload that "
            "explicitly supports sequence samples."
        )
    transport = str(cfg.replay.transport)
    transport_options = None
    if transport == "distributed":
        transport_options = {
            "backend": str(cfg.replay.transport_backend),
            "timeout": float(cfg.replay.transport_timeout),
            "extend_spec": extend_spec,
            "sample_spec": sample_spec,
        }
    return TensorDictReplayBuffer(
        storage=partial(
            LazyTensorStorage,
            int(cfg.replay.capacity),
            device="cpu",
        ),
        batch_size=int(cfg.replay.batch_size),
        service_backend="ray",
        service_backend_options={
            "ray_init_config": _ray_init_config(cfg),
            "remote_config": {"num_cpus": int(cfg.replay.num_cpus)},
        },
        transport=transport,
        transport_options=transport_options,
    )


def _make_collector(
    cfg: DictConfig,
    policy: nn.Module,
    replay: TensorDictReplayBuffer,
) -> RayCollector:
    num_collectors = int(cfg.environment.num_collectors)
    envs_per_collector = int(cfg.environment.total_num_envs) // num_collectors
    frames_per_batch = envs_per_collector * int(cfg.collection.frames_per_env)
    use_gpu = cfg.algorithm.environment_kind == "mujoco"
    device = "cuda" if use_gpu else "cpu"
    remote_configs = {
        "num_cpus": 1,
        "num_gpus": 1 if use_gpu else 0,
    }
    collector = RayCollector(
        create_env_fn=_make_env_factories(cfg),
        policy=policy,
        replay_buffer=replay,
        collector_class="single",
        collector_kwargs={
            # Direct replay is owned by the inner collectors. The outer
            # RayCollector postprocessor never sees these batches.
            "postproc": flatten_replay_postproc,
            "track_policy_version": True,
        },
        frames_per_batch=frames_per_batch,
        total_frames=int(cfg.collection.total_frames),
        init_random_frames=int(cfg.collection.init_random_frames),
        device=device,
        storing_device="cpu",
        env_device=device,
        policy_device=device,
        exploration_type=ExplorationType.RANDOM,
        remote_configs=remote_configs,
        ray_init_config=_ray_init_config(cfg),
        sync=False,
    )
    # Random-only collection and replay prefill are separate concerns. The
    # inner collectors receive init_random_frames above, while the controller
    # uses this threshold to delay learning until stochastic-policy data has
    # filled replay. This also avoids per-collector random-frame thresholds
    # being compared against the shared replay write count.
    collector.init_random_frames = int(cfg.collection.learning_starts)
    return collector


def _git_commit() -> str:
    configured = os.environ.get("TORCHRL_GIT_COMMIT")
    if configured:
        return configured
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return "unknown"


def _make_logger(cfg: DictConfig, run_name: str, commit: str) -> Logger:
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    config["git_commit"] = commit
    logger = get_logger(
        "wandb",
        logger_name=str(cfg.artifacts_dir),
        experiment_name=run_name,
        wandb_kwargs={
            "project": str(cfg.logging.project),
            "group": str(cfg.logging.group),
            "mode": str(cfg.logging.mode),
            "config": config,
            "log_env_packages": bool(cfg.logging.log_env_packages),
            "tags": [
                "offpolicy-dp",
                str(cfg.algorithm.name),
                f"learners-{cfg.learner.world_size}",
                f"collectors-{cfg.environment.num_collectors}",
            ],
        },
    )
    if logger is None:
        raise RuntimeError("W&B logger construction unexpectedly returned None.")
    return logger


def _replay_write_count(replay: TensorDictReplayBuffer) -> int:
    value = replay.write_count
    if callable(value):
        value = value()
    return int(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _validate_transition_sample(
    sample: TensorDictBase, expected_num_transitions: int
) -> None:
    if sample.ndim != 1 or sample.numel() != expected_num_transitions:
        raise RuntimeError(
            "Standard off-policy replay must return a flat [N] transition "
            f"batch, got batch_size={sample.batch_size} for N="
            f"{expected_num_transitions}."
        )


def _diagnostic_replay_sample_size(cfg: DictConfig) -> int:
    sample_size = int(cfg.logging.sample_size)
    if str(cfg.replay.transport) != "distributed":
        return sample_size

    world_size = int(cfg.learner.world_size)
    global_batch_size = int(cfg.replay.batch_size)
    if global_batch_size % world_size:
        raise ValueError("replay.batch_size must be divisible by learner.world_size.")
    transport_sample_size = global_batch_size // world_size
    if sample_size != transport_sample_size:
        raise ValueError(
            "logging.sample_size must match the distributed replay transport's "
            "fixed per-rank sample schema: expected replay.batch_size / "
            f"learner.world_size = {transport_sample_size}, got {sample_size}."
        )
    return sample_size


class RuntimeValidationHook:
    """Log controller-side throughput and fail promptly on non-finite metrics."""

    def __init__(
        self,
        trainer: Trainer,
        replay: TensorDictReplayBuffer,
        *,
        interval_frames: int,
        sample_size: int,
    ) -> None:
        self.trainer = trainer
        self.replay = replay
        self.interval_frames = interval_frames
        self.sample_size = sample_size
        self.last_frames = int(trainer.collected_frames)
        self.last_optim_steps = int(trainer._optim_count)
        self.last_time = time.monotonic()
        self.last_sample_frames = 0
        self.checked_counts: dict[str, int] = {}

    def _check_finite(self) -> None:
        for key, values in self.trainer._log_dict.items():
            start = self.checked_counts.get(key, 0)
            for value in values[start:]:
                scalar = _as_float(value)
                if scalar is not None and not math.isfinite(scalar):
                    raise RuntimeError(f"Non-finite training metric {key}={scalar}.")
            self.checked_counts[key] = len(values)

    def _sample_metrics(self) -> dict[str, float]:
        size = len(self.replay)
        if size < self.sample_size:
            return {}
        sample = self.replay.sample(self.sample_size)
        _validate_transition_sample(sample, self.sample_size)
        metrics: dict[str, float] = {"replay/sample_batch_ndim": sample.ndim}
        policy_version = sample.get("policy_version", None)
        if policy_version is not None:
            metrics["policy_version/sample_min"] = float(policy_version.min())
            metrics["policy_version/sample_max"] = float(policy_version.max())
            metrics["policy_version/sample_mean"] = float(policy_version.float().mean())
        action = sample.get("action", None)
        if action is not None and action.is_floating_point():
            action = action.float()
            metrics["exploration/action_mean"] = float(action.mean())
            metrics["exploration/action_std"] = float(action.std())
            metrics["exploration/action_abs_max"] = float(action.abs().max())
            metrics["exploration/action_saturation_fraction"] = float(
                (action.abs() >= 0.99).float().mean()
            )
        done = sample.get(("next", "done"), None)
        episode_return = sample.get(("next", "episode_reward"), None)
        if done is not None and episode_return is not None:
            done = done.squeeze(-1).bool()
            if done.any():
                terminal_return = episode_return.squeeze(-1)[done]
                metrics["returns/replay_terminal_mean"] = float(
                    terminal_return.float().mean()
                )
                metrics["returns/replay_terminal_count"] = float(done.sum())
        return metrics

    def __call__(self) -> None:
        self._check_finite()
        frames = int(self.trainer.collected_frames)
        if frames <= self.last_frames:
            return
        now = time.monotonic()
        elapsed = max(now - self.last_time, 1e-9)
        optim_steps = int(self.trainer._optim_count)
        metrics: dict[str, float] = {
            "throughput/collection_fps": (frames - self.last_frames) / elapsed,
            "throughput/optimization_sps": (optim_steps - self.last_optim_steps)
            / elapsed,
            "replay/size": float(len(self.replay)),
            "replay/write_count": float(_replay_write_count(self.replay)),
            "distributed/published_model_version": float(
                self.trainer._published_model_version
            ),
            "distributed/learner_world_size": float(
                self.trainer._execution_backend.world_size
            ),
        }
        if frames - self.last_sample_frames >= self.interval_frames:
            metrics.update(self._sample_metrics())
            self.last_sample_frames = frames
        self.trainer._log(**metrics)
        self.last_frames = frames
        self.last_optim_steps = optim_steps
        self.last_time = now


class EvaluationHook:
    """Evaluate current learner weights and record the rendered rollout."""

    def __init__(
        self,
        trainer: Trainer,
        evaluator: Evaluator,
        logger: NamespacedLogger,
        *,
        interval_frames: int,
        shutdown_timeout: float,
        log_results: bool = True,
    ) -> None:
        self.trainer = trainer
        self.evaluator = evaluator
        self.logger = logger
        self.interval_frames = interval_frames
        self.shutdown_timeout = shutdown_timeout
        self.log_results = log_results
        self.next_frames = interval_frames
        self.pending_policy_version: int | None = None

    def _log_result(self, result: dict[str, Any]) -> None:
        result = dict(result)
        step = int(result.pop("evaluation/step"))
        if self.log_results:
            if self.pending_policy_version is not None:
                result["evaluation/policy_version"] = self.pending_policy_version
            self.logger.log_metrics(result, step=step)
        self.pending_policy_version = None

    def _poll(self) -> None:
        result = self.evaluator.poll()
        if result is not None:
            self._log_result(result)

    def __call__(self) -> None:
        self._poll()
        frames = int(self.trainer.collected_frames)
        if frames < self.next_frames or self.evaluator.pending:
            return
        policy_version = int(self.trainer._published_model_version)
        weights = self.trainer._execution_backend.get_weights(
            expected_version=policy_version
        )
        if self.evaluator.trigger_eval(weights, step=frames):
            self.pending_policy_version = policy_version
            self.next_frames = frames + self.interval_frames

    def shutdown(self) -> None:
        self._poll()
        if self.evaluator.pending:
            result = self.evaluator.wait(timeout=self.shutdown_timeout)
            if result is None:
                raise TimeoutError(
                    "The final evaluator rollout did not finish before shutdown."
                )
            self._log_result(result)
        self.evaluator.shutdown(timeout=self.shutdown_timeout)


def _make_trainer(
    cfg: DictConfig,
    objects: AlgorithmObjects,
    collector: RayCollector,
    replay: TensorDictReplayBuffer,
    logger: NamespacedLogger,
) -> Trainer:
    learner_options = {
        "world_size": int(cfg.learner.world_size),
        "resources_per_rank": {
            "num_cpus": int(cfg.learner.num_cpus_per_rank),
            "num_gpus": int(cfg.learner.num_gpus_per_rank),
        },
        "backend": str(cfg.learner.backend),
        "setup_timeout": float(cfg.learner.setup_timeout),
        "command_timeout": float(cfg.learner.command_timeout),
        "ray_init_config": _ray_init_config(cfg),
    }
    kwargs: dict[str, Any] = {
        "collector": collector,
        "total_frames": int(cfg.collection.total_frames),
        "frame_skip": 1,
        "optim_steps_per_batch": int(cfg.learner.optim_steps_per_batch),
        "loss_module": objects.loss_module,
        "optimizer": objects.optimizer,
        "replay_buffer": replay,
        "batch_size": int(cfg.replay.batch_size),
        "target_net_updater": objects.target_updater,
        "learner_backend": "ray",
        "learner_backend_options": learner_options,
        "learner_poll_interval": float(cfg.learner.poll_interval),
        "logger": logger,
        "enable_logging": False,
        "async_collection": True,
        "progress_bar": True,
        "seed": int(cfg.seed),
        "log_interval": int(cfg.logging.interval_frames),
        "save_trainer_file": None,
    }
    if objects.optimization_stepper is not None:
        kwargs["optimization_stepper"] = objects.optimization_stepper
    kwargs.update(objects.trainer_kwargs)
    return objects.trainer_cls(**kwargs)


def _all_metrics_finite(trainer: Trainer) -> bool:
    for values in trainer._log_dict.values():
        for value in values:
            scalar = _as_float(value)
            if scalar is not None and not math.isfinite(scalar):
                return False
    return True


def _validate_final_state(
    cfg: DictConfig,
    trainer: Trainer,
    replay: TensorDictReplayBuffer,
) -> dict[str, Any]:
    optim_steps = int(trainer._optim_count)
    published_version = int(trainer._published_model_version)
    write_count = _replay_write_count(replay)
    collected_frames = int(trainer.collected_frames)
    if optim_steps <= 0:
        raise RuntimeError("The learner completed without an optimization step.")
    if published_version != optim_steps:
        raise RuntimeError(
            "Published policy version does not match optimization count: "
            f"published={published_version}, optim_steps={optim_steps}."
        )
    if collected_frames < int(cfg.collection.total_frames):
        raise RuntimeError(
            f"Collector reported {collected_frames} transitions, expected at "
            f"least {cfg.collection.total_frames}."
        )
    if write_count != collected_frames:
        raise RuntimeError(
            "Direct replay transition accounting diverged from collection: "
            f"write_count={write_count}, collected_frames={collected_frames}."
        )
    if not _all_metrics_finite(trainer):
        raise RuntimeError("At least one recorded training metric is non-finite.")
    sample_size = _diagnostic_replay_sample_size(cfg)
    if len(replay) < sample_size:
        raise RuntimeError(
            "Replay does not contain enough transitions for the final diagnostic "
            f"sample: size={len(replay)}, sample_size={sample_size}."
        )
    sample = replay.sample(sample_size)
    _validate_transition_sample(sample, sample_size)
    policy_version = sample.get("policy_version", None)
    if policy_version is None:
        raise RuntimeError("Replay samples do not contain policy_version.")
    max_policy_version = int(policy_version.max())
    if max_policy_version <= 0:
        raise RuntimeError("Replay contains no data from an updated policy.")
    return {
        "algorithm": str(cfg.algorithm.name),
        "status": "completed",
        "collected_frames": collected_frames,
        "replay_write_count": write_count,
        "replay_size": len(replay),
        "replay_sample_batch_size": list(sample.batch_size),
        "replay_sample_batch_ndim": sample.ndim,
        "replay_sample_numel": sample.numel(),
        "optimization_steps": optim_steps,
        "published_model_version": published_version,
        "max_sampled_policy_version": max_policy_version,
        "num_collectors": int(cfg.environment.num_collectors),
        "total_num_envs": int(cfg.environment.total_num_envs),
        "learner_world_size": int(cfg.learner.world_size),
    }


def _write_summary(cfg: DictConfig, run_name: str, summary: dict[str, Any]) -> Path:
    artifact_dir = Path(str(cfg.artifacts_dir))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{run_name}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Construct and run one configured distributed off-policy experiment."""
    torch.manual_seed(int(cfg.seed))
    commit = _git_commit()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{cfg.algorithm.name}-{cfg.algorithm.environment_name}-"
        f"{cfg.environment.total_num_envs}env-"
        f"{cfg.environment.num_collectors}c-{cfg.learner.world_size}l-{timestamp}"
    )
    logger = NamespacedLogger(_make_logger(cfg, run_name, commit))
    experiment = logger.experiment
    replay: TensorDictReplayBuffer | None = None
    collector: RayCollector | None = None
    evaluators: list[Evaluator] = []
    exit_code = 1
    try:
        diagnostic_sample_size = _diagnostic_replay_sample_size(cfg)
        proof_env = _make_proof_env(cfg)
        try:
            objects = _make_algorithm(cfg, proof_env)
            extend_spec, sample_spec = _make_replay_transport_specs(
                cfg, objects.policy, proof_env
            )
        finally:
            proof_env.close()
        replay = _make_replay(cfg, extend_spec, sample_spec)
        collector = _make_collector(cfg, objects.policy, replay)
        trainer = _make_trainer(cfg, objects, collector, replay, logger)
        validation_hook = RuntimeValidationHook(
            trainer,
            replay,
            interval_frames=int(cfg.logging.interval_frames),
            sample_size=diagnostic_sample_size,
        )
        trainer.register_op("post_steps", validation_hook)
        if bool(cfg.evaluation.enabled):
            eval_device = str(cfg.evaluation.device)
            eval_num_envs = (
                1
                if cfg.algorithm.environment_kind == "gym"
                else int(cfg.evaluation.num_envs)
            )
            evaluator = Evaluator(
                partial(_make_evaluation_env, cfg, logger, record_video=False),
                policy_factory=partial(
                    _make_evaluation_policy,
                    policy=objects.evaluation_policy,
                    device=eval_device,
                ),
                num_trajectories=int(cfg.evaluation.num_trajectories),
                max_steps=int(cfg.evaluation.max_steps),
                frames_per_batch=int(cfg.evaluation.max_steps) * eval_num_envs,
                collector_kwargs={"traj_format": "padded"},
                log_prefix="evaluation",
                device=eval_device,
                exploration_type=ExplorationType.DETERMINISTIC,
                dump_video=False,
                busy_policy="skip",
            )
            _initialize_evaluator(evaluator)
            evaluators.append(evaluator)
            evaluation_hook = EvaluationHook(
                trainer,
                evaluator,
                logger,
                interval_frames=int(cfg.evaluation.interval_frames),
                shutdown_timeout=float(cfg.evaluation.shutdown_timeout),
            )
            trainer.register_op("post_steps", evaluation_hook)
            trainer.register_op("shutdown", evaluation_hook.shutdown)
            if bool(cfg.evaluation.video):
                video_num_envs = (
                    1
                    if cfg.algorithm.environment_kind == "gym"
                    else int(cfg.evaluation.video_num_envs)
                )
                video_evaluator = Evaluator(
                    partial(_make_evaluation_env, cfg, logger, record_video=True),
                    policy_factory=partial(
                        _make_evaluation_policy,
                        policy=objects.evaluation_policy,
                        device=eval_device,
                    ),
                    num_trajectories=int(cfg.evaluation.video_num_trajectories),
                    max_steps=int(cfg.evaluation.video_max_steps),
                    frames_per_batch=(
                        int(cfg.evaluation.video_max_steps) * video_num_envs
                    ),
                    collector_kwargs={"traj_format": "padded"},
                    log_prefix="evaluation",
                    device=eval_device,
                    exploration_type=ExplorationType.DETERMINISTIC,
                    dump_video=True,
                    busy_policy="skip",
                )
                _initialize_evaluator(video_evaluator)
                evaluators.append(video_evaluator)
                video_hook = EvaluationHook(
                    trainer,
                    video_evaluator,
                    logger,
                    interval_frames=int(cfg.evaluation.video_interval_frames),
                    shutdown_timeout=float(cfg.evaluation.shutdown_timeout),
                    log_results=False,
                )
                trainer.register_op("post_steps", video_hook)
                trainer.register_op("shutdown", video_hook.shutdown)
        trainer.train()
        summary = _validate_final_state(cfg, trainer, replay)
        summary.update(
            {
                "git_commit": commit,
                "run_name": run_name,
                "wandb_url": getattr(experiment, "url", None),
            }
        )
        path = _write_summary(cfg, run_name, summary)
        experiment.summary.update(summary)
        torchrl_logger.info(f"Validation summary written to {path}.")
        exit_code = 0
    except BaseException as err:
        failure = {
            "algorithm": str(cfg.algorithm.name),
            "status": "failed",
            "error_type": type(err).__name__,
            "error": str(err),
            "git_commit": commit,
            "run_name": run_name,
            "wandb_url": getattr(experiment, "url", None),
        }
        _write_summary(cfg, run_name, failure)
        experiment.summary.update(failure)
        raise
    finally:
        if collector is not None:
            with contextlib.suppress(Exception):
                collector.shutdown()
        for evaluator in evaluators:
            with contextlib.suppress(Exception):
                evaluator.shutdown()
        if replay is not None:
            with contextlib.suppress(Exception):
                replay.shutdown()
        experiment.finish(exit_code=exit_code)
        wandb.finish(exit_code=exit_code)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
