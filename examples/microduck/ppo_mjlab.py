# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train MicroDuck's velocity task with TorchRL PPO and MJLab.

This example deliberately uses the task registered by ``mjlab_microduck``
instead of rebuilding its reward in TorchRL. Consequently, the BAM actuator
model, observations, rewards, curricula, domain randomization, and termination
rules remain those of the MicroDuck project. This is an MJLab example, not a
backend-neutral MuJoCo task.

Run it from a checkout of ``pollen-robotics/microduck_rl`` so that the
MicroDuck lockfile supplies its pinned simulator dependencies::

    uv run --with-editable /path/to/rl --with cmake \
        --with 'torchcodec>=0.10.0' \
        --no-build-isolation-package tensordict \
        python /path/to/rl/examples/microduck/ppo_mjlab.py --smoke

The smoke preset runs 64 environments for five PPO iterations. Remove
``--smoke`` for the default 4096-environment, 5000-iteration run. Metrics,
checkpoints, and evaluation videos are written below ``--output-dir``.
Training requires an NVIDIA CUDA GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl import timeit
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import (
    Compose,
    ExplorationType,
    RewardSum,
    set_exploration_type,
    StepCounter,
    TransformedEnv,
    VecNormV2,
)
from torchrl.envs.libs.mjlab import MJLabWrapper
from torchrl.modules import IndependentNormal, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import CSVLogger

_has_mjlab = importlib.util.find_spec("mjlab") is not None
_has_microduck = importlib.util.find_spec("mjlab_microduck") is not None
_has_torchcodec = importlib.util.find_spec("torchcodec") is not None

TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
ACTOR_OBS_KEY = "actor"
CRITIC_OBS_KEY = "critic"
NUM_MINIBATCHES = 4
NUM_EPOCHS = 5
TARGET_KL = 0.01
MIN_LR = 1.0e-5
MAX_LR = 1.0e-2
LR_FACTOR = 1.5


class MetricsMJLabWrapper(MJLabWrapper):
    """Retain numeric MJLab log entries without changing the public wrapper."""

    def __init__(self, *args, **kwargs):
        self._metric_sums: dict[str, torch.Tensor] = {}
        self._metric_counts: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def _capture_metrics(self) -> None:
        extras = getattr(self._env, "extras", None)
        if not isinstance(extras, Mapping):
            return
        values = extras.get("log")
        if not isinstance(values, Mapping):
            return
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                if not value.numel():
                    continue
                scalar = value.detach().float().mean()
            elif isinstance(value, Real):
                scalar = torch.as_tensor(float(value), device=self.device)
            else:
                continue
            key = str(key)
            previous = self._metric_sums.get(key)
            self._metric_sums[key] = scalar if previous is None else previous + scalar
            self._metric_counts[key] = self._metric_counts.get(key, 0) + 1

    def _reset(
        self,
        tensordict: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        result = super()._reset(tensordict, **kwargs)
        self._capture_metrics()
        return result

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        result = super()._step(tensordict)
        self._capture_metrics()
        return result

    def pop_metrics(self) -> dict[str, float]:
        result = {}
        for key, value in self._metric_sums.items():
            mean = (value / self._metric_counts[key]).item()
            if math.isfinite(mean):
                result[key] = mean
        self._metric_sums.clear()
        self._metric_counts.clear()
        return result


class GaussianPolicyHead(nn.Module):
    """ELU policy network with a learned scalar Gaussian standard deviation."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LazyLinear(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(()))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.backbone(observation)
        return loc, self.log_std.exp().expand_as(loc)


def make_env(
    *,
    num_envs: int,
    device: torch.device,
    seed: int,
    play: bool = False,
    from_pixels: bool = False,
) -> tuple[TransformedEnv, Any, MetricsMJLabWrapper, VecNormV2]:
    # These imports are intentionally lazy: MJLab and MicroDuck are optional
    # dependencies of this standalone example.
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = deepcopy(load_env_cfg(TASK_ID, play=play))
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = False
    raw_env = ManagerBasedRlEnv(
        cfg=cfg,
        device=str(device),
        render_mode="rgb_array" if from_pixels else None,
    )
    wrapped_env = MetricsMJLabWrapper(
        raw_env,
        native_autoreset=False,
        from_pixels=from_pixels,
    )
    normalizer = VecNormV2(
        in_keys=[ACTOR_OBS_KEY, CRITIC_OBS_KEY],
        reduce_batch_dims=True,
    )
    env = TransformedEnv(
        wrapped_env,
        Compose(
            normalizer,
            RewardSum(in_keys=[wrapped_env.reward_key], out_keys=["episode_reward"]),
            StepCounter(),
        ),
    )
    return env, raw_env, wrapped_env, normalizer


def make_models(
    env: TransformedEnv,
    device: torch.device,
) -> tuple[ProbabilisticActor, ValueOperator]:
    action_dim = env.action_spec_unbatched.shape[-1]
    policy_module = TensorDictModule(
        GaussianPolicyHead(action_dim),
        in_keys=[ACTOR_OBS_KEY],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=["loc", "scale"],
        distribution_class=IndependentNormal,
        return_log_prob=True,
    ).to(device)
    critic = ValueOperator(
        nn.Sequential(
            nn.LazyLinear(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ),
        in_keys=[CRITIC_OBS_KEY],
    ).to(device)

    # Materialize lazy layers before checkpoint loading or optimizer creation.
    fake_tensordict = env.fake_tensordict()
    with torch.no_grad():
        actor(fake_tensordict)
        critic(fake_tensordict)
    return actor, critic


def _mjlab_counters(raw_env: Any) -> dict[str, Any]:
    counters = {}
    for name in ("common_step_counter", "_sim_step_counter"):
        value = getattr(raw_env, name, None)
        if isinstance(value, torch.Tensor):
            counters[name] = value.detach().cpu()
        elif value is not None:
            counters[name] = value
    return counters


def _restore_mjlab_counters(raw_env: Any, counters: Mapping[str, Any]) -> None:
    for name, saved_value in counters.items():
        current_value = getattr(raw_env, name, None)
        if isinstance(current_value, torch.Tensor):
            current_value.copy_(
                torch.as_tensor(saved_value, device=current_value.device)
            )
        elif current_value is not None:
            setattr(raw_env, name, saved_value)


def save_checkpoint(
    path: Path,
    *,
    iteration: int,
    total_frames: int,
    actor: ProbabilisticActor,
    critic: ValueOperator,
    optimizer: torch.optim.Optimizer,
    normalizer: VecNormV2,
    raw_env: Any,
) -> None:
    state = {
        "iteration": iteration,
        "total_frames": total_frames,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "mjlab_counters": _mjlab_counters(raw_env),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    torch.save(state, path.parent / "latest.pt")


def load_checkpoint(
    path: Path,
    *,
    actor: ProbabilisticActor,
    critic: ValueOperator,
    optimizer: torch.optim.Optimizer,
    normalizer: VecNormV2,
    raw_env: Any,
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    optimizer.load_state_dict(state["optimizer"])
    normalizer.load_state_dict(state["normalizer"])
    _restore_mjlab_counters(raw_env, state.get("mjlab_counters", {}))
    torch.set_rng_state(state["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state["iteration"]) + 1, int(state["total_frames"])


def _assert_finite_batch(batch: TensorDictBase) -> None:
    for key in (ACTOR_OBS_KEY, CRITIC_OBS_KEY, ("next", "reward")):
        value = batch.get(key)
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite values found under TensorDict key {key!r}.")


def _set_adaptive_learning_rate(
    optimizer: torch.optim.Optimizer,
    mean_kl: float,
) -> float:
    learning_rate = float(optimizer.param_groups[0]["lr"])
    if mean_kl > 2.0 * TARGET_KL:
        learning_rate = max(MIN_LR, learning_rate / LR_FACTOR)
    elif 0.0 < mean_kl < 0.5 * TARGET_KL:
        learning_rate = min(MAX_LR, learning_rate * LR_FACTOR)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def evaluate(
    *,
    actor: ProbabilisticActor,
    train_normalizer: VecNormV2,
    device: torch.device,
    seed: int,
    logger: CSVLogger,
    iteration: int,
) -> float:
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all()
    eval_env = None
    try:
        eval_env, raw_eval_env, _, eval_normalizer = make_env(
            num_envs=1,
            device=device,
            seed=seed + 1,
            play=True,
            from_pixels=True,
        )
        eval_normalizer.load_state_dict(train_normalizer.state_dict())
        eval_normalizer.freeze()
        max_steps = int(getattr(raw_eval_env, "max_episode_length", 1000))
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            rollout = eval_env.rollout(
                max_steps=max_steps,
                policy=actor,
                break_when_any_done=True,
                return_contiguous=True,
            )
        episode_return = rollout.get(("next", "reward")).sum().item()
        pixels = rollout.get(("next", "pixels"))
        logger.log_video(
            "evaluation",
            pixels.movedim(-1, -3).cpu(),
            step=iteration,
        )
        logger.log_scalar("evaluation/return", episode_return, step=iteration)
        return episode_return
    finally:
        if eval_env is not None and not eval_env.is_closed:
            eval_env.close()
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_state)


def _log_iteration(
    *,
    logger: CSVLogger,
    iteration: int,
    batch: TensorDictBase,
    loss_metrics: Mapping[str, float],
    upstream_metrics: Mapping[str, float],
    timings: Mapping[str, float],
    learning_rate: float,
) -> None:
    reward = batch.get(("next", "reward")).mean().item()
    logger.log_scalar("train/reward", reward, step=iteration)
    logger.log_scalar("train/learning_rate", learning_rate, step=iteration)

    done = batch.get(("next", "done")).squeeze(-1)
    if done.any():
        episode_returns = batch.get(("next", "episode_reward")).squeeze(-1)[done]
        episode_lengths = batch.get(("next", "step_count")).squeeze(-1)[done]
        logger.log_scalar(
            "train/episode_return", episode_returns.mean().item(), step=iteration
        )
        logger.log_scalar(
            "train/episode_length",
            episode_lengths.float().mean().item(),
            step=iteration,
        )

    for name, value in loss_metrics.items():
        logger.log_scalar(f"loss/{name}", value, step=iteration)
    for name, value in upstream_metrics.items():
        logger.log_scalar(f"mjlab/{name}", value, step=iteration)
    for name, value in timings.items():
        logger.log_scalar(name, value, step=iteration)

    elapsed = sum(timings.values())
    if elapsed > 0.0:
        logger.log_scalar(
            "train/frames_per_second",
            batch.numel() / elapsed,
            step=iteration,
        )


def train(args: argparse.Namespace) -> None:
    if args.smoke:
        args.num_envs = 64
        args.iterations = 5
    if args.num_envs < 1 or args.iterations < 1 or args.rollout_steps < 1:
        raise ValueError("num-envs, iterations, and rollout-steps must be positive.")
    if args.checkpoint_interval < 1 or args.eval_interval < 1:
        raise ValueError("checkpoint-interval and eval-interval must be positive.")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "MicroDuck MJLab training requires an available CUDA device."
        )
    if not _has_mjlab or not _has_microduck:
        raise RuntimeError(
            "Install the MicroDuck environment and run this example from its uv "
            "environment; see the command in this module's docstring."
        )
    if not _has_torchcodec:
        raise RuntimeError(
            "Evaluation videos require torchcodec>=0.10.0. Add "
            "`--with 'torchcodec>=0.10.0'` to the uv command."
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    logger = CSVLogger(
        exp_name="metrics",
        log_dir=str(run_dir),
        video_format="mp4",
        video_fps=50,
    )
    logger.log_hparams(
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
    )

    env, raw_env, wrapped_env, normalizer = make_env(
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
    )
    actor, critic = make_models(env, device)
    advantage = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
        average_gae=False,
        device=device,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=0.2,
        clip_value=True,
        entropy_bonus=True,
        entropy_coeff=0.01,
        critic_coeff=1.0,
        loss_critic_type="l2",
        normalize_advantage=True,
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=1.0e-3)

    start_iteration = 1
    total_frames = 0
    if args.resume is not None:
        start_iteration, total_frames = load_checkpoint(
            args.resume.expanduser().resolve(),
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            normalizer=normalizer,
            raw_env=raw_env,
        )
    if start_iteration > args.iterations:
        raise ValueError(
            f"Checkpoint resumes at iteration {start_iteration}, beyond the requested "
            f"total of {args.iterations}."
        )

    frames_per_batch = args.num_envs * args.rollout_steps
    minibatch_size = frames_per_batch // NUM_MINIBATCHES
    if minibatch_size * NUM_MINIBATCHES != frames_per_batch:
        raise ValueError(
            f"num-envs * rollout-steps ({frames_per_batch}) must be divisible by "
            f"{NUM_MINIBATCHES}."
        )
    collector = Collector(
        env,
        actor,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=(args.iterations - start_iteration + 1) * frames_per_batch,
        max_frames_per_traj=-1,
    )
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    collector_iterator = iter(collector)
    try:
        for iteration in range(start_iteration, args.iterations + 1):
            timeit.reset()
            with timeit("collect"):
                batch = next(collector_iterator)
            _assert_finite_batch(batch)
            total_frames += batch.numel()

            with timeit("train"):
                with torch.no_grad():
                    advantage(batch)
                flat_batch = batch.reshape(-1)
                accumulated: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
                for _ in range(NUM_EPOCHS):
                    replay_buffer.empty()
                    replay_buffer.extend(flat_batch)
                    for _ in range(NUM_MINIBATCHES):
                        sample = replay_buffer.sample()
                        losses = loss_module(sample)
                        total_loss = (
                            losses["loss_objective"]
                            + losses["loss_critic"]
                            + losses["loss_entropy"]
                        )
                        if not torch.isfinite(total_loss):
                            raise RuntimeError("PPO produced a non-finite loss.")
                        optimizer.zero_grad(set_to_none=True)
                        total_loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            loss_module.parameters(), 1.0
                        )
                        if not torch.isfinite(grad_norm):
                            raise RuntimeError("PPO produced non-finite gradients.")
                        optimizer.step()
                        for key in (
                            "loss_objective",
                            "loss_critic",
                            "loss_entropy",
                            "entropy",
                            "kl_approx",
                            "clip_fraction",
                            "value_clip_fraction",
                            "explained_variance",
                        ):
                            if key in losses:
                                accumulated[key].append(losses[key].detach())
                        accumulated["grad_norm"].append(grad_norm.detach())

            collector.update_policy_weights_()
            loss_metrics = {
                key: torch.stack(values).mean().item()
                for key, values in accumulated.items()
            }
            learning_rate = _set_adaptive_learning_rate(
                optimizer, loss_metrics["kl_approx"]
            )
            timings = timeit.todict(prefix="time")
            upstream_metrics = wrapped_env.pop_metrics()
            _log_iteration(
                logger=logger,
                iteration=iteration,
                batch=batch,
                loss_metrics=loss_metrics,
                upstream_metrics=upstream_metrics,
                timings=timings,
                learning_rate=learning_rate,
            )

            final_iteration = iteration == args.iterations
            if iteration % args.eval_interval == 0 or final_iteration:
                eval_return = evaluate(
                    actor=actor,
                    train_normalizer=normalizer,
                    device=device,
                    seed=args.seed,
                    logger=logger,
                    iteration=iteration,
                )
            else:
                eval_return = None
            if iteration % args.checkpoint_interval == 0 or final_iteration:
                save_checkpoint(
                    checkpoint_dir / f"model_{iteration:06d}.pt",
                    iteration=iteration,
                    total_frames=total_frames,
                    actor=actor,
                    critic=critic,
                    optimizer=optimizer,
                    normalizer=normalizer,
                    raw_env=raw_env,
                )

            message = (
                f"iteration={iteration:05d} frames={total_frames} "
                f"reward={batch['next', 'reward'].mean().item():+.4f} "
                f"kl={loss_metrics['kl_approx']:.5f} lr={learning_rate:.2e}"
            )
            if eval_return is not None:
                message += f" eval_return={eval_return:+.3f}"
            print(message, flush=True)
    finally:
        collector.shutdown()
        if not env.is_closed:
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("microduck_ppo_runs/default"),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 64 environments for five iterations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
