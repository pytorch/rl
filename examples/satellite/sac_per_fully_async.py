"""SAC + Prioritized Replay Buffer with fully-async collection.

Like ``sac_per.py`` but the collector and trainer are fully decoupled:
the collector is constructed with ``replay_buffer=rb`` and started via
``collector.start()`` so it pushes new transitions to the buffer in the
background. The main loop only iterates over gradient steps, sampling
from the buffer and updating the policy/value/alpha networks.

Defaults match the ``1rlrmzo4`` configuration (the strongest
synchronous run so far): 16k vmapped envs, 64 grad steps per "outer
chunk", 6400 outer chunks => 409,600 gradient updates total. UTD ratio
is no longer enforced -- collection and training proceed at their own
hardware-determined rates.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import torch

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Evaluator, MultiCollector
from torchrl.data import (
    LazyTensorStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs.utils import ExplorationType
from torchrl.objectives import SACLoss, SoftUpdate
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.weight_update import MultiProcessWeightSyncScheme

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent.parent))

from examples.satellite._utils import (  # noqa: E402
    DEFAULT_OBS_NORM_PATH,
    DEFAULT_TEST_SET_PATH,
    load_test_set_csv,
    make_actor,
    make_eval_metrics_fn,
    make_qvalue_critic,
    make_train_env,
    pick_device,
    setup_wandb_key,
)
from examples.satellite.sac_per import (  # noqa: E402
    _eval_env_factory,
    _eval_policy_factory,
    _train_env_factory,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI mirroring ``sac_per.py`` so launch commands are interchangeable.

    The async-only knobs are at the bottom. Unused-in-async flags
    (``--init-random-frames-per-env``, ``--gradient-steps``,
    ``--frames-per-env``) are still parsed so launch scripts don't break,
    but their semantics shift -- see help text on each.
    """
    p = argparse.ArgumentParser(description=__doc__)
    # Env / collection sizing.
    p.add_argument("--num-envs", type=int, default=16_384)
    p.add_argument("--num-cmgs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--frame-skip", type=int, default=50)
    p.add_argument("--min-random-horizon", type=int, default=100)
    p.add_argument("--random-horizon-prob", type=float, default=0.02)
    p.add_argument("--frames-per-env", type=int, default=1)
    p.add_argument(
        "--total-iters",
        type=int,
        default=6400,
        help=(
            "Number of outer chunks. Each chunk runs --gradient-steps "
            "grad updates. Total grad updates = total_iters * "
            "gradient_steps. Eval cadence and LR scheduler horizon both "
            "use total_iters."
        ),
    )
    p.add_argument("--gradient-steps", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument(
        "--init-buffer-size",
        type=int,
        default=16_384,
        help=(
            "Number of transitions the collector must produce before "
            "the trainer starts taking gradient steps. Default = "
            "num_envs (one outer-iter's worth)."
        ),
    )
    p.add_argument(
        "--init-random-frames-per-env",
        type=int,
        default=0,
        help="Parsed for CLI parity; not used by the async loop.",
    )
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--prb-alpha", type=float, default=0.7)
    p.add_argument("--prb-beta", type=float, default=0.5)
    p.add_argument("--no-prb", action="store_true")
    # SAC.
    p.add_argument("--lr", type=float, default=9e-4)
    p.add_argument("--critic-lr", type=float, default=9e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha-init", type=float, default=1.0)
    p.add_argument("--fixed-alpha", action="store_true")
    p.add_argument("--min-alpha", type=float, default=None)
    p.add_argument("--max-alpha", type=float, default=None)
    p.add_argument("--target-update-polyak", type=float, default=0.995)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--lr-decay-end-frac",
        type=float,
        default=0.1,
        help="Cosine LR end value as a fraction of starting lr.",
    )
    p.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=0,
        help=(
            "Linear warmup over this many outer iters from "
            "lr * 1/start_div -> lr, then cosine decay over the remaining "
            "(total_iters - lr_warmup_iters) iters."
        ),
    )
    p.add_argument(
        "--lr-warmup-start-factor",
        type=float,
        default=1e-2,
        help="Start factor for the warmup phase. lr_start = lr * start_factor.",
    )
    # Reward / env physics.
    p.add_argument("--action-scale", type=float, default=3.0)
    p.add_argument("--singularity-weight", type=float, default=0.0)
    p.add_argument("--singularity-clamp-min", type=float, default=1e-6)
    p.add_argument(
        "--singularity-mode",
        type=str,
        default="inverse",
        choices=["inverse", "exp"],
    )
    p.add_argument("--singularity-exp-k", type=float, default=5.0)
    p.add_argument("--omega-weight", type=float, default=0.1)
    p.add_argument("--ctrl-cost-weight", type=float, default=0.0)
    p.add_argument("--reward-scale", type=float, default=0.333)
    # Networks.
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256, 256])
    p.add_argument("--activation", type=str, default="tanh")
    p.add_argument("--small-init-last-layer", action="store_true", default=True)
    p.add_argument(
        "--no-small-init-last-layer", dest="small_init_last_layer", action="store_false"
    )
    p.add_argument("--scale-init", type=float, default=0.31)
    p.add_argument("--layer-norm", action="store_true")
    p.add_argument("--no-obs-norm", action="store_true", default=True)
    p.add_argument("--obs-norm-path", type=str, default=str(DEFAULT_OBS_NORM_PATH))
    p.add_argument("--obs-norm-warmup", type=int, default=10_000)
    # Devices / compile.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--eval-device", default=None)
    p.add_argument(
        "--buffer-device",
        default="cpu",
        help=(
            "Device for the replay buffer storage. Must be 'cpu' for "
            "shared-memory multiprocess use unless TorchRL was built "
            "with CUDA support."
        ),
    )
    p.add_argument("--compile-env", action="store_true")
    p.add_argument("--compile-eval-env", action="store_true")
    p.add_argument("--compile-policy", action="store_true")
    p.add_argument("--compile-loss", action="store_true")
    # Eval.
    p.add_argument(
        "--eval-every", type=int, default=50, help="Eval every N outer chunks."
    )
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--test-set-csv", type=str, default=str(DEFAULT_TEST_SET_PATH))
    # WandB.
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="torchrl-sat")
    p.add_argument("--wandb-group", default="sac-per-async")
    p.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    device = pick_device(cfg.device)
    eval_device = (
        torch.device(cfg.eval_device) if cfg.eval_device is not None else device
    )
    total_grad_steps = cfg.total_iters * cfg.gradient_steps
    torchrl_logger.info(
        f"Device: {device} | eval_device: {eval_device} | "
        f"num_envs={cfg.num_envs} | total_iters={cfg.total_iters} | "
        f"gradient_steps={cfg.gradient_steps} | "
        f"total_grad_steps={total_grad_steps}"
    )

    # ----- Reference env (for spec extraction only) -----
    # The async collector builds its own env in a worker. We only need
    # one in the main process so we can read obs/action specs to build
    # the actor + critic. Closed immediately after.
    spec_env, _ = make_train_env(
        num_envs=cfg.num_envs,
        device=device,
        max_steps=cfg.max_steps,
        min_random_horizon=cfg.min_random_horizon,
        random_horizon_prob=cfg.random_horizon_prob,
        compile_step=False,
        obs_norm_stats=None,
        use_obs_norm=not cfg.no_obs_norm,
        num_cmgs=cfg.num_cmgs,
        action_scale=cfg.action_scale,
        singularity_weight=cfg.singularity_weight,
        singularity_clamp_min=cfg.singularity_clamp_min,
        singularity_mode=cfg.singularity_mode,
        singularity_exp_k=cfg.singularity_exp_k,
        omega_weight=cfg.omega_weight,
        ctrl_cost_weight=cfg.ctrl_cost_weight,
        frame_skip=cfg.frame_skip,
        reward_scale=cfg.reward_scale,
        seed=cfg.seed,
    )
    obs_spec = spec_env.observation_spec
    action_spec = spec_env.action_spec
    spec_env.close()
    obs_norm_stats = None  # --no-obs-norm only path supported in async

    # ----- Networks + loss -----
    hidden = tuple(cfg.hidden)
    actor = make_actor(
        obs_spec=obs_spec,
        action_spec=action_spec,
        device=device,
        hidden=hidden,
        activation=cfg.activation,
        state_independent_scale=False,
        layer_norm=cfg.layer_norm,
        small_init_last_layer=cfg.small_init_last_layer,
        scale_init=cfg.scale_init,
    )
    qvalue = make_qvalue_critic(
        obs_spec=obs_spec,
        action_spec=action_spec,
        device=device,
        hidden=hidden,
        activation=cfg.activation,
        layer_norm=cfg.layer_norm,
    )
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        loss_function="l2",
        alpha_init=cfg.alpha_init,
        min_alpha=cfg.min_alpha,
        max_alpha=cfg.max_alpha,
        fixed_alpha=cfg.fixed_alpha,
        delay_actor=False,
        delay_qvalue=True,
        delay_value=True,
        target_entropy="auto",
        action_spec=action_spec,
    )
    loss_module.make_value_estimator(gamma=cfg.gamma)
    target_updater = SoftUpdate(loss_module, eps=cfg.target_update_polyak)

    if cfg.compile_loss:
        torchrl_logger.info("torch.compile(loss_module) enabled.")
        loss_call = torch.compile(loss_module)
    else:
        loss_call = loss_module

    critic_lr = cfg.critic_lr if cfg.critic_lr is not None else cfg.lr
    optim_actor = torch.optim.Adam(
        loss_module.actor_network_params.flatten_keys().values(),
        lr=cfg.lr,
    )
    optim_critic = torch.optim.Adam(
        loss_module.qvalue_network_params.flatten_keys().values(),
        lr=critic_lr,
    )
    optim_alpha = None
    if not cfg.fixed_alpha:
        optim_alpha = torch.optim.Adam([loss_module.log_alpha], lr=cfg.lr)

    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []

    def _build_lr_scheduler(optim, eta_min):
        """Linear warmup -> cosine decay (or just cosine if warmup=0)."""
        decay_iters = max(1, cfg.total_iters - cfg.lr_warmup_iters)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=decay_iters,
            eta_min=eta_min,
        )
        if cfg.lr_warmup_iters > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=cfg.lr_warmup_start_factor,
                end_factor=1.0,
                total_iters=cfg.lr_warmup_iters,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optim,
                schedulers=[warmup, cosine],
                milestones=[cfg.lr_warmup_iters],
            )
        return cosine

    if cfg.lr_decay_end_frac is not None:
        eta_min_actor = cfg.lr * cfg.lr_decay_end_frac
        eta_min_critic = critic_lr * cfg.lr_decay_end_frac
        schedulers.append(_build_lr_scheduler(optim_actor, eta_min_actor))
        schedulers.append(_build_lr_scheduler(optim_critic, eta_min_critic))
        if optim_alpha is not None:
            schedulers.append(_build_lr_scheduler(optim_alpha, eta_min_actor))
        if cfg.lr_warmup_iters > 0:
            torchrl_logger.info(
                f"LR schedule: linear warmup {cfg.lr * cfg.lr_warmup_start_factor:.4g} "
                f"-> {cfg.lr:.4g} over {cfg.lr_warmup_iters} iters, then "
                f"cosine decay -> {eta_min_actor:.4g} over the remaining "
                f"{cfg.total_iters - cfg.lr_warmup_iters} iters."
            )
        else:
            torchrl_logger.info(
                f"Cosine LR decay: lr {cfg.lr:.4g} -> {eta_min_actor:.4g} "
                f"over {cfg.total_iters} outer chunks."
            )

    # ----- Replay buffer (shared across processes) -----
    buffer_device = torch.device(cfg.buffer_device)
    storage = LazyTensorStorage(cfg.buffer_size, device=buffer_device)
    # Note: ``prefetch`` is incompatible with ``shared=True`` (the
    # multiprocess SyncManager wrappers can't pickle the prefetch
    # thread state).
    if cfg.no_prb:
        replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            batch_size=cfg.batch_size,
            shared=True,
        )
        torchrl_logger.info("Using uniform TensorDictReplayBuffer (no PER), shared.")
    else:
        # ``sync=False`` (added in PR pytorch/rl#3714): writer procs use a
        # shareable RandomSampler, while the learner owns a local
        # PrioritizedSampler that catches up from shared write_count
        # before sampling. Required for the fully-async collector flow.
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=cfg.prb_alpha,
            beta=cfg.prb_beta,
            storage=storage,
            batch_size=cfg.batch_size,
            priority_key="td_error",
            shared=True,
            sync=False,
        )
        torchrl_logger.info(
            f"Using TensorDictPrioritizedReplayBuffer "
            f"(alpha={cfg.prb_alpha}, beta={cfg.prb_beta}), shared, sync=False."
        )

    # ----- Async collector -----
    train_env_factory = partial(
        _train_env_factory,
        num_envs=cfg.num_envs,
        device_str=str(device),
        max_steps=cfg.max_steps,
        min_random_horizon=cfg.min_random_horizon,
        random_horizon_prob=cfg.random_horizon_prob,
        compile_step=cfg.compile_env,
        obs_norm_stats=obs_norm_stats,
        use_obs_norm=not cfg.no_obs_norm,
        num_cmgs=cfg.num_cmgs,
        action_scale=cfg.action_scale,
        singularity_weight=cfg.singularity_weight,
        singularity_clamp_min=cfg.singularity_clamp_min,
        singularity_mode=cfg.singularity_mode,
        singularity_exp_k=cfg.singularity_exp_k,
        omega_weight=cfg.omega_weight,
        ctrl_cost_weight=cfg.ctrl_cost_weight,
        frame_skip=cfg.frame_skip,
        reward_scale=cfg.reward_scale,
        seed=cfg.seed,
    )
    frames_per_batch = cfg.num_envs * cfg.frames_per_env
    collector = MultiCollector(
        create_env_fn=[train_env_factory],
        policy=actor,
        replay_buffer=replay_buffer,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        device=device,
        exploration_type=ExplorationType.RANDOM,
        update_at_each_batch=True,
        sync=False,
    )
    torchrl_logger.info(
        "MultiCollector(sync=False, replay_buffer=rb) ready -- starting "
        "async collection now."
    )

    # ----- WandB -----
    logger = None
    if not cfg.no_wandb:
        setup_wandb_key()
        exp = generate_exp_name("SAC-PER-async", "satellite")
        logger = get_logger(
            "wandb",
            logger_name="torchrl-sat",
            experiment_name=exp,
            wandb_kwargs={
                "project": cfg.wandb_project,
                "group": cfg.wandb_group,
                "mode": cfg.wandb_mode,
                "config": vars(cfg),
            },
        )

    # ----- Eval setup (process backend, identical to sac_per.py) -----
    metrics_fn = None
    evaluator = None
    if not cfg.no_eval:
        _, _, cats = load_test_set_csv(cfg.test_set_csv)
        metrics_fn = make_eval_metrics_fn(cats)

        def _on_eval_result(result):
            torchrl_logger.info(
                f"[eval] step={int(result.get('eval/step', -1))} "
                f"return={float(result.get('eval/eval/return', float('nan'))):.3f} "
                f"final_err={float(result.get('eval/eval/final_attitude_error_rad', float('nan'))):.3f} "
                f"success@0.10={float(result.get('eval/eval/success_rate@0.10', float('nan'))):.3f}"
            )
            if logger is not None:
                flat = {
                    k: v.item() if hasattr(v, "item") else float(v)
                    for k, v in result.items()
                }
                step = int(flat.pop("eval/step", -1))
                logger.log_metrics(flat, step=max(0, step))

        env_factory = partial(
            _eval_env_factory,
            device_str=str(eval_device),
            test_set_csv=cfg.test_set_csv,
            max_steps=cfg.max_steps,
            obs_norm_stats=obs_norm_stats,
            use_obs_norm=not cfg.no_obs_norm,
            num_cmgs=cfg.num_cmgs,
            action_scale=cfg.action_scale,
            singularity_weight=cfg.singularity_weight,
            singularity_clamp_min=cfg.singularity_clamp_min,
            singularity_mode=cfg.singularity_mode,
            singularity_exp_k=cfg.singularity_exp_k,
            omega_weight=cfg.omega_weight,
            ctrl_cost_weight=cfg.ctrl_cost_weight,
            frame_skip=cfg.frame_skip,
            compile_step=cfg.compile_eval_env,
            reward_scale=cfg.reward_scale,
        )
        policy_factory = partial(
            _eval_policy_factory,
            obs_spec=obs_spec,
            action_spec=action_spec,
            hidden=tuple(cfg.hidden),
            activation=cfg.activation,
            device_str=str(eval_device),
        )
        evaluator = Evaluator(
            env=env_factory,
            policy_factory=policy_factory,
            num_trajectories=len(cats),
            max_steps=None,
            exploration_type=ExplorationType.DETERMINISTIC,
            metrics_fn=metrics_fn,
            on_result=_on_eval_result,
            backend="process",
            weight_sync_schemes={"policy": MultiProcessWeightSyncScheme()},
        )

    # ----- Start async collection -----
    collector.start()

    # Wait for the buffer to warm up enough to start sampling.
    while len(replay_buffer) < cfg.init_buffer_size:
        torchrl_logger.info(
            f"warmup: buffer size = {len(replay_buffer)} "
            f"< {cfg.init_buffer_size}; sleeping 1s"
        )
        time.sleep(1.0)
    torchrl_logger.info(
        f"warmup done: buffer size = {len(replay_buffer)}; "
        f"starting gradient updates."
    )

    # ----- Training loop -----
    pbar_t0 = time.perf_counter()
    grad_step = 0
    last_log_t = pbar_t0
    last_loss = None

    for outer_iter in range(cfg.total_iters):
        for _ in range(cfg.gradient_steps):
            sample = replay_buffer.sample().to(device)
            losses = loss_call(sample)

            optim_actor.zero_grad(set_to_none=True)
            optim_critic.zero_grad(set_to_none=True)
            if optim_alpha is not None:
                optim_alpha.zero_grad(set_to_none=True)
            loss = losses["loss_actor"] + losses["loss_qvalue"] + losses["loss_alpha"]
            if not torch.isfinite(loss):
                loss_summary = {
                    key: value.detach().item()
                    for key, value in losses.items()
                    if value.numel() == 1
                }
                raise RuntimeError(
                    f"Non-finite SAC loss at grad_step {grad_step}: " f"{loss_summary}"
                )
            loss.backward()
            grad_norm_actor = torch.nn.utils.clip_grad_norm_(
                list(loss_module.actor_network_params.flatten_keys().values()),
                cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"),
            )
            grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                list(loss_module.qvalue_network_params.flatten_keys().values()),
                cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"),
            )
            if optim_alpha is not None:
                grad_norm_alpha = torch.nn.utils.clip_grad_norm_(
                    [loss_module.log_alpha],
                    cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"),
                )
            else:
                grad_norm_alpha = torch.zeros((), device=device)
            optim_actor.step()
            optim_critic.step()
            if optim_alpha is not None:
                optim_alpha.step()
            target_updater.step()

            if not cfg.no_prb:
                replay_buffer.update_tensordict_priority(sample)
            last_loss = losses
            grad_step += 1

        for sched in schedulers:
            sched.step()

        # Per-outer-iter logging.
        now = time.perf_counter()
        log_dt = now - last_log_t
        last_log_t = now
        # Reproduce the same step semantic as ``sac_per.py``: "frames"
        # in wandb step indexing tracks collector frames; use the
        # collector's running write counter as the canonical step.
        try:
            collected_frames = int(replay_buffer.write_count)
        except (AttributeError, TypeError):
            collected_frames = len(replay_buffer)
        torchrl_logger.info(
            f"iter={outer_iter} grad_step={grad_step} "
            f"buffer={len(replay_buffer)} "
            f"frames={collected_frames} "
            f"loss_q={last_loss['loss_qvalue'].item():.3f} "
            f"loss_a={last_loss['loss_actor'].item():.3f} "
            f"alpha={loss_module.log_alpha.detach().exp().item():.3f} "
            f"dt={log_dt:.2f}s"
        )
        # Diagnostic batch metrics from the most recent sampled minibatch.
        # Mirrors what ``sac_per.py`` and ``ppo_buffer.py`` log so all
        # three scripts share a comparable training-quality dashboard.
        with torch.no_grad():
            batch_reward_per_step = sample.get(("next", "reward")).mean().item()
            qerr_t = sample.get(("next", "quat_err"), default=None)
            batch_attitude_error = (
                qerr_t.norm(dim=-1).mean().item()
                if qerr_t is not None
                else float("nan")
            )
            omega_t = sample.get(("next", "bus_omega"), default=None)
            batch_omega_sq = (
                (omega_t**2).sum(dim=-1).mean().item()
                if omega_t is not None
                else float("nan")
            )
            done_mask = sample.get(("next", "done"))
            ep_r_t = sample.get(("next", "episode_reward"), default=None)
            episode_reward = (
                ep_r_t[done_mask].mean().item()
                if ep_r_t is not None and done_mask.any()
                else None
            )

        if logger is not None and last_loss is not None:
            metrics = {
                "train/loss_actor": last_loss["loss_actor"].item(),
                "train/loss_qvalue": last_loss["loss_qvalue"].item(),
                "train/loss_alpha": last_loss["loss_alpha"].item(),
                "train/alpha": loss_module.log_alpha.detach().exp().item(),
                "train/n_updates": grad_step,
                "train/buffer_size": len(replay_buffer),
                # ``train/lr`` is the canonical key shared with PPO; keep
                # ``train/critic_lr`` as a back-compat alias.
                "train/lr": optim_critic.param_groups[0]["lr"],
                "train/critic_lr": optim_critic.param_groups[0]["lr"],
                "train/grad_norm_actor": grad_norm_actor.item(),
                "train/grad_norm_critic": grad_norm_critic.item(),
                "train/grad_norm_alpha": grad_norm_alpha.item()
                if isinstance(grad_norm_alpha, torch.Tensor)
                else grad_norm_alpha,
                "train/iter_per_sec": (outer_iter + 1) / max(1e-6, now - pbar_t0),
                "train/iter_dt_sec": log_dt,
                "train/batch_reward_per_step": batch_reward_per_step,
                "train/batch_attitude_error": batch_attitude_error,
                "train/batch_omega_sq": batch_omega_sq,
            }
            if episode_reward is not None:
                metrics["train/episode_reward"] = episode_reward
            logger.log_metrics(metrics, step=collected_frames)

        # Eval trigger -- skip if the previous eval is still running
        # (the async Evaluator's default ``busy_policy='error'`` raises
        # when overlapping triggers come in faster than eval can drain).
        if (
            evaluator is not None
            and (outer_iter + 1) % cfg.eval_every == 0
            and not evaluator.pending
        ):
            evaluator.trigger_eval(actor, step=collected_frames)

    # ----- Shutdown -----
    torchrl_logger.info(
        f"Training complete: {grad_step} gradient updates. " "Shutting down collector."
    )
    try:
        collector.async_shutdown(timeout=30.0)
    except Exception as e:
        torchrl_logger.warning(f"collector.async_shutdown failed: {e}")
    if evaluator is not None:
        if evaluator.pending:
            try:
                evaluator.wait(timeout=120.0)
            except Exception as e:
                torchrl_logger.warning(f"Final eval wait failed: {e}")
        try:
            evaluator.shutdown()
        except Exception as e:
            torchrl_logger.warning(f"evaluator.shutdown failed: {e}")
    if logger is not None:
        try:
            logger.experiment.finish()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
