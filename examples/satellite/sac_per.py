"""SAC + Prioritized Replay Buffer training for the satellite task.

Same env / eval / logging conventions as ``examples/satellite/ppo.py`` so the
two are directly comparable on the CSV-backed test set.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import torch

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector, Evaluator, MultiCollector
from torchrl.data import (
    LazyTensorStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.postprocs import MultiStep
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
    make_eval_env,
    make_eval_metrics_fn,
    make_qvalue_critic,
    make_train_env,
    pick_device,
    setup_wandb_key,
)


# Module-level eval-process factories. The ``backend="process"``
# Evaluator pickles these and re-constructs the env + policy inside a
# child process, so they must be importable top-level callables (not
# closures over local state).


def _train_env_factory(
    *,
    num_envs: int,
    device_str: str,
    max_steps: int,
    min_random_horizon: int | None,
    random_horizon_prob: float,
    compile_step: bool,
    obs_norm_stats: tuple[torch.Tensor, torch.Tensor] | None,
    use_obs_norm: bool,
    num_cmgs: int,
    action_scale: float,
    singularity_weight: float,
    singularity_clamp_min: float,
    singularity_mode: str,
    singularity_exp_k: float,
    omega_weight: float,
    ctrl_cost_weight: float,
    frame_skip: int,
    reward_scale: float = 1.0,
    seed: int | None = None,
):
    """Picklable training-env factory for ``aSyncDataCollector``."""
    env, _ = make_train_env(
        num_envs=num_envs,
        device=torch.device(device_str),
        max_steps=max_steps,
        min_random_horizon=min_random_horizon,
        random_horizon_prob=random_horizon_prob,
        compile_step=compile_step,
        obs_norm_stats=obs_norm_stats,
        use_obs_norm=use_obs_norm,
        num_cmgs=num_cmgs,
        action_scale=action_scale,
        singularity_weight=singularity_weight,
        singularity_clamp_min=singularity_clamp_min,
        singularity_mode=singularity_mode,
        singularity_exp_k=singularity_exp_k,
        omega_weight=omega_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        frame_skip=frame_skip,
        reward_scale=reward_scale,
        seed=seed,
    )
    return env


def _eval_env_factory(
    *,
    device_str: str,
    test_set_csv: str,
    max_steps: int,
    obs_norm_stats: tuple[torch.Tensor, torch.Tensor] | None,
    use_obs_norm: bool,
    num_cmgs: int,
    action_scale: float,
    singularity_weight: float,
    singularity_clamp_min: float,
    singularity_mode: str,
    singularity_exp_k: float,
    omega_weight: float,
    ctrl_cost_weight: float,
    frame_skip: int,
    compile_step: bool,
    reward_scale: float = 1.0,
):
    return make_eval_env(
        device=torch.device(device_str),
        test_set_csv=test_set_csv,
        max_steps=max_steps,
        obs_norm_stats=obs_norm_stats,
        use_obs_norm=use_obs_norm,
        num_cmgs=num_cmgs,
        action_scale=action_scale,
        singularity_weight=singularity_weight,
        singularity_clamp_min=singularity_clamp_min,
        singularity_mode=singularity_mode,
        singularity_exp_k=singularity_exp_k,
        omega_weight=omega_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        frame_skip=frame_skip,
        compile_step=compile_step,
        reward_scale=reward_scale,
    )


def _eval_policy_factory(
    env=None,
    *,
    obs_spec=None,
    action_spec=None,
    hidden: tuple[int, ...],
    activation: str,
    device_str: str,
):
    """Build an actor identical to the trainer's.

    The ``MultiProcessWeightSyncScheme`` calls this factory **without**
    args on the sender side (just to enumerate parameter shapes), and
    again **with** an env on the receiver side. We accept either by
    falling back to specs passed via :func:`functools.partial`.
    """
    if env is not None:
        obs_spec = env.observation_spec
        action_spec = env.action_spec
    if obs_spec is None or action_spec is None:
        raise RuntimeError(
            "_eval_policy_factory needs either env or pre-bound "
            "(obs_spec, action_spec) via partial."
        )
    return make_actor(
        obs_spec=obs_spec,
        action_spec=action_spec,
        device=torch.device(device_str),
        hidden=hidden,
        activation=activation,
        state_independent_scale=False,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=32_768)
    p.add_argument("--num-cmgs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument(
        "--frame-skip",
        type=int,
        default=50,
        help=(
            "Number of physics sub-steps per agent step. dt=0.001s, so "
            "frame_skip=50 means each step is 50ms of sim time. Larger "
            "frame skip = bigger per-step dynamics = stronger Q-action "
            "signal."
        ),
    )
    p.add_argument(
        "--min-random-horizon",
        type=int,
        default=None,
        help=(
            "If set, decorrelate vectorized env phases by sampling train "
            "episode truncation horizons in [min_random_horizon, max_steps]."
        ),
    )
    p.add_argument(
        "--random-horizon-prob",
        type=float,
        default=0.0,
        help="Probability of resampling a shortened horizon after first reset.",
    )
    p.add_argument(
        "--frames-per-env",
        type=int,
        default=8,
        help=(
            "Env-steps per collection iteration. Higher = fewer "
            "Python collector cycles per env transition; lower = more "
            "frequent grad-step updates with fresher data."
        ),
    )
    p.add_argument("--total-iters", type=int, default=3000)
    # Replay buffer
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=8_192)
    p.add_argument("--prb-alpha", type=float, default=0.7)
    p.add_argument("--prb-beta", type=float, default=0.5)
    p.add_argument(
        "--no-prb",
        action="store_true",
        help=(
            "Use a uniform replay buffer instead of prioritized. "
            "Disables priority updates (no td_error key required)."
        ),
    )
    p.add_argument(
        "--lr-decay-end-frac",
        type=float,
        default=None,
        help=(
            "If set (e.g. 0.1), cosine-anneal actor/critic/alpha LR "
            "from --lr to lr*frac over the full --total-iters span."
        ),
    )
    p.add_argument("--gradient-steps", type=int, default=8)
    p.add_argument(
        "--init-random-frames-per-env",
        type=int,
        default=4,
        help="Random-action warm-up steps per env before SAC updates start.",
    )
    # Optim / SAC
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--critic-lr",
        type=float,
        default=None,
        help="Optional critic learning-rate override. Defaults to --lr.",
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha-init", type=float, default=1.0)
    p.add_argument(
        "--fixed-alpha",
        action="store_true",
        help="Keep SAC temperature fixed at --alpha-init.",
    )
    p.add_argument("--min-alpha", type=float, default=None)
    p.add_argument("--max-alpha", type=float, default=None)
    p.add_argument("--target-update-polyak", type=float, default=0.995)
    p.add_argument("--max-grad-norm", type=float, default=0.0)
    p.add_argument(
        "--action-scale",
        type=float,
        default=3.0,
        help="Scaling from agent action [-1, 1] to commanded gimbal rate (rad/s).",
    )
    p.add_argument(
        "--singularity-weight",
        type=float,
        default=0.5,
        help="Weight on -1/manip_norm in the reward (rotor-speed-invariant).",
    )
    p.add_argument(
        "--singularity-clamp-min",
        type=float,
        default=1e-6,
        help=(
            "Floor on manip_norm before division (only used when "
            "--singularity-mode=inverse). Higher values bound the worst-case "
            "spike: max penalty = singularity_weight / singularity_clamp_min."
        ),
    )
    p.add_argument(
        "--singularity-mode",
        type=str,
        default="inverse",
        choices=["inverse", "exp"],
        help=(
            "Singularity penalty form. 'inverse' (default): -w/manip_norm "
            "(unbounded near singularity, controllable via --singularity-clamp-min). "
            "'exp': -w*exp(-k*manip_norm), bounded at -w."
        ),
    )
    p.add_argument(
        "--singularity-exp-k",
        type=float,
        default=5.0,
        help="Curvature for --singularity-mode=exp. Larger = steeper falloff.",
    )
    p.add_argument(
        "--omega-weight",
        type=float,
        default=0.1,
        help="Weight on -||bus_omega||^2 in the reward (slew-and-stop incentive).",
    )
    p.add_argument(
        "--ctrl-cost-weight",
        type=float,
        default=0.01,
        help="Weight on -||action||^2. Set to 0 to remove the control penalty.",
    )
    p.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplicative scaling on the per-step reward (applied as "
            "a transform). Brings raw reward in [-3.5, 0] into a Q-friendly "
            "range. 1.0=no change; 0.333 maps to roughly [-1, 0]."
        ),
    )
    p.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden-layer sizes for both actor and critic MLPs.",
    )
    p.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "elu", "gelu", "silu"],
        help="Activation for both actor and critic hidden layers.",
    )
    p.add_argument(
        "--layer-norm",
        action="store_true",
        help="Add LayerNorm in actor + critic MLP hidden layers.",
    )
    p.add_argument(
        "--small-init-last-layer",
        action="store_true",
        help=(
            "Orthogonal-init the last Linear of the actor and critic "
            "MLPs with gain=0.01 (zero bias). Initial actor mean is 0, "
            "initial Q is ~0 -- avoids 'confidently wrong' default init."
        ),
    )
    p.add_argument(
        "--scale-init",
        type=float,
        default=1.0,
        help=(
            "Initial value of the actor's TanhNormal scale at "
            "zero-input (via biased_softplus). 1.0 saturates samples "
            "near +/-1 (heavy exploration); 0.31 gives moderate "
            "samples in [-0.3, 0.3]; 0.5 is a middle ground."
        ),
    )
    p.add_argument(
        "--n-step",
        type=int,
        default=1,
        help=(
            "n in n-step returns. >1 wraps the collector with a "
            "MultiStep postproc (gamma already taken from --gamma) so "
            "Q learns from n-step rewards."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--eval-device",
        default=None,
        help=(
            "Device for the eval-process env+policy. Defaults to --device. "
            "Use a separate CUDA index (e.g. cuda:1) to run eval on a "
            "different GPU than training."
        ),
    )
    p.add_argument(
        "--buffer-device",
        default=None,
        help=(
            "Device for the replay buffer storage. Defaults to --device. "
            "Use 'cpu' on environments where TorchRL was not built with "
            "CUDA support (no CUDA PRB extension)."
        ),
    )
    p.add_argument("--compile-env", action="store_true")
    p.add_argument(
        "--compile-eval-env",
        action="store_true",
        help="Pass compile_step=True to the eval env (separate from --compile-env).",
    )
    p.add_argument(
        "--compile-policy",
        action="store_true",
        help="torch.compile the policy forward pass during collection.",
    )
    p.add_argument(
        "--compile-loss",
        action="store_true",
        help=(
            "torch.compile(loss_module) so the SAC objective is JIT-traced "
            "and runs as a fused graph each gradient step."
        ),
    )
    p.add_argument(
        "--async-collector",
        action="store_true",
        help=(
            "Use aSyncDataCollector (collection runs in a separate process "
            "and overlaps with gradient updates). Requires that the env "
            "constructor be picklable; weights are synced once per outer "
            "iter via MultiProcessWeightSyncScheme."
        ),
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="torchrl-sat")
    p.add_argument("--wandb-group", default="sac-per")
    p.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help=(
            "wandb run mode. Use 'offline' when wandb.init() "
            "handshake keeps timing out; sync later with "
            "'wandb sync torchrl-sat/wandb/offline-run-*'."
        ),
    )
    p.add_argument("--test-set-csv", default=str(DEFAULT_TEST_SET_PATH))
    p.add_argument("--obs-norm-path", default=str(DEFAULT_OBS_NORM_PATH))
    p.add_argument(
        "--no-obs-norm",
        action="store_true",
        help=(
            "Skip the ObservationNorm transform; the policy sees raw "
            "observations (already in physical units after the env's "
            "sin/cos gimbal encoding)."
        ),
    )
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the periodic eval rollout entirely (focus on train metrics only).",
    )
    p.add_argument("--obs-norm-warmup", type=int, default=1024)
    return p.parse_args(argv)


def _ensure_obs_norm_stats(
    *,
    num_envs: int,
    device: torch.device,
    max_steps: int,
    num_cmgs: int,
    action_scale: float,
    singularity_weight: float,
    omega_weight: float,
    ctrl_cost_weight: float,
    frame_skip: int,
    seed: int,
    path: Path,
    warmup: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if path.exists():
        torchrl_logger.info(f"Loading ObservationNorm stats from {path}")
        d = torch.load(path, map_location="cpu", weights_only=True)
        loc, scale = d["loc"], d["scale"]
        expected_dim = 6 + 3 * num_cmgs
        if loc.shape[-1] == expected_dim and scale.shape[-1] == expected_dim:
            return loc, scale
        torchrl_logger.warning(
            f"Ignoring stale ObservationNorm stats at {path}: expected "
            f"last dim {expected_dim}, got loc={tuple(loc.shape)} "
            f"scale={tuple(scale.shape)}."
        )
    torchrl_logger.info(
        f"No ObservationNorm stats at {path}; running {warmup} warm-up steps."
    )
    env, obs_norm = make_train_env(
        num_envs=min(num_envs, 1024),
        device=device,
        max_steps=max_steps,
        min_random_horizon=None,
        compile_step=False,
        obs_norm_stats=None,
        num_cmgs=num_cmgs,
        action_scale=action_scale,
        singularity_weight=singularity_weight,
        omega_weight=omega_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        frame_skip=frame_skip,
        seed=seed,
    )
    obs_norm.init_stats(
        num_iter=warmup, reduce_dim=(0, 1), cat_dim=1, key="observation"
    )
    loc = obs_norm.loc.detach().cpu()
    scale = obs_norm.scale.detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"loc": loc, "scale": scale}, path)
    torchrl_logger.info(f"Wrote ObservationNorm stats to {path}")
    env.close()
    return loc, scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    device = pick_device(cfg.device)
    eval_device = (
        torch.device(cfg.eval_device) if cfg.eval_device is not None else device
    )
    torchrl_logger.info(
        f"Device: {device} | eval_device: {eval_device} | "
        f"num_envs={cfg.num_envs} | frames_per_env={cfg.frames_per_env} | "
        f"total_iters={cfg.total_iters}"
    )

    if not cfg.no_wandb:
        setup_wandb_key()

    obs_norm_stats: tuple[torch.Tensor, torch.Tensor] | None
    if cfg.no_obs_norm:
        torchrl_logger.info("ObservationNorm disabled (--no-obs-norm).")
        obs_norm_stats = None
    else:
        loc, scale = _ensure_obs_norm_stats(
            num_envs=cfg.num_envs,
            device=device,
            max_steps=cfg.max_steps,
            num_cmgs=cfg.num_cmgs,
            action_scale=cfg.action_scale,
            singularity_weight=cfg.singularity_weight,
            omega_weight=cfg.omega_weight,
            ctrl_cost_weight=cfg.ctrl_cost_weight,
            frame_skip=cfg.frame_skip,
            seed=cfg.seed,
            path=Path(cfg.obs_norm_path),
            warmup=cfg.obs_norm_warmup,
        )
        obs_norm_stats = (loc, scale)

    train_env, _ = make_train_env(
        num_envs=cfg.num_envs,
        device=device,
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
    obs_spec = train_env.observation_spec
    action_spec = train_env.action_spec

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
        small_init_last_layer=cfg.small_init_last_layer,
    )

    with torch.no_grad():
        td0 = train_env.reset()
        actor(td0)
        qvalue(actor(td0))

    # ----- Loss + target net -----
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

    # Optional torch.compile of the loss objective. The SACLoss forward
    # is a fixed graph over a single sample TD; compiling it fuses
    # actor + twin-Q + alpha computation per gradient step. The
    # uncompiled module is kept around for soft-update bookkeeping
    # (target_updater walks loss_module.qvalue_network_params).
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

    # Optional cosine LR decay across the full training span. Decays
    # to ``lr * lr_decay_end_frac`` by ``total_iters``.
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
    if cfg.lr_decay_end_frac is not None:
        eta_min_actor = cfg.lr * cfg.lr_decay_end_frac
        eta_min_critic = critic_lr * cfg.lr_decay_end_frac
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_actor,
                T_max=cfg.total_iters,
                eta_min=eta_min_actor,
            )
        )
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_critic,
                T_max=cfg.total_iters,
                eta_min=eta_min_critic,
            )
        )
        if optim_alpha is not None:
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim_alpha,
                    T_max=cfg.total_iters,
                    eta_min=eta_min_actor,
                )
            )
        torchrl_logger.info(
            f"Cosine LR decay enabled: lr {cfg.lr:.4g} -> "
            f"{eta_min_actor:.4g} over {cfg.total_iters} iters."
        )

    # ----- Replay buffer -----
    buffer_device = (
        torch.device(cfg.buffer_device) if cfg.buffer_device is not None else device
    )
    storage = LazyTensorStorage(cfg.buffer_size, device=buffer_device)
    if cfg.no_prb:
        replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            batch_size=cfg.batch_size,
            prefetch=3,
        )
        torchrl_logger.info("Using uniform TensorDictReplayBuffer (no PER).")
    else:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=cfg.prb_alpha,
            beta=cfg.prb_beta,
            storage=storage,
            batch_size=cfg.batch_size,
            priority_key="td_error",
            prefetch=3,
        )

    # ----- Collector -----
    frames_per_batch = cfg.num_envs * cfg.frames_per_env
    init_random_frames = cfg.num_envs * cfg.init_random_frames_per_env
    total_frames = frames_per_batch * cfg.total_iters
    postproc = (
        MultiStep(gamma=cfg.gamma, n_steps=cfg.n_step) if cfg.n_step > 1 else None
    )
    if postproc is not None:
        torchrl_logger.info(
            f"Using {cfg.n_step}-step returns "
            f"(MultiStep postproc with gamma={cfg.gamma})."
        )
    if cfg.async_collector:
        # Build a picklable env factory; the worker process re-creates
        # the env there. The actor lives in the main process and gets
        # synced to the worker once per outer iter via update_policy_weights_.
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
        # Free the in-process train env: the worker will build its own.
        train_env.close()
        torchrl_logger.info(
            "Async collector enabled (MultiCollector sync=False, 1 worker). "
            "Collection runs in a separate process and overlaps with grad updates."
        )
        collector = MultiCollector(
            create_env_fn=[train_env_factory],
            policy=actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            init_random_frames=init_random_frames,
            exploration_type=ExplorationType.RANDOM,
            postproc=postproc,
            update_at_each_batch=True,
            sync=False,
        )
    else:
        collector = Collector(
            train_env,
            policy=actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            init_random_frames=init_random_frames,
            exploration_type=ExplorationType.RANDOM,
            compile_policy=cfg.compile_policy,
            postproc=postproc,
        )

    # ----- WandB -----
    logger = None
    if not cfg.no_wandb:
        exp = generate_exp_name("SAC-PER", "satellite")
        # ``--wandb-mode offline`` avoids the 90s wandb.init() handshake
        # timeout that has been intermittently failing on this host.
        # Sync offline runs with
        # ``wandb sync torchrl-sat/wandb/offline-run-*``.
        wandb_mode = cfg.wandb_mode
        logger = get_logger(
            "wandb",
            logger_name="torchrl-sat",
            experiment_name=exp,
            wandb_kwargs={
                "project": cfg.wandb_project,
                "group": cfg.wandb_group,
                "config": vars(cfg),
                "mode": wandb_mode,
            },
        )

    # ----- Eval (process-isolated, non-blocking) -----
    evaluator = None
    if not cfg.no_eval:
        _, _, cats = load_test_set_csv(cfg.test_set_csv)
        metrics_fn = make_eval_metrics_fn(cats)

        def _on_eval_result(result):
            # Logged from the evaluator's coordination thread.
            torchrl_logger.info(
                f"[eval] step={int(result.get('eval/step', -1))} "
                f"return={float(result.get('eval/eval/return', float('nan'))):.3f} "
                f"final_err={float(result.get('eval/eval/final_attitude_error_rad', float('nan'))):.3f} "
                f"success@0.10={float(result.get('eval/eval/success_rate@0.10', float('nan'))):.3f}"
            )
            if logger is not None:
                # The Evaluator emits keys with the prefix it was
                # configured with (default ``eval/``); we store them as-is.
                flat = {
                    k: v.item() if hasattr(v, "item") else float(v)
                    for k, v in result.items()
                }
                step = int(flat.pop("eval/step", -1))
                logger.log_metrics(flat, step=max(0, step))

        n_eval_envs = len(cats)
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
            num_trajectories=n_eval_envs,
            # ``max_steps`` is intentionally None: the eval env already
            # has a :class:`StepCounter` set to ``cfg.max_steps``, and
            # the process-backend collector raises if both sources try
            # to enforce the same horizon.
            max_steps=None,
            exploration_type=ExplorationType.DETERMINISTIC,
            metrics_fn=metrics_fn,
            on_result=_on_eval_result,
            backend="process",
            weight_sync_schemes={"policy": MultiProcessWeightSyncScheme()},
        )
        torchrl_logger.info(
            "Evaluator(backend='process') ready; first eval will spawn "
            "the child process on the first trigger."
        )
    else:
        torchrl_logger.info("Eval disabled (--no-eval); training-only metrics.")

    def _check_finite(td, key: tuple[str, ...], context: str) -> None:
        value = td.get(key)
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite tensor at {context}: {key}")

    # ----- Training loop -----
    pbar_t0 = time.perf_counter()
    collected_frames = 0
    for it, batch in enumerate(collector):
        _check_finite(batch, ("next", "reward"), f"collector iter {it}")
        _check_finite(batch, ("next", "observation"), f"collector iter {it}")
        # Flatten (num_envs, frames_per_env, ...) -> (N, ...)
        flat = batch.reshape(-1)
        replay_buffer.extend(flat)
        collected_frames += frames_per_batch

        # Skip SAC updates while warming up the buffer.
        if collected_frames < init_random_frames:
            torchrl_logger.info(f"iter={it} (warmup) frames={collected_frames}")
            continue

        # ----- SAC updates -----
        last_loss = None
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
                raise RuntimeError(f"Non-finite SAC loss at iter {it}: {loss_summary}")
            loss.backward()
            # Per-component grad-norm logging. ``clip_grad_norm_``
            # returns the total norm BEFORE clipping, so we can log it
            # directly. Calling it per parameter group also clips each
            # group independently rather than treating actor + critic
            # + alpha as one big vector.
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

            # Push the per-sample TD error back as the priority signal
            # (only meaningful for prioritized replay).
            if not cfg.no_prb:
                replay_buffer.update_tensordict_priority(sample)
            last_loss = losses

        done_mask = batch["next", "done"]
        ep_reward = batch["next", "episode_reward"][done_mask]
        ep_length = batch["next", "step_count"][done_mask].to(torch.float32)
        ep_reward_mean = ep_reward.mean().item() if ep_reward.numel() else None
        ep_length_mean = ep_length.mean().item() if ep_length.numel() else None
        # Per-step reward is the policy-quality signal: cumulative
        # reward divided by episode length removes the artifact where
        # ``episode_reward`` drifts more negative simply because
        # ``RandomTruncationTransform`` lengthens episodes once the
        # first uniformly-sampled horizons clear out.
        ep_reward_per_step = (
            (ep_reward / ep_length.clamp_min(1.0)).mean().item()
            if ep_reward.numel()
            else None
        )
        ep_reward_text = (
            f"{ep_reward_mean:.3f}" if ep_reward_mean is not None else "n/a"
        )
        # ``batch_reward_per_step`` is the snapshot of "how is the
        # policy doing right now": mean reward across the current
        # 1024-sample collection batch. Decoupled from episode length
        # and episode-completion rate, so it is the cleanest training
        # signal.
        batch_reward_per_step = batch.get(("next", "reward")).mean().item()
        # Mean ||bus_omega||² this batch -- shows whether the policy
        # is actually moving the satellite (>>0) or sitting still (~0).
        batch_omega_sq = (
            (batch.get(("next", "bus_omega")) ** 2).sum(dim=-1).mean().item()
        )
        # Mean ||quat_err|| this batch -- direct read of how far the
        # bus is from the target on average right now.
        batch_att_err = batch.get(("next", "quat_err")).norm(dim=-1).mean().item()
        if logger is not None and last_loss is not None:
            metrics = {
                "train/loss_actor": last_loss["loss_actor"].item(),
                "train/loss_qvalue": last_loss["loss_qvalue"].item(),
                "train/loss_alpha": last_loss["loss_alpha"].item(),
                "train/alpha": loss_module.log_alpha.detach().exp().item(),
                "train/iter_per_sec": (it + 1)
                / max(1e-6, time.perf_counter() - pbar_t0),
                "train/buffer_size": len(replay_buffer),
                "train/critic_lr": optim_critic.param_groups[0]["lr"],
                "train/batch_reward_per_step": batch_reward_per_step,
                "train/batch_omega_sq": batch_omega_sq,
                "train/batch_attitude_error": batch_att_err,
                "train/grad_norm_actor": grad_norm_actor.item(),
                "train/grad_norm_critic": grad_norm_critic.item(),
                "train/grad_norm_alpha": grad_norm_alpha.item(),
            }
            if ep_reward_mean is not None:
                metrics["train/episode_reward"] = ep_reward_mean
                metrics["train/episode_reward_per_step"] = ep_reward_per_step
                metrics["train/episode_length"] = ep_length_mean
            logger.log_metrics(metrics, step=collected_frames)
        if last_loss is not None:
            torchrl_logger.info(
                f"iter={it} frames={collected_frames} "
                f"r/step={batch_reward_per_step:.3f} "
                f"|omega|²={batch_omega_sq:.3f} "
                f"|q_err|={batch_att_err:.3f} "
                f"ep_r={ep_reward_text} "
                f"loss_q={last_loss['loss_qvalue'].item():.3f} "
                f"loss_a={last_loss['loss_actor'].item():.3f} "
                f"alpha={loss_module.log_alpha.detach().exp().item():.3f}"
            )

        # Step LR schedulers once per outer iteration (each iter is one
        # collector batch; gradient_steps inner steps share the same lr).
        for sched in schedulers:
            sched.step()

        if (
            evaluator is not None
            and (it + 1) % cfg.eval_every == 0
            and not evaluator.pending
        ):
            evaluator.trigger_eval(actor, step=collected_frames)

    if evaluator is not None:
        # Drain any pending eval, then shut the worker process down.
        if evaluator.pending:
            try:
                evaluator.wait(timeout=120.0)
            except Exception as e:  # noqa: BLE001
                torchrl_logger.warning(f"Final eval wait failed: {e}")
        evaluator.shutdown()
    collector.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
