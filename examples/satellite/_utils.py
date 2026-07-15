"""Shared utilities for the satellite PPO and SAC training scripts.

Houses everything both scripts need so the entry-points stay short and the
PPO-vs-SAC comparison is apples-to-apples (same env wrapping, same eval
protocol, same metric definitions, same WandB project / group).

Public surface:

* :func:`pick_device` -- one-line GPU/MPS/CPU selection.
* :func:`make_train_env` -- vmapped :class:`SatelliteEnv` + transform stack.
* :func:`make_eval_env` -- same stack plus a stateless :class:`TestSetPrimer`
  feeding fixed `(init_bus_quat, target_quat)` rows from a CSV.
* :class:`TestSetPrimer` -- thin :class:`TensorDictPrimer` subclass that
  emits a deterministic batch of test-set rows on every reset (with safe
  tiling when ``num_envs != len(test_set)``).
* :func:`generate_test_set` / :func:`dump_test_set_csv` /
  :func:`load_test_set_csv` -- (re)producible eval set on disk.
* :func:`make_actor` / :func:`make_value_critic` /
  :func:`make_qvalue_critic` -- shared TanhNormal actor and MLP critics.
* :func:`eval_metrics_fn` -- per-category metrics for the
  :class:`Evaluator`.
* :func:`setup_wandb_key` -- compatibility no-op; configure WandB with
  ``WANDB_API_KEY`` or ``wandb login`` before running.

Run ``python -m examples.satellite._utils generate-test-set`` to materialize
the deterministic eval CSV; otherwise the default eval set is generated in memory.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tensordict import NestedKey, TensorDictBase
from tensordict.nn import (
    AddStateIndependentNormalScale,
    NormalParamExtractor,
    TensorDictModule,
)

from torchrl.data.tensor_specs import Composite, TensorSpec, Unbounded
from torchrl.envs import (
    CatTensors,
    ObservationNorm,
    RandomTruncationTransform,
    RewardScaling,
    RewardSum,
    StepCounter,
    TensorDictPrimer,
    TransformedEnv,
)
from torchrl.envs.custom.mujoco import SatelliteEnv
from torchrl.envs.custom.mujoco._math import (
    cmg_jacobian,
    manipulability,
    pyramid_4cmg_geometry,
    quat_conj,
    quat_log,
    quat_mul,
)
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator


# ---------------------------------------------------------------------------
# Constants -- a single source of truth for layout / file paths.
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_SET_PATH = PACKAGE_DIR / "test_set.csv"
DEFAULT_OBS_NORM_PATH = PACKAGE_DIR / "obs_norm_stats.pt"

# Observation sub-keys fed to the policy / critic. Manipulability is
# kept *outside* this list so the network never sees it -- it's a
# logging-only signal.
POLICY_OBS_KEYS: list[str] = [
    "quat_err",
    "bus_omega",
    "gimbal_angles",
    "gimbal_rates",
]

# Eval / test-set categories. Kept here so the metric reporter and the
# generator stay in sync.
CATEGORIES: tuple[str, ...] = (
    "uniform",
    "large_err",
    "near_singular",
    "precision",
    "off_axis",
)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Order: explicit ``prefer`` argument > ``CUDA`` > ``CPU``.

    .. note::

        We deliberately skip MPS: the ``mujoco-torch`` backend keeps its
        physics model in float64 internally, which MPS doesn't support.
        For these backends, "GPU if available, CPU otherwise" maps Apple
        silicon to CPU rather than MPS.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Test-set generation
# ---------------------------------------------------------------------------


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _attitude_error_norm(init_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """Magnitude of the quaternion log-map between two attitudes (radians)."""
    q_err = quat_mul(quat_conj(init_q), target_q)
    return quat_log(q_err).norm(dim=-1)


def _manip_at_angles(angles: torch.Tensor, num_cmgs: int = 4) -> torch.Tensor:
    """Manipulability at given gimbal angles for the 4-CMG pyramid geometry."""
    g, r0 = pyramid_4cmg_geometry(device=angles.device, dtype=angles.dtype)
    jac = cmg_jacobian(angles, g, r0, 100.0)
    return manipulability(jac)


def generate_test_set(
    n: int = 256,
    seed: int = 42,
    num_cmgs: int = 4,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | list[str]]:
    """Generate a deterministic test set covering the four hard categories.

    Returns a dict with keys ``init_bus_quat`` (n, 4), ``target_quat``
    (n, 4) and ``category`` (list[str]). Every category gets ``n // 5``
    rows (the residual is padded with ``uniform``).

    The generation is fully seeded by ``seed`` so re-running this
    function with the same arguments yields a byte-identical CSV.
    """
    if num_cmgs != 4:
        # The generator below only knows the 4-CMG pyramid geometry --
        # extending to 6-CMG is straightforward but unnecessary now.
        raise NotImplementedError("Test-set generation currently assumes 4 CMGs.")

    g = torch.Generator(device="cpu").manual_seed(seed)
    per_cat = n // len(CATEGORIES)
    residual = n - per_cat * len(CATEGORIES)
    counts = {c: per_cat for c in CATEGORIES}
    counts["uniform"] += residual  # absorb any leftover

    inits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    cats: list[str] = []

    # uniform -- pure random pairs; the dynamics-relevant baseline.
    n_u = counts["uniform"]
    inits.append(_normalize_quat(torch.randn(n_u, 4, generator=g)))
    targets.append(_normalize_quat(torch.randn(n_u, 4, generator=g)))
    cats.extend(["uniform"] * n_u)

    # large_err -- reject any pair with attitude error < 120 degrees.
    n_l = counts["large_err"]
    LARGE_THRESHOLD = math.radians(120.0)
    accepted_init: list[torch.Tensor] = []
    accepted_target: list[torch.Tensor] = []
    while len(accepted_init) < n_l:
        batch = max(64, n_l * 4)
        i = _normalize_quat(torch.randn(batch, 4, generator=g))
        t = _normalize_quat(torch.randn(batch, 4, generator=g))
        err = _attitude_error_norm(i, t)
        keep = err >= LARGE_THRESHOLD
        for j in range(batch):
            if keep[j]:
                accepted_init.append(i[j])
                accepted_target.append(t[j])
                if len(accepted_init) == n_l:
                    break
    inits.append(torch.stack(accepted_init))
    targets.append(torch.stack(accepted_target))
    cats.extend(["large_err"] * n_l)

    # near_singular -- pick (init, target) pairs whose initial slew
    # direction (the quat-log of the relative attitude) aligns with a
    # near-null direction of the gimbal Jacobian at the satellite's
    # nominal posture. We can't prescribe ``qpos[7:]`` (gimbal angles)
    # via the reset API, so the proxy here is: among many random pairs,
    # pick the ones where the slew axis is most poorly conditioned for
    # the *nominal* CMG configuration, i.e. has the lowest projection
    # onto the column span of the Jacobian at gimbal angles=0. The
    # bottom-N by that score is the "hardest to slew" family.
    n_s = counts["near_singular"]
    pool = max(8 * n_s, 4096)
    pool_init = _normalize_quat(torch.randn(pool, 4, generator=g))
    pool_target = _normalize_quat(torch.randn(pool, 4, generator=g))
    # Initial slew direction: the quaternion log of the relative
    # attitude error.
    pool_qerr = quat_mul(quat_conj(pool_init), pool_target)
    pool_log = quat_log(pool_qerr)  # (pool, 3)
    # Jacobian at nominal (zero) gimbal angles; shape (3, 4). We
    # measure how poorly the slew direction aligns with the J column
    # span by 1 - (||J^T s|| / ||s|| / sigma_max), where s is the slew
    # direction. A small alignment score => the satellite has to push
    # mostly through the null space => harder maneuver.
    g_axes, r0 = pyramid_4cmg_geometry(device="cpu", dtype=torch.float32)
    jac = cmg_jacobian(torch.zeros(1, 4), g_axes, r0, 100.0).squeeze(0)  # (3, 4)
    sigma_max = torch.linalg.svdvals(jac).max()
    s_unit = pool_log / pool_log.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    alignment = (jac.T @ s_unit.T).norm(dim=0) / sigma_max  # (pool,)
    worst_idx = torch.topk(-alignment, k=n_s).indices
    inits.append(pool_init.index_select(0, worst_idx))
    targets.append(pool_target.index_select(0, worst_idx))
    cats.extend(["near_singular"] * n_s)

    # precision -- pairs with moderate attitude error so success at the
    # 0.05 rad threshold is genuinely informative rather than trivial.
    n_p = counts["precision"]
    accepted_init = []
    accepted_target = []
    LO, HI = math.radians(30.0), math.radians(90.0)
    while len(accepted_init) < n_p:
        batch = max(64, n_p * 4)
        i = _normalize_quat(torch.randn(batch, 4, generator=g))
        t = _normalize_quat(torch.randn(batch, 4, generator=g))
        err = _attitude_error_norm(i, t)
        keep = (err >= LO) & (err <= HI)
        for j in range(batch):
            if keep[j]:
                accepted_init.append(i[j])
                accepted_target.append(t[j])
                if len(accepted_init) == n_p:
                    break
    inits.append(torch.stack(accepted_init))
    targets.append(torch.stack(accepted_target))
    cats.extend(["precision"] * n_p)

    # off_axis -- targets oriented along non-principal axes. We sample
    # a random axis with all components > 0.4 in absolute value and a
    # random angle, then build the quaternion (cos a/2, sin a/2 * axis).
    n_o = counts["off_axis"]
    accepted_init = []
    accepted_target = []
    while len(accepted_init) < n_o:
        ax = torch.randn(1, 3, generator=g)
        if (ax.abs() < 0.4).any():
            continue
        ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        angle = torch.empty(1).uniform_(
            math.radians(60.0), math.radians(160.0), generator=g
        )
        half = 0.5 * angle
        target_q = torch.cat(
            [half.cos().unsqueeze(0), half.sin().unsqueeze(0) * ax], dim=-1
        ).squeeze(0)
        init_q = _normalize_quat(torch.randn(1, 4, generator=g)).squeeze(0)
        accepted_init.append(init_q)
        accepted_target.append(_normalize_quat(target_q.unsqueeze(0)).squeeze(0))
    inits.append(torch.stack(accepted_init))
    targets.append(torch.stack(accepted_target))
    cats.extend(["off_axis"] * n_o)

    init_t = torch.cat(inits, dim=0).to(device).float()
    target_t = torch.cat(targets, dim=0).to(device).float()
    assert init_t.shape == (n, 4) and target_t.shape == (n, 4)
    return {"init_bus_quat": init_t, "target_quat": target_t, "category": cats}


def dump_test_set_csv(test_set: dict[str, Any], path: str | Path) -> None:
    """Write the test set to a CSV (12 floats + 1 category string per row)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    init = test_set["init_bus_quat"]
    target = test_set["target_quat"]
    cats = test_set["category"]
    n = init.shape[0]
    cols = (
        ["init_w", "init_x", "init_y", "init_z"]
        + ["target_w", "target_x", "target_y", "target_z"]
        + ["category"]
    )
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for k in range(n):
            row = (
                [f"{init[k, j].item():.10f}" for j in range(4)]
                + [f"{target[k, j].item():.10f}" for j in range(4)]
                + [cats[k]]
            )
            f.write(",".join(row) + "\n")


def load_test_set_csv(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Read the CSV back into ``(init_bus_quat, target_quat, categories)``."""
    path = Path(path)
    init_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    cats: list[str] = []
    with path.open() as f:
        header = f.readline().strip().split(",")  # noqa: F841 -- documented schema
        for line in f:
            parts = line.strip().split(",")
            init_rows.append([float(x) for x in parts[:4]])
            target_rows.append([float(x) for x in parts[4:8]])
            cats.append(parts[8])
    init = torch.tensor(init_rows, dtype=torch.float32)
    target = torch.tensor(target_rows, dtype=torch.float32)
    return init, target, cats


# ---------------------------------------------------------------------------
# Stateless test-set primer
# ---------------------------------------------------------------------------


class TestSetPrimer(TensorDictPrimer):
    """Inject fixed ``(init_bus_quat, target_quat)`` pairs at every reset.

    Stateless and vectorized: the primer holds the full test set and
    deterministically slices / tiles it to fill a reset of any size.
    Designed to pair with the modified :class:`SatelliteEnv` whose
    :meth:`_reset` reads these keys from the input tensordict.

    With ``num_envs == len(test_set)`` and full resets only (the standard
    eval flow), each env always receives the same row -- so two
    consecutive eval rollouts replay the exact same starts.
    """

    def __init__(
        self,
        init_bus_quat: torch.Tensor,
        target_quat: torch.Tensor,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        n = init_bus_quat.shape[0]
        if target_quat.shape[0] != n:
            raise ValueError(
                f"init/target length mismatch: {init_bus_quat.shape[0]} vs "
                f"{target_quat.shape[0]}"
            )
        device = torch.device(device)
        self._init = init_bus_quat.to(device)
        self._target = target_quat.to(device)
        self._n = n

        # Specs declare the keys to the framework. ``random=False`` and
        # ``default_value=callable`` triggers the per-reset closure
        # below, which sees the reset mask shape and returns matching
        # rows.
        primer_specs = Composite(
            init_bus_quat=Unbounded(shape=(4,), dtype=torch.float32, device=device),
            target_quat=Unbounded(shape=(4,), dtype=torch.float32, device=device),
        )
        super().__init__(
            primers=primer_specs,
            random=False,
            default_value=self._sample,
            single_default_value=True,
            expand_specs=True,
            call_before_env_reset=True,
        )

    def _sample(
        self, reset: torch.Tensor | None = None, **_: Any
    ) -> dict[str, torch.Tensor]:
        # ``reset`` is the bool mask, shape ``(B,)`` or ``(B, 1)``;
        # may be ``None`` when called from a full-batch reset path.
        # Strategy: generate ``arange(B) % n`` over the *full* batch
        # then sub-select by the mask. This keeps row k always paired
        # with env k for the standard ``B == n`` setup, and gracefully
        # tiles / truncates otherwise.
        if reset is None:
            B = self._n
            sel = torch.arange(B, device=self._init.device) % self._n
        else:
            mask = reset
            if mask.ndim > 1:
                mask = mask.squeeze(-1)
            B = mask.shape[0]
            full = torch.arange(B, device=self._init.device) % self._n
            sel = full[mask]
        return {
            "init_bus_quat": self._init.index_select(0, sel),
            "target_quat": self._target.index_select(0, sel),
        }


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------


def _attach_obs_norm(
    env: TransformedEnv,
    obs_norm_stats: tuple[torch.Tensor, torch.Tensor] | None,
    device: torch.device,
) -> ObservationNorm:
    """Append an :class:`ObservationNorm`, optionally pre-loaded with stats.

    Returns the transform instance so the caller can ``init_stats`` on
    it later when ``obs_norm_stats is None``.
    """
    if obs_norm_stats is not None:
        loc, scale = obs_norm_stats
        norm = ObservationNorm(
            loc=loc.to(device),
            scale=scale.to(device),
            in_keys=["observation"],
            standard_normal=True,
        )
    else:
        # Lazy: stats filled in later via ``init_stats``.
        norm = ObservationNorm(
            in_keys=["observation"],
            standard_normal=True,
        )
    env.append_transform(norm)
    return norm


def make_train_env(
    *,
    num_envs: int,
    device: torch.device,
    max_steps: int = 1500,
    min_random_horizon: int | None = None,
    random_horizon_prob: float = 0.0,
    compile_step: bool = False,
    obs_norm_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    use_obs_norm: bool = True,
    num_cmgs: int = 4,
    action_scale: float = 3.0,
    singularity_weight: float = 0.5,
    singularity_clamp_min: float = 1e-6,
    singularity_mode: str = "inverse",
    singularity_exp_k: float = 5.0,
    omega_weight: float = 0.1,
    ctrl_cost_weight: float = 0.01,
    frame_skip: int | None = None,
    reward_scale: float = 1.0,
    seed: int | None = None,
) -> tuple[TransformedEnv, ObservationNorm | None]:
    """Vmapped training env. Returns ``(env, observation_norm_transform)``.

    When ``use_obs_norm=False`` the :class:`ObservationNorm` transform
    is omitted entirely and the second tuple element is ``None``. The
    raw observation is in physically meaningful units already
    (``quat_err`` in radians, ``bus_omega`` in rad/s, gimbal sin/cos in
    [-1, 1], gimbal_rates in rad/s) so a network with reasonable init
    can train on them directly.

    When ``use_obs_norm=True``, the caller is expected to either:

    * pass ``obs_norm_stats=(loc, scale)`` so eval / training share stats; or
    * leave ``obs_norm_stats=None``, then call
      ``observation_norm_transform.init_stats(num_iter=N, ...)`` once.
    """
    base = SatelliteEnv(
        num_cmgs=num_cmgs,
        num_envs=num_envs,
        backend="mujoco-torch",
        device=device,
        seed=seed,
        max_episode_steps=max_steps,
        compile_step=compile_step,
        compile_kwargs={"dynamic": False} if compile_step else None,
        action_scale=action_scale,
        singularity_weight=singularity_weight,
        singularity_clamp_min=singularity_clamp_min,
        singularity_mode=singularity_mode,
        singularity_exp_k=singularity_exp_k,
        omega_weight=omega_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        frame_skip=frame_skip,
    )
    env = TransformedEnv(base)
    # 1) Pack the dynamics-relevant sub-keys into a single ``observation``
    #    tensor. Keep ``manipulability`` outside (logging-only).
    env.append_transform(
        CatTensors(
            in_keys=POLICY_OBS_KEYS,
            out_key="observation",
            dim=-1,
            del_keys=False,
            sort=False,
        )
    )
    # 2) Per-dim normalization, shared across PPO / SAC / eval. Skipped
    #    when ``use_obs_norm=False`` -- the network sees raw observations.
    obs_norm: ObservationNorm | None = None
    if use_obs_norm:
        obs_norm = _attach_obs_norm(env, obs_norm_stats, device)
    # 3) Episodic step counter (logging) + episode-return aggregator
    #    (consumed by the Evaluator).
    env.append_transform(StepCounter(max_steps=max_steps))
    if min_random_horizon is not None:
        env.append_transform(
            RandomTruncationTransform(
                min_horizon=min_random_horizon,
                max_horizon=max_steps,
                prob=random_horizon_prob,
            )
        )
    if reward_scale != 1.0:
        # ``reward = reward * scale + 0`` -- scale rewards before
        # downstream consumers (RewardSum, replay buffer, loss). With
        # raw reward in [-3.5, 0] and scale=1/3 the post-scale range
        # is [-1.17, 0], much friendlier for Q-target regression.
        env.append_transform(
            RewardScaling(loc=0.0, scale=reward_scale, in_keys=["reward"])
        )
    env.append_transform(RewardSum())
    return env, obs_norm


def make_eval_env(
    *,
    device: torch.device,
    test_set_csv: str | Path = DEFAULT_TEST_SET_PATH,
    test_set_size: int = 256,
    test_set_seed: int = 42,
    max_steps: int = 1500,
    obs_norm_stats: tuple[torch.Tensor, torch.Tensor] | None,
    use_obs_norm: bool = True,
    num_cmgs: int = 4,
    action_scale: float = 3.0,
    singularity_weight: float = 0.5,
    singularity_clamp_min: float = 1e-6,
    singularity_mode: str = "inverse",
    singularity_exp_k: float = 5.0,
    omega_weight: float = 0.1,
    ctrl_cost_weight: float = 0.01,
    frame_skip: int | None = None,
    compile_step: bool = False,
    reward_scale: float = 1.0,
    seed: int = 0,
) -> TransformedEnv:
    """Eval env: ``num_envs == len(test_set)`` so one rollout = full eval.

    The :class:`TestSetPrimer` injects matching ``init_bus_quat`` /
    ``target_quat`` rows on every reset so the eval is byte-stable across
    iterations. The ``action_scale`` / ``singularity_weight`` /
    ``use_obs_norm`` settings must match the training env so eval
    rewards are directly comparable. If the default CSV was not
    materialized, the same deterministic test set is generated in memory.
    """
    test_set_csv = Path(test_set_csv)
    if test_set_csv.exists():
        init_q, target_q, _cats = load_test_set_csv(test_set_csv)
    elif test_set_csv == DEFAULT_TEST_SET_PATH:
        test_set = generate_test_set(
            n=test_set_size, seed=test_set_seed, num_cmgs=num_cmgs
        )
        init_q = test_set["init_bus_quat"]
        target_q = test_set["target_quat"]
    else:
        raise FileNotFoundError(f"Could not find eval test-set CSV: {test_set_csv}")
    n = init_q.shape[0]
    base = SatelliteEnv(
        num_cmgs=num_cmgs,
        num_envs=n,
        backend="mujoco-torch",
        device=device,
        seed=seed,
        max_episode_steps=max_steps,
        action_scale=action_scale,
        singularity_weight=singularity_weight,
        singularity_clamp_min=singularity_clamp_min,
        singularity_mode=singularity_mode,
        singularity_exp_k=singularity_exp_k,
        omega_weight=omega_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        frame_skip=frame_skip,
        compile_step=compile_step,
        compile_kwargs={"dynamic": False} if compile_step else None,
    )
    env = TransformedEnv(base)
    env.append_transform(TestSetPrimer(init_q, target_q, device=device))
    env.append_transform(
        CatTensors(
            in_keys=POLICY_OBS_KEYS,
            out_key="observation",
            dim=-1,
            del_keys=False,
            sort=False,
        )
    )
    if use_obs_norm:
        if obs_norm_stats is None:
            raise ValueError(
                "make_eval_env(use_obs_norm=True) requires obs_norm_stats "
                "so the eval env shares normalization with training. "
                "Either compute stats once and save them, or load from "
                "disk -- or pass use_obs_norm=False to skip the transform."
            )
        _attach_obs_norm(env, obs_norm_stats, device)
    env.append_transform(StepCounter(max_steps=max_steps))
    if reward_scale != 1.0:
        env.append_transform(
            RewardScaling(loc=0.0, scale=reward_scale, in_keys=["reward"])
        )
    env.append_transform(RewardSum())
    return env


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def _obs_dim(obs_spec: TensorSpec) -> int:
    """Resolve the policy-input dimension from the (possibly composite) spec."""
    if isinstance(obs_spec, Composite):
        return obs_spec["observation"].shape[-1]
    return obs_spec.shape[-1]


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _activation_class(name: str | type[nn.Module]) -> type[nn.Module]:
    if isinstance(name, str):
        try:
            return _ACTIVATIONS[name.lower()]
        except KeyError as e:
            raise ValueError(
                f"Unknown activation {name!r}; valid: {sorted(_ACTIVATIONS)}."
            ) from e
    return name


def _small_gain_last_linear(module: nn.Module, gain: float = 0.01) -> None:
    """Re-init the *last* :class:`nn.Linear` in ``module`` with
    orthogonal weights at ``gain`` and zero bias.

    Standard policy-gradient init from Mnih et al. and the PPO paper:
    keeps the output near zero so the policy starts as a near-uniform
    sample (loc = 0, scale = whatever the param extractor emits) and
    avoids the "confidently wrong" behaviour from default Kaiming init
    on a 256-wide last layer.
    """
    last_linear = None
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            last_linear = sub
    if last_linear is None:
        raise RuntimeError("No nn.Linear found inside module to re-init.")
    nn.init.orthogonal_(last_linear.weight, gain=gain)
    if last_linear.bias is not None:
        nn.init.zeros_(last_linear.bias)


def make_actor(
    *,
    obs_spec: TensorSpec,
    action_spec: TensorSpec,
    device: torch.device,
    hidden: tuple[int, ...] = (256, 256),
    activation: str | type[nn.Module] = "relu",
    state_independent_scale: bool = False,
    layer_norm: bool = False,
    small_init_last_layer: bool = False,
    scale_init: float = 1.0,
) -> ProbabilisticActor:
    """TanhNormal actor.

    ``state_independent_scale=True`` matches the canonical PPO setup
    (scale is a learned bias). ``False`` matches SAC (head outputs both
    loc and scale). ``activation`` is the hidden-layer non-linearity;
    string aliases (``"relu"``, ``"tanh"``, ``"elu"``, ``"gelu"``,
    ``"silu"``) or a class are accepted.
    """
    in_dim = _obs_dim(obs_spec)
    action_dim = action_spec.shape[-1]
    act_cls = _activation_class(activation)
    norm_kwargs = (
        {
            "norm_class": nn.LayerNorm,
            "norm_kwargs": [{"normalized_shape": h} for h in hidden],
        }
        if layer_norm
        else {}
    )

    if state_independent_scale:
        body = MLP(
            in_features=in_dim,
            num_cells=list(hidden),
            out_features=action_dim,
            activation_class=act_cls,
            device=device,
            **norm_kwargs,
        )
        mlp = nn.Sequential(
            body,
            AddStateIndependentNormalScale(action_dim, scale_lb=1e-4).to(device),
        )
    else:
        body = MLP(
            in_features=in_dim,
            num_cells=list(hidden),
            out_features=2 * action_dim,
            activation_class=act_cls,
            device=device,
            **norm_kwargs,
        )
        mlp = nn.Sequential(
            body,
            NormalParamExtractor(
                scale_mapping=f"biased_softplus_{scale_init:.4f}",
                scale_lb=1e-4,
            ).to(device),
        )

    if small_init_last_layer:
        _small_gain_last_linear(body, gain=0.01)

    td_module = TensorDictModule(
        mlp, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    # Pass shape-``[action_dim]`` bounds (not batched per env): vmapped
    # specs make ``space.low/high`` shape ``[num_envs, action_dim]``, which
    # breaks broadcasting inside ``TanhNormal.update`` under ``torch.compile``
    # (the loc batch dim differs from num_envs after replay sampling).
    low_b = action_spec.space.low
    high_b = action_spec.space.high
    while low_b.ndim > 1:
        low_b = low_b[0]
    while high_b.ndim > 1:
        high_b = high_b[0]
    actor = ProbabilisticActor(
        module=td_module,
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": low_b,
            "high": high_b,
            "tanh_loc": True,
        },
        return_log_prob=True,
    )
    return actor


def make_value_critic(
    *,
    obs_spec: TensorSpec,
    device: torch.device,
    hidden: tuple[int, ...] = (256, 256),
    activation: str | type[nn.Module] = "tanh",
) -> ValueOperator:
    in_dim = _obs_dim(obs_spec)
    net = MLP(
        in_features=in_dim,
        num_cells=list(hidden),
        out_features=1,
        activation_class=_activation_class(activation),
        device=device,
    )
    return ValueOperator(net, in_keys=["observation"])


def make_qvalue_critic(
    *,
    obs_spec: TensorSpec,
    action_spec: TensorSpec,
    device: torch.device,
    hidden: tuple[int, ...] = (256, 256),
    activation: str | type[nn.Module] = "relu",
    layer_norm: bool = False,
    small_init_last_layer: bool = False,
) -> ValueOperator:
    in_dim = _obs_dim(obs_spec) + action_spec.shape[-1]
    norm_kwargs = (
        {
            "norm_class": nn.LayerNorm,
            "norm_kwargs": [{"normalized_shape": h} for h in hidden],
        }
        if layer_norm
        else {}
    )
    net = MLP(
        in_features=in_dim,
        num_cells=list(hidden),
        out_features=1,
        activation_class=_activation_class(activation),
        device=device,
        **norm_kwargs,
    )
    if small_init_last_layer:
        _small_gain_last_linear(net, gain=0.01)
    return ValueOperator(net, in_keys=["observation", "action"])


# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------


def make_eval_metrics_fn(
    categories: list[str],
) -> Callable[[TensorDictBase], dict[str, float]]:
    """Return a metrics function that breaks down by ``categories``.

    ``categories`` must be the per-row category list from the same CSV
    the eval primer uses, in the same order. The eval env is built with
    ``num_envs == len(categories)`` so positions line up.
    """
    cat_tensor_idx: dict[str, list[int]] = {}
    for i, c in enumerate(categories):
        cat_tensor_idx.setdefault(c, []).append(i)

    def _flatten(td: TensorDictBase, key: NestedKey) -> torch.Tensor:
        # Eval rollouts come back with shape ``(B, T, ...)``; reduce over T.
        return td.get(key)

    def fn(td: TensorDictBase) -> dict[str, float]:
        # Episode return: collected by ``RewardSum`` under ``("next",
        # "episode_reward")``. We take the *last* value per env.
        ep_ret = _flatten(td, ("next", "episode_reward"))[..., -1, :]
        # Final attitude error (radians) at the end of the rollout.
        quat_err = _flatten(td, ("next", "quat_err"))  # (B, T, 3)
        final_err = quat_err[..., -1, :].norm(dim=-1)
        # Manipulability per step: shape (B, T, 1).
        manip = _flatten(td, ("next", "manipulability")).squeeze(-1)
        min_manip = manip.min(dim=-1).values
        mean_manip = manip.mean(dim=-1)
        sum_inv_manip = (1.0 / manip.clamp_min(1e-6)).sum(dim=-1)

        out: dict[str, float] = {
            "eval/return": ep_ret.mean().item(),
            "eval/final_attitude_error_rad": final_err.mean().item(),
            "eval/success_rate@0.10": (final_err <= 0.10).float().mean().item(),
            "eval/success_rate@0.05": (final_err <= 0.05).float().mean().item(),
            "eval/min_manipulability": min_manip.mean().item(),
            "eval/mean_manipulability": mean_manip.mean().item(),
            "eval/sum_inv_manipulability": sum_inv_manip.mean().item(),
        }
        for cat, idx_list in cat_tensor_idx.items():
            if not idx_list:
                continue
            idx = torch.tensor(idx_list, device=ep_ret.device)
            out[f"eval/{cat}/return"] = ep_ret.index_select(0, idx).mean().item()
            out[f"eval/{cat}/final_err"] = final_err.index_select(0, idx).mean().item()
            out[f"eval/{cat}/success@0.10"] = (
                (final_err.index_select(0, idx) <= 0.10).float().mean().item()
            )
            out[f"eval/{cat}/min_manipulability"] = (
                min_manip.index_select(0, idx).mean().item()
            )
        return out

    return fn


# ---------------------------------------------------------------------------
# WandB key bootstrap
# ---------------------------------------------------------------------------


def setup_wandb_key() -> None:
    """No-op stub kept for backwards compatibility.

    Configure WandB outside this module, for example by exporting
    ``WANDB_API_KEY`` or running ``wandb login``, before enabling
    WandB logging in the example scripts.
    """


# ---------------------------------------------------------------------------
# CLI: dump the test set CSV
# ---------------------------------------------------------------------------


def _cli_generate_test_set(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="examples.satellite._utils generate-test-set")
    p.add_argument("--out", default=str(DEFAULT_TEST_SET_PATH))
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-cmgs", type=int, default=4)
    args = p.parse_args(argv)
    test_set = generate_test_set(n=args.n, seed=args.seed, num_cmgs=args.num_cmgs)
    dump_test_set_csv(test_set, args.out)
    cats = test_set["category"]
    counts = {c: cats.count(c) for c in CATEGORIES}
    print(f"Wrote {args.n} rows to {args.out}")
    print("Per-category counts:", counts)
    err = _attitude_error_norm(test_set["init_bus_quat"], test_set["target_quat"])
    print(f"Mean attitude error (rad): {err.mean().item():.3f}")
    print(f"Max  attitude error (rad): {err.max().item():.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: python -m examples.satellite._utils generate-test-set [...]",
            file=sys.stderr,
        )
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "generate-test-set":
        return _cli_generate_test_set(rest)
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
