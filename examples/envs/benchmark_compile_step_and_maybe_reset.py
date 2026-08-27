# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark ``env.step_and_maybe_reset`` with and without ``env.compile()``.

This benchmark uses a branchless native-autoreset counting environment so the
compiled graph measures TorchRL's transform/reset machinery rather than Python
control flow in the toy environment. The transform chain is:

    ``StepCounter -> [RandomTruncationTransform] -> RewardSum -> VecNormV2``

Reference results from an Apple Silicon CPU run with torch
``2.13.0.dev20260523``, 10 torch threads, batch size ``16_384``, compile time
excluded:

=======================  ======  ==========  =======  ===================
Case                     Mode    Median      Speedup  Reset envs / step
=======================  ======  ==========  =======  ===================
Synchronized resets      eager    584.86 us   1.00x   128.7
Synchronized resets      default  309.02 us   1.89x   128.7
Random truncation resets eager    996.22 us   1.00x   3708.0
Random truncation resets default  401.28 us   2.48x   3708.0
=======================  ======  ==========  =======  ===================

The random truncation case uses
``RandomTruncationTransform(prob=1.0, min_horizon=1, max_horizon=8)`` to
desynchronize resets across sub-envs.

Example:
    python examples/envs/benchmark_compile_step_and_maybe_reset.py \\
        --batch-size 16384 --steps 200 --repeats 7 --warmup 25
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch
from tensordict import TensorDict

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None

from torchrl.data.tensor_specs import Binary, Categorical, Composite, Unbounded
from torchrl.envs import (
    Compose,
    EnvBase,
    RandomTruncationTransform,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNormV2,
)


class BranchlessNativeAutoResetCountingEnv(EnvBase):
    """Tensor-only counting env that resets internally without Python branching."""

    def __init__(self, max_steps: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        shape = (*self.batch_size, 1)
        self.observation_spec = Composite(
            observation=Unbounded(shape, dtype=torch.float32, device=self.device),
            shape=self.batch_size,
            device=self.device,
        )
        self.reward_spec = Unbounded(shape, device=self.device)
        self.done_spec = Categorical(
            2, dtype=torch.bool, shape=shape, device=self.device
        )
        self.action_spec = Binary(n=1, shape=shape, device=self.device)
        self.register_buffer("count", torch.zeros(shape, dtype=torch.float32))

    def _reset(self, tensordict=None, **kwargs):
        self.count.zero_()
        done = self.count.bool()
        return TensorDict(
            {
                "observation": self.count.clone(),
                "done": done,
                "terminated": done,
            },
            self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        next_count = self.count + tensordict["action"].to(torch.float32)
        done = next_count > self.max_steps
        count = torch.where(done, torch.zeros_like(next_count), next_count)
        self.count.copy_(count)
        return TensorDict(
            {
                "observation": count.clone(),
                "done": done,
                "terminated": done,
                "reward": torch.ones_like(count),
            },
            self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        return None


def make_env(args: argparse.Namespace, *, random_truncation: bool):
    transforms = [StepCounter()]
    if random_truncation:
        transforms.append(
            RandomTruncationTransform(
                prob=args.truncation_prob,
                min_horizon=args.min_horizon,
                max_horizon=args.max_horizon,
            )
        )
    transforms.extend([RewardSum(), VecNormV2(in_keys=["observation"])])

    env = TransformedEnv(
        BranchlessNativeAutoResetCountingEnv(
            args.env_max_steps,
            batch_size=[args.batch_size],
            device=args.device,
        ),
        Compose(*transforms),
    )
    env._torchrl_native_autoreset = True
    env.full_observation_spec
    tensordict = env.reset()
    return env, tensordict


def run_steps(env, tensordict, action, num_steps: int):
    reset_count = 0
    for _ in range(num_steps):
        tensordict.update(action)
        step_data, tensordict = env.step_and_maybe_reset(tensordict)
        reset_count += int(step_data["next", "done"].sum().item())
    return tensordict, reset_count


COMPILE_MODES = {
    "default": {},
    "reduce-overhead": {"mode": "reduce-overhead"},
    "max-autotune": {"mode": "max-autotune"},
}


def measure(args: argparse.Namespace, *, compile_mode: str, random_truncation: bool):
    torch.manual_seed(args.seed)
    env, tensordict = make_env(args, random_truncation=random_truncation)
    action = env.full_action_spec.one()
    if compile_mode != "eager":
        if dynamo is not None:
            dynamo.reset()
        compile_kwargs = dict(COMPILE_MODES[compile_mode])
        compile_kwargs["fullgraph"] = True
        if args.backend is not None:
            compile_kwargs["backend"] = args.backend
        env.compile(warmup=args.compile_warmup, **compile_kwargs)

    tensordict, _ = run_steps(env, tensordict, action, args.warmup)
    times = []
    resets = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        tensordict, reset_count = run_steps(env, tensordict, action, args.steps)
        if env.device is not None and torch.device(env.device).type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / args.steps)
        resets.append(reset_count / args.steps)

    env.eager()
    env.close()
    gc.collect()
    return {
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "resets": statistics.mean(resets),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark env.step_and_maybe_reset with env.compile()."
    )
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument(
        "--compile-warmup",
        type=int,
        default=1,
        help=(
            "Number of eager calls before tracing kicks in. Forwarded to "
            "env.compile(warmup=...). Stabilizes the input TensorDict layout "
            "so the first compiled call sees the steady-state schema."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument(
        "--compile-modes",
        nargs="+",
        default=["default", "reduce-overhead", "max-autotune"],
        choices=sorted(COMPILE_MODES),
        help="Compile modes to compare against eager.",
    )
    parser.add_argument("--env-max-steps", type=int, default=128)
    parser.add_argument("--min-horizon", type=int, default=1)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--truncation-prob", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"torch={torch.__version__} threads={torch.get_num_threads()} "
        f"batch={args.batch_size} device={args.device or 'cpu'}"
    )
    print("case,mode,median_us,mean_us,speedup_vs_eager,resets_per_step")
    for random_truncation, name in (
        (False, "sync_resets"),
        (True, "with_random_truncation"),
    ):
        eager = measure(args, compile_mode="eager", random_truncation=random_truncation)
        print(
            f"{name},"
            "eager,"
            f"{eager['median'] * 1e6:.2f},"
            f"{eager['mean'] * 1e6:.2f},"
            "1.000,"
            f"{eager['resets']:.1f}"
        )
        for compile_mode in args.compile_modes:
            compiled = measure(
                args, compile_mode=compile_mode, random_truncation=random_truncation
            )
            print(
                f"{name},"
                f"{compile_mode},"
                f"{compiled['median'] * 1e6:.2f},"
                f"{compiled['mean'] * 1e6:.2f},"
                f"{eager['median'] / compiled['median']:.3f},"
                f"{compiled['resets']:.1f}"
            )


if __name__ == "__main__":
    main()
