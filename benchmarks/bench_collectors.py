"""Benchmark: AsyncBatchedCollector vs Collector vs MultiCollector.

Supports a mock pixel environment or a real Atari env (ALE/Pong-v5)
paired with a Nature-CNN policy.

Usage:
    python benchmarks/bench_collectors.py                    # mock env
    python benchmarks/bench_collectors.py --env ALE/Pong-v5  # real Atari

Collectors tested:
  1. Collector (1 env)                  -- single-process, single env
  2. Collector (ParallelEnv x N)        -- single-process, N envs in sub-procs
  3. MultiCollector (sync, x N)         -- N sub-processes, sync delivery
  4. MultiCollector (async, x N)        -- N sub-processes, async delivery
  5. AsyncBatched (env=thread, pol=thread)  -- threading pool + threading transport
  6. AsyncBatched (env=mp, pol=thread)      -- multiprocessing pool + threading transport
"""
from __future__ import annotations

import argparse
import time as time_module

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.collectors import AsyncBatchedCollector, Collector, MultiCollector
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import ConvNet, MLP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_ENVS = 8
FRAMES_PER_BATCH = 800
TOTAL_FRAMES = 8_000
WARMUP_BATCHES = NUM_ENVS
OBS_SHAPE = (4, 84, 84)  # used by mock env and policy
NUM_ACTIONS = 6
MAX_STEPS = 200


# ---------------------------------------------------------------------------
# Mock pixel environment (for --env mock)
# ---------------------------------------------------------------------------


class PixelMockEnv(EnvBase):
    """Fake pixel env producing (4, 84, 84) float observations with configurable step cost."""

    def __init__(self, max_steps=MAX_STEPS, step_cost=0.001, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.observation_spec = Composite(
            pixels=Unbounded(
                shape=(*self.batch_size, *OBS_SHAPE),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=self.batch_size,
        )
        self.action_spec = Categorical(
            n=NUM_ACTIONS,
            shape=(*self.batch_size,),
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_spec = Unbounded(
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.done_spec = Categorical(
            n=2,
            dtype=torch.bool,
            shape=(*self.batch_size, 1),
            device=self.device,
        )
        self._step_count = 0

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def _reset(self, tensordict=None, **kwargs):
        self._step_count = 0
        return TensorDict(
            {
                "pixels": torch.randn(*self.batch_size, *OBS_SHAPE, device=self.device),
                "done": torch.zeros(
                    *self.batch_size, 1, dtype=torch.bool, device=self.device
                ),
                "terminated": torch.zeros(
                    *self.batch_size, 1, dtype=torch.bool, device=self.device
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        if self.step_cost > 0:
            time_module.sleep(self.step_cost)
        self._step_count += 1
        done = self._step_count >= self.max_steps
        return TensorDict(
            {
                "pixels": torch.randn(*self.batch_size, *OBS_SHAPE, device=self.device),
                "reward": torch.randn(*self.batch_size, 1, device=self.device),
                "done": torch.full(
                    (*self.batch_size, 1), done, dtype=torch.bool, device=self.device
                ),
                "terminated": torch.full(
                    (*self.batch_size, 1), done, dtype=torch.bool, device=self.device
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )


# ---------------------------------------------------------------------------
# Atari env factory (for --env ALE/Pong-v5 etc.)
# ---------------------------------------------------------------------------


def make_atari_env(env_name, frame_skip=4):
    """Atari env with standard DQN preprocessing pipeline."""
    from torchrl.envs import (
        CatFrames,
        DoubleToFloat,
        GrayScale,
        GymEnv,
        Resize,
        StepCounter,
        ToTensorImage,
        TransformedEnv,
    )

    env = GymEnv(
        env_name,
        frame_skip=frame_skip,
        from_pixels=True,
        pixels_only=False,
        categorical_action_encoding=True,
    )
    env = TransformedEnv(env)
    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(StepCounter(max_steps=MAX_STEPS))
    env.append_transform(DoubleToFloat())
    return env


# ---------------------------------------------------------------------------
# Policy (module-level so it's picklable by MultiCollector)
# ---------------------------------------------------------------------------


class ArgmaxHead(nn.Module):
    """Wraps a Q-network and returns the argmax action."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pixels):
        logits = self.net(pixels)
        return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Picklable env factory (module-level for multiprocessing)
# ---------------------------------------------------------------------------

_ENV_NAME = "mock"  # set by main() before any collector is created


def make_env_fn():
    """Module-level factory dispatching on _ENV_NAME."""
    if _ENV_NAME == "mock":
        return PixelMockEnv()
    return make_atari_env(_ENV_NAME)


# ---------------------------------------------------------------------------


def make_policy(num_actions=NUM_ACTIONS):
    """Nature-CNN: Conv(32,64,64) + MLP(512) -> argmax action."""
    cnn = ConvNet(
        activation_class=nn.ReLU,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    )
    with torch.no_grad():
        cnn_out = cnn(torch.ones(OBS_SHAPE))
    mlp = MLP(
        in_features=cnn_out.shape[-1],
        activation_class=nn.ReLU,
        out_features=num_actions,
        num_cells=[512],
    )
    return TensorDictModule(
        ArgmaxHead(nn.Sequential(cnn, mlp)),
        in_keys=["pixels"],
        out_keys=["action"],
    )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def bench(name: str, factory, warmup=WARMUP_BATCHES, target_frames=TOTAL_FRAMES):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    collector = factory()
    total = 0
    idx = 0
    t0 = None

    try:
        for batch in collector:
            idx += 1
            n = batch.numel()
            if idx <= warmup:
                print(f"  [warmup {idx}/{warmup}] {n} frames")
                continue
            if t0 is None:
                t0 = time_module.perf_counter()
            total += n
            elapsed_so_far = time_module.perf_counter() - t0
            fps_so_far = total / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"  [batch {idx}] {n} frames  total={total}  ({fps_so_far:.0f} fps)")
            if total >= target_frames:
                break
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        elapsed = time_module.perf_counter() - t0 if t0 is not None else float("inf")
        try:
            collector.shutdown()
        except Exception:
            pass

    fps = total / elapsed if elapsed > 0 else 0
    print(f"  --> {total} frames in {elapsed:.2f}s = {fps:.0f} fps")
    return name, fps, elapsed, total


def main():
    parser = argparse.ArgumentParser(description="Collector throughput benchmark")
    parser.add_argument(
        "--env",
        default="mock",
        help="Environment: 'mock' for PixelMockEnv, or a gym env id like 'ALE/Pong-v5'",
    )
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--frames-per-batch", type=int, default=FRAMES_PER_BATCH)
    parser.add_argument("--total-frames", type=int, default=TOTAL_FRAMES)
    args = parser.parse_args()

    num_envs = args.num_envs
    frames_per_batch = args.frames_per_batch
    total_frames = args.total_frames

    # Set module-level env name so make_env_fn is picklable
    global _ENV_NAME
    _ENV_NAME = args.env

    if args.env == "mock":
        env_label = f"PixelMockEnv {OBS_SHAPE}, step_cost=1ms"
        num_actions = NUM_ACTIONS
    else:
        env_label = args.env
        probe = make_env_fn()
        num_actions = probe.action_spec.space.n
        print(
            f"Env: {args.env}, pixels: {probe.observation_spec['pixels'].shape}, actions: {num_actions}"
        )
        probe.close()

    def policy_factory():
        return make_policy(num_actions=num_actions)

    # Verify
    env = make_env_fn()
    check_env_specs(env)
    env.close()

    results = []

    # All collectors use total_frames=-1 (endless); the bench harness
    # controls stopping after `target_frames` timed frames so that
    # warmup batches don't eat into the measurement window.

    # 1. Collector -- single env
    results.append(
        bench(
            "Collector (1 env)",
            lambda: Collector(
                create_env_fn=make_env_fn,
                policy=policy_factory(),
                frames_per_batch=frames_per_batch,
                total_frames=-1,
                trust_policy=True,
                use_buffers=False,
            ),
            target_frames=total_frames,
        )
    )

    # 2. Collector -- ParallelEnv
    results.append(
        bench(
            f"Collector (ParallelEnv x{num_envs})",
            lambda: Collector(
                create_env_fn=ParallelEnv(num_envs, make_env_fn),
                policy=policy_factory(),
                frames_per_batch=frames_per_batch,
                total_frames=-1,
                trust_policy=True,
                use_buffers=False,
            ),
            target_frames=total_frames,
        )
    )

    # 3. MultiCollector sync
    results.append(
        bench(
            f"MultiCollector sync (x{num_envs})",
            lambda: MultiCollector(
                create_env_fn=[make_env_fn] * num_envs,
                policy_factory=policy_factory,
                frames_per_batch=frames_per_batch,
                total_frames=-1,
                sync=True,
            ),
            target_frames=total_frames,
        )
    )

    # 4. MultiCollector async
    results.append(
        bench(
            f"MultiCollector async (x{num_envs})",
            lambda: MultiCollector(
                create_env_fn=[make_env_fn] * num_envs,
                policy_factory=policy_factory,
                frames_per_batch=frames_per_batch,
                total_frames=-1,
                sync=False,
            ),
            target_frames=total_frames,
        )
    )

    # 5. AsyncBatchedCollector (env=threading, policy=threading)
    results.append(
        bench(
            f"AsyncBatched env=thread pol=thread (x{num_envs})",
            lambda: AsyncBatchedCollector(
                create_env_fn=[make_env_fn] * num_envs,
                policy=policy_factory(),
                frames_per_batch=frames_per_batch,
                total_frames=-1,
                max_batch_size=num_envs,
                env_backend="threading",
            ),
            target_frames=total_frames,
        )
    )

    # 6. AsyncBatchedCollector (env=multiprocessing, policy=threading)
    results.append(
        bench(
            f"AsyncBatched env=mp pol=thread (x{num_envs})",
            lambda: AsyncBatchedCollector(
                create_env_fn=[make_env_fn] * num_envs,
                policy=policy_factory(),
                frames_per_batch=frames_per_batch,
                total_frames=-1,
                max_batch_size=num_envs,
                env_backend="multiprocessing",
            ),
            target_frames=total_frames,
        )
    )

    # Summary
    print("\n")
    print("=" * 70)
    print("  THROUGHPUT SUMMARY  (higher FPS is better)")
    print(f"  Env: {env_label}")
    print(f"  Nature-CNN policy, {num_envs} envs")
    print(f"  {total_frames} total frames, {frames_per_batch} frames/batch")
    print("=" * 70)
    print(f"  {'Collector':<45s} {'FPS':>8s}  {'Time':>7s}")
    print(f"  {'-'*45} {'-'*8}  {'-'*7}")
    for name, fps, elapsed, _total in sorted(results, key=lambda x: -x[1]):
        print(f"  {name:<45s} {fps:>8.0f}  {elapsed:>6.2f}s")
    print()


if __name__ == "__main__":
    main()
