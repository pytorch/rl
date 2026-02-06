# IsaacLab Integration Guide

This document covers everything learned about integrating IsaacLab with TorchRL and Dreamer, including traps, gotchas, and how-to.

## What is IsaacLab?

IsaacLab (formerly Isaac Gym) is NVIDIA's GPU-accelerated robotics simulation platform built on Omniverse. It runs thousands of parallel environments on a single GPU, giving orders-of-magnitude higher throughput than CPU-based simulators like MuJoCo/DMControl.

- **Docker image**: `nvcr.io/nvidia/isaac-lab:2.3.0`
- **Python env**: Inside the Docker image at `/workspace/isaaclab/isaaclab.sh`
- **Docs**: https://isaac-sim.github.io/IsaacLab/v2.3.0/

## Critical: Import Order

**The single most important thing to know**: IsaacLab's `AppLauncher` MUST be initialized BEFORE importing `torch`.

```python
# CORRECT -- AppLauncher first, then torch
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args(["--headless"])
AppLauncher(args_cli)

# NOW safe to import torch
import torch
import torchrl
```

```python
# WRONG -- will cause mysterious errors or crashes
import torch
from isaaclab.app import AppLauncher  # too late!
```

This is why we have a separate `dreamer_isaac.py` script instead of adding IsaacLab support directly to `dreamer.py`.

## Key Properties of IsaacLab Environments

### Pre-vectorized

IsaacLab environments are already batched internally. A single `gym.make("Isaac-Ant-v0")` creates ~4096 parallel environments running on the GPU. There is **no need** to wrap with `ParallelEnv`.

```python
env = gym.make("Isaac-Ant-v0")
env = IsaacLabWrapper(env)
print(env.batch_size)  # (4096,)
```

### GPU-native

All computation happens on `cuda:0`. The environment does NOT support CPU execution. The `device` parameter to `IsaacLabWrapper` defaults to `cuda:0` and should not be changed.

### State-based observations

Standard IsaacLab environments use vector observations, not pixels. The observation key is `"policy"` (not `"observation"` or `"pixels"`).

```python
env.observation_spec["policy"].shape  # (4096, obs_dim)
```

Some environments (e.g., `Isaac-Cartpole-RGB-v0`) support pixel observations, but these require `--enable_cameras` and are not the default.

### Auto-reset

IsaacLab environments auto-reset individual sub-environments when they reach a terminal state. The `IsaacLabWrapper` handles this via `VecGymEnvTransform`. The flag `allow_done_after_reset=True` is set because IsaacLab can report done immediately after reset.

### In-place tensor modification

IsaacLab modifies `terminated` and `truncated` tensors in-place. The `IsaacLabWrapper._output_transform()` clones these tensors to prevent data corruption.

## TorchRL Integration

### IsaacLabWrapper

```python
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

env = gym.make("Isaac-Ant-v0")
env = IsaacLabWrapper(env)
```

Key defaults:
- `device=cuda:0`
- `allow_done_after_reset=True`
- `convert_actions_to_numpy=False` (actions stay as tensors)

### Collector

Use a **single `Collector`** (not `MultiCollector`). The pre-vectorized env already provides massive throughput:

```python
from torchrl.collectors import Collector

collector = Collector(
    create_env_fn=env,
    policy=policy,
    frames_per_batch=40960,   # 10 env steps * 4096 envs
    storing_device="cpu",
    no_cuda_sync=True,        # IMPORTANT for CUDA envs
)
```

- `no_cuda_sync=True`: Avoids unnecessary CUDA synchronization that can cause hangs
- `storing_device="cpu"`: Move collected data to CPU for the replay buffer

### RayCollector (alternative)

If you need distributed collection across multiple GPUs/nodes, use `RayCollector`:

```python
from torchrl.collectors.distributed import RayCollector

collector = RayCollector(
    [make_env] * num_collectors,
    policy,
    frames_per_batch=8192,
    collector_kwargs={
        "trust_policy": True,
        "no_cuda_sync": True,
    },
)
```

### Replay Buffer

The `SliceSampler` needs enough sequential data. With `batch_length=50`, you need at least 50 time steps per trajectory before sampling:

```
init_random_frames >= batch_length * num_envs
                    = 50 * 4096
                    = 204,800
```

## Available Environments

List environments by running (inside the Isaac container):
```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

### Classic (MuJoCo-style)
- `Isaac-Ant-v0` -- Ant locomotion (good for validation, tested in CI)
- `Isaac-Humanoid-v0` -- Humanoid locomotion
- `Isaac-Cartpole-v0` -- Cart-pole balance

### Locomotion (quadruped/biped)
- `Isaac-Velocity-Flat-Anymal-C-v0` -- Anymal-C on flat terrain
- `Isaac-Velocity-Rough-Anymal-C-v0` -- Anymal-C on rough terrain
- `Isaac-Velocity-Flat-Unitree-Go2-v0` -- Unitree Go2 on flat terrain
- `Isaac-Velocity-Flat-H1-v0` -- Unitree H1 humanoid

### Manipulation
- `Isaac-Reach-Franka-v0` -- Franka reach
- `Isaac-Lift-Cube-Franka-v0` -- Franka lift cube
- `Isaac-Open-Drawer-Franka-v0` -- Franka open drawer
- `Isaac-Repose-Cube-Allegro-v0` -- Allegro hand in-hand manipulation

## Running on Steve

### 1) Create a job with Isaac Docker image

```bash
JOBID=$(steve job \
    --partition h200-high \
    --gpus-per-task 1 \
    --ntasks 1 \
    --time 24:00:00 \
    --job-name "dreamer-isaac" \
    --container-image nvcr.io/nvidia/isaac-lab:2.3.0 \
    --jobid-only)
```

### 2) Copy and run setup

```bash
steve cp "$JOBID" ./setup-and-run.sh :/root/setup-and-run.sh
steve step "$JOBID" 'bash /root/setup-and-run.sh --build-only'
```

### 3) Run training (detached)

```bash
steve step -d "$JOBID" 'WANDB_MODE=online bash /root/setup-and-run.sh'
```

### 4) Override the task

```bash
steve step -d "$JOBID" 'WANDB_MODE=online bash /root/setup-and-run.sh env.name=Isaac-Ant-v0'
```

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `OMNI_KIT_ACCEPT_EULA` | `yes` | Accept IsaacLab EULA (required) |
| `PYTHONNOUSERSITE` | `1` | Avoid user-site packages conflicting |
| `TD_GET_DEFAULTS_TO_NONE` | `1` | TensorDict default behavior |
| `WANDB_MODE` | `online` | Weights & Biases logging mode |

## Dreamer-Specific Notes

### Observation Key

IsaacLab uses `"policy"` as the observation key. In Dreamer's pipeline, this means:
- Encoder reads from `"policy"` (not `"observation"` or `"pixels"`)
- Decoder writes to `"reco_policy"`
- Loss keys: `world_model_loss.set_keys(pixels="policy", reco_pixels="reco_policy")`

This is handled automatically when `cfg.env.backend == "isaaclab"` in `make_dreamer()`.

### No Separate Eval Environment

IsaacLab typically runs one simulation per process. Creating a second env for evaluation is unreliable. Instead:
- Track episode rewards from completed episodes in the training data
- The 4096 parallel envs provide statistically robust reward estimates
- Periodic deterministic evaluation can be done by switching the policy's exploration type

### Throughput

Measured on a single NVIDIA H200 with 4096 parallel envs:
- **~15,600 fps** data collection throughput
- **~7.5 optim steps/sec** during gradient updates (without compile/autocast)
- One env step = 4096 frames
- 50 steps per collection = 204,800 frames
- This is orders of magnitude faster than DMControl (which needs 7 multiprocess workers)

### Config Differences from DMControl

| Parameter | DMControl | IsaacLab |
|-----------|-----------|----------|
| `env.backend` | `dm_control` | `isaaclab` |
| `env.from_pixels` | `True` | `False` |
| `collector.num_collectors` | `7` | N/A (single Collector) |
| `collector.frames_per_batch` | `1000` | `204800` |
| `collector.init_random_frames` | `10000` | `204800` |
| `optimization.compile.enabled` | `True` | `False` (container compat) |
| `optimization.autocast` | `bfloat16` | `false` (container compat) |

## Gotchas and Traps

1. **Import order**: AppLauncher MUST be initialized before `import torch`. This cannot be stressed enough.

2. **Single simulation per process**: Don't try to create two IsaacLab environments in the same process.

3. **`no_cuda_sync=True`**: Always set this for collectors with CUDA environments. Without it, you get mysterious hangs.

4. **EULA acceptance**: Set `OMNI_KIT_ACCEPT_EULA=yes` or the simulation won't start.

5. **Batched specs**: IsaacLab env specs include the batch dimension (e.g., shape `(4096, obs_dim)`). Use `*_spec_unbatched` properties when you need per-env shapes.

6. **Reward shape**: IsaacLab rewards are `(num_envs,)`. The wrapper unsqueezes to `(num_envs, 1)` for TorchRL compatibility.

7. **`--headless` flag**: Always pass `--headless` for server/cluster training (no display).

8. **Zombie processes**: IsaacLab can leave orphan processes. Always `pkill -9 python` before relaunching (handled in `setup-and-run.sh`).

9. **Installing torchrl in Isaac container**: Use `--no-build-isolation --no-deps` to avoid conflicts with Isaac's pre-installed torch/numpy.

10. **`TensorDictPrimer` expand_specs**: When adding primers (e.g., `state`, `belief`) to a pre-vectorized env, you MUST pass `expand_specs=True` to `TensorDictPrimer`. Otherwise the primer shapes `()` conflict with the env's batch_size `(4096,)`.

11. **Model-based env spec double-batching**: `model_based_env.set_specs_from_env(batched_env)` copies specs with batch dims baked in. The model-based env then double-batches actions during sampling (e.g., `(4096, 4096, 8)` instead of `(4096, 8)`). **Fix**: unbatch the model-based env's specs after copying:
    ```python
    model_based_env.set_specs_from_env(test_env)
    if test_env.batch_size:
        idx = (0,) * len(test_env.batch_size)
        model_based_env.__dict__["_output_spec"] = model_based_env.__dict__["_output_spec"][idx]
        model_based_env.__dict__["_input_spec"] = model_based_env.__dict__["_input_spec"][idx]
        model_based_env.empty_cache()
    ```

12. **`torch.compile` incompatibility**: The `tensordict` version bundled in the IsaacLab container (`_isaac_sim/kit/python/lib/python3.11/site-packages/tensordict/`) has a different internal API (`_tensordict` attribute) than the latest git version. `torch.compile`/dynamo traces through TensorDict internals and triggers `AttributeError: 'TensorDict' object has no attribute '_tensordict'`. **Fix**: disable `torch.compile` and `autocast` in the Isaac config until the container's tensordict is updated.

13. **SliceSampler with strict_length=False**: The sampler may return fewer elements than `batch_size` (e.g., 9999 instead of 10000) when some trajectories are shorter than `slice_len`. This causes `reshape(-1, batch_length)` to fail. **Fix**: truncate the sample to make `numel` divisible by `batch_length`:
    ```python
    sample = replay_buffer.sample()
    numel = sample.numel()
    usable = (numel // batch_length) * batch_length
    if usable < numel:
        sample = sample[:usable]
    sample = sample.reshape(-1, batch_length)
    ```

14. **`frames_per_batch` vs `batch_length`**: Each collection adds `frames_per_batch / num_envs` time steps per env. The `SliceSampler` needs contiguous sequences of at least `batch_length` steps within a single trajectory. Ensure `frames_per_batch >= batch_length * num_envs` for the initial collection, or that `init_random_frames >= batch_length * num_envs`.

15. **TERM environment variable**: The Isaac container may not have `TERM` set, causing `isaaclab.sh` to print `'': unknown terminal type`. **Fix**: `export TERM="${TERM:-xterm}"`.

16. **`rsync` not installed**: The IsaacLab container does not ship with `rsync`. Install with `apt-get install -y rsync` if needed for file transfer.

17. **`gym.make` requires `cfg` argument**: IsaacLab environments require an explicit configuration object. Resolve it dynamically:
    ```python
    spec = gymnasium.spec(env_name)
    entry = spec.kwargs["env_cfg_entry_point"]
    module_path, class_name = entry.rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(module_path), class_name)()
    env = gymnasium.make(env_name, cfg=env_cfg)
    ```
