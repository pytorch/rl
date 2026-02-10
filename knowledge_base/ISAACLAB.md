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

### 2-GPU Async Pipeline (recommended for production)

For maximum throughput, use 2 GPUs with a background collection thread:
- **GPU 0 (`sim_device`)**: IsaacLab simulation + collection policy inference
- **GPU 1 (`train_device`)**: Model training (world model, actor, value gradients)

```python
import copy, threading
from tensordict import TensorDict

# Deep copy policy to sim_device for collection
collector_policy = copy.deepcopy(policy).to(sim_device)

# Background thread for continuous collection
def collect_loop(collector, replay_buffer, stop_event):
    for data in collector:
        replay_buffer.extend(data)
        if stop_event.is_set():
            break

# Main thread: train on train_device
for optim_step in range(total_steps):
    batch = replay_buffer.sample()
    train(batch)  # All on cuda:1
    # Periodic weight sync: training policy -> collector policy
    if optim_step % sync_every == 0:
        weights = TensorDict.from_module(policy)
        collector.update_policy_weights_(weights)
```

Key points:
- Both CUDA operations release the GIL, so they truly overlap
- Must pass `TensorDict.from_module(policy)` to `update_policy_weights_()`, not the module itself (the container's old tensordict doesn't handle `.data` on modules)
- Set `CUDA_VISIBLE_DEVICES=0,1` to expose 2 GPUs (IsaacLab defaults to only GPU 0)
- Falls back gracefully to single-GPU if only 1 GPU available

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

#### Single GPU (synchronous collect-then-train)
Measured on a single NVIDIA H200, 4096 parallel envs, batch_size=10k:
- **~15,600 fps** data collection throughput
- **~7.0 optim steps/sec** during gradient updates (FP32, no compile)
- **~55s for 200 steps** end-to-end (3.6 ops/s)
- 50% time collecting, 50% training (perfectly balanced bottleneck)

#### 2-GPU async pipeline (recommended)
GPU 0 = IsaacLab sim + collection, GPU 1 = training. Background thread.
With bfloat16 autocast, torch.compile (sub-modules), batch_size=50k:
- **~78k-137k fps** continuous collection (never pauses)
- **~2.9-3.5 optim steps/sec** with 50k batch (5x more gradient signal per step)
- Collection and training fully overlap on separate GPUs
- **~9.5h ETA for 100k steps** on H200 (Anymal-C locomotion)

#### Key insight: CPU replay buffer is the bottleneck
With async 2-GPU, the replay buffer `extend()` (collector thread) and `sample()` (training thread) both touch CPU memory, causing GIL contention. GPU-resident replay buffer (`LazyTensorStorage` on cuda:1) eliminates the CPU-GPU transfer at sample time but adds it at extend time.

### Config Differences from DMControl

| Parameter | DMControl | IsaacLab |
|-----------|-----------|----------|
| `env.backend` | `dm_control` | `isaaclab` |
| `env.from_pixels` | `True` | `False` |
| `collector.num_collectors` | `7` | N/A (single Collector) |
| `collector.frames_per_batch` | `1000` | `204800` |
| `collector.init_random_frames` | `10000` | `204800` |
| `optimization.compile.enabled` | `True` | `True` (sub-modules only) |
| `optimization.autocast` | `bfloat16` | `bfloat16` |
| `replay_buffer.batch_size` | `10000` | `50000` |
| `replay_buffer.gpu_storage` | N/A | `true` |

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

12. **`torch.compile` with TensorDict**: Compiling full loss modules crashes because dynamo traces through TensorDict internals. **Fix**: compile individual MLP sub-modules (encoder, decoder, reward_model, value_model) with `torch._dynamo.config.suppress_errors = True`. Do NOT compile RSSM (sequential, shared with collector) or loss modules (heavy TensorDict use). This gives dynamo the tensor-level optimizations while falling back to eager for TensorDict ops.

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

## Pixel-Based Observations (TiledCamera)

IsaacLab's default ManagerBased environments only provide state vectors (joint positions, velocities, etc.). For pixel-based RL (e.g., Dreamer with CNN encoder/decoder), you need to add a `TiledCamera` sensor.

### How It Works

`TiledCamera` renders all environments in a single batched pass on the GPU, producing `[num_envs, H, W, C]` uint8 tensors efficiently. Much faster than per-env camera rendering.

### Setup Steps

1. **Enable cameras**: Pass `--enable_cameras` to `AppLauncher`. Without this, rendering APIs are not initialised.

2. **Add TiledCameraCfg to the scene config** before calling `gym.make`:
    ```python
    from isaaclab.sensors import TiledCameraCfg
    import isaaclab.sim as sim_utils

    env_cfg.scene.tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-3.0, 0.0, 2.0),  # behind and above the robot
            rot=(0.9945, 0.0, 0.1045, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 20.0),
        ),
        width=64, height=64,
    )
    ```

3. **Read camera data**: After each `env.step()`, read from `scene["tiled_camera"].data.output["rgb"]`. In TorchRL, this is done via `IsaacCameraReadTransform`.

4. **Reduce num_envs**: Rendering is expensive. 512 cameras on RTX 4090 is the recommended max. For 64×64 images, 256 envs is a safe starting point on A100/H100.

5. **Increase env_spacing**: Set `env_cfg.scene.env_spacing = 8.0` or higher to prevent the camera from seeing neighbouring environments.

### Camera Position for ANYmal-C

The `offset.pos` is in world coordinates relative to the environment origin. For the ANYmal-C quadruped (base at ~0.5m height):
- **Rear-elevated**: `pos=(-3.0, 0.0, 2.0)` – sees the full body from behind
- **Side view**: `pos=(0.0, -3.0, 1.5)` – captures gait from the side
- **Top-down**: `pos=(0.0, 0.0, 5.0)` – overhead view

The rotation quaternion `(w, x, y, z) = (0.9945, 0.0, 0.1045, 0.0)` applies a slight downward pitch (≈12°).

### Gotchas

18. **`from_pixels` ignored for IsaacLab**: The `from_pixels` parameter in `GymEnv` / `IsaacLabWrapper` does NOT add pixel observations. You must add a TiledCamera sensor to the scene config manually.

19. **Camera data is NOT in the observation manager**: ManagerBased envs don't include camera data in their observation groups. The camera data must be read separately from `scene["tiled_camera"].data.output[...]` via a custom transform.

20. **Memory**: Pixel replay buffers are large. 500K frames at 64×64×3 float32 ≈ 24 GB on disk (memmap). Strip intermediate `pixels_int` tensors before storing.
