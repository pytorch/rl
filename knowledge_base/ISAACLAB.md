# IsaacLab Guide

This document covers how to install, run, and debug IsaacLab on a cluster, plus common pitfalls.

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
```

```python
# WRONG -- will cause mysterious errors or crashes
import torch
from isaaclab.app import AppLauncher  # too late!
```

## Key Properties of IsaacLab Environments

### Pre-vectorized

IsaacLab environments are already batched internally. A single `gym.make("Isaac-Ant-v0")` creates ~4096 parallel environments running on the GPU. There is **no need** to wrap with `ParallelEnv`.

```python
env = gym.make("Isaac-Ant-v0")
print(env.num_envs)  # 4096
```

### GPU-native

All computation happens on `cuda:0`. The environment does NOT support CPU execution.

### State-based observations

Standard IsaacLab environments use vector observations, not pixels. The observation key is `"policy"` (not `"observation"` or `"pixels"`).

Some environments (e.g., `Isaac-Cartpole-RGB-v0`) support pixel observations, but these require `--enable_cameras` and are not the default.

### Auto-reset

IsaacLab environments auto-reset individual sub-environments when they reach a terminal state. Done can be reported immediately after reset.

### In-place tensor modification

IsaacLab modifies `terminated` and `truncated` tensors in-place. Downstream code should clone these tensors to prevent data corruption.

## Pixel-Based Observations (TiledCamera)

IsaacLab's default ManagerBased environments only provide state vectors. For pixel-based RL you need to add a `TiledCamera` sensor.

### How It Works

`TiledCamera` renders all environments in a single batched pass on the GPU, producing `[num_envs, H, W, C]` uint8 tensors efficiently.

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

3. **Read camera data**: After each `env.step()`, read from `scene["tiled_camera"].data.output["rgb"]`.

4. **Reduce num_envs**: Rendering is expensive. 256 envs is a safe starting point for 64×64 images on A100/H100.

5. **Increase env_spacing**: Set `env_cfg.scene.env_spacing = 8.0` or higher to prevent the camera from seeing neighbouring environments.

### Camera Position for ANYmal-C

The `offset.pos` is in world coordinates relative to the environment origin. For the ANYmal-C quadruped (base at ~0.5m height):
- **Rear-elevated**: `pos=(-3.0, 0.0, 2.0)` – sees the full body from behind
- **Side view**: `pos=(0.0, -3.0, 1.5)` – captures gait from the side
- **Top-down**: `pos=(0.0, 0.0, 5.0)` – overhead view

The rotation quaternion `(w, x, y, z) = (0.9945, 0.0, 0.1045, 0.0)` applies a slight downward pitch (≈12°).

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
| `WANDB_MODE` | `online` | Weights & Biases logging mode |

## Gotchas and Pitfalls

1. **Import order**: AppLauncher MUST be initialized before `import torch`. This cannot be stressed enough.

2. **Single simulation per process**: Don't try to create two IsaacLab environments in the same process.

3. **EULA acceptance**: Set `OMNI_KIT_ACCEPT_EULA=yes` or the simulation won't start.

4. **`--headless` flag**: Always pass `--headless` for server/cluster training (no display).

5. **Zombie processes**: IsaacLab can leave orphan processes. Always `pkill -9 python` before relaunching.

6. **`TERM` environment variable**: The Isaac container may not have `TERM` set, causing `isaaclab.sh` to print `'': unknown terminal type`. **Fix**: `export TERM="${TERM:-xterm}"`.

7. **`rsync` not installed**: The IsaacLab container does not ship with `rsync`. Install with `apt-get install -y rsync` if needed.

8. **`gym.make` requires `cfg` argument**: IsaacLab environments require an explicit configuration object. Resolve it dynamically:
    ```python
    spec = gymnasium.spec(env_name)
    entry = spec.kwargs["env_cfg_entry_point"]
    module_path, class_name = entry.rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(module_path), class_name)()
    env = gymnasium.make(env_name, cfg=env_cfg)
    ```

9. **`from_pixels` ignored for IsaacLab**: The `from_pixels` parameter in `GymEnv` does NOT add pixel observations. You must add a TiledCamera sensor to the scene config manually.

10. **Camera data is NOT in the observation manager**: ManagerBased envs don't include camera data in their observation groups. The camera data must be read separately from `scene["tiled_camera"].data.output[...]`.

11. **Memory**: Pixel replay buffers are large. 500K frames at 64×64×3 float32 ≈ 24 GB on disk (memmap).

12. **Installing torchrl in Isaac container**: Use `--no-build-isolation --no-deps` to avoid conflicts with Isaac's pre-installed torch/numpy.
