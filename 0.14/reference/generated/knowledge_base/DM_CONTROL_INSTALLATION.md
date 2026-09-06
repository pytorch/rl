# Installing dm_control

dm_control is DeepMind's software stack for physics-based simulation and
reinforcement learning environments, built on top of MuJoCo.

GitHub: https://github.com/deepmind/dm_control

## Basic Installation

```
pip install dm_control
```

This should work on most platforms with Python 3.8-3.12.

## Common Issues

### 1. labmaze build failure with Bazel 8 (Python 3.13+)

**Error:**

```
× Failed to build `labmaze==1.0.6`
ERROR: Skipping '//labmaze/cc/python:_defaults': error loading package 'labmaze/cc/python':
Unable to find package for @@[unknown repo 'bazel_skylib' requested from @@]//lib:collections.bzl
The WORKSPACE file is disabled by default in Bazel 8 (late 2024) and will be removed in Bazel 9
```

**Cause:**

- `labmaze` (a dependency of dm_control) only has prebuilt wheels for Python 3.7-3.12
- Python 3.13+ requires building from source
- Bazel 8 has deprecated WORKSPACE files, which labmaze still uses

**Solutions:**

#### Option 1: Use Python 3.12 (Recommended)

```
# Create a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install dm_control
```

#### Option 2: Downgrade Bazel to version 7.x

```
# Using Homebrew on macOS
brew install bazel@7
export PATH="/opt/homebrew/opt/bazel@7/bin:$PATH"

# Or using bazelisk with version pinning
export USE_BAZEL_VERSION=7.4.1

# Then install
pip install dm_control
```

#### Option 3: Install dm_control without labmaze

If you don't need the maze/locomotion environments that require labmaze:

```
pip install dm_control --no-deps
pip install dm-env dm-tree glfw lxml mujoco numpy pyopengl pyparsing scipy
```

### 2. EGL/rendering issues

See MUJOCO_INSTALLATION.md for rendering-related
issues, as dm_control uses MuJoCo for rendering.

#### EGL multi-GPU device selection in containers (Docker / SLURM)

When running `ParallelEnv` with pixel-based dm_control environments on a
multi-GPU machine, all rendering contends on a **single GPU** -- even if the
host has 8 GPUs. This inflates per-worker render time by ~3x (e.g. 17ms serial
→ 54ms with 8 workers sharing one GPU's EGL queue).

**Common root causes:** Inside Docker or SLURM containers, the NVIDIA
container runtime may expose only a subset of devices to EGL, or a minimal CUDA
image may omit the NVIDIA graphics userspace libraries entirely. In those
cases, `eglQueryDevicesEXT()` can return fewer devices than the node has, or
EGL initialization can fail even though CUDA and `nvidia-smi` work. Setting
`MUJOCO_EGL_DEVICE_ID` or `EGL_DEVICE_ID` to an unavailable EGL device raises:

```
RuntimeError: MUJOCO_EGL_DEVICE_ID must be an integer between 0 and 0 (inclusive), got 1.
```

Unsetting `CUDA_VISIBLE_DEVICES` in the worker does **not** help once the
container runtime has hidden devices from the driver. Conversely,
`NVIDIA_DRIVER_CAPABILITIES=compute,utility` by itself does not prove EGL is
impossible: if matching NVIDIA EGL/GLVND userspace libraries are installed
inside the container, EGL may still work.

**Note on variable naming:** dm_control uses `MUJOCO_EGL_DEVICE_ID` internally
(which maps to the same thing as MuJoCo's variable). Historically there was
also `EGL_DEVICE_ID` used by older dm_control versions. See
[dm_control#345](https://github.com/google-deepmind/dm_control/issues/345)
for the unification discussion.

**Upstream issues:**

- [mujoco#572 -- Cannot access all GPUs through EGL devices when using docker](https://github.com/google-deepmind/mujoco/issues/572)
- [dm_control#345 -- Unify EGL_DEVICE_ID with MUJOCO_EGL_DEVICE_ID](https://github.com/google-deepmind/dm_control/issues/345)

**Workarounds:**

1. **Verify the graphics userspace stack first.** Minimal CUDA containers often
omit the EGL/GLVND loader packages and NVIDIA graphics libraries. On
Debian/Ubuntu images, install the generic runtime packages:

```
sudo apt-get update
sudo apt-get install -y libegl-dev libglvnd0 libglx0 libgles2
```

Then verify the NVIDIA pieces are visible and match the host driver:

```
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head
ldconfig -p | grep -E 'libEGL_nvidia|libnvidia-eglcore|libGLX_nvidia'
ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

If the NVIDIA libraries are missing, install a matching
`libnvidia-gl-<driver-version>` package or provide a matching userspace
bundle and point `LD_LIBRARY_PATH` / `ldconfig` at it. The GLVND vendor JSON
should point EGL at `libEGL_nvidia.so.0`.
2. **Configure container for full GPU access.** If you control the container
runtime, set `NVIDIA_VISIBLE_DEVICES=all` and
include `graphics` in `NVIDIA_DRIVER_CAPABILITIES` (or use `all`) so the
driver stack and all intended GPUs are mounted. Then assign
`MUJOCO_EGL_DEVICE_ID=<worker_idx % num_gpus>` per worker process
**before** dm_control is imported (the EGL display is created at import
time). For LIBERO / robosuite environments, prefer passing
`render_gpu_device_id=<worker_idx % num_gpus>` to the environment
constructor.
3. **Run outside containers.** On bare metal, `eglQueryDevicesEXT()` correctly
returns all GPUs (plus the X server display, if any).
4. **Reduce rendering overhead.** If multi-GPU rendering is not possible:

- Lower the rendering resolution (e.g. 64x64 instead of 84x84)
- Render at a lower frequency than the simulation step (frame-skip)
- Use state-only observations where possible -- the IPC overhead is small
compared to rendering

#### No batched rendering support in MuJoCo

MuJoCo does not support batched GPU rendering -- each environment renders its
scene independently through its own OpenGL context. There is no API to submit
multiple scenes to the GPU in one call.

MuJoCo XLA (MJX) accelerates *simulation* on GPU via JAX but still requires
copying data back to CPU for rendering through the standard `mujoco.Renderer`
pipeline. See [mujoco#1604](https://github.com/google-deepmind/mujoco/issues/1604)
for discussion on batched rendering support.

### 3. macOS ARM64 (Apple Silicon) specific issues

On Apple Silicon Macs, ensure you're using native ARM Python, not Rosetta:

```
# Check architecture
python -c "import platform; print(platform.machine())"
# Should output: arm64
```

If it outputs `x86_64`, you're running under Rosetta. Install native ARM Python:

```
# Using Homebrew
brew install python@3.12

# Or using pyenv
arch -arm64 pyenv install 3.12
```

## Verifying Installation

```
from dm_control import suite

# List available environments
print(suite.BENCHMARKING)

# Create an environment
env = suite.load("cheetah", "run")
timestep = env.reset()
print(timestep.observation)
```