# Installing dm_control

dm_control is DeepMind's software stack for physics-based simulation and 
reinforcement learning environments, built on top of MuJoCo.

GitHub: https://github.com/deepmind/dm_control

## Basic Installation

```bash
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
```bash
# Create a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install dm_control
```

#### Option 2: Downgrade Bazel to version 7.x
```bash
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
```bash
pip install dm_control --no-deps
pip install dm-env dm-tree glfw lxml mujoco numpy pyopengl pyparsing scipy
```

### 2. EGL/rendering issues

See [MUJOCO_INSTALLATION.md](./MUJOCO_INSTALLATION.md) for rendering-related 
issues, as dm_control uses MuJoCo for rendering.

### 3. macOS ARM64 (Apple Silicon) specific issues

On Apple Silicon Macs, ensure you're using native ARM Python, not Rosetta:
```bash
# Check architecture
python -c "import platform; print(platform.machine())"
# Should output: arm64
```

If it outputs `x86_64`, you're running under Rosetta. Install native ARM Python:
```bash
# Using Homebrew
brew install python@3.12

# Or using pyenv
arch -arm64 pyenv install 3.12
```

## Verifying Installation

```python
from dm_control import suite

# List available environments
print(suite.BENCHMARKING)

# Create an environment
env = suite.load("cheetah", "run")
timestep = env.reset()
print(timestep.observation)
```
