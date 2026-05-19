# IsaacLab Guide

This guide covers the pieces that tend to matter when running IsaacLab with
TorchRL on headless GPU machines: import order, rendering dependencies,
TorchRL wrappers, and common installation/debugging issues.

## Must-know rules

### Launch Isaac before importing torch

IsaacLab's `AppLauncher` should be initialized before importing `torch` in a
process that will own an Isaac simulation.

```python
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args(["--headless"])
AppLauncher(args_cli)

import torch
```

Avoid this order:

```python
import torch
from isaaclab.app import AppLauncher
```

### One simulation per process

Do not create multiple IsaacLab simulations in the same process. Use separate
processes for independent collectors, evaluators, or renderers.

### Environments are GPU-native and pre-vectorized

IsaacLab environments are already batched internally. A single `gym.make(...)`
can create thousands of sub-environments on the GPU, so wrapping with a generic
`ParallelEnv` is usually unnecessary.

Standard manager-based tasks expose state observations under the `"policy"`
observation key. Pixel observations are not added by `from_pixels`; add a camera
sensor explicitly when pixels are needed.

### Set non-interactive runtime variables

For cluster or container jobs, set the EULA/headless-related environment
variables before runtime:

```bash
export OMNI_KIT_ACCEPT_EULA=yes
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_DISABLE_CUP=1
export OMNI_KIT_ALLOW_ROOT=1
export TERM="${TERM:-xterm}"
```

## Minimal state-based environment

```python
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args(["--headless"])
AppLauncher(args_cli)

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

cfg = AntEnvCfg()
cfg.scene.num_envs = 4096
cfg.sim.device = "cuda:0"
env = IsaacLabWrapper(gym.make("Isaac-Ant-v0", cfg=cfg), device="cuda:0")
td = env.reset()
print(td["policy"].shape)
```

## Tiled-camera rendering

Manager-based environments do not put camera data in the observation manager by
default. Add a tiled camera to the scene config before `gym.make(...)` and launch
with `--enable_cameras`; without it, camera/rendering APIs are not initialized.

For TorchRL, prefer `IsaacLabWrapper.add_tiled_camera_config(...)` and
`IsaacLabWrapper(..., from_tiled_camera=True)` so pixels are inserted into the
TensorDict under `"pixels"`.

```python
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

IsaacLabWrapper.add_tiled_camera_config(env_cfg, width=64, height=64)
env = IsaacLabWrapper(
    gym.make("Isaac-Ant-v0", cfg=env_cfg),
    device="cuda:0",
    from_tiled_camera=True,
)
```

If configuring the sensor manually, add a `TiledCameraCfg` to the scene config:

```python
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

env_cfg.scene.tiled_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-3.0, 0.0, 2.0),
        rot=(0.9945, 0.0, 0.1045, 0.0),
        convention="world",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    ),
    width=64,
    height=64,
)
```

`TiledCamera` renders all environments in a single batched pass on the GPU,
producing `[num_envs, H, W, C]` tensors efficiently. Rendering many
environments is expensive: start with a smaller number of envs for pixels, and
increase `env_cfg.scene.env_spacing` if neighboring envs appear in camera views.

For the ANYmal-C quadruped (base at approximately 0.5 m height), useful camera
positions include:

- rear-elevated: `pos=(-3.0, 0.0, 2.0)`;
- side view: `pos=(0.0, -3.0, 1.5)`;
- top-down: `pos=(0.0, 0.0, 5.0)`.

The rotation quaternion `(w, x, y, z) = (0.9945, 0.0, 0.1045, 0.0)` applies a
slight downward pitch (approximately 12 degrees).

## Auto-reset and per-index reset

IsaacLab environments auto-reset individual sub-environments when they reach a
terminal state. Done can be reported immediately after reset.

`IsaacLabWrapper(env, native_autoreset=True)` keeps Isaac's native post-reset
observation in `tensordict_["policy"]` and marks the terminal
`("next", "policy")` with NaN; `EnvBase.step_and_maybe_reset` then skips the
synthetic reset call. The same bridge is installed for Direct-workflow envs
(`DirectRLEnv`, `DirectMARLEnv`), not just Manager-based envs.

`IsaacLabWrapper` surfaces Isaac Lab's per-index reset and `reset_to` APIs
through the standard torchrl `"_reset"` boolean mask:

```python
env = IsaacLabWrapper(gym.make("Isaac-Ant-v0", cfg=AntEnvCfg()), native_autoreset=True)
td = env.reset()

# ... step the env a few times ...

# Reset half of the sub-envs without disturbing the others. The transform
# stack (RewardSum, InitTracker, recurrent primers, VecNormV2, ...) fires
# on the masked rows only, exactly like a normal reset.
reset_mask = torch.zeros(env.batch_size[0], 1, dtype=torch.bool, device=env.device)
reset_mask[: env.batch_size[0] // 2] = True
td.set("_reset", reset_mask)
env.reset(td)

# Snapshot and branch from a deterministic state (manager-based envs only).
snapshot = env.base_env.get_state()
# ... evolve env from `snapshot` to explore one branch ...
env.reset(td, isaac_reset_state=snapshot)  # rewind to snapshot

# Convenience method (equivalent to the call above):
env.reset_to_state(snapshot, td)
```

Gotchas:

- The per-index reset path is gated on `native_autoreset=True`. With the
  default `native_autoreset=False`, the `VecGymEnvTransform`-based obs-swap path
  already handles `step_and_maybe_reset`-driven partial resets implicitly; an
  explicit per-index reset would double-reset those envs.
- `reset_to_state` is only available on manager-based envs (`ManagerBasedEnv` /
  `ManagerBasedRLEnv`). Direct envs do not expose `reset_to`.
- `is_relative=True` interprets the snapshot pose relative to the env origin,
  which is useful for terrain-relative pose reuse.

## In-place tensor modification

IsaacLab modifies `terminated` and `truncated` tensors in-place. Downstream code
should clone these tensors to prevent data corruption.

### Headless EGL/Vulkan dependencies

Headless tiled-camera RGB rendering needs the NVIDIA graphics userspace stack,
not just CUDA. Minimal CUDA images often omit EGL/GLVND and Vulkan runtime
packages. On Debian/Ubuntu images, install the generic loader/runtime packages:

```bash
sudo apt-get update
sudo apt-get install -y libegl-dev libglvnd0 libglx0 libvulkan1 vulkan-tools
```

If using distro NVIDIA packages, install a `libnvidia-gl-<driver-version>`
package matching the host driver when available. The NVIDIA userspace libraries
must match the host driver; if package repositories provide a different patch
release, Vulkan may report `ERROR_INCOMPATIBLE_DRIVER`.

Useful checks:

```bash
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head
ldconfig -p | grep -E 'libEGL_nvidia|libnvidia-eglcore|libGLX_nvidia'
ls /usr/share/glvnd/egl_vendor.d/
ls /usr/share/vulkan/icd.d/
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary
```

When a matching driver userspace bundle is outside the default library path,
point the process at it explicitly:

```bash
export LD_LIBRARY_PATH=/path/to/nvidia/lib:${LD_LIBRARY_PATH}
export VK_ICD_FILENAMES=/path/to/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-${USER}
mkdir -p "${XDG_RUNTIME_DIR}" && chmod 700 "${XDG_RUNTIME_DIR}"
```

Some Isaac/Kit rendering paths expect the rendering device to be `cuda:0`. For a
dedicated render/eval GPU, expose a single physical GPU to the process and use
`cuda:0` inside that process:

```bash
CUDA_VISIBLE_DEVICES=<physical-render-gpu> python render_or_eval.py --device cuda:0
```

## TorchRL integration

### IsaacLab wrapper

Use `IsaacLabWrapper` around an IsaacLab Gymnasium environment. The wrapper
handles IsaacLab's in-place `terminated`/`truncated` tensors and can read tiled
camera pixels when configured with `from_tiled_camera=True`.

```python
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

IsaacLabWrapper.add_tiled_camera_config(env_cfg, width=320, height=240)
env = IsaacLabWrapper(
    gym.make("Isaac-Ant-v0", cfg=env_cfg),
    device="cuda:0",
    from_tiled_camera=True,
)
```

### Recurrent PPO collector/evaluator notes

For IsaacLab PPO with recurrent policies, keep the dataflow explicitly
on-policy:

1. Build collector policies with `policy_factory`.
2. Use collector auto-registration for `InitTracker` and recurrent-state
   primers; on transitional branches, pass `auto_register_policy_transforms=True`.
3. Use separate devices for large runs when possible: one GPU for Isaac
   collection/inference, one GPU for learner updates, and another for rendered
   eval.
4. Collect one rollout window, compute GAE over that window, train for the
   configured PPO epochs, empty the training buffer, sync weights, then collect
   again.
5. With compact rollout data, prefer shifted value estimation so batches do not
   need `("next", "policy")` rehydration.

For robust explicit weight sync in fallback collector paths:

```python
from tensordict import TensorDict

collector.update_policy_weights_(weights=TensorDict.from_module(actor).data)
```

### Rendered async eval in the recurrent PPO example

The recurrent PPO example can run rendered eval in a separate evaluator process.
Use a dedicated physical GPU for the worker and remap it to `cuda:0` inside the
worker:

```bash
python examples/collectors/isaaclab_rnn_ppo_memory.py \
  --eval \
  --eval-cuda-visible-devices <physical-render-gpu> \
  --eval-worker-device cuda:0 \
  --eval-nvidia-lib-dir /path/to/nvidia/lib \
  --eval-vulkan-icd /path/to/nvidia_icd.json \
  --eval-xdg-runtime-dir /tmp/xdg-runtime-eval
```

By default, eval should collect a requested number of complete trajectories.
Only pass `--eval-max-steps` when a hard cap is desired.

## Pip/source installation

Prefer an IsaacLab container when possible. If starting from a generic CUDA
image, a pip/source flow can work, but keep the order explicit.

### Create the venv

Some images ship Python without `ensurepip`; use `uv venv` directly:

```bash
VENV_DIR=/root/.venv/rl-isaac
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
cd /root
unset UV_EXCLUDE_NEWER UV_EXCLUDE_NEWER_PACKAGE
```

### Install Isaac Sim, then IsaacLab source packages

Pin versions for the experiment. Example:

```bash
uv pip install "isaacsim[all,extscache]==6.0.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow

git clone https://github.com/isaac-sim/IsaacLab.git /root/IsaacLab
cd /root/IsaacLab
./isaaclab.sh --install || true
```

The `|| true` is for non-interactive setup scripts: package installation can
succeed even if a later VSCode/Kit bootstrap step exits after an EULA prompt.
The runtime job must set the EULA variables above.

Check package presence without importing Isaac, since imports can initialize Kit
or trigger prompts:

```bash
python -c "import importlib.util; raise SystemExit(importlib.util.find_spec('isaacsim') is None)"
python -c "import importlib.util; raise SystemExit(importlib.util.find_spec('isaaclab') is None)"
```

### Restore the target torch/TorchRL stack

IsaacLab installers may replace torch with their preferred build. Reinstall the
experiment's target torch after IsaacLab if needed:

```bash
uv pip install --upgrade --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu130
```

For cu130 nightlies, if `import torch` fails with an NCCL symbol such as
`ncclCommResume`, force reinstall the matching cu13 NCCL wheel after all
cu12/cu128 installs and prepend the venv NVIDIA library paths to
`LD_LIBRARY_PATH`.

```bash
uv pip install --force-reinstall "nvidia-nccl-cu13==2.30.4" \
    --index-url https://download.pytorch.org/whl/nightly/cu130
```

After torch is correct, install local TensorDict and TorchRL without
dependencies so pip does not replace torch:

```bash
uv pip install -e /root/tensordict --no-deps
uv pip install -e /root/rl --no-deps
python -c "import torch, tensordict, torchrl; print(torch.__version__, tensordict.__version__, torchrl.__version__)"
```

## Environment discovery

List registered IsaacLab environments inside the Isaac environment:

```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

Useful validation tasks include:

- `Isaac-Ant-v0`
- `Isaac-Humanoid-v0`
- `Isaac-Cartpole-v0`
- `Isaac-Velocity-Flat-Anymal-C-v0`
- `Isaac-Reach-Franka-v0`

## Troubleshooting checklist

- AppLauncher was created before importing torch in each Isaac-owning process.
- The job sets EULA/headless variables and `TERM`.
- Only one Isaac simulation is created per process.
- `gym.make` receives an explicit IsaacLab config object (`cfg=...`).
- `from_pixels` is not used as a substitute for adding a `TiledCameraCfg`.
- Camera rendering was launched with `--enable_cameras`.
- EGL/GLVND/Vulkan packages are installed in minimal CUDA images.
- NVIDIA GL/EGL userspace libraries match the host driver.
- Dedicated render workers use `CUDA_VISIBLE_DEVICES=<gpu>` and `cuda:0` inside
  the worker.
- Setup checks use `importlib.util.find_spec`, not real Isaac imports.
- Torch was verified after IsaacLab installation and reinstalled if necessary.
- TensorDict/TorchRL were installed with `--no-deps` after torch was correct.
- Recurrent collectors use policy factories and recurrent-state auto transforms.
- PPO updates use one collected rollout window at a time; avoid continuously
  filling replay buffers for on-policy updates.
- If Isaac leaves orphan processes after failed runs, clean up stale Python/Kit
  processes before relaunching.
- If a container lacks `rsync`, install it with `apt-get install -y rsync`.
