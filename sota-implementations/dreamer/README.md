# Dreamer V1

This is an implementation of the Dreamer algorithm from the paper 
["Dream to Control: Learning Behaviors by Latent Imagination"](https://arxiv.org/abs/1912.01603) (Hafner et al., ICLR 2020).

Dreamer is a model-based reinforcement learning algorithm that:
1. Learns a **world model** (RSSM) from experience
2. **Imagines** future trajectories in latent space
3. Trains **actor and critic** using analytic gradients through the imagined rollouts

## Setup

### Dependencies

```bash
# Create virtual environment
uv venv torchrl --python 3.12
source torchrl/bin/activate

# Install PyTorch (adjust for your CUDA version)
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install TorchRL and TensorDict
uv pip install tensordict torchrl

# Install additional dependencies
uv pip install mujoco dm_control wandb tqdm hydra-core
```

### System Dependencies (for MuJoCo rendering)

```bash
apt-get update && apt-get install -y \
    libegl1 \
    libgl1 \
    libgles2 \
    libglvnd0
```

### Environment Variables

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

## Running

```bash
python dreamer.py
```

### Configuration

The default configuration trains on DMControl's `cheetah-run` task. You can override settings via command line:

```bash
# Different environment
python dreamer.py env.name=walker env.task=walk

# Mixed precision options: false, true (=bfloat16), float16, bfloat16
python dreamer.py optimization.autocast=bfloat16  # default
python dreamer.py optimization.autocast=float16   # for older GPUs
python dreamer.py optimization.autocast=false     # disable autocast

# Adjust batch size
python dreamer.py replay_buffer.batch_size=1000
```

## Known Caveats

### 1. Mixed Precision (Autocast) Compatibility

Some GPU/cuBLAS combinations have issues with `bfloat16` autocast, resulting in:
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasGemmEx
```

**Solutions:**
- Try float16: `optimization.autocast=float16`
- Or disable autocast entirely: `optimization.autocast=false`

Note: Ensure your PyTorch CUDA version matches your driver. For example, with CUDA 13.0:
```bash
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

### 2. Benchmarking Status

This implementation has not been fully benchmarked against the original paper's results.
Performance may differ from published numbers.

### 3. Video Logging

To enable video logging of both real and imagined rollouts:
```bash
python dreamer.py logger.video=True
```

This requires additional setup for rendering and significantly increases computation time.

## Architecture Overview

```
World Model:
  - ObsEncoder: pixels -> encoded_latents
  - RSSMPrior: (state, belief, action) -> next_belief, prior_dist
  - RSSMPosterior: (belief, encoded_latents) -> posterior_dist, state
  - ObsDecoder: (state, belief) -> reconstructed_pixels
  - RewardModel: (state, belief) -> predicted_reward

Actor: (state, belief) -> action_distribution
Critic: (state, belief) -> state_value
```

## Training Loop

1. **Collect** real experience from environment
2. **Train world model** on sequences from replay buffer (KL + reconstruction + reward loss)
3. **Imagine** trajectories starting from encoded real states
4. **Train actor** to maximize imagined returns (gradients flow through dynamics)
5. **Train critic** to predict lambda returns on imagined trajectories

## IsaacLab Support

Dreamer can be trained with [IsaacLab](https://isaac-sim.github.io/IsaacLab/v2.3.0/)
environments using the dedicated script:

```bash
python dreamer_isaac.py
```

See `config_isaac.yaml` for the full configuration. IsaacLab requires its own
entry-point because `AppLauncher` **must** be initialised before `import torch`.

### Observation key

IsaacLab uses `"policy"` as the observation key. In Dreamer's pipeline this
means:

- Encoder reads from `"policy"` (not `"observation"` or `"pixels"`).
- Decoder writes to `"reco_policy"`.
- Loss keys: `world_model_loss.set_keys(pixels="policy", reco_pixels="reco_policy")`.

This is handled automatically when `cfg.env.backend == "isaaclab"` in
`make_dreamer()`.

### No separate eval environment

IsaacLab typically runs one simulation per process. Creating a second env for
evaluation is unreliable. Instead:

- Track episode rewards from completed episodes in the training data.
- The 4096 parallel envs provide statistically robust reward estimates.
- Periodic deterministic evaluation can be done by switching the policy's
  exploration type.

### Throughput benchmarks

**Single GPU** (synchronous collect-then-train, NVIDIA H200, 4096 envs):

- ~15,600 fps data collection
- ~7.0 optim steps/sec (FP32, no compile)
- ~55 s for 200 steps end-to-end (3.6 ops/s)
- 50 % time collecting, 50 % training

**2-GPU async pipeline** (GPU 0 = sim, GPU 1 = train, bfloat16, torch.compile):

- ~78k–137k fps continuous collection
- ~2.9–3.5 optim steps/sec with 50k batch
- Collection and training fully overlap on separate GPUs
- ~9.5 h ETA for 100k steps on H200 (Anymal-C locomotion)

**Key insight**: with async 2-GPU, the replay buffer `extend()` (collector
thread) and `sample()` (training thread) both touch CPU memory, causing GIL
contention. A GPU-resident replay buffer eliminates CPU-GPU transfer at sample
time.

### Config differences from DMControl

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

## References

- Original Paper: [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
- PlaNet (predecessor): [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
- DreamerV2: [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)
- DreamerV3: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
