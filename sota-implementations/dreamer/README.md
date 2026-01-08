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

## References

- Original Paper: [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
- PlaNet (predecessor): [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
- DreamerV2: [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)
- DreamerV3: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
