## Reproducing Proximal Policy Optimization (PPO) Algorithm Results

This repository contains scripts that enable training agents using the Proximal Policy Optimization (PPO) Algorithm on MuJoCo and Atari environments. We follow the original paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) by Schulman et al. (2017) to implement the PPO algorithm but introduce the improvement of computing the Generalised Advantage Estimator (GAE) at every epoch.


## Examples Structure

Please note that each example is independent of each other for the sake of simplicity. Each example contains the following files:

1. **Main Script:** The definition of algorithm components and the training loop can be found in the main script  (e.g. ppo_atari.py).

2. **Utils File:** A utility file is provided to contain various helper functions, generally to create the environment and the models (e.g. utils_atari.py).

3. **Configuration File:** This file includes default hyperparameters specified in the original paper. Users can modify these hyperparameters to customize their experiments  (e.g. config_atari.yaml).


## Running the Examples

You can execute the PPO algorithm on Atari environments by running the following command:

```bash
python ppo_atari.py
```

You can execute the PPO algorithm on MuJoCo environments by running the following command:

```bash
python ppo_mujoco.py
``` 

## MuJoCo Playground Franka/Panda example

The MuJoCo PPO script can also use the optional TorchRL MuJoCo Playground
wrapper. This makes it possible to run PPO against MJX-backed manipulation
tasks such as `PandaPickCube` from the Franka Emika Panda task suite.

Install the optional dependencies through the dedicated extra and select the
MuJoCo Playground backend from Hydra:

```bash
uv run --frozen --extra mujoco_playground python sota-implementations/ppo/ppo_mujoco.py \
  env.backend=mujoco_playground \
  env.env_name=PandaPickCube \
  +env.config_overrides.impl=jax \
  env.num_envs=1 \
  env.batch_mode=parallel \
  env.normalize_observation=false \
  env.max_episode_steps=150 \
  optim.device=cpu \
  logger.backend=null \
  collector.total_frames=512 \
  collector.frames_per_batch=128 \
  loss.mini_batch_size=32 \
  loss.ppo_epochs=2 \
  logger.test_interval=256 \
  logger.num_test_episodes=1 \
  checkpoint.path=/tmp/torchrl_ppo_panda_pick_smoke.pt
```

MuJoCo model assets are downloaded when absent, and the MJX step function is
JIT-compiled according to the JAX compilation cache. On local CPU runs, the JAX
implementation is the most portable choice for this task; if the default
backend selects Warp and fails during MJX model construction, keep the
`+env.config_overrides.impl=jax` override.

A longer local validation command is:

```bash
uv run --frozen --extra mujoco_playground python sota-implementations/ppo/ppo_mujoco.py \
  env.backend=mujoco_playground \
  env.env_name=PandaPickCube \
  +env.config_overrides.impl=jax \
  env.num_envs=1 \
  env.batch_mode=parallel \
  env.normalize_observation=false \
  env.max_episode_steps=150 \
  optim.device=cpu \
  logger.backend=null \
  logger.test_interval=1024 \
  logger.num_test_episodes=2 \
  collector.total_frames=4096 \
  collector.frames_per_batch=256 \
  loss.mini_batch_size=64 \
  loss.ppo_epochs=4 \
  checkpoint.path=/tmp/torchrl_ppo_panda_pick.pt \
  checkpoint.interval=1024
```

This run is intended as a functional training/checkpointing check for a richer
manipulation environment rather than a solved-policy benchmark. The checkpoint
contains the MuJoCo Playground backend name and resolved config overrides, so
`utils_mujoco.py:make_render_policy` can reconstruct the same policy
architecture from the checkpoint metadata. Pixel video rendering is unavailable
for this backend because the MuJoCo Playground wrapper does not support
`from_pixels=True`.

MuJoCo Playground supports two multi-environment paths:

- `env.batch_mode=parallel` creates one scalar MuJoCo Playground environment per
  TorchRL `ParallelEnv` worker. This is the simple CPU path.
- `env.batch_mode=vmap` creates one batched MuJoCo Playground environment with
  `batch_size=[env.num_envs]`. The MuJoCo Playground wrapper internally uses
  JAX `vmap` and `jit` for this path, which is the path to benchmark on GPU/TPU.

For the simple worker-parallel path, increase `env.num_envs` while keeping
`env.batch_mode=parallel`:

```bash
uv run --frozen --extra mujoco_playground python sota-implementations/ppo/ppo_mujoco.py \
  env.backend=mujoco_playground \
  env.env_name=PandaPickCube \
  +env.config_overrides.impl=jax \
  env.num_envs=4 \
  env.batch_mode=parallel \
  env.normalize_observation=false \
  env.max_episode_steps=150 \
  optim.device=cpu \
  logger.backend=null \
  logger.test_interval=8192 \
  logger.num_test_episodes=1 \
  collector.total_frames=8192 \
  collector.frames_per_batch=1024 \
  loss.mini_batch_size=256 \
  loss.ppo_epochs=2 \
  checkpoint.path=/tmp/torchrl_ppo_panda_pick_parallel.pt
```

For the in-process vmap/JIT path, set `env.batch_mode=vmap`:

```bash
uv run --frozen --extra mujoco_playground python sota-implementations/ppo/ppo_mujoco.py \
  env.backend=mujoco_playground \
  env.env_name=PandaPickCube \
  +env.config_overrides.impl=jax \
  env.num_envs=16 \
  env.batch_mode=vmap \
  env.normalize_observation=false \
  env.max_episode_steps=150 \
  optim.device=cpu \
  logger.backend=null \
  logger.test_interval=8192 \
  logger.num_test_episodes=1 \
  collector.total_frames=8192 \
  collector.frames_per_batch=1024 \
  loss.mini_batch_size=256 \
  loss.ppo_epochs=2 \
  checkpoint.path=/tmp/torchrl_ppo_panda_pick_batched.pt
```

On local CPU runs, benchmark both settings before scaling a long job. The scalar
`env.num_envs=1` setup can be faster than the vmap path because XLA compile,
MJX stepping, and DLPack transfer overheads are machine dependent.

## Rendering an InvertedPendulum checkpoint with MuJoCo-WASM

The MuJoCo PPO script can write a local checkpoint that is directly loadable by
`rlrender`. The checkpoint stores the actor state dict, Gymnasium environment
name, observation-normalization setting, resolved Hydra config, frame count, and
metrics recorded at checkpoint time.

`InvertedPendulum-v4` is a lightweight MuJoCo-WASM render target. On macOS,
pass `optim.device=cpu`; MuJoCo specs use
`float64`, which is not supported by the MPS backend.

### Smoke test

Use a small run to validate checkpointing:

```bash
uv run --frozen python sota-implementations/ppo/ppo_mujoco.py \
  env.env_name=InvertedPendulum-v4 \
  env.normalize_observation=false \
  optim.device=cpu \
  logger.backend=null \
  collector.total_frames=2048 \
  collector.frames_per_batch=512 \
  loss.mini_batch_size=64 \
  loss.ppo_epochs=2 \
  logger.test_interval=1024 \
  logger.num_test_episodes=1 \
  checkpoint.path=/tmp/torchrl_ppo_inverted_pendulum_smoke.pt
```

### Training run with W&B

Set ``WANDB_API_KEY`` in the environment and leave ``logger.backend=wandb``
enabled for an online W&B run:

```bash
uv run --frozen python sota-implementations/ppo/ppo_mujoco.py \
  env.env_name=InvertedPendulum-v4 \
  env.normalize_observation=false \
  optim.device=cpu \
  logger.backend=wandb \
  logger.project_name=torchrl_rlrender \
  logger.group_name=ppo_inverted_pendulum_mujoco_wasm \
  logger.test_interval=10000 \
  logger.num_test_episodes=5 \
  collector.total_frames=100000 \
  collector.frames_per_batch=2048 \
  loss.mini_batch_size=64 \
  loss.ppo_epochs=10 \
  checkpoint.path=/tmp/torchrl_ppo_inverted_pendulum.pt \
  checkpoint.interval=10000
```

This scale is intended to produce a visible training curve and a solved
InvertedPendulum checkpoint.

### MuJoCo-WASM notebook render

Render a notebook that opens a live MuJoCo-WASM viewer and plays the saved
`qpos` trajectory:

`rlrender` records `qpos` through TorchRL's `MujocoStateReader`; the PPO
environment therefore keeps the same observation keys used for training.

```bash
MODEL_PATH="$(uv run --frozen python - <<'PY'
from pathlib import Path
import gymnasium.envs.mujoco
print(Path(gymnasium.envs.mujoco.__file__).parent / "assets" / "inverted_pendulum.xml")
PY
)"

uv run --frozen --extra rendering python -m torchrl.render \
  --ckpt /tmp/torchrl_ppo_inverted_pendulum.pt \
  --policy sota-implementations/ppo/utils_mujoco.py:make_render_policy \
  --env sota-implementations/ppo/utils_mujoco.py:make_render_env \
  --env-kwargs '{"env_name":"InvertedPendulum-v4"}' \
  --render-backend null \
  --max-steps 1000 \
  --num-trajs 1 \
  --format ipynb \
  --out /tmp/torchrl_ppo_inverted_pendulum_mujoco_wasm.ipynb \
  --notebook-render-backend mujoco-wasm \
  --mujoco-model-path "$MODEL_PATH" \
  --mujoco-qpos-key qpos \
  --overwrite
```

To generate trajectories inside the notebook instead of before notebook
creation, add `--notebook-rollout-mode live`. The generated notebook will
construct the configured policy and environment in the kernel, collect
rollouts when the rollout cell is executed, and then play the resulting `qpos`
trajectory in the live MuJoCo-WASM iframe:

```bash
uv run --frozen --extra rendering python -m torchrl.render \
  --ckpt /tmp/torchrl_ppo_inverted_pendulum.pt \
  --policy sota-implementations/ppo/utils_mujoco.py:make_render_policy \
  --env sota-implementations/ppo/utils_mujoco.py:make_render_env \
  --env-kwargs '{"env_name":"InvertedPendulum-v4"}' \
  --render-backend null \
  --max-steps 1000 \
  --num-trajs 1 \
  --format ipynb \
  --out /tmp/torchrl_ppo_inverted_pendulum_mujoco_wasm_live.ipynb \
  --notebook-render-backend mujoco-wasm \
  --notebook-rollout-mode live \
  --mujoco-model-path "$MODEL_PATH" \
  --mujoco-qpos-key qpos \
  --overwrite
```

Open the notebook with the project's locked dependency resolution:

```bash
uv run --frozen --extra notebook jupyter-lab /tmp/torchrl_ppo_inverted_pendulum_mujoco_wasm.ipynb
```
