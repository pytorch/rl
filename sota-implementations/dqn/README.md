## Reproducing Deep Q-Learning (DQN) Algorithm Results

This repository contains scripts that enable training agents using the Deep Q-Learning (DQN) Algorithm on CartPole and Atari environments. For Atari, We follow the original paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) by Mnih et al. (2013).


## Examples Structure

Please note that each example is independent of each other for the sake of simplicity. Each example contains the following files:

1. **Main Script:** The definition of algorithm components and the training loop can be found in the main script  (e.g. dqn_atari.py).

2. **Utils File:** A utility file is provided to contain various helper functions, generally to create the environment and the models (e.g. utils_atari.py).

3. **Configuration File:** This file includes default hyperparameters specified in the original paper. Users can modify these hyperparameters to customize their experiments  (e.g. config_atari.yaml).


## Running the Examples

You can execute the DQN algorithm on the CartPole environment by running the following command:

```bash
python dqn_cartpole.py

``` 

You can execute the DQN algorithm on Atari environments by running the following command:

```bash
python dqn_atari.py
```

## Rendering a CartPole checkpoint with `rlrender`

The CartPole DQN script can now write a local checkpoint that is directly
loadable by `rlrender`. The checkpoint stores the policy state dict, the Gym
environment name, the resolved Hydra config, the frame count, and the latest
training metrics.

### Smoke test

Use a tiny run first to validate checkpointing and rendering:

```bash
uv run --frozen python sota-implementations/dqn/dqn_cartpole.py \
  logger.backend=null \
  collector.total_frames=400 \
  collector.frames_per_batch=100 \
  collector.init_random_frames=100 \
  collector.annealing_frames=400 \
  loss.num_updates=1 \
  buffer.batch_size=32 \
  logger.test_interval=200 \
  logger.num_test_episodes=1 \
  checkpoint.path=/tmp/torchrl_dqn_cartpole_smoke.pt
```

### Training run with W&B

Set ``WANDB_API_KEY`` in the environment and leave ``logger.backend=wandb``
enabled for an online W&B run:

```bash
uv run --frozen python sota-implementations/dqn/dqn_cartpole.py \
  logger.backend=wandb \
  logger.project_name=torchrl_rlrender \
  logger.group_name=dqn_cartpole_rlrender \
  collector.total_frames=500100 \
  collector.frames_per_batch=1000 \
  collector.init_random_frames=10000 \
  collector.annealing_frames=250000 \
  loss.num_updates=100 \
  buffer.batch_size=128 \
  logger.test_interval=25000 \
  logger.num_test_episodes=5 \
  checkpoint.path=/tmp/torchrl_dqn_cartpole.pt \
  checkpoint.interval=50000
```

This default-scale command is intended to produce a visible training curve and
a solved CartPole checkpoint.

The default Gymnasium CartPole RGB renderer requires the optional `pygame`
dependency. To avoid making `pygame` a hard dependency, the render environment
factory below draws a lightweight CartPole RGB frame from the observation state.

Render an MP4 from the checkpoint:

```bash
uv run --frozen --extra rendering python -m torchrl.render \
  --ckpt /tmp/torchrl_dqn_cartpole.pt \
  --policy sota-implementations/dqn/utils_cartpole.py:make_render_policy \
  --env sota-implementations/dqn/utils_cartpole.py:make_render_env \
  --from-pixels \
  --max-steps 500 \
  --num-trajs 1 \
  --format mp4 \
  --out /tmp/torchrl_dqn_cartpole.mp4 \
  --overwrite
```

Render an inspection notebook from the same checkpoint:

```bash
uv run --frozen --extra rendering python -m torchrl.render \
  --ckpt /tmp/torchrl_dqn_cartpole.pt \
  --policy sota-implementations/dqn/utils_cartpole.py:make_render_policy \
  --env sota-implementations/dqn/utils_cartpole.py:make_render_env \
  --from-pixels \
  --max-steps 500 \
  --num-trajs 1 \
  --format ipynb \
  --out /tmp/torchrl_dqn_cartpole_render.ipynb \
  --overwrite
```

Open the notebook with the locked environment to avoid triggering a fresh
cross-version dependency resolution:

```bash
uv run --frozen --extra notebook jupyter-lab /tmp/torchrl_dqn_cartpole_render.ipynb
```
