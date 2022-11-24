#!/usr/bin/env bash

# this code is supposed to run on CPU
# rendering with the combination of packages we have here in headless mode
# is hard to nail.
# IMPORTANT: As a consequence, we can't guarantee TorchRL compatibility with
# rendering with this version of gym / mujoco-py.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU

coverage run -m pytest test/smoke_test.py -v --durations 20
coverage run -m pytest test/smoke_test_deps.py -v --durations 20
coverage run examples/ddpg/ddpg.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
coverage run examples/a2c/a2c.py \
  total_frames=48 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  logger=csv
coverage run examples/dqn/dqn.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
coverage run examples/redq/redq.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
coverage run examples/sac/sac.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
coverage run examples/ppo/ppo.py \
  total_frames=48 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  lr_scheduler=
coverage run examples/dreamer/dreamer.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=200 \
  num_workers=2 \
  env_per_collector=1 \
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120 \
  rssm_hidden_dim=17
coverage xml -i
