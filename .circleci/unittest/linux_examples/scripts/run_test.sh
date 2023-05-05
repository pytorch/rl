#!/usr/bin/env bash

# this code is supposed to run on CPU
# rendering with the combination of packages we have here in headless mode
# is hard to nail.
# IMPORTANT: As a consequence, we can't guarantee TorchRL compatibility with
# rendering with this version of gym / mujoco-py.

set -e
set -v

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

python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 20
python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 20

# With batched environments
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ppo/ppo.py \
  env.num_envs=1 \
  collector.total_frames=48 \
  collector.frames_per_batch=16 \
  collector.collector_device=cuda:0 \
  optim.device=cuda:0 \
  loss.mini_batch_size=10 \
  loss.ppo_epochs=1 \
  logger.backend= \
  logger.log_interval=4 \
  optim.lr_scheduler=False
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ddpg/ddpg.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
python .circleci/unittest/helpers/coverage_run_parallel.py examples/a2c/a2c.py \
  env.num_envs=1 \
  collector.total_frames=48 \
  collector.frames_per_batch=16 \
  collector.collector_device=cuda:0 \
  logger.backend= \
  logger.log_interval=4 \
  optim.lr_scheduler=False \
  optim.device=cuda:0
python .circleci/unittest/helpers/coverage_run_parallel.py examples/dqn/dqn.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
python .circleci/unittest/helpers/coverage_run_parallel.py examples/redq/redq.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
python .circleci/unittest/helpers/coverage_run_parallel.py examples/sac/sac.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
python .circleci/unittest/helpers/coverage_run_parallel.py examples/dreamer/dreamer.py \
  total_frames=200 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=200 \
  num_workers=4 \
  env_per_collector=2 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120 \
  rssm_hidden_dim=17
python .circleci/unittest/helpers/coverage_run_parallel.py examples/td3/td3.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_device=cuda:0 \
  mode=offline
python .circleci/unittest/helpers/coverage_run_parallel.py examples/iql/iql_online.py \
  total_frames=48 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_device=cuda:0 \
  device=cuda:0 \
  mode=offline

# With single envs
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ddpg/ddpg.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
python .circleci/unittest/helpers/coverage_run_parallel.py examples/a2c/a2c.py \
  env.num_envs=1 \
  collector.total_frames=48 \
  collector.frames_per_batch=16 \
  collector.collector_device=cuda:0 \
  logger.backend= \
  logger.log_interval=4 \
  optim.lr_scheduler=False \
  optim.device=cuda:0
python .circleci/unittest/helpers/coverage_run_parallel.py examples/dqn/dqn.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
python .circleci/unittest/helpers/coverage_run_parallel.py examples/redq/redq.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
python .circleci/unittest/helpers/coverage_run_parallel.py examples/sac/sac.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ppo/ppo.py \
  env.num_envs=1 \
  collector.total_frames=48 \
  collector.frames_per_batch=16 \
  collector.collector_device=cuda:0 \
  optim.device=cuda:0 \
  loss.mini_batch_size=10 \
  loss.ppo_epochs=1 \
  logger.backend= \
  logger.log_interval=4 \
  optim.lr_scheduler=False
python .circleci/unittest/helpers/coverage_run_parallel.py examples/dreamer/dreamer.py \
  total_frames=200 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=200 \
  num_workers=2 \
  env_per_collector=1 \
  collector_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120 \
  rssm_hidden_dim=17
python .circleci/unittest/helpers/coverage_run_parallel.py examples/td3/td3.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  mode=offline \
  collector_device=cuda:0
python .circleci/unittest/helpers/coverage_run_parallel.py examples/iql/iql_online.py \
  total_frames=48 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  mode=offline \
  device=cuda:0 \
  collector_device=cuda:0

python .circleci/unittest/helpers/coverage_run_parallel.py examples/bandits/dqn.py --n_steps=100

coverage combine
coverage xml -i
