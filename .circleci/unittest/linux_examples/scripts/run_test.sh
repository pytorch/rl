#!/usr/bin/env bash

# Leave blank as code needs to start on line 29 for run_local.sh
#
#
#
#
#
#
#

set -e
set -v

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

python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200

# With batched environments
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ppo/ppo.py \
  env.num_envs=1 \
  env.device=cuda:0 \
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
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optimization.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=4 \
  collector.env_per_collector=2 \
  collector.collector_device=cuda:0 \
  network.device=cuda:0 \
  optimization.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
#  record_video=True \
#  record_frames=4 \
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
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=4 \
  collector.env_per_collector=2 \
  collector.collector_device=cuda:0 \
  optimization.batch_size=10 \
  optimization.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
#  logger.record_video=True \
#  logger.record_frames=4 \
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
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optimization.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=4 \
  collector.env_per_collector=2 \
  collector.collector_device=cuda:0 \
  network.device=cuda:0 \
  logger.mode=offline \
  env.name=Pendulum-v1 \
  logger.backend=
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
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optimization.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=2 \
  collector.env_per_collector=1 \
  collector.collector_device=cuda:0 \
  network.device=cuda:0 \
  optimization.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
#  record_video=True \
#  record_frames=4 \
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
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ppo/ppo.py \
  env.num_envs=1 \
  env.device=cuda:0 \
  collector.total_frames=48 \
  collector.frames_per_batch=16 \
  collector.collector_device=cuda:0 \
  optim.device=cuda:0 \
  loss.mini_batch_size=10 \
  loss.ppo_epochs=1 \
  logger.backend= \
  logger.log_interval=4 \
  optim.lr_scheduler=False
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
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=2 \
  collector.env_per_collector=1 \
  collector.collector_device=cuda:0 \
  optimization.batch_size=10 \
  optimization.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
#  record_video=True \
#  record_frames=4 \
python .circleci/unittest/helpers/coverage_run_parallel.py examples/iql/iql_online.py \
  total_frames=48 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  mode=offline \
  device=cuda:0 \
  collector_device=cuda:0
python .circleci/unittest/helpers/coverage_run_parallel.py examples/td3/td3.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optimization.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=2 \
  collector.env_per_collector=1 \
  logger.mode=offline \
  collector.collector_device=cuda:0 \
  env.name=Pendulum-v1 \
  logger.backend=

python .circleci/unittest/helpers/coverage_run_parallel.py examples/bandits/dqn.py --n_steps=100

coverage combine
coverage xml -i
