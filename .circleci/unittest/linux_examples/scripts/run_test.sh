#!/usr/bin/env bash

# this code is supposed to run on CPU
# rendering with the combination of packages we have here in headless mode
# is hard to nail.
# IMPORTANT: As a consequence, we can't guarantee TorchRL compatibility with
# rendering with this version of gym / mujoco-py.

set -e

apt-get update && apt-get remove swig -y && apt-get install -y git gcc patchelf libosmesa6-dev libgl1-mesa-glx libglfw3 swig3.0 wget freeglut3 freeglut3-dev

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

# ========================================================================================
# DDPG
# ----
#
# Modalities:
# ^^^^^^^^^^^
#
# pixels on/off
# Batched on/off
#
# With batched environments
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.num_collectors=4 \
  collector.collector_devices=cuda:0 \
  env.num_envs=2 \
  optim.batch_size=10 \
  optim.optim_steps_per_batch=1 \
  recorder.video=True \
  recorder.frames=4 \
  replay_buffer.capacity=120 \
  env.from_pixels=False \
  logger.backend=csv
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.num_collectors=4 \
  collector.collector_devices=cuda:0 \
  env.num_envs=2 \
  optim.batch_size=10 \
  optim.optim_steps_per_batch=1 \
  recorder.video=True \
  recorder.frames=4 \
  replay_buffer.capacity=120 \
  env.from_pixels=True \
  logger.backend=csv
# With single envs
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.num_collectors=4 \
  collector.collector_devices=cuda:0 \
  env.num_envs=1 \
  optim.batch_size=10 \
  optim.optim_steps_per_batch=1 \
  recorder.video=True \
  recorder.frames=4 \
  replay_buffer.capacity=120 \
  env.from_pixels=False \
  logger.backend=csv
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.num_collectors=4 \
  collector.collector_devices=cuda:0 \
  env.num_envs=1 \
  optim.batch_size=10 \
  optim.optim_steps_per_batch=1 \
  recorder.video=True \
  recorder.frames=4 \
  replay_buffer.capacity=120 \
  env.from_pixels=True \
  logger.backend=csv

python .circleci/unittest/helpers/coverage_run_parallel.py examples/a2c/a2c.py \
  total_frames=48 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  logger=csv
python .circleci/unittest/helpers/coverage_run_parallel.py examples/dqn/dqn.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_devices=cuda:0 \
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
  collector_devices=cuda:0 \
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
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ppo/ppo.py \
  total_frames=48 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_devices=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  lr_scheduler=
python .circleci/unittest/helpers/coverage_run_parallel.py examples/dreamer/dreamer.py \
  total_frames=200 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=200 \
  num_workers=4 \
  env_per_collector=2 \
  collector_devices=cuda:0 \
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
  collector_devices=cuda:0 \
  mode=offline 
python .circleci/unittest/helpers/coverage_run_parallel.py examples/iql/iql_online.py \
  total_frames=48 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=4 \
  env_per_collector=2 \
  collector_devices=cuda:0 \
  mode=offline 

python .circleci/unittest/helpers/coverage_run_parallel.py examples/a2c/a2c.py \
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
python .circleci/unittest/helpers/coverage_run_parallel.py examples/dqn/dqn.py \
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
python .circleci/unittest/helpers/coverage_run_parallel.py examples/redq/redq.py \
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
python .circleci/unittest/helpers/coverage_run_parallel.py examples/sac/sac.py \
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
python .circleci/unittest/helpers/coverage_run_parallel.py examples/ppo/ppo.py \
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
python .circleci/unittest/helpers/coverage_run_parallel.py examples/dreamer/dreamer.py \
  total_frames=200 \
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
python .circleci/unittest/helpers/coverage_run_parallel.py examples/td3/td3.py \
  total_frames=48 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  mode=offline \
  collector_devices=cuda:0 
python .circleci/unittest/helpers/coverage_run_parallel.py examples/iql/iql_online.py \
  total_frames=48 \
  batch_size=10 \
  frames_per_batch=16 \
  num_workers=2 \
  env_per_collector=1 \
  mode=offline \
  collector_devices=cuda:0

python .circleci/unittest/helpers/coverage_run_parallel.py examples/bandits/dqn.py --n_steps=100

coverage combine
coverage xml -i
