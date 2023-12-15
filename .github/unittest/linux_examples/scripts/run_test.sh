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

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
#python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200

# ==================================================================================== #
# ================================ gym 0.23 ========================================== #

# With batched environments
python .github/unittest/helpers/coverage_run_parallel.py examples/decision_transformer/dt.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10 \
  optim.device=cuda:0 \
  logger.backend= \
  env.backend=gymnasium \
  env.name=HalfCheetah-v4
python .github/unittest/helpers/coverage_run_parallel.py examples/decision_transformer/online_dt.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10 \
  optim.device=cuda:0 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/iql/iql_offline.py \
  optim.gradient_steps=55 \
  optim.device=cuda:0 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/cql/cql_offline.py \
  optim.gradient_steps=55 \
  optim.device=cuda:0 \
  logger.backend=

# ==================================================================================== #
# ================================ Gymnasium ========================================= #

python .github/unittest/helpers/coverage_run_parallel.py examples/impala/impala_single_node.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  collector.num_workers=1 \
  logger.backend= \
  logger.test_interval=10
python .github/unittest/helpers/coverage_run_parallel.py examples/ppo/ppo_mujoco.py \
  env.env_name=HalfCheetah-v4 \
  collector.total_frames=40 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=10 \
  loss.ppo_epochs=2 \
  logger.backend= \
  logger.test_interval=10
python .github/unittest/helpers/coverage_run_parallel.py examples/ppo/ppo_atari.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=20 \
  loss.ppo_epochs=2 \
  logger.backend= \
  logger.test_interval=10
python .github/unittest/helpers/coverage_run_parallel.py examples/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  collector.device=cuda:0 \
  network.device=cuda:0 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
#  record_video=True \
#  record_frames=4 \
python .github/unittest/helpers/coverage_run_parallel.py examples/a2c/a2c_mujoco.py \
  env.env_name=HalfCheetah-v4 \
  collector.total_frames=40 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=10 \
  logger.backend= \
  logger.test_interval=40
python .github/unittest/helpers/coverage_run_parallel.py examples/a2c/a2c_atari.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=20 \
  logger.backend= \
  logger.test_interval=40
python .github/unittest/helpers/coverage_run_parallel.py examples/dqn/dqn_atari.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  buffer.batch_size=10 \
  device=cuda:0 \
  loss.num_updates=1 \
  buffer.buffer_size=120
python .github/unittest/helpers/coverage_run_parallel.py examples/cql/discrete_cql_online.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  collector.device=cuda:0 \
  replay_buffer.size=120 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/redq/redq.py \
  num_workers=4 \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  collector.device=cuda:0 \
  buffer.batch_size=10 \
  optim.steps_per_batch=1 \
  logger.record_video=True \
  logger.record_frames=4 \
  buffer.size=120 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/sac/sac.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  collector.device=cuda:0 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  network.device=cuda:0 \
  logger.backend=
#  logger.record_video=True \
#  logger.record_frames=4 \
python .github/unittest/helpers/coverage_run_parallel.py examples/dreamer/dreamer.py \
  total_frames=200 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=200 \
  num_workers=4 \
  env_per_collector=2 \
  collector_device=cuda:0 \
  model_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120 \
  rssm_hidden_dim=17
python .github/unittest/helpers/coverage_run_parallel.py examples/td3/td3.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=4 \
  collector.env_per_collector=2 \
  collector.device=cuda:0 \
  collector.device=cuda:0 \
  network.device=cuda:0 \
  logger.mode=offline \
  env.name=Pendulum-v1 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/iql/iql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=2 \
  optim.device=cuda:0 \
  collector.device=cuda:0 \
  logger.mode=offline \
  logger.backend=
  python .github/unittest/helpers/coverage_run_parallel.py examples/cql/cql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=2 \
  collector.device=cuda:0 \
  optim.device=cuda:0 \
  logger.mode=offline \
  logger.backend=

# With single envs
python .github/unittest/helpers/coverage_run_parallel.py examples/dreamer/dreamer.py \
  total_frames=200 \
  init_random_frames=10 \
  batch_size=10 \
  frames_per_batch=200 \
  num_workers=2 \
  env_per_collector=1 \
  collector_device=cuda:0 \
  model_device=cuda:0 \
  optim_steps_per_batch=1 \
  record_video=True \
  record_frames=4 \
  buffer_size=120 \
  rssm_hidden_dim=17
python .github/unittest/helpers/coverage_run_parallel.py examples/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  collector.device=cuda:0 \
  network.device=cuda:0 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
#  record_video=True \
#  record_frames=4 \
python .github/unittest/helpers/coverage_run_parallel.py examples/dqn/dqn_atari.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  buffer.batch_size=10 \
  device=cuda:0 \
  loss.num_updates=1 \
  buffer.buffer_size=120
python .github/unittest/helpers/coverage_run_parallel.py examples/redq/redq.py \
  num_workers=2 \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  buffer.batch_size=10 \
  collector.device=cuda:0 \
  optim.steps_per_batch=1 \
  logger.record_video=True \
  logger.record_frames=4 \
  buffer.size=120 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/sac/sac.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  collector.device=cuda:0 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  network.device=cuda:0 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/iql/iql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=1 \
  logger.mode=offline \
  optim.device=cuda:0 \
  collector.device=cuda:0 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/cql/cql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  logger.mode=offline \
  optim.device=cuda:0 \
  collector.device=cuda:0 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/td3/td3.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=2 \
  collector.env_per_collector=1 \
  collector.device=cuda:0 \
  logger.mode=offline \
  optim.batch_size=10 \
  env.name=Pendulum-v1 \
  network.device=cuda:0 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/multiagent/mappo_ippo.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/multiagent/maddpg_iddpg.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/multiagent/iql.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/multiagent/qmix_vdn.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
python .github/unittest/helpers/coverage_run_parallel.py examples/multiagent/sac.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=

python .github/unittest/helpers/coverage_run_parallel.py examples/bandits/dqn.py --n_steps=100

## RLHF
# RLHF tests are executed in the dedicated workflow

coverage combine
coverage xml -i
