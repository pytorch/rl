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
#python .circleci/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200

# ==================================================================================== #
# ================================ gym 0.23 ========================================== #

# With batched environments
python .circleci/unittest/helpers/coverage_run_parallel.py examples/decision_transformer/dt.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10 \
  logger.backend=
python .circleci/unittest/helpers/coverage_run_parallel.py examples/decision_transformer/online_td.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10 \
  logger.backend=

# ==================================================================================== #
# ================================ Gymnasium ========================================= #

# install ale-py: manylinux names are broken for CentOS so we need to manually download and
# rename them
PY_VERSION=$(python --version)
if [[ $PY_VERSION == *"3.7"* ]]; then
  wget https://files.pythonhosted.org/packages/ab/fd/6615982d9460df7f476cad265af1378057eee9daaa8e0026de4cedbaffbd/ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  pip install ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  rm ale_py-0.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
elif [[ $PY_VERSION == *"3.8"* ]]; then
  wget https://files.pythonhosted.org/packages/0f/8a/feed20571a697588bc4bfef05d6a487429c84f31406a52f8af295a0346a2/ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  pip install ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  rm ale_py-0.8.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
elif [[ $PY_VERSION == *"3.9"* ]]; then
  wget https://files.pythonhosted.org/packages/a0/98/4316c1cedd9934f9a91b6e27a9be126043b4445594b40cfa391c8de2e5e8/ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  pip install ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  rm ale_py-0.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
elif [[ $PY_VERSION == *"3.10"* ]]; then
  wget https://files.pythonhosted.org/packages/60/1b/3adde7f44f79fcc50d0a00a0643255e48024c4c3977359747d149dc43500/ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
  mv ale_py-0.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  pip install ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  rm ale_py-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
fi
pip install "gymnasium[atari,accept-rom-license]"

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
