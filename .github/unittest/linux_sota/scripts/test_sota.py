# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import subprocess
from pathlib import Path

import pytest

commands = {
    "dt": """python sota-implementations/decision_transformer/dt.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10 \
  logger.backend= \
  env.backend=gymnasium \
  env.name=HalfCheetah-v4
""",
    "online_dt": """"python sota-implementations/decision_transformer/online_dt.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10 \
  env.backend=gymnasium \
  logger.backend=
""",
    "td3_bc": """python sota-implementations/td3_bc/td3_bc.py \
  optim.gradient_steps=55 \
  logger.backend=
""",
    "impala_single_node": """"python sota-implementations/impala/impala_single_node.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  collector.num_workers=1 \
  logger.backend= \
  logger.test_interval=10
""",
    "ppo_mujoco": """"python sota-implementations/ppo/ppo_mujoco.py \
  env.env_name=HalfCheetah-v4 \
  collector.total_frames=40 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=10 \
  loss.ppo_epochs=2 \
  logger.backend= \
  logger.test_interval=10
""",
    "ppo_atari": """"python sota-implementations/ppo/ppo_atari.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=20 \
  loss.ppo_epochs=2 \
  logger.backend= \
  logger.test_interval=10
""",
    "ddpg": """"python sota-implementations/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
""",
    "a2c_mujoco": """"python sota-implementations/a2c/a2c_mujoco.py \
  env.env_name=HalfCheetah-v4 \
  collector.total_frames=40 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=10 \
  logger.backend= \
  logger.test_interval=40
""",
    "a2c_atari": """"python sota-implementations/a2c/a2c_atari.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=20 \
  logger.backend= \
  logger.test_interval=40
""",
    "dqn_atari": """"python sota-implementations/dqn/dqn_atari.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  buffer.batch_size=10 \
  loss.num_updates=1 \
  logger.backend= \
  buffer.buffer_size=120
""",
    "discrete_cql_online": """"python sota-implementations/cql/discrete_cql_online.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  replay_buffer.size=120 \
  logger.backend=
""",
    "redq": """"python sota-implementations/redq/redq.py \
  num_workers=4 \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  buffer.batch_size=10 \
  optim.steps_per_batch=1 \
  logger.video=True \
  logger.record_frames=4 \
  buffer.size=120 \
  logger.backend=
""",
    "sac": """"python sota-implementations/sac/sac.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
""",
    "discrete_sac": """"python sota-implementations/discrete_sac/discrete_sac.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=CartPole-v1 \
  logger.backend=
""",
    "crossq": """"python sota-implementations/crossq/crossq.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  collector.device= \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  network.device= \
  logger.backend=
""",
    "td3": """"python sota-implementations/td3/td3.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=4 \
  collector.env_per_collector=2 \
  logger.mode=offline \
  env.name=Pendulum-v1 \
  logger.backend=
""",
    "iql_online": """"python sota-implementations/iql/iql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=2 \
  logger.mode=offline \
  logger.backend=
""",
    "discrete_iql": """"python sota-implementations/iql/discrete_iql.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=2 \
  logger.mode=offline \
  logger.backend=
""",
    "cql_online": """"python sota-implementations/cql/cql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=2 \
  logger.mode=offline \
  logger.backend=
""",
    "gail": """"python sota-implementations/gail/gail.py \
  ppo.collector.total_frames=48 \
  replay_buffer.batch_size=16 \
  ppo.loss.mini_batch_size=10 \
  ppo.collector.frames_per_batch=16 \
  logger.mode=offline \
  logger.backend=
""",
    "dreamer": """"python sota-implementations/dreamer/dreamer.py \
  collector.total_frames=200 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=200 \
  env.n_parallel_envs=1 \
  optimization.optim_steps_per_batch=1 \
  logger.video=True \
  logger.backend=csv \
  replay_buffer.buffer_size=120 \
  replay_buffer.batch_size=24 \
  replay_buffer.batch_length=12 \
  networks.rssm_hidden_dim=17
""",
    "ddpg-single": """"python sota-implementations/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
""",
    "redq-single": """"python sota-implementations/redq/redq.py \
  num_workers=2 \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  buffer.batch_size=10 \
  optim.steps_per_batch=1 \
  logger.video=True \
  logger.record_frames=4 \
  buffer.size=120 \
  logger.backend=
""",
    "iql_online-single": """"python sota-implementations/iql/iql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=1 \
  logger.mode=offline \
  logger.backend=
""",
    "cql_online-single": """"python sota-implementations/cql/cql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  logger.mode=offline \
  logger.backend=
""",
    "td3-single": """"python sota-implementations/td3/td3.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=2 \
  collector.env_per_collector=1 \
  logger.mode=offline \
  optim.batch_size=10 \
  env.name=Pendulum-v1 \
  logger.backend=
""",
    "mappo_ippo": """"python sota-implementations/multiagent/mappo_ippo.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "maddpg_iddpg": """"python sota-implementations/multiagent/maddpg_iddpg.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "iql_marl": """"python sota-implementations/multiagent/iql.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "qmix_vdn": """python sota-implementations/multiagent/qmix_vdn.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "marl_sac": """"python sota-implementations/multiagent/sac.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "bandits": """"python sota-implementations/bandits/dqn.py --n_steps=100
""",
}


def run_command(command):
    try:
        # Get the current coverage settings
        cov_settings = os.environ.get('COVERAGE_PROCESS_START')

        if cov_settings:
            # If coverage is enabled, run the command with coverage
            command = f"coverage run --parallel-mode {command}"

        subprocess.check_call(
            command, shell=True, cwd=Path(__file__).parent.parent.parent.parent.parent
        )
    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Command failed with return code {e.returncode}")


@pytest.mark.parametrize("algo", list(commands))
def test_multiagent_commands(algo):
    run_command(commands[algo])


# Test bandits separately
def test_bandits():
    command = ""
    run_command(command)
