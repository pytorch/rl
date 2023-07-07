#!/usr/bin/env bash

set -euxo pipefail
set -v

# ==================================================================================== #
# ================================ Init ============================================== #


apt-get update && apt-get upgrade -y
apt-get install -y vim git wget

apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev libosmesa6-dev
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libgles2
apt-get install -y g++ gcc patchelf

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# from cudagl docker image
cp $this_dir/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json


# ==================================================================================== #
# ================================ Setup env ========================================= #

# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

cd "${root_dir}"

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# 1. Install conda at ./conda
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    wget -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-x86_64.sh"
    bash ./miniconda.sh -b -f -p "${conda_dir}"
fi
eval "$(${conda_dir}/bin/conda shell.bash hook)"

# 2. Create test environment at ./env
printf "python: ${PYTHON_VERSION}\n"
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment\n"
    conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"
fi
conda activate "${env_dir}"

# 3. Install mujoco
printf "* Installing mujoco and related\n"
mkdir -p $root_dir/.mujoco
cd $root_dir/.mujoco/
#wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
#tar -xf mujoco-2.1.1-linux-x86_64.tar.gz
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
cd "${root_dir}"

# 4. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
cat "${this_dir}/environment.yml"

export MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210
export DISPLAY=unix:0.0
#export MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/lib/libmujoco.so.2.1.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root_dir/.mujoco/mujoco210/bin
export SDL_VIDEODRIVER=dummy
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

conda env config vars set MUJOCO_PY_MUJOCO_PATH=$root_dir/.mujoco/mujoco210 \
  DISPLAY=unix:0.0 \
#  MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/lib/libmujoco.so.2.1.1 \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root_dir/.mujoco/mujoco210/bin \
  SDL_VIDEODRIVER=dummy \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

conda deactivate
conda activate "${env_dir}"

# install d4rl
pip install free-mujoco-py
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

python -c """import gym;import d4rl"""

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

# ============================================================================================ #
# ================================ PyTorch & TorchRL ========================================= #


if [[ ${#CU_VERSION} -eq 4 ]]; then
    CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
elif [[ ${#CU_VERSION} -eq 5 ]]; then
    CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
fi
echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/$CU_VERSION

# smoke test
python -c "import functorch"

# install snapshot
pip install git+https://github.com/pytorch/torchsnapshot

# install tensordict
pip install git+https://github.com/pytorch-labs/tensordict.git

printf "* Installing torchrl\n"
python setup.py develop

# ==================================================================================== #
# ================================ Run tests ========================================= #



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
python .circleci/unittest/helpers/coverage_run_parallel.py examples/decision_transformer/dt.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10
python .circleci/unittest/helpers/coverage_run_parallel.py examples/decision_transformer/online_td.py \
  optim.pretrain_gradient_steps=55 \
  optim.updates_per_episode=3 \
  optim.warmup_steps=10

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

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
