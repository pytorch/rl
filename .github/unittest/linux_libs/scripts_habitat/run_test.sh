#!/usr/bin/env bash

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# Note: We no longer install libstdcxx-ng=12 here since we're building habitat-sim from source
# and the pip-installed PyTorch has its own compatible libstdc++

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False

# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# Set library path to include environment lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU

# Verify torch works
python -m torch.utils.collect_env

# Set habitat environment variables
conda env config vars set \
  MAX_IDLE_COUNT=1000 \
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=true
conda deactivate && conda activate ./env

# smoke test
python -c "import habitat;import habitat.gym"
python -c """from torchrl.envs.libs.habitat import HabitatEnv
env = HabitatEnv('HabitatRenderPick-v0')
env.reset()
"""

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 --capture no -k TestHabitat --error-for-skips
coverage combine -q
coverage xml -i
