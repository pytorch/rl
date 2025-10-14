#!/usr/bin/env bash

set -e
set -v

root_dir="$(git rev-parse --show-toplevel)"
source "${root_dir}/.venv/bin/activate"

# we can install this now but not before installing tensordict and torchrl, g++ version will break the compilation
# https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin
# Install libstdc++ if needed via apt
apt-get install -y libstdc++6

## find libstdc
STDC_LOC=$(find ${root_dir}/.venv/ -name "libstdc++.so.6" 2>/dev/null | head -1 || echo "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")

export MAX_IDLE_COUNT=1000
export LD_PRELOAD=$LD_PRELOAD:$STDC_LOC
export TOKENIZERS_PARALLELISM=true
export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export MKL_THREADING_LAYER=GNU

python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

env_dir="${root_dir}/.venv"
lib_dir="${env_dir}/lib"

# smoke test
python -c "import habitat;import habitat.gym"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir


# this workflow only tests the libs
python -c "import habitat;import habitat.gym"
python -c """from torchrl.envs.libs.habitat import HabitatEnv
env = HabitatEnv('HabitatRenderPick-v0')
env.reset()
"""

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 --capture no -k TestHabitat --error-for-skips
coverage combine
coverage xml -i
