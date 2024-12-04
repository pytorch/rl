#!/usr/bin/env bash

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# we can install this now but not before installing tensordict and torchrl, g++ version will break the compilation
# https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin
#conda install -y -c conda-forge gcc=12.1.0
conda install -y -c conda-forge libstdcxx-ng=12
conda env config vars set LD_PRELOAD=$LD_PRELOAD:$STDC_LOC

## find libstdc
STDC_LOC=$(find conda/ -name "libstdc++.so.6" | head -1)

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# smoke test
python -c "import habitat;import habitat.gym"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU
# more logging

#wget https://github.com/openai/mujoco-py/blob/master/vendor/10_nvidia.json
#mv 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

conda env config vars set MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

conda deactivate && conda activate ./env


# this workflow only tests the libs
mkdir data
git lfs update

python -m habitat_sim.utils.datasets_download --uids rearrange_pick_dataset_v0 rearrange_task_assets --data-path ./data --no-prune
#python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path ./data --no-prune

python -c "import habitat;import habitat.gym"
python -c """from torchrl.envs.libs.habitat import HabitatEnv
env = HabitatEnv('HabitatRenderPick-v0')
env.reset()
"""

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/test_libs.py --instafail -v --durations 200 --capture no -k TestHabitat --error-for-skips
coverage combine
coverage xml -i
