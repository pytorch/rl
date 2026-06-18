#!/usr/bin/env bash

set -euxo pipefail

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
libero_dir="${root_dir}/libero-src"
source "${env_dir}/bin/activate"

export PYTHONPATH="${libero_dir}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${root_dir}/.libero-ci"
export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export DISPLAY=:99
export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=0

python -m torch.utils.collect_env
git config --global --add safe.directory '*'

Xvfb :99 -screen 0 1024x768x24 &

python -c 'import torch; t = torch.ones([2, 2], device="cuda:0" if torch.cuda.is_available() else "cpu"); print(t); print("tensor device:" + str(t.device))'
python -c 'from torchrl.envs.libs.libero import _ensure_libero_config; _ensure_libero_config()'
python -c 'from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; print("LIBERO suites:", sorted(benchmark.get_benchmark_dict()))'

json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
json_report_args="--json-report --json-report-file=${json_report_dir}/test-results-libero.json --json-report-indent=2"

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/libs/test_libero.py \
  ${json_report_args} \
  --instafail -v --durations 200 --capture no \
  -k 'not demo_replay_success' \
  --error-for-skips
coverage combine -q
coverage xml -i

python .github/unittest/helpers/upload_test_results.py || echo "Warning: Failed to process test results for flaky tracking"
