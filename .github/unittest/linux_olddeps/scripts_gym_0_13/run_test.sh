#!/usr/bin/env bash

set -e
set -v

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pytorch/rl/mujoco-py/mujoco_py/binaries/linux/mujoco210/bin
export MKL_THREADING_LAYER=GNU
export BATCHED_PIPE_TIMEOUT=60

python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test.py -v --durations 200
python .github/unittest/helpers/coverage_run_parallel.py -m pytest test/smoke_test_deps.py -v --durations 200 -k 'test_gym'

export DISPLAY=:99
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &

# Test sharding: the single serial run of the full suite reached 84 minutes,
# making olddeps the critical path of the unit-test workflows. The workflow
# fans the job out over three shards (TORCHRL_TEST_SHARD); "all" keeps the
# previous single-invocation behavior. The union of the three shards equals
# "all". Every shard stays serial inside its job: these tests share one GPU
# and the process-spawning set flakes next to parallel workers.
# - transforms: test/transforms plus the test_setup.py install tests (both
#   off the critical path of the big remainder);
# - quarantine: the process-spawning set (mirrors linux/scripts/run_all.sh);
# - remainder: everything else.
TORCHRL_TEST_SHARD="${TORCHRL_TEST_SHARD:-all}"

quarantine_tests="test/envs/test_parallel.py test/envs/test_special.py \
test/envs/test_auto_reset.py test/envs/test_env_base.py \
test/envs/test_nested.py test/envs/test_step_mdp.py test/test_collectors.py \
test/services test/test_inference_server.py test/test_loggers.py"
quarantine_ignores=""
for quarantine_path in ${quarantine_tests}; do
    quarantine_ignores+="--ignore ${quarantine_path} "
done

# JSON report for flaky-test tracking and future shard balancing; lands in
# the job artifact when RUNNER_ARTIFACT_DIR is set.
json_report_dir="${RUNNER_ARTIFACT_DIR:-${root_dir}}"
json_report_args="--json-report --json-report-file=${json_report_dir}/test-results-olddeps-shard-${TORCHRL_TEST_SHARD}.json --json-report-indent=2"

common_args=(--instafail -v --durations 200 -k "not HalfCheetah-v2" --mp_fork_if_no_cuda ${json_report_args})

case "${TORCHRL_TEST_SHARD}" in
    transforms)
        CKPT_BACKEND=torch MUJOCO_GL=egl python .github/unittest/helpers/coverage_run_parallel.py -m pytest \
            test/transforms test/test_setup.py \
            "${common_args[@]}"
        ;;
    quarantine)
        CKPT_BACKEND=torch MUJOCO_GL=egl python .github/unittest/helpers/coverage_run_parallel.py -m pytest \
            ${quarantine_tests} \
            "${common_args[@]}"
        ;;
    remainder)
        CKPT_BACKEND=torch MUJOCO_GL=egl python .github/unittest/helpers/coverage_run_parallel.py -m pytest \
            test \
            --ignore test/test_distributed.py \
            --ignore test/test_rlhf.py \
            --ignore test/llm \
            --ignore test/transforms \
            --ignore test/test_setup.py \
            ${quarantine_ignores} \
            "${common_args[@]}"
        ;;
    all)
        CKPT_BACKEND=torch MUJOCO_GL=egl python .github/unittest/helpers/coverage_run_parallel.py -m pytest \
            --ignore test/test_distributed.py \
            --ignore test/test_rlhf.py \
            --ignore test/llm \
            "${common_args[@]}"
        ;;
    *)
        echo "Unknown TORCHRL_TEST_SHARD='${TORCHRL_TEST_SHARD}'. Expected: all|transforms|quarantine|remainder."
        exit 2
        ;;
esac

#pytest --instafail -v --durations 200
#python test/libs
coverage combine -q
coverage xml -i
