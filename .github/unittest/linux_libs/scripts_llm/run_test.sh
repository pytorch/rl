#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

apt-get update && apt-get install -y git gcc
ln -s /usr/bin/swig3.0 /usr/bin/swig

export PYTORCH_TEST_WITH_SLOW='1'
export LAZY_LEGACY_OP=False

# to solve RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

conda deactivate && conda activate ./env

python -c "import transformers, datasets"

pytest test/test_rlhf.py --instafail -v --durations 200 --capture no --error-for-skips

pytest test/test_actors.py test/test_collector.py -k llm --instafail -v --durations 200 --capture no --error-for-skips --runslow

pytest examples/rlhf/train_rlhf.py \
  sys.device=cuda:0 sys.ref_device=cuda:0 \
  model.name_or_path=gpt2 train.max_epochs=2 \
  data.batch_size=2 train.ppo.ppo_batch_size=2 \
  train.ppo.ppo_num_epochs=1 reward_model.name_or_path= \
  train.ppo.episode_length=8 train.ppo.num_rollouts_per_epoch=4 \
  data.block_size=110 io.logger=csv

coverage combine
coverage xml -i
