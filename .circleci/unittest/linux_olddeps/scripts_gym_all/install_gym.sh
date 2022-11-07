# TODO: update gym to the given version.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

conda install gym==$GYM_VERSION
