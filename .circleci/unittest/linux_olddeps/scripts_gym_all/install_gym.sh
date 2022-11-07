# TODO: update gym to the given version.

echo "Installing gym version ${GYM_VERSION}"

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

conda install gym==$GYM_VERSION
