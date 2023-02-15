root_dir="$(pwd)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"


printf "reactivating conda\n"
eval "$(${conda_dir}/bin/conda shell.bash hook)"
conda activate "${env_dir}"

printf "building docs...\n"
cd ./docs
timeout 7m bash -ic "MUJOCO_GL=egl sphinx-build SPHINXOPTS=-v ./source _local_build" || code=$?; if [[ $code -ne 124 && $code -ne 0 ]]; then exit $code; fi
cd ..
printf "done!\n"

git clone --branch gh-pages https://github.com/pytorch-labs/tensordict.git docs/_local_build/tensordict
rm -rf docs/_local_build/tensordict/.git
