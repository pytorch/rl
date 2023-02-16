#apt-get update -y
#apt-get install software-properties-common -y
#add-apt-repository ppa:git-core/candidate -y
#apt-get update -y
#apt-get upgrade -y
#apt-get -y install libglfw3 libglew2.0 gcc curl g++ unzip \
#  wget sudo git cmake libz-dev \
#  zlib1g-dev python3.8 python3-pip ninja

#yum install -y mesa-libGL freeglut egl-utils glew glfw
#yum install -y glew glfw
apt-get update && apt-get install -y git wget gcc g++

root_dir="$(pwd)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

os=Linux

# 1. Install conda at ./conda
printf "* Installing conda\n"
wget -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-x86_64.sh"
bash ./miniconda.sh -b -f -p "${conda_dir}"

eval "$(${conda_dir}/bin/conda shell.bash hook)"

printf "* Creating a test environment\n"
conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"

printf "* Activating\n"
conda activate "${env_dir}"

conda install -c conda-forge zlib -y

pip3 install --upgrade pip --quiet --root-user-action=ignore

printf "python version\n"
python --version

pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118 --quiet --root-user-action=ignore
#pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu --quiet --root-user-action=ignore

printf "Installing tensordict\n"
pip3 install git+https://github.com/pytorch-labs/tensordict.git --quiet --root-user-action=ignore

printf "Installing torchrl\n"
pip3 install -e . --quiet --root-user-action=ignore

printf "Installing requirements\n"
pip3 install -r docs/requirements.txt --quiet --root-user-action=ignore
printf "Installed all dependencies\n"

printf "smoke test\n"
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python3 -c """from torchrl.envs.libs.dm_control import DMControlEnv
print(DMControlEnv('cheetah', 'run').reset())
"""

printf "building docs...\n"
cd ./docs
#timeout 7m bash -ic "MUJOCO_GL=egl sphinx-build SPHINXOPTS=-v ./source _local_build" || code=$?; if [[ $code -ne 124 && $code -ne 0 ]]; then exit $code; fi
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl sphinx-build ./source _local_build
cd ..
printf "done!\n"

git clone --branch gh-pages https://github.com/pytorch-labs/tensordict.git docs/_local_build/tensordict
rm -rf docs/_local_build/tensordict/.git
