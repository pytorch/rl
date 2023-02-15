apt-get update -y
apt-get install software-properties-common -y
add-apt-repository ppa:git-core/candidate -y
apt-get update -y
apt-get upgrade -y
apt-get -y install libglu1-mesa libgl1-mesa-glx libosmesa6 gcc curl g++ unzip wget libglfw3-dev libgles2-mesa-dev libglew-dev sudo git cmake libz-dev zlib1g-dev python3.8 python3-pip ninja


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
conda activate "${env_dir}"

conda install -c conda-forge zlib -y

pip3 install --upgrade pip

python --version

pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118

python -m pip install git+https://github.com/pytorch-labs/tensordict.git

python -m pip install -e .

mkdir _tmp
cd _tmp
python3 -c "import torchrl;from torchrl.data import ReplayBuffer"
cd ..
rm -rf _tmp

python -m pip install -r docs/requirements.txt
cd ./docs
timeout 7m bash -ic "MUJOCO_GL=egl sphinx-build SPHINXOPTS=-v ./source _local_build" || code=$?; if [[ $code -ne 124 && $code -ne 0 ]]; then exit $code; fi
cd ..

git clone --branch gh-pages https://github.com/pytorch-labs/tensordict.git docs/_local_build/tensordict
rm -rf docs/_local_build/tensordict/.git


