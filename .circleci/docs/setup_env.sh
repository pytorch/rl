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

printf "python version"
python --version

pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118

printf "Installing tensordict"
python -m pip install git+https://github.com/pytorch-labs/tensordict.git --quiet

printf "Installing torchrl"
python -m pip install -e . --quiet

printf "Installing requirements"
python -m pip install -r docs/requirements.txt --quiet
printf "Installed all dependencies"
