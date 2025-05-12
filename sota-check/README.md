# SOTA Performance checks

This folder contains a `submitit-release-check.sh` file that executes all
the training scripts using `sbatch` with the default configuration and long them
into a common WandB project.

This script is to be executed before every release to assess the performance of
the various algorithms available in torchrl. The name of the project will include
the specific commit of torchrl used to run the scripts (e.g. `torchrl-examples-check-<commit>`).

## Usage

To display the script usage, you can use the `--help` option:

```bash
./submitit-release-check.sh --help
```

## Setup

The following setup should allow you to run the scripts:

```bash
export MUJOCO_GL=egl

conda create -n rl-sota-bench python=3.10 -y 
conda install anaconda::libglu -y
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
pip3 install "gymnasium[atari,mujoco]" vmas tqdm wandb pygame "moviepy<2.0.0" imageio submitit hydra-core transformers

cd /path/to/tensordict
python setup.py develop
cd /path/to/torchrl
python setup.py develop
```
