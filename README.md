[![facebookresearch](https://circleci.com/gh/facebookresearch/rl.svg?style=shield)](https://circleci.com/gh/facebookresearch/rl)

# TorchRL

## Disclaimer

This library is not officially released yet and is subject to change.

The features are available before an official release so that users and collaborators can get early access and provide feedback. No guarantee of stability, robustness or backward compatibility is provided.

---

TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch. 

It provides pytorch and python-first, low and high level abstractions for RL that are intended to be efficient, documented and properly tested. 
The code is aimed at supporting research in RL. Most of it is written in python in a highly modular way, such that researchers can easily swap components, transform them or write new ones with little effort.

This repo attempts to align with the existing pytorch ecosystem libraries in that it has a dataset pillar ([torchrl/envs](torchrl/envs)), [transforms](torchrl/envs/transforms), [models](torchrl/modules), data utilities (e.g. collectors and containers), etc. 
TorchRL aims at having as few dependencies as possible (python standard library, numpy and pytorch). Common environment libraries (e.g. OpenAI gym) are only optional.

On the low-level end, torchrl comes with a set of highly re-usable functionals for [cost functions](torchrl/objectives/costs), [returns](torchrl/objectives/returns) and data processing.

On the high-level end, torchrl provides:
- multiprocess [data collectors](torchrl/collectors/collectors.py);
- efficient and generic [replay buffers](torchrl/data/replay_buffers/replay_buffers.py);
- [TensorDict](torchrl/data/tensordict/tensordict.py), a convenient data structure to pass data from one object to another without friction;
- An associated [`TensorDictModule` class](torchrl/modules/tensordict_module/common.py) which is [functorch](https://github.com/pytorch/functorch)-compatible! 
- [interfaces for environments](torchrl/envs) from common libraries (OpenAI gym, deepmind control lab, etc.) and [wrappers for parallel execution](torchrl/envs/vec_env.py), as well as a new pytorch-first class of [tensor-specification class](torchrl/data/tensor_specs.py);
- [environment transforms](torchrl/envs/transforms/transforms.py), which process and prepare the data coming out of the environments to be used by the agent;
- various tools for distributed learning (e.g. [memory mapped tensors](torchrl/data/tensordict/memmap.py));
- various [architectures](torchrl/modules/models/) and models (e.g. [actor-critic](torchrl/modules/tensordict_module/actors.py));
- [exploration wrappers](torchrl/modules/tensordict_module/exploration.py) and [modules](torchrl/modules/models/exploration.py);
- various [recipes](torchrl/trainers/helpers/models.py) to build models that correspond to the environment being deployed;
- a generic [trainer class](torchrl/trainers/trainers.py).

A series of [examples](examples/) are provided with an illustrative purpose:
- [DQN (and add-ons up to Rainbow)](examples/dqn/dqn.py)
- [DDPG](examples/ddpg/ddpg.py)
- [PPO](examples/ppo/ppo.py)
- [SAC](examples/sac/sac.py)
- [REDQ](examples/redq/redq.py)

and many more to come!

We also provide [tutorials and demos](tutorials) that give a sense of what the 
library can do.

## Installation
Create a conda environment where the packages will be installed. 
Before installing anything, make sure you have the latest version of the `ninja` library:

```
conda create --name torch_rl python=3.9
conda activate torch_rl
pip install ninja
```

Depending on the use of functorch that you want to make, you may want to install the latest (nightly) pytorch release or the latest stable version of pytorch:

**Stable**

```
# For CUDA 10.2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# For CUDA 11.3
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
# For CPU-only build
conda install pytorch torchvision cpuonly -c pytorch

pip install functorch
```

**Nightly**
```
# For CUDA 10.2
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu102
# For CUDA 11.3
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu113
# For CPU-only build
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

and functorch
```
pip install "git+https://github.com/pytorch/functorch.git"
```

If the generation of this artifact in MacOs M1 doesn't work correctly or in the execution the message 
`(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e'))` appears, then try

```
ARCHFLAGS="-arch arm64" pip install "git+https://github.com/pytorch/functorch.git"
```

**Torchrl**

You can install the latest release by using
```
pip install torchrl
```
This should work on linux and MacOs (not M1). For Windows and M1/M2 machines, one 
should install the library locally (see below).

To install extra dependencies, call
```
pip install "torchrl[atari,dm_control,gym_continuous,rendering,tests,utils]"
```
or a subset of these.

Alternatively, as the library is at an early stage, it may be wise to install 
it in develop mode as this will make it possible to pull the latest changes and 
benefit from them immediately. 
Start by cloning the repo:
```
git clone https://github.com/facebookresearch/rl
```

Go to the directory where you have cloned the torchrl repo and install it
```
cd /path/to/torchrl/
python setup.py develop
```

If the generation of this artifact in MacOs M1 doesn't work correctly or in the execution the message 
`(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e'))` appears, then try

```
ARCHFLAGS="-arch arm64" python setup.py develop
```

To run a quick sanity check, leave that directory (e.g. by executing `cd ~/`) 
and try to import the library.
```
python -c "import torchrl"
```
This should not return any warning or error.

**Optional dependencies**

The following libraries can be installed depending on the usage one wants to 
make of torchrl:
```
# diverse
pip install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher

# rendering
pip install moviepy

# deepmind control suite
pip install dm_control 

# gym, atari games
pip install gym "gym[accept-rom-license]" pygame gym_retro

# tests
pip install pytest pyyaml pytest-instafail
```

**Troubleshooting**

If a `ModuleNotFoundError: No module named ‘torchrl._torchrl` errors occurs, 
it means that the C++ extensions were not installed or not found. 
One common reason might be that you are trying to import torchrl from within the 
git repo location. Indeed the following code snippet should return an error if 
torchrl has not been installed in `develop` mode:
```
cd ~/path/to/rl/repo
python -c 'from torchrl.envs import GymEnv'
```
If this is the case, consider executing torchrl from another location.

On **MacOs**, we recommend installing XCode first. 
With Apple Silicon M1 chips, make sure you are using the arm64-built python 
(e.g. [here](https://betterprogramming.pub/how-to-install-pytorch-on-apple-m1-series-512b3ad9bc6)). Running the following lines of code

```
wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py
python collect_env.py
```
should display
```
OS: macOS *** (arm64)
```
and not
```
OS: macOS **** (x86_64)
```

## Running examples
Examples are coded in a very similar way but the configuration may change from one algorithm to another (e.g. async/sync data collection, hyperparameters, ratio of model updates / frame etc.)
To train an algorithm it is therefore advised to use the predefined configurations that are found in the `configs` sub-folder in each algorithm directory:
```
python examples/ppo/ppo.py --config=examples/ppo/configs/humanoid.txt
```
Note that using the config files requires the [configargparse](https://pypi.org/project/ConfigArgParse/) library. 

One can also overwrite the config parameters using flags, e.g.
```
python examples/ppo/ppo.py --config=examples/ppo/configs/humanoid.txt --frame_skip=2 --collection_devices=cuda:1
```

Each example will write a tensorboard log in a dedicated folder, e.g. `ppo_logging/...`.

## Contributing

Internal collaborations to torchrl are welcome! Feel free to fork, submit issues and PRs.
You can checkout the detailed contribution guide [here](CONTRIBUTING.md).

Contributors are recommended to install [pre-commit hooks](https://pre-commit.com/) (using `pre-commit install`). pre-commit will check for linting related issues when the code is commited locally. You can disable th check by appending `-n` to your commit command: `git commit -m <commit message> -n`


## Upcoming features

In the near future, we plan to:
- provide tutorials on how to design new actors or environment wrappers;
- implement IMPALA (as a distributed RL example) and Meta-RL algorithms;
- improve the tests, documentation and nomenclature.

# License
TorchRL is licensed under the MIT License. See [LICENSE](LICENSE) for details.
