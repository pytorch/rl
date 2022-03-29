# TorchRL

TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch. 

It provides pytorch and python-first, low and high level abstractions for RL that are intended to be efficient, docummented and properly tested. 
The code is aimed at supporting research in RL. Most of it is written in python in a highly modular way, such that researchers can easily swap components, transform them or write new ones with little effort.

This repo attempts to align with the existing pytorch ecosystem libraries in that it has a dataset pillar ([torchrl/envs](torchrl/envs)), [transforms](torchrl/envs/transforms), [models](torchrl/modules), data utilities (e.g. collectors and containers)... 
TorchRL aims at having as few dependencies as possible (python standard library, numpy and pytorch). Common environment libraries (e.g. OpenAI gym) are only optional.

On the low-level end, torchrl comes with a set of highly re-usable functionals for [cost functions](torchrl/objectives/costs), [returns](torchrl/objectives/returns) and data processing.

On the high-level end, it provides:
- multiprocess [data collectors](torchrl/collectors/collectors.py);
- a generic [agent class](torchrl/agents/agents.py);
- efficient and generic [replay buffers](torchrl/data/replay_buffers/replay_buffers.py);
- [TensorDict](torchrl/data/tensordict/tensordict.py), a convenient data structure to pass data from one object to another without friction;
- An associated [`TDModule` class](torchrl/modules/td_module/common.py) which is [functorch](https://github.com/pytorch/functorch)-compatible! 
- [interfaces for environments](torchrl/envs) from common libraries (OpenAI gym, deepmind control lab, etc.) and [wrappers for parallel execution](torchrl/envs/vec_env.py), as well as a new pytorch-first class of [tensor-specification class](torchrl/data/tensor_specs.py);
- [environment transforms](torchrl/envs/transforms/transforms.py), which process and prepare the data coming out of the environments to be used by the agent;
- various tools for distributed learning (e.g. [memory mapped tensors](torchrl/data/tensordict/memmap.py));
- various [architectures](torchrl/modules/models/) and models (e.g. [actor-critic](torchrl/modules/td_module/actors.py));
- [exploration wrappers](torchrl/modules/td_module/exploration.py);
- various [recipes](torchrl/agents/helpers/models.py) to build models that correspond to the environment being deployed.

A series of [examples](examples/) are provided with an illustrative purpose:
- [DQN (and add-ons up to Rainbow)](examples/dqn/dqn.py)
- [DDPG](examples/ddpg/ddpg.py)
- [PPO](examples/ppo/ppo.py)
- [SAC](examples/sac/sac.py)
- [REDQ](examples/redq/redq.py)

and many more to come!

## Installation
Create a conda environment where the packages will be installed. 
Before installing anything, make sure you have the latest version of `cmake` and `ninja` libraries:

```
conda create --name torch_rl python=3.9
conda activate torch_rl
conda install cmake -c conda-forge
pip install ninja
```

Depending on the use of functorch that you want to make, you may want to install the latest (nightly) pytorch release or the latest stable version of pytorch:

**Stable**

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch  # refer to pytorch official website for cudatoolkit installation
pip install functorch
```

**Nightly**
```
# For CUDA 10.2
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html --upgrade
# For CUDA 11.1
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html --upgrade
# For CPU-only build
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html --upgrade
```

and functorch
```
pip install --user "git+https://github.com/pytorch/functorch.git"
```

**Torchrl**

Go to the directory where you have cloned the torchrl repo and install it
```
cd /path/to/torchrl/
python setup.py install
```
To run a quick sanity check, leave that directory and try to import the library.
```
python -c "import torchrl"
```

**Optional dependencies**

The following libraries can be installed depending on the usage one wants to make of torchrl:
```
# diverse
pip install tqdm pyyaml configargparse

# rendering
pip install moviepy

# deepmind control suite
pip install dm_control 

# gym, atari games
pip install gym gym[accept-rom-license] pygame gym_retro

# tests
pip install pytest
```

## Running examples
Examples are coded in a very similar way but the configuration may change from one algorithm to the other (e.g. async/sync data collection, hyperparameters, ratio of model updates / frame etc.)
To train an algorithm it is therefore advised to do use the predefined configurations that are found in the `configs` sub-folder in each algorithm directory:
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

## Upcoming features
In the near future, we plan to:
- provide tutorials on how to design new actors or environment wrappers;
- implement IMPALA (as a distributed RL example) and Meta-RL algorithms;
- improve the tests, documentation and nomenclature.
