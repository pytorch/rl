# TorchRL

TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch. 

It provides pytorch and python-first, low and high level abstractions for RL that are intended to be efficient, docummented and properly tested. 
The code is aimed at supporting research in RL. Most of it is written in python in a highly modular way, such that researchers can easily swap components, transform them or write new ones with little effort.

This repo attempts to align with the existing pytorch ecosystem libraries in that it has a dataset pillar ([torchrl/envs](torchrl/envs)), [transforms](torchrl/data/transforms), [models](torchrl/modules), data utilities (e.g. collectors and containers)... 
TorchRL aims at having as few dependencies as possible (python standard library, numpy and pytorch). Common environment libraries (e.g. OpenAI gym) are only optional.

On the low-level end, torchrl comes with a set of highly re-usable functionals for [cost functions](torchrl/objectives/costs), [returns](torchrl/objectives/returns) and data processing.

On the high-level end, it provides:
- multiprocess [data collectors](torchrl/collectors/collectors.py);
- a generic [agent class](torchrl/agents/agents.py);
- efficient and generic [replay buffers](torchrl/data/replay_buffers/replay_buffers.py);
- [TensorDict](torchrl/data/tensordict/tensordict.py), a convenient data structure to pass data from one object to another without friction;
- [interfaces for environments](torchrl/envs) from common libraries (OpenAI gym, deepmind control lab, etc.) and [wrappers for parallel execution](torchrl/envs/vec_env.py), as well as a new pytorch-first class of [tensor-specification class](torchrl/data/tensor_specs.py);
- [environment transforms](torchrl/data/transforms/transforms.py), which process and prepare the data coming out of the environments to be used by the agent;
- various tools for distributed learning (e.g. [memory mapped tensors](torchrl/data/tensordict/memmap.py));
- various [architectures](torchrl/modules/models/) and models (e.g. [actor-critic](torchrl/modules/td_module/actors.py));
- [exploration wrappers](torchrl/modules/td_module/exploration.py);
- various [recipes](torchrl/agents/helpers/models.py) to build models that correspond to the environment being deployed.

A series of [examples](examples/) are provided with an illustrative purpose:
- [DQN (and add-ons up to Rainbow)](examples/dqn/dqn.py)
- [DDPG](examples/ddpg/ddpg.py)
- [PPO](examples/ppo/ppo.py)
- [SAC](examples/sac/sac.py)
and many more to come!

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
