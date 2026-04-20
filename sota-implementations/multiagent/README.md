# Multi-agent examples

In this folder we provide a set of multi-agent example scripts using the [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) simulator.

<p align="center">
<img src="https://pytorch.s3.amazonaws.com/torchrl/github-artifacts/img/marl_vmas.png" width="600px">
</p>

<center><i>The MARL algorithms contained in the scripts of this folder run on three multi-robot tasks in VMAS.</i></center>

For more details on the experiment setup and the environments please refer to the corresponding section of the appendix in the [TorchRL paper](https://arxiv.org/abs/2306.00577).

> [!NOTE]  
> If you are interested in Multi-Agent Reinforcement Learning (MARL) in TorchRL, check out [BenchMARL](https://github.com/facebookresearch/BenchMARL):
> a benchmarking library where you 
> can train and compare MARL algorithms, tasks, and models using TorchRL!

## Using the scripts

### Install

First you need to install vmas and the dependencies of the scripts.

Install torchrl and tensordict following repo instructions.

Install vmas and dependencies:

```bash
pip install vmas
pip install wandb "moviepy<2.0.0"
pip install hydra-core
```

### Run

To run the scripts just execute the corresponding python file after having modified the corresponding config 
according to your needs.
The config can be found in the .yaml file with the same name.

For example:
```bash
python mappo_ippo.py
```

You can even change the config from the command line like:

```bash
python mappo_ippo.py --m env.scenario_name=navigation
```

### Computational demand
The scripts are set up for collecting many frames, if your compute is limited, you can change the "frames_per_batch" 
and "num_epochs" parameters to reduce compute requirements.

### Script structure

The scripts are self-contained.
This means that all the code you will need to look at is contained in the script file. 
No helper functions are used.

The structure of scripts follows this order:
- Configuration dictionary for the script
- Environment creation
- Modules creation
- Collector instantiation
- Replay buffer instantiation
- Loss module creation
- Training loop (with inner minibatch loops)
- Evaluation run (at the desired frequency)

Logging is done by default to wandb.
The logging backend can be changed in the config files to one of "wandb", "tensorboard", "csv", "mlflow".

All the scripts follow the same on-policy training structure so that results can be compared across different algorithms.
