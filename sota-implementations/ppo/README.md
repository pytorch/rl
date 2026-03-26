## Reproducing Proximal Policy Optimization (PPO) Algorithm Results

This repository contains scripts that enable training agents using the Proximal Policy Optimization (PPO) Algorithm on MuJoCo and Atari environments. We follow the original paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) by Schulman et al. (2017) to implement the PPO algorithm but introduce the improvement of computing the Generalised Advantage Estimator (GAE) at every epoch.


## Examples Structure

Please note that each example is independent of each other for the sake of simplicity. Each example contains the following files:

1. **Main Script:** The definition of algorithm components and the training loop can be found in the main script  (e.g. ppo_atari.py).

2. **Utils File:** A utility file is provided to contain various helper functions, generally to create the environment and the models (e.g. utils_atari.py).

3. **Configuration File:** This file includes default hyperparameters specified in the original paper. Users can modify these hyperparameters to customize their experiments  (e.g. config_atari.yaml).


## Running the Examples

You can execute the PPO algorithm on Atari environments by running the following command:

```bash
python ppo_atari.py
```

You can execute the PPO algorithm on MuJoCo environments by running the following command:

```bash
python ppo_mujoco.py
``` 
