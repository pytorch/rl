## Reproducing Deep Q-Learning (DQN) Algorithm Results

This repository contains scripts that enable training agents using the Deep Q-Learning (DQN) Algorithm on CartPole and Atari environments. For Atari, We follow the original paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) by Mnih et al. (2013).


## Examples Structure

Please note that each example is independent of each other for the sake of simplicity. Each example contains the following files:

1. **Main Script:** The definition of algorithm components and the training loop can be found in the main script  (e.g. dqn_atari.py).

2. **Utils File:** A utility file is provided to contain various helper functions, generally to create the environment and the models (e.g. utils_atari.py).

3. **Configuration File:** This file includes default hyperparameters specified in the original paper. Users can modify these hyperparameters to customize their experiments  (e.g. config_atari.yaml).


## Running the Examples

You can execute the DQN algorithm on the CartPole environment by running the following command:

```bash
python dqn_cartpole.py

``` 

You can execute the DQN algorithm on Atari environments by running the following command:

```bash
python dqn_atari.py
```
