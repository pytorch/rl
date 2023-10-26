## Reproducing Advantage Actor Critic (A2C) Algorithm Results

This repository contains scripts that enable training agents using the Advantage Actor Critic (A2C) Algorithm on MuJoCo and Atari environments. We follow the original paper [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) by Mnih et al. (2016) to implement the A2C algorithm but fix the number of steps during the collection phase.


## Examples Structure

Please note that each example is independent of each other for the sake of simplicity. Each example contains the following files:

1. **Main Script:** The definition of algorithm components and the training loop can be found in the main script  (e.g. a2c_atari.py).

2. **Utils File:** A utility file is provided to contain various helper functions, generally to create the environment and the models (e.g. utils_atari.py).

3. **Configuration File:** This file includes default hyperparameters specified in the original paper. Users can modify these hyperparameters to customize their experiments  (e.g. config_atari.yaml).


## Running the Examples

You can execute the A2C algorithm on Atari environments by running the following command:

```bash
python a2c_atari.py
```

You can execute the A2C algorithm on MuJoCo environments by running the following command:

```bash
python a2c_mujoco.py
``` 
