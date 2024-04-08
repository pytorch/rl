## Reproducing Importance Weighted Actor-Learner Architecture (IMPALA) Algorithm Results

This repository contains scripts that enable training agents using the IMPALA Algorithm on MuJoCo and Atari environments. We follow the original paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) by Espeholt et al. 2018.

## Examples Structure

Please note that we provide 2 examples, one for single node training and one for distributed training. Both examples rely on the same utils file, but besides that are independent. Each example contains the following files:

1. **Main Script:** The definition of algorithm components and the training loop can be found in the main script  (e.g. impala_single_node_ray.py).

2. **Utils File:** A utility file is provided to contain various helper functions, generally to create the environment and the models (e.g. utils.py).

3. **Configuration File:** This file includes default hyperparameters specified in the original paper. For the multi-node case, the file also includes the configuration file of the Ray cluster. Users can modify these hyperparameters to customize their experiments  (e.g. config_single_node.yaml).


## Running the Examples

You can execute the single node IMPALA algorithm on Atari environments by running the following command:

```bash
python impala_single_node.py
```

You can execute the multi-node IMPALA algorithm on Atari environments by running the following command:

```bash
python impala_single_node_ray.py
``` 
or 

```bash
python impala_single_node_submitit.py
``` 
