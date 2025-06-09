# GRPO: Generalized Reward-Conditioned Policy Optimization

This is an implementation of GRPO for language models, built on top of TorchRL.

## Overview

GRPO is a method for training language models using reinforcement learning, with the following key features:
- Multi-GPU support with efficient device management
- Mixed precision training
- Gradient accumulation
- Automatic checkpointing
- Comprehensive logging with Weights & Biases
- Hydra configuration system
- Asynchronous training support with Ray

## Installation

1. Install dependencies:
```bash
# GSM8K deps
pip install -r sota-implementations/grpo/requirements_gsm8k.txt
# IFEval deps
pip install -r sota-implementations/grpo/requirements_ifeval.txt
```

2. Set required environment variables:
```bash
export VLLM_USE_V1=0  # Required for vLLM compatibility
```

## Hardware Requirements

- At least 3 CUDA-capable GPUs:
  - Training device(s)
  - vLLM inference device
  - Reference model device

### Device Management

There are two ways to specify device allocation:

1. Using `num_devices` (Recommended):
```bash
train_model.num_devices=2 ref_model.num_devices=2 inference_model.num_devices=2
```
This approach automatically manages device allocation based on the training mode (sync/async) and prevents device conflicts.

2. Using `devices` (Manual):
```bash
train_model.devices=[0,1] ref_model.devices=[2,3] inference_model.devices=[4,5]
```
This approach requires manual device management and is more error-prone.

The `num_devices` approach is recommended as it:
- Automatically handles device allocation
- Works correctly in both sync and async modes
- Prevents device conflicts between model components
- Is more portable across different machine configurations

## Configuration

The training configuration is managed through Hydra. There are two main configuration files:
- `config/grpo_gsm8k.yaml`: Default configuration for GSM8K tasks (default)
- `config/grpo_ifeval.yaml`: Configuration optimized for IFEval tasks

## Usage

### Basic Training

There are two training modes available:

#### Synchronous Mode (Default)
```bash
VLLM_USE_V1=0 python sota-implementations/grpo/grpo.py train_model.num_devices=2 ref_model.num_devices=2 inference_model.num_devices=2
```

#### Asynchronous Mode (Recommended)
```bash
VLLM_USE_V1=0 python sota-implementations/grpo/grpo-async.py train_model.num_devices=2 ref_model.num_devices=2 inference_model.num_devices=2
```

The async mode offers better performance by:
- Running data collection and optimization concurrently
- More efficient GPU utilization
- Reduced memory overhead
- Better throughput

### Run with IFEval Config

```bash
python grpo.py --config-name grpo_ifeval
```

### Override Config Values

```bash
# Change dataset
python grpo.py env.dataset=ifeval

# Modify training parameters
python grpo.py optimizer.lr=2e-5 optimizer.weight_decay=0.01

# Change model
python grpo.py model.name=meta-llama/Llama-2-7b-hf
```

### Hyperparameter Sweeps

```bash
# Learning rate sweep
python grpo.py --multirun optimizer.lr=1e-4,1e-5,1e-6

# Multiple parameters
python grpo.py --multirun \
  optimizer.lr=1e-4,1e-5 \
  policy.kl_coef=0.01,0.1
```

Don't forget to set the number of value of `train.total_dialog_turns` to a reasonable value!

## Monitoring

Training progress is logged to Weights & Biases with the following metrics:
- Reward
- Advantage
- KL penalty
- Sequence length
- ESS (Effective Sample Size)
- Loss metrics (objective, clip fraction, etc.)
- Gradient norm
- Throughput metrics (in async mode)

## Checkpointing

Checkpoints are saved every `train.checkpoint_frequency` steps and contain:
- Model state
- Optimizer state
- Gradient scaler state (for mixed precision)
- Full configuration

## Debugging Out-of-memory issues

- vLLM: Reduce `inference_model.gpu_memory_utilization=FRACTION` or number of environments run
  in parallel (`env.num_envs=N`).
- KL scoring: If the KL scoring is achieved on the batch of data,
  reduce the number of environments (`env.num_envs=N`) run in parallel.
- Training: Reduce batch size (`train.optim_batch_size`)

## Directory Structure

```
sota-implementations/grpo/
├── config/
│   └── grpo_gsm8k.yaml       # Main configuration file
│   └── grpo_ifeval.yaml      # config file for IFEval task
├── grpo.py            # Synchronous training script
├── grpo-async.py      # Asynchronous training script
├── grpo_utils.py      # Utility functions
└── README.md          # This file
```

## Output Structure

Each run creates a timestamped directory under `outputs/`:
```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── checkpoints/
        │   └── checkpoint_*.pt
        └── .hydra/
            └── config.yaml
```

For hyperparameter sweeps, outputs are stored under `multirun/`.
