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

## Installation

1. Install dependencies:
```bash
# GSM8K deps
pip install -r sota-implementations/llm/requirements_gsm8k.txt
# IFEval deps
pip install -r sota-implementations/llm/requirements_ifeval.txt
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

## Configuration

The training configuration is managed through Hydra. The main config file is `config/grpo.yaml`:

```yaml
env:
  dataset: gsm8k  # choices: [gsm8k, ifeval]
  num_envs: 8     # number of parallel environments
  repeats: 16     # action repeats for GRPO

model:
  name: Qwen/Qwen2.5-3B  # HuggingFace model name
  compile: false         # enable torch.compile

policy:
  kl_coef: 1e-2  # KL penalty coefficient

train:
  epochs: 1
  steps_per_batch: 64
  optim_batch_size: 4
  gradient_accumulation_steps: 1
  mixed_precision: true
  optimizer:
    name: AdamW
    lr: 1e-5
    clip_grad_norm: 0.5

system:
  gpu_memory_utilization: 0.5

logging:
  checkpoint_dir: checkpoints
  experiment_name: null  # auto-generated if null
  checkpoint_frequency: 10  # save every N batches
```

## Usage

### Basic Training

```bash
python grpo.py
```

### Run with IFEval Config

```bash
python grpo.py --config-name grpo_ifeval
```

### Override Config Values

```bash
# Change dataset
python grpo.py env.dataset=ifeval

# Modify training parameters
python grpo.py train.epochs=2 train.optimizer.lr=2e-5

# Change model
python grpo.py model.name=meta-llama/Llama-2-7b-hf
```

### Hyperparameter Sweeps

```bash
# Learning rate sweep
python grpo.py --multirun train.optimizer.lr=1e-4,1e-5,1e-6

# Multiple parameters
python grpo.py --multirun \
  train.optimizer.lr=1e-4,1e-5 \
  policy.kl_coef=0.01,0.1
```

## Monitoring

Training progress is logged to Weights & Biases with the following metrics:
- Reward
- Advantage
- KL penalty
- Sequence length
- ESS (Effective Sample Size)
- Loss metrics (objective, clip fraction, etc.)
- Gradient norm

## Checkpointing

Checkpoints are saved every `logging.checkpoint_frequency` batches and contain:
- Model state
- Optimizer state
- Gradient scaler state (for mixed precision)
- Full configuration

## Debugging

### Out-of-memory issues

- vLLM: Reduce `inference_model.gpu_memory_utilization=FRACTION` or number of environments run
  in parallel (`env.num_envs=N`).
- KL scoring: If the KL scoring is achieved on the batch of data,
  reduce the number of environments (`env.num_envs=N`) run in parallel.
- Training: Reduce batch size (`train.optim_batch_size`)

### vLLM / Ray 

## Directory Structure

```
sota-implementations/llm/
├── config/
│   └── grpo.yaml       # Main configuration file
├── grpo.py            # Training script
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

## Citation

If you use this implementation in your research, please cite:
```bibtex
@misc{grpo2024,
  title={GRPO: Generalized Reward-Conditioned Policy Optimization},
  author={[Authors]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
