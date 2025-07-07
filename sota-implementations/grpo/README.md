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

The number of devices for each model component is specified using `num_devices`:

```bash
train_model.num_devices=2 ref_model.num_devices=2 inference_model.num_devices=2
```

This approach:
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
VLLM_USE_V1=0 python sota-implementations/grpo/grpo-sync.py mode=sync train_model.num_devices=2 ref_model.num_devices=2 inference_model.num_devices=2
```

#### Asynchronous Mode (Recommended)
```bash
VLLM_USE_V1=0 python sota-implementations/grpo/grpo-async.py mode=async train_model.num_devices=2 ref_model.num_devices=2 inference_model.num_devices=2
```

The key difference between sync and async modes is how data collection and optimization are handled:

**Synchronous Mode (grpo-sync.py)**:
```python
# Three nested loops:
for data in collector:  # Data collection loop
    for epoch in range(epochs):  # Epoch loop
        for batch in replay_buffer:  # Buffer consumption loop
            # Optimize on batch
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()
    # Weight update
    weight_updater.push_weights(policy_training)
```

**Asynchronous Mode (grpo-async.py)**:
```python
# Start data collection in background
collector.start()

# Single optimization loop
for step in range(total_steps):
    # Sample and optimize
    batch = replay_buffer.sample()
    loss = loss_fn(batch)
    loss.backward()
    optimizer.step()
    # Update weights once in a while
    if cond():
      weight_updater.push_weights(policy_training)

```

Key differences:
1. **Data Collection**: 
   - Sync: Data collection and optimization happen sequentially.
     
     *Note*: The `train.sync_iter=False` argument can be used to collect data whilst optimizing. In this context, the
     maximum policy age will be 1. If `train.sync_iter=True` (default), the maximum policy age is `0`.

   - Async: Data collection runs in background while optimization happens

2. **Buffer Size**:
   - Sync: Buffer size must equal the batch size returned by collector (`buffer_size = dialog_turns_per_batch`)
   - Async: Buffer can be larger than the batch size, allowing for more diverse sampling

3. **Data Processing**:
   - Sync: Processes the same data multiple times (epochs)
   - Async: Each piece of data is processed a non-deterministic number of times.

4. **Weight updates**:
   - Sync: Weights are updated befor every collection of data.
   - Async: Weights are updated at a given interval (in gradient steps). This will require a synchronization between the training
     and inference processes, and frequent updates will cause both workers to often wait for each other.

The async mode offers better performance by:
- Running data collection and optimization concurrently
- More efficient GPU utilization
- Reduced memory overhead
- Better throughput
- More flexible buffer management

### Running GRPO on More Than One Node with SLURM

GRPO can be run across more than one node using SLURM, enabling distributed training for moderately scaled workloads.

Two scripts are provided for launching multi-node runs:

- `grpo-sync-multi-node.sbatch`: SLURM job script that launches sync GRPO across multiple nodes using Ray.
- `grpo-async-multi-node.sbatch`: SLURM job script that launches async GRPO across multiple nodes using Ray.

Example Usage:

```bash
sbatch sota-implementations/grpo/grpo-sync-multi-node.sbatch

### KL Divergences in PPO: Reference vs Inference

KL divergence is a key regularization term in policy optimization algorithms like PPO and in LLM post-training. It measures how much the updated policy diverges from a baseline or reference policy, helping to prevent the new policy from drifting too far and ensuring stable learning.

There are two main types of KL divergences commonly used:

#### 1. KL to Reference Policy (KL[ref || policy])
- **Definition:** Measures how much the new (learned) policy diverges from a fixed reference policy (often the original, pre-trained model).
- **Implementation:** In GRPO, this is computed as `(ref_log_prob - cur_log_prob).expm1() - (ref_log_prob - cur_log_prob)`, which is a numerically stable way to compute KL for log probabilities.
- **Usage:**
  - **LLM Post-Training:** This is the canonical choice in LLM post-training (e.g., RLHF, DPO, GRPO). The reference is usually the original language model before any RL fine-tuning. Penalizing KL[ref || policy] ensures the fine-tuned model stays close to the original, preserving language quality and preventing over-optimization.
  - **Effect:** Encourages the new policy to not deviate too much from the reference, maintaining fluency and generalization.

#### 2. KL to Inference Policy (KL[policy || inference])
- **Definition:** Measures how much the current policy diverges from the policy used to generate the data (the inference policy, sometimes called the behavior policy).
- **Implementation:** In GRPO, this is approximated as `prev_log_prob - cur_log_prob`, where `prev_log_prob` is from the inference policy that generated the data.
- **Usage:**
  - **Canonical PPO:** In standard PPO (especially in RL for control), this is the canonical KL: KL[policy || inference]. The inference policy is the one that generated the trajectories in the replay buffer. Penalizing this KL ensures that the updated policy does not move too far from the data distribution, stabilizing importance sampling and learning.
  - **Effect:** Prevents the policy from making large, unstable updates relative to the data it was trained on.

#### Summary Table
| Setting            | Canonical KL Term         | Purpose                                    |
|--------------------|--------------------------|---------------------------------------------|
| PPO (RL control)   | KL[policy || inference]  | Stabilize updates, match data distribution  |
| LLM Post-Training  | KL[ref || policy]        | Stay close to pre-trained model             |

In GRPO, both types of KL can be used and controlled via configuration. Typically, for LLM post-training, the KL to reference is the most important for preserving model quality, while the KL to inference is more about stabilizing the optimization process.

The KL contributions to the loss can be controlled via the `train.kl_to_ref_coeff` and `train.kl_to_inference_coeff`, respectively.

Additionally, the KL to ref loss contribution can be either added to the reward during the grading of the LLM response, or added directly to the loss given by the `train.kl_coef_in_loss` config option.

In the original GRPO paper, the KL to reference (KL[ref || policy]) is added **directly to the loss function**, not to the reward. This means that the KL penalty acts as a regularizer during optimization, discouraging the policy from drifting too far from the reference model at every update step. This is in contrast to some RLHF-style approaches, where the KL penalty is added to the reward signal during data collection (i.e., the environment's reward is modified). 

**Why does this matter?**
- **KL in the loss (as in GRPO):** The optimization explicitly balances the policy objective and the KL penalty at each gradient step, making the trade-off more direct and stable. This is the canonical approach in GRPO and is controlled by setting `train.kl_coef_in_loss=True` in the config.
- **KL in the reward:** The KL penalty is treated as part of the environment's reward, so the policy is trained to maximize this modified reward. This can sometimes make the effect of the KL less direct, as it is mixed with the task reward during data collection.

In summary, GRPO's approach of adding the KL to reference directly to the loss provides more explicit and stable regularization, and is the recommended setting for most LLM post-training scenarios.

### Run with IFEval Config

```bash
python grpo-sync.py mode=sync --config-name grpo_ifeval
```

### Override Config Values

```bash
# Change dataset
python grpo-sync.py mode=sync env.dataset=ifeval

# Modify training parameters
python grpo-sync.py mode=sync optimizer.lr=2e-5 optimizer.weight_decay=0.01

# Change model
python grpo-sync.py mode=sync model.name=meta-llama/Llama-2-7b-hf
```

### Hyperparameter Sweeps

```bash
# Learning rate sweep
python grpo-sync.py mode=sync --multirun optimizer.lr=1e-4,1e-5,1e-6

# Multiple parameters
python grpo-sync.py mode=sync --multirun \
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
├── grpo-sync.py       # Synchronous training script
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
