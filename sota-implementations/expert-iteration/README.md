# Expert Iteration: Learning from Top-K Responses

This is an implementation of Expert Iteration for language models, built on top of TorchRL. 
Expert Iteration is a reinforcement learning-like method that learns from the best performing responses in a batch, rather than using all responses equally.

The idea of these scripts is extremely simple:
- Collect some trajectories with a pre-trained version of the model;
- Select the top-K best trajectories of the batch (based on their reward);
- Train the model using SFT of these trajectories;
- Update the inference model.

## Overview

The version of Expert Iteration presented here has the following features:

- **Top-K Selection**: Only the best performing responses are used for training, improving sample efficiency
- **KL Regularization**: Maintains model quality by penalizing divergence from a reference model
- **Multi-GPU support** with efficient device management
- **Mixed precision training** for memory efficiency
- **Gradient accumulation** for larger effective batch sizes
- **Automatic checkpointing** and comprehensive logging with Weights & Biases
- **Hydra configuration system** for easy experimentation
- **Asynchronous training support** with Ray for improved throughput
- **Prioritized sampling** such that samples with higher rewards have more chances of being sampled

## Key Differences from GRPO and other RL algorithms

### 1. Top-K Reward Selection

Unlike other RL post-training recipes (e.g. GRPO) which uses all responses, 
Expert Iteration employs a `TopKRewardSelector` transform that:

- Collects multiple responses for each prompt (controlled by `env.repeats`)
- Selects only the top-k responses based on reward (controlled by `train.topk_size`)
- Writes only the best responses to the replay buffer, improving training efficiency

```python
# Example: For each prompt, generate 32 responses but only keep the best 4
env.repeats = 32  # Generate 32 responses per prompt
train.topk_size = 4  # Keep only the top 4 responses
```

### 2. KL Divergence Handling

Expert Iteration uses a different approach to KL regularization:
- **No KL in reward**: Unlike GRPO's `KLRewardTransform`, Expert Iteration doesn't add KL penalties to the reward signal
- **KL in loss function**: KL divergence is computed directly in the loss function using `SFTLoss` with `kl_to_ref_coeff`
- **Reference log probabilities**: The `RetrieveLogProb` transform extracts reference model log probabilities for KL computation

```python
# KL is handled in the loss function, not in the reward
loss_fn = SFTLoss(
    actor_network=policy_training,
    kl_to_ref_coeff=cfg.train.kl_to_ref_coeff,  # KL penalty coefficient
    tokenizer=train_tokenizer,
    tokenizer_kwargs={"chat_template_name": "qwen"},
    device=train_device,
)
```

### 3. Reduced Weight Updates

Expert Iteration can afford fewer policy weight updates due to its selective training approach. One can freely choose longer intervals for the `update_weight_frequency` (e.g., every 100 or more optimization steps).

## Installation

1. Install dependencies:
```bash
# GSM8K deps
pip install -r sota-implementations/expert-iteration/requirements_gsm8k.txt
# IFEval deps
pip install -r sota-implementations/expert-iteration/requirements_ifeval.txt
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

- Automatically handles device allocation;
- Works correctly in both sync and async modes;
- Prevents device conflicts between model components;
- Is more portable across different machine configurations.

## Configuration

The training configuration is managed through Hydra. There are two main configuration files:
- `config/ei_gsm8k.yaml`: Default configuration for GSM8K tasks (default)
- `config/ei_ifeval.yaml`: Configuration optimized for IFEval tasks

## Usage

### Basic Training

There are two training modes available:

#### Synchronous Mode (Default)
```bash
VLLM_USE_V1=0 python sota-implementations/expert-iteration/expert-iteration-sync.py mode=sync train_model.num_devices=2 ref_model.num_devices=2 inference_model.num_devices=2
```

#### Asynchronous Mode (Recommended)
```bash
VLLM_USE_V1=0 python sota-implementations/expert-iteration/expert-iteration-async.py mode=async train_model.num_devices=2 ref_model.num_devices=2 inference_model.num_devices=2
```

The key difference between sync and async modes is how data collection and optimization are handled:

**Synchronous Mode (expert-iteration-sync.py)**:
```python
# Three nested loops:
for data in collector:  # Data collection loop
    for epoch in range(epochs):  # Epoch loop
        for batch in replay_buffer:  # Buffer consumption loop
            # Optimize on batch (only top-k responses)
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()
    # Weight update
    weight_updater.push_weights(policy_training)
```

**Asynchronous Mode (expert-iteration-async.py)**:
```python
# Start data collection in background
collector.start()

# Single optimization loop
for step in range(total_steps):
    # Sample and optimize (only top-k responses)
    batch = replay_buffer.sample()
    loss = loss_fn(batch)
    loss.backward()
    optimizer.step()
    # Update weights once in a while
    if step % weight_update_frequency == 0:
        weight_updater.push_weights(policy_training)
```

Key differences:
1. **Data Collection**: 
   - Sync: Data collection and optimization happen sequentially (unless `train.sync_iter=false`)
   - Async: Data collection runs in background while optimization happens

2. **Buffer Size**:
   - Sync: Buffer size must equal the batch size returned by collector
   - Async: Buffer can be larger than the batch size, allowing for more diverse sampling

3. **Data Processing**:
   - Sync: Processes the same data multiple times (epochs)
   - Async: Each piece of data is processed a non-deterministic number of times

4. **Weight updates**:
   - Sync: Weights are updated before every collection of data
   - Async: Weights are updated at a given interval (in gradient steps)

The async mode offers better performance by:

- Running data collection and optimization concurrently
- More efficient GPU utilization
- Reduced memory overhead
- Better throughput
- More flexible buffer management

### Top-K Configuration

The key parameters for top-k selection are:

```yaml
env:
  repeats: 32  # Number of responses to generate per prompt
train:
  topk_size: 4  # Number of best responses to keep for training
```

**Recommendations**:

- Higher `repeats` values provide more diversity but increase computation
- `topk_size` should be 10-20% of `repeats` for good selection pressure
- Typical values: `repeats=32, topk_size=4` or `repeats=64, topk_size=8`

It is critical to have a reward function that is granular enough for `top-k` to be of any use: a binary reward will have a median value
will not provide much insight into what outputs outrank others. 

### KL Regularization

KL divergence is controlled via the `kl_to_ref_coeff` parameter:

```yaml
train:
  kl_to_ref_coeff: 1.0  # KL penalty coefficient
```

**Recommendations**:

- Start with `kl_to_ref_coeff=1.0` and adjust based on model quality.
- Higher values keep the model closer to the reference.
- Lower values allow more exploration but risk quality degradation.
  **Note**: Expert iteration is a rather simple algorithm with little convergence guarantees. Using high KL regularization coefficient and setting it to lower values progressively is advisable.

### Run with IFEval Config

```bash
python expert-iteration-sync.py mode=sync --config-name ei_ifeval
```

### Override Config Values

```bash
# Change dataset
python expert-iteration-sync.py mode=sync env.dataset=ifeval

# Modify top-k parameters
python expert-iteration-sync.py mode=sync env.repeats=64 train.topk_size=8

# Adjust KL regularization
python expert-iteration-sync.py mode=sync train.kl_to_ref_coeff=0.5

# Change model
python expert-iteration-sync.py mode=sync model.name=meta-llama/Llama-2-7b-hf
```

### Hyperparameter Sweeps

```bash
# Top-k size sweep
python expert-iteration-sync.py mode=sync --multirun train.topk_size=2,4,8

# KL coefficient sweep
python expert-iteration-sync.py mode=sync --multirun train.kl_to_ref_coeff=0.5,1.0,2.0

# Multiple parameters
python expert-iteration-sync.py mode=sync --multirun \
  train.topk_size=4,8 \
  train.kl_to_ref_coeff=0.5,1.0
```

Don't forget to set the number of value of `train.total_dialog_turns` to a reasonable value!

## Monitoring

Training progress is logged to Weights & Biases with the following metrics:

- **Reward**: Average reward of responses in the buffer
- **Sequence length**: Average length of generated responses
- **KL divergence**: KL divergence from reference model
- **Loss metrics**: SFT loss, KL loss, and total loss
- **Gradient norm**: Gradient clipping statistics
- **Throughput metrics**: Steps per second, gradient steps per write
- **Buffer statistics**: Write count, policy version tracking

### Collector Logging

The collector is given a `RemoteDataLogger` postproc hook that passes the data to a Ray queue, consumed by the training node for logging.

This approach ensures:
- Single wandb run with all metrics (training + collector)
- No conflicts between multiple wandb loggers
- Centralized logging through the main process

The collector logs the following metrics:
- **Collector rewards**: Mean, std, min, max of rewards from collected data
- **Response lengths**: Mean, std, min, max of response lengths
- **Policy versions**: Mean, min, max of policy versions (for async mode)
- **Time elapsed**: Time between collection batches

To add new collector metrics, modify the `log_data` method in `RemoteDataLogger` in `ei_utils.py`.

## Checkpointing

Checkpoints are saved every `train.checkpoint_frequency` steps and contain:
- Model state
- Optimizer state
- Gradient scaler state (for mixed precision)
- Full configuration

## Debugging Out-of-memory issues

- **vLLM**: Reduce `inference_model.gpu_memory_utilization=FRACTION` or number of environments run in parallel (`env.num_envs=N`)
- **Reference model**: If the reference model computation is memory-intensive, reduce the number of environments (`env.num_envs=N`) run in parallel
- **Training**: Reduce batch size (`train.optim_batch_size`)
- **Top-k**: Reduce `env.repeats` to generate fewer responses per prompt

## Directory Structure

```
sota-implementations/expert-iteration/
├── config/
│   ├── ei_gsm8k.yaml       # Main configuration file
│   ├── ei_ifeval.yaml      # Configuration for IFEval task
│   └── mode/
│       ├── async.yaml      # Async mode settings
│       └── sync.yaml       # Sync mode settings
├── expert-iteration-sync.py       # Synchronous training script
├── expert-iteration-async.py      # Asynchronous training script
├── ei_utils.py            # Utility functions
└── README.md              # This file
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

## Theoretical Background

Expert Iteration is based on the principle of learning from the best examples rather than all examples. The key insights are:

1. **Selective Learning**: By only training on high-quality responses, the model learns more efficiently
2. **Quality over Quantity**: A smaller dataset of high-quality examples can be more effective than a larger dataset of mixed quality
3. **Iterative Improvement**: Each iteration produces better responses, which become the training data for the next iteration

This approach is particularly effective for language model training where:

- Response quality varies significantly
- High-quality responses are rare but valuable
- The model can learn to imitate good responses more effectively than avoid bad ones

In theory, one could use Exp. It. with samples gathered from other LLMs or expert datasets, although convergence will be harder to control due to
the inability to use the KL regularization factor.

## Comparison with Other Methods

| Method | Training Data | KL Handling | Update Frequency |
|--------|---------------|-------------|------------------|
| **Expert Iteration** | Top-k responses | In loss function | Reduced (can be less frequent) |
| **GRPO** | All responses | In reward / loss | Standard |
| **DPO** | Preference pairs | Implicit in loss | Standard |

Expert Iteration's key advantage is its sample efficiency - by focusing on the best responses, it can achieve better performance with fewer training examples and less frequent policy updates. 
