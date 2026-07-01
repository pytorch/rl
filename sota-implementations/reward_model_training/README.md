# RLHF reward-model training

Trains a scalar **reward model** from pairwise human-preference data using the
Bradley-Terry objective
[`RewardModelLoss`](../../torchrl/objectives/llm/reward.py). Given a prompt and two
responses -- a `chosen` one preferred by an annotator and a `rejected` one -- the model
learns to assign a higher score to the chosen response:

```
loss = -log σ(r_θ(x, y_chosen) - r_θ(x, y_rejected))
```

This is the reward-modelling stage that precedes policy optimization (e.g.
[GRPO](../grpo/)) in RLHF pipelines.

This recipe is a minimal **single-node reference implementation**, not a large-scale
training stack. See [Scaling and integration](#scaling-and-integration) for how it is
meant to fit into serious training.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Train on the default summarization-comparisons dataset with a small Qwen backbone:

```bash
python reward_model.py model.name=Qwen/Qwen2.5-0.5B
```

The backbone is **model-agnostic**: `model.name` accepts any Hugging Face model id
usable with `AutoModelForSequenceClassification` (a single-output head is attached
automatically), e.g. `facebook/opt-125m` or a larger instruct model.

Use a different preference dataset (must expose `chosen`/`rejected`, and optionally
`prompt` fields):

```bash
python reward_model.py model.name=Qwen/Qwen2.5-0.5B data.dataset_name=Anthropic/hh-rlhf \
  data.split_train=train data.split_val=test
```

Add the score-centering regularizer or freeze part of the backbone:

```bash
python reward_model.py loss.center_coeff=0.01 optim.freeze_frac=0.7
```

## Configuration

Key fields in [`config.yaml`](config.yaml):

| Group | Field | Description |
|-------|-------|-------------|
| `model` | `name` | HF backbone id; empty -> tiny from-scratch model (CI). |
| `model` | `device` | Training device; `null` auto-selects CUDA/MPS/CPU. |
| `data` | `dataset_name` | HF preference dataset; empty -> synthetic data (CI). |
| `data` | `max_length`, `batch_size`, `max_samples` | Tokenization / sampling. |
| `optim` | `max_iters`, `lr`, `freeze_frac`, `clip_grad` | Optimization. |
| `loss` | `reduction`, `center_coeff` | Bradley-Terry loss options. |
| `logger` | `backend`, `eval_iter`, `eval_iters` | Logging / evaluation. |

## Hermetic smoke run

Leaving `model.name` and `data.dataset_name` empty builds a tiny from-scratch model and
a synthetic preference dataset, so the recipe runs end-to-end with no download and
without the `datasets` dependency (this is what CI exercises):

```bash
python reward_model.py model.name= data.dataset_name= \
  optim.max_iters=3 logger.eval_iter=2 logger.eval_iters=1 logger.backend=
```

## Scaling and integration

This recipe is a minimal single-node baseline: a readable, single-process training loop
that exercises the loss and data format end to end. It is **not** a large-scale
reward-model training stack, and is not meant to become one.

TorchRL owns the **contract**, not the parallelism. What this recipe provides:

- a canonical TensorDict **preference-data format** (prompt / chosen / rejected),
- a canonical **reward-model loss**
  ([`RewardModelLoss`](../../torchrl/objectives/llm/reward.py), Bradley-Terry),
- a small **reference trainer** (this recipe),
- and, in the future, **adapters/scorers** so an externally trained reward model plugs
  back into TorchRL GRPO/PPO/SFT pipelines.

For serious distributed reward-model training (parallelism, sharded checkpointing,
fault tolerance, packed/streamed datasets, mixed precision, activation checkpointing,
multi-node orchestration), use a dedicated backend such as TRL/Accelerate/FSDP,
TorchTitan/FSDP2, Megatron/NeMo, or OpenRLHF, while **preserving the TorchRL data and
scoring contract** so the resulting model plugs straight back into TorchRL's RLHF
pipelines. Interfacing with those backends, rather than reimplementing them, is the
intended direction for scaling this and the rest of the TorchRL LLM stack.
