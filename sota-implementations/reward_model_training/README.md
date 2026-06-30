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

## Install

```bash
pip install -r requirements.txt
```

## Usage

Train on the default summarization-comparisons dataset with a GPT2 backbone:

```bash
python reward_model.py model.name=gpt2
```

The backbone is **model-agnostic**: `model.name` accepts any Hugging Face model id
usable with `AutoModelForSequenceClassification` (a single-output head is attached
automatically), e.g. `facebook/opt-125m` or a larger instruct model.

Use a different preference dataset (must expose `chosen`/`rejected`, and optionally
`prompt` fields):

```bash
python reward_model.py model.name=gpt2 data.dataset_name=Anthropic/hh-rlhf \
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
