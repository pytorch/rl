Note

Go to the end
to download the full example code.

# TRL Interoperability: Using TorchRL Buffers and HF Reward Models Together

**Author**: TorchRL contributors

This tutorial demonstrates how to bridge TorchRL and Hugging Face `trl`
using the adapter classes documented in [TRL Interoperability](../reference/llms.html#trl-interop-section):

- [`TorchRLBufferDataset`](../reference/generated/torchrl.modules.llm.TorchRLBufferDataset.html#torchrl.modules.llm.TorchRLBufferDataset): expose replay samples as
PyTorch or Hugging Face iterable datasets.
- [`HFRewardModelWrapper`](../reference/generated/torchrl.modules.llm.HFRewardModelWrapper.html#torchrl.modules.llm.HFRewardModelWrapper): consume a Hugging Face
reward model inside any TorchRL training loop.

## What you will learn

- How to wrap a TorchRL replay buffer as a `torch.utils.data.IterableDataset`
and bridge it to the Hugging Face `datasets` interface required by `trl`.
- How to key-filter the yielded samples and move tensors to a target device.
- How to wrap an HF reward model as a `TensorDictModuleBase` and plug it
directly into a TorchRL GRPO / PPO rollout.
- How to chain both adapters in a complete round-trip test.

## Imports and setup

Neither `trl` nor `transformers` is required to *import* the adapters -
they are loaded lazily on first use. In this tutorial we use toy models so
no real checkpoints need to be downloaded.

```
import torch
from tensordict import TensorDict

from torchrl.data import ListStorage, ReplayBuffer
```

# Part 1 - TorchRL -> TRL: TorchRLBufferDataset

We start with the `TorchRL -> TRL` direction. The goal is to produce a
[`torch.utils.data.IterableDataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset). Current `trl` trainers require
a Hugging Face `datasets.IterableDataset`; the adapter exposes that form
through `as_hf_dataset()` when the optional dependency is installed.

## Step 1: build and populate a replay buffer

In a real GRPO pipeline the buffer would be filled by an
[`LLMCollector`](../reference/generated/torchrl.collectors.llm.LLMCollector.html#torchrl.collectors.llm.LLMCollector). Here we fill it manually
with synthetic token tensors.

```
SEQ_LEN = 16
N_SAMPLES = 40
BATCH_SIZE = 8

rb = ReplayBuffer(storage=ListStorage(max_size=200), batch_size=BATCH_SIZE)

for _ in range(N_SAMPLES):
 rb.add(
 TensorDict(
 {
 "prompt": "Explain one reinforcement-learning concept.",
 "input_ids": torch.randint(0, 1000, (SEQ_LEN,)),
 "attention_mask": torch.ones(SEQ_LEN, dtype=torch.long),
 "labels": torch.randint(0, 2, (SEQ_LEN,)),
 "reward": torch.randn(()),
 },
 batch_size=[],
 )
 )
```

## Step 2: wrap the buffer as an IterableDataset

Each call to `__iter__` draws one batch and yields individual samples as
plain Python dicts - the exact format expected by `DataCollatorWithPadding`
and trl trainers.

```
from torchrl.modules.llm import TorchRLBufferDataset

dataset = TorchRLBufferDataset(rb, batch_size=BATCH_SIZE, num_batches=1)
```

Iterate and inspect

```
samples = list(dataset)
# Each yielded item is a single sample (no batch dimension)
sample = samples[0]
assert "input_ids" in sample
assert sample["input_ids"].shape == torch.Size([SEQ_LEN])
assert len(samples) == BATCH_SIZE
```

`trl.GRPOTrainer` requires a Hugging Face `datasets.IterableDataset`
with a top-level `"prompt"` field. With `datasets` installed, create an
unbounded stream and set a finite `max_steps` on the trainer:

```
trl_dataset = TorchRLBufferDataset(
 rb,
 batch_size=BATCH_SIZE,
 keys=["prompt"],
 num_batches=None,
).as_hf_dataset()
```

## Step 3: key filtering

Pass `keys` to expose only a subset of TensorDict keys. This is useful
when the buffer stores extra bookkeeping tensors you do not want to send to
the model.

```
dataset_filtered = TorchRLBufferDataset(
 rb,
 batch_size=BATCH_SIZE,
 keys=["input_ids", "attention_mask"],
)
filtered_sample = next(iter(dataset_filtered))
assert set(filtered_sample.keys()) == {"input_ids", "attention_mask"}
```

## Step 4: device placement

Pass `device` to move tensors automatically before they are yielded. This
avoids boilerplate `{k: v.to(device) for k, v in sample.items()}` in the
training loop.

```
dataset_cpu = TorchRLBufferDataset(rb, batch_size=BATCH_SIZE, device="cpu")
for s in dataset_cpu:
 for v in s.values():
 if isinstance(v, torch.Tensor):
 assert v.device == torch.device("cpu")
```

## Step 5: nested keys

TorchRL stores many signals under nested keys, e.g. `("tokens", "full")`.
Pass a list of `NestedKey` tuples to `keys`; they are serialised to
dot-separated strings (`"tokens.full"`) so HuggingFace collators
(which require `str` keys) work without modification.

```
rb_nested = ReplayBuffer(storage=ListStorage(max_size=200), batch_size=BATCH_SIZE)
for _ in range(N_SAMPLES):
 rb_nested.add(
 TensorDict(
 {
 "tokens": TensorDict(
 {"full": torch.randint(0, 1000, (SEQ_LEN,))}, batch_size=[]
 ),
 "masks": TensorDict(
 {"all_attention_mask": torch.ones(SEQ_LEN, dtype=torch.long)},
 batch_size=[],
 ),
 },
 batch_size=[],
 )
 )

dataset_nested = TorchRLBufferDataset(
 rb_nested,
 batch_size=BATCH_SIZE,
 keys=[("tokens", "full"), ("masks", "all_attention_mask")],
)
nested_sample = next(iter(dataset_nested))
assert "tokens.full" in nested_sample
assert "masks.all_attention_mask" in nested_sample
```

# Part 2 - TRL -> TorchRL: HFRewardModelWrapper

The second direction: consume an HF reward model inside a TorchRL loop.
[`HFRewardModelWrapper`](../reference/generated/torchrl.modules.llm.HFRewardModelWrapper.html#torchrl.modules.llm.HFRewardModelWrapper) adapts any HF
`AutoModelForSequenceClassification`-style model to the TensorDict API.

```
from torchrl.modules.llm import HFRewardModelWrapper
```

## Step 1: define a stand-in reward model

In production you would load a real checkpoint, for example:

```
from transformers import AutoModelForSequenceClassification
hf_model = AutoModelForSequenceClassification.from_pretrained(
 "my-org/reward-model-v1", num_labels=1
)
```

Here we use a toy that mimics the HF API - returns an object with
`logits` of shape `[B, 1]`.

```
class ToyRewardModel(torch.nn.Linear):
 """A toy reward model mimicking the HF ``AutoModelForSequenceClassification`` API."""

 def __init__(self):
 super().__init__(SEQ_LEN, 1, bias=False)

 def forward(self, input_ids, attention_mask=None):
 # Convert to float and reduce over sequence; return logits [B, 1]
 x = input_ids.float()
 if attention_mask is not None:
 x = x * attention_mask.float()
 logits = super().forward(x) # [B, 1]

 class _Out:
 pass

 out = _Out()
 out.logits = logits
 return out
```

## Step 2: wrap with default TorchRL token keys

The default `token_key=("tokens", "full")` and
`attention_mask_key=("masks", "all_attention_mask")` match the layout
produced by TorchRL's [`TransformersWrapper`](../reference/generated/torchrl.modules.llm.TransformersWrapper.html#torchrl.modules.llm.TransformersWrapper).

```
reward_fn = HFRewardModelWrapper(
 ToyRewardModel(),
 token_key=("tokens", "full"),
 attention_mask_key=("masks", "all_attention_mask"),
 reward_key="reward",
 inference_mode=True, # disable grad for pure reward inference
)

assert ("tokens", "full") in reward_fn.in_keys
assert "reward" in reward_fn.out_keys
```

## Step 3: run a batch through the wrapper

```
B = 4
td = TensorDict(
 {
 "tokens": TensorDict(
 {"full": torch.randint(0, 1000, (B, SEQ_LEN))}, batch_size=[B]
 ),
 "masks": TensorDict(
 {"all_attention_mask": torch.ones(B, SEQ_LEN, dtype=torch.long)},
 batch_size=[B],
 ),
 },
 batch_size=[B],
)

result = reward_fn(td)
assert result["reward"].shape == torch.Size([B])
assert result["reward"].dtype == torch.float32
```

## Step 4: custom keys and nested reward key

You can adapt the wrapper to any key layout by passing custom `token_key`,
`attention_mask_key`, and `reward_key`. Nested reward keys (a tuple)
are fully supported.

```
reward_fn_custom = HFRewardModelWrapper(
 ToyRewardModel(),
 token_key="input_ids",
 attention_mask_key="attention_mask",
 reward_key=("reward", "value"),
 inference_mode=True,
)

td2 = TensorDict(
 {
 "input_ids": torch.randint(0, 1000, (B, SEQ_LEN)),
 "attention_mask": torch.ones(B, SEQ_LEN, dtype=torch.long),
 },
 batch_size=[B],
)
result2 = reward_fn_custom(td2)
assert result2.get(("reward", "value")).shape == torch.Size([B])
```

# Part 3 - Round-trip: buffer -> dataset -> reward wrapper

We now chain both adapters:

1. TorchRL collector fills a buffer (simulated here with synthetic data).
2. [`TorchRLBufferDataset`](../reference/generated/torchrl.modules.llm.TorchRLBufferDataset.html#torchrl.modules.llm.TorchRLBufferDataset) samples from it and
yields dicts of tensors.
3. Those dicts are re-batched into a TensorDict.
4. [`HFRewardModelWrapper`](../reference/generated/torchrl.modules.llm.HFRewardModelWrapper.html#torchrl.modules.llm.HFRewardModelWrapper) scores the batch.

```
# Build buffer
rb_rt = ReplayBuffer(storage=ListStorage(max_size=200), batch_size=8)
for _ in range(20):
 rb_rt.add(
 TensorDict(
 {
 "tokens": TensorDict(
 {"full": torch.randint(0, 1000, (SEQ_LEN,))}, batch_size=[]
 ),
 "masks": TensorDict(
 {"all_attention_mask": torch.ones(SEQ_LEN, dtype=torch.long)},
 batch_size=[],
 ),
 },
 batch_size=[],
 )
 )

# Sample via dataset
ds_rt = TorchRLBufferDataset(
 rb_rt,
 batch_size=8,
 keys=[("tokens", "full"), ("masks", "all_attention_mask")],
)
samples_rt = list(ds_rt) # 8 individual sample dicts

# Re-batch
batch_ids = torch.stack([s["tokens.full"] for s in samples_rt]) # [8, SEQ_LEN]
batch_mask = torch.stack(
 [s["masks.all_attention_mask"] for s in samples_rt]
) # [8, SEQ_LEN]

td_rt = TensorDict(
 {
 "tokens": TensorDict({"full": batch_ids}, batch_size=[8]),
 "masks": TensorDict({"all_attention_mask": batch_mask}, batch_size=[8]),
 },
 batch_size=[8],
)

# Score with reward wrapper
reward_fn_rt = HFRewardModelWrapper(ToyRewardModel(), inference_mode=True)
result_rt = reward_fn_rt(td_rt)
assert result_rt["reward"].shape == torch.Size([8])
```

## Conclusion

In this tutorial you have seen how to:

- Wrap a TorchRL [`ReplayBuffer`](../reference/generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) as a
`torch.utils.data.IterableDataset` consumable by `trl` trainers.
- Filter keys, handle nested `NestedKey` tuples, and
move tensors to a specific device, all without any custom collators.
- Adapt any HuggingFace reward model to the TensorDict API with
[`HFRewardModelWrapper`](../reference/generated/torchrl.modules.llm.HFRewardModelWrapper.html#torchrl.modules.llm.HFRewardModelWrapper), controlling gradient
flow and output key names at construction time.
- Chain the two adapters in a complete round-trip pipeline.

## Further reading

- [TRL Interoperability](../reference/llms.html#trl-interop-section) - Reference documentation for both adapters.
- [`TorchRLBufferDataset`](../reference/generated/torchrl.modules.llm.TorchRLBufferDataset.html#torchrl.modules.llm.TorchRLBufferDataset) - API reference.
- [`HFRewardModelWrapper`](../reference/generated/torchrl.modules.llm.HFRewardModelWrapper.html#torchrl.modules.llm.HFRewardModelWrapper) - API reference.
- [`LLMCollector`](../reference/generated/torchrl.collectors.llm.LLMCollector.html#torchrl.collectors.llm.LLMCollector) - the recommended way to
generate rollout data for a TorchRL replay buffer.
- [`GRPOLoss`](../reference/generated/torchrl.objectives.llm.GRPOLoss.html#torchrl.objectives.llm.GRPOLoss) - GRPO training objective that
pairs naturally with [`HFRewardModelWrapper`](../reference/generated/torchrl.modules.llm.HFRewardModelWrapper.html#torchrl.modules.llm.HFRewardModelWrapper).

**Total running time of the script:** (0 minutes 0.015 seconds)

[`Download Jupyter notebook: trl_interop.ipynb`](../_downloads/fec1854d700353a6eedacea49f9886d1/trl_interop.ipynb)

[`Download Python source code: trl_interop.py`](../_downloads/db36c047ff34975bb8dc922dc3f4dd38/trl_interop.py)

[`Download zipped: trl_interop.zip`](../_downloads/618a20a3dde0d81bfebc7e6c845162ad/trl_interop.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)