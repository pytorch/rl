"""
.. _coding_grpo_trl:

Training a Reward Model with TorchRL and HuggingFace TRL
=========================================================

**Author**: `Jay Prajapati <https://github.com/coder-jayp>`_

This tutorial demonstrates an end-to-end workflow for training a reward model
using TorchRL and Hugging Face ``trl``.  It showcases two adapter classes that
make the two libraries interoperable:

* :class:`~torchrl.modules.llm.TorchRLBufferDataset` — exposes any TorchRL
  :class:`~torchrl.data.ReplayBuffer` as a Hugging Face
  ``datasets.IterableDataset``, the format expected by ``trl`` trainers.
* :class:`~torchrl.modules.llm.HFRewardModelWrapper` — wraps a trained HF
  reward model so it can be used inside any TorchRL training loop or rollout.

What you will learn
-------------------

* How to store real human-preference data inside a TorchRL
  :class:`~torchrl.data.ReplayBuffer`.
* How :class:`~torchrl.modules.llm.TorchRLBufferDataset` acts as a bridge
  that lets ``trl.RewardTrainer`` consume that buffer with zero custom
  data-loading code.
* How to use :class:`~torchrl.modules.llm.HFRewardModelWrapper` to score new
  responses with the trained reward model inside a TorchRL workflow.
* How to plot a real training loss curve directly from the TRL trainer logs.

Why this matters
----------------

In a full GRPO / RLHF pipeline, an :class:`~torchrl.collectors.llm.LLMCollector`
continuously generates (prompt, response) pairs that flow into a
:class:`~torchrl.data.ReplayBuffer`.  That buffer must then feed a ``trl``
trainer for reward-model updates.  The two adapters eliminate the glue code
that would otherwise be required at this boundary.
"""

# %%
# Setup and Imports
# -----------------
# We install the optional dependencies if they are not already present.
#
# .. code-block:: bash
#
#    pip install trl transformers datasets matplotlib
#

import os
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; Sphinx-Gallery captures the figure
import matplotlib.pyplot as plt
import torch
from tensordict import set_list_to_stack, TensorDict
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.modules.llm import HFRewardModelWrapper, TorchRLBufferDataset

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

    _has_trl = True
except ImportError:
    _has_trl = False

# %%
# 1. Loading a Real Preference Dataset into a TorchRL ReplayBuffer
# ----------------------------------------------------------------
#
# We use ``trl-lib/ultrafeedback_binarized``, a standard human-preference
# dataset widely used for reward-model training.  Each example contains a
# ``chosen`` response (preferred by annotators) and a ``rejected`` response
# stored as conversation-history dicts (the standard HF chat format).
#
# We load the tokenizer first so we can call
# ``tokenizer.apply_chat_template()`` to convert those dicts to plain
# strings *before* writing to the buffer.  Plain strings are what
# ``trl.RewardTrainer`` expects, and they travel through the buffer without
# any serialisation overhead.
#
# In production the buffer would be filled online by a
# :class:`~torchrl.collectors.llm.LLMCollector`; the chat-template step
# would happen inside the collector's data-processing transforms.

N_SAMPLES = 128  # small enough for CI, large enough to show a real trend
BATCH_SIZE = 8
MODEL_NAME = "distilbert-base-uncased"

if _has_trl:
    # Enable TensorDict to store Python lists (strings) transparently.
    # Scoped here so it only applies when trl is available.
    set_list_to_stack(True).set()

    # Load tokenizer first so we can flatten conversation dicts → strings.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # DistilBERT / BERT use [SEP] as the sequence-boundary token.
    # TRL's RewardTrainer appends eos_token to each example, so we map it.
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token

    raw = load_dataset(
        "trl-lib/ultrafeedback_binarized",
        split=f"train[:{N_SAMPLES}]",
    )

    rb = ReplayBuffer(storage=ListStorage(max_size=N_SAMPLES))

    # Extract the most recent assistant turn from each conversation.
    # Reward models score *responses*, so we strip away the system/user
    # context and pass just the assistant's text.  In a real pipeline, the
    # :class:`~torchrl.collectors.llm.LLMCollector` produces the response
    # text directly, so no extra extraction is needed there.
    def get_assistant_response(messages: list) -> str:
        return next(
            m["content"] for m in reversed(messages) if m["role"] == "assistant"
        )

    for row in raw:
        chosen_str = get_assistant_response(row["chosen"])
        rejected_str = get_assistant_response(row["rejected"])

        # Store as a plain string alongside any TorchRL-specific metadata
        # (log-probs, token tensors, …) without impacting the TRL trainer.
        rb.add(
            TensorDict(
                {"chosen": chosen_str, "rejected": rejected_str},
                batch_size=[],
            )
        )

    print(f"ReplayBuffer populated: {len(rb)} preference pairs.")
    print(f"  Sample chosen  : {rb[0]['chosen'][:120]}...")
    print(f"  Sample rejected: {rb[0]['rejected'][:120]}...")

# %%
# 2. Bridging TorchRL → TRL with TorchRLBufferDataset
# ----------------------------------------------------
#
# :class:`~torchrl.modules.llm.TorchRLBufferDataset` samples from the buffer
# and yields individual ``dict[str, Any]`` objects — exactly the format
# ``trl.RewardTrainer`` expects.  Calling :meth:`as_hf_dataset` wraps the
# result in a Hugging Face ``datasets.IterableDataset`` without copying data.
#
# ``num_batches=None`` creates an unbounded stream; the trainer's ``max_steps``
# imposes the training budget.

if _has_trl:
    trl_dataset = TorchRLBufferDataset(
        rb,
        batch_size=BATCH_SIZE,
        keys=["chosen", "rejected"],
        num_batches=None,  # online / unbounded stream
    ).as_hf_dataset()

    # Quick sanity check: one yielded item should be a single example dict
    first_item = next(iter(trl_dataset))
    assert "chosen" in first_item and "rejected" in first_item
    print("\nTorchRLBufferDataset ready. First item keys:", list(first_item.keys()))

# %%
# 3. Training the Reward Model
# ----------------------------
#
# We use ``distilbert-base-uncased`` (66 M parameters) as the base model — small
# enough to train on CPU in about a minute while still producing a meaningful
# learning signal on real preference data.  The classification head
# (``pre_classifier`` + ``classifier``) is initialised from scratch; this is
# expected and correct — the MISSING entries in the load report confirm it.

MAX_STEPS = 20  # increase to 200+ for a production-quality reward model

if _has_trl:
    # Tokenizer was already loaded above; reuse it here.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = RewardConfig(
        output_dir=tempfile.mkdtemp(prefix="torchrl_reward_model_"),
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
        logging_steps=1,
        learning_rate=2e-5,
        report_to="none",
        remove_unused_columns=False,
        max_length=256,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=trl_dataset,
        processing_class=tokenizer,
    )

    print(f"\nTraining reward model ({MODEL_NAME}) for {MAX_STEPS} steps …")
    trainer.train()
    print("Training complete!")

# %%
# 4. Plotting the Learning Curve
# ------------------------------
#
# TRL logs ``loss``, ``grad_norm``, and reward metrics at every step.  We
# extract them from ``trainer.state.log_history`` and plot the training loss.
# Sphinx-Gallery automatically captures ``plt.show()`` calls and embeds the
# figure in the documentation page.

if _has_trl:
    history = [log for log in trainer.state.log_history if "loss" in log]
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    rewards_accuracy = [h.get("accuracy", None) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Loss curve ---
    axes[0].plot(steps, losses, marker="o", linewidth=2, color="#4C72B0")
    axes[0].set_title("Reward Model Training Loss", fontsize=13)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Reward accuracy curve (margin: how often chosen > rejected) ---
    if any(v is not None for v in rewards_accuracy):
        acc = [v for v in rewards_accuracy if v is not None]
        axes[1].plot(steps[: len(acc)], acc, marker="s", linewidth=2, color="#55A868")
        axes[1].axhline(0.5, color="gray", linestyle="--", label="random baseline")
        axes[1].set_title("Reward Accuracy (chosen > rejected)", fontsize=13)
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# %%
# 5. Using the Trained Reward Model Back in TorchRL
# -------------------------------------------------
#
# Once training is complete, wrap the reward model with
# :class:`~torchrl.modules.llm.HFRewardModelWrapper` so it integrates
# natively with any TorchRL rollout.  The wrapper maps
# ``("tokens", "full")`` / ``("masks", "all_attention_mask")`` → ``"reward"``
# by default, matching the layout produced by
# :class:`~torchrl.modules.llm.TransformersWrapper`.

if _has_trl:
    SEQ_LEN = 32
    # After accelerate training the model may live on GPU; infer its device
    # so the rollout batch can be placed on the same device.
    model_device = next(model.parameters()).device

    reward_fn = HFRewardModelWrapper(
        model,
        token_key="input_ids",
        attention_mask_key="attention_mask",
        reward_key="reward",
        inference_mode=True,
    )

    # Simulate a batch of tokenised responses arriving from an LLMCollector
    B = 4
    rollout_batch = TensorDict(
        {
            "input_ids": torch.randint(
                0, tokenizer.vocab_size, (B, SEQ_LEN), device=model_device
            ),
            "attention_mask": torch.ones(
                B, SEQ_LEN, dtype=torch.long, device=model_device
            ),
        },
        batch_size=[B],
    )

    scored = reward_fn(rollout_batch)
    print(f"\nReward scores for {B} responses: {scored['reward'].tolist()}")
    assert scored["reward"].shape == torch.Size([B])
    assert scored["reward"].dtype == torch.float32

# %%
# Conclusion
# ----------
#
# In this tutorial you have seen how to build a complete reward-model training
# pipeline that bridges TorchRL and Hugging Face ``trl``:
#
# 1. **Buffer as data store** — real human-preference pairs live in a TorchRL
#    :class:`~torchrl.data.ReplayBuffer`, giving you TorchRL's sampling,
#    prioritisation, and device-management features for free.
# 2. **TorchRLBufferDataset** — a one-line bridge that turns the buffer into an
#    unbounded ``datasets.IterableDataset`` consumable by any ``trl`` trainer.
# 3. **Meaningful training signal** — using a real preference dataset
#    (``trl-lib/ultrafeedback_binarized``) produces an actual learning curve
#    rather than a flat loss at ``log(2)``.
# 4. **HFRewardModelWrapper** — closes the loop by making the trained reward
#    model a first-class TorchRL module that can score rollouts from an
#    :class:`~torchrl.collectors.llm.LLMCollector`.
#
# Further reading
# ---------------
#
# * :ref:`trl_interop_tutorial` — API deep-dive for both adapter classes.
# * :ref:`trl_interop_section` — Reference documentation.
# * :class:`~torchrl.modules.llm.TorchRLBufferDataset` — API reference.
# * :class:`~torchrl.modules.llm.HFRewardModelWrapper` — API reference.
# * :class:`~torchrl.collectors.llm.LLMCollector` — recommended way to
#   generate rollout data for a TorchRL replay buffer.
# * :class:`~torchrl.objectives.llm.GRPOLoss` — GRPO training objective that
#   pairs naturally with :class:`~torchrl.modules.llm.HFRewardModelWrapper`.
