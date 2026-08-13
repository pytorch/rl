"""
TorchRL Component Contracts for External Loops (WS1)
=====================================================

This tutorial documents the **stable component boundaries** introduced by
Workstream 1 of `RFC #3948 <https://github.com/pytorch/rl/issues/3948>`_.

If you are building a custom training loop (TRL, NeMo-RL, or in-house) and
want to consume individual TorchRL components, this guide explains exactly
what interface each component exposes and how to verify your own objects
satisfy those contracts.

.. note::

    **WS1** deliverable.  See also:

    * `WS2 — TRL Interoperability <trl_interop.html>`_
    * `WS4 — Post-Training Observability <PostTrainingLogger>`

Why stable boundaries?
-----------------------
TorchRL's components (replay buffers, collectors, loss modules) work together
through implicit TensorDict key contracts.  Without a formal definition, it is
hard to know:

* Which keys a ``ReplayBuffer`` sample must carry.
* What a loss module writes into its output.
* How a collector exposes policy-weight synchronisation.

The :mod:`torchrl.data.llm.contracts` module formalises these expectations as
Python ``typing.Protocol`` classes with full docstrings, so adapters stay thin
and don't break silently across TorchRL version bumps.
"""

# %%
# Setup
# -----

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.llm.contracts import (
    PostTrainingBufferProtocol,
    PostTrainingCollectorProtocol,
    PostTrainingLossOutputProtocol,
    assert_satisfies_protocol,
)

# %%
# 1. Buffer contract
# ------------------
# Any replay buffer used in a post-training loop must expose:
#
# * ``extend(data)`` — write a batch of trajectories.
# * ``sample(batch_size)`` — draw a batch.
# * ``write_count`` property — total samples written.
#
# TorchRL's built-in ``ReplayBuffer`` already satisfies this:

rb = ReplayBuffer(storage=LazyTensorStorage(1_000), batch_size=8)
assert isinstance(rb, PostTrainingBufferProtocol)
print("ReplayBuffer satisfies PostTrainingBufferProtocol:", True)

# %%
# You can also verify your own buffer with ``assert_satisfies_protocol``:


class MyCustomBuffer:
    """A minimal external buffer that satisfies the protocol."""

    def __init__(self):
        self._data = []
        self._write_count = 0

    def extend(self, data):
        self._data.append(data)
        self._write_count += len(data)

    def sample(self, batch_size=None):
        return TensorDict({"x": torch.zeros(batch_size or 4)}, [batch_size or 4])

    @property
    def write_count(self) -> int:
        return self._write_count


buf = MyCustomBuffer()
assert_satisfies_protocol(buf, PostTrainingBufferProtocol, name="my_buffer")
print("MyCustomBuffer satisfies PostTrainingBufferProtocol:", True)

# %%
# If your buffer is missing a required field, ``assert_satisfies_protocol``
# raises a ``TypeError`` with a helpful message:


class IncompleteBuffer:
    def extend(self, data):
        pass

    def sample(self, batch_size=None):
        return TensorDict({}, [])

    # Missing: write_count property


try:
    assert_satisfies_protocol(IncompleteBuffer(), PostTrainingBufferProtocol)
except TypeError as e:
    print(f"Caught expected TypeError: {e}")

# %%
# 2. Collector contract
# ---------------------
# Any collector must expose:
#
# * ``update_policy_weights_(...)`` — push updated weights to inference workers.
# * ``__iter__`` / ``__next__`` — yield rollout batches as ``TensorDictBase``.
#
# Here is a minimal external collector that satisfies the contract:


class MyCollector:
    def update_policy_weights_(self, policy_weights=None, *, worker_ids=None):
        pass  # No-op for demonstration

    def __iter__(self):
        # In practice, this would drive rollout generation
        return iter([TensorDict({"tokens": {"full": torch.zeros(4, 16, dtype=torch.long)}}, [])])

    def __next__(self):
        return next(iter(self))


col = MyCollector()
assert_satisfies_protocol(col, PostTrainingCollectorProtocol, name="my_collector")
print("MyCollector satisfies PostTrainingCollectorProtocol:", True)

# %%
# 3. Loss output contract
# -----------------------
# A loss output object must expose at least one of the primary loss fields:
#
# * ``loss_objective`` (GRPO / PPO)
# * ``loss_sft`` (SFT / Expert Iteration)
#
# Additional optional fields are documented in
# :class:`~torchrl.data.llm.contracts.PostTrainingLossOutputProtocol`.
#
# .. note::
#
#     ``GRPOLossOutput`` and ``SFTLossOutput`` are ``TensorClass`` subclasses,
#     so standard ``isinstance`` checks do not work.  Always use
#     ``assert_satisfies_protocol`` for loss outputs.

from torchrl.objectives.llm.sft import SFTLossOutput

sft_out = SFTLossOutput(loss_sft=torch.tensor(0.42))
assert_satisfies_protocol(sft_out, PostTrainingLossOutputProtocol, name="sft_out")
print("SFTLossOutput satisfies PostTrainingLossOutputProtocol:", True)

# %%
# 4. TensorDict key reference
# ----------------------------
# The table below summarises the TensorDict keys that cross component
# boundaries.  These are the same keys ``PostTrainingLogger`` reads.
#
# .. list-table:: Key contract
#    :header-rows: 1
#    :widths: 30 20 50
#
#    * - Key
#      - Source
#      - Description
#    * - ``("next", "reward")``
#      - Collector / Buffer
#      - Per-token or per-sequence reward. Shape ``[B, ...]``.
#    * - ``("tokens", "full")``
#      - Collector / Buffer
#      - Full token sequence (prompt + response). Shape ``[B, T]``.
#    * - ``("tokens", "response")``
#      - Collector / Buffer
#      - Response-only tokens. Shape ``[B, R]`` or ragged list.
#    * - ``("tokens", "prompt")``
#      - Collector
#      - Prompt-only tokens. Shape ``[B, P]``.
#    * - ``("masks", "all_attention_mask")``
#      - Collector
#      - Boolean attention mask. Same shape as ``("tokens", "full")``.
#    * - ``advantage``
#      - Loss (GRPO input)
#      - Per-token advantage estimates.
#    * - ``("log_probs", "full")``
#      - Loss (GRPO input)
#      - Log probabilities from the current policy.
#

print("Key reference table: see docstring above.")

# %%
# 5. Asserting contracts at training loop startup
# ------------------------------------------------
# The recommended pattern is to call ``assert_satisfies_protocol`` once at the
# beginning of your training loop, before the hot path:


def my_post_training_loop(buffer, collector, loss_fn_output):
    """Example external training loop using TorchRL components."""
    # Gate all three components up front — catches integration bugs early
    assert_satisfies_protocol(buffer, PostTrainingBufferProtocol, name="buffer")
    assert_satisfies_protocol(collector, PostTrainingCollectorProtocol, name="collector")
    assert_satisfies_protocol(
        loss_fn_output, PostTrainingLossOutputProtocol, name="loss_output"
    )
    print("All component contracts verified — proceeding with training.")
    # ... rest of your training loop ...


my_post_training_loop(rb, col, sft_out)
