"""
TorchRL Component Contracts for External Loops
==============================================

Documents the stable component boundaries.
Shows how to verify that custom buffers, collectors, and loss outputs
satisfy TorchRL's interfaces using the provided helper functions.
"""

# %%
# Setup
# -----

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.llm.contracts import (
    assert_buffer_contract,
    assert_collector_contract,
    assert_loss_contract,
)
from torchrl.objectives.llm.sft import SFTLossOutput

# %%
# Buffer contract
# ---------------
# Buffers must expose ``extend``, ``sample``, and ``write_count``.
# TorchRL's built-in ``ReplayBuffer`` already satisfies this:

rb = ReplayBuffer(storage=LazyTensorStorage(1_000), batch_size=8)
assert_buffer_contract(rb)

# %%
# Custom buffers can also be verified with the helper:


class MyCustomBuffer:
    def __init__(self):
        self._write_count = 0

    def extend(self, data):
        self._write_count += len(data)

    def sample(self, batch_size=None):
        return TensorDict({"x": torch.zeros(batch_size or 4)}, [batch_size or 4])

    @property
    def write_count(self) -> int:
        return self._write_count


buf = MyCustomBuffer()
assert_buffer_contract(buf)

# %%
# Missing a required field raises ``TypeError`` with a helpful message:


class IncompleteBuffer:
    def extend(self, data):
        pass

    def sample(self, batch_size=None):
        return TensorDict({}, [])


try:
    assert_buffer_contract(IncompleteBuffer())
except TypeError:
    pass  # expected — write_count is missing

# %%
# Collector contract
# ------------------
# Collectors must expose ``update_policy_weights_``, ``__iter__``, and ``__next__``.


class MyCollector:
    def update_policy_weights_(self, policy_weights=None, *, worker_ids=None):
        pass

    def __iter__(self):
        return iter([TensorDict({}, [])])

    def __next__(self):
        return next(iter(self))


col = MyCollector()
assert_collector_contract(col)

# %%
# Loss output contracts
# ---------------------
# TorchRL provides explicit field validation for two separate loss types:
#
# * GRPO — requires ``loss_objective``.
# * SFT — requires ``loss_sft``.
#
# Because ``GRPOLossOutput`` and ``SFTLossOutput`` are ``TensorClass`` subclasses,
# use ``assert_loss_contract`` (not ``isinstance``):

sft_out = SFTLossOutput(loss_sft=torch.tensor(0.42))
assert_loss_contract(sft_out, loss_type="sft")

# %%
# Asserting all contracts at startup
# ------------------------------------
# Recommended pattern: call once at the top of your training loop.


def my_training_loop(buffer, collector, loss_output):
    assert_buffer_contract(buffer)
    assert_collector_contract(collector)
    assert_loss_contract(loss_output, loss_type="sft")


my_training_loop(rb, col, sft_out)
