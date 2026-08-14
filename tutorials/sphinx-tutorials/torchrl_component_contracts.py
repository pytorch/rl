"""
TorchRL Component Contracts for External Loops (WS1)
=====================================================

Documents the stable component boundaries from RFC #3948 WS1.
Shows how to verify that custom buffers, collectors, and loss outputs
satisfy TorchRL's interfaces using :func:`~torchrl.data.llm.contracts.assert_satisfies_protocol`.
"""

# %%
# Setup
# -----

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.llm.contracts import (
    GRPOLossOutputProtocol,
    PostTrainingBufferProtocol,
    PostTrainingCollectorProtocol,
    SFTLossOutputProtocol,
    assert_satisfies_protocol,
)
from torchrl.objectives.llm.sft import SFTLossOutput

# %%
# Buffer contract
# ---------------
# Buffers must expose ``extend``, ``sample``, and ``write_count``.
# TorchRL's built-in ``ReplayBuffer`` already satisfies this:

rb = ReplayBuffer(storage=LazyTensorStorage(1_000), batch_size=8)
assert isinstance(rb, PostTrainingBufferProtocol)

# %%
# Custom buffers can be verified with ``assert_satisfies_protocol``:


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
assert_satisfies_protocol(buf, PostTrainingBufferProtocol, name="my_buffer")

# %%
# Missing a required field raises ``TypeError`` with a helpful message:


class IncompleteBuffer:
    def extend(self, data):
        pass

    def sample(self, batch_size=None):
        return TensorDict({}, [])


try:
    assert_satisfies_protocol(IncompleteBuffer(), PostTrainingBufferProtocol)
except TypeError as e:
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
assert_satisfies_protocol(col, PostTrainingCollectorProtocol, name="my_collector")

# %%
# Loss output contracts
# ---------------------
# TorchRL provides two separate loss-output protocols, one per loss type:
#
# * :class:`~torchrl.data.llm.contracts.GRPOLossOutputProtocol` — requires ``loss_objective``.
# * :class:`~torchrl.data.llm.contracts.SFTLossOutputProtocol` — requires ``loss_sft``.
#
# Because ``GRPOLossOutput`` and ``SFTLossOutput`` are ``TensorClass`` subclasses,
# use ``assert_satisfies_protocol`` (not ``isinstance``):

sft_out = SFTLossOutput(loss_sft=torch.tensor(0.42))
assert_satisfies_protocol(sft_out, SFTLossOutputProtocol, name="sft_out")

# %%
# Asserting all contracts at startup
# ------------------------------------
# Recommended pattern: call once at the top of your training loop.


def my_training_loop(buffer, collector, loss_output):
    assert_satisfies_protocol(buffer, PostTrainingBufferProtocol, name="buffer")
    assert_satisfies_protocol(collector, PostTrainingCollectorProtocol, name="collector")
    assert_satisfies_protocol(loss_output, SFTLossOutputProtocol, name="loss_output")


my_training_loop(rb, col, sft_out)
