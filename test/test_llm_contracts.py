# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for WS1: Stable component boundary Protocols (torchrl/data/llm/contracts.py)."""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.llm.contracts import (
    PostTrainingBufferProtocol,
    PostTrainingCollectorProtocol,
    PostTrainingLossOutputProtocol,
    assert_satisfies_protocol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rb(capacity: int = 100) -> ReplayBuffer:
    return ReplayBuffer(storage=LazyTensorStorage(capacity), batch_size=4)


class _MinimalBuffer:
    """A hand-rolled buffer that satisfies PostTrainingBufferProtocol."""

    def __init__(self):
        self._count = 0

    def extend(self, data):
        self._count += len(data)

    def sample(self, batch_size=None):
        return TensorDict({"x": torch.zeros(batch_size or 4)}, [batch_size or 4])

    @property
    def write_count(self) -> int:
        return self._count


class _BrokenBuffer:
    """Missing write_count — does NOT satisfy the protocol."""

    def extend(self, data):
        pass

    def sample(self, batch_size=None):
        return TensorDict({}, [])


class _MinimalCollector:
    """Satisfies PostTrainingCollectorProtocol without inheriting anything."""

    def update_policy_weights_(self, policy_weights=None, *, worker_ids=None):
        pass

    def __iter__(self):
        return iter([TensorDict({}, [])])

    def __next__(self):
        return TensorDict({}, [])


class _MinimalLossOutput:
    """Satisfies PostTrainingLossOutputProtocol."""

    @property
    def loss_objective(self) -> torch.Tensor | None:
        """Primary policy optimisation loss, or ``None`` if not computed.

        .. note::
            :class:`~torchrl.objectives.llm.SFTLossOutput` exposes
            ``loss_sft`` rather than ``loss_objective``.  Both satisfy this
            protocol because ``loss_objective`` is optional (may be
            ``None``).  :class:`~torchrl.record.loggers.llm.PostTrainingLogger`
            reads all fields via ``getattr`` duck-typing.
        """
        return torch.tensor(0.5)

    @property
    def loss_sft(self) -> torch.Tensor | None:
        """SFT loss term — present on SFTLossOutput/EI outputs."""
        return None


class _BrokenLossOutput:
    """Missing loss_objective — does NOT satisfy the protocol."""
    pass


# ---------------------------------------------------------------------------
# PostTrainingBufferProtocol
# ---------------------------------------------------------------------------


class TestPostTrainingBufferProtocol:
    def test_torchrl_replay_buffer_satisfies(self):
        """Built-in ReplayBuffer must satisfy the protocol."""
        rb = _make_rb()
        assert isinstance(rb, PostTrainingBufferProtocol)

    def test_minimal_custom_buffer_satisfies(self):
        """A hand-rolled buffer with the required attrs satisfies the protocol."""
        buf = _MinimalBuffer()
        assert isinstance(buf, PostTrainingBufferProtocol)

    def test_broken_buffer_does_not_satisfy(self):
        """A buffer missing write_count does NOT satisfy the protocol."""
        broken = _BrokenBuffer()
        assert not isinstance(broken, PostTrainingBufferProtocol)

    def test_assert_satisfies_passes_silently(self):
        """assert_satisfies_protocol must not raise for a compliant object."""
        rb = _make_rb()
        assert_satisfies_protocol(rb, PostTrainingBufferProtocol, name="rb")

    def test_assert_satisfies_raises_for_broken(self):
        """assert_satisfies_protocol must raise TypeError for a violating object."""
        broken = _BrokenBuffer()
        with pytest.raises(TypeError, match="write_count"):
            assert_satisfies_protocol(broken, PostTrainingBufferProtocol, name="broken_buf")

    def test_assert_satisfies_error_includes_name(self):
        """Error message must include the name argument."""
        broken = _BrokenBuffer()
        with pytest.raises(TypeError, match="my_buffer"):
            assert_satisfies_protocol(broken, PostTrainingBufferProtocol, name="my_buffer")


# ---------------------------------------------------------------------------
# PostTrainingCollectorProtocol
# ---------------------------------------------------------------------------


class TestPostTrainingCollectorProtocol:
    def test_minimal_collector_satisfies(self):
        col = _MinimalCollector()
        assert isinstance(col, PostTrainingCollectorProtocol)

    def test_plain_object_does_not_satisfy(self):
        assert not isinstance(object(), PostTrainingCollectorProtocol)

    def test_assert_satisfies_passes(self):
        col = _MinimalCollector()
        assert_satisfies_protocol(col, PostTrainingCollectorProtocol, name="col")

    def test_assert_satisfies_raises(self):
        with pytest.raises(TypeError):
            assert_satisfies_protocol(object(), PostTrainingCollectorProtocol, name="bad")


# ---------------------------------------------------------------------------
# PostTrainingLossOutputProtocol
# ---------------------------------------------------------------------------


class TestPostTrainingLossOutputProtocol:
    def test_minimal_loss_output_satisfies(self):
        out = _MinimalLossOutput()
        # Use assert_satisfies_protocol (getattr-based) instead of isinstance
        assert_satisfies_protocol(out, PostTrainingLossOutputProtocol, name="minimal")

    def test_broken_loss_output_does_not_satisfy(self):
        with pytest.raises(TypeError):
            assert_satisfies_protocol(
                _BrokenLossOutput(), PostTrainingLossOutputProtocol
            )

    def test_grpo_loss_output_satisfies(self):
        """GRPOLossOutput must satisfy the protocol (via assert_satisfies_protocol)."""
        pytest.importorskip("torchrl.objectives.llm.grpo")
        from torchrl.objectives.llm.grpo import GRPOLossOutput
        out = GRPOLossOutput(
            loss_objective=torch.tensor(0.1),
            clip_fraction=torch.tensor(0.0),
            kl_approx=torch.tensor(0.0),
            ESS=torch.tensor(1.0),
        )
        # GRPOLossOutput is a TensorClass: isinstance won't work, but
        # assert_satisfies_protocol uses getattr-based checking.
        assert_satisfies_protocol(out, PostTrainingLossOutputProtocol, name="grpo_out")

    def test_sft_loss_output_satisfies(self):
        """SFTLossOutput must satisfy the protocol (it has loss_sft)."""
        pytest.importorskip("torchrl.objectives.llm.sft")
        from torchrl.objectives.llm.sft import SFTLossOutput
        out = SFTLossOutput(loss_sft=torch.tensor(0.2))
        assert_satisfies_protocol(out, PostTrainingLossOutputProtocol, name="sft_out")

    def test_assert_satisfies_passes(self):
        out = _MinimalLossOutput()
        assert_satisfies_protocol(out, PostTrainingLossOutputProtocol, name="loss_out")

    def test_assert_satisfies_raises(self):
        with pytest.raises(TypeError):
            assert_satisfies_protocol(_BrokenLossOutput(), PostTrainingLossOutputProtocol)


# ---------------------------------------------------------------------------
# assert_satisfies_protocol — edge cases
# ---------------------------------------------------------------------------


class TestAssertSatisfiesProtocol:
    def test_no_name_in_error(self):
        """Without a name argument the error must still be raised."""
        broken = _BrokenBuffer()
        with pytest.raises(TypeError):
            assert_satisfies_protocol(broken, PostTrainingBufferProtocol)

    def test_importable_from_top_level_data_llm(self):
        """Must be importable from the torchrl.data.llm namespace."""
        from torchrl.data.llm import assert_satisfies_protocol as asp
        assert asp is assert_satisfies_protocol

    def test_protocols_importable_from_top_level_data_llm(self):
        from torchrl.data.llm import (
            PostTrainingBufferProtocol as PBP,
            PostTrainingCollectorProtocol as PCP,
            PostTrainingLossOutputProtocol as PLOP,
        )
        assert PBP is PostTrainingBufferProtocol
        assert PCP is PostTrainingCollectorProtocol
        assert PLOP is PostTrainingLossOutputProtocol
