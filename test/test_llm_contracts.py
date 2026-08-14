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
    GRPOLossOutputProtocol,
    PostTrainingBufferProtocol,
    PostTrainingCollectorProtocol,
    SFTLossOutputProtocol,
    assert_satisfies_protocol,
)
from torchrl.objectives.llm.grpo import GRPOLossOutput
from torchrl.objectives.llm.sft import SFTLossOutput


def _make_rb(capacity: int = 100) -> ReplayBuffer:
    return ReplayBuffer(storage=LazyTensorStorage(capacity), batch_size=4)


class _MinimalBuffer:
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
    def update_policy_weights_(self, policy_weights=None, *, worker_ids=None):
        pass

    def __iter__(self):
        return iter([TensorDict({}, [])])

    def __next__(self):
        return TensorDict({}, [])


class _MinimalGRPOOutput:
    @property
    def loss_objective(self) -> torch.Tensor:
        return torch.tensor(0.5)


class _MinimalSFTOutput:
    @property
    def loss_sft(self) -> torch.Tensor:
        return torch.tensor(0.3)


class _BrokenOutput:
    pass


class TestPostTrainingBufferProtocol:
    def test_torchrl_replay_buffer_satisfies(self):
        assert isinstance(_make_rb(), PostTrainingBufferProtocol)

    def test_minimal_custom_buffer_satisfies(self):
        assert isinstance(_MinimalBuffer(), PostTrainingBufferProtocol)

    def test_broken_buffer_does_not_satisfy(self):
        assert not isinstance(_BrokenBuffer(), PostTrainingBufferProtocol)

    def test_assert_satisfies_passes_silently(self):
        assert_satisfies_protocol(_make_rb(), PostTrainingBufferProtocol, name="rb")

    def test_assert_satisfies_raises_for_broken(self):
        with pytest.raises(TypeError, match="write_count"):
            assert_satisfies_protocol(_BrokenBuffer(), PostTrainingBufferProtocol, name="broken_buf")

    def test_assert_satisfies_error_includes_name(self):
        with pytest.raises(TypeError, match="my_buffer"):
            assert_satisfies_protocol(_BrokenBuffer(), PostTrainingBufferProtocol, name="my_buffer")


class TestPostTrainingCollectorProtocol:
    def test_minimal_collector_satisfies(self):
        assert isinstance(_MinimalCollector(), PostTrainingCollectorProtocol)

    def test_plain_object_does_not_satisfy(self):
        assert not isinstance(object(), PostTrainingCollectorProtocol)

    def test_assert_satisfies_passes(self):
        assert_satisfies_protocol(_MinimalCollector(), PostTrainingCollectorProtocol, name="col")

    def test_assert_satisfies_raises(self):
        with pytest.raises(TypeError):
            assert_satisfies_protocol(object(), PostTrainingCollectorProtocol, name="bad")


class TestGRPOLossOutputProtocol:
    def test_minimal_grpo_output_satisfies(self):
        assert_satisfies_protocol(_MinimalGRPOOutput(), GRPOLossOutputProtocol)

    def test_broken_output_does_not_satisfy(self):
        with pytest.raises(TypeError):
            assert_satisfies_protocol(_BrokenOutput(), GRPOLossOutputProtocol)

    def test_grpo_loss_output_satisfies(self):
        out = GRPOLossOutput(
            loss_objective=torch.tensor(0.1),
            clip_fraction=torch.tensor(0.0),
            kl_approx=torch.tensor(0.0),
            ESS=torch.tensor(1.0),
        )
        assert_satisfies_protocol(out, GRPOLossOutputProtocol, name="grpo_out")

    def test_sft_output_does_not_satisfy_grpo_protocol(self):
        """SFTLossOutput must NOT satisfy GRPOLossOutputProtocol (has loss_sft, not loss_objective)."""
        out = SFTLossOutput(loss_sft=torch.tensor(0.2))
        with pytest.raises(TypeError):
            assert_satisfies_protocol(out, GRPOLossOutputProtocol)


class TestSFTLossOutputProtocol:
    def test_minimal_sft_output_satisfies(self):
        assert_satisfies_protocol(_MinimalSFTOutput(), SFTLossOutputProtocol)

    def test_broken_output_does_not_satisfy(self):
        with pytest.raises(TypeError):
            assert_satisfies_protocol(_BrokenOutput(), SFTLossOutputProtocol)

    def test_sft_loss_output_satisfies(self):
        out = SFTLossOutput(loss_sft=torch.tensor(0.2))
        assert_satisfies_protocol(out, SFTLossOutputProtocol, name="sft_out")

    def test_grpo_output_does_not_satisfy_sft_protocol(self):
        """GRPOLossOutput must NOT satisfy SFTLossOutputProtocol (has loss_objective, not loss_sft)."""
        out = GRPOLossOutput(
            loss_objective=torch.tensor(0.1),
            clip_fraction=torch.tensor(0.0),
            kl_approx=torch.tensor(0.0),
            ESS=torch.tensor(1.0),
        )
        with pytest.raises(TypeError):
            assert_satisfies_protocol(out, SFTLossOutputProtocol)


class TestAssertSatisfiesProtocol:
    def test_no_name_in_error(self):
        with pytest.raises(TypeError):
            assert_satisfies_protocol(_BrokenBuffer(), PostTrainingBufferProtocol)

    def test_importable_from_top_level_data_llm(self):
        from torchrl.data.llm import assert_satisfies_protocol as asp
        assert asp is assert_satisfies_protocol

    def test_protocols_importable_from_top_level_data_llm(self):
        from torchrl.data.llm import (
            GRPOLossOutputProtocol as GLOP,
            PostTrainingBufferProtocol as PBP,
            PostTrainingCollectorProtocol as PCP,
            SFTLossOutputProtocol as SLOP,
        )
        assert PBP is PostTrainingBufferProtocol
        assert PCP is PostTrainingCollectorProtocol
        assert GLOP is GRPOLossOutputProtocol
        assert SLOP is SFTLossOutputProtocol


if __name__ == "__main__":
    pytest.main([__file__])
