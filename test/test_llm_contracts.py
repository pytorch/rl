# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.llm.contracts import (
    assert_buffer_contract,
    assert_collector_contract,
    assert_loss_contract,
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
    """Missing write_count — does NOT satisfy the contract."""

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


class TestBufferContract:
    def test_torchrl_replay_buffer_satisfies(self):
        assert_buffer_contract(_make_rb())

    def test_minimal_custom_buffer_satisfies(self):
        assert_buffer_contract(_MinimalBuffer())

    def test_assert_raises_for_broken(self):
        with pytest.raises(TypeError, match="write_count"):
            assert_buffer_contract(_BrokenBuffer())


class TestCollectorContract:
    def test_minimal_collector_satisfies(self):
        assert_collector_contract(_MinimalCollector())

    def test_plain_object_does_not_satisfy(self):
        with pytest.raises(TypeError, match="update_policy_weights_"):
            assert_collector_contract(object())


class TestLossContract:
    def test_minimal_grpo_output_satisfies(self):
        assert_loss_contract(_MinimalGRPOOutput(), loss_type="grpo")

    def test_grpo_loss_output_satisfies(self):
        out = GRPOLossOutput(
            loss_objective=torch.tensor(0.1),
            clip_fraction=torch.tensor(0.0),
            kl_approx=torch.tensor(0.0),
            ESS=torch.tensor(1.0),
        )
        assert_loss_contract(out, loss_type="grpo")

    def test_minimal_sft_output_satisfies(self):
        assert_loss_contract(_MinimalSFTOutput(), loss_type="sft")

    def test_sft_loss_output_satisfies(self):
        out = SFTLossOutput(loss_sft=torch.tensor(0.2))
        assert_loss_contract(out, loss_type="sft")

    def test_grpo_output_fails_sft_contract(self):
        out = GRPOLossOutput(
            loss_objective=torch.tensor(0.1),
            clip_fraction=torch.tensor(0.0),
            kl_approx=torch.tensor(0.0),
            ESS=torch.tensor(1.0),
        )
        with pytest.raises(TypeError, match="loss_sft"):
            assert_loss_contract(out, loss_type="sft")

    def test_sft_output_fails_grpo_contract(self):
        out = SFTLossOutput(loss_sft=torch.tensor(0.2))
        with pytest.raises(TypeError, match="loss_objective"):
            assert_loss_contract(out, loss_type="grpo")

    def test_broken_output_does_not_satisfy(self):
        with pytest.raises(TypeError):
            assert_loss_contract(_BrokenOutput(), loss_type="grpo")

    def test_invalid_loss_type(self):
        with pytest.raises(ValueError, match="Unknown loss_type"):
            assert_loss_contract(_MinimalGRPOOutput(), loss_type="ppo")


class TestInitExports:
    def test_importable_from_top_level_data_llm(self):
        from torchrl.data.llm import (
            assert_buffer_contract,
            assert_collector_contract,
            assert_loss_contract,
        )

        assert assert_buffer_contract is not None
        assert assert_collector_contract is not None
        assert assert_loss_contract is not None


if __name__ == "__main__":
    pytest.main([__file__])
