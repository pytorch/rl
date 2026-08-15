# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.nn import TensorDictModule

from torchrl.envs.batched_inference import FixedBatchedInference


CUDA_AVAILABLE = torch.cuda.is_available()


def _make_policy(in_features: int = 4, out_features: int = 2, device="cpu"):
    return TensorDictModule(
        torch.nn.Linear(in_features, out_features).to(device),
        in_keys=["obs"],
        out_keys=["action"],
    )


def _make_batch(B: int, obs_dim: int = 4, with_env_index: bool = False) -> TensorDict:
    td = TensorDict({"obs": torch.randn(B, obs_dim)}, batch_size=[B])
    if with_env_index:
        td.set("env_index", NonTensorStack(*list(range(B))))
    return td


class TestConstruction:
    def test_empty_bucket_sizes_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[])

    def test_zero_bucket_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[0, 8])

    def test_duplicate_bucket_sizes_deduplicated(self):
        helper = FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[8, 8, 16])
        assert helper.bucket_sizes == [8, 16]

    def test_bucket_sizes_sorted(self):
        helper = FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[64, 8, 32])
        assert helper.bucket_sizes == [8, 32, 64]

    def test_nn_module_moved_to_device(self):
        policy = torch.nn.Linear(4, 2)
        helper = FixedBatchedInference(policy, "cpu")
        # On CPU the policy should remain accessible and on cpu
        assert next(helper.policy.parameters()).device.type == "cpu"


class TestBucketSelection:
    def setup_method(self):
        self.helper = FixedBatchedInference(
            _make_policy(), "cpu", bucket_sizes=[8, 16, 32]
        )

    def test_exact_match(self):
        assert self.helper._pick_bucket(8) == 8
        assert self.helper._pick_bucket(16) == 16

    def test_rounds_up(self):
        assert self.helper._pick_bucket(1) == 8
        assert self.helper._pick_bucket(9) == 16
        assert self.helper._pick_bucket(17) == 32

    def test_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="exceeds the largest bucket"):
            self.helper._pick_bucket(33)


class TestCPUHotPath:
    def _make_helper(self, **kwargs):
        return FixedBatchedInference(
            _make_policy(), "cpu", bucket_sizes=[4, 8, 16], **kwargs
        )

    def test_output_shape_matches_input(self):
        helper = self._make_helper()
        for B in (1, 4, 5, 8, 16):
            out = helper(_make_batch(B))
            assert out.batch_size == torch.Size([B]), f"B={B}"

    def test_valid_mask_stripped_from_output(self):
        out = self._make_helper(add_valid_mask=True)(_make_batch(3))
        assert "valid_mask" not in out.keys()

    def test_no_valid_mask_when_disabled(self):
        out = self._make_helper(add_valid_mask=False)(_make_batch(3))
        assert "valid_mask" not in out.keys()

    def test_action_key_present(self):
        out = self._make_helper()(_make_batch(5))
        assert "action" in out.keys()
        assert out["action"].shape == torch.Size([5, 2])

    def test_padding_rows_do_not_corrupt_output(self):
        """Identical batches must produce identical actions regardless of padding."""
        torch.manual_seed(0)
        helper = FixedBatchedInference(
            _make_policy(), "cpu", bucket_sizes=[8], double_buffer=False
        )
        batch = _make_batch(3)
        torch.testing.assert_close(helper(batch)["action"], helper(batch)["action"])

    def test_zero_batch_raises(self):
        with pytest.raises(ValueError, match="empty batch"):
            self._make_helper()(_make_batch(0))

    def test_non_1d_batch_raises(self):
        td = TensorDict({"obs": torch.randn(2, 4, 4)}, batch_size=[2, 4])
        with pytest.raises(ValueError, match="1-D"):
            self._make_helper()(td)

    def test_reset_clears_state(self):
        helper = self._make_helper()
        helper(_make_batch(3))
        assert helper._initialized
        helper.reset()
        assert not helper._initialized
        assert not helper._staging

    def test_context_manager_calls_reset(self):
        helper = self._make_helper()
        with helper:
            helper(_make_batch(3))
            assert helper._initialized
        assert not helper._initialized

    def test_env_index_preserved(self):
        """Non-tensor env_index must survive the staging round-trip."""
        helper = self._make_helper()
        batch = _make_batch(3, with_env_index=True)
        out = helper(batch)
        assert "env_index" in out.keys()
        indices = [out.get("env_index")[i].data for i in range(3)]
        assert indices == [0, 1, 2]

    def test_env_index_not_in_staging(self):
        """env_index must not be written into the pinned staging buffer."""
        helper = self._make_helper()
        batch = _make_batch(3, with_env_index=True)
        helper(batch)
        for buf in helper._staging[4]:
            assert "env_index" not in buf.keys()


class TestDoubleBuffer:
    def test_buf_idx_advances(self):
        helper = FixedBatchedInference(
            _make_policy(), "cpu", bucket_sizes=[8], double_buffer=True
        )
        helper(_make_batch(4))
        assert helper._buf_idx[8] == 1
        helper(_make_batch(4))
        assert helper._buf_idx[8] == 0

    def test_single_buffer_stays_at_zero(self):
        helper = FixedBatchedInference(
            _make_policy(), "cpu", bucket_sizes=[8], double_buffer=False
        )
        helper(_make_batch(4))
        assert helper._buf_idx[8] == 0
        helper(_make_batch(4))
        assert helper._buf_idx[8] == 0

    def test_two_buffers_allocated(self):
        helper = FixedBatchedInference(
            _make_policy(), "cpu", bucket_sizes=[8], double_buffer=True
        )
        helper(_make_batch(4))
        assert len(helper._staging[8]) == 2

    def test_one_buffer_allocated_when_disabled(self):
        helper = FixedBatchedInference(
            _make_policy(), "cpu", bucket_sizes=[8], double_buffer=False
        )
        helper(_make_batch(4))
        assert len(helper._staging[8]) == 1


class TestMultipleBuckets:
    def test_correct_bucket_used(self):
        helper = FixedBatchedInference(
            _make_policy(), "cpu", bucket_sizes=[4, 8, 16], double_buffer=False
        )
        helper(_make_batch(3))
        helper(_make_batch(5))
        helper(_make_batch(12))
        assert set(helper._staging.keys()) == {4, 8, 16}


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDA:
    def test_policy_moved_to_cuda(self):
        policy = torch.nn.Linear(4, 2)  # starts on CPU
        helper = FixedBatchedInference(policy, "cuda:0", bucket_sizes=[8])
        assert next(helper.policy.parameters()).device.type == "cuda"

    def test_output_on_device(self):
        helper = FixedBatchedInference(
            _make_policy(), "cuda:0", bucket_sizes=[8]
        )
        out = helper(_make_batch(3))
        assert out["action"].device.type == "cuda"

    def test_output_shape_on_cuda(self):
        helper = FixedBatchedInference(
            _make_policy(), "cuda:0", bucket_sizes=[8, 16]
        )
        for B in (1, 5, 8, 16):
            out = helper(_make_batch(B))
            assert out.batch_size == torch.Size([B])

    def test_valid_mask_in_pinned_staging(self):
        """valid_mask must be pre-allocated in pinned memory, not added after."""
        helper = FixedBatchedInference(
            _make_policy(), "cuda:0", bucket_sizes=[8], add_valid_mask=True
        )
        helper(_make_batch(3))
        for buf in helper._staging[8]:
            assert "valid_mask" in buf.keys()
            assert buf.get("valid_mask").is_pinned()

    def test_output_readable_after_stream_handoff(self):
        """Calling stream must not race with compute stream on result tensors."""
        helper = FixedBatchedInference(
            _make_policy(), "cuda:0", bucket_sizes=[8]
        )
        batch = _make_batch(5)
        out = helper(batch)
        # If stream sync is broken this may return zeros or garbage.
        # At minimum it must not hang or raise.
        torch.cuda.synchronize()
        assert out["action"].shape == torch.Size([5, 2])
        assert not out["action"].isnan().any()

    def test_separate_copy_and_compute_streams(self):
        helper = FixedBatchedInference(
            _make_policy(), "cuda:0", bucket_sizes=[8]
        )
        assert helper._copy_stream is not None
        assert helper._compute_stream is not None
        assert helper._copy_stream != helper._compute_stream

    def test_env_index_preserved_on_cuda(self):
        helper = FixedBatchedInference(
            _make_policy(), "cuda:0", bucket_sizes=[8]
        )
        batch = _make_batch(3, with_env_index=True)
        out = helper(batch)
        assert "env_index" in out.keys()
        indices = [out.get("env_index")[i].data for i in range(3)]
        assert indices == [0, 1, 2]


def test_importable_from_torchrl_envs():
    from torchrl.envs import FixedBatchedInference as FI  # noqa: F401

    assert FI is FixedBatchedInference
