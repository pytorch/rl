# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.tensorclass import NonTensorStack

from torchrl.envs.batched_inference import FixedBatchedInference


def _make_policy(in_features: int = 4, out_features: int = 2):
    return TensorDictModule(
        torch.nn.Linear(in_features, out_features),
        in_keys=["obs"],
        out_keys=["action"],
    )


def _make_batch(
    B: int,
    obs_dim: int = 4,
    with_env_index: bool = False,
    nested: bool = False,
) -> TensorDict:
    batch = TensorDict({"obs": torch.randn(B, obs_dim)}, batch_size=[B])
    if with_env_index:
        batch.set("env_index", NonTensorStack(*range(B)))
    if nested:
        batch.set("state", TensorDict({"hidden": torch.randn(B, 8)}, batch_size=[B]))
    return batch


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
        helper = self._make_helper()
        out = helper(_make_batch(3, with_env_index=True))

        assert list(out["env_index"]) == [0, 1, 2]

    def test_nested_tensordict_available_to_policy(self):
        policy = TensorDictModule(
            torch.nn.Linear(8, 2),
            in_keys=[("state", "hidden")],
            out_keys=["action"],
        )
        helper = FixedBatchedInference(policy, "cpu", bucket_sizes=[4])

        out = helper(_make_batch(3, nested=True))

        assert out["action"].shape == torch.Size([3, 2])

    def test_changed_keys_raise_instead_of_reusing_stale_values(self):
        helper = self._make_helper()
        helper(_make_batch(3))
        changed_batch = TensorDict(
            {"unrelated": torch.randn(3, 4)}, batch_size=[3]
        )

        with pytest.raises(ValueError, match="Tensor keys changed"):
            helper(changed_batch)

    def test_output_does_not_alias_reused_staging(self):
        policy = TensorDictModule(
            torch.nn.Identity(), in_keys=["obs"], out_keys=["action"]
        )
        helper = FixedBatchedInference(
            policy, "cpu", bucket_sizes=[4], double_buffer=False
        )
        first = helper(TensorDict({"obs": torch.zeros(3, 4)}, batch_size=[3]))
        expected = first["action"].clone()

        helper(TensorDict({"obs": torch.ones(3, 4)}, batch_size=[3]))

        torch.testing.assert_close(first["action"], expected)


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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_output_device():
    helper = FixedBatchedInference(
        _make_policy().to("cuda:0"), "cuda:0", bucket_sizes=[8]
    )

    out = helper(_make_batch(3))

    assert out["action"].device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_output_readable_after_stream_handoff():
    policy = TensorDictModule(
        torch.nn.Identity(), in_keys=["obs"], out_keys=["action"]
    ).to("cuda:0")
    helper = FixedBatchedInference(policy, "cuda:0", bucket_sizes=[8])
    batch = _make_batch(5)
    caller_stream = torch.cuda.Stream()

    with torch.cuda.stream(caller_stream):
        out = helper(batch)
        observed = out["action"].clone()
    caller_stream.synchronize()

    torch.testing.assert_close(observed.cpu(), batch["obs"])


def test_importable_from_torchrl_envs():
    from torchrl.envs import FixedBatchedInference as FI  # noqa: F401

    assert FI is FixedBatchedInference
