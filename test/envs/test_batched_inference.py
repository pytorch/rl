# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from functools import partial
from unittest import mock

import pytest
import torch
from tensordict import set_capture_non_tensor_stack, TensorDict
from tensordict.nn import TensorDictModule

from torchrl.envs import AsyncEnvPool, FixedBatchedInference
from torchrl.testing.mocking_classes import CountingEnv


def _make_policy(
    in_features: int = 4,
    out_features: int = 2,
    device: torch.device | str | None = None,
):
    return TensorDictModule(
        torch.nn.Linear(in_features, out_features, device=device),
        in_keys=["obs"],
        out_keys=["action"],
    )


def _make_batch(batch_size: int, obs_dim: int = 4, nested: bool = False) -> TensorDict:
    td = TensorDict({"obs": torch.randn(batch_size, obs_dim)}, batch_size=[batch_size])
    if nested:
        td.set(
            "state",
            TensorDict({"hidden": torch.randn(batch_size, 8)}, batch_size=[batch_size]),
        )
    return td


@pytest.mark.parametrize(
    ("bucket_sizes", "match"),
    [([], "must not be empty"), ([0, 8], ">= 1")],
)
def test_invalid_bucket_sizes(bucket_sizes, match):
    with pytest.raises(ValueError, match=match):
        FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=bucket_sizes)


def test_policy_placement_is_left_to_caller():
    policy = _make_policy()
    with mock.patch.object(
        policy,
        "to",
        side_effect=AssertionError("policy placement must not be changed"),
    ):
        helper = FixedBatchedInference(policy, "cpu")
    assert helper.policy is policy


def test_output_shape_across_buckets():
    helper = FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[4, 8, 16])
    for batch_size in (1, 4, 5, 8, 16):
        out = helper(_make_batch(batch_size))
        assert out.batch_size == torch.Size([batch_size])
        assert out["action"].shape == torch.Size([batch_size, 2])


def test_batch_larger_than_max_bucket_raises():
    helper = FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[4, 8])
    with pytest.raises(ValueError, match="exceeds the largest bucket"):
        helper(_make_batch(9))


def test_valid_mask_available_to_policy_and_stripped_from_output():
    policy = TensorDictModule(
        torch.nn.Identity(), in_keys=["valid_mask"], out_keys=["action"]
    )
    helper = FixedBatchedInference(policy, "cpu", bucket_sizes=[8])

    out = helper(_make_batch(3))

    assert "valid_mask" not in out.keys()
    torch.testing.assert_close(out["action"], torch.ones(3, dtype=torch.bool))


def test_output_does_not_alias_reused_staging():
    policy = TensorDictModule(torch.nn.Identity(), in_keys=["obs"], out_keys=["action"])
    helper = FixedBatchedInference(policy, "cpu", bucket_sizes=[4], double_buffer=False)
    first = helper(TensorDict({"obs": torch.zeros(3, 4)}, batch_size=[3]))
    expected = first["action"].clone()

    helper(TensorDict({"obs": torch.ones(3, 4)}, batch_size=[3]))

    torch.testing.assert_close(first["action"], expected)


def test_zero_batch_raises():
    helper = FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[4])
    with pytest.raises(ValueError, match="empty batch"):
        helper(_make_batch(0))


def test_non_1d_batch_raises():
    helper = FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[8])
    batch = TensorDict({"obs": torch.randn(2, 4, 4)}, batch_size=[2, 4])
    with pytest.raises(ValueError, match="1-D"):
        helper(batch)


def test_nested_tensordict_available_to_policy():
    policy = TensorDictModule(
        torch.nn.Linear(8, 2),
        in_keys=[("state", "hidden")],
        out_keys=["action"],
    )
    helper = FixedBatchedInference(policy, "cpu", bucket_sizes=[4])

    out = helper(_make_batch(3, nested=True))

    assert out["action"].shape == torch.Size([3, 2])


def test_changed_keys_raise_instead_of_reusing_stale_values():
    helper = FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[4])
    helper(_make_batch(3))
    changed_batch = TensorDict({"unrelated": torch.randn(3, 4)}, batch_size=[3])

    with pytest.raises(ValueError, match="Tensor keys changed"):
        helper(changed_batch)


def test_only_policy_input_keys_are_staged():
    helper = FixedBatchedInference(_make_policy(), "cpu", bucket_sizes=[4])
    batch = _make_batch(3)
    batch.set("reward", torch.randn(3, 1))

    out = helper(batch)

    staged = set(helper._staging[4][0].keys())
    assert "obs" in staged
    assert "reward" not in staged
    assert "reward" not in out.keys()
    # Staged inputs are not echoed back to the caller.
    assert "obs" not in out.keys()
    assert out["action"].shape == torch.Size([3, 2])


def test_select_keys_for_plain_callables():
    def policy(td):
        return td.set("action", td["obs"].sum(-1, keepdim=True))

    helper = FixedBatchedInference(policy, "cpu", bucket_sizes=[4], select_keys=["obs"])
    batch = _make_batch(3)
    batch.set("reward", torch.randn(3, 1))

    out = helper(batch)

    assert "reward" not in helper._staging[4][0].keys()
    torch.testing.assert_close(out["action"], batch["obs"].sum(-1, keepdim=True))


@pytest.mark.parametrize(
    "backend,exchange",
    [("threading", None), ("multiprocessing", "shm")],
)
@set_capture_non_tensor_stack(False)
def test_async_env_pool_routes_inference_by_env_index(backend, exchange):
    kwargs = {"exchange": exchange} if exchange is not None else {}
    pool = AsyncEnvPool(
        [partial(CountingEnv, start_val=index, max_steps=100) for index in range(3)],
        backend=backend,
        **kwargs,
    )
    helper = FixedBatchedInference(
        TensorDictModule(
            torch.nn.Identity(),
            in_keys=["observation"],
            out_keys=["action"],
        ),
        "cpu",
        bucket_sizes=[4],
    )
    try:
        pool.async_reset_send(env_index=[0, 1])
        observations = pool.async_reset_recv(min_get=2)
        actions = helper(observations)
        assert list(actions["env_index"]) == [0, 1]

        pool.async_step_and_maybe_reset_send(actions)
        _, next_observations = pool.async_step_and_maybe_reset_recv(min_get=2)

        assert list(next_observations["env_index"]) == [0, 1]
        torch.testing.assert_close(
            next_observations["observation"].squeeze(-1),
            torch.tensor([0, 2], dtype=torch.int32),
        )
    finally:
        pool._maybe_shutdown()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    def test_output_device_and_shape(self):
        helper = FixedBatchedInference(
            _make_policy(device="cuda:0"), "cuda:0", bucket_sizes=[8, 16]
        )
        for batch_size in (1, 5, 8, 16):
            out = helper(_make_batch(batch_size))
            assert out.batch_size == torch.Size([batch_size])
            assert out["action"].device.type == "cuda"

    def test_valid_mask_in_pinned_staging(self):
        helper = FixedBatchedInference(
            _make_policy(device="cuda:0"),
            "cuda:0",
            bucket_sizes=[8],
            add_valid_mask=True,
        )
        helper(_make_batch(3))
        for buffer in helper._staging[8]:
            assert buffer["valid_mask"].is_pinned()

    def test_output_readable_after_stream_handoff(self):
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

    def test_device_buffer_is_persistent(self):
        """With double_buffer=False every call reuses the same device storage.

        Stable input pointers are what manual CUDA-graph capture of the
        policy requires; a fresh device allocation per call would silently
        break replay.
        """
        helper = FixedBatchedInference(
            _make_policy(device="cuda:0"),
            "cuda:0",
            bucket_sizes=[8],
            double_buffer=False,
        )
        out_first = helper(_make_batch(3))
        pointer = helper._device_batches[8][0]["obs"].data_ptr()
        out_second = helper(_make_batch(5))
        assert helper._device_batches[8][0]["obs"].data_ptr() == pointer
        assert out_first["action"].shape == torch.Size([3, 2])
        assert out_second["action"].shape == torch.Size([5, 2])


if __name__ == "__main__":
    pytest.main([__file__])
