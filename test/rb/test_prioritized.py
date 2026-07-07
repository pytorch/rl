# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import functools
import warnings

import numpy as np
import pytest
import torch
from tensordict import assert_allclose_td, is_tensorclass, TensorDict

from torchrl.data import (
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import samplers
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
)
from torchrl.testing import get_default_devices, make_tc


@pytest.mark.parametrize("priority_key", ["pk", "td_error"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_ptdrb(priority_key, contiguous, alpha, device):
    torch.manual_seed(0)
    np.random.seed(0)
    rb = TensorDictReplayBuffer(
        sampler=samplers.PrioritizedSampler(5, alpha=alpha, beta=0.9),
        priority_key=priority_key,
        batch_size=5,
    )
    td1 = TensorDict(
        source={
            "a": torch.randn(3, 1),
            priority_key: torch.rand(3, 1) / 10,
            "_idx": torch.arange(3).view(3, 1),
        },
        batch_size=[3],
        device=device,
    )
    rb.extend(td1)
    s = rb.sample()
    assert s.batch_size == torch.Size([5])
    assert (td1[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td1[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test replacement
    td2 = TensorDict(
        source={
            "a": torch.randn(5, 1),
            priority_key: torch.rand(5, 1) / 10,
            "_idx": torch.arange(5).view(5, 1),
        },
        batch_size=[5],
        device=device,
    )
    rb.extend(td2)
    s = rb.sample()
    assert s.batch_size == torch.Size([5])
    assert (td2[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td2[s.get("_idx").squeeze()].select("a"), s.select("a"))

    if (
        alpha == 0.0
    ):  # when alpha is 0.0, sampling is uniform, so no need to check priority sampling
        return

    # test strong update
    # get all indices that match first item
    idx = s.get("_idx")
    idx_match = (idx == idx[0]).nonzero()[:, 0]
    s.set_at_(
        priority_key,
        torch.ones(idx_match.numel(), 1, device=device) * 100000000,
        idx_match,
    )
    val = s.get("a")[0]

    idx0 = s.get("_idx")[0]
    rb.update_tensordict_priority(s)
    s = rb.sample()
    assert (val == s.get("a")).sum() >= 1
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))

    # test updating values of original td
    td2.set_("a", torch.ones_like(td2.get("a")))
    s = rb.sample()
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_segment_tree_parity():
    ext = pytest.importorskip("torchrl._torchrl")
    if not hasattr(ext, "CudaSumSegmentTreeFp32"):
        pytest.skip("TorchRL was not built with CUDA segment tree support")
    CudaMinSegmentTreeFp32 = ext.CudaMinSegmentTreeFp32
    CudaSumSegmentTreeFp32 = ext.CudaSumSegmentTreeFp32
    MinSegmentTreeFp32 = ext.MinSegmentTreeFp32
    SumSegmentTreeFp32 = ext.SumSegmentTreeFp32

    device = torch.device("cuda:0")
    size = 16
    index = torch.tensor([0, 3, 4, 7, 12, 15], device=device)
    value = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], device=device)

    cpu_sum = SumSegmentTreeFp32(size)
    cpu_min = MinSegmentTreeFp32(size)
    cuda_sum = CudaSumSegmentTreeFp32(size, device)
    cuda_min = CudaMinSegmentTreeFp32(size, device)

    cpu_sum[index.cpu()] = value.cpu()
    cpu_min[index.cpu()] = value.cpu()
    cuda_sum[index] = value
    cuda_min[index] = value

    left = torch.tensor([0, 3, 4, 7], device=device)
    right = torch.tensor([16, 8, 13, 16], device=device)
    torch.testing.assert_close(
        cuda_sum.query(left, right).cpu(), cpu_sum.query(left.cpu(), right.cpu())
    )
    torch.testing.assert_close(
        cuda_min.query(left, right).cpu(), cpu_min.query(left.cpu(), right.cpu())
    )

    mass = torch.tensor([0.5, 1.0, 2.9, 7.1, 30.0], device=device)
    torch.testing.assert_close(
        cuda_sum.scan_lower_bound(mass).cpu(),
        cpu_sum.scan_lower_bound(mass.cpu()),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_prioritized_replay_buffer_samples_on_cuda():
    ext = pytest.importorskip("torchrl._torchrl")
    if not hasattr(ext, "CudaSumSegmentTreeFp32"):
        pytest.skip("TorchRL was not built with CUDA segment tree support")
    device = torch.device("cuda:0")
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(32, device=device),
        sampler=PrioritizedSampler(max_capacity=32, alpha=0.7, beta=0.5),
        batch_size=8,
        priority_key="td_error",
    )
    data = TensorDict(
        {
            "obs": torch.arange(16, device=device).float().unsqueeze(-1),
            "td_error": torch.linspace(0.1, 1.0, 16, device=device),
        },
        batch_size=[16],
        device=device,
    )

    rb.extend(data)
    sample = rb.sample()

    assert sample.device == device
    assert sample["index"].device == device
    assert sample["priority_weight"].device == device

    sample["td_error"] = torch.ones_like(sample["td_error"]) * 10
    rb.update_tensordict_priority(sample)
    sample = rb.sample()
    assert sample["index"].device == device
    assert sample["priority_weight"].device == device


def test_tensordict_prioritized_replay_buffer_sampler_device_cpu():
    rb = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.5,
        storage=LazyTensorStorage(32),
        sampler_device="cpu",
        batch_size=8,
        priority_key="td_error",
    )
    data = TensorDict(
        {
            "obs": torch.arange(16).float().unsqueeze(-1),
            "td_error": torch.linspace(0.1, 1.0, 16),
        },
        batch_size=[16],
    )

    rb.extend(data)
    sample = rb.sample()

    assert rb._sampler.device == torch.device("cpu")
    assert sample["index"].device == torch.device("cpu")
    assert sample["priority_weight"].device == torch.device("cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tensordict_prioritized_replay_buffer_memmap_storage_cuda_sampler(tmpdir):
    ext = pytest.importorskip("torchrl._torchrl")
    if not hasattr(ext, "CudaSumSegmentTreeFp32"):
        pytest.skip("TorchRL was not built with CUDA segment tree support")

    rb = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.5,
        storage=LazyMemmapStorage(32, scratch_dir=tmpdir),
        sampler_device="cuda:0",
        batch_size=8,
        priority_key="td_error",
    )
    data = TensorDict(
        {
            "obs": torch.arange(16).float().unsqueeze(-1),
            "td_error": torch.linspace(0.1, 1.0, 16),
        },
        batch_size=[16],
    )

    rb.extend(data)
    sample = rb.sample()

    assert rb._sampler.device == torch.device("cuda:0")
    assert sample["obs"].device.type == "cpu"
    assert sample["index"].device.type == "cpu"
    assert sample["priority_weight"].device.type == "cpu"

    sample["td_error"] = torch.ones_like(sample["td_error"]) * 10
    rb.update_tensordict_priority(sample)
    sample = rb.sample()
    assert sample["index"].device.type == "cpu"
    assert rb._sampler.device == torch.device("cuda:0")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tensordict_prioritized_replay_buffer_cuda_storage_cpu_sampler():
    device = torch.device("cuda:0")
    rb = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.5,
        storage=LazyTensorStorage(32, device=device),
        sampler_device="cpu",
        batch_size=8,
        priority_key="td_error",
    )
    data = TensorDict(
        {
            "obs": torch.arange(16, device=device).float().unsqueeze(-1),
            "td_error": torch.linspace(0.1, 1.0, 16, device=device),
        },
        batch_size=[16],
        device=device,
    )

    rb.extend(data)
    sample = rb.sample()

    assert rb._sampler.device == torch.device("cpu")
    assert sample.device == device
    assert sample["index"].device == device
    assert sample["priority_weight"].device == device

    sample["td_error"] = torch.ones_like(sample["td_error"]) * 10
    rb.update_tensordict_priority(sample)
    sample = rb.sample()
    assert sample.device == device
    assert rb._sampler.device == torch.device("cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_prioritized_replay_buffer_weight_matches_cpu_formula():
    ext = pytest.importorskip("torchrl._torchrl")
    if not hasattr(ext, "CudaSumSegmentTreeFp32"):
        pytest.skip("TorchRL was not built with CUDA segment tree support")

    size = 64
    batch_size = 16
    alpha = 0.7
    beta = 0.5
    eps = 1e-8
    priorities = torch.linspace(0.1, 2.0, size)
    expected_tree_priority = (priorities + eps).pow(alpha)
    min_tree_priority = expected_tree_priority.min()

    def make_rb(device):
        device = torch.device(device)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(size, device=device),
            sampler=PrioritizedSampler(
                max_capacity=size,
                alpha=alpha,
                beta=beta,
                eps=eps,
                device=device,
            ),
            batch_size=batch_size,
            priority_key="td_error",
        )
        data = TensorDict(
            {
                "obs": torch.arange(size, device=device),
                "td_error": priorities.to(device),
            },
            batch_size=[size],
            device=device,
        )
        rb.extend(data)
        return rb

    cpu_rb = make_rb("cpu")
    cuda_rb = make_rb("cuda:0")

    for rb, device in (
        (cpu_rb, torch.device("cpu")),
        (cuda_rb, torch.device("cuda:0")),
    ):
        for _ in range(8):
            sample = rb.sample()
            index = sample["index"].to("cpu")
            expected_weight = (expected_tree_priority[index] / min_tree_priority).pow(
                -beta
            )
            torch.testing.assert_close(sample["obs"].to("cpu"), index)
            torch.testing.assert_close(sample["td_error"].to("cpu"), priorities[index])
            torch.testing.assert_close(
                sample["priority_weight"].to("cpu"), expected_weight
            )
            assert sample["index"].device == device
            assert sample["priority_weight"].device == device


@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("datatype", ["tc", "tb"])
@pytest.mark.parametrize("reduction", ["min", "max", "median", "mean"])
def test_replay_buffer_trajectories(stack, reduction, datatype):
    traj_td = TensorDict(
        {"obs": torch.randn(3, 4, 5), "actions": torch.randn(3, 4, 2)},
        batch_size=[3, 4],
    )
    rbcls = functools.partial(TensorDictReplayBuffer, priority_key="td_error")
    if datatype == "tc":
        c = make_tc(traj_td)
        rbcls = functools.partial(ReplayBuffer, storage=LazyTensorStorage(100))
        traj_td = c(**traj_td, batch_size=traj_td.batch_size)
        assert is_tensorclass(traj_td)
    elif datatype != "tb":
        raise NotImplementedError

    if stack:
        traj_td = torch.stack(list(traj_td), 0)

    rb = rbcls(
        sampler=samplers.PrioritizedSampler(
            5,
            alpha=0.7,
            beta=0.9,
            reduction=reduction,
        ),
        batch_size=3,
    )
    rb.extend(traj_td)
    if datatype == "tc":
        sampled_td, info = rb.sample(return_info=True)
        index = info["index"]
    else:
        sampled_td = rb.sample()
    if datatype == "tc":
        assert is_tensorclass(traj_td)
        return

    sampled_td.set("td_error", torch.rand(sampled_td.shape))
    if datatype == "tc":
        rb.update_priority(index, sampled_td)
        sampled_td, info = rb.sample(return_info=True)
        assert (info["priority_weight"] > 0).all()
        assert sampled_td.batch_size == torch.Size([3, 4])
    else:
        rb.update_tensordict_priority(sampled_td)
        sampled_td = rb.sample(include_info=True)
        assert (sampled_td.get("priority_weight") > 0).all()
        assert sampled_td.batch_size == torch.Size([3, 4])

    # # set back the trajectory length
    # sampled_td_filtered = sampled_td.to_tensordict().exclude(
    #     "priority_weight", "index", "td_error"
    # )
    # sampled_td_filtered.batch_size = [3, 4]


@pytest.mark.parametrize("priority_key", ["pk", "td_error"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", get_default_devices())
def test_prb(priority_key, contiguous, device):
    torch.manual_seed(0)
    np.random.seed(0)
    rb = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.9,
        priority_key=priority_key,
        storage=ListStorage(5),
        batch_size=5,
    )
    td1 = TensorDict(
        source={
            "a": torch.randn(3, 1),
            priority_key: torch.rand(3, 1) / 10,
            "_idx": torch.arange(3).view(3, 1),
        },
        batch_size=[3],
    ).to(device)

    rb.extend(td1)
    s = rb.sample()
    assert s.batch_size == torch.Size([5])
    assert (td1[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td1[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test replacement
    td2 = TensorDict(
        source={
            "a": torch.randn(5, 1),
            priority_key: torch.rand(5, 1) / 10,
            "_idx": torch.arange(5).view(5, 1),
        },
        batch_size=[5],
    ).to(device)
    rb.extend(td2)
    s = rb.sample()
    assert s.batch_size == torch.Size([5])
    assert (td2[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td2[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test strong update
    # get all indices that match first item
    idx = s.get("_idx")
    idx_match = (idx == idx[0]).nonzero()[:, 0]
    s.set_at_(
        priority_key,
        torch.ones(idx_match.numel(), 1, device=device) * 100000000,
        idx_match,
    )
    val = s.get("a")[0]

    idx0 = s.get("_idx")[0]
    rb.update_tensordict_priority(s)
    s = rb.sample()
    assert (val == s.get("a")).sum() >= 1
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))

    # test updating values of original td
    td2.set_("a", torch.ones_like(td2.get("a")))
    s = rb.sample()
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))


@pytest.mark.parametrize("alpha", [0.4, 0.7])
@pytest.mark.parametrize("max_priority_within_buffer", [False, True])
def test_prb_new_item_gets_max_priority(alpha, max_priority_within_buffer):
    """A freshly added item with no priority key must be written to the sum-tree at
    the current max priority, transformed by ``alpha`` exactly once.

    This is PER's "new experience is sampled at least once" guarantee. Regression
    test for the double-``alpha`` transform in
    :meth:`~torchrl.data.replay_buffers.samplers.PrioritizedSampler.default_priority`,
    which wrote new items at ``((p + eps) ** alpha + eps) ** alpha`` and so
    systematically under-prioritized them for ``alpha < 1``.
    """
    sampler = PrioritizedSampler(
        max_capacity=10,
        alpha=alpha,
        beta=0.9,
        max_priority_within_buffer=max_priority_within_buffer,
    )
    # index 0 receives a large TD-error priority
    sampler.update_priority(torch.tensor([0]), torch.tensor([100.0]))
    # index 1 is a fresh item written at the default (max) priority, as writers do
    sampler.mark_update(torch.tensor([1]))
    eps = sampler._eps
    expected = (100.0 + eps) ** alpha
    got = float(sampler._sum_tree[1])
    assert got == pytest.approx(expected, rel=1e-4), (got, expected)
    # the new item is exactly as sample-able as the current max item
    assert got == pytest.approx(float(sampler._sum_tree[0]), rel=1e-4)


def test_prb_within_buffer_max_priority_stays_raw():
    """In ``max_priority_within_buffer`` mode, updating the current max item must keep
    ``_max_priority`` as a raw priority (not the transformed sum-tree value), so a
    subsequently added default item still lands at the true max tree priority.
    """
    alpha = 0.6
    sampler = PrioritizedSampler(
        max_capacity=5, alpha=alpha, beta=0.9, max_priority_within_buffer=True
    )
    sampler.update_priority(torch.tensor([0, 1]), torch.tensor([100.0, 10.0]))
    # updating the current max index (0) triggers the within-buffer recompute
    sampler.update_priority(torch.tensor([0]), torch.tensor([50.0]))
    # _max_priority must hold the raw current max (50), not (50 + eps) ** alpha
    assert float(sampler._max_priority[0]) == pytest.approx(50.0, rel=1e-3)
    # a fresh default item then lands at (50 + eps) ** alpha
    sampler.mark_update(torch.tensor([2]))
    expected = (50.0 + sampler._eps) ** alpha
    assert float(sampler._sum_tree[2]) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("max_priority_within_buffer", [False, True])
def test_prb_alpha_zero_keeps_raw_max(max_priority_within_buffer):
    """With ``alpha == 0`` every tree entry is ``(p + eps) ** 0 == 1``, so the
    within-buffer recompute cannot recover raw priorities from the tree. It must
    keep the previously tracked raw max rather than store the transformed value
    (1.0) into ``_max_priority``."""
    sampler = PrioritizedSampler(
        max_capacity=5,
        alpha=0.0,
        beta=0.9,
        max_priority_within_buffer=max_priority_within_buffer,
    )
    sampler.update_priority(torch.tensor([0, 1]), torch.tensor([100.0, 10.0]))
    # updating the tracked max index triggers the within-buffer recompute
    sampler.update_priority(torch.tensor([0]), torch.tensor([50.0]))
    assert float(sampler._max_priority[0]) == pytest.approx(100.0)
    assert float(sampler.default_priority) == pytest.approx(100.0)


def test_prb_alpha_setter_validates_and_warns():
    """In ``max_priority_within_buffer`` mode the max-priority recomputation
    inverts sum-tree values with the current ``alpha``, so changing ``alpha``
    once priorities have been written must warn (once per sampler); the setter
    must also reject negative values like ``__init__`` does. Without
    ``max_priority_within_buffer`` the setter stays silent so that alpha
    annealing (e.g. via ``LinearScheduler``) is warning-free."""
    sampler = PrioritizedSampler(
        max_capacity=5, alpha=0.7, beta=0.9, max_priority_within_buffer=True
    )
    # empty sampler: changing alpha is safe and must not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler.alpha = 0.6
    sampler.update_priority(torch.tensor([0]), torch.tensor([100.0]))
    # setting the same value is a no-op and must not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler.alpha = 0.6
    with pytest.warns(UserWarning, match="does not re-transform"):
        sampler.alpha = 0.5
    # the warning is emitted once per sampler
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler.alpha = 0.4
    with pytest.raises(ValueError, match="alpha must be greater or equal"):
        sampler.alpha = -1.0
    # without max_priority_within_buffer, annealing alpha is warning-free
    sampler = PrioritizedSampler(max_capacity=5, alpha=0.7, beta=0.9)
    sampler.update_priority(torch.tensor([0]), torch.tensor([100.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler.alpha = 0.5


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
