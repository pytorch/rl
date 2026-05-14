# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pytest
import torch
from _rb_common import _has_snapshot, TORCH_VERSION
from packaging import version
from tensordict import TensorDict

from torchrl._utils import _replace_last
from torchrl.collectors.utils import split_trajectories
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    PrioritizedSliceSampler,
    RandomSampler,
    Sampler,
    SamplerWithoutReplacement,
    SliceSampler,
    SliceSamplerWithoutReplacement,
    StalenessAwareSampler,
)
from torchrl.data.replay_buffers.scheduler import (
    LinearScheduler,
    SchedulerList,
    StepScheduler,
)
from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
)
from torchrl.modules import GRUModule, set_recurrent_mode
from torchrl.testing import get_default_devices


@pytest.mark.parametrize("size", [10, 15, 20])
@pytest.mark.parametrize("samples", [5, 9, 11, 14, 16])
@pytest.mark.parametrize("drop_last", [True, False])
def test_samplerwithoutrep(size, samples, drop_last):
    torch.manual_seed(0)
    storage = ListStorage(size)
    storage.set(range(size), range(size))
    assert len(storage) == size
    sampler = SamplerWithoutReplacement(drop_last=drop_last)
    visited = False
    for _ in range(10):
        _n_left = (
            sampler._sample_list.numel() if sampler._sample_list is not None else size
        )
        if samples > size and drop_last:
            with pytest.raises(
                ValueError,
                match=r"The batch size .* is greater than the storage capacity",
            ):
                idx, _ = sampler.sample(storage, samples)
            break
        idx, _ = sampler.sample(storage, samples)
        if drop_last or _n_left >= samples:
            assert idx.numel() == samples
            assert idx.unique().numel() == idx.numel()
        else:
            assert idx.numel() == _n_left
            visited = True
    if not drop_last and (size % samples > 0):
        assert visited
    else:
        assert not visited


class TestSamplers:
    @pytest.mark.parametrize(
        "backend", ["torch"] + (["torchsnapshot"] if _has_snapshot else [])
    )
    def test_sampler_without_rep_state_dict(self, backend):
        os.environ["CKPT_BACKEND"] = backend
        torch.manual_seed(0)

        n_samples = 3
        buffer_size = 100
        storage_in = LazyTensorStorage(buffer_size, device="cpu")
        storage_out = LazyTensorStorage(buffer_size, device="cpu")

        replay_buffer = TensorDictReplayBuffer(
            storage=storage_in,
            sampler=SamplerWithoutReplacement(),
        )
        # fill replay buffer with random data
        transition = TensorDict(
            {
                "observation": torch.ones(1, 4),
                "action": torch.ones(1, 2),
                "reward": torch.ones(1, 1),
                "dones": torch.ones(1, 1),
                "next": {"observation": torch.ones(1, 4)},
            },
            batch_size=1,
        )
        for _ in range(n_samples):
            replay_buffer.extend(transition.clone())
        for _ in range(n_samples):
            s = replay_buffer.sample(batch_size=1)
            assert (s.exclude("index") == 1).all()

        replay_buffer.extend(torch.zeros_like(transition))

        state_dict = replay_buffer.state_dict()

        new_replay_buffer = TensorDictReplayBuffer(
            storage=storage_out,
            batch_size=state_dict["_batch_size"],
            sampler=SamplerWithoutReplacement(),
        )

        new_replay_buffer.load_state_dict(state_dict)
        s = new_replay_buffer.sample(batch_size=1)
        assert (s.exclude("index") == 0).all()

    def test_sampler_without_rep_dumps_loads(self, tmpdir):
        d0 = tmpdir + "/save0"
        d1 = tmpdir + "/save1"
        d2 = tmpdir + "/dump"
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(max_size=100, scratch_dir=d0, device="cpu"),
            sampler=SamplerWithoutReplacement(drop_last=True),
            batch_size=8,
        )
        replay_buffer2 = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(max_size=100, scratch_dir=d1, device="cpu"),
            sampler=SamplerWithoutReplacement(drop_last=True),
            batch_size=8,
        )
        td = TensorDict(
            {"a": torch.arange(0, 27), ("b", "c"): torch.arange(1, 28)}, batch_size=[27]
        )
        replay_buffer.extend(td)
        for _ in replay_buffer:
            break
        replay_buffer.dumps(d2)
        replay_buffer2.loads(d2)
        assert (
            replay_buffer.sampler._sample_list == replay_buffer2.sampler._sample_list
        ).all()
        s = replay_buffer2.sample(3)
        assert (s["a"] == s["b", "c"] - 1).all()

    @pytest.mark.parametrize("drop_last", [False, True])
    def test_sampler_without_replacement_cap_prefetch(self, drop_last):
        torch.manual_seed(0)
        data = TensorDict({"a": torch.arange(11)}, batch_size=[11])
        rb = ReplayBuffer(
            storage=LazyTensorStorage(11),
            sampler=SamplerWithoutReplacement(drop_last=drop_last),
            batch_size=2,
            prefetch=3,
        )
        rb.extend(data)

        for _ in range(100):
            s = set()
            for i, d in enumerate(rb):
                assert i <= (4 + int(not drop_last)), i
                s = s.union(set(d["a"].tolist()))
            assert i == (4 + int(not drop_last)), i
            if drop_last:
                assert s != set(range(11))
            else:
                assert s == set(range(11))

    @pytest.mark.parametrize(
        "batch_size,num_slices,slice_len,prioritized",
        [
            [100, 20, None, True],
            [100, 20, None, False],
            [120, 30, None, False],
            [100, None, 5, False],
            [120, None, 4, False],
            [101, None, 101, False],
        ],
    )
    @pytest.mark.parametrize("episode_key", ["episode", ("some", "episode")])
    @pytest.mark.parametrize("done_key", ["done", ("some", "done")])
    @pytest.mark.parametrize("match_episode", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_slice_sampler(
        self,
        batch_size,
        num_slices,
        slice_len,
        prioritized,
        episode_key,
        done_key,
        match_episode,
        device,
    ):
        torch.manual_seed(0)
        storage = LazyMemmapStorage(100)
        episode = torch.zeros(100, dtype=torch.int, device=device)
        episode[:30] = 1
        episode[30:55] = 2
        episode[55:70] = 3
        episode[70:] = 4
        steps = torch.cat(
            [torch.arange(30), torch.arange(25), torch.arange(15), torch.arange(30)], 0
        )

        done = torch.zeros(100, 1, dtype=torch.bool)
        done[torch.tensor([29, 54, 69, 99])] = 1

        data = TensorDict(
            {
                # we only use episode_key if we want the sampler to access it
                episode_key if match_episode else "whatever_episode": episode,
                "another_episode": episode,
                "obs": torch.randn((3, 4, 5)).expand(100, 3, 4, 5),
                "act": torch.randn((20,)).expand(100, 20),
                "steps": steps,
                "count": torch.arange(100),
                "other": torch.randn((20, 50)).expand(100, 20, 50),
                done_key: done,
                _replace_last(done_key, "terminated"): done,
            },
            [100],
            device=device,
        )
        storage.set(range(100), data)
        if slice_len is not None and slice_len > 15:
            # we may have to sample trajs shorter than slice_len
            strict_length = False
        else:
            strict_length = True

        if prioritized:
            num_steps = data.shape[0]
            sampler = PrioritizedSliceSampler(
                max_capacity=num_steps,
                alpha=0.7,
                beta=0.9,
                num_slices=num_slices,
                traj_key=episode_key,
                end_key=done_key,
                slice_len=slice_len,
                strict_length=strict_length,
                truncated_key=_replace_last(done_key, "truncated"),
            )
            index = torch.arange(0, num_steps, 1)
            sampler.extend(index)
            sampler.update_priority(index, 1)
        else:
            sampler = SliceSampler(
                num_slices=num_slices,
                traj_key=episode_key,
                end_key=done_key,
                slice_len=slice_len,
                strict_length=strict_length,
                truncated_key=_replace_last(done_key, "truncated"),
            )
        if slice_len is not None:
            num_slices = batch_size // slice_len
        trajs_unique_id = set()
        too_short = False
        count_unique = set()
        for _ in range(50):
            index, info = sampler.sample(storage, batch_size=batch_size)
            samples = storage._storage[index]
            if strict_length:
                # check that trajs are ok
                samples = samples.view(num_slices, -1)

                unique_another_episode = (
                    samples["another_episode"].unique(dim=1).squeeze()
                )
                assert unique_another_episode.shape == torch.Size([num_slices]), (
                    num_slices,
                    samples,
                )
                assert (
                    samples["steps"][..., 1:] - 1 == samples["steps"][..., :-1]
                ).all()
            if isinstance(index, tuple):
                index_numel = index[0].numel()
            else:
                index_numel = index.numel()

            too_short = too_short or index_numel < batch_size
            trajs_unique_id = trajs_unique_id.union(
                samples["another_episode"].view(-1).tolist()
            )
            count_unique = count_unique.union(samples.get("count").view(-1).tolist())

            truncated = info[_replace_last(done_key, "truncated")]
            terminated = info[_replace_last(done_key, "terminated")]
            assert (truncated | terminated).view(num_slices, -1)[:, -1].all()
            assert (
                terminated
                == samples[_replace_last(done_key, "terminated")].view_as(terminated)
            ).all()
            done = info[done_key]
            assert done.view(num_slices, -1)[:, -1].all()

            if len(count_unique) == 100:
                # all items have been sampled
                break
        else:
            raise AssertionError(
                f"Not all items can be sampled: {set(range(100)) - count_unique} are missing"
            )

        if strict_length:
            assert not too_short
        else:
            assert too_short

        assert len(trajs_unique_id) == 4

    @pytest.mark.parametrize("sampler", [SliceSampler, SliceSamplerWithoutReplacement])
    def test_slice_sampler_at_capacity(self, sampler):
        torch.manual_seed(0)

        trajectory0 = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        trajectory1 = torch.arange(2).repeat_interleave(6)
        trajectory = torch.stack([trajectory0, trajectory1], 0)

        td = TensorDict(
            {"trajectory": trajectory, "steps": torch.arange(12).expand(2, 12)}, [2, 12]
        )

        rb = ReplayBuffer(
            sampler=sampler(traj_key="trajectory", num_slices=2),
            storage=LazyTensorStorage(20, ndim=2),
            batch_size=6,
        )

        rb.extend(td)

        for s in rb:
            if (s["steps"] == 9).any():
                break
        else:
            raise AssertionError

    def test_slice_sampler_errors(self):
        device = "cpu"
        batch_size, num_slices = 100, 20

        episode = torch.zeros(100, dtype=torch.int, device=device)
        episode[:30] = 1
        episode[30:55] = 2
        episode[55:70] = 3
        episode[70:] = 4
        steps = torch.cat(
            [torch.arange(30), torch.arange(25), torch.arange(15), torch.arange(30)], 0
        )

        done = torch.zeros(100, 1, dtype=torch.bool)
        done[torch.tensor([29, 54, 69])] = 1

        data = TensorDict(
            {
                # we only use episode_key if we want the sampler to access it
                "episode": episode,
                "another_episode": episode,
                "obs": torch.randn((3, 4, 5)).expand(100, 3, 4, 5),
                "act": torch.randn((20,)).expand(100, 20),
                "steps": steps,
                "other": torch.randn((20, 50)).expand(100, 20, 50),
                ("next", "done"): done,
            },
            [100],
            device=device,
        )

        data_wrong_done = data.clone(False)
        data_wrong_done.rename_key_("episode", "_")
        data_wrong_done["next", "done"] = done.unsqueeze(1).expand(100, 5, 1)
        storage = LazyMemmapStorage(100)
        storage.set(range(100), data_wrong_done)
        sampler = SliceSampler(num_slices=num_slices)
        with pytest.raises(
            RuntimeError,
            match="Expected the end-of-trajectory signal to be 1-dimensional",
        ):
            index, _ = sampler.sample(storage, batch_size=batch_size)

        storage = ListStorage(100)
        storage.set(range(100), data)
        sampler = SliceSampler(num_slices=num_slices)
        with pytest.raises(
            RuntimeError,
            match="Could not get a tensordict out of the storage, which is required for SliceSampler to compute the trajectories.",
        ):
            index, _ = sampler.sample(storage, batch_size=batch_size)

    @pytest.mark.parametrize("batch_size,num_slices", [[20, 4], [4, 2]])
    @pytest.mark.parametrize("episode_key", ["episode", ("some", "episode")])
    @pytest.mark.parametrize("done_key", ["done", ("some", "done")])
    @pytest.mark.parametrize("match_episode", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_slice_sampler_without_replacement(
        self,
        batch_size,
        num_slices,
        episode_key,
        done_key,
        match_episode,
        device,
    ):
        torch.manual_seed(0)
        storage = LazyMemmapStorage(100)
        episode = torch.zeros(100, dtype=torch.int, device=device)
        steps = []
        done = torch.zeros(100, 1, dtype=torch.bool)
        for i in range(0, 100, 5):
            episode[i : i + 5] = i // 5
            steps.append(torch.arange(5))
            done[i + 4] = 1
        steps = torch.cat(steps)

        data = TensorDict(
            {
                # we only use episode_key if we want the sampler to access it
                episode_key if match_episode else "whatever_episode": episode,
                "another_episode": episode,
                "obs": torch.randn((3, 4, 5)).expand(100, 3, 4, 5),
                "act": torch.randn((20,)).expand(100, 20),
                "steps": steps,
                "other": torch.randn((20, 50)).expand(100, 20, 50),
                done_key: done,
            },
            [100],
            device=device,
        )
        storage.set(range(100), data)
        sampler = SliceSamplerWithoutReplacement(
            num_slices=num_slices, traj_key=episode_key, end_key=done_key
        )
        trajs_unique_id = set()
        for i in range(5):
            index, info = sampler.sample(storage, batch_size=batch_size)
            samples = storage._storage[index]

            # check that trajs are ok
            samples = samples.view(num_slices, -1)
            assert samples["another_episode"].unique(
                dim=1
            ).squeeze().shape == torch.Size([num_slices])
            assert (samples["steps"][..., 1:] - 1 == samples["steps"][..., :-1]).all()
            cur_episodes = samples["another_episode"].view(-1).tolist()
            for ep in cur_episodes:
                assert ep not in trajs_unique_id, i
            trajs_unique_id = trajs_unique_id.union(
                cur_episodes,
            )
        done_recon = info[("next", "truncated")] | info[("next", "terminated")]
        assert done_recon.view(num_slices, -1)[:, -1].all()
        done = info[("next", "done")]
        assert done.view(num_slices, -1)[:, -1].all()

    def test_slice_sampler_left_right(self):
        torch.manual_seed(0)
        data = TensorDict(
            {"obs": torch.arange(1, 11).repeat(10), "eps": torch.arange(100) // 10 + 1},
            [100],
        )

        for N in (2, 4):
            rb = TensorDictReplayBuffer(
                sampler=SliceSampler(num_slices=10, traj_key="eps", span=(N, N)),
                batch_size=50,
                storage=LazyMemmapStorage(100),
            )
            rb.extend(data)

            for _ in range(10):
                sample = rb.sample()
                sample = split_trajectories(sample)
                assert (sample["next", "truncated"].squeeze(-1).sum(-1) == 1).all()
                assert ((sample["obs"] == 0).sum(-1) <= N).all(), sample["obs"]
                assert ((sample["eps"] == 0).sum(-1) <= N).all()
                for i in range(sample.shape[0]):
                    curr_eps = sample[i]["eps"]
                    curr_eps = curr_eps[curr_eps != 0]
                    assert curr_eps.unique().numel() == 1

    def test_slice_sampler_left_right_ndim(self):
        torch.manual_seed(0)
        data = TensorDict(
            {"obs": torch.arange(1, 11).repeat(12), "eps": torch.arange(120) // 10 + 1},
            [120],
        )
        data = data.reshape(4, 30)

        for N in (2, 4):
            rb = TensorDictReplayBuffer(
                sampler=SliceSampler(num_slices=10, traj_key="eps", span=(N, N)),
                batch_size=50,
                storage=LazyMemmapStorage(100, ndim=2),
            )
            rb.extend(data)

            for _ in range(10):
                sample = rb.sample()
                sample = split_trajectories(sample)
                assert (sample["next", "truncated"].squeeze(-1).sum(-1) <= 1).all()
                assert ((sample["obs"] == 0).sum(-1) <= N).all(), sample["obs"]
                assert ((sample["eps"] == 0).sum(-1) <= N).all()
                for i in range(sample.shape[0]):
                    curr_eps = sample[i]["eps"]
                    curr_eps = curr_eps[curr_eps != 0]
                    assert curr_eps.unique().numel() == 1

    def test_slice_sampler_strictlength(self):
        torch.manual_seed(0)

        data = TensorDict(
            {
                "traj": torch.cat(
                    [
                        torch.ones(2, dtype=torch.int),
                        torch.zeros(10, dtype=torch.int),
                    ],
                    dim=0,
                ),
                "x": torch.arange(12),
            },
            [12],
        )

        buffer = ReplayBuffer(
            storage=LazyTensorStorage(12),
            sampler=SliceSampler(num_slices=2, strict_length=True, traj_key="traj"),
            batch_size=8,
        )
        buffer.extend(data)

        for _ in range(50):
            sample = buffer.sample()
            assert sample.shape == torch.Size([8])
            assert (sample["traj"] == 0).all()

        buffer = ReplayBuffer(
            storage=LazyTensorStorage(12),
            sampler=SliceSampler(num_slices=2, strict_length=False, traj_key="traj"),
            batch_size=8,
        )
        buffer.extend(data)

        for _ in range(50):
            sample = buffer.sample()
            if sample.shape == torch.Size([6]):
                assert (sample["traj"] != 0).any()
            else:
                assert len(sample["traj"].unique()) == 1

    # ------------------------------------------------------------------
    # traj_key auto-detection tests
    # ------------------------------------------------------------------

    def test_slice_sampler_auto_traj_key_collector_ids(self):
        """Auto-detection should prefer ("collector", "traj_ids") over "episode"."""
        torch.manual_seed(0)
        # Build data with both keys present; sampler should pick collector key
        # and warn that this changes the pre-0.13 default.
        traj_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int)
        data = TensorDict(
            {
                ("collector", "traj_ids"): traj_ids,
                "episode": torch.zeros(8, dtype=torch.int),  # wrong, should be ignored
                "obs": torch.arange(8).float(),
            },
            batch_size=[8],
        )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(8),
            sampler=SliceSampler(num_slices=2),
            batch_size=6,
        )
        rb.extend(data)
        # Force resolution — with both keys present we must see a FutureWarning.
        with pytest.warns(FutureWarning, match="auto-detected"):
            sample = rb.sample()
        assert rb.sampler.traj_key == ("collector", "traj_ids")
        assert rb.sampler._fetch_traj is True
        assert rb.sampler._traj_key_auto is False
        # Each slice should come from a single trajectory
        sample_reshaped = sample.reshape(2, 3)
        for i in range(2):
            traj_vals = sample_reshaped[i][("collector", "traj_ids")]
            assert traj_vals.unique().numel() == 1

    def test_slice_sampler_auto_traj_key_no_warning_single_key(self):
        """No FutureWarning when only one of the two candidate keys is present."""
        torch.manual_seed(0)
        traj_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int)
        data = TensorDict(
            {
                ("collector", "traj_ids"): traj_ids,
                "obs": torch.arange(8).float(),
            },
            batch_size=[8],
        )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(8),
            sampler=SliceSampler(num_slices=2),
            batch_size=6,
        )
        rb.extend(data)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            rb.sample()
        assert rb.sampler.traj_key == ("collector", "traj_ids")

    def test_slice_sampler_auto_traj_key_episode(self):
        """Auto-detection falls back to 'episode' when collector key is absent."""
        torch.manual_seed(0)
        traj_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int)
        data = TensorDict(
            {
                "episode": traj_ids,
                "obs": torch.arange(8).float(),
            },
            batch_size=[8],
        )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(8),
            sampler=SliceSampler(num_slices=2),
            batch_size=6,
        )
        rb.extend(data)
        rb.sample()
        assert rb.sampler.traj_key == "episode"
        assert rb.sampler._fetch_traj is True

    def test_slice_sampler_auto_traj_key_fallback_to_done(self):
        """Auto-detection falls back to end_key reconstruction when no traj key."""
        torch.manual_seed(0)
        done = torch.zeros(9, 1, dtype=torch.bool)
        done[[2, 5, 8]] = True
        data = TensorDict(
            {
                ("next", "done"): done,
                ("next", "truncated"): done,
                ("next", "terminated"): done,
                "obs": torch.arange(9).float(),
            },
            batch_size=[9],
        )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(9),
            sampler=SliceSampler(num_slices=3),
            batch_size=9,
        )
        rb.extend(data)
        rb.sample()
        assert rb.sampler._fetch_traj is False

    def test_slice_sampler_explicit_traj_key_no_auto(self):
        """Explicit traj_key should bypass auto-detection entirely."""
        torch.manual_seed(0)
        traj_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int)
        data = TensorDict(
            {
                "my_traj": traj_ids,
                ("collector", "traj_ids"): torch.zeros(8, dtype=torch.int),
                "obs": torch.arange(8).float(),
            },
            batch_size=[8],
        )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(8),
            sampler=SliceSampler(num_slices=2, traj_key="my_traj"),
            batch_size=6,
        )
        rb.extend(data)
        rb.sample()
        assert rb.sampler.traj_key == "my_traj"
        assert getattr(rb.sampler, "_traj_key_auto", False) is False

    # ------------------------------------------------------------------
    # mask / lengths tests (strict_length=False)
    # ------------------------------------------------------------------

    def _make_rb_with_short_trajs(self, traj_lengths, slice_len, num_slices):
        """Helper: build a TensorDictReplayBuffer with trajectories of given lengths."""
        parts = []
        for t_id, length in enumerate(traj_lengths):
            is_init = torch.zeros(length, 1, dtype=torch.bool)
            is_init[0] = True  # episode reset at the first step of each trajectory
            parts.append(
                TensorDict(
                    {
                        "traj": torch.full((length,), t_id, dtype=torch.int),
                        "obs": torch.arange(length).float(),
                        "is_init": is_init,
                    },
                    batch_size=[length],
                )
            )
        data = torch.cat(parts)
        total = sum(traj_lengths)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(total),
            sampler=SliceSampler(
                slice_len=slice_len,
                traj_key="traj",
                strict_length=False,
                pad_output=True,
            ),
            batch_size=num_slices * slice_len,
        )
        rb.extend(data)
        return rb

    def test_slice_sampler_mask_present_when_short_trajs(self):
        """mask appears in output when short trajectories force padding."""
        torch.manual_seed(0)
        rb = self._make_rb_with_short_trajs(
            traj_lengths=[3, 6, 2], slice_len=5, num_slices=3
        )
        sample = rb.sample()
        assert ("collector", "mask") in sample.keys(True)

    def test_slice_sampler_mask_shape_dtype(self):
        """mask is bool with shape [B*T] (matches batch shape, no trailing 1)."""
        torch.manual_seed(0)
        B, T = 4, 6
        rb = self._make_rb_with_short_trajs(
            traj_lengths=[2, 5, 3, 4], slice_len=T, num_slices=B
        )
        sample = rb.sample()
        mask = sample[("collector", "mask")]
        assert mask.shape == torch.Size([B * T])
        assert mask.dtype == torch.bool
        # mask must match the leading batch dim so trainer code can index
        # batch[batch.get(("collector", "mask"))] without broadcasting tricks.
        assert mask.shape[0] == sample.batch_size[0]

    def test_slice_sampler_mask_correctness(self):
        """mask rows are contiguous: True prefix followed by False suffix."""
        torch.manual_seed(0)
        B, T = 6, 8
        rb = self._make_rb_with_short_trajs(
            traj_lengths=[3, 8, 2, 7, 1, 5], slice_len=T, num_slices=B
        )
        for _ in range(20):
            sample = rb.sample()
            mask = sample[("collector", "mask")].reshape(B, T)
            # derive lengths from the mask itself
            lengths = mask.sum(-1)  # [B]
            for i in range(B):
                length = lengths[i].item()
                assert length >= 1
                assert length <= T
                assert mask[
                    i, :length
                ].all(), f"slice {i}: first {length} steps should be True"
                assert not mask[
                    i, length:
                ].any(), f"slice {i}: steps after {length} should be False"

    def test_slice_sampler_mask_padded_obs_is_valid(self):
        """Padded positions repeat the last real index — obs values must be finite."""
        torch.manual_seed(0)
        rb = self._make_rb_with_short_trajs(
            traj_lengths=[2, 6, 3], slice_len=5, num_slices=3
        )
        sample = rb.sample()
        assert torch.isfinite(sample["obs"]).all()

    def test_slice_sampler_strict_length_no_mask(self):
        """With pad_output=False, no mask is emitted regardless of strict_length."""
        torch.manual_seed(0)
        data = TensorDict(
            {
                "traj": torch.cat(
                    [torch.zeros(6, dtype=torch.int), torch.ones(6, dtype=torch.int)]
                ),
                "obs": torch.arange(12).float(),
            },
            batch_size=[12],
        )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(12),
            sampler=SliceSampler(
                slice_len=4, traj_key="traj", strict_length=True, pad_output=False
            ),
            batch_size=8,
        )
        rb.extend(data)
        sample = rb.sample()
        assert ("collector", "mask") not in sample.keys(True)

    def test_slice_sampler_pad_output_strict_length_raises(self):
        """pad_output=True + strict_length=True is rejected at construction."""
        with pytest.raises(ValueError, match="pad_output=True is incompatible"):
            SliceSampler(
                slice_len=4, traj_key="traj", strict_length=True, pad_output=True
            )

    def test_slice_sampler_pad_output_marks_slice_starts(self):
        """pad_output=True writes is_init=True at every slice start.

        This is what lets a recurrent policy in `set_recurrent_mode("recurrent")`
        consume the flat [B*T] sample directly: the RNN splits on `is_init`
        and uses each slice's stored hidden state at position 0.
        """
        torch.manual_seed(0)
        B, T = 4, 8
        rb = self._make_rb_with_short_trajs(
            traj_lengths=[3, 8, 2, 7, 1, 5], slice_len=T, num_slices=B
        )
        for _ in range(10):
            sample = rb.sample()
            is_init = sample["is_init"].reshape(B, T)
            # Position 0 of every slice must be True regardless of where the
            # slice landed within its source trajectory.
            assert is_init[:, 0].all(), "every slice must start with is_init=True"

    def test_slice_sampler_marks_slice_starts_no_pad(self):
        """Default (no pad_output) flow: is_init=True at every slice start.

        This is the workflow most users will hit: trajectories are written
        end-to-end into the buffer, the sampler returns concatenated
        variable-length slices, and the RNN splits on `is_init`. No mask, no
        padding involved.
        """
        torch.manual_seed(0)
        traj_lengths = [3, 8, 2, 7, 5]
        parts = []
        for t_id, length in enumerate(traj_lengths):
            init = torch.zeros(length, 1, dtype=torch.bool)
            init[0] = True
            parts.append(
                TensorDict(
                    {
                        "traj": torch.full((length,), t_id, dtype=torch.int),
                        "is_init": init,
                    },
                    batch_size=[length],
                )
            )
        data = torch.cat(parts)
        B = 4
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(data.numel()),
            sampler=SliceSampler(num_slices=B, traj_key="traj", strict_length=False),
            batch_size=B * 6,
        )
        rb.extend(data)
        for _ in range(10):
            sample = rb.sample()
            assert "is_init" in sample.keys(True)
            is_init = sample["is_init"].squeeze(-1)
            trunc = sample[("next", "truncated")].squeeze(-1)
            # Slice 0 always starts at position 0.
            assert is_init[0].item(), "first slice must start with is_init=True"
            # Every position right after a truncated flag must be is_init=True
            # (next slice's start). The last truncated marks the end of the
            # batch; nothing follows it.
            slice_ends = trunc.nonzero().squeeze(-1).tolist()
            for end in slice_ends[:-1]:
                assert is_init[
                    end + 1
                ].item(), f"slice starting at index {end + 1} missing is_init=True"

    def test_slice_sampler_pad_output_no_is_init_no_marker(self):
        """Without is_init in the storage we don't introduce one out of thin air."""
        torch.manual_seed(0)
        # Build a buffer *without* is_init.
        data = TensorDict(
            {
                "traj": torch.cat(
                    [
                        torch.full((3,), 0, dtype=torch.int),
                        torch.full((6,), 1, dtype=torch.int),
                        torch.full((2,), 2, dtype=torch.int),
                    ]
                ),
                "obs": torch.arange(11).float(),
            },
            batch_size=[11],
        )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(11),
            sampler=SliceSampler(
                slice_len=5, traj_key="traj", strict_length=False, pad_output=True
            ),
            batch_size=15,
        )
        rb.extend(data)
        sample = rb.sample()
        # is_init must not appear if it wasn't in the storage
        assert "is_init" not in sample.keys(True)

    def test_slice_sampler_flat_sample_matches_batched_recurrent_module(self):
        """A flat padded sample must match an explicit [B, T] recurrent call."""
        torch.manual_seed(0)
        B, T = 4, 5
        input_size, hidden_size = 3, 7
        parts = []
        for traj_id, length in enumerate([11, 9, 10, 12]):
            is_init = torch.zeros(length, 1, dtype=torch.bool)
            is_init[0] = True
            parts.append(
                TensorDict(
                    {
                        "traj": torch.full((length,), traj_id, dtype=torch.int),
                        "embed": torch.randn(length, input_size),
                        "recurrent_state": torch.randn(length, 1, hidden_size),
                        "is_init": is_init,
                    },
                    batch_size=[length],
                )
            )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(sum(part.shape[0] for part in parts)),
            sampler=SliceSampler(
                slice_len=T,
                traj_key="traj",
                strict_length=False,
                pad_output=True,
            ),
            batch_size=B * T,
        )
        rb.extend(torch.cat(parts))
        sample = rb.sample()
        assert sample["is_init"].reshape(B, T)[:, 0].all()

        gru = GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            in_keys=["embed", "recurrent_state", "is_init"],
            out_keys=["features", ("next", "recurrent_state")],
        )
        with set_recurrent_mode("recurrent"):
            flat_out = gru(sample.clone())
            batched_out = gru(sample.clone().reshape(B, T))

        torch.testing.assert_close(
            flat_out["features"].reshape(B, T, hidden_size), batched_out["features"]
        )
        torch.testing.assert_close(
            flat_out[("next", "recurrent_state")].reshape(B, T, 1, hidden_size),
            batched_out[("next", "recurrent_state")],
        )

    def test_slice_sampler_mask_all_long_trajs_no_mask(self):
        """When all trajs >= slice_len, pad_output=True still emits no mask (nothing to pad)."""
        torch.manual_seed(0)
        data = TensorDict(
            {
                "traj": torch.cat(
                    [torch.zeros(8, dtype=torch.int), torch.ones(8, dtype=torch.int)]
                ),
                "obs": torch.arange(16).float(),
            },
            batch_size=[16],
        )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(16),
            sampler=SliceSampler(
                slice_len=4, traj_key="traj", strict_length=False, pad_output=True
            ),
            batch_size=8,
        )
        rb.extend(data)
        sample = rb.sample()
        # No short trajectories → no padding needed → no mask emitted
        assert ("collector", "mask") not in sample.keys(True)

    def test_slice_sampler_truncated_marks_last_real_step(self):
        """truncated flag should sit at the last *real* timestep, not the padded end."""
        torch.manual_seed(0)
        B, T = 4, 6
        rb = self._make_rb_with_short_trajs(
            traj_lengths=[2, 5, 3, 4], slice_len=T, num_slices=B
        )
        sample = rb.sample()
        mask = sample[("collector", "mask")].reshape(B, T)
        lengths = mask.sum(-1)  # [B] — derived from mask
        trunc = sample[("next", "truncated")].reshape(B, T)
        for i in range(B):
            length = lengths[i].item()
            # truncated should be True exactly at position length-1
            assert trunc[
                i, length - 1
            ].item(), f"slice {i}: truncated missing at last real step"
            # no truncated flag in padded region
            if length < T:
                assert not trunc[
                    i, length:
                ].any(), f"slice {i}: spurious truncated in padding"

    @pytest.mark.parametrize("ndim", [1, 2])
    @pytest.mark.parametrize("strict_length", [True, False])
    @pytest.mark.parametrize("circ", [False, True])
    @pytest.mark.parametrize("at_capacity", [False, True])
    def test_slice_sampler_prioritized(self, ndim, strict_length, circ, at_capacity):
        torch.manual_seed(0)
        out = []
        for t in range(5):
            length = (t + 1) * 5
            done = torch.zeros(length, 1, dtype=torch.bool)
            done[-1] = 1
            priority = 10 if t == 0 else 1
            traj = TensorDict(
                {
                    "traj": torch.full((length,), t),
                    "step_count": torch.arange(length),
                    "done": done,
                    "priority": torch.full((length,), priority),
                },
                batch_size=length,
            )
            out.append(traj)
        data = torch.cat(out)
        if ndim == 2:
            data = torch.stack([data, data])
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(data.numel() - at_capacity, ndim=ndim),
            sampler=PrioritizedSliceSampler(
                max_capacity=data.numel() - at_capacity,
                alpha=1.0,
                beta=1.0,
                end_key="done",
                slice_len=10,
                strict_length=strict_length,
                cache_values=True,
            ),
            batch_size=50,
        )
        if not circ:
            # Simplest case: the buffer is full but no overlap
            index = rb.extend(data, update_priority=False)
        else:
            # The buffer is 2/3 -> 1/3 overlapping
            rb.extend(data[..., : data.shape[-1] // 3], update_priority=False)
            index = rb.extend(data, update_priority=False)
        rb.update_priority(index, data["priority"])
        samples = []
        found_shorter_batch = False
        for _ in range(100):
            samples.append(rb.sample())
            if samples[-1].numel() < 50:
                found_shorter_batch = True
        samples = torch.cat(samples)
        if strict_length:
            assert not found_shorter_batch
        else:
            assert found_shorter_batch
        # the first trajectory has a very high priority, but should only appear
        # if strict_length=False.
        if strict_length:
            assert (samples["traj"] != 0).all(), samples["traj"].unique()
        else:
            assert (samples["traj"] == 0).any()
            # Check that all samples of the first traj contain all elements (since it's too short to fulfill 10 elts)
            sc = samples[samples["traj"] == 0]["step_count"]
            assert (sc == 1).sum() == (sc == 2).sum()
            assert (sc == 1).sum() == (sc == 4).sum()
        assert rb.sampler._cache
        rb.extend(data, update_priority=False)
        assert not rb.sampler._cache

    @pytest.mark.parametrize("ndim", [1, 2])
    @pytest.mark.parametrize("strict_length", [True, False])
    @pytest.mark.parametrize("circ", [False, True])
    @pytest.mark.parametrize(
        "span", [False, [False, False], [False, True], 3, [False, 3]]
    )
    def test_slice_sampler_prioritized_span(self, ndim, strict_length, circ, span):
        torch.manual_seed(0)
        out = []
        # 5 trajs of length 3, 6, 9, 12 and 15
        for t in range(5):
            length = (t + 1) * 3
            done = torch.zeros(length, 1, dtype=torch.bool)
            done[-1] = 1
            priority = 1
            traj = TensorDict(
                {
                    "traj": torch.full((length,), t),
                    "step_count": torch.arange(length),
                    "done": done,
                    "priority": torch.full((length,), priority),
                },
                batch_size=length,
            )
            out.append(traj)
        data = torch.cat(out)
        if ndim == 2:
            data = torch.stack([data, data])
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(data.numel(), ndim=ndim),
            sampler=PrioritizedSliceSampler(
                max_capacity=data.numel(),
                alpha=1.0,
                beta=1.0,
                end_key="done",
                slice_len=5,
                strict_length=strict_length,
                cache_values=True,
                span=span,
            ),
            batch_size=5,
        )
        if not circ:
            # Simplest case: the buffer is full but no overlap
            index = rb.extend(data)
        else:
            # The buffer is 2/3 -> 1/3 overlapping
            rb.extend(data[..., : data.shape[-1] // 3])
            index = rb.extend(data)
        rb.update_priority(index, data["priority"])
        found_traj_0 = False
        found_traj_4_truncated_right = False
        for i, s in enumerate(rb):
            t = s["traj"].unique().tolist()
            assert len(t) == 1
            t = t[0]
            if t == 0:
                found_traj_0 = True
            if t == 4 and s.numel() < 5:
                if s["step_count"][0] > 10:
                    found_traj_4_truncated_right = True
                if s["step_count"][0] == 0:
                    pass
            if i == 1000:
                break
        assert not rb.sampler.span[0]
        # if rb.sampler.span[0]:
        #     assert found_traj_4_truncated_left
        if rb.sampler.span[1]:
            assert found_traj_4_truncated_right
        else:
            assert not found_traj_4_truncated_right
        if strict_length and not rb.sampler.span[1]:
            assert not found_traj_0
        else:
            assert found_traj_0

    @pytest.mark.parametrize("max_priority_within_buffer", [True, False])
    def test_prb_update_max_priority(self, max_priority_within_buffer):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(11),
            sampler=PrioritizedSampler(
                max_capacity=11,
                alpha=1.0,
                beta=1.0,
                max_priority_within_buffer=max_priority_within_buffer,
            ),
        )
        for data in torch.arange(20):
            idx = rb.add(data)
            rb.update_priority(idx, 21 - data)
            if data <= 10:
                # The max is always going to be the first value
                assert rb.sampler._max_priority[0] == 21
                assert rb.sampler._max_priority[1] == 0
            elif not max_priority_within_buffer:
                # The max is the historical max, which was at idx 0
                assert rb.sampler._max_priority[0] == 21
                assert rb.sampler._max_priority[1] == 0
            else:
                # the max is the current max. Find it and compare
                sumtree = torch.as_tensor(
                    [rb.sampler._sum_tree[i] for i in range(rb.sampler._max_capacity)]
                )
                assert rb.sampler._max_priority[0] == sumtree.max()
                assert rb.sampler._max_priority[1] == sumtree.argmax()
        idx = rb.extend(torch.arange(10))
        rb.update_priority(idx, 12)
        if max_priority_within_buffer:
            assert rb.sampler._max_priority[0] == 12
            assert rb.sampler._max_priority[1] == 0
        else:
            assert rb.sampler._max_priority[0] == 21
            assert rb.sampler._max_priority[1] == 0

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
    def test_prb_serialization(self, tmpdir):
        rb = ReplayBuffer(
            storage=LazyMemmapStorage(max_size=10),
            sampler=PrioritizedSampler(max_capacity=10, alpha=0.8, beta=0.6),
        )

        td = TensorDict(
            {
                "observations": torch.zeros(1, 3),
                "actions": torch.zeros(1, 1),
                "rewards": torch.zeros(1, 1),
                "next_observations": torch.zeros(1, 3),
                "terminations": torch.zeros(1, 1, dtype=torch.bool),
            },
            batch_size=[1],
        )
        rb.extend(td)

        rb.save(tmpdir)

        rb2 = ReplayBuffer(
            storage=LazyMemmapStorage(max_size=10),
            sampler=PrioritizedSampler(max_capacity=10, alpha=0.5, beta=0.5),
        )

        td = TensorDict(
            {
                "observations": torch.ones(1, 3),
                "actions": torch.ones(1, 1),
                "rewards": torch.ones(1, 1),
                "next_observations": torch.ones(1, 3),
                "terminations": torch.ones(1, 1, dtype=torch.bool),
            },
            batch_size=[1],
        )
        rb2.extend(td)
        rb2.load(tmpdir)
        assert len(rb) == 1
        assert rb.sampler._alpha == rb2.sampler._alpha
        assert rb.sampler._beta == rb2.sampler._beta
        assert rb.sampler._max_priority[0] == rb2.sampler._max_priority[0]
        assert rb.sampler._max_priority[1] == rb2.sampler._max_priority[1]

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
    def test_prb_new_sampler_with_loaded_storage(self, tmpdir):
        """Test that creating a new PrioritizedSampler with loaded storage works correctly.

        This test reproduces the issue from scratch8.py where creating a new
        PrioritizedSampler instance with storage that already contains data
        would fail with "RuntimeError: non-positive p_sum".
        """
        device = torch.device("cpu")

        # Create and populate original buffer
        original_rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(10, device=device),
            sampler=PrioritizedSampler(max_capacity=10, alpha=0.7, beta=0.5),
            batch_size=2,
            priority_key="td_error",
        )

        data = TensorDict(
            {
                "state": torch.ones(4, 2, dtype=torch.float32, device=device),
                "td_error": torch.ones(4) * 0.5,
            },
            batch_size=torch.Size((4,)),
        )
        original_rb.extend(data)

        # Update priorities
        td = original_rb.sample()
        td["td_error"] = torch.arange(2, device=device) + 1.0
        original_rb.update_tensordict_priority(td)

        # Get original priorities for comparison
        original_priorities = torch.tensor(
            [original_rb._sampler._sum_tree[i] for i in range(len(original_rb))]
        )

        # Save and load normally
        original_rb.dumps(tmpdir)
        del original_rb

        loaded_rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(10, device=device),
            sampler=PrioritizedSampler(max_capacity=10, alpha=0.7, beta=0.5),
            batch_size=2,
            priority_key="td_error",
        )
        loaded_rb.loads(tmpdir)

        # Create a new buffer with the loaded storage but NEW sampler
        # This was failing before the fix with "RuntimeError: non-positive p_sum"
        new_rb_with_loaded_storage = TensorDictReplayBuffer(
            storage=loaded_rb.storage,  # Use the loaded storage
            sampler=PrioritizedSampler(  # But create a NEW sampler instance
                max_capacity=len(loaded_rb), alpha=0.7, beta=0.5
            ),
            batch_size=2,
            priority_key="td_error",
        )

        # This should work now thanks to our fix
        td = new_rb_with_loaded_storage.sample()
        assert td.batch_size == torch.Size([2])

        # Verify the storage has the expected data
        assert len(new_rb_with_loaded_storage) == 4

        # Verify priorities were properly initialized with default values
        # When creating a new sampler with existing storage, it should initialize with default priorities
        new_priorities = torch.tensor(
            [
                new_rb_with_loaded_storage._sampler._sum_tree[i]
                for i in range(len(new_rb_with_loaded_storage))
            ]
        )
        expected_default_priority = new_rb_with_loaded_storage._sampler.default_priority
        expected_priorities = torch.full(
            (len(new_rb_with_loaded_storage),),
            expected_default_priority,
            dtype=torch.float,
        )

        # All priorities should be positive and equal to the default priority
        assert (new_priorities > 0).all(), "All priorities should be positive"
        torch.testing.assert_close(
            new_priorities,
            expected_priorities,
            msg="New sampler should initialize with default priorities",
        )

        # Also verify that the loaded buffer maintains the original priorities
        loaded_priorities = torch.tensor(
            [loaded_rb._sampler._sum_tree[i] for i in range(len(loaded_rb))]
        )
        torch.testing.assert_close(
            loaded_priorities,
            original_priorities,
            msg="Loaded buffer should maintain original priorities",
        )

    def test_prb_ndim(self):
        """This test lists all the possible ways of updating the priority of a PRB with RB, TRB and TPRB.

        All tests are done for 1d and 2d TDs.

        """
        torch.manual_seed(0)
        np.random.seed(0)

        # first case: 1d, RB
        rb = ReplayBuffer(
            sampler=PrioritizedSampler(max_capacity=100, alpha=1.0, beta=1.0),
            storage=LazyTensorStorage(100),
            batch_size=4,
        )
        data = TensorDict({"a": torch.arange(10), "p": torch.ones(10) / 2}, [10])
        idx = rb.extend(data)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 2).all()
        s, info = rb.sample(return_info=True)
        rb.update_priority(info["index"], 3)
        assert (
            torch.tensor([rb.sampler._sum_tree[i] for i in range(10)])[info["index"]]
            == 3
        ).all()

        # second case: 1d, TRB
        rb = TensorDictReplayBuffer(
            sampler=PrioritizedSampler(max_capacity=100, alpha=1.0, beta=1.0),
            storage=LazyTensorStorage(100),
            batch_size=4,
        )
        data = TensorDict({"a": torch.arange(10), "p": torch.ones(10) / 2}, [10])
        idx = rb.extend(data)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 2).all()
        s = rb.sample()
        rb.update_priority(s["index"], 3)
        assert (
            torch.tensor([rb.sampler._sum_tree[i] for i in range(10)])[s["index"]] == 3
        ).all()

        # third case: 1d TPRB
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=1.0,
            beta=1.0,
            storage=LazyTensorStorage(100),
            batch_size=4,
            priority_key="p",
        )
        data = TensorDict({"a": torch.arange(10), "p": torch.ones(10) / 2}, [10])
        idx = rb.extend(data)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 0.5).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 2).all()
        s = rb.sample()

        s["p"] = torch.ones(4) * 10_000
        rb.update_tensordict_priority(s)
        assert (
            torch.tensor([rb.sampler._sum_tree[i] for i in range(10)])[s["index"]]
            == 10_000
        ).all()

        s2 = rb.sample()
        # All indices in s2 must be from s since we set a very high priority to these items
        assert (s2["index"].unsqueeze(0) == s["index"].unsqueeze(1)).any(0).all()

        # fourth case: 2d RB
        rb = ReplayBuffer(
            sampler=PrioritizedSampler(max_capacity=100, alpha=1.0, beta=1.0),
            storage=LazyTensorStorage(100, ndim=2),
            batch_size=4,
        )
        data = TensorDict(
            {"a": torch.arange(5).expand(2, 5), "p": torch.ones(2, 5) / 2}, [2, 5]
        )
        idx = rb.extend(data)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 2).all()

        s, info = rb.sample(return_info=True)
        rb.update_priority(info["index"], 3)
        priorities = torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]).reshape(
            (5, 2)
        )
        assert (priorities[info["index"]] == 3).all()

        # fifth case: 2d TRB
        # 2d
        rb = TensorDictReplayBuffer(
            sampler=PrioritizedSampler(max_capacity=100, alpha=1.0, beta=1.0),
            storage=LazyTensorStorage(100, ndim=2),
            batch_size=4,
        )
        data = TensorDict(
            {"a": torch.arange(5).expand(2, 5), "p": torch.ones(2, 5) / 2}, [2, 5]
        )
        idx = rb.extend(data)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 2).all()

        s = rb.sample()
        rb.update_priority(s["index"], 10_000)
        priorities = torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]).reshape(
            (5, 2)
        )
        assert (priorities[s["index"].unbind(-1)] == 10_000).all()

        s2 = rb.sample()
        assert (
            (s2["index"].unsqueeze(0) == s["index"].unsqueeze(1)).all(-1).any(0).all()
        )

        # Sixth case: 2d TDPRB
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=1.0,
            beta=1.0,
            storage=LazyTensorStorage(100, ndim=2),
            batch_size=4,
            priority_key="p",
        )
        data = TensorDict(
            {"a": torch.arange(5).expand(2, 5), "p": torch.ones(2, 5) / 2}, [2, 5]
        )
        idx = rb.extend(data)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 0.5).all()
        rb.update_priority(idx, torch.ones(()) * 2)
        assert (torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]) == 2).all()
        s = rb.sample()
        # setting the priorities to a value that is so big that the buffer will resample them
        s["p"] = torch.ones(4) * 10_000
        rb.update_tensordict_priority(s)
        priorities = torch.tensor([rb.sampler._sum_tree[i] for i in range(10)]).reshape(
            (5, 2)
        )
        assert (priorities[s["index"].unbind(-1)] == 10_000).all()

        s2 = rb.sample()
        assert (
            (s2["index"].unsqueeze(0) == s["index"].unsqueeze(1)).all(-1).any(0).all()
        )

    def test_replacement_kwarg_random(self):
        # RandomSampler(replacement=True) is a regular RandomSampler
        s = RandomSampler()
        assert type(s) is RandomSampler
        s = RandomSampler(replacement=True)
        assert type(s) is RandomSampler

        # RandomSampler(replacement=False) dispatches to SamplerWithoutReplacement
        s = RandomSampler(replacement=False)
        assert type(s) is SamplerWithoutReplacement
        # default kwargs propagate
        assert s.drop_last is False
        assert s.shuffle is True

        # Extra kwargs are forwarded to SamplerWithoutReplacement
        s = RandomSampler(replacement=False, drop_last=True, shuffle=False)
        assert type(s) is SamplerWithoutReplacement
        assert s.drop_last is True
        assert s.shuffle is False

        # isinstance is preserved
        assert isinstance(s, Sampler)
        assert isinstance(s, SamplerWithoutReplacement)

    def test_replacement_kwarg_slice(self):
        # SliceSampler(replacement=True) is a regular SliceSampler
        s = SliceSampler(slice_len=5)
        assert type(s) is SliceSampler
        s = SliceSampler(replacement=True, slice_len=5)
        assert type(s) is SliceSampler

        # SliceSampler(replacement=False) dispatches to SliceSamplerWithoutReplacement
        s = SliceSampler(replacement=False, slice_len=5)
        assert type(s) is SliceSamplerWithoutReplacement
        assert s.slice_len == 5
        assert s.drop_last is False
        assert s.shuffle is True

        # Extra without-replacement kwargs forward correctly
        s = SliceSampler(
            replacement=False,
            slice_len=5,
            drop_last=True,
            shuffle=False,
            traj_key="episode",
            strict_length=False,
        )
        assert type(s) is SliceSamplerWithoutReplacement
        assert s.slice_len == 5
        assert s.drop_last is True
        assert s.shuffle is False
        assert s.traj_key == "episode"
        assert s.strict_length is False

        # isinstance preserves the SliceSampler hierarchy
        assert isinstance(s, SliceSampler)
        assert isinstance(s, SamplerWithoutReplacement)

    def test_replacement_kwarg_subclass_unaffected(self):
        # PrioritizedSliceSampler inherits from SliceSampler but should NOT dispatch
        s = PrioritizedSliceSampler(slice_len=5, max_capacity=10, alpha=0.5, beta=0.5)
        assert type(s) is PrioritizedSliceSampler

        # SamplerWithoutReplacement(replacement=...) is a no-op pop
        s = SamplerWithoutReplacement(replacement=False, drop_last=True)
        assert type(s) is SamplerWithoutReplacement
        assert s.drop_last is True
        s = SliceSamplerWithoutReplacement(replacement=False, slice_len=5)
        assert type(s) is SliceSamplerWithoutReplacement
        assert s.slice_len == 5

    def test_replacement_kwarg_no_variant_errors(self):
        # PrioritizedSampler has no without-replacement variant -> TypeError
        with pytest.raises(TypeError, match="no without-replacement variant"):
            PrioritizedSampler(max_capacity=10, alpha=0.5, beta=0.5, replacement=False)

    def test_replacement_kwarg_in_replay_buffer(self):
        # End-to-end: a buffer using RandomSampler(replacement=False) should
        # exhaust the storage without duplicate indices (like SamplerWithoutReplacement).
        torch.manual_seed(0)
        data = TensorDict({"a": torch.arange(11)}, batch_size=[11])
        rb = ReplayBuffer(
            storage=LazyTensorStorage(11),
            sampler=RandomSampler(replacement=False, drop_last=False),
            batch_size=3,
        )
        rb.extend(data)
        seen = set()
        for _ in range(4):
            seen.update(rb.sample()["a"].tolist())
        assert seen == set(range(11))

    def test_replacement_kwarg_slice_in_replay_buffer(self):
        # End-to-end: SliceSampler(replacement=False) returns sub-trajectories
        torch.manual_seed(0)
        episodes = torch.zeros(60, dtype=torch.long)
        episodes[:20] = 0
        episodes[20:40] = 1
        episodes[40:] = 2
        data = TensorDict(
            {"episode": episodes, "obs": torch.arange(60)},
            batch_size=[60],
        )
        rb = ReplayBuffer(
            storage=LazyTensorStorage(60),
            sampler=SliceSampler(
                replacement=False,
                slice_len=5,
                traj_key="episode",
                strict_length=True,
            ),
            batch_size=10,
        )
        rb.extend(data)
        sample = rb.sample()
        # batch_size=10, slice_len=5 -> 2 slices of 5 contiguous obs each
        obs = sample["obs"].view(2, 5)
        diffs = obs[:, 1:] - obs[:, :-1]
        assert (diffs == 1).all(), obs


class TestStalenessAwareSampler:
    """Tests for StalenessAwareSampler."""

    def _make_buffer_with_versions(self, n_entries=100, version_range=(0, 5)):
        """Create a replay buffer populated with data containing policy_version."""
        sampler = StalenessAwareSampler(max_staleness=-1)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(n_entries),
            sampler=sampler,
            batch_size=16,
        )
        # Fill with data having varying policy versions
        for v in range(version_range[0], version_range[1] + 1):
            batch = TensorDict(
                {
                    "observation": torch.randn(20, 4),
                    "action": torch.randn(20, 2),
                    "policy_version": torch.full((20,), float(v)),
                },
                batch_size=[20],
            )
            rb.extend(batch)
        return rb, sampler

    def test_basic_sampling(self):
        """Test that StalenessAwareSampler can sample from a buffer."""
        rb, sampler = self._make_buffer_with_versions()
        sampler.consumer_version = 5
        batch = rb.sample()
        assert batch is not None
        assert batch.shape[0] == 16

    def test_freshness_weighting(self):
        """Test that fresher entries are sampled more frequently."""
        sampler = StalenessAwareSampler(max_staleness=-1)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(200),
            sampler=sampler,
            batch_size=32,
        )
        # Add 100 entries at version 0 (stale) and 100 at version 9 (fresh)
        stale = TensorDict(
            {
                "observation": torch.zeros(100, 4),
                "policy_version": torch.full((100,), 0.0),
            },
            batch_size=[100],
        )
        fresh = TensorDict(
            {
                "observation": torch.ones(100, 4),
                "policy_version": torch.full((100,), 9.0),
            },
            batch_size=[100],
        )
        rb.extend(stale)
        rb.extend(fresh)
        sampler.consumer_version = 10

        # Sample many times and count how often fresh vs stale entries appear
        fresh_count = 0
        total = 0
        for _ in range(100):
            batch = rb.sample()
            # Fresh entries have observation == 1, stale have observation == 0
            fresh_count += (batch["observation"][:, 0] > 0.5).sum().item()
            total += batch.shape[0]

        fresh_ratio = fresh_count / total
        # Fresh entries (staleness=1) should be sampled ~10x more than stale (staleness=10)
        # So fresh_ratio should be significantly above 0.5
        assert (
            fresh_ratio > 0.7
        ), f"Expected fresh entries to dominate, got {fresh_ratio:.2f}"

    def test_hard_staleness_gate(self):
        """Test that entries beyond max_staleness are never sampled."""
        sampler = StalenessAwareSampler(max_staleness=3)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(200),
            sampler=sampler,
            batch_size=32,
        )
        # Add entries at version 0 (stale) and version 8 (fresh)
        stale = TensorDict(
            {
                "observation": torch.zeros(100, 4),
                "policy_version": torch.full((100,), 0.0),
            },
            batch_size=[100],
        )
        fresh = TensorDict(
            {
                "observation": torch.ones(100, 4),
                "policy_version": torch.full((100,), 8.0),
            },
            batch_size=[100],
        )
        rb.extend(stale)
        rb.extend(fresh)
        sampler.consumer_version = 10

        # All sampled entries should be fresh (staleness=2 <= 3)
        # Stale entries have staleness=10 > 3, so they're excluded
        for _ in range(50):
            batch = rb.sample()
            assert (
                batch["observation"][:, 0] > 0.5
            ).all(), (
                "Stale entries should never be sampled when max_staleness is exceeded"
            )

    def test_all_stale_raises(self):
        """Test that an error is raised when all entries exceed max_staleness."""
        sampler = StalenessAwareSampler(max_staleness=2)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(50),
            sampler=sampler,
            batch_size=8,
        )
        data = TensorDict(
            {
                "observation": torch.randn(50, 4),
                "policy_version": torch.full((50,), 0.0),
            },
            batch_size=[50],
        )
        rb.extend(data)
        sampler.consumer_version = 100  # Everything is very stale

        with pytest.raises(RuntimeError, match="max_staleness"):
            rb.sample()

    def test_consumer_version_increment(self):
        """Test consumer version tracking."""
        sampler = StalenessAwareSampler()
        assert sampler.consumer_version == 0
        sampler.increment_consumer_version()
        assert sampler.consumer_version == 1
        sampler.consumer_version = 42
        assert sampler.consumer_version == 42

    def test_staleness_in_info(self):
        """Test that staleness values are returned in sample info."""
        sampler = StalenessAwareSampler(max_staleness=-1)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(50),
            sampler=sampler,
            batch_size=8,
        )
        data = TensorDict(
            {
                "observation": torch.randn(50, 4),
                "policy_version": torch.full((50,), 3.0),
            },
            batch_size=[50],
        )
        rb.extend(data)
        sampler.consumer_version = 5

        index, info = sampler.sample(rb._storage, 8)
        assert "staleness" in info
        assert (info["staleness"] == 2.0).all()  # consumer=5 - version=3 = 2

    def test_missing_version_key_raises(self):
        """Test that a clear error is raised when version key is missing."""
        sampler = StalenessAwareSampler()
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(50),
            sampler=sampler,
            batch_size=8,
        )
        data = TensorDict(
            {"observation": torch.randn(50, 4)},
            batch_size=[50],
        )
        rb.extend(data)

        with pytest.raises(KeyError, match="policy_version"):
            rb.sample()

    def test_state_dict_roundtrip(self):
        """Test that state_dict/load_state_dict preserves sampler state."""
        sampler = StalenessAwareSampler(max_staleness=7)
        sampler.consumer_version = 42

        sd = sampler.state_dict()
        assert sd["consumer_version"] == 42
        assert sd["max_staleness"] == 7

        sampler2 = StalenessAwareSampler()
        sampler2.load_state_dict(sd)
        assert sampler2.consumer_version == 42
        assert sampler2.max_staleness == 7

    def test_no_staleness_limit(self):
        """Test sampling with max_staleness=-1 (no limit)."""
        sampler = StalenessAwareSampler(max_staleness=-1)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(50),
            sampler=sampler,
            batch_size=8,
        )
        data = TensorDict(
            {
                "observation": torch.randn(50, 4),
                "policy_version": torch.full((50,), 0.0),
            },
            batch_size=[50],
        )
        rb.extend(data)
        sampler.consumer_version = 1000  # Very stale, but no limit

        # Should not raise
        batch = rb.sample()
        assert batch.shape[0] == 8


def test_prioritized_slice_sampler_doc_example():
    sampler = PrioritizedSliceSampler(max_capacity=9, num_slices=3, alpha=0.7, beta=0.9)
    rb = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(9), sampler=sampler, batch_size=6
    )
    data = TensorDict(
        {
            "observation": torch.randn(9, 16),
            "action": torch.randn(9, 1),
            "episode": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long),
            "steps": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long),
            ("next", "observation"): torch.randn(9, 16),
            ("next", "reward"): torch.randn(9, 1),
            ("next", "done"): torch.tensor(
                [0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.bool
            ).unsqueeze(1),
        },
        batch_size=[9],
    )
    rb.extend(data)
    sample, info = rb.sample(return_info=True)
    # print("episode", sample["episode"].tolist())
    # print("steps", sample["steps"].tolist())
    # print("weight", info["priority_weight"].tolist())

    priority = torch.tensor([0, 3, 3, 0, 0, 0, 1, 1, 1])
    rb.update_priority(torch.arange(0, 9, 1), priority=priority)
    sample, info = rb.sample(return_info=True)
    # print("episode", sample["episode"].tolist())
    # print("steps", sample["steps"].tolist())
    # print("weight", info["priority_weight"].tolist())


@pytest.mark.parametrize("device", get_default_devices())
def test_prioritized_slice_sampler_episodes(device):
    num_slices = 10
    batch_size = 20

    episode = torch.zeros(100, dtype=torch.int, device=device)
    episode[:30] = 1
    episode[30:55] = 2
    episode[55:70] = 3
    episode[70:] = 4
    steps = torch.cat(
        [torch.arange(30), torch.arange(25), torch.arange(15), torch.arange(30)], 0
    )
    done = torch.zeros(100, 1, dtype=torch.bool)
    done[torch.tensor([29, 54, 69])] = 1

    data = TensorDict(
        {
            "observation": torch.randn(100, 16),
            "action": torch.randn(100, 4),
            "episode": episode,
            "steps": steps,
            ("next", "observation"): torch.randn(100, 16),
            ("next", "reward"): torch.randn(100, 1),
            ("next", "done"): done,
        },
        batch_size=[100],
        device=device,
    )

    num_steps = data.shape[0]
    sampler = PrioritizedSliceSampler(
        max_capacity=num_steps,
        alpha=0.7,
        beta=0.9,
        num_slices=num_slices,
    )

    rb = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(100),
        sampler=sampler,
        batch_size=batch_size,
    )
    rb.extend(data)

    episodes = []
    for _ in range(10):
        sample = rb.sample()
        episodes.append(sample["episode"])
    assert {1, 2, 3, 4} == set(
        torch.cat(episodes).cpu().tolist()
    ), "all episodes are expected to be sampled at least once"

    index = torch.arange(0, num_steps, 1)
    new_priorities = torch.cat(
        [torch.ones(30), torch.zeros(25), torch.ones(15), torch.zeros(30)], 0
    )
    sampler.update_priority(index, new_priorities)

    episodes = []
    for _ in range(10):
        sample = rb.sample()
        episodes.append(sample["episode"])
    assert {1, 3} == set(
        torch.cat(episodes).cpu().tolist()
    ), "after priority update, only episode 1 and 3 are expected to be sampled"


@pytest.mark.parametrize("alpha", [0.6, torch.tensor(1.0)])
@pytest.mark.parametrize("beta", [0.7, torch.tensor(0.1)])
@pytest.mark.parametrize("gamma", [0.1])
@pytest.mark.parametrize("total_steps", [200])
@pytest.mark.parametrize("n_annealing_steps", [100])
@pytest.mark.parametrize("anneal_every_n", [10, 159])
@pytest.mark.parametrize("alpha_min", [0, 0.2])
@pytest.mark.parametrize("beta_max", [1, 1.4])
def test_prioritized_parameter_scheduler(
    alpha,
    beta,
    gamma,
    total_steps,
    n_annealing_steps,
    anneal_every_n,
    alpha_min,
    beta_max,
):
    rb = TensorDictPrioritizedReplayBuffer(
        alpha=alpha, beta=beta, storage=ListStorage(max_size=1000)
    )
    data = TensorDict({"data": torch.randn(1000, 5)}, batch_size=1000)
    rb.extend(data)
    alpha_scheduler = LinearScheduler(
        rb, param_name="alpha", final_value=alpha_min, num_steps=n_annealing_steps
    )
    beta_scheduler = StepScheduler(
        rb,
        param_name="beta",
        gamma=gamma,
        n_steps=anneal_every_n,
        max_value=beta_max,
        mode="additive",
    )

    scheduler = SchedulerList(schedulers=(alpha_scheduler, beta_scheduler))

    alpha = alpha if torch.is_tensor(alpha) else torch.tensor(alpha)
    alpha_min = torch.tensor(alpha_min)
    expected_alpha_vals = torch.linspace(alpha, alpha_min, n_annealing_steps + 1)
    expected_alpha_vals = torch.nn.functional.pad(
        expected_alpha_vals, (0, total_steps - n_annealing_steps), value=alpha_min
    )

    expected_beta_vals = [beta]
    annealing_steps = total_steps // anneal_every_n
    gammas = torch.arange(0, annealing_steps + 1, dtype=torch.float32) * gamma
    expected_beta_vals = (
        (beta + gammas).repeat_interleave(anneal_every_n).clip(None, beta_max)
    )
    for i in range(total_steps):
        curr_alpha = rb.sampler.alpha
        torch.testing.assert_close(
            curr_alpha
            if torch.is_tensor(curr_alpha)
            else torch.tensor(curr_alpha).float(),
            expected_alpha_vals[i],
            msg=f"expected {expected_alpha_vals[i]}, got {curr_alpha}",
        )
        curr_beta = rb.sampler.beta
        torch.testing.assert_close(
            curr_beta
            if torch.is_tensor(curr_beta)
            else torch.tensor(curr_beta).float(),
            expected_beta_vals[i],
            msg=f"expected {expected_beta_vals[i]}, got {curr_beta}",
        )
        rb.sample(20)
        scheduler.step()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
