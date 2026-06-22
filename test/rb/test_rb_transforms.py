# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
from functools import partial
from unittest import mock

import pytest
import torch
from _rb_common import _has_tv
from tensordict import TensorDict
from torch.utils._pytree import tree_map

from torchrl.data import ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler, SliceSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
from torchrl.envs.transforms import NextStateReconstructor
from torchrl.envs.transforms.transforms import (
    BinarizeReward,
    CatFrames,
    CatTensors,
    CenterCrop,
    DiscreteActionProjection,
    DoubleToFloat,
    FiniteTensorDictCheck,
    FlattenObservation,
    GrayScale,
    gSDENoise,
    ObservationNorm,
    PinMemoryTransform,
    Resize,
    RewardClipping,
    RewardScaling,
    SqueezeTransform,
    ToTensorImage,
    UnsqueezeTransform,
    VecNorm,
)


class TestTransforms:
    def test_append_transform(self):
        rb = ReplayBuffer(collate_fn=lambda x: torch.stack(x, 0), batch_size=1)
        td = TensorDict(
            {
                "observation": torch.randn(2, 4, 3, 16),
                "observation2": torch.randn(2, 4, 3, 16),
            },
            [],
        )
        rb.add(td)
        flatten = CatTensors(
            in_keys=["observation", "observation2"], out_key="observation_cat"
        )

        rb.append_transform(flatten)

        sampled = rb.sample()
        assert sampled.get("observation_cat").shape[-1] == 32

    def test_init_transform(self):
        flatten = FlattenObservation(
            -2, -1, in_keys=["observation"], out_keys=["flattened"]
        )

        rb = ReplayBuffer(
            collate_fn=lambda x: torch.stack(x, 0), transform=flatten, batch_size=1
        )

        td = TensorDict({"observation": torch.randn(2, 4, 3, 16)}, [])
        rb.add(td)
        sampled = rb.sample()
        assert sampled.get("flattened").shape[-1] == 48

    def test_insert_transform(self):
        flatten = FlattenObservation(
            -2, -1, in_keys=["observation"], out_keys=["flattened"]
        )
        rb = ReplayBuffer(
            collate_fn=lambda x: torch.stack(x, 0), transform=flatten, batch_size=1
        )
        td = TensorDict({"observation": torch.randn(2, 4, 3, 16, 1)}, [])
        rb.add(td)

        rb.insert_transform(0, SqueezeTransform(-1, in_keys=["observation"]))

        sampled = rb.sample()
        assert sampled.get("flattened").shape[-1] == 48

        with pytest.raises(ValueError):
            rb.insert_transform(10, SqueezeTransform(-1, in_keys=["observation"]))

    transforms = [
        ToTensorImage,
        pytest.param(
            partial(RewardClipping, clamp_min=0.1, clamp_max=0.9), id="RewardClipping"
        ),
        BinarizeReward,
        pytest.param(
            partial(Resize, w=2, h=2),
            id="Resize",
            marks=pytest.mark.skipif(
                not _has_tv, reason="needs torchvision dependency"
            ),
        ),
        pytest.param(
            partial(CenterCrop, w=1),
            id="CenterCrop",
            marks=pytest.mark.skipif(
                not _has_tv, reason="needs torchvision dependency"
            ),
        ),
        pytest.param(partial(UnsqueezeTransform, dim=-1), id="UnsqueezeTransform"),
        pytest.param(partial(SqueezeTransform, dim=-1), id="SqueezeTransform"),
        GrayScale,
        pytest.param(partial(ObservationNorm, loc=1, scale=2), id="ObservationNorm"),
        pytest.param(partial(CatFrames, dim=-3, N=4), id="CatFrames"),
        pytest.param(partial(RewardScaling, loc=1, scale=2), id="RewardScaling"),
        DoubleToFloat,
        VecNorm,
    ]

    @pytest.mark.parametrize("transform", transforms)
    def test_smoke_replay_buffer_transform(self, transform):
        rb = TensorDictReplayBuffer(
            transform=transform(in_keys=["observation"]), batch_size=1
        )

        # td = TensorDict({"observation": torch.randn(3, 3, 3, 16, 1), "action": torch.randn(3)}, [])
        td = TensorDict({"observation": torch.randn(3, 3, 3, 16, 3)}, [])
        rb.add(td)

        m = mock.Mock()
        m.side_effect = [td.unsqueeze(0)]
        rb._transform.forward = m
        # rb._transform.__len__ = lambda *args: 3
        rb.sample()
        assert rb._transform.forward.called

        # was_called = [False]
        # forward = rb._transform.forward
        # def new_forward(*args, **kwargs):
        #     was_called[0] = True
        #     return forward(*args, **kwargs)
        # rb._transform.forward = new_forward
        # rb.sample()
        # assert was_called[0]

    transforms2 = [
        partial(DiscreteActionProjection, num_actions_effective=1, max_actions=3),
        FiniteTensorDictCheck,
        gSDENoise,
        PinMemoryTransform,
    ]

    @pytest.mark.parametrize("transform", transforms2)
    def test_smoke_replay_buffer_transform_no_inkeys(self, transform):
        if transform == PinMemoryTransform and not torch.cuda.is_available():
            raise pytest.skip("No CUDA device detected, skipping PinMemory")
        rb = ReplayBuffer(
            collate_fn=lambda x: torch.stack(x, 0), transform=transform(), batch_size=1
        )

        action = torch.zeros(3)
        action[..., 0] = 1
        td = TensorDict(
            {"observation": torch.randn(3, 3, 3, 16, 1), "action": action}, []
        )
        rb.add(td)
        rb.sample()

        rb._transform = mock.MagicMock()
        rb._transform.__len__ = lambda *args: 3
        rb.sample()
        assert rb._transform.called

    @pytest.mark.parametrize("at_init", [True, False])
    def test_transform_nontensor(self, at_init):
        def t(x):
            return tree_map(lambda y: y * 0, x)

        if at_init:
            rb = ReplayBuffer(storage=LazyMemmapStorage(100), transform=t)
        else:
            rb = ReplayBuffer(storage=LazyMemmapStorage(100))
            rb.append_transform(t)
        data = {
            "a": torch.randn(3),
            "b": {"c": (torch.zeros(2), [torch.ones(1)])},
            30: -torch.ones(()),
        }
        rb.add(data)

        def assert0(x):
            assert (x == 0).all()

        s = rb.sample(10)
        tree_map(assert0, s)

    def test_transform_inv(self):
        rb = ReplayBuffer(storage=LazyMemmapStorage(10), batch_size=4)
        data = TensorDict({"a": torch.zeros(10)}, [10])

        def t(data):
            data += 1
            return data

        rb.append_transform(t, invert=True)
        rb.extend(data)
        assert (data == 1).all()


class TestNextStateReconstructor:
    """Tests for :class:`~torchrl.envs.transforms.NextStateReconstructor`."""

    _DEFAULT_TRAJ_KEY = ("collector", "traj_ids")

    @classmethod
    def _make_data(
        cls,
        n_traj=3,
        traj_len=4,
        obs_dim=2,
        traj_key: tuple | str | None = None,
    ):
        if traj_key is None:
            traj_key = cls._DEFAULT_TRAJ_KEY
        n = n_traj * traj_len
        obs = torch.arange(n * obs_dim, dtype=torch.float32).reshape(n, obs_dim)
        done = torch.zeros(n, 1, dtype=torch.bool)
        done[traj_len - 1 :: traj_len] = True
        traj_ids = torch.repeat_interleave(torch.arange(n_traj), traj_len)
        return TensorDict(
            {
                "observation": obs,
                ("next", "done"): done,
                ("next", "reward"): torch.zeros(n, 1),
                traj_key: traj_ids,
            },
            batch_size=[n],
        )

    def test_slice_sampler_default(self):
        """With ``SliceSampler`` + default ``traj_key``, slices mirror cleanly."""
        data = self._make_data(n_traj=3, traj_len=4)
        rb = ReplayBuffer(
            storage=LazyTensorStorage(12),
            sampler=SliceSampler(slice_len=4, traj_key=self._DEFAULT_TRAJ_KEY),
            transform=NextStateReconstructor(),
            batch_size=8,
        )
        rb.extend(data)
        sample = rb.sample()
        assert sample.batch_size == torch.Size([8])
        next_obs = sample.get(("next", "observation"))
        root_obs = sample.get("observation")
        traj = sample.get(self._DEFAULT_TRAJ_KEY)
        # Within each slice (4 entries), positions 0..2 mirror to 1..3 of the same traj.
        for slice_start in (0, 4):
            assert (traj[slice_start : slice_start + 4] == traj[slice_start]).all()
            for i in range(3):
                torch.testing.assert_close(
                    next_obs[slice_start + i], root_obs[slice_start + i + 1]
                )
            # Last position of each slice belongs to a different trajectory
            # in the (i, i+1) pair (or has no i+1 at all) → NaN.
            assert torch.isnan(next_obs[slice_start + 3]).all()

    def test_single_trajectory_full_batch(self):
        """Whole trajectory as one batch: every transition reconstructed, last NaN."""
        n = 6
        td = TensorDict(
            {
                "observation": torch.arange(n, dtype=torch.float32).view(n, 1),
                self._DEFAULT_TRAJ_KEY: torch.zeros(n, dtype=torch.long),
                # No terminal in the middle; explicit final done for completeness.
                ("next", "done"): torch.tensor([[False]] * (n - 1) + [[True]]),
            },
            batch_size=[n],
        )
        out = NextStateReconstructor()(td)
        next_obs = out.get(("next", "observation"))
        torch.testing.assert_close(next_obs[:-1], td.get("observation")[1:])
        assert torch.isnan(next_obs[-1]).all()

    def test_done_catches_slice_repetition(self):
        """SliceSampler can place two slices of the same trajectory in one batch.

        Trajectory ids match across the splice; ``done`` at the slice end of the
        first copy disambiguates. Without the done check, the first slice's
        last position would silently borrow the *second slice's first frame*
        (same trajectory, but not its temporal successor) and the user would
        never know.
        """
        n = 8  # two identical trajectories of length 4, glued together
        obs = torch.tensor([[0.0], [1.0], [2.0], [3.0]] * 2, dtype=torch.float32)
        td = TensorDict(
            {
                "observation": obs,
                self._DEFAULT_TRAJ_KEY: torch.tensor([0] * 8),  # all same id
                ("next", "done"): torch.tensor([[False], [False], [False], [True]] * 2),
            },
            batch_size=[n],
        )
        out = NextStateReconstructor()(td)
        next_obs = out.get(("next", "observation"))
        # Position 3: traj id matches position 4, but done[3]=True → NaN
        assert torch.isnan(next_obs[3]).all()
        # Positions 0..2 mirror to 1..3
        torch.testing.assert_close(next_obs[:3], obs[1:4])
        # Positions 4..6 mirror to 5..7
        torch.testing.assert_close(next_obs[4:7], obs[5:8])
        # Position 7: no i+1 → NaN
        assert torch.isnan(next_obs[7]).all()

    def test_random_sampler_is_mostly_nan(self):
        """Random sampling yields mismatched traj ids between neighbors → NaN.

        Documents the honest failure mode: when the user picks a sampler that
        doesn't preserve trajectory adjacency, the transform refuses to invent
        a next observation.
        """
        data = self._make_data(n_traj=8, traj_len=4)  # 32 entries
        rb = ReplayBuffer(
            storage=LazyTensorStorage(32),
            sampler=RandomSampler(),
            transform=NextStateReconstructor(),
            batch_size=16,
        )
        rb.extend(data)
        torch.manual_seed(0)
        sample = rb.sample()
        next_obs = sample.get(("next", "observation"))
        # With 8 trajectories random-sampled into a 16-batch, the chance that
        # two adjacent picks share a trajectory id (≈ 1/8) is low. Assert that
        # the *vast majority* of positions are NaN — both that the check is
        # firing and that we aren't accidentally fabricating next obs.
        nan_frac = torch.isnan(next_obs).all(dim=-1).float().mean().item()
        assert nan_frac > 0.7, f"expected mostly-NaN, got nan_frac={nan_frac:.2f}"

    def test_nested_keys(self):
        n = 8
        td = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "pos": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
                        "vel": torch.arange(n * 2, dtype=torch.float32).reshape(n, 2),
                    },
                    [n],
                ),
                ("next", "done"): torch.tensor([[False], [False], [False], [True]] * 2),
                ("next", "reward"): torch.zeros(n, 1),
                self._DEFAULT_TRAJ_KEY: torch.tensor([0] * 4 + [1] * 4),
            },
            batch_size=[n],
        )
        rb = ReplayBuffer(
            storage=LazyTensorStorage(n),
            sampler=SliceSampler(slice_len=4, traj_key=self._DEFAULT_TRAJ_KEY),
            transform=NextStateReconstructor(
                keys=[("agents", "pos"), ("agents", "vel")],
            ),
            batch_size=4,
        )
        rb.extend(td)
        sample = rb.sample()
        for k in (("agents", "pos"), ("agents", "vel")):
            next_k = ("next", *k)
            torch.testing.assert_close(sample.get(next_k)[:3], sample.get(k)[1:4])
            assert torch.isnan(sample.get(next_k)[3]).all()

    def test_explicit_fill_value(self):
        data = self._make_data(n_traj=2, traj_len=4)
        rb = ReplayBuffer(
            storage=LazyTensorStorage(8),
            sampler=SliceSampler(slice_len=4, traj_key=self._DEFAULT_TRAJ_KEY),
            transform=NextStateReconstructor(fill_value=-1.0),
            batch_size=8,
        )
        rb.extend(data)
        sample = rb.sample()
        next_obs = sample.get(("next", "observation"))
        # The last position of each slice belongs to a different trajectory
        # in (i, i+1), so it gets the fill value.
        for slice_start in (0, 4):
            assert (next_obs[slice_start + 3] == -1.0).all()

    def test_overwrites_existing_next_obs(self):
        """If ``("next", k)`` is already in storage, the transform overwrites it."""
        n = 8
        td = TensorDict(
            {
                "observation": torch.arange(n, dtype=torch.float32).view(n, 1),
                ("next", "observation"): torch.full(
                    (n, 1), -999.0, dtype=torch.float32
                ),
                ("next", "done"): torch.tensor([[False], [False], [False], [True]] * 2),
                ("next", "reward"): torch.zeros(n, 1),
                self._DEFAULT_TRAJ_KEY: torch.tensor([0] * 4 + [1] * 4),
            },
            batch_size=[n],
        )
        rb = ReplayBuffer(
            storage=LazyTensorStorage(n),
            sampler=SliceSampler(slice_len=4, traj_key=self._DEFAULT_TRAJ_KEY),
            transform=NextStateReconstructor(),
            batch_size=8,
        )
        rb.extend(td)
        sample = rb.sample()
        assert not (sample.get(("next", "observation")) == -999.0).any()

    def test_step_count_cross_check(self):
        """``step_count_key`` adds a stricter "consecutive in time" check."""
        n = 4
        td = TensorDict(
            {
                "observation": torch.arange(n, dtype=torch.float32).view(n, 1),
                self._DEFAULT_TRAJ_KEY: torch.zeros(n, dtype=torch.long),
                ("next", "done"): torch.zeros(n, 1, dtype=torch.bool),
                # Same traj id and no done, but step counts disagree at i=1
                # (jumps from 0 to 5, then 5 -> 6 -> 7).
                ("collector", "step_count"): torch.tensor([0, 5, 6, 7]),
            },
            batch_size=[n],
        )
        t = NextStateReconstructor(step_count_key=("collector", "step_count"))
        out = t(td)
        next_obs = out.get(("next", "observation"))
        # Position 0 → step_count[1] - step_count[0] = 5 ≠ 1, so NaN.
        assert torch.isnan(next_obs[0]).all()
        # Positions 1 and 2 are consecutive (5→6, 6→7) → reconstructed.
        torch.testing.assert_close(next_obs[1], td.get("observation")[2])
        torch.testing.assert_close(next_obs[2], td.get("observation")[3])
        # Position 3 has no i+1 → NaN.
        assert torch.isnan(next_obs[3]).all()

    def test_strict_missing_traj_key_raises(self):
        td = TensorDict(
            {"observation": torch.arange(4, dtype=torch.float32).view(4, 1)},
            batch_size=[4],
        )
        with pytest.raises(KeyError, match="trajectory key"):
            NextStateReconstructor()(td)

    def test_strict_missing_done_key_raises(self):
        td = TensorDict(
            {
                "observation": torch.arange(4, dtype=torch.float32).view(4, 1),
                self._DEFAULT_TRAJ_KEY: torch.zeros(4, dtype=torch.long),
            },
            batch_size=[4],
        )
        with pytest.raises(KeyError, match="done key"):
            NextStateReconstructor()(td)

    def test_strict_false_single_traj_fallback(self):
        td = TensorDict(
            {"observation": torch.arange(4, dtype=torch.float32).view(4, 1)},
            batch_size=[4],
        )
        out = NextStateReconstructor(strict=False)(td)
        next_obs = out.get(("next", "observation"))
        torch.testing.assert_close(next_obs[:-1], td.get("observation")[1:])
        assert torch.isnan(next_obs[-1]).all()

    def test_traj_key_none_disables_check(self):
        td = TensorDict(
            {
                "observation": torch.arange(4, dtype=torch.float32).view(4, 1),
                # Different traj ids, but check is disabled → all-shift, no NaN
                # except the last position.
                self._DEFAULT_TRAJ_KEY: torch.tensor([0, 1, 2, 3]),
            },
            batch_size=[4],
        )
        out = NextStateReconstructor(traj_key=None, done_key=None)(td)
        next_obs = out.get(("next", "observation"))
        torch.testing.assert_close(next_obs[:-1], td.get("observation")[1:])
        assert torch.isnan(next_obs[-1]).all()

    def test_int_obs_requires_explicit_fill_value(self):
        td = TensorDict(
            {
                "observation": torch.arange(4, dtype=torch.int64).view(4, 1),
                self._DEFAULT_TRAJ_KEY: torch.zeros(4, dtype=torch.long),
                ("next", "done"): torch.zeros(4, 1, dtype=torch.bool),
            },
            batch_size=[4],
        )
        with pytest.raises(TypeError, match="non-floating dtype"):
            NextStateReconstructor()(td)
        # Explicit integer fill works
        out = NextStateReconstructor(fill_value=-1)(td)
        next_obs = out.get(("next", "observation"))
        assert next_obs[-1].item() == -1

    def test_bad_batch_dims_errors(self):
        td = TensorDict(
            {
                "observation": torch.arange(8, dtype=torch.float32).view(2, 4, 1),
                self._DEFAULT_TRAJ_KEY: torch.zeros(2, 4, dtype=torch.long),
            },
            batch_size=[2, 4],
        )
        with pytest.raises(ValueError, match="flat"):
            NextStateReconstructor()(td)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
