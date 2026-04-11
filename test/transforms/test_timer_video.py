# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest

import torch

from _transforms_common import TransformBase
from tensordict import LazyStackedTensorDict, TensorDict
from torch import nn

from torchrl.envs import Compose, SerialEnv, StepCounter, Timer, TransformedEnv
from torchrl.envs.utils import check_env_specs
from torchrl.record.recorder import VideoRecorder

from torchrl.testing import (  # noqa
    BREAKOUT_VERSIONED,
    dtype_fixture,
    get_default_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rand_reset,
    retry,
)
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv


class TestTimer(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), Timer())
        check_env_specs(env)
        env.close()

    def test_serial_trans_env_check(self):
        env = SerialEnv(
            2, lambda: TransformedEnv(ContinuousActionVecMockEnv(), Timer())
        )
        check_env_specs(env)
        env.close()

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(
            2, lambda: TransformedEnv(ContinuousActionVecMockEnv(), Timer())
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv()), Timer()
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            Timer(),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        torch.manual_seed(0)
        t = Timer()
        with pytest.raises(NotImplementedError):
            t(TensorDict())

    def test_transform_compose(self):
        torch.manual_seed(0)
        t = Compose(Timer())
        with pytest.raises(NotImplementedError):
            t(TensorDict())

    def test_transform_env(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), Timer())
        rollout = env.rollout(3)
        # The stack must be contiguous
        assert not isinstance(rollout, LazyStackedTensorDict)
        assert (rollout["time_policy"] >= 0).all()
        assert (rollout["time_step"] >= 0).all()
        env.append_transform(StepCounter(max_steps=5))
        rollout = env.rollout(10, break_when_any_done=False)
        assert (rollout["time_reset"] > 0).sum() == 2
        assert (rollout["time_policy"] == 0).sum() == 2
        assert (rollout["time_step"] == 0).sum() == 2
        assert (rollout["next", "time_reset"] == 0).all()
        assert (rollout["next", "time_policy"] > 0).all()
        assert (rollout["next", "time_step"] > 0).all()

    def test_transform_model(self):
        torch.manual_seed(0)
        t = nn.Sequential(Timer())
        with pytest.raises(NotImplementedError):
            t(TensorDict())

    def test_transform_rb(self):
        # NotImplemented tested elsewhere
        return

    def test_transform_inverse(self):
        raise pytest.skip("Tested elsewhere")


class TestVideoRecorder:
    # TODO: add more tests
    def test_can_init_with_fps(self):
        recorder = VideoRecorder(None, None, fps=30)

        assert recorder is not None

    def test_video_recorder_grayscale(self):
        """Test that VideoRecorder handles 1-channel (grayscale) observations."""
        recorder = VideoRecorder(None, None, fps=30)
        # Simulate a grayscale observation: (1, H, W) — single channel
        obs = torch.randint(0, 255, (1, 64, 64), dtype=torch.uint8)
        recorder._apply_transform(obs)
        # The stored frame should be expanded to 3-channel for video codecs
        assert len(recorder.obs) == 1
        stored = recorder.obs[0]
        assert stored.shape == (3, 64, 64)

    def test_video_recorder_grayscale_batched(self):
        """Test that VideoRecorder handles batched grayscale observations."""
        recorder = VideoRecorder(None, None, fps=30)
        # Batched grayscale: (B, 1, H, W)
        obs = torch.randint(0, 255, (4, 1, 64, 64), dtype=torch.uint8)
        recorder._apply_transform(obs)
        # Batched observations get flattened into individual frames
        assert len(recorder.obs) == 4
        for frame in recorder.obs:
            assert frame.shape == (3, 64, 64)
