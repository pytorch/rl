# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import logging
import sys

import pytest

import torch

from _transforms_common import TransformBase
from tensordict import LazyStackedTensorDict, TensorDict
from torch import nn

from torchrl._utils import logger as torchrl_logger
from torchrl.envs import Compose, SerialEnv, StepCounter, Timer, TransformedEnv
from torchrl.envs.utils import check_env_specs
from torchrl.record.recorder import VideoRecorder

from torchrl.testing import (  # noqa
    AddPixelsTransform,
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

_has_matplotlib = importlib.util.find_spec("matplotlib", None) is not None


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

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Timer resolution on Windows causes flaky time_reset assertions",
    )
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


class _CaptureVideoLogger:
    """Duck-typed logger that stores each ``log_video`` call."""

    def __init__(self):
        self.calls = []

    def log_video(self, name=None, video=None, step=None, **kwargs):
        self.calls.append({"name": name, "video": video, "step": step, **kwargs})


def _make_pixels_env(max_steps=None):
    transforms = [AddPixelsTransform()]
    if max_steps is not None:
        transforms.append(StepCounter(max_steps=max_steps))
    return TransformedEnv(ContinuousActionVecMockEnv(), Compose(*transforms))


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

    @pytest.mark.skipif(not _has_matplotlib, reason="matplotlib not installed")
    def test_video_recorder_to_animation(self):
        from matplotlib.animation import ArtistAnimation

        recorder = VideoRecorder(None, None, fps=30)
        obs = torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8)
        recorder._apply_transform(obs)
        obs_buffer = recorder.obs

        anim = recorder.to_animation(title="test", clear=True)

        assert isinstance(anim, ArtistAnimation)
        assert recorder.obs == []
        assert recorder.obs is obs_buffer
        assert recorder.count == 0

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

    def test_video_recorder_batched_env_grid(self):
        """A recorder attached outside a batched env records one grid video."""
        logger = _CaptureVideoLogger()
        env = TransformedEnv(
            SerialEnv(2, _make_pixels_env),
            VideoRecorder(logger, tag="grid_video"),
        )
        env.rollout(3)
        env.transform.dump(step=0)
        env.close()
        assert len(logger.calls) == 1
        video = logger.calls[0]["video"]
        # one reset frame + 3 step frames, each a 1x2 grid of 8x8 workers
        assert video.shape == (1, 4, 3, 8, 16)

    def test_video_recorder_max_frames(self):
        with pytest.raises(ValueError, match="max_frames"):
            VideoRecorder(None, None, max_frames=0)

        recorder = VideoRecorder(None, None, max_frames=3)
        for _ in range(5):
            recorder._apply_transform(torch.zeros(3, 16, 16, dtype=torch.uint8))
        assert len(recorder.obs) == 3
        # dump empties the buffer, then recording resumes
        recorder.dump()
        recorder._apply_transform(torch.zeros(3, 16, 16, dtype=torch.uint8))
        assert len(recorder.obs) == 1

    def test_video_recorder_max_frames_batched(self):
        """The frame cap also truncates flattened batch extensions."""
        recorder = VideoRecorder(None, None, max_frames=5, make_grid=False)
        recorder._apply_transform(torch.zeros(4, 3, 16, 16, dtype=torch.uint8))
        recorder._apply_transform(torch.zeros(4, 3, 16, 16, dtype=torch.uint8))
        recorder._apply_transform(torch.zeros(4, 3, 16, 16, dtype=torch.uint8))
        assert len(recorder.obs) == 5

    @pytest.mark.parametrize("batched", [False, True])
    def test_video_recorder_dump_on_done(self, batched):
        logger = _CaptureVideoLogger()
        recorder = VideoRecorder(logger, tag="done_video", dump_on_done=True, skip=1)
        if batched:
            base = SerialEnv(2, lambda: _make_pixels_env(max_steps=4))
        else:
            base = _make_pixels_env(max_steps=4)
        env = TransformedEnv(base, recorder)
        env.rollout(10)
        env.close()
        # the episode-ending step (StepCounter truncation) triggered the dump
        assert len(logger.calls) == 1
        video = logger.calls[0]["video"]
        # one reset frame + 4 step frames
        assert video.shape[1] == 5
        assert not recorder.obs

    def test_video_recorder_in_batched_worker_warns(self):
        records = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Handler()
        torchrl_logger.addHandler(handler)
        try:
            SerialEnv(
                2,
                lambda: TransformedEnv(
                    ContinuousActionVecMockEnv(),
                    Compose(AddPixelsTransform(), VideoRecorder(None, None)),
                ),
            )
        finally:
            torchrl_logger.removeHandler(handler)
        assert any("VideoRecorder" in message for message in records)
