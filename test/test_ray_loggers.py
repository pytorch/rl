# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os
import tempfile

import pytest
import torch

_has_ray = importlib.util.find_spec("ray") is not None


@pytest.fixture(autouse=True)
def _ray_init_shutdown():
    """Initialize Ray before each test and shut it down after."""
    if not _has_ray:
        yield
        return
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=2, log_to_driver=False)
    yield
    ray.shutdown()


@pytest.mark.skipif(not _has_ray, reason="Ray not available")
class TestRayLoggerMetaclass:
    """Test that the _RayServiceMetaClass routing works for loggers."""

    def test_csv_logger_returns_ray_logger(self):
        from torchrl.record.loggers.csv import CSVLogger
        from torchrl.record.loggers.ray import RayLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(exp_name="test", log_dir=tmpdir, use_ray_service=True)
            assert isinstance(logger, RayLogger)

    def test_csv_logger_returns_csv_logger(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(exp_name="test", log_dir=tmpdir)
            assert isinstance(logger, CSVLogger)

    def test_use_ray_service_false_is_default(self):
        from torchrl.record.loggers.csv import CSVLogger
        from torchrl.record.loggers.ray import RayLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(exp_name="test", log_dir=tmpdir, use_ray_service=False)
            assert isinstance(logger, CSVLogger)
            assert not isinstance(logger, RayLogger)


@pytest.mark.skipif(not _has_ray, reason="Ray not available")
class TestRayLoggerOperations:
    """Test logging operations through the Ray wrapper using CSVLogger."""

    def test_log_scalar(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_scalar",
                log_dir=tmpdir,
                use_ray_service=True,
            )
            logger.log_scalar("loss", 0.5, step=0)
            logger.log_scalar("loss", 0.3, step=1)

            # Verify the CSV file was written
            scalars_dir = os.path.join(tmpdir, "test_scalar", "scalars")
            assert os.path.isdir(scalars_dir)
            csv_file = os.path.join(scalars_dir, "loss.csv")
            assert os.path.isfile(csv_file)

    def test_log_metrics(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_metrics",
                log_dir=tmpdir,
                use_ray_service=True,
            )
            metrics = {"train/loss": 0.5, "train/reward": 1.0}
            result = logger.log_metrics(metrics, step=0)
            assert result == metrics

    def test_log_metrics_with_tensors(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_tensor_metrics",
                log_dir=tmpdir,
                use_ray_service=True,
            )
            metrics = {
                "loss": torch.tensor(0.5),
                "reward": torch.tensor(1.0),
            }
            result = logger.log_metrics(metrics, step=0)
            assert isinstance(result["loss"], float)
            assert isinstance(result["reward"], float)

    def test_log_hparams(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_hparams",
                log_dir=tmpdir,
                use_ray_service=True,
            )
            logger.log_hparams({"lr": 0.001, "batch_size": 32})

    def test_log_video(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_video",
                log_dir=tmpdir,
                video_format="pt",
                use_ray_service=True,
            )
            video = torch.randint(0, 255, (1, 10, 3, 64, 64), dtype=torch.uint8)
            logger.log_video("test_vid", video, step=0)

            # Verify the video file was written
            videos_dir = os.path.join(tmpdir, "test_video", "videos")
            assert os.path.isdir(videos_dir)

    def test_repr(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_repr",
                log_dir=tmpdir,
                use_ray_service=True,
            )
            r = repr(logger)
            assert isinstance(r, str)
            assert "CSVLogger" in r


@pytest.mark.skipif(not _has_ray, reason="Ray not available")
class TestRayLoggerGetattr:
    """Test __getattr__ fallback for logger-specific methods."""

    def test_print_log_dir(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_getattr",
                log_dir=tmpdir,
                use_ray_service=True,
            )
            # CSVLogger.print_log_dir() is a non-standard method
            # Should not raise
            logger.print_log_dir()

    def test_private_attr_raises(self):
        from torchrl.record.loggers.csv import CSVLogger
        from torchrl.record.loggers.ray import RayLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_private",
                log_dir=tmpdir,
                use_ray_service=True,
            )
            assert isinstance(logger, RayLogger)
            with pytest.raises(AttributeError):
                logger._nonexistent


@pytest.mark.skipif(not _has_ray, reason="Ray not available")
class TestRayLoggerOptions:
    """Test ray_actor_options passthrough."""

    def test_ray_actor_options(self):
        from torchrl.record.loggers.csv import CSVLogger
        from torchrl.record.loggers.ray import RayLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_options",
                log_dir=tmpdir,
                use_ray_service=True,
                ray_actor_options={"num_cpus": 1},
            )
            assert isinstance(logger, RayLogger)
            logger.log_scalar("x", 1.0, step=0)


@pytest.mark.skipif(not _has_ray, reason="Ray not available")
class TestRayLoggerProperties:
    """Test delegated property access."""

    def test_exp_name(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="my_experiment",
                log_dir=tmpdir,
                use_ray_service=True,
            )
            assert logger.exp_name == "my_experiment"

    def test_log_dir(self):
        from torchrl.record.loggers.csv import CSVLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_logdir",
                log_dir=tmpdir,
                use_ray_service=True,
            )
            assert logger.log_dir == tmpdir


@pytest.mark.skipif(not _has_ray, reason="Ray not available")
class TestGetLoggerRay:
    """Test get_logger with use_ray_service."""

    def test_get_logger_csv_with_ray(self):
        from torchrl.record.loggers.ray import RayLogger
        from torchrl.record.loggers.utils import get_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger("csv", tmpdir, "test_get_logger", use_ray_service=True)
            assert isinstance(logger, RayLogger)
            logger.log_scalar("x", 1.0, step=0)

    def test_get_logger_csv_without_ray(self):
        from torchrl.record.loggers.csv import CSVLogger
        from torchrl.record.loggers.utils import get_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger("csv", tmpdir, "test_no_ray")
            assert isinstance(logger, CSVLogger)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
