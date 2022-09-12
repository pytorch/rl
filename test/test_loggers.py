import argparse
import os.path
import pathlib
import tempfile
from time import sleep

import pytest
import torch
import torchvision
import os
from torchrl.trainers.loggers.csv import CSVLogger
from torchrl.trainers.loggers.tensorboard import TensorboardLogger, _has_tb
from torchrl.trainers.loggers.wandb import WandbLogger, _has_wandb
from torchrl.trainers.loggers.mlflow import MLFlowLogger, _has_mlflow


@pytest.mark.skipif(not _has_tb, reason="TensorBoard not installed")
class TestTensorboard:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as log_dir:
            exp_name = "ramala"
            logger = TensorboardLogger(log_dir=log_dir, exp_name=exp_name)

            values = torch.rand(3)
            for i in range(3):
                scalar_name = "foo"
                scalar_value = values[i].item()
                logger.log_scalar(
                    value=scalar_value,
                    name=scalar_name,
                    step=steps[i] if steps else None,
                )

            sleep(0.01)  # wait until events are registered
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            event_acc = EventAccumulator(logger.experiment.get_logdir())
            event_acc.Reload()
            assert len(event_acc.Scalars("foo")) == 3, str(event_acc.Scalars("foo"))
            for i in range(3):
                assert event_acc.Scalars("foo")[i].value == values[i]
                if steps:
                    assert event_acc.Scalars("foo")[i].step == steps[i]


class TestCSVLogger:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as log_dir:
            exp_name = "ramala"
            logger = CSVLogger(log_dir=log_dir, exp_name=exp_name)

            values = torch.rand(3)
            for i in range(3):
                scalar_name = "foo"
                scalar_value = values[i].item()
                logger.log_scalar(
                    value=scalar_value,
                    name=scalar_name,
                    step=steps[i] if steps else None,
                )

            with open(
                os.path.join(log_dir, exp_name, "scalars", "foo.csv"), "r"
            ) as file:
                for i, row in enumerate(file.readlines()):
                    step = steps[i] if steps else i
                    assert row == f"{step},{values[i].item()}\n"


@pytest.mark.skipif(not _has_wandb, reason="Wandb not installed")
class TestWandbLogger:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as log_dir:
            exp_name = "ramala"
            logger = WandbLogger(log_dir=log_dir, exp_name=exp_name, offline=True)

            values = torch.rand(3)
            for i in range(3):
                scalar_name = "foo"
                scalar_value = values[i].item()
                logger.log_scalar(
                    value=scalar_value,
                    name=scalar_name,
                    step=steps[i] if steps else None,
                )

            assert logger.experiment.summary["foo"] == values[-1].item()
            assert logger.experiment.summary["_step"] == i if not steps else steps[i]

            logger.experiment.finish()
            del logger


@pytest.mark.skipif(not _has_mlflow, reason="MLFlow not installed")
class TestMLFlowLogger:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps):
        torch.manual_seed(0)
        import mlflow

        with tempfile.TemporaryDirectory() as log_dir:
            exp_name = "ramala"
            log_dir_uri = pathlib.Path(log_dir).as_uri()
            logger = MLFlowLogger(exp_name=exp_name, tracking_uri=log_dir_uri)

            values = torch.rand(3)
            for i in range(3):
                scalar_name = "foo"
                scalar_value = values[i].item()
                logger.log_scalar(
                    value=scalar_value,
                    name=scalar_name,
                    step=steps[i] if steps else None,
                )
            run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
            client = mlflow.MlflowClient()
            for i, metric in enumerate(client.get_metric_history(run_id, "foo")):
                assert metric.key == "foo"
                assert metric.step == (steps[i] if steps else 0)
                assert metric.value == values[i].item()

    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_video(self, steps):
        torch.manual_seed(0)
        import mlflow

        with tempfile.TemporaryDirectory() as log_dir:
            exp_name = "ramala"
            log_dir_uri = pathlib.Path(log_dir).as_uri()
            logger = MLFlowLogger(exp_name=exp_name, tracking_uri=log_dir_uri)

            test_frames = torch.rand((3, 60, 64, 64, 3)) * 255  # (N, T, W, H, C)
            test_frames = test_frames.int()

            fps = 6
            for i in range(3):
                logger.log_video(
                    name="test_video",
                    video=test_frames[i],
                    fps=fps,
                    step=steps[i] if steps else None,
                )
            run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
            client = mlflow.MlflowClient()
            with tempfile.TemporaryDirectory() as artifacts_dir:
                videos_dir = client.download_artifacts(run_id, "videos", artifacts_dir)
                for i, video_name in enumerate(os.listdir(videos_dir)):
                    video_path = os.path.join(videos_dir, video_name)
                    loaded_frames, _, _ = torchvision.io.read_video(
                        video_path, pts_unit="sec"
                    )
                    if steps:
                        assert torch.equal(loaded_frames.int(), test_frames[i])
                    else:
                        assert torch.equal(loaded_frames.int(), test_frames[-1])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
