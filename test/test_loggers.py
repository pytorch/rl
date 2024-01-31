# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path
import pathlib
import tempfile
from time import sleep

import pytest
import torch

from tensordict import MemoryMappedTensor
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record.loggers.mlflow import _has_mlflow, _has_tv, MLFlowLogger
from torchrl.record.loggers.tensorboard import _has_tb, TensorboardLogger
from torchrl.record.loggers.wandb import _has_wandb, WandbLogger

if _has_tv:
    import torchvision

if _has_tb:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if _has_mlflow:
    import mlflow


@pytest.fixture
def tb_logger(tmp_path_factory):
    tmpdir1 = tmp_path_factory.mktemp("tmpdir1")
    exp_name = "ramala"
    logger = TensorboardLogger(log_dir=tmpdir1, exp_name=exp_name)
    yield logger
    del logger


@pytest.fixture
def config():
    return {
        "value": "value",
        "nested": {"inner": 3, "value": "value"},
        "int": 3,
        "list": [3, 4, 5],
        "tuple": (2,),
        "float": 3.45,
        "bool": True,
    }


@pytest.mark.skipif(not _has_tb, reason="TensorBoard not installed")
class TestTensorboard:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps, tb_logger):
        torch.manual_seed(0)

        values = torch.rand(3)
        for i in range(3):
            scalar_name = "foo"
            scalar_value = values[i].item()
            tb_logger.log_scalar(
                value=scalar_value,
                name=scalar_name,
                step=steps[i] if steps else None,
            )

        sleep(0.01)  # wait until events are registered

        event_acc = EventAccumulator(tb_logger.experiment.get_logdir())
        event_acc.Reload()
        assert len(event_acc.Scalars("foo")) == 3, str(event_acc.Scalars("foo"))
        for i in range(3):
            assert event_acc.Scalars("foo")[i].value == values[i]
            if steps:
                assert event_acc.Scalars("foo")[i].step == steps[i]

    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_video(self, steps, tb_logger):
        torch.manual_seed(0)

        # creating a sample video (T, C, H, W), where T - number of frames,
        # C - number of image channels (e.g. 3 for RGB), H, W - image dimensions.
        # the first 64 frames are black and the next 64 are white
        video = torch.cat(
            (torch.zeros(64, 1, 32, 32), torch.full((64, 1, 32, 32), 255))
        )
        video = video[None, :]
        for i in range(3):
            tb_logger.log_video(
                name="foo",
                video=video,
                step=steps[i] if steps else None,
                fps=6,  # we can't test for the difference between fps, because the result is an encoded_string
            )

        sleep(0.01)  # wait until events are registered

        event_acc = EventAccumulator(tb_logger.experiment.get_logdir())
        event_acc.Reload()
        assert len(event_acc.Images("foo")) == 3, str(event_acc.Images("foo"))

        # check that we catch the error in case the format of the tensor is wrong
        # here the number of color channels is set to 2, which is not correct
        video_wrong_format = torch.zeros(64, 2, 32, 32)
        video_wrong_format = video_wrong_format[None, :]
        with pytest.raises(Exception):
            tb_logger.log_video(
                name="foo",
                video=video_wrong_format,
                step=steps[i] if steps else None,
            )

    def test_log_hparams(self, tb_logger, config):
        del config["nested"]  # not supported in tensorboard
        del config["list"]  # not supported in tensorboard
        del config["tuple"]  # not supported in tensorboard
        tb_logger.log_hparams(config)

    def test_log_histogram(self, tb_logger):
        torch.manual_seed(0)
        # test with torch
        data = torch.randn(10)
        tb_logger.log_histogram("hist", data, step=0, bins=2)
        # test with np
        data = torch.randn(10).numpy()
        tb_logger.log_histogram("hist", data, step=1, bins=2)


class TestCSVLogger:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps, tmpdir):
        torch.manual_seed(0)
        exp_name = "ramala"
        logger = CSVLogger(log_dir=tmpdir, exp_name=exp_name)

        values = torch.rand(3)
        for i in range(3):
            scalar_name = "foo"
            scalar_value = values[i].item()
            logger.log_scalar(
                value=scalar_value,
                name=scalar_name,
                step=steps[i] if steps else None,
            )

        with open(os.path.join(tmpdir, exp_name, "scalars", "foo.csv"), "r") as file:
            for i, row in enumerate(file.readlines()):
                step = steps[i] if steps else i
                assert row == f"{step},{values[i].item()}\n"

    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    @pytest.mark.parametrize(
        "video_format", ["pt", "memmap"] + ["mp4"] if _has_tv else []
    )
    def test_log_video(self, steps, video_format, tmpdir):
        torch.manual_seed(0)
        exp_name = "ramala"
        logger = CSVLogger(log_dir=tmpdir, exp_name=exp_name, video_format=video_format)

        # creating a sample video (T, C, H, W), where T - number of frames,
        # C - number of image channels (e.g. 3 for RGB), H, W - image dimensions.
        # the first 64 frames are black and the next 64 are white
        video = torch.cat(
            (
                torch.zeros(64, 1, 32, 32, dtype=torch.uint8),
                torch.full((64, 1, 32, 32), 255, dtype=torch.uint8),
            )
        )
        video = video[None, :]
        for i in range(3):
            logger.log_video(
                name="foo",
                video=video,
                step=steps[i] if steps else None,
            )
        sleep(0.01)  # wait until events are registered

        # check that the logged videos are the same as the initial video
        extention = (
            ".pt"
            if video_format == "pt"
            else ".memmap"
            if video_format == "memmap"
            else ".mp4"
        )
        video_file_name = "foo_" + ("0" if not steps else str(steps[0])) + extention
        path = os.path.join(tmpdir, exp_name, "videos", video_file_name)
        if video_format == "pt":
            logged_video = torch.load(path)
            assert torch.equal(video, logged_video), logged_video
        elif video_format == "memmap":
            logged_video = MemoryMappedTensor.from_filename(
                path, dtype=torch.uint8, shape=(1, 128, 1, 32, 32)
            )
            assert torch.equal(video, logged_video), logged_video
        elif video_format == "mp4":
            import torchvision

            logged_video = torchvision.io.read_video(path, output_format="TCHW")[0][
                :, :1
            ]
            logged_video = logged_video.unsqueeze(0)
            torch.testing.assert_close(video, logged_video)

        # check that we catch the error in case the format of the tensor is wrong
        video_wrong_format = torch.zeros(64, 2, 32, 32)
        video_wrong_format = video_wrong_format[None, :]
        with pytest.raises(Exception):
            logger.log_video(
                name="foo",
                video=video_wrong_format,
                step=steps[i] if steps else None,
            )

    def test_log_histogram(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as log_dir:
            exp_name = "ramala"
            logger = CSVLogger(log_dir=log_dir, exp_name=exp_name)
            with pytest.raises(NotImplementedError):
                data = torch.randn(10)
                logger.log_histogram("hist", data, step=0, bins=2)

    def test_log_config(self, tmpdir, config):
        torch.manual_seed(0)

        exp_name = "ramala"
        logger = CSVLogger(log_dir=tmpdir, exp_name=exp_name)
        logger.log_hparams(cfg=config)

        with open(os.path.join(tmpdir, exp_name, "texts", "hparams0.txt"), "r") as file:
            txt = "\n".join([f"{k}: {val}" for k, val in sorted(config.items())])
            text = "".join(file.readlines())
            assert text == txt


@pytest.fixture(scope="class")
def wandb_logger(tmp_path_factory):
    tmpdir1 = tmp_path_factory.mktemp("tmpdir1")
    exp_name = "ramala"
    logger = WandbLogger(log_dir=tmpdir1, exp_name=exp_name, offline=True)
    yield logger
    logger.experiment.finish()
    del logger


@pytest.mark.skipif(not _has_wandb, reason="Wandb not installed")
class TestWandbLogger:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps, wandb_logger):
        torch.manual_seed(0)

        values = torch.rand(3)
        for i in range(3):
            scalar_name = "foo"
            scalar_value = values[i].item()
            wandb_logger.log_scalar(
                value=scalar_value,
                name=scalar_name,
                step=steps[i] if steps else None,
            )

        assert wandb_logger.experiment.summary["foo"] == values[-1].item()
        assert wandb_logger.experiment.summary["_step"] == i if not steps else steps[i]

    def test_log_video(self, wandb_logger):
        torch.manual_seed(0)

        # creating a sample video (T, C, H, W), where T - number of frames,
        # C - number of image channels (e.g. 3 for RGB), H, W - image dimensions.
        # the first 64 frames are black and the next 64 are white
        video = torch.cat(
            (torch.zeros(64, 1, 32, 32), torch.full((64, 1, 32, 32), 255))
        )
        video = video[None, :]
        wandb_logger.log_video(
            name="foo",
            video=video,
            fps=6,
        )
        wandb_logger.log_video(
            name="foo_12fps",
            video=video,
            fps=24,
        )
        sleep(0.01)  # wait until events are registered

        # check that fps can be passed and that it has impact on the length of the video
        video_6fps_size = wandb_logger.experiment.summary["foo"]["size"]
        video_24fps_size = wandb_logger.experiment.summary["foo_12fps"]["size"]
        assert video_6fps_size > video_24fps_size, video_6fps_size

        # check that we catch the error in case the format of the tensor is wrong
        video_wrong_format = torch.zeros(64, 2, 32, 32)
        video_wrong_format = video_wrong_format[None, :]
        with pytest.raises(Exception):
            wandb_logger.log_video(
                name="foo",
                video=video_wrong_format,
            )

    def test_log_hparams(self, wandb_logger, config):
        wandb_logger.log_hparams(config)
        for key, value in config.items():
            if isinstance(value, tuple):
                value = list(value)  # wandb converts tuples to lists
            assert wandb_logger.experiment.config[key] == value

    def test_log_histogram(self, wandb_logger):
        torch.manual_seed(0)
        # test with torch
        data = torch.randn(10)
        wandb_logger.log_histogram("hist", data, step=0, bins=2)
        # test with np
        data = torch.randn(10).numpy()
        wandb_logger.log_histogram("hist", data, step=1, bins=2)


@pytest.fixture
def mlflow_fixture():
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as log_dir:
        exp_name = "ramala"
        log_dir_uri = pathlib.Path(log_dir).as_uri()
        logger = MLFlowLogger(exp_name=exp_name, tracking_uri=log_dir_uri)
        client = mlflow.MlflowClient()
        yield logger, client
        mlflow.end_run()


@pytest.mark.skipif(not _has_mlflow, reason="MLFlow not installed")
class TestMLFlowLogger:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps, mlflow_fixture):

        logger, client = mlflow_fixture
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
        for i, metric in enumerate(client.get_metric_history(run_id, "foo")):
            assert metric.key == "foo"
            assert metric.step == (steps[i] if steps else 0)
            assert metric.value == values[i].item()

    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    @pytest.mark.skipif(not _has_tv, reason="torchvision not installed")
    def test_log_video(self, steps, mlflow_fixture):

        logger, client = mlflow_fixture
        videos = torch.cat(
            (torch.full((3, 64, 3, 32, 32), 255), torch.zeros(3, 64, 3, 32, 32)),
            dim=1,
        )
        fps = 6
        for i in range(3):
            logger.log_video(
                name="test_video",
                video=videos[i],
                fps=fps,
                step=steps[i] if steps else None,
            )
        run_id = mlflow.active_run().info.run_id
        with tempfile.TemporaryDirectory() as artifacts_dir:
            videos_dir = client.download_artifacts(run_id, "videos", artifacts_dir)
            for i, video_name in enumerate(os.listdir(videos_dir)):
                video_path = os.path.join(videos_dir, video_name)
                loaded_video, _, _ = torchvision.io.read_video(
                    video_path, pts_unit="sec", output_format="TCHW"
                )
                if steps:
                    assert torch.allclose(loaded_video.int(), videos[i].int(), rtol=0.1)
                else:
                    assert torch.allclose(
                        loaded_video.int(), videos[-1].int(), rtol=0.1
                    )

    def test_log_histogram(self, mlflow_fixture):
        logger, client = mlflow_fixture
        torch.manual_seed(0)
        with pytest.raises(NotImplementedError):
            data = torch.randn(10)
            logger.log_histogram("hist", data, step=0, bins=2)

    def test_log_hparams(self, mlflow_fixture, config):
        logger, client = mlflow_fixture
        logger.log_hparams(config)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
