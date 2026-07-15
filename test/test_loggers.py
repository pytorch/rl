# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import os.path
import pathlib
import tempfile
import threading
from time import sleep

import pytest
import torch
from tensordict import MemoryMappedTensor

from torchrl._comm import MailboxPeerClosedError
from torchrl.checkpoint import Checkpoint
from torchrl.envs import check_env_specs, GymEnv, ParallelEnv
from torchrl.record.loggers.common import _has_torchcodec, Logger
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record.loggers.mlflow import _has_mlflow, MLFlowLogger
from torchrl.record.loggers.process import ProcessLogger
from torchrl.record.loggers.ray import _RayLoggerClient, RayLogger
from torchrl.record.loggers.tensorboard import _has_tb, TensorboardLogger
from torchrl.record.loggers.trackio import _has_trackio, TrackioLogger
from torchrl.record.loggers.utils import get_logger
from torchrl.record.loggers.wandb import _has_moviepy, _has_wandb, WandbLogger
from torchrl.record.recorder import PixelRenderTransform, VideoRecorder

_has_mp4 = _has_torchcodec

if _has_tb:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if _has_mlflow:
    import mlflow

_has_ray = importlib.util.find_spec("ray") is not None
_has_gym = (
    importlib.util.find_spec("gym", None) is not None
    or importlib.util.find_spec("gymnasium", None) is not None
)


class _ProcessTestLogger:
    def __init__(self, log_dir):
        self.exp_name = "process-test"
        self.log_dir = str(log_dir)
        with open(os.path.join(self.log_dir, "constructed"), "a") as file:
            file.write("1\n")

    def log_scalar(self, name, value, step=None, **kwargs):
        del kwargs
        with open(os.path.join(self.log_dir, "events"), "a") as file:
            file.write(f"{name}:{value}:{step}\n")

    def log_video(self, name, video, step=None, **kwargs):
        torch.save(
            {"name": name, "video": video, "step": step, "kwargs": kwargs},
            os.path.join(self.log_dir, "video.pt"),
        )

    def log_fail(self):
        raise RuntimeError("expected logger failure")

    def log_with_result(self, value):
        return value + 1

    def flush(self):
        return None

    def __repr__(self):
        return "_ProcessTestLogger()"


class _SlowRestartProcessTestLogger(_ProcessTestLogger):
    def __init__(self, log_dir):
        restart_marker = pathlib.Path(log_dir) / "restart-marker"
        if restart_marker.exists():
            sleep(0.3)
        else:
            restart_marker.touch()
        super().__init__(log_dir)


def _log_process_scalars(client, name, count):
    for step in range(count):
        client.log_scalar(name, step, step=step)


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
            (
                torch.zeros(64, 1, 32, 32, dtype=torch.uint8),
                torch.full((64, 1, 32, 32), 255, dtype=torch.uint8),
            )
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
    def test_checkpoint_state(self, tmp_path):
        logger = CSVLogger(log_dir=tmp_path, exp_name="source")
        restored = CSVLogger(log_dir=tmp_path, exp_name="restored")
        logger.log_scalar("reward", 2.0)
        logger.experiment.videos_counter["evaluation"] = 3
        checkpoint_path = tmp_path / "checkpoint"
        Checkpoint(logger=logger).save(checkpoint_path)
        Checkpoint(logger=restored).load(
            checkpoint_path,
            tensor_load_kwargs={"weights_only": True, "mmap": True},
        )
        assert restored.experiment.scalars["reward"] == [(0, 2.0)]
        assert restored.experiment.videos_counter["evaluation"] == 3

    def test_direct_service_client_is_identity(self, tmpdir):
        logger = CSVLogger(log_dir=tmpdir, exp_name="direct")
        assert logger.client() is logger
        assert logger.start() is logger
        assert logger.service_backend == "direct"
        assert logger.is_alive
        logger.shutdown()
        assert not logger.is_alive
        logger.shutdown()

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

        with open(os.path.join(tmpdir, exp_name, "scalars", "foo.csv")) as file:
            for i, row in enumerate(file.readlines()):
                step = steps[i] if steps else i
                assert row == f"{step},{values[i].item()}\n"

    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    @pytest.mark.parametrize(
        "video_format", ["pt", "memmap"] + (["mp4"] if _has_mp4 else [])
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
        extension = (
            ".pt"
            if video_format == "pt"
            else ".memmap"
            if video_format == "memmap"
            else ".mp4"
        )
        video_file_name = "foo_" + ("0" if not steps else str(steps[0])) + extension
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
            from torchcodec.decoders import VideoDecoder

            logged_video = (
                VideoDecoder(path).get_frames_in_range(start=0, stop=128).data[:, :1]
            )
            logged_video = logged_video.unsqueeze(0)
            torch.testing.assert_close(video, logged_video, atol=2, rtol=0)

        # check that we catch the error in case the format of the tensor is wrong
        video_wrong_format = torch.zeros(64, 2, 32, 32)
        video_wrong_format = video_wrong_format[None, :]
        with pytest.raises(Exception):
            logger.log_video(
                name="foo",
                video=video_wrong_format,
                step=steps[i] if steps else None,
            )

    def test_log_video_mp4_requires_torchcodec(self, monkeypatch, tmpdir):
        import torchrl.record.loggers.common as logger_common

        monkeypatch.setattr(logger_common, "_has_torchcodec", False)
        logger = CSVLogger(log_dir=tmpdir, exp_name="ramala", video_format="mp4")
        video = torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8)

        with pytest.raises(ModuleNotFoundError, match="uv run --extra video"):
            logger.log_video(name="foo", video=video)

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

        with open(os.path.join(tmpdir, exp_name, "texts", "hparams0.txt")) as file:
            txt = "\n".join([f"{k}: {val}" for k, val in sorted(config.items())])
            text = "".join(file.readlines())
            assert text == txt


@pytest.fixture(scope="class")
def wandb_logger(tmp_path_factory):
    import wandb

    wandb.finish()
    tmpdir1 = tmp_path_factory.mktemp("tmpdir1")
    exp_name = "ramala"
    logger = WandbLogger(log_dir=tmpdir1, exp_name=exp_name, offline=True)
    yield logger
    logger.experiment.finish()
    wandb.finish()
    del logger


@pytest.fixture
def wandb_tmp_logger(tmp_path):
    import wandb

    wandb.finish()
    logger = WandbLogger(log_dir=tmp_path, exp_name="ramala", offline=True)
    yield logger
    logger.experiment.finish()
    wandb.finish()
    del logger


@pytest.mark.skipif(not _has_wandb, reason="Wandb not installed")
class TestWandbLogger:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps, wandb_tmp_logger, monkeypatch):
        torch.manual_seed(0)

        logged = []
        defined = []

        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "log",
            lambda payload, **kwargs: logged.append((payload, kwargs)),
        )
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "define_metric",
            lambda name, step_metric=None: defined.append((name, step_metric)),
        )

        values = torch.rand(3)
        for i in range(3):
            scalar_name = "foo"
            scalar_value = values[i].item()
            wandb_tmp_logger.log_scalar(
                value=scalar_value,
                name=scalar_name,
                step=steps[i] if steps else None,
                commit=True,
            )

        assert len(logged) == 3
        assert defined == [("step", None), ("foo", "step")]

        for i, (payload, kwargs) in enumerate(logged):
            expected_step = i if not steps else steps[i]
            assert payload == {"foo": values[i].item(), "step": expected_step}
            assert kwargs == {"commit": True}

    @pytest.mark.skipif(not _has_moviepy, reason="moviepy not installed")
    def test_log_video(self, wandb_logger):
        torch.manual_seed(0)

        # creating a sample video (T, C, H, W), where T - number of frames,
        # C - number of image channels (e.g. 3 for RGB), H, W - image dimensions.
        # the first 64 frames are black and the next 64 are white
        video = torch.cat(
            (
                torch.zeros(128, 1, 32, 32, dtype=torch.uint8),
                torch.full((128, 1, 32, 32), 255, dtype=torch.uint8),
            )
        )
        video = video[None, :]
        wandb_logger.log_video(
            name="foo",
            video=video,
            fps=4,
            format="mp4",
        )
        wandb_logger.log_video(
            name="foo_16fps",
            video=video,
            fps=16,
            format="mp4",
        )
        sleep(0.01)  # wait until events are registered

        # check that fps can be passed and that it has impact on the length of the video
        video_4fps_size = wandb_logger.experiment.summary["foo"]["size"]
        video_16fps_size = wandb_logger.experiment.summary["foo_16fps"]["size"]
        assert video_4fps_size > video_16fps_size, (video_4fps_size, video_16fps_size)

        # check that we catch the error in case the format of the tensor is wrong
        video_wrong_format = torch.zeros(2, 32, 32)
        with pytest.raises(ValueError, match="Video must be at least"):
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

    def test_logs_env_packages(self, wandb_logger):
        env = wandb_logger.experiment.config["env"]
        assert "python" in env
        assert "packages" in env
        assert "tensordict" in {key.lower() for key in env["packages"]}

    def test_log_histogram(self, wandb_logger):
        torch.manual_seed(0)
        # test with torch
        data = torch.randn(10)
        wandb_logger.log_histogram("hist", data, step=0, bins=2)
        # test with np
        data = torch.randn(10).numpy()
        wandb_logger.log_histogram("hist", data, step=1, bins=2)

    def test_log_metrics_infers_nested_steps(self, wandb_tmp_logger, monkeypatch):
        logged = []
        defined = []
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "log",
            lambda payload, **kwargs: logged.append((payload, kwargs)),
        )
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "define_metric",
            lambda name, step_metric=None: defined.append((name, step_metric)),
        )

        metrics = {"eval/reward": 1.0, "eval/other/something": 2.0}
        result = wandb_tmp_logger.log_metrics(metrics, step=7)

        assert result == metrics
        assert logged == [
            (
                {
                    "eval/reward": 1.0,
                    "eval/other/something": 2.0,
                    "eval/step": 7,
                    "eval/other/step": 7,
                },
                {},
            )
        ]
        assert defined == [
            ("eval/step", None),
            ("eval/other/step", None),
            ("eval/reward", "eval/step"),
            ("eval/other/something", "eval/other/step"),
        ]

    def test_log_metrics_preserves_explicit_step_keys(
        self, wandb_tmp_logger, monkeypatch
    ):
        logged = []
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "log",
            lambda payload, **kwargs: logged.append((payload, kwargs)),
        )
        monkeypatch.setattr(
            wandb_tmp_logger.experiment, "define_metric", lambda *args, **kwargs: None
        )

        wandb_tmp_logger.log_metrics({"eval/reward": 1.0, "eval/step": 4}, step=99)
        wandb_tmp_logger.log_metrics({"eval/reward": 2.0})

        assert logged[0][0]["eval/step"] == 4
        assert logged[1][0]["eval/step"] == 5

    def test_log_metrics_auto_increments_per_group(self, wandb_tmp_logger, monkeypatch):
        logged = []
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "log",
            lambda payload, **kwargs: logged.append((payload, kwargs)),
        )
        monkeypatch.setattr(
            wandb_tmp_logger.experiment, "define_metric", lambda *args, **kwargs: None
        )

        wandb_tmp_logger.log_metrics({"eval/reward": 1.0})
        wandb_tmp_logger.log_metrics({"train/loss": 0.5})
        wandb_tmp_logger.log_metrics({"eval/reward": 2.0})

        assert logged[0][0]["eval/step"] == 0
        assert logged[1][0]["train/step"] == 0
        assert logged[2][0]["eval/step"] == 1

    def test_override_global_step_uses_legacy_wandb_step(
        self, wandb_tmp_logger, monkeypatch
    ):
        logged = []
        defined = []
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "log",
            lambda payload, **kwargs: logged.append((payload, kwargs)),
        )
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "define_metric",
            lambda name, step_metric=None: defined.append((name, step_metric)),
        )

        wandb_tmp_logger.log_metrics(
            {"eval/reward": 1.0}, step=123, override_global_step=True
        )

        assert logged == [({"eval/reward": 1.0}, {"step": 123})]
        assert defined == []

    @pytest.mark.skipif(not _has_moviepy, reason="moviepy not installed")
    def test_log_video_uses_inferred_step(self, wandb_tmp_logger, monkeypatch):
        logged = []
        defined = []
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "log",
            lambda payload, **kwargs: logged.append((payload, kwargs)),
        )
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "define_metric",
            lambda name, step_metric=None: defined.append((name, step_metric)),
        )

        video = torch.randint(0, 255, (1, 4, 3, 8, 8), dtype=torch.uint8)
        wandb_tmp_logger.log_video("eval/video", video, step=11)

        payload, kwargs = logged[0]
        assert payload["eval/step"] == 11
        assert "eval/video" in payload
        assert kwargs == {}
        assert defined == [("eval/step", None), ("eval/video", "eval/step")]

    def test_log_histogram_uses_inferred_step(self, wandb_tmp_logger, monkeypatch):
        logged = []
        defined = []
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "log",
            lambda payload, **kwargs: logged.append((payload, kwargs)),
        )
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "define_metric",
            lambda name, step_metric=None: defined.append((name, step_metric)),
        )

        wandb_tmp_logger.log_histogram("eval/hist", torch.randn(10), step=3, bins=4)

        payload, kwargs = logged[0]
        assert payload["eval/step"] == 3
        assert "eval/hist" in payload
        assert kwargs == {}
        assert defined == [("eval/step", None), ("eval/hist", "eval/step")]

    def test_log_str_uses_inferred_step(self, wandb_tmp_logger, monkeypatch):
        logged = []
        defined = []
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "log",
            lambda payload, **kwargs: logged.append((payload, kwargs)),
        )
        monkeypatch.setattr(
            wandb_tmp_logger.experiment,
            "define_metric",
            lambda name, step_metric=None: defined.append((name, step_metric)),
        )

        wandb_tmp_logger.log_str("eval/text", "hello", step=9)

        assert logged == [({"eval/text": "hello", "eval/step": 9}, {})]
        assert defined == [("eval/step", None), ("eval/text", "eval/step")]


@pytest.fixture
def mlflow_fixture():
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as log_dir:
        exp_name = "ramala"
        artifact_uri = pathlib.Path(log_dir).as_uri()
        # MLflow >= 3.10 no longer supports the filesystem tracking backend, so
        # we use a SQLite database for tracking and keep artifacts on disk.
        tracking_uri = f"sqlite:///{log_dir}/mlflow.db"
        logger = MLFlowLogger(
            exp_name=exp_name,
            tracking_uri=tracking_uri,
            artifact_location=artifact_uri,
        )
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
    @pytest.mark.skipif(not _has_mp4, reason="no mp4 video backend available")
    def test_log_video(self, steps, mlflow_fixture):

        logger, client = mlflow_fixture
        videos = torch.cat(
            (
                torch.full((3, 64, 3, 32, 32), 255, dtype=torch.uint8),
                torch.zeros(3, 64, 3, 32, 32, dtype=torch.uint8),
            ),
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
                from torchcodec.decoders import VideoDecoder

                loaded_video = (
                    VideoDecoder(video_path).get_frames_in_range(start=0, stop=128).data
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


@pytest.mark.skipif(not _has_gym, reason="gym required to test rendering")
class TestPixelRenderTransform:
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("in_key", ["pixels", ("nested", "pix")])
    def test_pixel_render(self, parallel, in_key, tmpdir):
        def make_env():
            env = GymEnv("CartPole-v1", render_mode="rgb_array", device=None)
            env = env.append_transform(PixelRenderTransform(out_keys=in_key))
            return env

        try:
            # Try to render an image
            dummy_env = make_env()
            dummy_env.reset()
            dummy_env.base_env._env.render()
        except Exception:
            pytest.skip("Skipping as an exception was raised during rendering.")
        if parallel:
            env = ParallelEnv(2, make_env, mp_start_method="spawn")
        else:
            env = make_env()
        logger = CSVLogger("dummy", log_dir=tmpdir)
        try:
            env = env.append_transform(
                VideoRecorder(logger=logger, in_keys=[in_key], tag="pixels_record")
            )
            check_env_specs(env)
            env.rollout(10)
            env.transform.dump()
            assert os.path.isfile(
                os.path.join(tmpdir, "dummy", "videos", "pixels_record_0.pt")
            )
        finally:
            if not env.is_closed:
                env.close()


@pytest.fixture()
def trackio_logger():
    pytest.importorskip("trackio")
    exp_name = "ramala"
    logger = TrackioLogger(project="test", exp_name=exp_name)
    yield logger
    logger.experiment.finish()
    del logger


@pytest.mark.skipif(not _has_trackio, reason="trackio not installed")
class TestTrackioLogger:
    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_scalar(self, steps, trackio_logger):
        torch.manual_seed(0)

        values = torch.rand(3)
        for i in range(3):
            scalar_name = "foo"
            scalar_value = values[i].item()
            trackio_logger.log_scalar(
                value=scalar_value,
                name=scalar_name,
                step=steps[i] if steps else None,
            )

    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_str(self, steps, trackio_logger):
        for i in range(3):
            trackio_logger.log_str(
                name="foo",
                value="bar",
                step=steps[i] if steps else None,
            )

    def test_log_video(self, trackio_logger):
        torch.manual_seed(0)

        # creating a sample video (T, C, H, W), where T - number of frames,
        # C - number of image channels (e.g. 3 for RGB), H, W - image dimensions.
        # the first 64 frames are black and the next 64 are white
        video = torch.cat(
            (
                torch.zeros(128, 3, 32, 32, dtype=torch.uint8),
                torch.full((128, 3, 32, 32), 255, dtype=torch.uint8),
            )
        )
        video = video[None, :]
        trackio_logger.log_video(
            name="foo",
            video=video,
            fps=4,
            format="mp4",
        )
        trackio_logger.log_video(
            name="foo_16fps",
            video=video,
            fps=16,
            format="mp4",
        )

    def test_log_hparams(self, trackio_logger, config):
        trackio_logger.log_hparams(config)
        for key, value in config.items():
            assert trackio_logger.experiment.config[key] == value

    @pytest.mark.parametrize("steps", [None, [1, 10, 11]])
    def test_log_histogram(self, steps, trackio_logger):
        torch.manual_seed(0)
        for i in range(3):
            data = torch.randn(100)
            trackio_logger.log_histogram(
                "hist", data, step=steps[i] if steps else None, bins=10
            )


@pytest.fixture(autouse=False)
def ray_init_shutdown():
    """Initialize Ray before a test and shut it down after."""
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=2, log_to_driver=False)
    yield
    ray.shutdown()


def test_ray_logger_log_metrics_forwards_override_global_step():
    class _FakeRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.calls[-1]

    class _FakeActor:
        def __init__(self):
            self._execute = _FakeRemoteMethod()

    class _FakeRay:
        @staticmethod
        def get(value, timeout=None):
            del timeout
            return value

    logger = RayLogger.__new__(RayLogger)
    logger._actor = _FakeActor()
    logger._client_id = 3
    logger._sequence = 0
    logger._sequence_lock = threading.Lock()
    logger._ray = _FakeRay()

    result = logger.log_metrics(
        {"loss": torch.tensor(0.5)},
        step=12,
        override_global_step=True,
    )

    assert result == {"loss": 0.5}
    assert logger._actor._execute.calls == [
        (
            (
                3,
                0,
                "log_metrics",
                ({"loss": 0.5},),
                {
                    "step": 12,
                    "keys_sep": "/",
                    "override_global_step": True,
                },
                True,
            ),
            {},
        )
    ]


def test_ray_logger_custom_log_method_returns_synchronously():
    class _FakeRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "remote-result"

    class _FakeActor:
        def __init__(self):
            self._execute = _FakeRemoteMethod()

    class _FakeRay:
        @staticmethod
        def get(value, timeout=None):
            del timeout
            return value

    logger = RayLogger.__new__(RayLogger)
    logger._actor = _FakeActor()
    logger._client_id = 4
    logger._sequence = 0
    logger._sequence_lock = threading.Lock()
    logger._ray = _FakeRay()

    assert logger.log_with_result(4) == "remote-result"
    assert logger._actor._execute.calls == [
        ((4, 0, "log_with_result", (4,), {}, True), {})
    ]


def test_ray_logger_client_sequences_concurrent_submissions_in_call_order():
    class _FakeRemoteMethod:
        def __init__(self):
            self.sequences = []

        def remote(self, client_id, sequence, method, args, kwargs, wait):
            del client_id, method, args, kwargs, wait
            if sequence == 0:
                sleep(0.05)
            self.sequences.append(sequence)

    class _FakeActor:
        def __init__(self):
            self._execute = _FakeRemoteMethod()

    actor = _FakeActor()
    client = _RayLoggerClient(actor, 0, exp_name="test", log_dir=".")
    barrier = threading.Barrier(3)

    def submit():
        barrier.wait()
        client._submit("log_scalar", (), {}, wait=False)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()

    assert actor._execute.sequences == [0, 1]


class TestProcessLogger:
    def test_owner_clients_fifo_video_and_errors(self, tmp_path):
        logger = ProcessLogger(_ProcessTestLogger, tmp_path)
        try:
            assert logger.is_alive
            assert logger.service_backend == "process"
            assert logger.exp_name == "process-test"
            client = logger.client()
            assert not hasattr(client, "start")
            assert not hasattr(client, "shutdown")
            assert not hasattr(client, "client")

            parent_client = logger.client()
            worker_client = logger.client()
            ctx = mp.get_context("spawn")
            worker = ctx.Process(
                target=_log_process_scalars,
                args=(worker_client, "worker", 5),
            )
            worker.start()
            _log_process_scalars(parent_client, "parent", 5)
            worker.join(timeout=20)
            assert worker.exitcode == 0
            logger.flush(timeout=20)

            construction_lines = (tmp_path / "constructed").read_text().splitlines()
            assert construction_lines == ["1"]
            event_lines = (tmp_path / "events").read_text().splitlines()
            for name in ("parent", "worker"):
                assert [line for line in event_lines if line.startswith(name)] == [
                    f"{name}:{step}:{step}" for step in range(5)
                ]

            video = torch.arange(2 * 3 * 4 * 5, dtype=torch.uint8).reshape(
                1, 2, 3, 4, 5
            )
            client.log_video("eval/video", video, step=7, fps=8)
            payload = torch.load(tmp_path / "video.pt", weights_only=True)
            assert payload["name"] == "eval/video"
            assert payload["step"] == 7
            assert payload["kwargs"] == {"fps": 8}
            torch.testing.assert_close(payload["video"], video)
            assert payload["video"].dtype is torch.uint8

            with pytest.raises(RuntimeError, match="expected logger failure"):
                client.log_fail()
            assert client.log_with_result(4) == 5
        finally:
            logger.shutdown(timeout=20)
        assert not logger.is_alive
        logger.shutdown()

    def test_csv_logger_service_backend(self, tmp_path):
        logger = CSVLogger(
            exp_name="process-csv",
            log_dir=tmp_path,
            service_backend="process",
            service_backend_options={"max_queue_size": 4},
        )
        try:
            assert isinstance(logger, ProcessLogger)
            assert isinstance(logger, CSVLogger)
            logger.log_scalar("loss", 1.0, step=0)
            assert (tmp_path / "process-csv" / "scalars" / "loss.csv").is_file()
        finally:
            logger.shutdown(timeout=20)

    def test_dead_process_is_reported_without_an_unbounded_flush(self, tmp_path):
        logger = ProcessLogger(_ProcessTestLogger, tmp_path)
        process = logger._process
        process.terminate()
        process.join(timeout=10)
        try:
            with pytest.raises(MailboxPeerClosedError, match="peer closed"):
                logger.flush()
        finally:
            logger.shutdown(timeout=2)

    def test_restart_waits_for_readiness_and_replaces_the_monitor(self, tmp_path):
        logger = ProcessLogger(_SlowRestartProcessTestLogger, tmp_path)
        process = logger._process
        previous_monitor = logger._process_monitor
        process.terminate()
        process.join(timeout=10)
        try:
            logger.start()
            assert not previous_monitor.is_alive()
            assert logger.is_alive
            assert logger._service_alive.is_set()
            logger.log_scalar("after-restart", 1.0, step=1)

            construction_lines = (tmp_path / "constructed").read_text().splitlines()
            assert construction_lines == ["1", "1"]
            assert (tmp_path / "events").read_text().splitlines() == [
                "after-restart:1.0:1"
            ]
        finally:
            logger.shutdown(timeout=20)

    def test_shutdown_preserves_the_first_error(self, tmp_path, monkeypatch):
        logger = ProcessLogger(_ProcessTestLogger, tmp_path)
        original_submit = logger._submit
        original_manager_shutdown = logger._manager.shutdown

        def submit(method, *args, **kwargs):
            if method == "__flush__":
                raise RuntimeError("primary flush failure")
            return original_submit(method, *args, **kwargs)

        def manager_shutdown():
            original_manager_shutdown()
            raise RuntimeError("manager teardown failure")

        monkeypatch.setattr(logger, "_submit", submit)
        monkeypatch.setattr(logger._manager, "shutdown", manager_shutdown)
        with pytest.raises(RuntimeError, match="primary flush failure"):
            logger.shutdown(timeout=20)


def test_use_ray_service_warns_once_with_removal_version(monkeypatch):
    sentinel = object()

    def _service_factory(
        service_backend,
        *args,
        service_backend_options=None,
        **kwargs,
    ):
        del args, service_backend_options, kwargs
        assert service_backend == "ray"
        return sentinel

    monkeypatch.setattr(CSVLogger, "_ServiceClass", staticmethod(_service_factory))
    with pytest.warns(FutureWarning) as warnings:
        result = CSVLogger(
            exp_name="deprecated-ray",
            log_dir=".",
            use_ray_service=True,
        )
    assert result is sentinel
    assert len(warnings) == 1
    assert "removed in v0.16" in str(warnings[0].message)


def test_video_recorder_uses_service_client_and_default_vector_grid():
    class _CaptureClient:
        def __init__(self):
            self.payload = None

        def log_video(self, **kwargs):
            self.payload = kwargs

    class _LoggerOwner:
        def __init__(self):
            self.log_client = _CaptureClient()

        def client(self):
            return self.log_client

    owner = _LoggerOwner()
    recorder = VideoRecorder(owner, tag="eval/video")
    assert recorder.logger is owner.log_client
    assert recorder.in_keys == ["pixels"]

    # A recorder attached to an environment sees the vector batch at each
    # step and records it as one synchronized grid frame.
    recorder.__dict__["_parent"] = type("_VectorEnv", (), {"batch_size": (4,)})()
    pixels = torch.zeros(4, 3, 8, 8, dtype=torch.uint8)
    pixels[:, :, :, :] = torch.arange(4, dtype=torch.uint8).view(4, 1, 1, 1)
    recorder._apply_transform(pixels)
    assert len(recorder.obs) == 1
    recorder.dump(step=12)

    payload = owner.log_client.payload
    assert payload["name"] == "eval/video"
    assert payload["step"] == 12
    assert payload["video"].shape[:3] == (1, 1, 3)
    assert payload["video"].dtype is torch.uint8


@pytest.mark.skipif(not _has_ray, reason="Ray not available")
class TestRayLogger:
    @pytest.fixture(autouse=True)
    def _setup_ray(self, ray_init_shutdown):
        pass

    def test_csv_logger_returns_ray_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(exp_name="test", log_dir=tmpdir, use_ray_service=True)
            assert isinstance(logger, RayLogger)
            # The metaclass __instancecheck__ makes isinstance transparent
            assert isinstance(logger, CSVLogger)
            assert isinstance(logger, Logger)
            assert not isinstance(logger, WandbLogger)

    def test_csv_logger_returns_csv_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(exp_name="test", log_dir=tmpdir)
            assert isinstance(logger, CSVLogger)

    def test_log_scalar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_scalar", log_dir=tmpdir, use_ray_service=True
            )
            logger.log_scalar("loss", 0.5, step=0)
            logger.log_scalar("loss", 0.3, step=1)

            scalars_dir = os.path.join(tmpdir, "test_scalar", "scalars")
            assert os.path.isdir(scalars_dir)
            assert os.path.isfile(os.path.join(scalars_dir, "loss.csv"))

    def test_errors_are_synchronous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_sync_errors",
                log_dir=tmpdir,
                service_backend="ray",
            )
            try:
                with pytest.raises(RuntimeError, match="Logging histograms"):
                    logger.log_histogram("values", [1, 2, 3])
            finally:
                logger.shutdown()

    def test_log_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_metrics", log_dir=tmpdir, use_ray_service=True
            )
            metrics = {"train/loss": 0.5, "train/reward": 1.0}
            result = logger.log_metrics(metrics, step=0)
            assert result == metrics

    def test_log_metrics_with_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_tensor_metrics", log_dir=tmpdir, use_ray_service=True
            )
            metrics = {"loss": torch.tensor(0.5), "reward": torch.tensor(1.0)}
            result = logger.log_metrics(metrics, step=0)
            assert isinstance(result["loss"], float)
            assert isinstance(result["reward"], float)

    def test_log_hparams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_hparams", log_dir=tmpdir, use_ray_service=True
            )
            logger.log_hparams({"lr": 0.001, "batch_size": 32})

    def test_log_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_video",
                log_dir=tmpdir,
                video_format="pt",
                use_ray_service=True,
            )
            video = torch.randint(0, 255, (1, 10, 3, 64, 64), dtype=torch.uint8)
            logger.log_video("test_vid", video, step=0)

            videos_dir = os.path.join(tmpdir, "test_video", "videos")
            assert os.path.isdir(videos_dir)

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_repr", log_dir=tmpdir, use_ray_service=True
            )
            r = repr(logger)
            assert isinstance(r, str)
            assert "CSVLogger" in r

    def test_non_logging_method_is_not_exposed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_getattr", log_dir=tmpdir, use_ray_service=True
            )
            with pytest.raises(AttributeError):
                logger.client().print_log_dir()

    def test_private_attr_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_private", log_dir=tmpdir, use_ray_service=True
            )
            assert isinstance(logger, RayLogger)
            with pytest.raises(AttributeError):
                logger._nonexistent

    def test_ray_actor_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_options",
                log_dir=tmpdir,
                use_ray_service=True,
                ray_actor_options={"num_cpus": 1},
            )
            assert isinstance(logger, RayLogger)
            logger.log_scalar("x", 1.0, step=0)

    def test_exp_name_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="my_experiment", log_dir=tmpdir, use_ray_service=True
            )
            assert logger.exp_name == "my_experiment"

    def test_log_dir_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                exp_name="test_logdir", log_dir=tmpdir, use_ray_service=True
            )
            assert logger.log_dir == tmpdir

    def test_get_logger_with_ray(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_logger("csv", tmpdir, "test_get_logger", use_ray_service=True)
            assert isinstance(logger, RayLogger)
            logger.log_scalar("x", 1.0, step=0)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
