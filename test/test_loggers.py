import argparse
import os.path
import tempfile
from time import sleep

import pytest
import torch
from torchrl.trainers.loggers.csv import CSVLogger
from torchrl.trainers.loggers.tensorboard import TensorboardLogger, _has_tb
from torchrl.trainers.loggers.wandb import WandbLogger, _has_wandb


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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
