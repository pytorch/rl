# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains distributed tests which are expected to be a considerable burden for the CI
====================================================================================
"""
import abc
import argparse
import os
import sys
import time

import pytest
from tensordict.nn import TensorDictModuleBase
from torchrl._utils import logger as torchrl_logger

try:
    import ray

    _has_ray = True
    RAY_ERR = None
except ModuleNotFoundError as err:
    _has_ray = False
    RAY_ERR = err

import torch

from mocking_classes import ContinuousActionVecMockEnv, CountingEnv
from torch import multiprocessing as mp, nn

from torchrl.collectors.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.collectors.distributed import (
    DistributedDataCollector,
    DistributedSyncDataCollector,
    RayCollector,
    RPCDataCollector,
)
from torchrl.collectors.distributed.ray import DEFAULT_RAY_INIT_CONFIG
from torchrl.envs.utils import RandomPolicy

TIMEOUT = 200

if sys.platform.startswith("win"):
    pytest.skip("skipping windows tests in windows", allow_module_level=True)


class CountingPolicy(TensorDictModuleBase):
    """A policy for counting env.

    Returns a step of 1 by default but weights can be adapted.

    """

    def __init__(self):
        weight = 1.0
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))
        self.in_keys = []
        self.out_keys = ["action"]

    def forward(self, tensordict):
        tensordict.set("action", self.weight.expand(tensordict.shape).clone())
        return tensordict


class DistributedCollectorBase:
    @classmethod
    @abc.abstractmethod
    def distributed_class(self) -> type:
        raise ImportError

    @classmethod
    @abc.abstractmethod
    def distributed_kwargs(self) -> dict:
        raise ImportError

    @classmethod
    @abc.abstractmethod
    def _start_worker(cls):
        raise NotImplementedError

    @classmethod
    def _test_distributed_collector_basic(cls, queue, frames_per_batch):
        cls._start_worker()
        env = ContinuousActionVecMockEnv
        policy = RandomPolicy(env().action_spec)
        torchrl_logger.info("creating collector")
        collector = cls.distributed_class()(
            [env] * 2,
            policy,
            total_frames=1000,
            frames_per_batch=frames_per_batch,
            **cls.distributed_kwargs(),
        )
        total = 0
        torchrl_logger.info("getting data...")
        for data in collector:
            total += data.numel()
            assert data.numel() == frames_per_batch
        assert data.names[-1] == "time"
        collector.shutdown()
        assert total == 1000
        queue.put("passed")

    @pytest.mark.parametrize("frames_per_batch", [50, 100])
    def test_distributed_collector_basic(self, frames_per_batch):
        """Basic functionality test."""
        queue = mp.Queue(1)
        proc = mp.Process(
            target=self._test_distributed_collector_basic,
            args=(queue, frames_per_batch),
        )
        proc.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "passed"
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

    @classmethod
    def _test_distributed_collector_mult(cls, queue, frames_per_batch):
        cls._start_worker()
        env = ContinuousActionVecMockEnv
        policy = RandomPolicy(env().action_spec)
        collector = cls.distributed_class()(
            [env] * 2,
            policy,
            total_frames=1000,
            frames_per_batch=frames_per_batch,
            **cls.distributed_kwargs(),
        )
        total = 0
        for data in collector:
            total += data.numel()
            assert data.numel() == frames_per_batch
        collector.shutdown()
        assert total == -frames_per_batch * (1000 // -frames_per_batch)
        queue.put("passed")

    def test_distributed_collector_mult(self, frames_per_batch=200):
        """Testing multiple nodes."""
        time.sleep(1.0)
        queue = mp.Queue(1)
        proc = mp.Process(
            target=self._test_distributed_collector_mult,
            args=(queue, frames_per_batch),
        )
        proc.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "passed"
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

    @classmethod
    def _test_distributed_collector_sync(cls, queue, sync):
        frames_per_batch = 50
        env = ContinuousActionVecMockEnv
        policy = RandomPolicy(env().action_spec)
        collector = cls.distributed_class()(
            [env] * 2,
            policy,
            total_frames=200,
            frames_per_batch=frames_per_batch,
            sync=sync,
            **cls.distributed_kwargs(),
        )
        total = 0
        for data in collector:
            total += data.numel()
            assert data.numel() == frames_per_batch
        collector.shutdown()
        assert total == 200
        queue.put("passed")

    @pytest.mark.parametrize("sync", [False, True])
    def test_distributed_collector_sync(self, sync):
        """Testing sync and async."""
        queue = mp.Queue(1)
        proc = mp.Process(
            target=TestDistributedCollector._test_distributed_collector_sync,
            args=(queue, sync),
        )
        proc.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "passed"
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

    @classmethod
    def _test_distributed_collector_class(cls, queue, collector_class):
        frames_per_batch = 50
        env = ContinuousActionVecMockEnv
        policy = RandomPolicy(env().action_spec)
        collector = cls.distributed_class()(
            [env] * 2,
            policy,
            collector_class=collector_class,
            total_frames=200,
            frames_per_batch=frames_per_batch,
            **cls.distributed_kwargs(),
        )
        total = 0
        for data in collector:
            total += data.numel()
            assert data.numel() == frames_per_batch
        collector.shutdown()
        assert total == 200
        queue.put("passed")

    @pytest.mark.parametrize(
        "collector_class",
        [
            MultiSyncDataCollector,
            MultiaSyncDataCollector,
            SyncDataCollector,
        ],
    )
    def test_distributed_collector_class(self, collector_class):
        """Testing various collector classes to be used in nodes."""
        queue = mp.Queue(1)
        proc = mp.Process(
            target=self._test_distributed_collector_class,
            args=(queue, collector_class),
        )
        proc.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "passed"
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

    @classmethod
    def _test_distributed_collector_updatepolicy(cls, queue, collector_class, sync):
        frames_per_batch = 50
        total_frames = 300
        env = CountingEnv
        policy = CountingPolicy()
        if collector_class is MultiaSyncDataCollector:
            # otherwise we may collect data from a collector that has not yet been
            # updated
            n_collectors = 1
        else:
            n_collectors = 2
        collector = cls.distributed_class()(
            [env] * n_collectors,
            policy,
            collector_class=collector_class,
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            sync=sync,
            **cls.distributed_kwargs(),
        )
        total = 0
        first_batch = None
        last_batch = None
        for i, data in enumerate(collector):
            total += data.numel()
            assert data.numel() == frames_per_batch
            if i == 0:
                first_batch = data
                policy.weight.data += 1
                collector.update_policy_weights_()
            elif total == total_frames - frames_per_batch:
                last_batch = data
        assert (first_batch["action"] == 1).all(), first_batch["action"]
        assert (last_batch["action"] == 2).all(), last_batch["action"]
        collector.shutdown()
        assert total == total_frames
        queue.put("passed")

    @pytest.mark.parametrize(
        "collector_class",
        [
            SyncDataCollector,
            MultiSyncDataCollector,
            MultiaSyncDataCollector,
        ],
    )
    @pytest.mark.parametrize("sync", [False, True])
    def test_distributed_collector_updatepolicy(self, collector_class, sync):
        """Testing various collector classes to be used in nodes."""
        queue = mp.Queue(1)

        proc = mp.Process(
            target=self._test_distributed_collector_updatepolicy,
            args=(queue, collector_class, sync),
        )
        proc.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "passed"
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()


class TestDistributedCollector(DistributedCollectorBase):
    @classmethod
    def distributed_class(cls) -> type:
        return DistributedDataCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        return {"launcher": "mp", "tcp_port": "4324"}

    @classmethod
    def _start_worker(cls):
        pass


class TestRPCCollector(DistributedCollectorBase):
    @classmethod
    def distributed_class(cls) -> type:
        return RPCDataCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        return {"launcher": "mp", "tcp_port": "4324"}

    @classmethod
    def _start_worker(cls):
        os.environ["RCP_IDLE_TIMEOUT"] = "10"


class TestSyncCollector(DistributedCollectorBase):
    @classmethod
    def distributed_class(cls) -> type:
        return DistributedSyncDataCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        return {"launcher": "mp", "tcp_port": "4324"}

    @classmethod
    def _start_worker(cls):
        os.environ["RCP_IDLE_TIMEOUT"] = "10"

    def test_distributed_collector_sync(self, *args):
        raise pytest.skip("skipping as only sync is supported")

    @classmethod
    def _test_distributed_collector_updatepolicy(
        cls, queue, collector_class, update_interval
    ):
        frames_per_batch = 50
        total_frames = 300
        env = CountingEnv
        policy = CountingPolicy()
        collector = cls.distributed_class()(
            [env] * 2,
            policy,
            collector_class=collector_class,
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            update_interval=update_interval,
            **cls.distributed_kwargs(),
        )
        total = 0
        first_batch = None
        last_batch = None
        for i, data in enumerate(collector):
            total += data.numel()
            assert data.numel() == frames_per_batch
            if i == 0:
                first_batch = data
                policy.weight.data += 1
            elif total == total_frames - frames_per_batch:
                last_batch = data
        assert (first_batch["action"] == 1).all(), first_batch["action"]
        if update_interval == 1:
            assert (last_batch["action"] == 2).all(), last_batch["action"]
        else:
            assert (last_batch["action"] == 1).all(), last_batch["action"]
        collector.shutdown()
        assert total == total_frames
        queue.put("passed")

    @pytest.mark.parametrize(
        "collector_class",
        [
            SyncDataCollector,
            MultiSyncDataCollector,
            MultiaSyncDataCollector,
        ],
    )
    @pytest.mark.parametrize("update_interval", [1_000_000, 1])
    def test_distributed_collector_updatepolicy(self, collector_class, update_interval):
        """Testing various collector classes to be used in nodes."""
        queue = mp.Queue(1)

        proc = mp.Process(
            target=self._test_distributed_collector_updatepolicy,
            args=(queue, collector_class, update_interval),
        )
        proc.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "passed"
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()


@pytest.mark.skipif(not _has_ray, reason=f"Ray not found (error: {RAY_ERR})")
class TestRayCollector(DistributedCollectorBase):
    """A testing distributed data collector class that runs tests without using a Queue,
    to avoid potential deadlocks when combining Ray and multiprocessing.
    """

    @classmethod
    def distributed_class(cls) -> type:
        return RayCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        ray.shutdown()  # make sure ray is not running
        ray_init_config = DEFAULT_RAY_INIT_CONFIG
        ray_init_config["runtime_env"] = {
            "working_dir": os.path.dirname(__file__),
            "env_vars": {"PYTHONPATH": os.path.dirname(__file__)},
            "pip": ["ray"],
        }  # for ray workers
        remote_configs = {
            "num_cpus": 1,
            "num_gpus": 0.0,
            "memory": 1024**2,
        }
        return {"ray_init_config": ray_init_config, "remote_configs": remote_configs}

    @classmethod
    def _start_worker(cls):
        pass

    @pytest.mark.parametrize("sync", [False, True])
    def test_distributed_collector_sync(self, sync, frames_per_batch=200):
        frames_per_batch = 50
        env = ContinuousActionVecMockEnv
        policy = RandomPolicy(env().action_spec)
        collector = self.distributed_class()(
            [env] * 2,
            policy,
            total_frames=200,
            frames_per_batch=frames_per_batch,
            sync=sync,
            **self.distributed_kwargs(),
        )
        total = 0
        for data in collector:
            total += data.numel()
            assert data.numel() == frames_per_batch
        collector.shutdown()
        assert total == 200

    @pytest.mark.parametrize(
        "collector_class",
        [
            MultiSyncDataCollector,
            MultiaSyncDataCollector,
            SyncDataCollector,
        ],
    )
    def test_distributed_collector_class(self, collector_class):
        frames_per_batch = 50
        env = ContinuousActionVecMockEnv
        policy = RandomPolicy(env().action_spec)
        collector = self.distributed_class()(
            [env] * 2,
            policy,
            collector_class=collector_class,
            total_frames=200,
            frames_per_batch=frames_per_batch,
            **self.distributed_kwargs(),
        )
        total = 0
        for data in collector:
            total += data.numel()
            assert data.numel() == frames_per_batch
        collector.shutdown()
        assert total == 200

    @pytest.mark.parametrize(
        "collector_class",
        [
            SyncDataCollector,
            MultiSyncDataCollector,
            MultiaSyncDataCollector,
        ],
    )
    @pytest.mark.parametrize("sync", [False, True])
    def test_distributed_collector_updatepolicy(self, collector_class, sync):
        frames_per_batch = 50
        total_frames = 300
        env = CountingEnv
        policy = CountingPolicy()
        if collector_class is MultiaSyncDataCollector:
            # otherwise we may collect data from a collector that has not yet been
            # updated
            n_collectors = 1
        else:
            n_collectors = 2
        collector = self.distributed_class()(
            [env] * n_collectors,
            policy,
            collector_class=collector_class,
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            sync=sync,
            **self.distributed_kwargs(),
        )
        total = 0
        first_batch = None
        last_batch = None
        for i, data in enumerate(collector):
            total += data.numel()
            assert data.numel() == frames_per_batch
            if i == 0:
                first_batch = data
                policy.weight.data += 1
                collector.update_policy_weights_()
            elif total == total_frames - frames_per_batch:
                last_batch = data
        assert (first_batch["action"] == 1).all(), first_batch["action"]
        assert (last_batch["action"] == 2).all(), last_batch["action"]
        collector.shutdown()
        assert total == total_frames


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
