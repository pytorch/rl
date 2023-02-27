"""
Contains distributed tests which are expected to be a considerable burden for the CI
====================================================================================
"""
import abc
import argparse
import os
import time

import pytest

from mocking_classes import ContinuousActionVecMockEnv
from torch import multiprocessing as mp

from torchrl.collectors.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
    RandomPolicy,
    SyncDataCollector,
)
from torchrl.collectors.distributed import DistributedDataCollector, RPCDataCollector


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
        env = ContinuousActionVecMockEnv()
        policy = RandomPolicy(env.action_spec)
        collector = cls.distributed_class()(
            [env],
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
            out = queue.get(timeout=100)
            assert out == "passed"
        finally:
            proc.join()
            queue.close()

    @classmethod
    def _test_distributed_collector_mult(cls, queue, frames_per_batch):
        cls._start_worker()
        env = ContinuousActionVecMockEnv()
        policy = RandomPolicy(env.action_spec)
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

    def test_distributed_collector_mult(self, frames_per_batch=300):
        """Testing multiple nodes."""
        time.sleep(1.0)
        queue = mp.Queue(1)
        proc = mp.Process(
            target=self._test_distributed_collector_mult,
            args=(queue, frames_per_batch),
        )
        proc.start()
        try:
            out = queue.get(timeout=100)
            assert out == "passed"
        finally:
            proc.join()
            queue.close()

    @classmethod
    def _test_distributed_collector_sync(cls, queue, sync):
        frames_per_batch = 50
        env = ContinuousActionVecMockEnv()
        policy = RandomPolicy(env.action_spec)
        collector = cls.distributed_class()(
            [env],
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
            out = queue.get(timeout=100)
            assert out == "passed"
        finally:
            proc.join()
            queue.close()

    @classmethod
    def _test_distributed_collector_class(cls, queue, collector_class):
        frames_per_batch = 50
        env = ContinuousActionVecMockEnv()
        policy = RandomPolicy(env.action_spec)
        collector = cls.distributed_class()(
            [env],
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
            out = queue.get(timeout=100)
            assert out == "passed"
        finally:
            proc.join()
            queue.close()


class TestDistributedCollector(DistributedCollectorBase):
    @classmethod
    def distributed_class(cls) -> type:
        return DistributedDataCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        return {"launcher": "mp", "tcp_port": "1234"}

    @classmethod
    def _start_worker(cls):
        pass


class TestRPCCollector(DistributedCollectorBase):
    @classmethod
    def distributed_class(cls) -> type:
        return RPCDataCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        return {"launcher": "mp", "tcp_port": "1234"}

    @classmethod
    def _start_worker(cls):
        os.environ["RCP_IDLE_TIMEOUT"] = "10"


#
# class TestRPCCollector:
#     @staticmethod
#     def _test_distributed_collector_basic(queue, frames_per_batch):
#         os.environ["RCP_IDLE_TIMEOUT"] = "10"
#         env = ContinuousActionVecMockEnv()
#         policy = RandomPolicy(env.action_spec)
#         collector = RPCDataCollector(
#             [env],
#             policy,
#             total_frames=1000,
#             frames_per_batch=frames_per_batch,
#             launcher="mp",
#             tcp_port=str(1234),
#         )
#         total = 0
#         for data in collector:
#             total += data.numel()
#             assert data.numel() == frames_per_batch
#         collector.shutdown()
#         assert total == 1000
#         queue.put("passed")
#
#     @pytest.mark.parametrize("frames_per_batch", [50, 100])
#     def test_distributed_collector_basic(self, frames_per_batch):
#         """Basic functionality test."""
#         time.sleep(1.0)
#         queue = mp.Queue(1)
#         proc = mp.Process(
#             target=TestRPCCollector._test_distributed_collector_basic,
#             args=(queue, frames_per_batch),
#         )
#         proc.start()
#         try:
#             out = queue.get(timeout=100)
#             assert out == "passed"
#         finally:
#             proc.join()
#             queue.close()
#
#     @staticmethod
#     def _test_distributed_collector_mult(queue, frames_per_batch):
#         os.environ["RCP_IDLE_TIMEOUT"] = "10"
#         env = ContinuousActionVecMockEnv()
#         policy = RandomPolicy(env.action_spec)
#         collector = RPCDataCollector(
#             [env] * 2,
#             policy,
#             total_frames=1000,
#             frames_per_batch=frames_per_batch,
#             launcher="mp",
#             tcp_port=str(1234),
#         )
#         total = 0
#         for data in collector:
#             total += data.numel()
#             assert data.numel() == frames_per_batch
#         collector.shutdown()
#         assert total == -frames_per_batch * (1000 // -frames_per_batch)
#         queue.put("passed")
#
#     def test_distributed_collector_mult(self, frames_per_batch=300):
#         """Testing multiple nodes."""
#         time.sleep(1.0)
#         queue = mp.Queue(1)
#         proc = mp.Process(
#             target=TestRPCCollector._test_distributed_collector_mult,
#             args=(queue, frames_per_batch),
#         )
#         proc.start()
#         try:
#             out = queue.get(timeout=100)
#             assert out == "passed"
#         finally:
#             proc.join()
#             queue.close()
#
#     @staticmethod
#     def _test_distributed_collector_sync(queue, sync):
#         os.environ["RCP_IDLE_TIMEOUT"] = "10"
#         frames_per_batch = 50
#         env = ContinuousActionVecMockEnv()
#         policy = RandomPolicy(env.action_spec)
#         collector = RPCDataCollector(
#             [env],
#             policy,
#             total_frames=200,
#             frames_per_batch=frames_per_batch,
#             launcher="mp",
#             sync=sync,
#             tcp_port=str(1234),
#         )
#         total = 0
#         for data in collector:
#             total += data.numel()
#             assert data.numel() == frames_per_batch
#         collector.shutdown()
#         assert total == 200
#         queue.put("passed")
#
#     @pytest.mark.parametrize("sync", [False, True])
#     def test_distributed_collector_sync(self, sync):
#         """Testing sync and async."""
#         time.sleep(1.0)
#         queue = mp.Queue(1)
#         proc = mp.Process(
#             target=TestRPCCollector._test_distributed_collector_sync,
#             args=(queue, sync),
#         )
#         proc.start()
#         try:
#             out = queue.get(timeout=100)
#             assert out == "passed"
#         finally:
#             proc.join()
#             queue.close()
#
#     @staticmethod
#     def _test_distributed_collector_class(queue, collector_class):
#         os.environ["RCP_IDLE_TIMEOUT"] = "10"
#         frames_per_batch = 50
#         env = ContinuousActionVecMockEnv()
#         policy = RandomPolicy(env.action_spec)
#         collector = RPCDataCollector(
#             [env],
#             policy,
#             collector_class=collector_class,
#             total_frames=200,
#             frames_per_batch=frames_per_batch,
#             launcher="mp",
#             tcp_port=str(1234),
#         )
#         total = 0
#         for data in collector:
#             total += data.numel()
#             assert data.numel() == frames_per_batch
#         collector.shutdown()
#         assert total == 200
#         queue.put("passed")
#
#     @pytest.mark.parametrize(
#         "collector_class",
#         [
#             MultiSyncDataCollector,
#             MultiaSyncDataCollector,
#             SyncDataCollector,
#         ],
#     )
#     def test_distributed_collector_class(self, collector_class):
#         """Testing various collector classes to be used in nodes."""
#         queue = mp.Queue(1)
#         time.sleep(1.0)
#         proc = mp.Process(
#             target=TestRPCCollector._test_distributed_collector_class,
#             args=(queue, collector_class),
#         )
#         proc.start()
#         try:
#             out = queue.get(timeout=100)
#             assert out == "passed"
#         finally:
#             proc.join()
#             queue.close()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
