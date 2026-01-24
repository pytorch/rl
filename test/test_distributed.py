# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains distributed tests which are expected to be a considerable burden for the CI
====================================================================================
"""
from __future__ import annotations

import abc
import argparse
import importlib
import os
import socket
import sys
import time
import traceback
from functools import partial

import pytest

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential

from torch import multiprocessing as mp, nn
from torchrl._utils import logger as torchrl_logger

from torchrl.collectors import Collector, MultiAsyncCollector, MultiSyncCollector
from torchrl.collectors.distributed import (
    DistributedCollector,
    RayCollector,
    RPCCollector,
)
from torchrl.collectors.distributed.ray import DEFAULT_RAY_INIT_CONFIG
from torchrl.data import (
    LazyTensorStorage,
    RandomSampler,
    RayReplayBuffer,
    RoundRobinWriter,
    SamplerWithoutReplacement,
)
from torchrl.modules import RandomPolicy
from torchrl.testing.dist_utils import (
    assert_no_new_python_processes,
    snapshot_python_processes,
)

from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv, CountingEnv

_has_ray = importlib.util.find_spec("ray") is not None

TIMEOUT = 200

if sys.platform.startswith("win"):
    pytest.skip("skipping windows tests in windows", allow_module_level=True)

# pytestmark = [pytest.mark.forked]


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
        try:
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
            for data in collector:
                total += data.numel()
                assert data.numel() == frames_per_batch
            assert data.names[-1] == "time"
            assert total == 1000
            queue.put(("passed", None))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put(("not passed", (e, tb)))
        finally:
            collector.shutdown()

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
            out, maybe_err = queue.get(timeout=TIMEOUT)
            if out != "passed":
                raise RuntimeError(f"Error with stack {maybe_err[1]}") from maybe_err[0]
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

    @classmethod
    def _test_distributed_collector_mult(cls, queue, frames_per_batch):
        try:
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
            queue.put(("passed", None))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put(("not passed", (e, tb)))

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
            out, maybe_err = queue.get(timeout=TIMEOUT)
            if out != "passed":
                raise RuntimeError(f"Error with stack {maybe_err[1]}") from maybe_err[0]
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

    @classmethod
    def _test_distributed_collector_sync(cls, queue, sync):
        try:
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
            queue.put(("passed", None))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put(("not passed", (e, tb)))

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
            out, maybe_err = queue.get(timeout=TIMEOUT)
            if out != "passed":
                raise RuntimeError(f"Error with stack {maybe_err[1]}") from maybe_err[0]
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

    @classmethod
    def _test_distributed_collector_updatepolicy_shutdown_only(cls, queue, sync):
        """Small rollout + weight sync + shutdown (used for leak checks in parent process)."""
        collector = None
        try:
            frames_per_batch = 50
            total_frames = 250
            env = CountingEnv
            policy = CountingPolicy()
            dcls = cls.distributed_class()
            collector = dcls(
                [env] * 2,
                policy,
                collector_class=Collector,
                total_frames=total_frames,
                frames_per_batch=frames_per_batch,
                sync=sync,
                **cls.distributed_kwargs(),
            )
            first_batch = None
            seen_updated = False
            total = 0
            for i, data in enumerate(collector):
                total += data.numel()
                if i == 0:
                    first_batch = data
                    policy.weight.data.add_(1)
                    collector.update_policy_weights_(policy)
                else:
                    if (data["action"] == 2).all():
                        seen_updated = True
            assert total == total_frames
            assert first_batch is not None
            assert (first_batch["action"] == 1).all(), first_batch["action"]
            assert (
                seen_updated
            ), "Updated weights were never observed in collected batches."
            queue.put(("passed", None))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put(("not passed", (e, tb)))
        finally:
            if collector is not None:
                collector.shutdown()

    @pytest.mark.parametrize("sync", [False, True])
    def test_collector_shutdown_clears_python_processes(self, sync):
        """Regression test: collector.shutdown() should not leak python processes."""
        queue = mp.Queue(1)
        # Creating multiprocessing primitives (Queue / SemLock) may spawn Python's
        # `multiprocessing.resource_tracker` helper process. That process is not owned
        # by the collector and may live for the duration of the test runner, so we
        # include it in the baseline.
        baseline = snapshot_python_processes()
        baseline_time = time.time()

        proc = mp.Process(
            target=self._test_distributed_collector_updatepolicy_shutdown_only,
            args=(queue, sync),
        )
        proc.start()
        try:
            out, maybe_err = queue.get(timeout=TIMEOUT)
            if out != "passed":
                raise RuntimeError(f"Error with stack {maybe_err[1]}") from maybe_err[0]
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

        assert_no_new_python_processes(
            baseline=baseline, baseline_time=baseline_time, timeout=20.0
        )

    @classmethod
    def _test_distributed_collector_class(cls, queue, collector_class):
        try:
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
            queue.put(("passed", None))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put(("not passed", (e, tb)))

    @pytest.mark.parametrize(
        "collector_class",
        [
            MultiSyncCollector,
            MultiAsyncCollector,
            Collector,
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
            out, maybe_err = queue.get(timeout=TIMEOUT)
            if out != "passed":
                raise RuntimeError(f"Error with stack {maybe_err[1]}") from maybe_err[0]
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

    @classmethod
    def _test_distributed_collector_updatepolicy(
        cls, queue, collector_class, sync, pfactory
    ):
        try:
            frames_per_batch = 50
            total_frames = 300
            env = CountingEnv
            if pfactory:
                policy_factory = CountingPolicy
                policy = None
            else:
                policy = CountingPolicy()
                policy_factory = None
            if collector_class is MultiAsyncCollector:
                # otherwise we may collect data from a collector that has not yet been
                # updated
                n_collectors = 1
            else:
                n_collectors = 2
            weights = None
            if policy is None and policy_factory is not None:
                policy_stateful = policy_factory()
                weights = TensorDict.from_module(policy_stateful).lock_()
            dcls = cls.distributed_class()
            torchrl_logger.info(f"Using distributed collector {dcls}")
            collector = None
            collector = dcls(
                [env] * n_collectors,
                policy,
                policy_factory=policy_factory,
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
                    if policy is not None:
                        # Avoid using `.data` (and avoid tracking in autograd).
                        policy.weight.data.add_(1)
                    else:
                        assert weights is not None
                        weights.data += 1
                    torchrl_logger.info("TEST -- Calling update_policy_weights_()")
                    collector.update_policy_weights_(weights)
                    torchrl_logger.info("TEST -- Done calling update_policy_weights_()")
                elif total == total_frames - frames_per_batch:
                    last_batch = data
            assert first_batch is not None
            assert last_batch is not None
            assert (first_batch["action"] == 1).all(), first_batch["action"]
            assert (last_batch["action"] == 2).all(), last_batch["action"]
            collector.shutdown()
            assert total == total_frames
            queue.put(("passed", None))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put(("not passed", (e, tb)))

    @pytest.mark.parametrize(
        "collector_class",
        [
            Collector,
            MultiSyncCollector,
            MultiAsyncCollector,
        ],
    )
    @pytest.mark.parametrize("sync", [False, True])
    @pytest.mark.parametrize("pfactory", [False, True])
    def test_distributed_collector_updatepolicy(self, collector_class, sync, pfactory):
        """Testing various collector classes to be used in nodes."""
        queue = mp.Queue(1)

        proc = mp.Process(
            target=self._test_distributed_collector_updatepolicy,
            args=(queue, collector_class, sync, pfactory),
        )
        proc.start()
        try:
            out, maybe_err = queue.get(timeout=TIMEOUT)
            if out != "passed":
                raise RuntimeError(f"Error with stack {maybe_err[1]}") from maybe_err[0]
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()

    @classmethod
    def _test_collector_next_method(cls, queue):
        """Non-regression test for iterator/flag bug.

        Previously, `__iter__` set `_iterator = True` as a flag, but `next()` expected
        `_iterator` to be either `None` or an actual iterator object. When Ray's remote
        collector called `next()` after `__iter__`, it tried `next(True)` which failed
        with `TypeError: 'bool' object is not an iterator`.

        This test ensures that calling `next()` works correctly regardless of whether
        `__iter__` has been called.
        """
        try:
            cls._start_worker()
            env = ContinuousActionVecMockEnv
            policy = RandomPolicy(env().action_spec)
            collector = cls.distributed_class()(
                [env] * 2,
                policy,
                total_frames=500,
                frames_per_batch=50,
                **cls.distributed_kwargs(),
            )

            # Test 1: Call next() directly without __iter__
            data1 = collector.next()
            assert data1 is not None, "next() should return data"
            assert data1.numel() == 50, f"Expected 50 frames, got {data1.numel()}"

            # Test 2: Call next() again
            data2 = collector.next()
            assert data2 is not None, "second next() should return data"
            assert data2.numel() == 50, f"Expected 50 frames, got {data2.numel()}"

            queue.put(("passed", None))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put(("not passed", (e, tb)))
        finally:
            collector.shutdown()

    def test_collector_next_method(self):
        """Non-regression test: next() should work correctly (iterator/flag bug fix)."""
        queue = mp.Queue(1)
        proc = mp.Process(
            target=self._test_collector_next_method,
            args=(queue,),
        )
        proc.start()
        try:
            out, maybe_err = queue.get(timeout=TIMEOUT)
            if out != "passed":
                raise RuntimeError(f"Error with stack {maybe_err[1]}") from maybe_err[0]
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()


class TestDistributedCollector(DistributedCollectorBase):
    @classmethod
    def distributed_class(cls) -> type:
        return DistributedCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        # Pick an ephemeral free TCP port on localhost for each test process to
        # avoid address-in-use errors when tests are run repeatedly or in quick
        # succession.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
        return {"launcher": "mp", "tcp_port": str(port)}

    @classmethod
    def _start_worker(cls):
        pass


class TestRPCCollector(DistributedCollectorBase):
    @classmethod
    def distributed_class(cls) -> type:
        return RPCCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
        return {"launcher": "mp", "tcp_port": str(port)}

    @classmethod
    def _start_worker(cls):
        os.environ["RCP_IDLE_TIMEOUT"] = "10"


class TestSyncCollector(DistributedCollectorBase):
    @classmethod
    def distributed_class(cls) -> type:
        return DistributedCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
        return {"launcher": "mp", "tcp_port": str(port)}

    @classmethod
    def _start_worker(cls):
        os.environ["RCP_IDLE_TIMEOUT"] = "10"

    def test_distributed_collector_sync(self, *args):
        raise pytest.skip("skipping as only sync is supported")

    @pytest.mark.parametrize("sync", [True])
    def test_collector_shutdown_clears_python_processes(self, sync):
        super().test_collector_shutdown_clears_python_processes(sync)

    @classmethod
    def _test_distributed_collector_updatepolicy(
        cls,
        queue,
        collector_class,
        update_interval,
        pfactory,
    ):
        frames_per_batch = 50
        total_frames = 300
        env = CountingEnv
        if pfactory:
            policy_factory = CountingPolicy
        else:
            policy_factory = None
        policy = CountingPolicy()
        collector = cls.distributed_class()(
            [env] * 2,
            policy,
            policy_factory=policy_factory,
            collector_class=collector_class,
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            update_interval=update_interval,
            **cls.distributed_kwargs(),
        )
        try:
            total = 0
            first_batch = None
            last_batch = None
            for i, data in enumerate(collector):
                total += data.numel()
                assert data.numel() == frames_per_batch
                if i == 0:
                    first_batch = data
                    policy.weight.data.add_(1)
                elif total == total_frames - frames_per_batch:
                    last_batch = data
            assert first_batch is not None
            assert last_batch is not None
            assert (first_batch["action"] == 1).all(), first_batch["action"]
            if update_interval == 1:
                assert (last_batch["action"] == 2).all(), last_batch["action"]
            else:
                assert (last_batch["action"] == 1).all(), last_batch["action"]
            assert total == total_frames
            queue.put(("passed", None))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put(("not passed", (e, tb)))
        finally:
            collector.shutdown()

    @pytest.mark.parametrize(
        "collector_class",
        [
            Collector,
            MultiSyncCollector,
            MultiAsyncCollector,
        ],
    )
    @pytest.mark.parametrize("update_interval", [1])
    @pytest.mark.parametrize("pfactory", [False, True])
    def test_distributed_collector_updatepolicy(
        self, collector_class, update_interval, pfactory
    ):
        """Testing various collector classes to be used in nodes."""
        queue = mp.Queue(1)

        proc = mp.Process(
            target=self._test_distributed_collector_updatepolicy,
            args=(queue, collector_class, update_interval, pfactory),
        )
        proc.start()
        try:
            out, maybe_err = queue.get(timeout=TIMEOUT)
            if out != "passed":
                raise RuntimeError(f"Error with stack {maybe_err[1]}") from maybe_err[0]
        finally:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
            queue.close()


@pytest.mark.skipif(
    not _has_ray, reason="Ray not found. Ray may be badly configured or not installed."
)
class TestRayCollector(DistributedCollectorBase):
    """A testing distributed data collector class that runs tests without using a Queue,
    to avoid potential deadlocks when combining Ray and multiprocessing.
    """

    @pytest.fixture(autouse=True, scope="class")
    def start_ray(self):
        import ray
        from torchrl.collectors.distributed.ray import DEFAULT_RAY_INIT_CONFIG

        # Ensure Ray is initialized with a runtime_env that lets workers import
        # this test module (e.g. `CountingPolicy`), otherwise actor unpickling can
        # fail with "No module named 'test_distributed'".
        ray.shutdown()
        ray_init_config = dict(DEFAULT_RAY_INIT_CONFIG)
        ray_init_config["runtime_env"] = {
            "working_dir": os.path.dirname(__file__),
            "env_vars": {"PYTHONPATH": os.path.dirname(__file__)},
        }
        ray.init(**ray_init_config)

        yield
        ray.shutdown()

    @pytest.fixture(autouse=True, scope="function")
    def reset_process_group(self):
        import torch.distributed as dist

        try:
            dist.destroy_process_group()
        except Exception:
            pass
        yield

    @classmethod
    def distributed_class(cls) -> type:
        return RayCollector

    @classmethod
    def distributed_kwargs(cls) -> dict:
        # Ray will be auto-initialized by RayCollector if not already started.
        # We need to provide runtime_env so workers can import this test module.
        ray_init_config = dict(DEFAULT_RAY_INIT_CONFIG)
        ray_init_config["runtime_env"] = {
            "working_dir": os.path.dirname(__file__),
            "env_vars": {"PYTHONPATH": os.path.dirname(__file__)},
        }
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
        try:
            total = 0
            for data in collector:
                total += data.numel()
                assert data.numel() == frames_per_batch
            assert total == 200
        finally:
            collector.shutdown()

    @pytest.mark.parametrize("sync", [False, True])
    def test_collector_shutdown_clears_python_processes(self, sync):
        """Regression test: collector.shutdown() should not leak python processes (ray)."""
        kwargs = self.distributed_kwargs()
        baseline = snapshot_python_processes()
        baseline_time = time.time()

        frames_per_batch = 50
        total_frames = 250
        env = CountingEnv
        policy = CountingPolicy()
        collector = self.distributed_class()(
            [env] * 2,
            policy,
            collector_class=Collector,
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            sync=sync,
            **kwargs,
        )
        try:
            total = 0
            first_batch = None
            seen_updated = False
            for i, data in enumerate(collector):
                total += data.numel()
                if i == 0:
                    first_batch = data
                    policy.weight.data.add_(1)
                    collector.update_policy_weights_(policy)
                else:
                    if (data["action"] == 2).all():
                        seen_updated = True
            assert total == total_frames
            assert first_batch is not None
            assert (first_batch["action"] == 1).all(), first_batch["action"]
            assert (
                seen_updated
            ), "Updated weights were never observed in collected batches."
        finally:
            collector.shutdown()

        def _is_ray_runtime_proc(info):
            args = info.get("args") or ""
            comm = info.get("comm") or ""
            return (
                " ray::" in args.lower()
                or "/site-packages/ray/" in args
                or comm in {"raylet", "gcs_server"}
            )

        assert_no_new_python_processes(
            baseline=baseline,
            baseline_time=baseline_time,
            timeout=30.0,
            # Ray's core daemons and prestarted workers can legitimately outlive a
            # collector. We only want to catch leaked *non-Ray* Python processes
            # spawned by the collector itself.
            ignore_info_fn=_is_ray_runtime_proc,
        )

    @pytest.mark.parametrize(
        "collector_class",
        [
            MultiSyncCollector,
            MultiAsyncCollector,
            Collector,
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
        try:
            total = 0
            for data in collector:
                total += data.numel()
                assert data.numel() == frames_per_batch
            assert total == 200
        finally:
            collector.shutdown()

    @pytest.mark.parametrize(
        "collector_class",
        [
            Collector,
            MultiSyncCollector,
            MultiAsyncCollector,
        ],
    )
    @pytest.mark.parametrize("sync", [False, True])
    @pytest.mark.parametrize("pfactory", [False, True])
    def test_distributed_collector_updatepolicy(self, collector_class, sync, pfactory):
        frames_per_batch = 50
        total_frames = 300
        env = CountingEnv
        if pfactory:
            policy_factory = CountingPolicy
            policy = None
        else:
            policy = CountingPolicy()
            policy_factory = None
        if collector_class is MultiAsyncCollector:
            # otherwise we may collect data from a collector that has not yet been
            # updated
            n_collectors = 1
        else:
            n_collectors = 2
        weights = None
        if policy is None and policy_factory is not None:
            policy_stateful = policy_factory()
            weights = TensorDict.from_module(policy_stateful)
        collector = self.distributed_class()(
            [env] * n_collectors,
            policy,
            policy_factory=policy_factory,
            collector_class=collector_class,
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            sync=sync,
            **self.distributed_kwargs(),
        )
        total = 0
        first_batch = None
        last_batch = None
        try:
            for i, data in enumerate(collector):
                total += data.numel()
                assert data.numel() == frames_per_batch
                if i == 0:
                    first_batch = data
                    if policy is not None:
                        policy.weight.data.add_(1)
                    else:
                        assert weights is not None
                        weights.data.add_(1)
                    collector.update_policy_weights_(weights)
                elif total == total_frames - frames_per_batch:
                    last_batch = data
            assert first_batch is not None
            assert last_batch is not None
            assert (first_batch["action"] == 1).all(), first_batch["action"]
            assert (last_batch["action"] == 2).all(), last_batch["action"]
            assert total == total_frames
        finally:
            collector.shutdown()

    @pytest.mark.parametrize("storage", [None, partial(LazyTensorStorage, 1000)])
    @pytest.mark.parametrize(
        "sampler", [None, partial(RandomSampler), SamplerWithoutReplacement]
    )
    @pytest.mark.parametrize("writer", [None, partial(RoundRobinWriter)])
    def test_ray_replaybuffer(self, storage, sampler, writer):
        kwargs = self.distributed_kwargs()
        kwargs["remote_config"] = kwargs.pop("remote_configs")
        rb = RayReplayBuffer(
            storage=storage,
            sampler=sampler,
            writer=writer,
            batch_size=32,
            **kwargs,
        )
        td = TensorDict(a=torch.arange(100, 200), batch_size=[100])
        index = rb.extend(td)
        assert (index == torch.arange(100)).all()
        for _ in range(10):
            sample = rb.sample()
            if sampler is SamplerWithoutReplacement:
                assert sample["a"].unique().numel() == sample.numel()

    # class CustomCollectorCls(Collector):
    #     def __init__(self, create_env_fn, **kwargs):
    #         policy = lambda td: td.set("action", torch.full(td.shape, 2))
    #         super().__init__(create_env_fn, policy, **kwargs)

    def test_ray_collector_policy_constructor(self):
        n_collectors = 2
        frames_per_batch = 50
        total_frames = 300
        env = CountingEnv

        def policy_constructor():
            return TensorDictSequential(
                TensorDictModule(
                    lambda x: x.float(),
                    in_keys=["observation"],
                    out_keys=["_obs_float"],
                ),
                TensorDictModule(
                    nn.Linear(1, 1), out_keys=["action"], in_keys=["_obs_float"]
                ),
                TensorDictModule(
                    lambda x: x.int(), in_keys=["action"], out_keys=["action"]
                ),
            )

        collector = self.distributed_class()(
            [env] * n_collectors,
            collector_class=Collector,
            policy_factory=policy_constructor,
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            **self.distributed_kwargs(),
        )
        p = policy_constructor()
        # p(env().reset())
        weights = TensorDict.from_module(p)
        # `TensorDict.__getitem__` returns tensors; use in-place ops directly.
        with torch.no_grad():
            weights["module", "1", "module", "weight"].fill_(0)
            weights["module", "1", "module", "bias"].fill_(2)
        collector.update_policy_weights_(weights)
        try:
            for data in collector:
                assert (data["action"] == 2).all()
                collector.update_policy_weights_(weights)
        finally:
            collector.shutdown()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
