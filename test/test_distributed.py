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
import torch.distributed as dist
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
    ReplayBuffer,
    RoundRobinWriter,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)
from torchrl.distributed import DataParallelContext
from torchrl.envs import StepCounter, TransformedEnv
from torchrl.modules import RandomPolicy
from torchrl.objectives import LossModule
from torchrl.testing.dist_utils import (
    assert_no_new_python_processes,
    snapshot_python_processes,
)

from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv, CountingEnv
from torchrl.trainers import Learner, Trainer
from torchrl.trainers.distributed import RayLearnerGroup

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


class _RayLearnerLoss(LossModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, batch):
        return TensorDict(
            {"loss": (self.linear(batch["x"]) - batch["y"]).square().mean()}, []
        )


def _make_ray_test_learner(replay_buffer, data_parallel_context):
    loss = _RayLearnerLoss().to(data_parallel_context.device)
    return Learner(
        loss,
        replay_buffer,
        optimizer=torch.optim.SGD(loss.parameters(), lr=0.05),
        data_parallel_context=data_parallel_context,
        models={"policy": loss.linear},
    )


class _RayCheckpointCollector:
    frames_per_batch = 8
    init_random_frames = 0

    def __init__(self):
        self.state_value = 0
        self.collected_frames = 0

    def state_dict(self):
        return {"state_value": self.state_value}

    def load_state_dict(self, state_dict):
        self.state_value = state_dict["state_value"]

    def update_policy_weights_(self, weights):
        del weights

    def shutdown(self):
        pass


class _FailingRayLearnerLoss(_RayLearnerLoss):
    def forward(self, batch):
        raise RuntimeError("intentional learner failure")


def _make_failing_ray_test_learner(replay_buffer, data_parallel_context):
    loss = _FailingRayLearnerLoss().to(data_parallel_context.device)
    return Learner(
        loss,
        replay_buffer,
        optimizer=torch.optim.SGD(loss.parameters(), lr=0.05),
        data_parallel_context=data_parallel_context,
        models={"policy": loss.linear},
    )


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
        torch.manual_seed(42)
        kwargs = self.distributed_kwargs()
        kwargs["remote_config"] = kwargs.pop("remote_configs")
        kwargs["remote_config"]["num_gpus"] = 0
        kwargs["remote_config"]["runtime_env"] = {
            "env_vars": {"CUDA_VISIBLE_DEVICES": ""},
        }
        rb = RayReplayBuffer(
            storage=storage,
            sampler=sampler,
            writer=writer,
            batch_size=32,
            **kwargs,
        )
        try:
            td = TensorDict(a=torch.arange(100, 200), batch_size=[100])
            index = rb.extend(td)
            assert (index == torch.arange(100)).all()
            assert len(rb) == 100
            for _ in range(10):
                sample = rb.sample()
                if sampler is SamplerWithoutReplacement:
                    assert sample["a"].unique().numel() == sample.numel()
        finally:
            rb.close()

    def test_background_pause_drains_writes_and_resumes(self):
        env = ContinuousActionVecMockEnv
        policy = RandomPolicy(env().action_spec)
        replay_buffer = RayReplayBuffer(
            replay_buffer_cls=TensorDictReplayBuffer,
            storage=partial(LazyTensorStorage, 1000),
            batch_size=16,
            remote_config={"num_cpus": 0},
        )
        collector = RayCollector(
            [env, env],
            policy,
            total_frames=100_000,
            frames_per_batch=16,
            sync=False,
            replay_buffer=replay_buffer,
            **self.distributed_kwargs(),
        )
        try:
            collector.start()
            deadline = time.monotonic() + 30
            while replay_buffer.write_count == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert replay_buffer.write_count > 0

            with collector.pause(timeout=30):
                paused_count = replay_buffer.write_count
                time.sleep(0.2)
                assert replay_buffer.write_count == paused_count

            deadline = time.monotonic() + 30
            while (
                replay_buffer.write_count == paused_count
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert replay_buffer.write_count > paused_count
        finally:
            collector.shutdown()
            replay_buffer.shutdown()

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


@pytest.mark.skipif(
    not _has_ray, reason="Ray not found. Ray may be badly configured or not installed."
)
class TestRayTrajsPerBatch:
    """Tests for trajs_per_batch + replay_buffer on RayCollector."""

    @pytest.fixture(autouse=True, scope="class")
    def start_ray(self):
        import ray

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
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        yield

    def test_ray_trajs_per_batch_replay_buffer_rejects_regular_rb(self):
        """RayCollector rejects a regular ReplayBuffer (must use RayReplayBuffer)."""
        max_steps = 4
        num_trajs = 2

        def env_fn():
            return TransformedEnv(
                CountingEnv(max_steps=max_steps), StepCounter(max_steps)
            )

        probe = env_fn()
        policy = RandomPolicy(probe.action_spec)
        probe.close(raise_if_closed=False)

        rb = ReplayBuffer(storage=LazyTensorStorage(200))
        ray_init_config = dict(DEFAULT_RAY_INIT_CONFIG)
        ray_init_config["runtime_env"] = {
            "working_dir": os.path.dirname(__file__),
            "env_vars": {"PYTHONPATH": os.path.dirname(__file__)},
        }
        remote_configs = {"num_cpus": 1, "num_gpus": 0.0}
        with pytest.raises(TypeError, match="RayReplayBuffer"):
            RayCollector(
                [env_fn, env_fn],
                policy,
                collector_class=Collector,
                replay_buffer=rb,
                frames_per_batch=max_steps * 4,
                total_frames=max_steps * 16,
                trajs_per_batch=num_trajs,
                ray_init_config=ray_init_config,
                remote_configs=remote_configs,
            )


class _GradientSyncTestModule(nn.Module):
    def __init__(self, rank: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.25, -0.5]) + 10 * rank)
        self.bias = nn.Parameter(torch.tensor(0.125 + 10 * rank))
        self.rank_only = nn.Parameter(torch.tensor(-0.75 + 10 * rank))
        self.never_used = nn.Parameter(torch.tensor(3.0 + 10 * rank))

    def forward(self, value, *, use_rank_only: bool):
        result = value @ self.weight + self.bias
        if use_rank_only:
            result = result + value[:, 0] * self.rank_only
        return result


def _gradient_sync_data(rank):
    if rank == 0:
        return (
            torch.tensor([[1.0, 2.0], [-1.0, 0.5]]),
            torch.tensor([0.4, -0.2]),
        )
    return (
        torch.tensor([[0.3, -0.7], [2.0, 1.0]]),
        torch.tensor([0.1, 1.5]),
    )


def _gloo_gradient_sync_worker(rank, world_size, init_method, output_dir):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        context = DataParallelContext.from_process_group(device="cpu", local_rank=rank)
        module = _GradientSyncTestModule(rank)
        context.broadcast_module(module)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
        value, target = _gradient_sync_data(rank)
        loss = (module(value, use_rank_only=rank == 0) - target).square().mean()
        loss.backward()
        if rank == 0:
            assert module.rank_only.grad is not None
        else:
            assert module.rank_only.grad is None
        assert module.never_used.grad is None
        context.sync_gradients(optimizer)
        assert module.rank_only.grad is not None
        assert module.never_used.grad is None
        optimizer.step()
        torch.save(module.state_dict(), os.path.join(output_dir, f"rank-{rank}.pt"))
        context.close()
        context.close()
        assert dist.is_initialized()
    finally:
        dist.destroy_process_group()


def _nccl_gradient_sync_worker(rank, world_size, init_method, output_dir):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        context = DataParallelContext.from_process_group(
            device=torch.device("cuda", rank), local_rank=rank
        )
        module = nn.Linear(2, 1).to(context.device)
        with torch.no_grad():
            module.weight.fill_(rank + 1.0)
            module.bias.fill_(rank + 2.0)
        context.broadcast_module(module)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        module(torch.full((2, 2), rank + 1.0, device=context.device)).sum().backward()
        context.sync_gradients(optimizer)
        optimizer.step()
        torch.save(
            {key: value.cpu() for key, value in module.state_dict().items()},
            os.path.join(output_dir, f"nccl-rank-{rank}.pt"),
        )
        context.close()
    finally:
        dist.destroy_process_group()


def _torchrun_context_worker(rank, world_size, init_method, output_dir):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    context = DataParallelContext.from_torchrun(
        backend="gloo", device="cpu", init_method=init_method
    )
    assert dist.is_initialized()
    assert context.rank == rank
    assert context.local_rank == rank
    assert context.world_size == world_size
    context.barrier()
    context.close()
    context.close()
    assert not dist.is_initialized()
    torch.save(True, os.path.join(output_dir, f"torchrun-rank-{rank}.pt"))


@pytest.mark.skipif(
    not _has_ray, reason="Ray not found. Ray may be badly configured or not installed."
)
class TestRayLearnerGroup:
    @pytest.fixture(autouse=True)
    def start_ray(self):
        import ray

        ray.shutdown()
        ray.init(
            num_cpus=4,
            include_dashboard=False,
            runtime_env={
                "working_dir": os.path.dirname(__file__),
                "env_vars": {"PYTHONPATH": os.path.dirname(__file__)},
            },
        )
        yield
        ray.shutdown()

    @staticmethod
    def _make_replay_buffer():
        replay_buffer = RayReplayBuffer(
            replay_buffer_cls=TensorDictReplayBuffer,
            storage=LazyTensorStorage(64),
            batch_size=8,
        )
        replay_buffer.extend(
            TensorDict({"x": torch.ones(64, 1), "y": torch.zeros(64, 1)}, [64])
        )
        return replay_buffer

    @staticmethod
    def _make_group(
        replay_buffer, factory=_make_ray_test_learner, *, num_gpus: float = 0
    ):
        return RayLearnerGroup(
            factory,
            replay_buffer.client(),
            world_size=2,
            global_batch_size=8,
            resources_per_rank={"num_cpus": 1, "num_gpus": num_gpus},
            setup_timeout=60,
            command_timeout=60,
            seed=0,
        )

    def test_gloo_step_matches_global_batch_and_restores(self):
        replay_buffer = self._make_replay_buffer()
        group = self._make_group(replay_buffer).start()
        try:
            metrics = group.step()
            assert group.last_round == 1
            assert group.model_version == 1
            assert metrics.device == torch.device("cpu")

            torch.manual_seed(0)
            reference = _RayLearnerLoss()
            optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
            batch = TensorDict({"x": torch.ones(8, 1), "y": torch.zeros(8, 1)}, [8])
            reference(batch)["loss"].backward()
            optimizer.step()
            expected = TensorDict.from_module(reference.linear)
            actual = group.get_weights(expected_version=group.model_version)
            for key in expected.keys(True, True):
                torch.testing.assert_close(actual.get(key), expected.get(key))

            state = group.state_dict()
            assert len(state["rng_by_rank"]) == 2
            group.shutdown()
            group.shutdown()

            restored = self._make_group(replay_buffer).start()
            restored.load_state_dict(state)
            restored.step()
            assert restored.model_version == 2
            restored.shutdown()
            assert replay_buffer.is_alive
            import ray

            assert ray.is_initialized()
        finally:
            group.shutdown()
            replay_buffer.close()

    def test_controller_checkpoint_restores_new_generation(self, tmp_path):
        replay_buffer = self._make_replay_buffer()
        group = self._make_group(replay_buffer).start()
        collector = _RayCheckpointCollector()
        checkpoint_root = tmp_path / "ray-checkpoints"
        try:
            group.step()
            trainer = Trainer(
                collector=collector,
                total_frames=64,
                frame_skip=1,
                optim_steps_per_batch=1,
                loss_module=None,
                optimizer=None,
                learner_group=group,
                replay_buffer=replay_buffer,
                progress_bar=False,
                save_trainer_file=checkpoint_root,
            )
            trainer.collected_frames = 8
            trainer._optim_count = 1
            trainer._learner_round = group.last_round
            trainer._published_model_version = group.model_version
            collector.state_value = 17
            trainer._save_distributed_checkpoint()
        finally:
            group.shutdown()
            replay_buffer.close()

        restored_replay = self._make_replay_buffer()
        restored_group = self._make_group(restored_replay)
        restored_collector = _RayCheckpointCollector()
        try:
            restored = Trainer(
                collector=restored_collector,
                total_frames=64,
                frame_skip=1,
                optim_steps_per_batch=1,
                loss_module=None,
                optimizer=None,
                learner_group=restored_group,
                replay_buffer=restored_replay,
                progress_bar=False,
            ).load_from_file(checkpoint_root)
            assert restored._learner_round == 1
            assert restored._optim_count == 1
            assert restored_group.model_version == 1
            assert restored_collector.state_value == 17
            assert len(restored_replay) == 64
            restored_group.step()
            assert restored_group.model_version == 2
        finally:
            restored_group.shutdown()
            restored_replay.close()

    def test_rank_failure_invalidates_the_group_only(self):
        import ray

        replay_buffer = self._make_replay_buffer()
        group = self._make_group(replay_buffer, _make_failing_ray_test_learner).start()
        try:
            with pytest.raises(RuntimeError, match="generation=1, round=1") as error:
                group.step()
            assert isinstance(error.value.__cause__, Exception)
            assert not group.is_alive
            assert replay_buffer.is_alive
            assert ray.is_initialized()
        finally:
            group.shutdown()
            replay_buffer.close()

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="two CUDA devices are required"
    )
    def test_nccl_smoke(self):
        replay_buffer = self._make_replay_buffer()
        group = self._make_group(replay_buffer, num_gpus=1).start()
        try:
            group.step()
            assert group.model_version == 1
        finally:
            group.shutdown()
            replay_buffer.close()


class TestDataParallelContext:
    @pytest.fixture(autouse=True)
    def reset_process_group(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        yield
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_from_rendezvous_single_process(self):
        context = DataParallelContext.from_rendezvous(
            rank=0,
            world_size=1,
            local_rank=0,
            device="cpu",
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
        )
        assert context.rank == 0
        assert context.device == torch.device("cpu")
        context.close()

    def test_from_torchrun_single_process(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "1")
        context = DataParallelContext.from_torchrun(device="cpu")
        assert context.rank == 0
        assert context.local_rank == 3
        assert context.world_size == 1
        assert context.device == torch.device("cpu")
        assert context.is_rank_zero
        context.barrier()
        context.close()
        context.close()
        assert context.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            context.barrier()

    def test_from_torchrun_requires_environment(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        with pytest.raises(RuntimeError, match="RANK is not set"):
            DataParallelContext.from_torchrun(device="cpu")

    def test_sparse_gradients_fail_explicitly(self):
        module = nn.Embedding(4, 2, sparse=True)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        module(torch.tensor([0, 1])).sum().backward()
        with pytest.raises(RuntimeError, match="sparse or non-strided"):
            DataParallelContext().sync_gradients(optimizer)

    def test_gloo_updates_match_global_batch_reference(self, tmp_path):
        init_path = tmp_path / "gloo-init"
        init_method = f"file://{init_path}"
        mp.spawn(
            _gloo_gradient_sync_worker,
            args=(2, init_method, str(tmp_path)),
            nprocs=2,
            join=True,
        )

        reference = _GradientSyncTestModule(rank=0)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        loss_sum = 0.0
        for rank in range(2):
            value, target = _gradient_sync_data(rank)
            loss_sum = (
                loss_sum
                + (reference(value, use_rank_only=rank == 0) - target).square().sum()
            )
        (loss_sum / 4).backward()
        optimizer.step()

        reference_state = reference.state_dict()
        rank_states = [torch.load(tmp_path / f"rank-{rank}.pt") for rank in range(2)]
        for rank_state in rank_states:
            for key, expected in reference_state.items():
                torch.testing.assert_close(rank_state[key], expected)
        for key in reference_state:
            torch.testing.assert_close(rank_states[0][key], rank_states[1][key])

    def test_from_torchrun_initializes_and_owns_group(self, tmp_path):
        init_path = tmp_path / "torchrun-init"
        mp.spawn(
            _torchrun_context_worker,
            args=(2, f"file://{init_path}", str(tmp_path)),
            nprocs=2,
            join=True,
        )
        for rank in range(2):
            assert torch.load(tmp_path / f"torchrun-rank-{rank}.pt")

    @pytest.mark.gpu
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="two CUDA devices are required"
    )
    def test_nccl_smoke(self, tmp_path):
        init_path = tmp_path / "nccl-init"
        init_method = f"file://{init_path}"
        mp.spawn(
            _nccl_gradient_sync_worker,
            args=(2, init_method, str(tmp_path)),
            nprocs=2,
            join=True,
        )
        rank_states = [
            torch.load(tmp_path / f"nccl-rank-{rank}.pt") for rank in range(2)
        ]
        for key in rank_states[0]:
            torch.testing.assert_close(rank_states[0][key], rank_states[1][key])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
