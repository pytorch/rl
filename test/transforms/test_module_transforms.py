# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from functools import partial

import pytest

import torch
import torch.distributed as dist

from _transforms_common import _has_ray, TransformBase
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl import logger as torchrl_logger, merge_ray_runtime_env

from torchrl.collectors import Collector
from torchrl.collectors.distributed.ray import DEFAULT_RAY_INIT_CONFIG
from torchrl.data import ReplayBuffer
from torchrl.envs import Compose, SerialEnv, TransformedEnv
from torchrl.envs.transforms import ModuleTransform
from torchrl.envs.transforms.module import RayModuleTransform
from torchrl.modules import RandomPolicy

from torchrl.testing import (  # noqa
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
from torchrl.testing.modules import BiasModule
from torchrl.weight_update import RayModuleTransformScheme


class TestModuleTransform(TransformBase):
    @property
    def _module_factory_samespec(self):
        return partial(
            TensorDictModule,
            nn.LazyLinear(7),
            in_keys=["observation"],
            out_keys=["observation"],
        )

    @property
    def _module_factory_samespec_inverse(self):
        return partial(
            TensorDictModule, nn.LazyLinear(7), in_keys=["action"], out_keys=["action"]
        )

    def _single_env_maker(self):
        base_env = ContinuousActionVecMockEnv()
        t = ModuleTransform(module_factory=self._module_factory_samespec)
        return base_env.append_transform(t)

    def test_single_trans_env_check(self):
        env = self._single_env_maker()
        env.check_env_specs()

    def test_serial_trans_env_check(self):
        env = SerialEnv(2, self._single_env_maker)
        try:
            env.check_env_specs()
        finally:
            env.close(raise_if_closed=False)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(2, self._single_env_maker)
        try:
            env.check_env_specs()
        finally:
            env.close(raise_if_closed=False)

    def test_trans_serial_env_check(self):
        env = SerialEnv(2, ContinuousActionVecMockEnv)
        try:
            env = env.append_transform(
                ModuleTransform(module_factory=self._module_factory_samespec)
            )
            env.check_env_specs()
        finally:
            env.close(raise_if_closed=False)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv)
        try:
            env = env.append_transform(
                ModuleTransform(module_factory=self._module_factory_samespec)
            )
            env.check_env_specs()
        finally:
            env.close(raise_if_closed=False)

    def test_transform_no_env(self):
        t = ModuleTransform(module_factory=self._module_factory_samespec)
        td = t(TensorDict(observation=torch.randn(2, 3), batch_size=[2]))
        assert td["observation"].shape == (2, 7)

    def test_transform_compose(self):
        t = Compose(ModuleTransform(module_factory=self._module_factory_samespec))
        td = t(TensorDict(observation=torch.randn(2, 3), batch_size=[2]))
        assert td["observation"].shape == (2, 7)

    def test_transform_env(self):
        # TODO: We should give users the opportunity to modify the specs
        env = self._single_env_maker()
        env.check_env_specs()

    def test_transform_model(self):
        t = nn.Sequential(
            Compose(ModuleTransform(module_factory=self._module_factory_samespec))
        )
        td = t(TensorDict(observation=torch.randn(2, 3), batch_size=[2]))
        assert td["observation"].shape == (2, 7)

    def test_transform_rb(self):
        t = ModuleTransform(module_factory=self._module_factory_samespec)
        rb = ReplayBuffer(transform=t)
        rb.extend(TensorDict(observation=torch.randn(2, 3), batch_size=[2]))
        assert rb._storage._storage[0]["observation"].shape == (3,)
        s = rb.sample(2)
        assert s["observation"].shape == (2, 7)

        rb = ReplayBuffer()
        rb.append_transform(t, invert=True)
        rb.extend(TensorDict(observation=torch.randn(2, 3), batch_size=[2]))
        assert rb._storage._storage[0]["observation"].shape == (7,)
        s = rb.sample(2)
        assert s["observation"].shape == (2, 7)

    def test_transform_inverse(self):
        t = ModuleTransform(
            module_factory=self._module_factory_samespec_inverse, inverse=True
        )
        env = ContinuousActionVecMockEnv().append_transform(t)
        env.check_env_specs()

    @pytest.mark.skipif(not _has_ray, reason="ray required")
    def test_ray_extension(self):
        import ray

        # Check if ray is initialized
        ray_init = ray.is_initialized
        try:
            t = ModuleTransform(
                module_factory=self._module_factory_samespec,
                use_ray_service=True,
                actor_name="my_transform",
            )
            env = ContinuousActionVecMockEnv().append_transform(t)
            assert isinstance(t, RayModuleTransform)
            env.check_env_specs()
            assert ray.get_actor("my_transform") is not None
        finally:
            if not ray_init:
                ray.stop()


@pytest.mark.skipif(not _has_ray, reason="ray required")
class TestRayModuleTransform:
    @pytest.fixture(autouse=True, scope="function")
    def start_ray(self):
        import ray

        if ray.is_initialized():
            ray.shutdown()

        # Use merge_ray_runtime_env to exclude large directories from the runtime environment
        # This prevents issues with Ray's working_dir size limits and GCS package expiration
        ray_init_config = merge_ray_runtime_env(dict(DEFAULT_RAY_INIT_CONFIG))
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

    def test_ray_module_transform_scheme_flow(self):
        bias_module = BiasModule(2.0)
        module_fact = lambda: TensorDictModule(
            bias_module,
            in_keys=["observation"],
            out_keys=["action"],
        )

        # Create scheme and transform
        scheme = RayModuleTransformScheme()
        transform = ModuleTransform(
            module_factory=module_fact,
            weight_sync_scheme=scheme,
            use_ray_service=True,
            actor_name="my_transform",
        )
        assert transform.in_keys == ["observation"]
        assert transform.out_keys == ["action"]
        dummy_data = TensorDict(observation=torch.zeros(2, 3), batch_size=[2])

        module = module_fact()
        assert (module(dummy_data)["action"] == 2).all()

        # test sending weights
        weights = TensorDict.from_module(module)
        d = weights.data
        d *= 0
        d += 1
        scheme.send(weights)
        assert (module(dummy_data)["action"] == 1).all()

    def test_ray_module_transform_scheme_collector(self):
        # Create a simple module that adds a learnable bias to observations
        # We use addition instead of scaling to avoid issues with observation values

        bias_module = BiasModule()
        module = TensorDictModule(
            bias_module,
            in_keys=["observation"],
            out_keys=["observation"],  # Transform in-place
        )

        # Create scheme and transform
        scheme = RayModuleTransformScheme()
        transform = RayModuleTransform(
            module=module,
            weight_sync_scheme=scheme,
        )

        # Create transformed env
        base_env = ContinuousActionVecMockEnv

        def make_env():
            return TransformedEnv(base_env(), transform)

        # Create collector with scheme registered
        torchrl_logger.debug("Creating collector")
        policy = RandomPolicy(base_env().action_spec)
        collector = Collector(
            make_env,
            policy,
            frames_per_batch=50,
            total_frames=200,
            weight_sync_schemes={"transform_module": scheme},
        )

        torchrl_logger.debug("Starting collector")
        first_batch_mean = None
        second_batch_mean = None
        try:
            for i, data in enumerate(collector):
                obs_mean = data["observation"].mean().item()

                if i == 0:
                    first_batch_mean = obs_mean

                    # Update weights: set bias to 100.0 (large value to be clearly visible)
                    torchrl_logger.debug("Updating weights")
                    new_weights = TensorDict.from_module(module)
                    new_weights["module", "bias"].data.fill_(100.0)
                    collector.update_policy_weights_(
                        new_weights, model_id="transform_module"
                    )
                elif i == 1:
                    second_batch_mean = obs_mean
                    break
        finally:
            collector.shutdown()

        # Verify that weights were updated
        # With bias=0.0, first batch should have observations around 0 (env default)
        # With bias=100.0, second batch should have observations shifted by 100
        assert first_batch_mean is not None, "First batch not collected"
        assert second_batch_mean is not None, "Second batch not collected"

        # The second batch should have significantly higher mean due to bias=100
        assert second_batch_mean > first_batch_mean + 50, (
            f"Weight update did not take effect: first_mean={first_batch_mean:.2f}, "
            f"second_mean={second_batch_mean:.2f}. Expected second to be at least 50 higher."
        )
