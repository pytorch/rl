# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn as nn
from mocking_classes import ContinuousActionVecMockEnv
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import multiprocessing as mp
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.weight_update.weight_sync_schemes import (
    _resolve_model,
    MPTransport,
    MultiProcessWeightSyncScheme,
    NoWeightSyncScheme,
    SharedMemTransport,
    SharedMemWeightSyncScheme,
    WeightStrategy,
)


def worker_update_policy(pipe, timeout=5.0):
    policy = nn.Linear(4, 2)
    with torch.no_grad():
        policy.weight.fill_(0.0)
        policy.bias.fill_(0.0)

    scheme = MultiProcessWeightSyncScheme(strategy="state_dict")
    scheme.init_on_worker(model_id="policy", pipe=pipe, model=policy)
    receiver = scheme.get_receiver()

    if receiver._transport.pipe.poll(timeout):
        data, msg = receiver._transport.pipe.recv()
        if msg == "update_weights":
            model_id, weights = data
            receiver.apply_weights(weights)

    return policy.weight.sum().item(), policy.bias.sum().item()


def worker_update_policy_tensordict(pipe, timeout=5.0):
    policy = nn.Linear(4, 2)
    with torch.no_grad():
        policy.weight.fill_(0.0)
        policy.bias.fill_(0.0)

    scheme = MultiProcessWeightSyncScheme(strategy="tensordict")
    scheme.init_on_worker(model_id="policy", pipe=pipe, model=policy)
    receiver = scheme.get_receiver()

    if receiver._transport.pipe.poll(timeout):
        data, msg = receiver._transport.pipe.recv()
        if msg == "update_weights":
            model_id, weights = data
            receiver.apply_weights(weights)

    return policy.weight.sum().item(), policy.bias.sum().item()


def worker_shared_mem(pipe, timeout=10.0):
    policy = nn.Linear(4, 2)

    if pipe.poll(timeout):
        data, msg = pipe.recv()
        if msg == "register_shared_weights":
            model_id, shared_weights = data
            shared_weights.to_module(policy)
            pipe.send((None, "registered"))

    import time

    time.sleep(0.5)

    return policy.weight.sum().item(), policy.bias.sum().item()


class TestTransportBackends:
    def test_mp_transport_basic(self):
        parent_pipe, child_pipe = mp.Pipe()
        transport = MPTransport(parent_pipe)

        assert transport.check_connection()

        proc = mp.Process(target=worker_update_policy, args=(child_pipe,))
        proc.start()

        test_weights = {"weight": torch.ones(2, 4), "bias": torch.ones(2)}
        transport.send_weights("policy", test_weights)

        proc.join(timeout=10.0)
        assert not proc.is_alive()

    def test_mp_transport_async(self):
        parent_pipe, child_pipe = mp.Pipe()
        transport = MPTransport(parent_pipe)

        proc = mp.Process(target=worker_update_policy, args=(child_pipe,))
        proc.start()

        test_weights = {"weight": torch.ones(2, 4), "bias": torch.ones(2)}
        transport.send_weights_async("policy", test_weights)
        transport.wait_ack()

        proc.join(timeout=10.0)
        assert not proc.is_alive()

    def test_shared_mem_transport(self):
        shared_buffer = TensorDict(
            {"weight": torch.zeros(2, 4), "bias": torch.zeros(2)}, batch_size=[]
        ).share_memory_()

        transport = SharedMemTransport({"policy": shared_buffer})

        new_weights = TensorDict(
            {"weight": torch.ones(2, 4), "bias": torch.ones(2)}, batch_size=[]
        )

        transport.send_weights("policy", new_weights)

        assert torch.allclose(shared_buffer["weight"], torch.ones(2, 4))
        assert torch.allclose(shared_buffer["bias"], torch.ones(2))


class TestWeightStrategies:
    def test_state_dict_strategy(self):
        strategy = WeightStrategy(extract_as="state_dict")

        policy = nn.Linear(3, 4)
        weights = strategy.extract_weights(policy)
        assert isinstance(weights, dict)
        assert "weight" in weights
        assert "bias" in weights

        target_policy = nn.Linear(3, 4)
        with torch.no_grad():
            target_policy.weight.fill_(0.0)
            target_policy.bias.fill_(0.0)

        strategy.apply_weights(target_policy, weights)

        assert torch.allclose(policy.weight, target_policy.weight)
        assert torch.allclose(policy.bias, target_policy.bias)

    def test_tensordict_strategy(self):
        strategy = WeightStrategy(extract_as="tensordict")

        policy = nn.Linear(3, 4)
        weights = strategy.extract_weights(policy)
        assert isinstance(weights, TensorDict)

        target_policy = nn.Linear(3, 4)
        with torch.no_grad():
            target_policy.weight.fill_(0.0)
            target_policy.bias.fill_(0.0)

        strategy.apply_weights(target_policy, weights)

        assert torch.allclose(policy.weight, target_policy.weight)
        assert torch.allclose(policy.bias, target_policy.bias)

    def test_cross_format_conversion(self):
        policy = nn.Linear(3, 4)

        state_dict_strategy = WeightStrategy(extract_as="state_dict")
        tensordict_strategy = WeightStrategy(extract_as="tensordict")

        state_dict_weights = state_dict_strategy.extract_weights(policy)
        tensordict_weights = tensordict_strategy.extract_weights(policy)

        target_policy_1 = nn.Linear(3, 4)
        target_policy_2 = nn.Linear(3, 4)

        with torch.no_grad():
            target_policy_1.weight.fill_(0.0)
            target_policy_1.bias.fill_(0.0)
            target_policy_2.weight.fill_(0.0)
            target_policy_2.bias.fill_(0.0)

        state_dict_strategy.apply_weights(target_policy_1, tensordict_weights)
        tensordict_strategy.apply_weights(target_policy_2, state_dict_weights)

        assert torch.allclose(policy.weight, target_policy_1.weight)
        assert torch.allclose(policy.weight, target_policy_2.weight)


class TestWeightSyncSchemes:
    """Tests for weight sync schemes using the new simplified API.

    Lower-level transport and legacy API tests are in TestTransportBackends.
    """

    def test_multiprocess_scheme_state_dict(self):
        parent_pipe, child_pipe = mp.Pipe()

        scheme = MultiProcessWeightSyncScheme(strategy="state_dict")
        scheme.init_on_sender(model_id="policy", pipes=[parent_pipe])
        sender = scheme.get_sender()

        proc = mp.Process(target=worker_update_policy, args=(child_pipe,))
        proc.start()

        weights = {"weight": torch.ones(2, 4), "bias": torch.ones(2)}
        sender.send(weights)

        proc.join(timeout=10.0)
        assert not proc.is_alive()

    def test_multiprocess_scheme_tensordict(self):
        parent_pipe, child_pipe = mp.Pipe()

        scheme = MultiProcessWeightSyncScheme(strategy="tensordict")
        scheme.init_on_sender(model_id="policy", pipes=[parent_pipe])
        sender = scheme.get_sender()

        proc = mp.Process(target=worker_update_policy_tensordict, args=(child_pipe,))
        proc.start()

        weights = TensorDict(
            {"weight": torch.ones(2, 4), "bias": torch.ones(2)}, batch_size=[]
        )
        sender.send(weights)

        proc.join(timeout=10.0)
        assert not proc.is_alive()

    def test_shared_mem_scheme(self):
        shared_buffer = TensorDict(
            {"weight": torch.zeros(2, 4), "bias": torch.zeros(2)}, batch_size=[]
        ).share_memory_()

        scheme = SharedMemWeightSyncScheme(
            policy_weights={"policy": shared_buffer},
            strategy="tensordict",
            auto_register=False,
        )

        transport = scheme.create_transport(None)

        new_weights = TensorDict(
            {"weight": torch.ones(2, 4), "bias": torch.ones(2)}, batch_size=[]
        )

        transport.send_weights("policy", new_weights)

        assert torch.allclose(shared_buffer["weight"], torch.ones(2, 4))
        assert torch.allclose(shared_buffer["bias"], torch.ones(2))

    def test_shared_mem_scheme_auto_register(self):
        scheme = SharedMemWeightSyncScheme(strategy="tensordict", auto_register=True)
        transport = scheme.create_transport(None)

        weights = TensorDict(
            {"weight": torch.ones(2, 4), "bias": torch.ones(2)}, batch_size=[]
        )

        transport.send_weights("policy", weights)

        assert "policy" in scheme.policy_weights
        assert torch.allclose(
            scheme.policy_weights["policy"]["weight"], torch.ones(2, 4)
        )

    def test_no_weight_sync_scheme(self):
        scheme = NoWeightSyncScheme()
        transport = scheme.create_transport(None)

        weights = {"weight": torch.ones(2, 4), "bias": torch.ones(2)}
        transport.send_weights("policy", weights)

    def test_receiver_receive_method(self):
        """Test the new non-blocking receive() method."""

        def worker_with_receive(pipe):
            policy = nn.Linear(4, 2)
            with torch.no_grad():
                policy.weight.fill_(0.0)
                policy.bias.fill_(0.0)

            scheme = MultiProcessWeightSyncScheme(strategy="state_dict")
            scheme.init_on_worker(model_id="policy", pipe=pipe, model=policy)
            receiver = scheme.get_receiver()

            # Non-blocking receive should return False when no data
            result = receiver.receive(timeout=0.001)
            assert result is False

            # Now actually receive the weights
            result = receiver.receive(timeout=5.0)
            assert result is True

            # Check weights were applied
            return policy.weight.sum().item(), policy.bias.sum().item()

        parent_pipe, child_pipe = mp.Pipe()

        scheme = MultiProcessWeightSyncScheme(strategy="state_dict")
        scheme.init_on_sender(model_id="policy", pipes=[parent_pipe])
        sender = scheme.get_sender()

        proc = mp.Process(target=worker_with_receive, args=(child_pipe,))
        proc.start()

        # Give worker time to call receive with no data
        import time

        time.sleep(0.1)

        weights = {"weight": torch.ones(2, 4), "bias": torch.ones(2)}
        sender.send(weights)

        proc.join(timeout=10.0)
        assert not proc.is_alive()


class TestCollectorIntegration:
    @pytest.fixture
    def simple_env(self):
        return ContinuousActionVecMockEnv()

    @pytest.fixture
    def simple_policy(self, simple_env):
        return TensorDictModule(
            nn.Linear(
                simple_env.observation_spec["observation"].shape[-1],
                simple_env.action_spec.shape[-1],
            ),
            in_keys=["observation"],
            out_keys=["action"],
        )

    def test_syncdatacollector_multiprocess_scheme(self, simple_policy):
        scheme = MultiProcessWeightSyncScheme(strategy="state_dict")

        collector = SyncDataCollector(
            create_env_fn=ContinuousActionVecMockEnv,
            policy=simple_policy,
            frames_per_batch=64,
            total_frames=128,
            weight_sync_schemes={"policy": scheme},
        )

        new_weights = simple_policy.state_dict()
        with torch.no_grad():
            for key in new_weights:
                new_weights[key].fill_(1.0)

        collector.update_policy_weights_(new_weights)

        for data in collector:
            assert data.numel() > 0
            break

        collector.shutdown()

    def test_multisyncdatacollector_multiprocess_scheme(self, simple_policy):
        scheme = MultiProcessWeightSyncScheme(strategy="state_dict")

        collector = MultiSyncDataCollector(
            create_env_fn=[
                ContinuousActionVecMockEnv,
                ContinuousActionVecMockEnv,
            ],
            policy=simple_policy,
            frames_per_batch=64,
            total_frames=128,
            weight_sync_schemes={"policy": scheme},
        )

        new_weights = simple_policy.state_dict()
        with torch.no_grad():
            for key in new_weights:
                new_weights[key].fill_(1.0)

        collector.update_policy_weights_(new_weights)

        for data in collector:
            assert data.numel() > 0
            break

        collector.shutdown()

    def test_multisyncdatacollector_shared_mem_scheme(self, simple_policy):
        scheme = SharedMemWeightSyncScheme(strategy="tensordict", auto_register=True)

        collector = MultiSyncDataCollector(
            create_env_fn=[
                ContinuousActionVecMockEnv,
                ContinuousActionVecMockEnv,
            ],
            policy=simple_policy,
            frames_per_batch=64,
            total_frames=128,
            weight_sync_schemes={"policy": scheme},
        )

        new_weights = TensorDict.from_module(simple_policy)
        with torch.no_grad():
            new_weights["module"]["weight"].fill_(1.0)
            new_weights["module"]["bias"].fill_(1.0)

        collector.update_policy_weights_(new_weights)

        for data in collector:
            assert data.numel() > 0
            break

        collector.shutdown()

    def test_collector_no_weight_sync(self, simple_policy):
        scheme = NoWeightSyncScheme()

        collector = SyncDataCollector(
            create_env_fn=ContinuousActionVecMockEnv,
            policy=simple_policy,
            frames_per_batch=64,
            total_frames=128,
            weight_sync_schemes={"policy": scheme},
        )

        for data in collector:
            assert data.numel() > 0
            break

        collector.shutdown()


class TestMultiModelUpdates:
    def test_multi_model_state_dict_updates(self):
        env = ContinuousActionVecMockEnv()

        policy = TensorDictModule(
            nn.Linear(
                env.observation_spec["observation"].shape[-1], env.action_spec.shape[-1]
            ),
            in_keys=["observation"],
            out_keys=["action"],
        )

        value = TensorDictModule(
            nn.Linear(env.observation_spec["observation"].shape[-1], 1),
            in_keys=["observation"],
            out_keys=["value"],
        )

        weight_sync_schemes = {
            "policy": MultiProcessWeightSyncScheme(strategy="state_dict"),
            "value": MultiProcessWeightSyncScheme(strategy="state_dict"),
        }

        collector = SyncDataCollector(
            create_env_fn=ContinuousActionVecMockEnv,
            policy=policy,
            frames_per_batch=64,
            total_frames=128,
            weight_sync_schemes=weight_sync_schemes,
        )

        policy_weights = policy.state_dict()
        value_weights = value.state_dict()

        with torch.no_grad():
            for key in policy_weights:
                policy_weights[key].fill_(1.0)
            for key in value_weights:
                value_weights[key].fill_(2.0)

        collector.update_policy_weights_(
            weights_dict={
                "policy": policy_weights,
                "value": value_weights,
            }
        )

        for data in collector:
            assert data.numel() > 0
            break

        collector.shutdown()
        env.close()

    def test_multi_model_tensordict_updates(self):
        env = ContinuousActionVecMockEnv()

        policy = TensorDictModule(
            nn.Linear(
                env.observation_spec["observation"].shape[-1], env.action_spec.shape[-1]
            ),
            in_keys=["observation"],
            out_keys=["action"],
        )

        value = TensorDictModule(
            nn.Linear(env.observation_spec["observation"].shape[-1], 1),
            in_keys=["observation"],
            out_keys=["value"],
        )

        weight_sync_schemes = {
            "policy": MultiProcessWeightSyncScheme(strategy="tensordict"),
            "value": MultiProcessWeightSyncScheme(strategy="tensordict"),
        }

        collector = SyncDataCollector(
            create_env_fn=ContinuousActionVecMockEnv,
            policy=policy,
            frames_per_batch=64,
            total_frames=128,
            weight_sync_schemes=weight_sync_schemes,
        )

        policy_weights = TensorDict.from_module(policy)
        value_weights = TensorDict.from_module(value)

        with torch.no_grad():
            policy_weights["module"]["weight"].fill_(1.0)
            policy_weights["module"]["bias"].fill_(1.0)
            value_weights["module"]["weight"].fill_(2.0)
            value_weights["module"]["bias"].fill_(2.0)

        collector.update_policy_weights_(
            weights_dict={
                "policy": policy_weights,
                "value": value_weights,
            }
        )

        for data in collector:
            assert data.numel() > 0
            break

        collector.shutdown()
        env.close()


class TestHelpers:
    def test_resolve_model_simple(self):
        class Context:
            def __init__(self):
                self.policy = nn.Linear(2, 3)

        context = Context()
        resolved = _resolve_model(context, "policy")
        assert resolved is context.policy

    def test_resolve_model_nested(self):
        class Inner:
            def __init__(self):
                self.value_net = nn.Linear(2, 3)

        class Context:
            def __init__(self):
                self.env = Inner()

        context = Context()
        resolved = _resolve_model(context, "env.value_net")
        assert resolved is context.env.value_net

    def test_resolve_model_with_index(self):
        class Context:
            def __init__(self):
                self.transform = [nn.Linear(2, 3), nn.Linear(3, 4)]

        context = Context()
        resolved = _resolve_model(context, "transform[0]")
        assert resolved is context.transform[0]

        resolved = _resolve_model(context, "transform[1]")
        assert resolved is context.transform[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDeviceHandling:
    def test_weight_update_cpu_to_cpu(self):
        policy = nn.Linear(3, 4)
        strategy = WeightStrategy(extract_as="state_dict")

        weights = strategy.extract_weights(policy)
        target = nn.Linear(3, 4)
        strategy.apply_weights(target, weights)

        assert torch.allclose(policy.weight, target.weight)

    def test_weight_update_cuda_to_cuda(self):
        policy = nn.Linear(3, 4).cuda()
        strategy = WeightStrategy(extract_as="tensordict")

        weights = strategy.extract_weights(policy)
        target = nn.Linear(3, 4).cuda()
        strategy.apply_weights(target, weights)

        assert torch.allclose(policy.weight, target.weight)


@pytest.mark.parametrize("strategy", ["state_dict", "tensordict"])
def test_weight_strategy_parametrized(strategy):
    weight_strategy = WeightStrategy(extract_as=strategy)

    policy = nn.Linear(3, 4)
    weights = weight_strategy.extract_weights(policy)

    target = nn.Linear(3, 4)
    with torch.no_grad():
        target.weight.fill_(0.0)
        target.bias.fill_(0.0)

    weight_strategy.apply_weights(target, weights)

    assert torch.allclose(policy.weight, target.weight)
    assert torch.allclose(policy.bias, target.bias)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst", "-v"] + unknown)
