# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict

import pytest
import torch
from torch.optim import Adam

from torchrl.agents import Agent
from torchrl.agents.agents import SelectKeys, ReplayBufferAgent, LogReward, \
    RewardNormalizer, mask_batch, BatchSubSampler, UpdateWeights, \
    CountFramesLog
from torchrl.data import TensorDict, TensorDictPrioritizedReplayBuffer, \
    TensorDictReplayBuffer


class MockingOptim:
    param_groups = [{'params': []}]


class MockingCollector:
    called_update_policy_weights_ = False

    def set_seed(self, seed):
        return seed

    def update_policy_weights_(self):
        self.called_update_policy_weights_ = True


def mocking_agent() -> Agent:
    agent = Agent(MockingCollector(), *[None, ] * 3, MockingOptim())
    agent.collected_frames = 0
    agent._pbar_str = OrderedDict()
    return agent


def test_selectkeys():
    agent = mocking_agent()
    key1 = "first key"
    key2 = "second key"
    td = TensorDict({key1: torch.randn(3), key2: torch.randn(3), }, [])
    agent.register_op("batch_process", SelectKeys([key1]))
    td_out = agent._process_batch_hook(td)
    assert key1 in td_out.keys()
    assert key2 not in td_out.keys()


@pytest.mark.parametrize("prioritized", [True, False])
def test_rb_agent(prioritized):
    agent = mocking_agent()
    S = 100
    if prioritized:
        replay_buffer = TensorDictPrioritizedReplayBuffer(S, 1.1, 0.9)
    else:
        replay_buffer = TensorDictReplayBuffer(S)

    N = 9
    rb_agent = ReplayBufferAgent(replay_buffer=replay_buffer, batch_size=N)

    agent.register_op("batch_process", rb_agent.extend)
    agent.register_op("process_optim_batch", rb_agent.sample)
    agent.register_op("post_loss", rb_agent.update_priority)

    key1 = "first key"
    key2 = "second key"
    batch = 101
    td = TensorDict(
        {key1: torch.randn(batch, 3), key2: torch.randn(batch, 3), }, [batch])
    td_out = agent._process_batch_hook(td)
    assert td_out is td

    td_out = agent._process_optim_batch_hook(td)
    assert td_out is not td
    assert td_out.shape[0] == N

    if prioritized:
        td_out.set(replay_buffer.priority_key, torch.rand(N))

    td_out = agent._post_loss_hook(td_out)
    if prioritized:
        for idx in range(min(S, batch)):
            if idx in td_out.get("index"):
                assert replay_buffer._sum_tree[idx] != 1.0
            else:
                assert replay_buffer._sum_tree[idx] == 1.0
    else:
        assert "index" not in td_out.keys()


@pytest.mark.parametrize("logname", ["a", "b"])
def test_log_reward(logname):
    agent = mocking_agent()
    agent.collected_frames = 0
    agent._pbar_str = OrderedDict()

    log_reward = LogReward(logname)
    agent.register_op("pre_steps_log", log_reward)
    td = TensorDict({"reward": torch.ones(3)}, [3])
    agent._pre_steps_log_hook(td)
    assert agent._pbar_str[logname] == 1


@pytest.mark.parametrize("logname", ["a", "b"])
def test_log_reward(logname):
    agent = mocking_agent()
    agent.collected_frames = 0
    agent._pbar_str = OrderedDict()

    log_reward = LogReward(logname)
    agent.register_op("pre_steps_log", log_reward)
    td = TensorDict({"reward": torch.ones(3)}, [3])
    agent._pre_steps_log_hook(td)
    assert agent._pbar_str[logname] == 1


def test_reward_norm():
    torch.manual_seed(0)
    agent = mocking_agent()

    reward_normalizer = RewardNormalizer()
    agent.register_op("batch_process", reward_normalizer.update_reward_stats)
    agent.register_op("process_optim_batch",
                      reward_normalizer.normalize_reward)

    batch = 10
    reward = torch.randn(batch, 1)
    td = TensorDict({"reward": reward.clone()}, [batch])
    td_out = agent._process_batch_hook(td)
    assert (td_out.get("reward") == reward).all()
    assert not reward_normalizer._normalize_has_been_called

    td_norm = agent._process_optim_batch_hook(td)
    assert reward_normalizer._normalize_has_been_called
    torch.testing.assert_close(td_norm.get("reward").mean(), torch.zeros([]))
    torch.testing.assert_close(td_norm.get("reward").std(), torch.ones([]))


def test_masking():
    torch.manual_seed(0)
    agent = mocking_agent()

    agent.register_op("batch_process", mask_batch)
    batch = 10
    td = TensorDict({'mask': torch.zeros(batch, dtype=torch.bool).bernoulli_(),
                     'tensor': torch.randn(batch, 51)}, [batch])
    td_out = agent._process_batch_hook(td)
    assert td_out.shape[0] == td.get("mask").sum()
    assert (td["tensor"][td["mask"].squeeze(-1)] == td_out["tensor"]).all()


def test_subsampler():
    torch.manual_seed(0)
    agent = mocking_agent()

    batch_size = 10
    sub_traj_len = 5

    key1 = "key1"
    key2 = "key2"

    agent.register_op(
        "process_optim_batch",
        BatchSubSampler(batch_size=batch_size,
                        sub_traj_len=sub_traj_len),
    )

    td = TensorDict(
        {
            key1: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
            key2: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
        },
        [2, 10]
    )

    td_out = agent._process_optim_batch_hook(td)
    assert td_out.shape == torch.Size(
        [batch_size // sub_traj_len, sub_traj_len])
    assert (td_out.get(key1) == td_out.get(key2)).all()


def test_recorder():
    pass


def test_updateweights():
    torch.manual_seed(0)
    agent = mocking_agent()

    T = 5
    update_weights = UpdateWeights(agent.collector, T)
    agent.register_op("post_steps", update_weights)
    for t in range(T):
        agent._post_steps_hook()
        assert agent.collector.called_update_policy_weights_ is (t == T - 1)
    assert agent.collector.called_update_policy_weights_

def test_countframes():
    torch.manual_seed(0)
    agent = mocking_agent()

    frame_skip = 3
    batch = 10
    count_frames = CountFramesLog(frame_skip=frame_skip)
    agent.register_op("pre_steps_log", count_frames)
    td = TensorDict({'mask': torch.zeros(batch, dtype=torch.bool).bernoulli_()}, [batch])
    agent._pre_steps_log_hook(td)
    assert count_frames.frame_count == td.get("mask").sum() * frame_skip

if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
