# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import tempfile
from argparse import Namespace
from collections import OrderedDict
from os import walk, path

import pytest
import torch

try:
    from tensorboard.backend.event_processing import event_accumulator
    from torchrl.trainers.loggers import TensorboardLogger

    _has_tb = True
except ImportError:
    _has_tb = False

from torchrl.data import (
    TensorDict,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs.libs.gym import _has_gym
from torchrl.trainers import Recorder
from torchrl.trainers import Trainer
from torchrl.trainers.helpers import transformed_env_constructor
from torchrl.trainers.trainers import (
    SelectKeys,
    ReplayBufferTrainer,
    LogReward,
    RewardNormalizer,
    mask_batch,
    BatchSubSampler,
    UpdateWeights,
    CountFramesLog,
)
from torchrl.trainers.trainers import _has_tqdm


class MockingOptim:
    param_groups = [{"params": []}]


class MockingCollector:
    called_update_policy_weights_ = False

    def set_seed(self, seed, **kwargs):
        return seed

    def update_policy_weights_(self):
        self.called_update_policy_weights_ = True

    def shutdown(self):
        pass


def mocking_trainer() -> Trainer:
    trainer = Trainer(
        MockingCollector(),
        *[
            None,
        ]
        * 3,
        MockingOptim()
    )
    trainer.collected_frames = 0
    trainer._pbar_str = OrderedDict()
    return trainer


def test_selectkeys():
    trainer = mocking_trainer()
    key1 = "first key"
    key2 = "second key"
    td = TensorDict(
        {
            key1: torch.randn(3),
            key2: torch.randn(3),
        },
        [],
    )
    trainer.register_op("batch_process", SelectKeys([key1]))
    td_out = trainer._process_batch_hook(td)
    assert key1 in td_out.keys()
    assert key2 not in td_out.keys()


@pytest.mark.parametrize("prioritized", [True, False])
def test_rb_trainer(prioritized):
    trainer = mocking_trainer()
    S = 100
    if prioritized:
        replay_buffer = TensorDictPrioritizedReplayBuffer(S, 1.1, 0.9)
    else:
        replay_buffer = TensorDictReplayBuffer(S)

    N = 9
    rb_trainer = ReplayBufferTrainer(replay_buffer=replay_buffer, batch_size=N)

    trainer.register_op("batch_process", rb_trainer.extend)
    trainer.register_op("process_optim_batch", rb_trainer.sample)
    trainer.register_op("post_loss", rb_trainer.update_priority)

    key1 = "first key"
    key2 = "second key"
    batch = 101
    td = TensorDict(
        {
            key1: torch.randn(batch, 3),
            key2: torch.randn(batch, 3),
        },
        [batch],
    )
    td_out = trainer._process_batch_hook(td)
    assert td_out is td

    td_out = trainer._process_optim_batch_hook(td)
    assert td_out is not td
    assert td_out.shape[0] == N

    if prioritized:
        td_out.set(replay_buffer.priority_key, torch.rand(N))

    td_out = trainer._post_loss_hook(td_out)
    if prioritized:
        for idx in range(min(S, batch)):
            if idx in td_out.get("index"):
                assert replay_buffer._sum_tree[idx] != 1.0
            else:
                assert replay_buffer._sum_tree[idx] == 1.0
    else:
        assert "index" not in td_out.keys()


@pytest.mark.parametrize("logname", ["a", "b"])
@pytest.mark.parametrize("pbar", [True, False])
def test_log_reward(logname, pbar):
    trainer = mocking_trainer()
    trainer.collected_frames = 0

    log_reward = LogReward(logname, log_pbar=pbar)
    trainer.register_op("pre_steps_log", log_reward)
    td = TensorDict({"reward": torch.ones(3)}, [3])
    trainer._pre_steps_log_hook(td)
    if _has_tqdm and pbar:
        assert trainer._pbar_str[logname] == 1
    else:
        assert logname not in trainer._pbar_str
    assert trainer._log_dict[logname][-1] == 1


def test_reward_norm():
    torch.manual_seed(0)
    trainer = mocking_trainer()

    reward_normalizer = RewardNormalizer()
    trainer.register_op("batch_process", reward_normalizer.update_reward_stats)
    trainer.register_op("process_optim_batch", reward_normalizer.normalize_reward)

    batch = 10
    reward = torch.randn(batch, 1)
    td = TensorDict({"reward": reward.clone()}, [batch])
    td_out = trainer._process_batch_hook(td)
    assert (td_out.get("reward") == reward).all()
    assert not reward_normalizer._normalize_has_been_called

    td_norm = trainer._process_optim_batch_hook(td)
    assert reward_normalizer._normalize_has_been_called
    torch.testing.assert_close(td_norm.get("reward").mean(), torch.zeros([]))
    torch.testing.assert_close(td_norm.get("reward").std(), torch.ones([]))


def test_masking():
    torch.manual_seed(0)
    trainer = mocking_trainer()

    trainer.register_op("batch_process", mask_batch)
    batch = 10
    td = TensorDict(
        {
            "mask": torch.zeros(batch, dtype=torch.bool).bernoulli_(),
            "tensor": torch.randn(batch, 51),
        },
        [batch],
    )
    td_out = trainer._process_batch_hook(td)
    assert td_out.shape[0] == td.get("mask").sum()
    assert (td["tensor"][td["mask"].squeeze(-1)] == td_out["tensor"]).all()


def test_subsampler():
    torch.manual_seed(0)
    trainer = mocking_trainer()

    batch_size = 10
    sub_traj_len = 5

    key1 = "key1"
    key2 = "key2"

    trainer.register_op(
        "process_optim_batch",
        BatchSubSampler(batch_size=batch_size, sub_traj_len=sub_traj_len),
    )

    td = TensorDict(
        {
            key1: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
            key2: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
        },
        [2, 10],
    )

    td_out = trainer._process_optim_batch_hook(td)
    assert td_out.shape == torch.Size([batch_size // sub_traj_len, sub_traj_len])
    assert (td_out.get(key1) == td_out.get(key2)).all()


@pytest.mark.skipif(not _has_gym, reason="No gym library")
@pytest.mark.skipif(not _has_tb, reason="No tensorboard library")
def test_recorder():
    with tempfile.TemporaryDirectory() as folder:
        print(folder)
        logger = TensorboardLogger(exp_name=folder)
        args = Namespace()
        args.env_name = "ALE/Pong-v5"
        args.env_task = ""
        args.grayscale = True
        args.env_library = "gym"
        args.frame_skip = 1
        args.center_crop = []
        args.from_pixels = True
        args.vecnorm = False
        args.norm_rewards = False
        args.reward_scaling = 1.0
        args.reward_loc = 0.0
        args.noops = 0
        args.record_frames = 24 // args.frame_skip
        args.record_interval = 2
        args.catframes = 4

        N = 8

        recorder = transformed_env_constructor(
            args,
            video_tag="tmp",
            norm_obs_only=True,
            stats={"loc": 0, "scale": 1},
            logger=logger,
        )()

        recorder = Recorder(
            record_frames=args.record_frames,
            frame_skip=args.frame_skip,
            policy_exploration=None,
            recorder=recorder,
            record_interval=args.record_interval,
        )

        for _ in range(N):
            recorder(None)

        for (_, _, filenames) in walk(folder):
            filename = filenames[0]
            break

        ea = event_accumulator.EventAccumulator(
            path.join(folder, filename),
            size_guidance={
                event_accumulator.IMAGES: 0,
            },
        )
        ea.Reload()
        print(ea.Tags())
        img = ea.Images("tmp_ALE/Pong-v5_video")
        assert len(img) == N // args.record_interval


def test_updateweights():
    torch.manual_seed(0)
    trainer = mocking_trainer()

    T = 5
    update_weights = UpdateWeights(trainer.collector, T)
    trainer.register_op("post_steps", update_weights)
    for t in range(T):
        trainer._post_steps_hook()
        assert trainer.collector.called_update_policy_weights_ is (t == T - 1)
    assert trainer.collector.called_update_policy_weights_


def test_countframes():
    torch.manual_seed(0)
    trainer = mocking_trainer()

    frame_skip = 3
    batch = 10
    count_frames = CountFramesLog(frame_skip=frame_skip)
    trainer.register_op("pre_steps_log", count_frames)
    td = TensorDict(
        {"mask": torch.zeros(batch, dtype=torch.bool).bernoulli_()}, [batch]
    )
    trainer._pre_steps_log_hook(td)
    assert count_frames.frame_count == td.get("mask").sum() * frame_skip


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
