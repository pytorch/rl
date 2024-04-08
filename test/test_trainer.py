# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import tempfile
from argparse import Namespace
from collections import OrderedDict
from os import path, walk
from time import sleep

import pytest
import torch
from torch import nn

try:
    from tensorboard.backend.event_processing import event_accumulator
    from torchrl.record.loggers.tensorboard import TensorboardLogger

    _has_tb = True
except ImportError:
    _has_tb = False

from _utils_internal import PONG_VERSIONED
from tensordict import TensorDict
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs.libs.gym import _has_gym
from torchrl.trainers import Recorder, Trainer
from torchrl.trainers.helpers import transformed_env_constructor
from torchrl.trainers.trainers import (
    _has_tqdm,
    _has_ts,
    BatchSubSampler,
    CountFramesLog,
    LogReward,
    mask_batch,
    OptimizerHook,
    ReplayBufferTrainer,
    REWARD_KEY,
    RewardNormalizer,
    SelectKeys,
    UpdateWeights,
)


def _fun_checker(fun, checker):
    def new_fun(*args, **kwargs):
        checker[0] = True
        return fun(*args, **kwargs)

    return new_fun, fun


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

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class MockingLossModule(nn.Module):
    pass


_mocking_optim = MockingOptim()


def mocking_trainer(file=None, optimizer=_mocking_optim) -> Trainer:
    trainer = Trainer(
        collector=MockingCollector(),
        total_frames=None,
        frame_skip=None,
        optim_steps_per_batch=None,
        loss_module=MockingLossModule(),
        optimizer=optimizer,
        save_trainer_file=file,
    )
    trainer._pbar_str = OrderedDict()
    return trainer


class TestSelectKeys:
    def test_selectkeys(self):
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

    def test_selectkeys_statedict(self):
        if not _has_ts:
            os.environ["CKPT_BACKEND"] = "torch"
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
        hook = SelectKeys([key1])
        hook.register(trainer)
        trainer._process_batch_hook(td)

        trainer2 = mocking_trainer()
        hook2 = SelectKeys([key1])
        hook2.register(trainer2)
        sd = trainer.state_dict()
        assert not len(sd["select_keys"])
        trainer2.load_state_dict(sd)

    @pytest.mark.parametrize("backend", ["torchsnapshot", "torch"])
    def test_selectkeys_save(self, backend):
        if not _has_ts and backend == "torchsnapshot":
            pytest.skip("torchsnapshot not found")
        # we overwrite the method to make sure that load_state_dict and state_dict are being called
        state_dict_has_been_called = [False]
        load_state_dict_has_been_called = [False]
        SelectKeys.state_dict, SelectKeys_state_dict = _fun_checker(
            SelectKeys.state_dict, state_dict_has_been_called
        )
        SelectKeys.load_state_dict, SelectKeys_load_state_dict = _fun_checker(
            SelectKeys.load_state_dict, load_state_dict_has_been_called
        )

        os.environ["CKPT_BACKEND"] = backend

        with tempfile.TemporaryDirectory() as tmpdirname:
            if backend == "torch":
                file = path.join(tmpdirname, "file.pt")
            elif backend == "torchsnapshot":
                file = tmpdirname
            else:
                raise NotImplementedError
            trainer = mocking_trainer(file=file)
            key1 = "first key"
            key2 = "second key"
            td = TensorDict(
                {
                    key1: torch.randn(3),
                    key2: torch.randn(3),
                },
                [],
            )
            select_keys = SelectKeys([key1])
            select_keys.register(trainer)
            trainer._process_batch_hook(td)
            trainer.save_trainer(force_save=True)
            assert state_dict_has_been_called[0]

            trainer2 = mocking_trainer()
            select_keys2 = SelectKeys([key1])
            select_keys2.register(trainer2)

            trainer2.load_from_file(file)
            assert state_dict_has_been_called[0]
            if backend == "torch":
                assert load_state_dict_has_been_called[0]

        SelectKeys.state_dict = SelectKeys_state_dict
        SelectKeys.load_state_dict = SelectKeys_load_state_dict


@pytest.mark.parametrize("prioritized", [False, True])
class TestRB:
    def test_rb_trainer(self, prioritized):
        torch.manual_seed(0)
        trainer = mocking_trainer()
        S = 100
        storage = ListStorage(S)
        if prioritized:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=1.1, beta=0.9, storage=storage
            )
        else:
            replay_buffer = TensorDictReplayBuffer(storage=storage)

        N = 9
        rb_trainer = ReplayBufferTrainer(replay_buffer=replay_buffer, batch_size=N)

        rb_trainer.register(trainer)

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
                    assert replay_buffer._sampler._sum_tree[idx] != 1.0
                else:
                    assert replay_buffer._sampler._sum_tree[idx] == 1.0

    @pytest.mark.parametrize(
        "storage_type",
        [
            "memmap",
            "list",
        ],
    )
    def test_rb_trainer_state_dict(self, prioritized, storage_type):
        torch.manual_seed(0)
        trainer = mocking_trainer()
        S = 100
        if storage_type == "list":
            storage = ListStorage(S)
        elif storage_type == "memmap":
            storage = LazyMemmapStorage(S)
        else:
            raise NotImplementedError

        if prioritized:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=1.1,
                beta=0.9,
                storage=storage,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                storage=storage,
            )

        N = 9
        rb_trainer = ReplayBufferTrainer(replay_buffer=replay_buffer, batch_size=N)

        rb_trainer.register(trainer)

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
        trainer._process_batch_hook(td)
        td_out = trainer._process_optim_batch_hook(td)
        if prioritized:
            td_out.unlock_().set(replay_buffer.priority_key, torch.rand(N))
        trainer._post_loss_hook(td_out)

        trainer2 = mocking_trainer()
        if prioritized:
            replay_buffer2 = TensorDictPrioritizedReplayBuffer(
                alpha=1.1, beta=0.9, storage=storage
            )
        else:
            replay_buffer2 = TensorDictReplayBuffer(storage=storage)
        N = 9
        rb_trainer2 = ReplayBufferTrainer(replay_buffer=replay_buffer2, batch_size=N)
        rb_trainer2.register(trainer2)
        sd = trainer.state_dict()
        trainer2.load_state_dict(sd)

        assert rb_trainer2.replay_buffer._writer._cursor > 0
        assert (
            rb_trainer2.replay_buffer._writer._cursor
            == rb_trainer.replay_buffer._writer._cursor
        )

        if storage_type == "list":
            assert len(rb_trainer2.replay_buffer._storage._storage) > 0
            assert len(rb_trainer2.replay_buffer._storage._storage) == len(
                rb_trainer.replay_buffer._storage._storage
            )
            for i, s in enumerate(rb_trainer2.replay_buffer._storage._storage):
                assert (s == rb_trainer.replay_buffer._storage._storage[i]).all()
        elif storage_type == "memmap":
            assert rb_trainer2.replay_buffer._storage._len > 0
            assert (
                rb_trainer2.replay_buffer._storage._storage
                == rb_trainer.replay_buffer._storage._storage
            ).all()

    @pytest.mark.parametrize(
        "storage_type",
        [
            "memmap",
            "list",
            "tensor",
        ],
    )
    @pytest.mark.parametrize(
        "backend",
        [
            "torchsnapshot",
            "torch",
        ],
    )
    @pytest.mark.parametrize(
        "re_init",
        [
            False,
            True,
        ],
    )
    def test_rb_trainer_save(
        self, prioritized, storage_type, backend, re_init, S=10, batch=11, N=3
    ):
        if not _has_ts and backend == "torchsnapshot":
            pytest.skip("torchsnapshot not found")

        torch.manual_seed(0)
        # we overwrite the method to make sure that load_state_dict and state_dict are being called
        state_dict_has_been_called = [False]
        load_state_dict_has_been_called = [False]
        state_dict_has_been_called_td = [False]
        load_state_dict_has_been_called_td = [False]
        ReplayBufferTrainer.state_dict, ReplayBufferTrainer_state_dict = _fun_checker(
            ReplayBufferTrainer.state_dict, state_dict_has_been_called
        )
        (
            ReplayBufferTrainer.load_state_dict,
            ReplayBufferTrainer_load_state_dict,
        ) = _fun_checker(
            ReplayBufferTrainer.load_state_dict, load_state_dict_has_been_called
        )
        TensorDict.state_dict, TensorDict_state_dict = _fun_checker(
            TensorDict.state_dict, state_dict_has_been_called_td
        )
        TensorDict.load_state_dict, TensorDict_load_state_dict = _fun_checker(
            TensorDict.load_state_dict, load_state_dict_has_been_called_td
        )

        os.environ["CKPT_BACKEND"] = backend

        def make_storage():
            if storage_type == "list":
                storage = ListStorage(S)
            elif storage_type == "tensor":
                storage = LazyTensorStorage(S)
            elif storage_type == "memmap":
                storage = LazyMemmapStorage(S)
            else:
                raise NotImplementedError
            return storage

        with tempfile.TemporaryDirectory() as tmpdirname:
            if backend == "torch":
                file = path.join(tmpdirname, "file.pt")
            elif backend == "torchsnapshot":
                file = tmpdirname
            else:
                raise NotImplementedError
            trainer = mocking_trainer(file)

            storage = make_storage()
            if prioritized:
                replay_buffer = TensorDictPrioritizedReplayBuffer(
                    alpha=1.1,
                    beta=0.9,
                    storage=storage,
                )
            else:
                replay_buffer = TensorDictReplayBuffer(
                    storage=storage,
                )

            rb_trainer = ReplayBufferTrainer(replay_buffer=replay_buffer, batch_size=N)
            rb_trainer.register(trainer)
            key1 = "first key"
            key2 = "second key"
            td = TensorDict(
                {
                    key1: torch.randn(batch, 3),
                    key2: torch.randn(batch, 3),
                },
                [batch],
            )
            trainer._process_batch_hook(td)
            # sample from rb
            td_out = trainer._process_optim_batch_hook(td)
            if prioritized:
                td_out.unlock_().set(replay_buffer.priority_key, torch.rand(N))
            trainer._post_loss_hook(td_out)
            trainer.save_trainer(True)

            trainer2 = mocking_trainer()
            storage2 = make_storage()
            if prioritized:
                replay_buffer2 = TensorDictPrioritizedReplayBuffer(
                    alpha=1.1,
                    beta=0.9,
                    storage=storage2,
                )
            else:
                replay_buffer2 = TensorDictReplayBuffer(
                    storage=storage2,
                )
            N = 9
            rb_trainer2 = ReplayBufferTrainer(
                replay_buffer=replay_buffer2, batch_size=N
            )
            rb_trainer2.register(trainer2)
            if re_init:
                trainer2._process_batch_hook(td.to_tensordict().zero_())
            trainer2.load_from_file(file)
            assert state_dict_has_been_called[0]
            assert load_state_dict_has_been_called[0]
            assert state_dict_has_been_called_td[0]
            if re_init:
                assert load_state_dict_has_been_called_td[0]
            if backend != "torch":
                td1 = (
                    storage._storage
                )  # trainer.app_state["state"]["replay_buffer.replay_buffer._storage._storage"]
                td2 = trainer2._modules["replay_buffer"].replay_buffer._storage._storage
                if storage_type == "list":
                    assert all((_td1 == _td2).all() for _td1, _td2 in zip(td1, td2))
                    assert all((_td1 is not _td2) for _td1, _td2 in zip(td1, td2))
                    assert storage2._storage is td2
                else:
                    assert (td1 == td2).all()
                    assert td1 is not td2
                    if storage_type == "memmap":
                        assert td2.is_memmap()
                    assert storage2._storage is td2

        ReplayBufferTrainer.state_dict = ReplayBufferTrainer_state_dict
        ReplayBufferTrainer.load_state_dict = ReplayBufferTrainer_load_state_dict
        TensorDict.state_dict = TensorDict_state_dict
        TensorDict.load_state_dict = TensorDict_load_state_dict


class TestOptimizer:
    @staticmethod
    def _setup():
        torch.manual_seed(0)
        x = torch.randn(5, 10)
        model1 = nn.Linear(10, 20)
        model2 = nn.Linear(10, 20)
        td = TensorDict(
            {
                "loss_1": model1(x).sum(),
                "loss_2": model2(x).sum(),
            },
            batch_size=[],
        )
        model1_params = list(model1.parameters())
        model2_params = list(model2.parameters())
        all_params = model1_params + model2_params
        return model1_params, model2_params, all_params, td

    def test_optimizer_set_as_argument(self):
        _, _, all_params, td = self._setup()

        optimizer = torch.optim.SGD(all_params, lr=1e-3)
        trainer = mocking_trainer(optimizer=optimizer)

        params_before = [torch.clone(p) for p in all_params]
        td_out = trainer._optimizer_hook(td)
        params_after = all_params

        assert "grad_norm_0" in td_out.keys()
        assert all(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, params_after)
        )

    def test_optimizer_set_as_hook(self):
        _, _, all_params, td = self._setup()

        optimizer = torch.optim.SGD(all_params, lr=1e-3)
        trainer = mocking_trainer(optimizer=None)
        hook = OptimizerHook(optimizer)
        hook.register(trainer)

        params_before = [torch.clone(p) for p in all_params]
        td_out = trainer._optimizer_hook(td)
        params_after = all_params

        assert "grad_norm_0" in td_out.keys()
        assert all(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, params_after)
        )

    def test_optimizer_no_optimizer(self):
        _, _, all_params, td = self._setup()

        trainer = mocking_trainer(optimizer=None)

        params_before = [torch.clone(p) for p in all_params]
        td_out = trainer._optimizer_hook(td)
        params_after = all_params

        assert not [key for key in td_out.keys() if key.startswith("grad_norm_")]
        assert all(
            torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, params_after)
        )

    def test_optimizer_hook_loss_components_empty(self):
        model = nn.Linear(10, 20)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="loss_components list cannot be empty"):
            OptimizerHook(optimizer, loss_components=[])

    def test_optimizer_hook_loss_components_partial(self):
        model1_params, model2_params, all_params, td = self._setup()

        optimizer = torch.optim.SGD(all_params, lr=1e-3)
        trainer = mocking_trainer(optimizer=None)
        hook = OptimizerHook(optimizer, loss_components=["loss_1"])
        hook.register(trainer)

        model1_params_before = [torch.clone(p) for p in model1_params]
        model2_params_before = [torch.clone(p) for p in model2_params]
        td_out = trainer._optimizer_hook(td)
        model1_params_after = model1_params
        model2_params_after = model2_params

        assert "grad_norm_0" in td_out.keys()
        assert all(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(model1_params_before, model1_params_after)
        )
        assert all(
            torch.equal(p_before, p_after)
            for p_before, p_after in zip(model2_params_before, model2_params_after)
        )

    def test_optimizer_hook_loss_components_none(self):
        model1_params, model2_params, all_params, td = self._setup()

        optimizer = torch.optim.SGD(all_params, lr=1e-3)
        trainer = mocking_trainer(optimizer=None)
        hook = OptimizerHook(optimizer, loss_components=None)
        hook.register(trainer)

        model1_params_before = [torch.clone(p) for p in model1_params]
        model2_params_before = [torch.clone(p) for p in model2_params]
        td_out = trainer._optimizer_hook(td)
        model1_params_after = model1_params
        model2_params_after = model2_params

        assert "grad_norm_0" in td_out.keys()
        assert all(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(model1_params_before, model1_params_after)
        )
        assert all(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(model2_params_before, model2_params_after)
        )

    def test_optimizer_multiple_hooks(self):
        model1_params, model2_params, _, td = self._setup()

        trainer = mocking_trainer(optimizer=None)

        optimizer1 = torch.optim.SGD(model1_params, lr=1e-3)
        hook1 = OptimizerHook(optimizer1, loss_components=["loss_1"])
        hook1.register(trainer, name="optimizer1")

        optimizer2 = torch.optim.Adam(model2_params, lr=1e-4)
        hook2 = OptimizerHook(optimizer2, loss_components=["loss_2"])
        hook2.register(trainer, name="optimizer2")

        model1_params_before = [torch.clone(p) for p in model1_params]
        model2_params_before = [torch.clone(p) for p in model2_params]
        td_out = trainer._optimizer_hook(td)
        model1_params_after = model1_params
        model2_params_after = model2_params

        assert "grad_norm_0" in td_out.keys()
        assert "grad_norm_1" in td_out.keys()
        assert all(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(model1_params_before, model1_params_after)
        )
        assert all(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(model2_params_before, model2_params_after)
        )


class TestLogReward:
    @pytest.mark.parametrize("logname", ["a", "b"])
    @pytest.mark.parametrize("pbar", [True, False])
    def test_log_reward(self, logname, pbar):
        trainer = mocking_trainer()
        trainer.collected_frames = 0

        log_reward = LogReward(logname, log_pbar=pbar)
        trainer.register_op("pre_steps_log", log_reward)
        td = TensorDict({REWARD_KEY: torch.ones(3)}, [3])
        trainer._pre_steps_log_hook(td)
        if _has_tqdm and pbar:
            assert trainer._pbar_str[logname] == 1
        else:
            assert logname not in trainer._pbar_str
        assert trainer._log_dict[logname][-1] == 1

    @pytest.mark.parametrize("logname", ["a", "b"])
    @pytest.mark.parametrize("pbar", [True, False])
    def test_log_reward_register(self, logname, pbar):
        trainer = mocking_trainer()
        trainer.collected_frames = 0

        log_reward = LogReward(logname, log_pbar=pbar)
        log_reward.register(trainer)
        td = TensorDict({REWARD_KEY: torch.ones(3)}, [3])
        trainer._pre_steps_log_hook(td)
        if _has_tqdm and pbar:
            assert trainer._pbar_str[logname] == 1
        else:
            assert logname not in trainer._pbar_str
        assert trainer._log_dict[logname][-1] == 1


class TestRewardNorm:
    def test_reward_norm(self):
        torch.manual_seed(0)
        trainer = mocking_trainer()

        reward_normalizer = RewardNormalizer()
        reward_normalizer.register(trainer)

        batch = 10
        reward = torch.randn(batch, 1)
        td = TensorDict({REWARD_KEY: reward.clone()}, [batch])
        td_out = trainer._process_batch_hook(td)
        assert (td_out.get(REWARD_KEY) == reward).all()
        assert not reward_normalizer._normalize_has_been_called

        td_norm = trainer._process_optim_batch_hook(td)
        assert reward_normalizer._normalize_has_been_called
        torch.testing.assert_close(td_norm.get(REWARD_KEY).mean(), torch.zeros([]))
        torch.testing.assert_close(td_norm.get(REWARD_KEY).std(), torch.ones([]))

    def test_reward_norm_state_dict(self):
        torch.manual_seed(0)
        trainer = mocking_trainer()

        reward_normalizer = RewardNormalizer()
        reward_normalizer.register(trainer)

        batch = 10
        reward = torch.randn(batch, 1)
        td = TensorDict({REWARD_KEY: reward.clone()}, [batch])
        trainer._process_batch_hook(td)
        trainer._process_optim_batch_hook(td)
        state_dict = trainer.state_dict()

        trainer2 = mocking_trainer()

        reward_normalizer2 = RewardNormalizer()
        reward_normalizer2.register(trainer2)
        trainer2.load_state_dict(state_dict)
        for key, item in reward_normalizer._reward_stats.items():
            assert item == reward_normalizer2._reward_stats[key]

    @pytest.mark.parametrize(
        "backend",
        [
            "torchsnapshot",
            "torch",
        ],
    )
    def test_reward_norm_save(self, backend):
        if not _has_ts and backend == "torchsnapshot":
            pytest.skip("torchsnapshot not found")

        os.environ["CKPT_BACKEND"] = backend

        state_dict_has_been_called = [False]
        load_state_dict_has_been_called = [False]
        RewardNormalizer.state_dict, RewardNormalizer_state_dict = _fun_checker(
            RewardNormalizer.state_dict, state_dict_has_been_called
        )
        (
            RewardNormalizer.load_state_dict,
            RewardNormalizer_load_state_dict,
        ) = _fun_checker(
            RewardNormalizer.load_state_dict, load_state_dict_has_been_called
        )

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmpdirname:
            if backend == "torch":
                file = path.join(tmpdirname, "file.pt")
            elif backend == "torchsnapshot":
                file = tmpdirname
            else:
                raise NotImplementedError
            trainer = mocking_trainer(file)
            reward_normalizer = RewardNormalizer()
            reward_normalizer.register(trainer)

            batch = 10
            reward = torch.randn(batch, 1)
            td = TensorDict({REWARD_KEY: reward.clone()}, [batch])
            trainer._process_batch_hook(td)
            trainer._process_optim_batch_hook(td)
            trainer.save_trainer(True)

            trainer2 = mocking_trainer()
            reward_normalizer2 = RewardNormalizer()
            reward_normalizer2.register(trainer2)
            trainer2.load_from_file(file)

        RewardNormalizer.state_dict = RewardNormalizer_state_dict
        RewardNormalizer.load_state_dict = RewardNormalizer_load_state_dict


def test_masking():
    torch.manual_seed(0)
    trainer = mocking_trainer()

    trainer.register_op("batch_process", mask_batch)
    batch = 10
    td = TensorDict(
        {
            ("collector", "mask"): torch.zeros(batch, dtype=torch.bool).bernoulli_(),
            "tensor": torch.randn(batch, 51),
        },
        [batch],
    )
    td_out = trainer._process_batch_hook(td)
    assert td_out.shape[0] == td.get(("collector", "mask")).sum()
    assert (td["tensor"][td[("collector", "mask")]] == td_out["tensor"]).all()


class TestSubSampler:
    def test_subsampler(self):
        torch.manual_seed(0)
        trainer = mocking_trainer()

        batch_size = 10
        sub_traj_len = 5

        key1 = "key1"
        key2 = "key2"

        subsampler = BatchSubSampler(batch_size=batch_size, sub_traj_len=sub_traj_len)
        subsampler.register(trainer)

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

    def test_subsampler_state_dict(self):
        trainer = mocking_trainer()

        batch_size = 10
        sub_traj_len = 5

        key1 = "key1"
        key2 = "key2"

        subsampler = BatchSubSampler(batch_size=batch_size, sub_traj_len=sub_traj_len)
        subsampler.register(trainer)

        td = TensorDict(
            {
                key1: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
                key2: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
            },
            [2, 10],
        )

        torch.manual_seed(0)
        td0 = trainer._process_optim_batch_hook(td)
        trainer2 = mocking_trainer()
        subsampler2 = BatchSubSampler(batch_size=batch_size, sub_traj_len=sub_traj_len)
        subsampler2.register(trainer2)
        trainer2.load_state_dict(trainer.state_dict())
        torch.manual_seed(0)
        td1 = trainer2._process_optim_batch_hook(td)
        assert (td0 == td1).all()


@pytest.mark.skipif(not _has_gym, reason="No gym library")
@pytest.mark.skipif(not _has_tb, reason="No tensorboard library")
class TestRecorder:
    def _get_args(self):
        args = Namespace()
        args.env_name = PONG_VERSIONED()
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
        args.image_size = 84
        args.collector_device = ["cpu"]
        args.categorical_action_encoding = False
        return args

    def test_recorder(self, N=8):
        args = self._get_args()
        with tempfile.TemporaryDirectory() as folder:
            logger = TensorboardLogger(exp_name=folder)

            environment = transformed_env_constructor(
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
                environment=environment,
                record_interval=args.record_interval,
            )
            trainer = mocking_trainer()
            recorder.register(trainer)

            for _ in range(N):
                recorder(None)

            for (_, _, filenames) in walk(folder):
                filename = filenames[0]
                break

            for _ in range(3):
                ea = event_accumulator.EventAccumulator(
                    path.join(folder, filename),
                    size_guidance={
                        event_accumulator.IMAGES: 0,
                    },
                )
                ea.Reload()
                img = ea.Images(f"tmp_{PONG_VERSIONED()}_video")
                try:
                    assert len(img) == N // args.record_interval
                    break
                except AssertionError:
                    sleep(0.1)

    @pytest.mark.parametrize(
        "backend",
        [
            "torchsnapshot",
            "torch",
        ],
    )
    def test_recorder_load(self, backend, N=8):
        if not _has_ts and backend == "torchsnapshot":
            pytest.skip("torchsnapshot not found")

        os.environ["CKPT_BACKEND"] = backend
        state_dict_has_been_called = [False]
        load_state_dict_has_been_called = [False]
        Recorder.state_dict, Recorder_state_dict = _fun_checker(
            Recorder.state_dict, state_dict_has_been_called
        )
        (
            Recorder.load_state_dict,
            Recorder_load_state_dict,
        ) = _fun_checker(Recorder.load_state_dict, load_state_dict_has_been_called)

        args = self._get_args()

        def _make_recorder_and_trainer(tmpdirname):
            logger = TensorboardLogger(exp_name=f"{tmpdirname}/tb")
            if backend == "torch":
                file = path.join(tmpdirname, "file.pt")
            elif backend == "torchsnapshot":
                file = tmpdirname
            else:
                raise NotImplementedError
            trainer = mocking_trainer(file)

            environment = transformed_env_constructor(
                args,
                video_tag="tmp",
                norm_obs_only=True,
                stats={"loc": 0, "scale": 1},
                logger=logger,
            )()
            environment.rollout(2)

            recorder = Recorder(
                record_frames=args.record_frames,
                frame_skip=args.frame_skip,
                policy_exploration=None,
                environment=environment,
                record_interval=args.record_interval,
            )
            recorder.register(trainer)
            return trainer, recorder, file

        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer, recorder, file = _make_recorder_and_trainer(tmpdirname)
            for _ in range(N):
                recorder(None)
            trainer.save_trainer(True)
            with tempfile.TemporaryDirectory() as tmpdirname2:
                trainer2, recorder2, _ = _make_recorder_and_trainer(tmpdirname2)
                trainer2.load_from_file(file)
            assert recorder2._count == 8
            assert state_dict_has_been_called[0]
            assert load_state_dict_has_been_called[0]
        Recorder.state_dict = Recorder_state_dict
        Recorder.load_state_dict = Recorder_load_state_dict


def test_updateweights():
    torch.manual_seed(0)
    trainer = mocking_trainer()

    T = 5
    update_weights = UpdateWeights(trainer.collector, T)
    update_weights.register(trainer)
    for t in range(T):
        trainer._post_steps_hook()
        assert trainer.collector.called_update_policy_weights_ is (t == T - 1)
    assert trainer.collector.called_update_policy_weights_


class TestCountFrames:
    def test_countframes(self):
        torch.manual_seed(0)
        trainer = mocking_trainer()

        frame_skip = 3
        batch = 10
        count_frames = CountFramesLog(frame_skip=frame_skip)
        count_frames.register(trainer)
        td = TensorDict(
            {("collector", "mask"): torch.zeros(batch, dtype=torch.bool).bernoulli_()},
            [batch],
        )
        trainer._pre_steps_log_hook(td)
        assert (
            count_frames.frame_count == td.get(("collector", "mask")).sum() * frame_skip
        )

    @pytest.mark.parametrize(
        "backend",
        [
            "torchsnapshot",
            "torch",
        ],
    )
    def test_countframes_load(self, backend):
        if not _has_ts and backend == "torchsnapshot":
            pytest.skip("torchsnapshot not found")

        os.environ["CKPT_BACKEND"] = backend
        state_dict_has_been_called = [False]
        load_state_dict_has_been_called = [False]
        CountFramesLog.state_dict, CountFramesLog_state_dict = _fun_checker(
            CountFramesLog.state_dict, state_dict_has_been_called
        )
        (
            CountFramesLog.load_state_dict,
            CountFramesLog_load_state_dict,
        ) = _fun_checker(
            CountFramesLog.load_state_dict, load_state_dict_has_been_called
        )

        def _make_countframe_and_trainer(tmpdirname):
            if backend == "torch":
                file = path.join(tmpdirname, "file.pt")
            elif backend == "torchsnapshot":
                file = tmpdirname
            else:
                raise NotImplementedError
            trainer = mocking_trainer(file)
            count_frames = CountFramesLog(frame_skip=frame_skip)
            count_frames.register(trainer)
            return trainer, count_frames, file

        torch.manual_seed(0)

        frame_skip = 3
        batch = 10
        with tempfile.TemporaryDirectory() as tmpdirname, tempfile.TemporaryDirectory() as tmpdirname2:
            trainer, count_frames, file = _make_countframe_and_trainer(tmpdirname)
            td = TensorDict(
                {
                    ("collector", "mask"): torch.zeros(
                        batch, dtype=torch.bool
                    ).bernoulli_()
                },
                [batch],
            )
            trainer._pre_steps_log_hook(td)
            trainer.save_trainer(True)
            trainer2, count_frames2, _ = _make_countframe_and_trainer(tmpdirname2)
            trainer2.load_from_file(file)
            assert (
                count_frames2.frame_count
                == td.get(("collector", "mask")).sum() * frame_skip
            )
            assert state_dict_has_been_called[0]
            assert load_state_dict_has_been_called[0]
        CountFramesLog.state_dict = CountFramesLog_state_dict
        CountFramesLog.load_state_dict = CountFramesLog_load_state_dict


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
