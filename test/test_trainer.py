# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import tempfile
import warnings
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from os import path, walk
from time import sleep

import pytest
import torch
from torch import nn

_has_tb = importlib.util.find_spec("tensorboard") is not None

from tensordict import TensorDict
from torchrl.checkpoint import Checkpoint
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs.libs.gym import _has_gym
from torchrl.objectives import LossModule
from torchrl.testing import PONG_VERSIONED
from torchrl.trainers import (
    Learner,
    LearnerStepRequest,
    LocalLearnerGroup,
    LogValidationReward,
    Trainer,
)
from torchrl.trainers.algorithms.a2c import A2CTrainer
from torchrl.trainers.algorithms.cql import CQLTrainer
from torchrl.trainers.algorithms.ddpg import DDPGTrainer
from torchrl.trainers.algorithms.dqn import DQNTrainer
from torchrl.trainers.algorithms.iql import IQLTrainer
from torchrl.trainers.algorithms.offline_to_online import OfflineToOnlineTrainer
from torchrl.trainers.algorithms.ppo import PPOTrainer
from torchrl.trainers.algorithms.reinforce import ReinforceTrainer
from torchrl.trainers.algorithms.sac import SACTrainer
from torchrl.trainers.algorithms.td3 import TD3OptimizationStepper, TD3Trainer
from torchrl.trainers.helpers import transformed_env_constructor
from torchrl.trainers.trainers import (
    _has_tqdm,
    _has_ts,
    _torch_load_defaults,
    BatchSubSampler,
    CountFramesLog,
    DefaultOptimizationStepper,
    EarlyStopping,
    LogScalar,
    LRSchedulerHook,
    mask_batch,
    OptimizationStepper,
    OptimizerHook,
    ReplayBufferTrainer,
    REWARD_KEY,
    RewardNormalizer,
    SelectKeys,
    UpdateWeights,
    ValueEstimatorHook,
)

_has_ale = importlib.util.find_spec("ale_py") is not None


def _fun_checker(fun, checker):
    def new_fun(*args, **kwargs):
        checker[0] = True
        return fun(*args, **kwargs)

    return new_fun, fun


class MockingOptim:
    param_groups = [{"params": []}]


class MockingCollector:
    called_update_policy_weights_ = False

    def __init__(self, source_policy=None):
        self.source_policy = source_policy
        self.policy = deepcopy(source_policy) if source_policy is not None else None
        self.called_update_policy_weights_ = False

    def set_seed(self, seed, **kwargs):
        return seed

    def update_policy_weights_(self, policy=None):
        self.called_update_policy_weights_ = True
        policy = self.source_policy if policy is None else policy
        if policy is not None:
            self.policy.load_state_dict(policy.state_dict())

    def shutdown(self):
        pass

    def state_dict(self):
        if self.policy is None:
            return {}
        return {"policy": self.policy.state_dict()}

    def load_state_dict(self, state_dict):
        if self.policy is not None:
            self.policy.load_state_dict(state_dict["policy"])


class MockingIterableCollector(MockingCollector):
    def __init__(self, batches):
        super().__init__()
        self._batches = batches
        self.init_random_frames = 10**9
        self.shutdown_calls = 0

    def __iter__(self):
        return iter(self._batches)

    def shutdown(self):
        self.shutdown_calls += 1


class MockingLossModule(nn.Module):
    pass


class MockingLogger:
    def __init__(self):
        self.value = 0

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict):
        self.value = state_dict["value"]


class MockingReplayBufferHook:
    def __init__(self, fraction=0.0):
        self.replay_buffer = object()
        self.fraction = fraction

    def state_dict(self):
        return {"fraction": self.fraction}

    def load_state_dict(self, state_dict):
        self.fraction = state_dict["fraction"]


_mocking_optim = MockingOptim()


def mocking_trainer(
    file=None,
    optimizer=_mocking_optim,
    checkpoint=None,
    logger=None,
    with_policy=False,
) -> Trainer:
    loss_module = MockingLossModule()
    if with_policy:
        loss_module.actor_network = nn.Linear(2, 2)
    collector = MockingCollector(getattr(loss_module, "actor_network", None))
    trainer = Trainer(
        collector=collector,
        total_frames=None,
        frame_skip=None,
        optim_steps_per_batch=None,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        save_trainer_file=file,
        checkpoint=checkpoint,
    )
    trainer._pbar_str = OrderedDict()
    return trainer


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_unified_trainer_checkpoint(tmp_path, format):
    path = tmp_path / "trainer-checkpoint"
    trainer = mocking_trainer(
        file=path,
        checkpoint=Checkpoint(format=format),
    )
    assert {"collector", "loss_module", "trainer_state"}.issubset(
        trainer.checkpoint.components
    )
    trainer.collected_frames = 41
    trainer._optim_count = 7
    trainer.save_trainer(force_save=True)
    manifest = Checkpoint.manifest(path)
    assert {"collector", "loss_module", "trainer_state"}.issubset(
        manifest["components"]
    )

    restored = mocking_trainer(checkpoint=Checkpoint())
    restored.load_from_file(path)
    assert restored.collected_frames == 41
    assert restored._optim_count == 7


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_unified_trainer_resyncs_collector_policy_after_load(tmp_path, format):
    path = tmp_path / "trainer-checkpoint"
    source = mocking_trainer(
        file=path,
        checkpoint=Checkpoint(format=format),
        with_policy=True,
    )
    with torch.no_grad():
        for parameter in source.loss_module.actor_network.parameters():
            parameter.fill_(3.0)
    assert any(
        not torch.equal(source.collector.policy.state_dict()[key], value)
        for key, value in source.loss_module.actor_network.state_dict().items()
    )
    source.save_trainer(force_save=True)

    restored = mocking_trainer(checkpoint=Checkpoint(), with_policy=True)
    restored.load_from_file(path)

    assert restored.collector.called_update_policy_weights_
    for key, value in restored.loss_module.actor_network.state_dict().items():
        torch.testing.assert_close(restored.collector.policy.state_dict()[key], value)


def test_unified_trainer_default_component_selection(tmp_path):
    path = tmp_path / "trainer-checkpoint"
    trainer = mocking_trainer(
        file=path,
        checkpoint=Checkpoint(save_components={"trainer_state"}),
    )
    trainer.save_trainer(force_save=True)
    assert set(Checkpoint.manifest(path)["components"]) == {"trainer_state"}


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

    @pytest.mark.parametrize("backend", ["torchsnapshot", "torch", "memmap"])
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
            elif backend == "memmap":
                file = path.join(tmpdirname, "ckpt")
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
            assert load_state_dict_has_been_called[0]

        SelectKeys.state_dict = SelectKeys_state_dict
        SelectKeys.load_state_dict = SelectKeys_load_state_dict


class TestLoadFromFile:
    def test_unified_load_options_strictness_and_save_format(
        self, tmp_path, monkeypatch
    ):
        file = tmp_path / "unified"
        source = mocking_trainer(file=file, checkpoint=Checkpoint())
        source.collected_frames = 11
        source.save_trainer(force_save=True)

        captured = []
        true_load = torch.load

        def spy_load(*args, **kwargs):
            captured.append(kwargs.copy())
            return true_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", spy_load)
        restored_default = mocking_trainer()
        restored_default.load_from_file(file)
        assert restored_default.collected_frames == 11
        assert restored_default.checkpoint is None
        assert not captured

        captured.clear()
        restored = mocking_trainer(logger=MockingLogger())
        restored.load_from_file(
            file,
            strict="ignore",
            map_location="cpu",
            weights_only=True,
            mmap=False,
        )
        assert restored.collected_frames == 11
        assert restored.checkpoint is None
        assert not captured

    @pytest.mark.parametrize("format", ["directory", "archive"])
    def test_unified_hook_replay_buffer_roundtrip(self, tmp_path, format):
        file = tmp_path / "unified"
        source = mocking_trainer(file=file, checkpoint=Checkpoint(format=format))
        replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(10))
        ReplayBufferTrainer(replay_buffer=replay_buffer, batch_size=2).register(source)
        data = TensorDict({"obs": torch.randn(6, 4)}, [6])
        replay_buffer.extend(data)
        source.save_trainer(force_save=True)
        assert "replay_buffer" in Checkpoint.manifest(file)["components"]

        restored = mocking_trainer(checkpoint=Checkpoint())
        restored_replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(10))
        ReplayBufferTrainer(
            replay_buffer=restored_replay_buffer, batch_size=2
        ).register(restored)
        restored.load_from_file(file)
        torch.testing.assert_close(restored_replay_buffer[:]["obs"], data["obs"])

    def test_unified_hook_owned_replay_buffer_state(self, tmp_path):
        file = tmp_path / "unified"
        source = mocking_trainer(file=file, checkpoint=Checkpoint())
        source_hook = MockingReplayBufferHook(0.75)
        source.register_module("replay_buffer", source_hook)
        assert source.checkpoint.components["replay_buffer"] is source_hook
        source.save_trainer(force_save=True)

        restored = mocking_trainer(checkpoint=Checkpoint())
        restored_hook = MockingReplayBufferHook()
        restored.register_module("replay_buffer", restored_hook)
        restored.load_from_file(file)
        assert restored_hook.fraction == 0.75

    def test_legacy_save_warning_categories(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CKPT_BACKEND", "torch")
        trainer = mocking_trainer(file=tmp_path / "legacy.pt")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.save_trainer(force_save=True)
        assert any(
            item.category is FutureWarning and "v0.15" in str(item.message)
            for item in caught
        )
        assert any(
            item.category is DeprecationWarning and "v0.16" in str(item.message)
            for item in caught
        )

    def test_torch_backend_mmap_default(self, monkeypatch):
        monkeypatch.setenv("CKPT_BACKEND", "torch")
        captured = {}
        true_load = torch.load

        def spy_load(*args, **kwargs):
            captured.update(kwargs)
            return true_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", spy_load)
        with tempfile.TemporaryDirectory() as tmpdirname:
            file = path.join(tmpdirname, "file.pt")
            trainer = mocking_trainer(file=file)
            trainer.save_trainer(force_save=True)

            trainer2 = mocking_trainer()
            trainer2.load_from_file(file)
            # mmap defaults to False on Windows (a mapped checkpoint locks the
            # file) and weights_only to False on torch < 2.4 (its weights-only
            # unpickler rejects torch.device).
            defaults = _torch_load_defaults()
            assert captured["mmap"] is defaults["mmap"]
            assert captured["weights_only"] is defaults["weights_only"]

            captured.clear()
            trainer2.load_from_file(file, mmap=False)
            assert captured["mmap"] is False

    def test_resume_and_resave_replay_buffer(self, monkeypatch):
        # Resuming with mmap=True leaves the replay buffer storage backed by
        # the checkpoint file; re-saving to the same path must not corrupt it.
        monkeypatch.setenv("CKPT_BACKEND", "torch")

        def make_trainer(file=None):
            trainer = mocking_trainer(file=file)
            replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(10))
            ReplayBufferTrainer(replay_buffer=replay_buffer, batch_size=2).register(
                trainer
            )
            return trainer, replay_buffer

        with tempfile.TemporaryDirectory() as tmpdirname:
            file = path.join(tmpdirname, "file.pt")
            data = TensorDict({"obs": torch.randn(10, 4)}, [10])

            trainer, replay_buffer = make_trainer(file=file)
            replay_buffer.extend(data)
            trainer.save_trainer(force_save=True)

            trainer2, replay_buffer2 = make_trainer(file=file)
            trainer2.load_from_file(file)
            trainer2.save_trainer(force_save=True)

            trainer3, replay_buffer3 = make_trainer()
            trainer3.load_from_file(file)
            torch.testing.assert_close(replay_buffer3[:]["obs"], data["obs"])

    def test_torch_backend_save_is_atomic(self, monkeypatch):
        monkeypatch.setenv("CKPT_BACKEND", "torch")
        true_save = torch.save

        def failing_save(*args, **kwargs):
            true_save(*args, **kwargs)
            raise RuntimeError("interrupted save")

        with tempfile.TemporaryDirectory() as tmpdirname:
            file = path.join(tmpdirname, "file.pt")
            trainer = mocking_trainer(file=file)
            trainer.save_trainer(force_save=True)
            reference = torch.load(file, weights_only=True, mmap=False)

            trainer.collected_frames = 42
            monkeypatch.setattr(torch, "save", failing_save)
            with pytest.raises(RuntimeError, match="interrupted save"):
                trainer.save_trainer(force_save=True)

            # the previous checkpoint is intact and no temp file is left over
            loaded = torch.load(file, weights_only=True, mmap=False)
            assert (
                loaded["state"]["collected_frames"]
                == reference["state"]["collected_frames"]
            )
            assert os.listdir(tmpdirname) == ["file.pt"]


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
        # The ReplayBufferTrainer.extend method calls .cpu() which creates a new TensorDict
        # so we can't expect the same object identity, but the content should be the same
        assert td_out.shape == td.shape
        assert td_out.device == torch.device("cpu")
        assert td.device is None or td.device == torch.device("cpu")

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
            trainer2.load_from_file(file, weights_only=False)
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

        log_reward = LogScalar(REWARD_KEY, logname, log_pbar=pbar)
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

        log_reward = LogScalar(REWARD_KEY, logname, log_pbar=pbar)
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

    def test_reward_norm_load_state_dict_decouples_from_source(self):
        # loading a state dict must clone the incoming tensors so that
        # mutating the source afterwards does not corrupt the normalizer
        # stats (e.g. tensors mmap-backed by torch.load(mmap=True))
        torch.manual_seed(0)
        reward_normalizer = RewardNormalizer()
        batch = 10
        reward = torch.randn(batch, 1)
        td = TensorDict({REWARD_KEY: reward.clone()}, [batch])
        reward_normalizer.update_reward_stats(td)
        state_dict = reward_normalizer.state_dict()

        reward_normalizer2 = RewardNormalizer()
        reward_normalizer2.load_state_dict(state_dict)
        before = {
            key: item.clone() if isinstance(item, torch.Tensor) else item
            for key, item in reward_normalizer2._reward_stats.items()
        }
        for item in state_dict["_reward_stats"].values():
            if isinstance(item, torch.Tensor):
                item.fill_(1e9)
        for key, item in before.items():
            assert reward_normalizer2._reward_stats[key] == item

    @pytest.mark.parametrize(
        "backend",
        [
            "torchsnapshot",
            "torch",
            "memmap",
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
            elif backend == "memmap":
                file = path.join(tmpdirname, "ckpt")
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
            assert state_dict_has_been_called[0]
            assert load_state_dict_has_been_called[0]

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
@pytest.mark.skipif(
    not _has_ale,
    reason="ALE not available (missing ale_py); skipping Atari gym tests.",
)
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
        from tensorboard.backend.event_processing import event_accumulator
        from torchrl.record.loggers.tensorboard import TensorboardLogger

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

            recorder = LogValidationReward(
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

            for _, _, filenames in walk(folder):
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
            "memmap",
        ],
    )
    def test_recorder_load(self, backend, N=8):
        from torchrl.record.loggers.tensorboard import TensorboardLogger

        if not _has_ts and backend == "torchsnapshot":
            pytest.skip("torchsnapshot not found")

        os.environ["CKPT_BACKEND"] = backend
        state_dict_has_been_called = [False]
        load_state_dict_has_been_called = [False]
        LogValidationReward.state_dict, Recorder_state_dict = _fun_checker(
            LogValidationReward.state_dict, state_dict_has_been_called
        )
        (LogValidationReward.load_state_dict, Recorder_load_state_dict,) = _fun_checker(
            LogValidationReward.load_state_dict, load_state_dict_has_been_called
        )

        args = self._get_args()

        def _make_recorder_and_trainer(tmpdirname):
            logger = TensorboardLogger(exp_name=f"{tmpdirname}/tb")
            if backend == "torch":
                file = path.join(tmpdirname, "file.pt")
            elif backend == "torchsnapshot":
                file = tmpdirname
            elif backend == "memmap":
                file = path.join(tmpdirname, "ckpt")
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

            recorder = LogValidationReward(
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
        LogValidationReward.state_dict = Recorder_state_dict
        LogValidationReward.load_state_dict = Recorder_load_state_dict


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


@pytest.mark.parametrize("format", ["directory", "archive"])
def test_updateweights_checkpoint_counter(tmp_path, format):
    checkpoint_path = tmp_path / "trainer-checkpoint"
    trainer = mocking_trainer(
        file=checkpoint_path, checkpoint=Checkpoint(format=format)
    )
    update_weights = UpdateWeights(trainer.collector, 5)
    update_weights.register(trainer)
    for _ in range(3):
        trainer._post_steps_hook()
    trainer.save_trainer(force_save=True)

    restored = mocking_trainer(checkpoint=Checkpoint())
    restored_update_weights = UpdateWeights(restored.collector, 5)
    restored_update_weights.register(restored)
    restored.load_from_file(checkpoint_path)

    assert restored_update_weights.counter == 3
    restored._post_steps_hook()
    assert not restored.collector.called_update_policy_weights_
    restored._post_steps_hook()
    assert restored.collector.called_update_policy_weights_


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
            "memmap",
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
            elif backend == "memmap":
                file = path.join(tmpdirname, "ckpt")
            else:
                raise NotImplementedError
            trainer = mocking_trainer(file)
            count_frames = CountFramesLog(frame_skip=frame_skip)
            count_frames.register(trainer)
            return trainer, count_frames, file

        torch.manual_seed(0)

        frame_skip = 3
        batch = 10
        with (
            tempfile.TemporaryDirectory() as tmpdirname,
            tempfile.TemporaryDirectory() as tmpdirname2,
        ):
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


class _CountingLossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, td: TensorDict):
        self.forward_calls += 1
        return TensorDict({"loss": torch.zeros(())}, [])


class _CountingStepper(OptimizationStepper):
    def __init__(self):
        self.calls = 0

    def step(self, trainer: Trainer, sub_batch: TensorDict) -> TensorDict:
        self.calls += 1
        return TensorDict({"loss": torch.zeros(())}, [])

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state_dict):
        self.calls = state_dict["calls"]


class TestOptimizationStepper:
    def _make_trainer(self, loss_module, optimization_stepper=None, optimizer=None):
        trainer = Trainer(
            collector=MockingCollector(),
            total_frames=None,
            frame_skip=None,
            optim_steps_per_batch=1,
            loss_module=loss_module,
            optimizer=optimizer,
            optimization_stepper=optimization_stepper,
        )
        trainer._pbar_str = OrderedDict()
        return trainer

    def test_custom_stepper_used(self):
        loss_module = _CountingLossModule()
        stepper = _CountingStepper()
        trainer = self._make_trainer(
            loss_module=loss_module,
            optimization_stepper=stepper,
        )
        td = TensorDict({"x": torch.randn(3)}, [])
        trainer.optim_steps(td)
        assert stepper.calls == 1
        assert loss_module.forward_calls == 0

    def test_stepper_checkpoint_roundtrip(self, tmp_path):
        """Stepper state survives save/load via Trainer checkpointing."""
        os.environ["CKPT_BACKEND"] = "torch"

        loss_module = _CountingLossModule()
        stepper = _CountingStepper()
        trainer1 = self._make_trainer(
            loss_module=loss_module,
            optimization_stepper=stepper,
            optimizer=None,
        )
        stepper.calls = 5

        file = str(tmp_path / "trainer.pt")
        trainer1.save_trainer_file = file
        trainer1.save_trainer(force_save=True)

        stepper2 = _CountingStepper()
        trainer2 = self._make_trainer(
            loss_module=_CountingLossModule(),
            optimization_stepper=stepper2,
            optimizer=None,
        )
        trainer2.load_from_file(file)

        assert stepper2.calls == 5
        sd = torch.load(file)
        assert "optimization_stepper" in sd


class TestPostOptimCompleteLog:
    def _make_trainer(
        self,
        loss_module,
        optimization_stepper=None,
        optimizer=None,
        auto_log_optim_steps=True,
    ):
        trainer = Trainer(
            collector=MockingCollector(),
            total_frames=None,
            frame_skip=None,
            optim_steps_per_batch=1,
            loss_module=loss_module,
            optimizer=optimizer,
            optimization_stepper=optimization_stepper,
            auto_log_optim_steps=auto_log_optim_steps,
        )
        trainer._pbar_str = OrderedDict()
        return trainer

    def test_hook_receives_optim_steps_and_averaged_losses(self):
        captured = {}

        def capture_hook(optim_steps, average_losses):
            captured["optim_steps"] = optim_steps
            captured["average_losses"] = average_losses
            return None

        stepper = _CountingStepper()
        loss_module = _CountingLossModule()
        trainer = self._make_trainer(
            loss_module=loss_module,
            optimization_stepper=stepper,
        )
        trainer.register_op("post_optim_complete_log", capture_hook)
        td = TensorDict({"x": torch.randn(3)}, [])
        trainer.optim_steps(td)

        assert stepper.calls == 1
        assert captured["optim_steps"] == trainer._optim_count
        assert "loss" in captured["average_losses"].keys()
        assert "optim_steps" in trainer._log_dict
        assert "loss" in trainer._log_dict
        assert trainer._log_dict["optim_steps"][-1] == trainer._optim_count

    def test_hook_return_logs_extra_metric(self):
        def extra_metric_hook(optim_steps, average_losses):
            return {"custom_metric": 1.0}

        stepper = _CountingStepper()
        trainer = self._make_trainer(
            loss_module=_CountingLossModule(),
            optimization_stepper=stepper,
        )
        trainer.register_op("post_optim_complete_log", extra_metric_hook)
        td = TensorDict({"x": torch.randn(3)}, [])
        trainer.optim_steps(td)

        assert "custom_metric" in trainer._log_dict
        assert trainer._log_dict["custom_metric"][-1] == 1.0
        assert "optim_steps" in trainer._log_dict
        assert "loss" in trainer._log_dict

    def test_auto_log_optim_steps_disabled(self):
        captured = {}

        def capture_hook(optim_steps, average_losses):
            captured["optim_steps"] = optim_steps
            captured["average_losses"] = average_losses
            return None

        stepper = _CountingStepper()
        trainer = self._make_trainer(
            loss_module=_CountingLossModule(),
            optimization_stepper=stepper,
            auto_log_optim_steps=False,
        )
        trainer.register_op("post_optim_complete_log", capture_hook)
        td = TensorDict({"x": torch.randn(3)}, [])
        trainer.optim_steps(td)

        # Hook still fires and sees the values
        assert captured["optim_steps"] == trainer._optim_count
        assert "loss" in captured["average_losses"].keys()
        # but the trainer doesn't auto-log them
        assert "optim_steps" not in trainer._log_dict
        assert "loss" not in trainer._log_dict

    @pytest.mark.parametrize(
        "trainer_cls",
        [
            SACTrainer,
            PPOTrainer,
            DQNTrainer,
            DDPGTrainer,
            IQLTrainer,
            CQLTrainer,
            TD3Trainer,
        ],
    )
    def test_subclass_exposes_auto_log_optim_steps(self, trainer_cls):
        """Every Trainer subclass must surface auto_log_optim_steps in its __init__."""
        sig = inspect.signature(trainer_cls.__init__)
        assert (
            "auto_log_optim_steps" in sig.parameters
        ), f"{trainer_cls.__name__}.__init__ must accept auto_log_optim_steps"
        assert sig.parameters["auto_log_optim_steps"].default is True

    @pytest.mark.parametrize(
        "trainer_cls",
        [
            A2CTrainer,
            CQLTrainer,
            DDPGTrainer,
            DQNTrainer,
            IQLTrainer,
            OfflineToOnlineTrainer,
            PPOTrainer,
            ReinforceTrainer,
            SACTrainer,
            TD3Trainer,
        ],
    )
    def test_subclass_exposes_checkpoint(self, trainer_cls):
        """Every algorithm trainer must expose the unified checkpoint argument."""
        parameter = inspect.signature(trainer_cls.__init__).parameters["checkpoint"]
        assert parameter.default is None


class TestDefaultOptimizationStepper:
    def test_loss_components_partial(self):
        torch.manual_seed(0)
        x = torch.randn(5, 10)
        model1 = nn.Linear(10, 20)
        model2 = nn.Linear(10, 20)
        all_params = list(model1.parameters()) + list(model2.parameters())
        optimizer = torch.optim.SGD(all_params, lr=1e-3)

        class TwoLossModule(nn.Module):
            def forward(self, td):
                return TensorDict(
                    {
                        "loss_actor": model1(x).sum(),
                        "loss_critic": model2(x).sum(),
                    },
                    [],
                )

        loss_module = TwoLossModule()
        stepper = DefaultOptimizationStepper(loss_components=["loss_actor"])

        trainer = Trainer(
            collector=MockingCollector(),
            total_frames=None,
            frame_skip=None,
            optim_steps_per_batch=None,
            loss_module=loss_module,
            optimizer=optimizer,
            optimization_stepper=stepper,
        )
        trainer._pbar_str = OrderedDict()

        model1_before = [p.clone() for p in model1.parameters()]
        model2_before = [p.clone() for p in model2.parameters()]

        td = TensorDict({"x": torch.randn(3)}, [])
        losses = stepper.step(trainer, td)

        assert "grad_norm" in losses.keys()
        assert all(
            not torch.equal(b, a) for b, a in zip(model1_before, model1.parameters())
        )
        assert all(
            torch.equal(b, a) for b, a in zip(model2_before, model2.parameters())
        )

    def test_gradient_sync_precedes_clipping_and_step(self):
        events = []
        model = nn.Linear(2, 1)

        class Loss(nn.Module):
            def forward(self, td):
                return TensorDict({"loss": model(td["x"]).sum()}, [])

        class RecordingOptimizer(torch.optim.SGD):
            def step(self, closure=None):
                events.append("step")
                return super().step(closure)

        class RecordingContext:
            def sync_gradients(self, optimizer):
                assert all(
                    parameter.grad is not None
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                )
                events.append("sync")

        optimizer = RecordingOptimizer(model.parameters(), lr=0.1)
        trainer = Trainer(
            collector=MockingCollector(),
            total_frames=None,
            frame_skip=None,
            optim_steps_per_batch=1,
            loss_module=Loss(),
            optimizer=optimizer,
            optimization_stepper=DefaultOptimizationStepper(),
            clip_norm=1.0,
        )
        trainer.data_parallel_context = RecordingContext()
        trainer.optimization_stepper.step(
            trainer, TensorDict({"x": torch.ones(3, 2)}, [3])
        )
        assert events == ["sync", "step"]


class _LearnerLoss(LossModule):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, batch):
        return TensorDict({"loss": self.weight * batch["x"].mean()}, [])


class TestLearnerGroup:
    @staticmethod
    def _make_group(replay_buffer):
        def learner_factory(context):
            loss = _LearnerLoss()
            return Learner(
                loss,
                context.replay_buffer,
                optimizer=torch.optim.SGD(loss.parameters(), lr=0.1),
                data_parallel_context=context.data_parallel_context,
            )

        return LocalLearnerGroup(learner_factory, replay_buffer, global_batch_size=4)

    def test_requires_loss_module(self):
        module = nn.Linear(1, 1)
        with pytest.raises(TypeError, match="torchrl.objectives.LossModule"):
            Learner(
                module,
                replay_buffer=None,
                optimizer=torch.optim.SGD(module.parameters(), lr=0.1),
            )

    def test_local_group_rounds_metrics_and_state(self):
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(16), batch_size=4
        )
        replay_buffer.extend(TensorDict({"x": torch.ones(16, 1)}, [16]))
        group = self._make_group(replay_buffer).start()

        result = group.step(LearnerStepRequest(1, 2, 4))
        assert result.round_id == 1
        assert result.optim_steps == 2
        assert result.model_version == 2
        assert result.metrics.device == torch.device("cpu")
        assert result.metrics["loss"].ndim == 0

        state = group.state_dict()
        weights = group.get_weights()
        assert weights.model_version == 2
        group.shutdown()
        group.shutdown()

        restored = self._make_group(replay_buffer).start()
        restored.load_state_dict(state)
        assert restored.step(LearnerStepRequest(2, 1, 4)).model_version == 3
        restored.shutdown()

    def test_rejects_nonconsecutive_round(self):
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(8), batch_size=4
        )
        replay_buffer.extend(TensorDict({"x": torch.ones(8, 1)}, [8]))
        group = self._make_group(replay_buffer).start()
        with pytest.raises(RuntimeError, match="Expected round_id=1"):
            group.step(LearnerStepRequest(2, 1, 4))
        group.shutdown()


class TestTD3OptimizationStepper:
    def test_synchronizes_each_optimizer_before_step(self):
        events = []
        critic = nn.Parameter(torch.ones(()))
        actor = nn.Parameter(torch.ones(()))

        class RecordingOptimizer(torch.optim.SGD):
            def __init__(self, name, params):
                self.name = name
                super().__init__(params, lr=0.1)

            def step(self, closure=None):
                events.append(f"step_{self.name}")
                return super().step(closure)

        critic_optim = RecordingOptimizer("critic", [critic])
        actor_optim = RecordingOptimizer("actor", [actor])
        stepper = TD3OptimizationStepper(
            actor_optim, critic_optim, policy_update_delay=1
        )

        class Loss:
            class tensor_keys:
                priority = "td_error"

            def value_loss(self, batch):
                return critic.square(), {"td_error": torch.ones(1, 1)}

            def actor_loss(self, batch):
                return actor.square(), None

        class Context:
            loss_module = Loss()
            clip_grad_norm = True
            clip_norm = 1.0
            replay_buffer = None

            def sync_gradients(self, optimizer):
                events.append(f"sync_{optimizer.name}")

        stepper.step(Context(), TensorDict())
        assert events == [
            "sync_critic",
            "step_critic",
            "sync_actor",
            "step_actor",
        ]


class TestLRSchedulerHook:
    def test_scheduler_steps_on_call(self):
        net = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        hook = LRSchedulerHook(scheduler)

        hook()
        assert optimizer.param_groups[0]["lr"] == 0.5
        hook()
        assert optimizer.param_groups[0]["lr"] == 0.25

    def test_invalid_interval_raises(self):
        net = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        with pytest.raises(ValueError, match="interval must be"):
            LRSchedulerHook(scheduler, interval="epoch")

    def test_state_dict_roundtrip(self):
        net = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        hook = LRSchedulerHook(scheduler)
        hook()
        hook()
        state = hook.state_dict()

        net2 = torch.nn.Linear(2, 2)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=1.0)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.5)
        hook2 = LRSchedulerHook(scheduler2)
        hook2.load_state_dict(state)

        assert scheduler2.state_dict() == scheduler.state_dict()

    @pytest.mark.parametrize("interval", ["batch", "optim"])
    def test_register(self, interval):
        trainer = mocking_trainer()
        net = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        hook = LRSchedulerHook(scheduler, interval=interval)
        hook.register(trainer)

        dest = "post_optim" if interval == "optim" else "post_steps"
        ops = getattr(trainer, f"_{dest}_ops")
        assert any(getattr(op, "__wrapped__", None) is hook for op, _ in ops)
        assert trainer._modules["lr_scheduler"] is hook

    def test_no_step_without_optimization(self):
        # once registered, the hook must not decay the learning rate while no
        # optimization step has run (e.g. during init_random_frames warmup)
        trainer = mocking_trainer()
        net = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        hook = LRSchedulerHook(scheduler)
        hook.register(trainer)

        hook()
        assert optimizer.param_groups[0]["lr"] == 1.0

        trainer._optim_count += 1
        hook()
        assert optimizer.param_groups[0]["lr"] == 0.5

        # no new optimization step: the scheduler must not advance
        hook()
        assert optimizer.param_groups[0]["lr"] == 0.5


class TestValueEstimatorHook:
    class _RecordingEstimator:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            batch.set("advantage", torch.ones(batch.shape))
            return batch

    def test_estimator_applied_to_batch(self):
        estimator = self._RecordingEstimator()
        hook = ValueEstimatorHook(estimator)
        batch = TensorDict({"observation": torch.randn(5, 3)}, [5])
        out = hook(batch)
        assert estimator.calls == 1
        assert "advantage" in out.keys()

    def test_none_batch_passthrough(self):
        # async collection hands the pre_epoch stage a None batch: the hook
        # must pass it through without calling the estimator
        estimator = self._RecordingEstimator()
        hook = ValueEstimatorHook(estimator)
        assert hook(None) is None
        assert estimator.calls == 0

    def test_state_dict_plain_callable(self):
        hook = ValueEstimatorHook(self._RecordingEstimator())
        assert hook.state_dict() == {}
        hook.load_state_dict({})

    def test_state_dict_module_delegation(self):
        module = torch.nn.Linear(3, 1)
        hook = ValueEstimatorHook(module)
        state = hook.state_dict()
        assert "value_estimator" in state

        module2 = torch.nn.Linear(3, 1)
        hook2 = ValueEstimatorHook(module2)
        hook2.load_state_dict(state)
        assert torch.equal(module2.weight, module.weight)

    def test_register(self):
        trainer = mocking_trainer()
        hook = ValueEstimatorHook(self._RecordingEstimator())
        hook.register(trainer)
        ops = trainer._pre_epoch_ops
        assert any(getattr(op, "__wrapped__", None) is hook for op, _ in ops)
        assert trainer._modules["value_estimator"] is hook


class TestProcessLossHook:
    @pytest.mark.parametrize("factor", [0.5, 2.0, 10.0])
    def test_scale_loss(self, factor):
        trainer = mocking_trainer()
        td_loss = TensorDict({"loss_a": torch.tensor(1.0)}, [])
        td_sub_batch = TensorDict({"sub_batch": torch.tensor(1.0)}, [])

        class ScaleLoss:
            def __init__(self, scale):
                self.scale = scale

            def __call__(self, sub_batch, losses):
                return losses.apply(lambda t: t * self.scale)

        scale_hook = ScaleLoss(factor)
        trainer.register_op("process_loss", scale_hook)
        td_out = trainer._process_loss_hook(td_sub_batch.clone(), td_loss.clone())
        assert torch.allclose(td_out["loss_a"], torch.tensor(factor))

    def test_chained_hooks(self):
        """Test that multiple process_loss hooks are applied in order."""
        trainer = mocking_trainer()
        td_loss = TensorDict({"loss_a": torch.tensor(2.0)}, [])
        td_sub_batch = TensorDict()

        call_order = []

        class AddOne:
            def __call__(self, sub_batch, losses):
                call_order.append("add")
                losses = losses.clone()
                losses["loss_a"] = losses["loss_a"] + 1
                return losses

        class MultiplyTwo:
            def __call__(self, sub_batch, losses):
                call_order.append("mul")
                losses = losses.clone()
                losses["loss_a"] = losses["loss_a"] * 2
                return losses

        trainer.register_op("process_loss", AddOne())
        trainer.register_op("process_loss", MultiplyTwo())

        td_out = trainer._process_loss_hook(td_sub_batch, td_loss.clone())
        # (2 + 1) * 2 = 6
        assert torch.allclose(td_out["loss_a"], torch.tensor(6.0))
        assert call_order == ["add", "mul"]

    def test_hook_receives_sub_batch(self):
        """Test that the hook can use information from the sub_batch."""
        trainer = mocking_trainer()
        td_loss = TensorDict({"loss_a": torch.tensor(1.0)}, [])
        td_sub_batch = TensorDict({"importance_weight": torch.tensor(0.5)}, [])

        class WeightedLoss:
            def __call__(self, sub_batch, losses):
                weight = sub_batch.get("importance_weight")
                losses = losses.clone()
                losses["loss_a"] = losses["loss_a"] * weight
                return losses

        trainer.register_op("process_loss", WeightedLoss())
        td_out = trainer._process_loss_hook(td_sub_batch, td_loss.clone())
        assert torch.allclose(td_out["loss_a"], torch.tensor(0.5))


class TestSetupShutdownHooks:
    @staticmethod
    def _make_trainer(collector):
        trainer = Trainer(
            collector=collector,
            total_frames=2,
            frame_skip=1,
            optim_steps_per_batch=1,
            loss_module=MockingLossModule(),
            optimizer=None,
            progress_bar=False,
        )
        trainer._pbar_str = OrderedDict()
        return trainer

    def test_setup_and_shutdown_hooks_order(self):
        batches = [
            TensorDict({"x": torch.zeros(1)}, [1]),
            TensorDict({"x": torch.ones(1)}, [1]),
        ]
        collector = MockingIterableCollector(batches)
        trainer = self._make_trainer(collector)

        call_order = []

        def setup_hook():
            call_order.append("setup")

        def pre_steps_hook(_batch):
            call_order.append("pre_steps")

        def shutdown_hook():
            call_order.append("shutdown")

        trainer.register_op("setup", setup_hook)
        trainer.register_op("pre_steps_log", pre_steps_hook)
        trainer.register_op("shutdown", shutdown_hook)

        trainer.train()

        assert call_order == ["setup", "pre_steps", "pre_steps", "shutdown"]
        assert collector.shutdown_calls == 1


class TestEarlyStopping:
    @pytest.mark.parametrize(
        "monitor,values,hook_kwargs,expected_stops,expected_reason",
        [
            (
                "r_training",
                [1.0, 1.1, 1.09, 1.08],
                {"patience": 2, "wait_for": 0},
                [False, False, False, True],
                "did not improve",
            ),
            (
                "score",
                [1.0, 0.0, -1.0, -2.0],
                {"patience": 1, "wait_for": 3},
                [False, False, False, True],
                "did not improve",
            ),
            (
                "validation_loss",
                [1.0, 0.85, 0.8],
                {"mode": "min", "min_delta": 0.1, "patience": 1, "wait_for": 0},
                [False, False, True],
                "did not improve",
            ),
            (
                "score",
                [1.0, 1.2, 1.25],
                {"mode": "max", "min_delta": 0.1, "patience": 1, "wait_for": 0},
                [False, False, True],
                "did not improve",
            ),
            (
                "score",
                [1.0, float("nan")],
                {"wait_for": 0, "check_finite": True, "patience": 10},
                [False, True],
                "non-finite",
            ),
        ],
    )
    def test_early_stopping_params(
        self, monitor, values, hook_kwargs, expected_stops, expected_reason
    ):
        trainer = mocking_trainer(optimizer=None)
        early_stopping = EarlyStopping(
            monitor=monitor,
            **hook_kwargs,
        )
        early_stopping.register(trainer)

        td = TensorDict()
        for value, should_stop in zip(values, expected_stops):
            trainer._log(**{monitor: value})
            trainer.collected_frames += 1
            trainer._post_steps_log_hook(td)
            assert trainer._stop_training == should_stop

        assert trainer._stop_training
        assert expected_reason in early_stopping.stop_reason

    def test_missing_monitor_raises(self):
        trainer = mocking_trainer(optimizer=None)
        early_stopping = EarlyStopping(
            monitor="missing",
        )
        early_stopping.register(trainer)

        with pytest.raises(RuntimeError, match="could not find monitored metric"):
            trainer._post_steps_log_hook(TensorDict())

    def test_train_stops_early(self):
        rewards = [1.0, 2.0, 2.0, 2.0, 2.0]
        batches = [
            TensorDict({("next", "reward"): torch.tensor([reward])}, [1])
            for reward in rewards
        ]
        collector = MockingIterableCollector(batches=batches)
        trainer = Trainer(
            collector=collector,
            total_frames=100,
            frame_skip=1,
            optim_steps_per_batch=1,
            loss_module=MockingLossModule(),
            optimizer=None,
            progress_bar=False,
        )

        trainer.register_op("pre_steps_log", LogScalar(REWARD_KEY, "r_training"))
        EarlyStopping(monitor="r_training", patience=1, wait_for=0).register(trainer)

        trainer.train()
        assert trainer.collected_frames < trainer.total_frames
        assert trainer._stop_training
        assert collector.shutdown_calls == 1


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
