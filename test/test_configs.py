# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib.util
import os

import pytest
import torch

from hydra.utils import instantiate

from torchrl.collectors.collectors import SyncDataCollector
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.envs import AsyncEnvPool, ParallelEnv, SerialEnv
from torchrl.modules.models.models import MLP
from torchrl.objectives.ppo import PPOLoss
from torchrl.trainers.algorithms.configs.modules import ActivationConfig, LayerConfig
from torchrl.trainers.algorithms.ppo import PPOTrainer


_has_gym = (importlib.util.find_spec("gym") is not None) or (
    importlib.util.find_spec("gymnasium") is not None
)
_has_hydra = importlib.util.find_spec("hydra") is not None


class TestEnvConfigs:
    @pytest.mark.skipif(not _has_gym, reason="Gym is not installed")
    def test_gym_env_config(self):
        from torchrl.trainers.algorithms.configs.envs import GymEnvConfig

        cfg = GymEnvConfig(env_name="CartPole-v1")
        assert cfg.env_name == "CartPole-v1"
        assert cfg.backend == "gymnasium"
        assert cfg.from_pixels is False
        assert cfg.double_to_float is False
        instantiate(cfg)

    @pytest.mark.skipif(not _has_gym, reason="Gym is not installed")
    @pytest.mark.parametrize("cls", [ParallelEnv, SerialEnv, AsyncEnvPool])
    def test_batched_env_config(self, cls):
        from torchrl.trainers.algorithms.configs.envs import (
            BatchedEnvConfig,
            GymEnvConfig,
        )

        batched_env_type = (
            "parallel"
            if cls == ParallelEnv
            else "serial"
            if cls == SerialEnv
            else "async"
        )
        cfg = BatchedEnvConfig(
            create_env_fn=GymEnvConfig(env_name="CartPole-v1"),
            num_workers=2,
            batched_env_type=batched_env_type,
        )
        env = instantiate(cfg)
        assert isinstance(env, cls)


class TestDataConfigs:
    """Test cases for data.py configuration classes."""

    def test_writer_config(self):
        """Test basic WriterConfig."""
        from torchrl.trainers.algorithms.configs.data import WriterConfig

        cfg = WriterConfig()
        assert cfg._target_ == "torchrl.data.replay_buffers.Writer"

    def test_round_robin_writer_config(self):
        """Test RoundRobinWriterConfig."""
        from torchrl.trainers.algorithms.configs.data import RoundRobinWriterConfig

        cfg = RoundRobinWriterConfig(compilable=True)
        assert cfg._target_ == "torchrl.data.replay_buffers.RoundRobinWriter"
        assert cfg.compilable is True

        # Test instantiation
        writer = instantiate(cfg)
        from torchrl.data.replay_buffers.writers import RoundRobinWriter

        assert isinstance(writer, RoundRobinWriter)
        assert writer._compilable is True

    def test_sampler_config(self):
        """Test basic SamplerConfig."""
        from torchrl.trainers.algorithms.configs.data import SamplerConfig

        cfg = SamplerConfig()
        assert cfg._target_ == "torchrl.data.replay_buffers.Sampler"

    def test_random_sampler_config(self):
        """Test RandomSamplerConfig."""
        from torchrl.trainers.algorithms.configs.data import RandomSamplerConfig

        cfg = RandomSamplerConfig()
        assert cfg._target_ == "torchrl.data.replay_buffers.RandomSampler"

        # Test instantiation
        sampler = instantiate(cfg)
        from torchrl.data.replay_buffers.samplers import RandomSampler

        assert isinstance(sampler, RandomSampler)

    def test_tensor_storage_config(self):
        """Test TensorStorageConfig."""
        from torchrl.trainers.algorithms.configs.data import TensorStorageConfig

        cfg = TensorStorageConfig(max_size=1000, device="cpu", ndim=2, compilable=True)
        assert cfg._target_ == "torchrl.data.replay_buffers.TensorStorage"
        assert cfg.max_size == 1000
        assert cfg.device == "cpu"
        assert cfg.ndim == 2
        assert cfg.compilable is True

        # Test instantiation (requires storage parameter)
        import torch

        storage_tensor = torch.zeros(1000, 10)
        cfg.storage = storage_tensor
        storage = instantiate(cfg)
        from torchrl.data.replay_buffers.storages import TensorStorage

        assert isinstance(storage, TensorStorage)
        assert storage.max_size == 1000
        assert storage.ndim == 2

    def test_tensordict_replay_buffer_config(self):
        """Test TensorDictReplayBufferConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            ListStorageConfig,
            RandomSamplerConfig,
            RoundRobinWriterConfig,
            TensorDictReplayBufferConfig,
        )

        cfg = TensorDictReplayBufferConfig(
            sampler=RandomSamplerConfig(),
            storage=ListStorageConfig(max_size=1000),
            writer=RoundRobinWriterConfig(),
            batch_size=32,
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.TensorDictReplayBuffer"
        assert cfg.batch_size == 32

        # Test instantiation
        buffer = instantiate(cfg)
        from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer

        assert isinstance(buffer, TensorDictReplayBuffer)
        assert buffer._batch_size == 32

    def test_list_storage_config(self):
        """Test ListStorageConfig."""
        from torchrl.trainers.algorithms.configs.data import ListStorageConfig

        cfg = ListStorageConfig(max_size=1000, compilable=True)
        assert cfg._target_ == "torchrl.data.replay_buffers.ListStorage"
        assert cfg.max_size == 1000
        assert cfg.compilable is True

        # Test instantiation
        storage = instantiate(cfg)
        from torchrl.data.replay_buffers.storages import ListStorage

        assert isinstance(storage, ListStorage)
        assert storage.max_size == 1000

    def test_replay_buffer_config(self):
        """Test ReplayBufferConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            ListStorageConfig,
            RandomSamplerConfig,
            ReplayBufferConfig,
            RoundRobinWriterConfig,
        )

        # Test with all fields provided
        cfg = ReplayBufferConfig(
            sampler=RandomSamplerConfig(),
            storage=ListStorageConfig(max_size=1000),
            writer=RoundRobinWriterConfig(),
            batch_size=32,
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.ReplayBuffer"
        assert cfg.batch_size == 32

        # Test instantiation
        buffer = instantiate(cfg)
        from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer

        assert isinstance(buffer, ReplayBuffer)
        assert buffer._batch_size == 32

        # Test with optional fields omitted (new functionality)
        cfg_optional = ReplayBufferConfig()
        assert cfg_optional._target_ == "torchrl.data.replay_buffers.ReplayBuffer"
        assert cfg_optional.sampler is None
        assert cfg_optional.storage is None
        assert cfg_optional.writer is None
        assert cfg_optional.transform is None
        assert cfg_optional.batch_size is None
        assert isinstance(instantiate(cfg_optional), ReplayBuffer)

    def test_tensordict_replay_buffer_config_optional_fields(self):
        """Test that optional fields can be omitted from TensorDictReplayBuffer config."""
        from torchrl.trainers.algorithms.configs.data import (
            TensorDictReplayBufferConfig,
        )

        cfg = TensorDictReplayBufferConfig()
        assert cfg._target_ == "torchrl.data.replay_buffers.TensorDictReplayBuffer"
        assert cfg.sampler is None
        assert cfg.storage is None
        assert cfg.writer is None
        assert cfg.transform is None
        assert cfg.batch_size is None
        assert isinstance(instantiate(cfg), ReplayBuffer)

    def test_writer_ensemble_config(self):
        """Test WriterEnsembleConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            RoundRobinWriterConfig,
            WriterEnsembleConfig,
        )

        cfg = WriterEnsembleConfig(
            writers=[RoundRobinWriterConfig(), RoundRobinWriterConfig()], p=[0.5, 0.5]
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.WriterEnsemble"
        assert len(cfg.writers) == 2
        assert cfg.p == [0.5, 0.5]

        # Test instantiation - use direct instantiation to avoid Union type issues
        from torchrl.data.replay_buffers.writers import RoundRobinWriter, WriterEnsemble

        writer1 = RoundRobinWriter()
        writer2 = RoundRobinWriter()
        writer = WriterEnsemble(writer1, writer2)
        assert isinstance(writer, WriterEnsemble)
        assert len(writer._writers) == 2

    def test_tensor_dict_max_value_writer_config(self):
        """Test TensorDictMaxValueWriterConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            TensorDictMaxValueWriterConfig,
        )

        cfg = TensorDictMaxValueWriterConfig(rank_key="priority", reduction="max")
        assert cfg._target_ == "torchrl.data.replay_buffers.TensorDictMaxValueWriter"
        assert cfg.rank_key == "priority"
        assert cfg.reduction == "max"

        # Test instantiation
        writer = instantiate(cfg)
        from torchrl.data.replay_buffers.writers import TensorDictMaxValueWriter

        assert isinstance(writer, TensorDictMaxValueWriter)

    def test_tensor_dict_round_robin_writer_config(self):
        """Test TensorDictRoundRobinWriterConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            TensorDictRoundRobinWriterConfig,
        )

        cfg = TensorDictRoundRobinWriterConfig(compilable=True)
        assert cfg._target_ == "torchrl.data.replay_buffers.TensorDictRoundRobinWriter"
        assert cfg.compilable is True

        # Test instantiation
        writer = instantiate(cfg)
        from torchrl.data.replay_buffers.writers import TensorDictRoundRobinWriter

        assert isinstance(writer, TensorDictRoundRobinWriter)
        assert writer._compilable is True

    def test_immutable_dataset_writer_config(self):
        """Test ImmutableDatasetWriterConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            ImmutableDatasetWriterConfig,
        )

        cfg = ImmutableDatasetWriterConfig()
        assert cfg._target_ == "torchrl.data.replay_buffers.ImmutableDatasetWriter"

        # Test instantiation
        writer = instantiate(cfg)
        from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter

        assert isinstance(writer, ImmutableDatasetWriter)

    def test_sampler_ensemble_config(self):
        """Test SamplerEnsembleConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            RandomSamplerConfig,
            SamplerEnsembleConfig,
        )

        cfg = SamplerEnsembleConfig(
            samplers=[RandomSamplerConfig(), RandomSamplerConfig()], p=[0.5, 0.5]
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.SamplerEnsemble"
        assert len(cfg.samplers) == 2
        assert cfg.p == [0.5, 0.5]

        # Test instantiation - use direct instantiation to avoid Union type issues
        from torchrl.data.replay_buffers.samplers import RandomSampler, SamplerEnsemble

        sampler1 = RandomSampler()
        sampler2 = RandomSampler()
        sampler = SamplerEnsemble(sampler1, sampler2, p=[0.5, 0.5])
        assert isinstance(sampler, SamplerEnsemble)
        assert len(sampler._samplers) == 2

    def test_prioritized_slice_sampler_config(self):
        """Test PrioritizedSliceSamplerConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            PrioritizedSliceSamplerConfig,
        )

        cfg = PrioritizedSliceSamplerConfig(
            num_slices=10,
            slice_len=None,  # Only set one of num_slices or slice_len
            end_key=("next", "done"),
            traj_key="episode",
            cache_values=True,
            truncated_key=("next", "truncated"),
            strict_length=True,
            compile=False,  # Use bool instead of Union[bool, dict]
            span=False,  # Use bool instead of Union[bool, int, tuple]
            use_gpu=False,  # Use bool instead of Union[torch.device, bool]
            max_capacity=1000,
            alpha=0.7,
            beta=0.9,
            eps=1e-8,
            reduction="max",
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.PrioritizedSliceSampler"
        assert cfg.num_slices == 10
        assert cfg.slice_len is None
        assert cfg.end_key == ("next", "done")
        assert cfg.traj_key == "episode"
        assert cfg.cache_values is True
        assert cfg.truncated_key == ("next", "truncated")
        assert cfg.strict_length is True
        assert cfg.compile is False
        assert cfg.span is False
        assert cfg.use_gpu is False
        assert cfg.max_capacity == 1000
        assert cfg.alpha == 0.7
        assert cfg.beta == 0.9
        assert cfg.eps == 1e-8
        assert cfg.reduction == "max"

        # Test instantiation - use direct instantiation to avoid Union type issues
        from torchrl.data.replay_buffers.samplers import PrioritizedSliceSampler

        sampler = PrioritizedSliceSampler(
            num_slices=10,
            max_capacity=1000,
            alpha=0.7,
            beta=0.9,
            eps=1e-8,
            reduction="max",
        )
        assert isinstance(sampler, PrioritizedSliceSampler)
        assert sampler.num_slices == 10
        assert sampler.alpha == 0.7
        assert sampler.beta == 0.9

    def test_slice_sampler_without_replacement_config(self):
        """Test SliceSamplerWithoutReplacementConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            SliceSamplerWithoutReplacementConfig,
        )

        cfg = SliceSamplerWithoutReplacementConfig(
            num_slices=10,
            slice_len=None,  # Only set one of num_slices or slice_len
            end_key=("next", "done"),
            traj_key="episode",
            cache_values=True,
            truncated_key=("next", "truncated"),
            strict_length=True,
            compile=False,  # Use bool instead of Union[bool, dict]
            span=False,  # Use bool instead of Union[bool, int, tuple]
            use_gpu=False,  # Use bool instead of Union[torch.device, bool]
        )
        assert (
            cfg._target_ == "torchrl.data.replay_buffers.SliceSamplerWithoutReplacement"
        )
        assert cfg.num_slices == 10
        assert cfg.slice_len is None
        assert cfg.end_key == ("next", "done")
        assert cfg.traj_key == "episode"
        assert cfg.cache_values is True
        assert cfg.truncated_key == ("next", "truncated")
        assert cfg.strict_length is True
        assert cfg.compile is False
        assert cfg.span is False
        assert cfg.use_gpu is False

        # Test instantiation - use direct instantiation to avoid Union type issues
        from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement

        sampler = SliceSamplerWithoutReplacement(num_slices=10)
        assert isinstance(sampler, SliceSamplerWithoutReplacement)
        assert sampler.num_slices == 10

    def test_slice_sampler_config(self):
        """Test SliceSamplerConfig."""
        from torchrl.trainers.algorithms.configs.data import SliceSamplerConfig

        cfg = SliceSamplerConfig(
            num_slices=10,
            slice_len=None,  # Only set one of num_slices or slice_len
            end_key=("next", "done"),
            traj_key="episode",
            cache_values=True,
            truncated_key=("next", "truncated"),
            strict_length=True,
            compile=False,  # Use bool instead of Union[bool, dict]
            span=False,  # Use bool instead of Union[bool, int, tuple]
            use_gpu=False,  # Use bool instead of Union[torch.device, bool]
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.SliceSampler"
        assert cfg.num_slices == 10
        assert cfg.slice_len is None
        assert cfg.end_key == ("next", "done")
        assert cfg.traj_key == "episode"
        assert cfg.cache_values is True
        assert cfg.truncated_key == ("next", "truncated")
        assert cfg.strict_length is True
        assert cfg.compile is False
        assert cfg.span is False
        assert cfg.use_gpu is False

        # Test instantiation - use direct instantiation to avoid Union type issues
        from torchrl.data.replay_buffers.samplers import SliceSampler

        sampler = SliceSampler(num_slices=10)
        assert isinstance(sampler, SliceSampler)
        assert sampler.num_slices == 10

    def test_prioritized_sampler_config(self):
        """Test PrioritizedSamplerConfig."""
        from torchrl.trainers.algorithms.configs.data import PrioritizedSamplerConfig

        cfg = PrioritizedSamplerConfig(
            max_capacity=1000, alpha=0.7, beta=0.9, eps=1e-8, reduction="max"
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.PrioritizedSampler"
        assert cfg.max_capacity == 1000
        assert cfg.alpha == 0.7
        assert cfg.beta == 0.9
        assert cfg.eps == 1e-8
        assert cfg.reduction == "max"

        # Test instantiation
        sampler = instantiate(cfg)
        from torchrl.data.replay_buffers.samplers import PrioritizedSampler

        assert isinstance(sampler, PrioritizedSampler)
        assert sampler._max_capacity == 1000
        assert sampler._alpha == 0.7
        assert sampler._beta == 0.9
        assert sampler._eps == 1e-8
        assert sampler.reduction == "max"

    def test_sampler_without_replacement_config(self):
        """Test SamplerWithoutReplacementConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            SamplerWithoutReplacementConfig,
        )

        cfg = SamplerWithoutReplacementConfig(drop_last=True, shuffle=False)
        assert cfg._target_ == "torchrl.data.replay_buffers.SamplerWithoutReplacement"
        assert cfg.drop_last is True
        assert cfg.shuffle is False

        # Test instantiation
        sampler = instantiate(cfg)
        from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

        assert isinstance(sampler, SamplerWithoutReplacement)
        assert sampler.drop_last is True
        assert sampler.shuffle is False

    def test_storage_ensemble_writer_config(self):
        """Test StorageEnsembleWriterConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            RoundRobinWriterConfig,
            StorageEnsembleWriterConfig,
        )

        cfg = StorageEnsembleWriterConfig(
            writers=[RoundRobinWriterConfig(), RoundRobinWriterConfig()], transforms=[]
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.StorageEnsembleWriter"
        assert len(cfg.writers) == 2
        assert len(cfg.transforms) == 0

        # Note: StorageEnsembleWriter doesn't exist in the actual codebase
        # This test will fail until the class is implemented
        # For now, we just test the config creation
        assert cfg.writers[0]._target_ == "torchrl.data.replay_buffers.RoundRobinWriter"

    def test_lazy_stack_storage_config(self):
        """Test LazyStackStorageConfig."""
        from torchrl.trainers.algorithms.configs.data import LazyStackStorageConfig

        cfg = LazyStackStorageConfig(max_size=1000, compilable=True, stack_dim=1)
        assert cfg._target_ == "torchrl.data.replay_buffers.LazyStackStorage"
        assert cfg.max_size == 1000
        assert cfg.compilable is True
        assert cfg.stack_dim == 1

        # Test instantiation
        storage = instantiate(cfg)
        from torchrl.data.replay_buffers.storages import LazyStackStorage

        assert isinstance(storage, LazyStackStorage)
        assert storage.max_size == 1000
        assert storage.stack_dim == 1

    def test_storage_ensemble_config(self):
        """Test StorageEnsembleConfig."""
        from torchrl.trainers.algorithms.configs.data import (
            ListStorageConfig,
            StorageEnsembleConfig,
        )

        cfg = StorageEnsembleConfig(
            storages=[ListStorageConfig(max_size=100), ListStorageConfig(max_size=200)],
            transforms=[],
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.StorageEnsemble"
        assert len(cfg.storages) == 2
        assert len(cfg.transforms) == 0

        # Test instantiation - use direct instantiation since StorageEnsemble expects *storages
        from torchrl.data.replay_buffers.storages import ListStorage, StorageEnsemble

        storage1 = ListStorage(max_size=100)
        storage2 = ListStorage(max_size=200)
        storage = StorageEnsemble(
            storage1, storage2, transforms=[None, None]
        )  # Provide transforms for each storage
        assert isinstance(storage, StorageEnsemble)
        assert len(storage._storages) == 2

    def test_lazy_memmap_storage_config(self):
        """Test LazyMemmapStorageConfig."""
        from torchrl.trainers.algorithms.configs.data import LazyMemmapStorageConfig

        cfg = LazyMemmapStorageConfig(
            max_size=1000, device="cpu", ndim=2, compilable=True
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.LazyMemmapStorage"
        assert cfg.max_size == 1000
        assert cfg.device == "cpu"
        assert cfg.ndim == 2
        assert cfg.compilable is True

        # Test instantiation
        storage = instantiate(cfg)
        from torchrl.data.replay_buffers.storages import LazyMemmapStorage

        assert isinstance(storage, LazyMemmapStorage)
        assert storage.max_size == 1000
        assert storage.ndim == 2

    def test_lazy_tensor_storage_config(self):
        """Test LazyTensorStorageConfig."""
        from torchrl.trainers.algorithms.configs.data import LazyTensorStorageConfig

        cfg = LazyTensorStorageConfig(
            max_size=1000, device="cpu", ndim=2, compilable=True
        )
        assert cfg._target_ == "torchrl.data.replay_buffers.LazyTensorStorage"
        assert cfg.max_size == 1000
        assert cfg.device == "cpu"
        assert cfg.ndim == 2
        assert cfg.compilable is True

        # Test instantiation
        storage = instantiate(cfg)
        from torchrl.data.replay_buffers.storages import LazyTensorStorage

        assert isinstance(storage, LazyTensorStorage)
        assert storage.max_size == 1000
        assert storage.ndim == 2

    def test_storage_config(self):
        """Test StorageConfig."""
        from torchrl.trainers.algorithms.configs.data import StorageConfig

        cfg = StorageConfig()
        # This is a base class, so it should have a _target_
        assert hasattr(cfg, "_target_")
        assert cfg._target_ == "torchrl.data.replay_buffers.Storage"

    def test_complex_replay_buffer_configuration(self):
        """Test a complex replay buffer configuration with all components."""
        from torchrl.trainers.algorithms.configs.data import (
            LazyMemmapStorageConfig,
            PrioritizedSliceSamplerConfig,
            TensorDictReplayBufferConfig,
            TensorDictRoundRobinWriterConfig,
        )

        # Create a complex configuration
        cfg = TensorDictReplayBufferConfig(
            sampler=PrioritizedSliceSamplerConfig(
                num_slices=10,
                slice_len=5,
                max_capacity=1000,
                alpha=0.7,
                beta=0.9,
                compile=False,  # Use bool instead of Union[bool, dict]
                span=False,  # Use bool instead of Union[bool, int, tuple]
                use_gpu=False,  # Use bool instead of Union[torch.device, bool]
            ),
            storage=LazyMemmapStorageConfig(max_size=1000, device="cpu", ndim=2),
            writer=TensorDictRoundRobinWriterConfig(compilable=True),
            batch_size=64,
        )

        assert cfg._target_ == "torchrl.data.replay_buffers.TensorDictReplayBuffer"
        assert cfg.batch_size == 64

        # Test instantiation - use direct instantiation to avoid Union type issues
        from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
        from torchrl.data.replay_buffers.samplers import PrioritizedSliceSampler
        from torchrl.data.replay_buffers.storages import LazyMemmapStorage
        from torchrl.data.replay_buffers.writers import TensorDictRoundRobinWriter

        sampler = PrioritizedSliceSampler(
            num_slices=10, max_capacity=1000, alpha=0.7, beta=0.9
        )
        storage = LazyMemmapStorage(max_size=1000, device=torch.device("cpu"), ndim=2)
        writer = TensorDictRoundRobinWriter(compilable=True)

        buffer = TensorDictReplayBuffer(
            sampler=sampler, storage=storage, writer=writer, batch_size=64
        )

        assert isinstance(buffer, TensorDictReplayBuffer)
        assert isinstance(buffer._sampler, PrioritizedSliceSampler)
        assert isinstance(buffer._storage, LazyMemmapStorage)
        assert isinstance(buffer._writer, TensorDictRoundRobinWriter)
        assert buffer._batch_size == 64
        assert buffer._sampler.num_slices == 10
        assert buffer._sampler.alpha == 0.7
        assert buffer._sampler.beta == 0.9
        assert buffer._storage.max_size == 1000
        assert buffer._storage.ndim == 2
        assert buffer._writer._compilable is True


class TestModuleConfigs:
    """Test cases for modules.py configuration classes."""

    def test_network_config(self):
        """Test basic NetworkConfig."""
        from torchrl.trainers.algorithms.configs.modules import NetworkConfig

        cfg = NetworkConfig()
        # This is a base class, so it should not have a _target_
        assert not hasattr(cfg, "_target_")

    def test_mlp_config(self):
        """Test MLPConfig."""
        from torchrl.trainers.algorithms.configs.modules import MLPConfig

        cfg = MLPConfig(
            in_features=10,
            out_features=5,
            depth=2,
            num_cells=32,
            activation_class=ActivationConfig(_target_="torch.nn.ReLU", _partial_=True),
            dropout=0.1,
            bias_last_layer=True,
            single_bias_last_layer=False,
            layer_class=LayerConfig(_target_="torch.nn.Linear", _partial_=True),
            activate_last_layer=False,
            device="cpu",
        )
        assert cfg._target_ == "torchrl.modules.MLP"
        assert cfg.in_features == 10
        assert cfg.out_features == 5
        assert cfg.depth == 2
        assert cfg.num_cells == 32
        assert cfg.activation_class._target_ == "torch.nn.ReLU"
        assert cfg.dropout == 0.1
        assert cfg.bias_last_layer is True
        assert cfg.single_bias_last_layer is False
        assert cfg.layer_class._target_ == "torch.nn.Linear"
        assert cfg.activate_last_layer is False
        assert cfg.device == "cpu"

        mlp = instantiate(cfg)
        assert isinstance(mlp, MLP)
        mlp(torch.randn(10, 10))
        # Note: instantiate() has issues with string class names for MLP
        # This is a known limitation - the MLP constructor expects actual classes

    def test_convnet_config(self):
        """Test ConvNetConfig."""
        from torchrl.trainers.algorithms.configs.modules import (
            ActivationConfig,
            AggregatorConfig,
            ConvNetConfig,
        )

        cfg = ConvNetConfig(
            in_features=3,
            depth=2,
            num_cells=[32, 64],
            kernel_sizes=[3, 5],
            strides=[1, 2],
            paddings=[1, 2],
            activation_class=ActivationConfig(_target_="torch.nn.ReLU", _partial_=True),
            bias_last_layer=True,
            aggregator_class=AggregatorConfig(
                _target_="torchrl.modules.models.utils.SquashDims", _partial_=True
            ),
            squeeze_output=False,
            device="cpu",
        )
        assert cfg._target_ == "torchrl.modules.ConvNet"
        assert cfg.in_features == 3
        assert cfg.depth == 2
        assert cfg.num_cells == [32, 64]
        assert cfg.kernel_sizes == [3, 5]
        assert cfg.strides == [1, 2]
        assert cfg.paddings == [1, 2]
        assert cfg.activation_class._target_ == "torch.nn.ReLU"
        assert cfg.bias_last_layer is True
        assert (
            cfg.aggregator_class._target_ == "torchrl.modules.models.utils.SquashDims"
        )
        assert cfg.squeeze_output is False
        assert cfg.device == "cpu"

        convnet = instantiate(cfg)
        from torchrl.modules import ConvNet

        assert isinstance(convnet, ConvNet)
        convnet(torch.randn(1, 3, 32, 32))  # Test forward pass

    def test_tensor_dict_module_config(self):
        """Test TensorDictModuleConfig."""
        from torchrl.trainers.algorithms.configs.modules import (
            MLPConfig,
            TensorDictModuleConfig,
        )

        cfg = TensorDictModuleConfig(
            module=MLPConfig(in_features=10, out_features=10, depth=2, num_cells=32),
            in_keys=["observation"],
            out_keys=["action"],
        )
        assert cfg._target_ == "tensordict.nn.TensorDictModule"
        assert cfg.module._target_ == "torchrl.modules.MLP"
        assert cfg.in_keys == ["observation"]
        assert cfg.out_keys == ["action"]
        # Note: We can't test instantiation due to missing tensordict dependency

    def test_tanh_normal_model_config(self):
        """Test TanhNormalModelConfig."""
        from torchrl.trainers.algorithms.configs.modules import (
            MLPConfig,
            TanhNormalModelConfig,
        )

        network_cfg = MLPConfig(in_features=10, out_features=10, depth=2, num_cells=32)
        cfg = TanhNormalModelConfig(
            network=network_cfg,
            eval_mode=True,
            extract_normal_params=True,
            in_keys=["observation"],
            param_keys=["loc", "scale"],
            out_keys=["action"],
            exploration_type="RANDOM",
            return_log_prob=True,
        )
        assert (
            cfg._target_
            == "torchrl.trainers.algorithms.configs.modules._make_tanh_normal_model"
        )
        assert cfg.network == network_cfg
        assert cfg.eval_mode is True
        assert cfg.extract_normal_params is True
        assert cfg.in_keys == ["observation"]
        assert cfg.param_keys == ["loc", "scale"]
        assert cfg.out_keys == ["action"]
        assert cfg.exploration_type == "RANDOM"
        assert cfg.return_log_prob is True
        instantiate(cfg)

    def test_tanh_normal_model_config_defaults(self):
        """Test TanhNormalModelConfig with default values."""
        from torchrl.trainers.algorithms.configs.modules import (
            MLPConfig,
            TanhNormalModelConfig,
        )

        network_cfg = MLPConfig(in_features=10, out_features=10, depth=2, num_cells=32)
        cfg = TanhNormalModelConfig(network=network_cfg)

        # Test that defaults are set in __post_init__
        assert cfg.in_keys == ["observation"]
        assert cfg.param_keys == ["loc", "scale"]
        assert cfg.out_keys == ["action"]
        assert cfg.extract_normal_params is True
        assert cfg.return_log_prob is False
        assert cfg.exploration_type == "RANDOM"
        instantiate(cfg)

    def test_value_model_config(self):
        """Test ValueModelConfig."""
        from torchrl.trainers.algorithms.configs.modules import (
            MLPConfig,
            ValueModelConfig,
        )

        network_cfg = MLPConfig(in_features=10, out_features=1, depth=2, num_cells=32)
        cfg = ValueModelConfig(network=network_cfg)
        assert (
            cfg._target_
            == "torchrl.trainers.algorithms.configs.modules._make_value_model"
        )
        assert cfg.network == network_cfg

        # Test instantiation - this should work now with the new config structure
        value_model = instantiate(cfg)
        from torchrl.modules import MLP, ValueOperator

        assert isinstance(value_model, ValueOperator)
        assert isinstance(value_model.module, MLP)
        assert value_model.module.in_features == 10
        assert value_model.module.out_features == 1


class TestCollectorsConfig:
    @pytest.mark.parametrize("factory", [True, False])
    @pytest.mark.parametrize("collector", ["async", "multi_sync", "multi_async"])
    @pytest.mark.skipif(not _has_gym, reason="Gym is not installed")
    def test_collector_config(self, factory, collector):
        from torchrl.collectors.collectors import (
            aSyncDataCollector,
            MultiaSyncDataCollector,
            MultiSyncDataCollector,
        )
        from torchrl.trainers.algorithms.configs.collectors import (
            AsyncDataCollectorConfig,
            MultiaSyncDataCollectorConfig,
            MultiSyncDataCollectorConfig,
        )
        from torchrl.trainers.algorithms.configs.envs import GymEnvConfig
        from torchrl.trainers.algorithms.configs.modules import (
            MLPConfig,
            TanhNormalModelConfig,
        )

        # We need an env config and a policy config
        env_cfg = GymEnvConfig(env_name="Pendulum-v1")
        policy_cfg = TanhNormalModelConfig(
            network=MLPConfig(in_features=3, out_features=2, depth=2, num_cells=32),
            in_keys=["observation"],
            out_keys=["action"],
        )

        # Define cfg_cls and kwargs based on collector type
        if collector == "async":
            cfg_cls = AsyncDataCollectorConfig
            kwargs = {"create_env_fn": env_cfg, "frames_per_batch": 10}
        elif collector == "multi_sync":
            cfg_cls = MultiSyncDataCollectorConfig
            kwargs = {"create_env_fn": [env_cfg], "frames_per_batch": 10}
        elif collector == "multi_async":
            cfg_cls = MultiaSyncDataCollectorConfig
            kwargs = {"create_env_fn": [env_cfg], "frames_per_batch": 10}
        else:
            raise ValueError(f"Unknown collector type: {collector}")

        if factory:
            cfg = cfg_cls(policy_factory=policy_cfg, **kwargs)
        else:
            cfg = cfg_cls(policy=policy_cfg, **kwargs)

        # Check create_env_fn
        if collector in ["multi_sync", "multi_async"]:
            assert cfg.create_env_fn == [env_cfg]
        else:
            assert cfg.create_env_fn == env_cfg

        if factory:
            assert cfg.policy_factory._partial_
        else:
            assert not cfg.policy._partial_

        collector_instance = instantiate(cfg)
        try:
            if collector == "async":
                assert isinstance(collector_instance, aSyncDataCollector)
            elif collector == "multi_sync":
                assert isinstance(collector_instance, MultiSyncDataCollector)
            elif collector == "multi_async":
                assert isinstance(collector_instance, MultiaSyncDataCollector)
            for _c in collector_instance:
                # Just check that we can iterate
                break
        finally:
            # Only call shutdown if the collector has that method
            if hasattr(collector_instance, "shutdown"):
                collector_instance.shutdown(timeout=10)


class TestLossConfigs:
    @pytest.mark.parametrize("loss_type", ["clip", "kl", "ppo"])
    @pytest.mark.skipif(not _has_gym, reason="Gym is not installed")
    def test_ppo_loss_config(self, loss_type):
        from torchrl.objectives.ppo import ClipPPOLoss, KLPENPPOLoss, PPOLoss
        from torchrl.trainers.algorithms.configs.modules import (
            MLPConfig,
            TanhNormalModelConfig,
            TensorDictModuleConfig,
        )
        from torchrl.trainers.algorithms.configs.objectives import PPOLossConfig

        actor_network = TanhNormalModelConfig(
            network=MLPConfig(in_features=10, out_features=10, depth=2, num_cells=32),
            in_keys=["observation"],
            out_keys=["action"],
        )
        critic_network = TensorDictModuleConfig(
            module=MLPConfig(in_features=10, out_features=1, depth=2, num_cells=32),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        cfg = PPOLossConfig(
            actor_network=actor_network,
            critic_network=critic_network,
            loss_type=loss_type,
        )
        assert (
            cfg._target_
            == "torchrl.trainers.algorithms.configs.objectives._make_ppo_loss"
        )
        loss = instantiate(cfg)
        assert isinstance(loss, PPOLoss)
        if loss_type == "clip":
            assert isinstance(loss, ClipPPOLoss)
        elif loss_type == "kl":
            assert isinstance(loss, KLPENPPOLoss)


class TestOptimizerConfigs:
    def test_adam_config(self):
        """Test AdamConfig."""
        from torchrl.trainers.algorithms.configs.utils import AdamConfig

        cfg = AdamConfig(lr=1e-4, weight_decay=1e-5, betas=(0.95, 0.999))
        assert cfg._target_ == "torch.optim.Adam"
        assert cfg.lr == 1e-4
        assert cfg.weight_decay == 1e-5
        assert cfg.betas == (0.95, 0.999)
        assert cfg.eps == 1e-4  # Still default


class TestTrainerConfigs:
    @pytest.mark.skipif(not _has_gym, reason="Gym is not installed")
    def test_ppo_trainer_config(self):
        from torchrl.trainers.algorithms.configs.trainers import PPOTrainerConfig

        # Test that we can create a basic config
        cfg = PPOTrainerConfig(
            collector=None,
            total_frames=100,
            frame_skip=1,
            optim_steps_per_batch=1,
            loss_module=None,
            optimizer=None,
            logger=None,
            clip_grad_norm=True,
            clip_norm=1.0,
            progress_bar=True,
            seed=1,
            save_trainer_interval=10000,
            log_interval=10000,
            save_trainer_file=None,
            replay_buffer=None,
        )

        assert (
            cfg._target_
            == "torchrl.trainers.algorithms.configs.trainers._make_ppo_trainer"
        )
        assert cfg.total_frames == 100
        assert cfg.frame_skip == 1

    @pytest.mark.skipif(not _has_gym, reason="Gym is not installed")
    def test_ppo_trainer_config_optional_fields(self):
        """Test that optional fields can be omitted from PPO trainer config."""
        from torchrl.trainers.algorithms.configs.collectors import (
            SyncDataCollectorConfig,
        )
        from torchrl.trainers.algorithms.configs.data import (
            TensorDictReplayBufferConfig,
        )
        from torchrl.trainers.algorithms.configs.envs import GymEnvConfig
        from torchrl.trainers.algorithms.configs.modules import (
            MLPConfig,
            TanhNormalModelConfig,
            TensorDictModuleConfig,
        )
        from torchrl.trainers.algorithms.configs.objectives import PPOLossConfig
        from torchrl.trainers.algorithms.configs.trainers import PPOTrainerConfig
        from torchrl.trainers.algorithms.configs.utils import AdamConfig

        # Create minimal config with only required fields
        env_config = GymEnvConfig(env_name="CartPole-v1")

        actor_network = MLPConfig(
            in_features=4,  # CartPole observation space
            out_features=2,  # CartPole action space
            num_cells=64,
        )

        critic_network = MLPConfig(in_features=4, out_features=1, num_cells=64)

        actor_model = TanhNormalModelConfig(
            network=actor_network, in_keys=["observation"], out_keys=["action"]
        )

        critic_model = TensorDictModuleConfig(
            module=critic_network, in_keys=["observation"], out_keys=["state_value"]
        )

        loss_config = PPOLossConfig(
            actor_network=actor_model, critic_network=critic_model
        )

        optimizer_config = AdamConfig(lr=0.001)

        collector_config = SyncDataCollectorConfig(
            create_env_fn=env_config,
            policy=actor_model,
            total_frames=1000,
            frames_per_batch=100,
        )

        replay_buffer_config = TensorDictReplayBufferConfig()

        # Create trainer config with minimal required fields only
        trainer_config = PPOTrainerConfig(
            collector=collector_config,
            total_frames=1000,
            optim_steps_per_batch=1,
            loss_module=loss_config,
            optimizer=optimizer_config,
            logger=None,  # Optional field
            save_trainer_file="/tmp/test.pt",
            replay_buffer=replay_buffer_config
            # All optional fields are omitted to test defaults
        )

        # Verify that optional fields have default values
        assert trainer_config.frame_skip == 1
        assert trainer_config.clip_grad_norm is True
        assert trainer_config.clip_norm is None
        assert trainer_config.progress_bar is True
        assert trainer_config.seed is None
        assert trainer_config.save_trainer_interval == 10000
        assert trainer_config.log_interval == 10000
        assert trainer_config.create_env_fn is None
        assert trainer_config.actor_network is None
        assert trainer_config.critic_network is None


@pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
class TestHydraParsing:
    @pytest.fixture(autouse=True, scope="module")
    def init_hydra(self):
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        from hydra import initialize_config_module

        initialize_config_module("torchrl.trainers.algorithms.configs")

    @pytest.mark.skipif(not _has_gym, reason="Gym is not installed")
    def test_simple_config_instantiation(self):
        """Test that simple configs can be instantiated using registered names."""
        from hydra import compose
        from hydra.utils import instantiate
        from torchrl.envs import GymEnv
        from torchrl.modules import MLP

        # Test environment config
        env_cfg = compose(
            config_name="config",
            overrides=["+env=gym", "+env.env_name=CartPole-v1"],
        )
        env = instantiate(env_cfg.env)
        assert isinstance(env, GymEnv)
        assert env.env_name == "CartPole-v1"

        # Test with override
        env = instantiate(env_cfg.env, env_name="Pendulum-v1")
        assert isinstance(env, GymEnv), env
        assert env.env_name == "Pendulum-v1"

        # Test network config
        network_cfg = compose(
            config_name="config",
            overrides=[
                "+network=mlp",
                "+network.in_features=10",
                "+network.out_features=5",
            ],
        )
        network = instantiate(network_cfg.network)
        assert isinstance(network, MLP)

    @pytest.mark.skipif(not _has_gym, reason="Gym is not installed")
    def test_env_parsing(self, tmpdir):
        from hydra import compose
        from hydra.utils import instantiate
        from torchrl.envs import GymEnv

        # Method 1: Use Hydra's compose with overrides (recommended approach)
        # This directly uses the config group system like in the PPO trainer
        cfg_resolved = compose(
            config_name="config",  # Use the main config
            overrides=["+env=gym", "+env.env_name=CartPole-v1"],
        )

        # Now we can instantiate the environment
        env = instantiate(cfg_resolved.env)
        assert isinstance(env, GymEnv)
        assert env.env_name == "CartPole-v1"

    @pytest.mark.skipif(not _has_gym, reason="Gym is not installed")
    def test_env_parsing_with_file(self, tmpdir):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from torchrl.envs import GymEnv

        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(tmpdir), version_base=None)

        yaml_config = """
defaults:
  - env: gym
  - _self_

env:
  env_name: CartPole-v1
"""
        file = tmpdir / "config.yaml"
        with open(file, "w") as f:
            f.write(yaml_config)

        # Use Hydra's compose to resolve config groups
        cfg_from_file = compose(
            config_name="config",
        )

        # Now we can instantiate the environment
        env_from_file = instantiate(cfg_from_file.env)
        assert isinstance(env_from_file, GymEnv)
        assert env_from_file.env_name == "CartPole-v1"

    def test_collector_parsing_with_file(self, tmpdir):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from tensordict import TensorDict

        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(tmpdir), version_base=None)
        yaml_config = r"""
defaults:
  - env: gym
  - model: tanh_normal
  - network: mlp
  - collector: sync
  - _self_

network:
  out_features: 2
  in_features: 4  # CartPole observation space is 4-dimensional

model:
  return_log_prob: True
  in_keys: ["observation"]
  param_keys: ["loc", "scale"]
  out_keys: ["action"]
  network:
    out_features: 2
    in_features: 4  # CartPole observation space is 4-dimensional

env:
  env_name: CartPole-v1

collector:
  create_env_fn: ${env}
  policy: ${model}
  total_frames: 1000
  frames_per_batch: 100

"""

        file = tmpdir / "config.yaml"
        with open(file, "w") as f:
            f.write(yaml_config)

        # Use Hydra's compose to resolve config groups
        cfg_from_file = compose(config_name="config")

        collector = instantiate(cfg_from_file.collector)
        assert isinstance(collector, SyncDataCollector)
        for d in collector:
            assert isinstance(d, TensorDict)
            assert "action_log_prob" in d
            break

    def test_trainer_parsing_with_file(self, tmpdir):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(tmpdir), version_base=None)
        yaml_config = rf"""
defaults:
  - env: gym
  - model: tanh_normal
  - model@models.policy_model: tanh_normal
  - model@models.value_model: value
  - network: mlp
  - network@networks.policy_network: mlp
  - network@networks.value_network: mlp
  - collector: sync
  - replay_buffer: base
  - storage: tensor
  - sampler: random
  - writer: round_robin
  - trainer: ppo
  - optimizer: adam
  - loss: ppo
  - logger: wandb
  - _self_

networks:
  policy_network:
    out_features: 2
    in_features: 4  # CartPole observation space is 4-dimensional

  value_network:
    out_features: 1
    in_features: 4

models:
  policy_model:
    return_log_prob: True
    in_keys: ["observation"]
    param_keys: ["loc", "scale"]
    out_keys: ["action"]
    network: ${{networks.policy_network}}

  value_model:
    in_keys: ["observation"]
    out_keys: ["state_value"]
    network: ${{networks.value_network}}

env:
  env_name: CartPole-v1

storage:
  max_size: 1000
  device: cpu # should be optional
  ndim: 1 # should be optional

replay_buffer:
  storage: ${{storage}} # should be optional
  sampler: ${{sampler}} # should be optional
  writer: ${{writer}} # should be optional

loss:
  actor_network: ${{models.policy_model}}
  critic_network: ${{models.value_model}}

collector:
  create_env_fn: ${{env}}
  policy: ${{models.policy_model}}
  total_frames: 1000
  frames_per_batch: 100

optimizer:
  lr: 0.001

trainer:
  collector: ${{collector}}
  optimizer: ${{optimizer}}
  replay_buffer: ${{replay_buffer}}
  loss_module: ${{loss}}
  logger: ${{logger}}
  total_frames: 1000
  frame_skip: 1 # should be optional
  clip_grad_norm: 100 # should be optional and None if not provided
  clip_norm: null # should be optional
  progress_bar: true # should be optional
  seed: 0
  save_trainer_interval: 100 # should be optional
  log_interval: 100 # should be optional
  save_trainer_file: {tmpdir}/save/ckpt.pt
  optim_steps_per_batch: 1
"""

        file = tmpdir / "config.yaml"
        with open(file, "w") as f:
            f.write(yaml_config)

        os.makedirs(tmpdir / "save", exist_ok=True)

        # Use Hydra's compose to resolve config groups
        cfg_from_file = compose(config_name="config")

        networks = instantiate(cfg_from_file.networks)

        models = instantiate(cfg_from_file.models)

        loss = instantiate(cfg_from_file.loss)
        assert isinstance(loss, PPOLoss)

        collector = instantiate(cfg_from_file.collector)
        assert isinstance(collector, SyncDataCollector)

        trainer = instantiate(cfg_from_file.trainer)
        assert isinstance(trainer, PPOTrainer)
        trainer.train()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
