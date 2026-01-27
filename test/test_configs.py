# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib.util
import sys

import pytest
import torch

from torchrl import logger as torchrl_logger

from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.envs import AsyncEnvPool, ParallelEnv, SerialEnv
from torchrl.modules.models.models import MLP

# Test if configs can be imported (requires hydra)
try:
    from torchrl.trainers.algorithms.configs.modules import (
        ActivationConfig,
        LayerConfig,
    )

    _configs_available = True
except ImportError:
    _configs_available = False
    ActivationConfig = LayerConfig = None


_has_gym = (importlib.util.find_spec("gym") is not None) or (
    importlib.util.find_spec("gymnasium") is not None
)
_has_gymnasium = importlib.util.find_spec("gymnasium") is not None
_has_hydra = importlib.util.find_spec("hydra") is not None
_python_version_compatible = sys.version_info >= (3, 10)

# Make sure that warnings raise an exception
pytestmark = [
    pytest.mark.filterwarnings("error"),
]


@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(
    not _configs_available, reason="Config system requires hydra-core and omegaconf"
)
class TestEnvConfigs:
    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_gym_env_config(self):
        from hydra.utils import instantiate
        from torchrl.trainers.algorithms.configs.envs_libs import GymEnvConfig

        cfg = GymEnvConfig(env_name="CartPole-v1")
        assert cfg.env_name == "CartPole-v1"
        assert cfg.backend == "gymnasium"
        assert cfg.from_pixels is False
        instantiate(cfg)

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    @pytest.mark.parametrize("cls", [ParallelEnv, SerialEnv, AsyncEnvPool])
    def test_batched_env_config(self, cls):
        from hydra.utils import instantiate
        from torchrl.trainers.algorithms.configs.envs import BatchedEnvConfig
        from torchrl.trainers.algorithms.configs.envs_libs import GymEnvConfig

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


@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(
    not _configs_available, reason="Config system requires hydra-core and omegaconf"
)
class TestDataConfigs:
    """Test cases for data.py configuration classes."""

    def test_writer_config(self):
        """Test basic WriterConfig."""
        from torchrl.trainers.algorithms.configs.data import WriterConfig

        cfg = WriterConfig()
        assert cfg._target_ == "torchrl.data.replay_buffers.Writer"

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_round_robin_writer_config(self):
        """Test RoundRobinWriterConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_random_sampler_config(self):
        """Test RandomSamplerConfig."""
        from hydra.utils import instantiate
        from torchrl.trainers.algorithms.configs.data import RandomSamplerConfig

        cfg = RandomSamplerConfig()
        assert cfg._target_ == "torchrl.data.replay_buffers.RandomSampler"

        # Test instantiation
        sampler = instantiate(cfg)
        from torchrl.data.replay_buffers.samplers import RandomSampler

        assert isinstance(sampler, RandomSampler)

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_tensor_storage_config(self):
        """Test TensorStorageConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_tensordict_replay_buffer_config(self):
        """Test TensorDictReplayBufferConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_list_storage_config(self):
        """Test ListStorageConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_replay_buffer_config(self):
        """Test ReplayBufferConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_tensordict_replay_buffer_config_optional_fields(self):
        """Test that optional fields can be omitted from TensorDictReplayBuffer config."""
        from hydra.utils import instantiate
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
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_tensor_dict_round_robin_writer_config(self):
        """Test TensorDictRoundRobinWriterConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_immutable_dataset_writer_config(self):
        """Test ImmutableDatasetWriterConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_prioritized_sampler_config(self):
        """Test PrioritizedSamplerConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_sampler_without_replacement_config(self):
        """Test SamplerWithoutReplacementConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_lazy_stack_storage_config(self):
        """Test LazyStackStorageConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_lazy_memmap_storage_config(self):
        """Test LazyMemmapStorageConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_lazy_tensor_storage_config(self):
        """Test LazyTensorStorageConfig."""
        from hydra.utils import instantiate
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


@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(
    not _configs_available, reason="Config system requires hydra-core and omegaconf"
)
class TestModuleConfigs:
    """Test cases for modules.py configuration classes."""

    def test_network_config(self):
        """Test basic NetworkConfig."""
        from torchrl.trainers.algorithms.configs.modules import NetworkConfig

        cfg = NetworkConfig()
        # This is a base class, so it should not have a _target_
        assert not hasattr(cfg, "_target_")

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_mlp_config(self):
        """Test MLPConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_convnet_config(self):
        """Test ConvNetConfig."""
        from hydra.utils import instantiate
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
        assert (
            cfg._target_
            == "torchrl.trainers.algorithms.configs.modules._make_tensordict_module"
        )
        assert cfg.module._target_ == "torchrl.modules.MLP"
        assert cfg.in_keys == ["observation"]
        assert cfg.out_keys == ["action"]
        # Note: We can't test instantiation due to missing tensordict dependency

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_tanh_normal_model_config(self):
        """Test TanhNormalModelConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_tanh_normal_model_config_defaults(self):
        """Test TanhNormalModelConfig with default values."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_tensordict_sequential_config(self):
        """Test TensorDictSequentialConfig."""
        from hydra.utils import instantiate
        from torchrl.trainers.algorithms.configs.modules import (
            MLPConfig,
            TensorDictModuleConfig,
            TensorDictSequentialConfig,
        )

        cfg = TensorDictSequentialConfig(
            modules=[
                TensorDictModuleConfig(
                    module=MLPConfig(
                        in_features=10, out_features=10, depth=2, num_cells=32
                    ),
                    in_keys=["observation"],
                    out_keys=["hidden"],
                ),
                TensorDictModuleConfig(
                    module=MLPConfig(
                        in_features=10, out_features=5, depth=2, num_cells=32
                    ),
                    in_keys=["hidden"],
                    out_keys=["action"],
                ),
            ],
            partial_tolerant=False,
            selected_out_keys=None,
            inplace=None,
        )
        assert (
            cfg._target_
            == "torchrl.trainers.algorithms.configs.modules._make_tensordict_sequential"
        )
        assert cfg.modules is not None
        assert len(cfg.modules) == 2
        assert cfg.partial_tolerant is False
        assert cfg.selected_out_keys is None
        assert cfg.inplace is None

        seq = instantiate(cfg)
        from tensordict.nn import TensorDictSequential

        assert isinstance(seq, TensorDictSequential)
        assert len(seq.module) == 2
        from tensordict.nn import TensorDictModule

        assert all(isinstance(m, TensorDictModule) for m in seq.module)
        assert seq.in_keys == ["observation"]
        assert "action" in seq.out_keys

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_tanh_module_config(self):
        """Test TanhModuleConfig."""
        from hydra.utils import instantiate
        from torchrl.trainers.algorithms.configs.modules import TanhModuleConfig

        cfg = TanhModuleConfig(
            in_keys=["action"],
            out_keys=["action"],
            low=-1.0,
            high=1.0,
            clamp=False,
        )
        assert (
            cfg._target_
            == "torchrl.trainers.algorithms.configs.modules._make_tanh_module"
        )
        assert cfg.in_keys == ["action"]
        assert cfg.out_keys == ["action"]
        assert cfg.low == -1.0
        assert cfg.high == 1.0
        assert cfg.clamp is False

        # Test instantiation
        tanh_module = instantiate(cfg)
        from torchrl.modules import TanhModule

        assert isinstance(tanh_module, TanhModule)
        assert tanh_module.in_keys == ["action"]
        assert tanh_module.out_keys == ["action"]

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_value_model_config(self):
        """Test ValueModelConfig."""
        from hydra.utils import instantiate
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

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_additive_gaussian_module_config(self):
        """Test AdditiveGaussianModuleConfig."""
        from hydra.utils import instantiate
        from torchrl.trainers.algorithms.configs.modules import (
            AdditiveGaussianModuleConfig,
        )

        cfg = AdditiveGaussianModuleConfig(
            spec=None,
            sigma_init=1.0,
            sigma_end=0.1,
            annealing_num_steps=1000,
            mean=0.0,
            std=0.1,
            action_key="action",
        )
        assert (
            cfg._target_
            == "torchrl.trainers.algorithms.configs.modules._make_additive_gaussian_module"
        )
        assert cfg.spec is None
        assert cfg.sigma_init == 1.0
        assert cfg.sigma_end == 0.1
        assert cfg.action_key == "action"

        module = instantiate(cfg)
        from torchrl.modules.tensordict_module.exploration import AdditiveGaussianModule

        assert isinstance(module, AdditiveGaussianModule)
        assert module._spec is None
        assert module.action_key == "action"


@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(
    not _configs_available, reason="Config system requires hydra-core and omegaconf"
)
class TestCollectorsConfig:
    @pytest.mark.parametrize("factory", [True, False])
    @pytest.mark.parametrize("collector", ["async", "multi_sync", "multi_async"])
    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_collector_config(self, factory, collector):
        from hydra.utils import instantiate
        from torchrl.collectors import (
            AsyncCollector,
            MultiAsyncCollector,
            MultiSyncCollector,
        )
        from torchrl.trainers.algorithms.configs.collectors import (
            AsyncDataCollectorConfig,
            MultiAsyncCollectorConfig,
            MultiSyncCollectorConfig,
        )
        from torchrl.trainers.algorithms.configs.envs_libs import GymEnvConfig
        from torchrl.trainers.algorithms.configs.modules import (
            MLPConfig,
            TanhNormalModelConfig,
        )
        from torchrl.trainers.algorithms.configs.weight_update import (
            RemoteModuleWeightUpdaterConfig,
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
            cfg_cls = MultiSyncCollectorConfig
            kwargs = {"create_env_fn": [env_cfg], "frames_per_batch": 10}
        elif collector == "multi_async":
            cfg_cls = MultiAsyncCollectorConfig
            kwargs = {"create_env_fn": [env_cfg], "frames_per_batch": 10}
        else:
            raise ValueError(f"Unknown collector type: {collector}")

        if factory:
            # When using policy_factory, use RemoteModuleWeightUpdater to suppress warnings
            cfg = cfg_cls(
                policy_factory=policy_cfg,
                weight_updater=RemoteModuleWeightUpdaterConfig(),
                **kwargs,
            )
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
                assert isinstance(collector_instance, AsyncCollector)
            elif collector == "multi_sync":
                assert isinstance(collector_instance, MultiSyncCollector)
            elif collector == "multi_async":
                assert isinstance(collector_instance, MultiAsyncCollector)
            for _c in collector_instance:
                # Just check that we can iterate
                break
        finally:
            collector_instance.shutdown(timeout=10)

    @pytest.mark.parametrize("factory", [True, False])
    @pytest.mark.parametrize("collector", ["async", "multi_sync", "multi_async"])
    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_collector_auto_configures_exploration_modules(self, factory, collector):
        """Test that collector instantiation auto-configures exploration modules.

        This tests that when an exploration module (e.g. AdditiveGaussianModule) is configured
        without a spec (spec=None), the collector's __init__ automatically sets the spec
        from the environment's action_spec.
        """
        from hydra.utils import instantiate
        from torchrl.collectors import (
            AsyncCollector,
            MultiAsyncCollector,
            MultiSyncCollector,
        )
        from torchrl.trainers.algorithms.configs.collectors import (
            AsyncDataCollectorConfig,
            MultiAsyncCollectorConfig,
            MultiSyncCollectorConfig,
        )
        from torchrl.trainers.algorithms.configs.envs_libs import GymEnvConfig
        from torchrl.trainers.algorithms.configs.modules import (
            AdditiveGaussianModuleConfig,
            MLPConfig,
            TanhNormalModelConfig,
            TensorDictSequentialConfig,
        )
        from torchrl.trainers.algorithms.configs.weight_update import (
            RemoteModuleWeightUpdaterConfig,
        )

        env_cfg = GymEnvConfig(env_name="Pendulum-v1")

        policy_cfg = TanhNormalModelConfig(
            network=MLPConfig(in_features=3, out_features=2, depth=2, num_cells=32),
            in_keys=["observation"],
            out_keys=["action"],
        )

        exploration_cfg = AdditiveGaussianModuleConfig(
            spec=None,  # Will be auto-set by collector from environment
            sigma_init=0.5,
            sigma_end=0.1,
            annealing_num_steps=100,
            action_key="action",
        )

        exploratory_policy_cfg = TensorDictSequentialConfig(
            modules=[policy_cfg, exploration_cfg],
        )

        if collector == "async":
            cfg_cls = AsyncDataCollectorConfig
            expected_cls = AsyncCollector
            kwargs = {"create_env_fn": env_cfg, "frames_per_batch": 10}
        elif collector == "multi_sync":
            cfg_cls = MultiSyncCollectorConfig
            expected_cls = MultiSyncCollector
            kwargs = {"create_env_fn": [env_cfg], "frames_per_batch": 10}
        elif collector == "multi_async":
            cfg_cls = MultiAsyncCollectorConfig
            expected_cls = MultiAsyncCollector
            kwargs = {"create_env_fn": [env_cfg], "frames_per_batch": 10}
        else:
            raise ValueError(f"Unknown collector type: {collector}")

        if factory:
            cfg = cfg_cls(
                policy_factory=exploratory_policy_cfg,
                weight_updater=RemoteModuleWeightUpdaterConfig(),
                **kwargs,
            )
        else:
            cfg = cfg_cls(policy=exploratory_policy_cfg, **kwargs)

        collector_instance = instantiate(cfg)

        try:
            assert isinstance(collector_instance, expected_cls)
            # This would raise RuntimeError if spec was not set on workers' exploration module
            for batch in collector_instance:
                assert "action" in batch.keys()
                assert batch["action"].shape[-1] == 1
                assert (batch["action"] >= -2.0).all() and (
                    batch["action"] <= 2.0
                ).all(), "Actions should be clipped to environment spec bounds [-2, 2]"
                break

        finally:
            collector_instance.shutdown(timeout=10)


@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
class TestLossConfigs:
    @pytest.mark.parametrize("loss_type", ["clip", "kl", "ppo"])
    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_ppo_loss_config(self, loss_type):
        from hydra.utils import instantiate
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


@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(
    not _configs_available, reason="Config system requires hydra-core and omegaconf"
)
class TestOptimizerConfigs:
    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_adam_config(self):
        """Test AdamConfig."""
        from torchrl.trainers.algorithms.configs.utils import AdamConfig

        cfg = AdamConfig(lr=1e-4, weight_decay=1e-5, betas=(0.95, 0.999))
        assert cfg._target_ == "torch.optim.Adam"
        assert cfg.lr == 1e-4
        assert cfg.weight_decay == 1e-5
        assert cfg.betas == (0.95, 0.999)
        assert cfg.eps == 1e-4  # Still default


@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(
    not _configs_available, reason="Config system requires hydra-core and omegaconf"
)
class TestTrainerConfigs:
    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
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

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_ppo_trainer_config_optional_fields(self):
        """Test that optional fields can be omitted from PPO trainer config."""
        from torchrl.trainers.algorithms.configs.collectors import CollectorConfig
        from torchrl.trainers.algorithms.configs.data import (
            TensorDictReplayBufferConfig,
        )
        from torchrl.trainers.algorithms.configs.envs_libs import GymEnvConfig
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

        collector_config = CollectorConfig(
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
@pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(
    not _configs_available, reason="Config system requires hydra-core and omegaconf"
)
@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="hydra-core argparse integration is not compatible with Python 3.14+",
)
class TestHydraParsing:
    @pytest.fixture(autouse=True, scope="module")
    def init_hydra(self):
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        from hydra import initialize_config_module
        from torchrl.trainers.algorithms.configs import _register_configs

        # Register the configs manually for testing
        _register_configs()
        initialize_config_module(
            "torchrl.trainers.algorithms.configs", version_base="1.1"
        )

    def _run_hydra_test(
        self, tmpdir, yaml_config, test_script_content, success_message="SUCCESS"
    ):
        """Helper function to run a Hydra test with subprocess approach."""
        import subprocess
        import sys

        # Create a test script that follows the pattern
        test_script = tmpdir / "test.py"

        script_content = f"""
import hydra
import torchrl
from torchrl.trainers.algorithms.configs.common import Config

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
{test_script_content}
    print("{success_message}")
    return True

if __name__ == "__main__":
    main()
"""

        with open(test_script, "w") as f:
            f.write(script_content)

        # Create the config directory structure
        config_dir = tmpdir / "config"
        config_dir.mkdir()

        config_file = config_dir / "config.yaml"
        with open(config_file, "w") as f:
            f.write(yaml_config)

        # Run the test script using subprocess
        try:
            result = subprocess.run(
                [sys.executable, str(test_script)],
                cwd=str(tmpdir),
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                assert success_message in result.stdout
                torchrl_logger.info("Test passed!")
            else:
                # Filter out known Hydra warnings that shouldn't cause test failures
                stderr_lines = (
                    result.stderr.strip().split("\n") if result.stderr.strip() else []
                )
                filtered_stderr = []
                for line in stderr_lines:
                    # Skip the Hydra working directory warning and related warnings
                    if any(
                        warning_text in line
                        for warning_text in [
                            "Future Hydra versions will no longer change working directory at job runtime by default",
                            "UserWarning: Future Hydra versions will no longer change working directory",
                            "hydra/_internal/hydra.py",
                        ]
                    ):
                        continue
                    filtered_stderr.append(line)

                filtered_stderr_text = "\n".join(filtered_stderr).strip()

                # Only fail if there are actual errors after filtering warnings
                if filtered_stderr_text:
                    torchrl_logger.error(f"Test failed: {filtered_stderr_text}")
                    torchrl_logger.error(f"stdout: {result.stdout}")
                    raise AssertionError(f"Test failed: {filtered_stderr_text}")
                else:
                    # No real errors, just warnings - treat as success if stdout contains success message
                    assert success_message in result.stdout
                    torchrl_logger.info("Test passed (with warnings filtered)!")

        except subprocess.TimeoutExpired:
            raise AssertionError("Test timed out")
        except Exception:
            raise

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_simple_env_config(self, tmpdir):
        """Test simple environment configuration without any transforms or batching."""
        yaml_config = """
defaults:
  - env: gym
  - _self_

env:
  env_name: CartPole-v1
"""

        test_code = """
    env = hydra.utils.instantiate(cfg.env)
    assert isinstance(env, torchrl.envs.EnvBase)
    assert env.env_name == "CartPole-v1"
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_batched_env_config(self, tmpdir):
        """Test batched environment configuration without transforms."""
        yaml_config = """
defaults:
  - env@training_env: batched_env
  - env@training_env.create_env_fn: gym
  - _self_

training_env:
  num_workers: 2
  create_env_fn:
    env_name: CartPole-v1
    _partial_: true
"""

        test_code = """
    env = hydra.utils.instantiate(cfg.training_env)
    assert isinstance(env, torchrl.envs.EnvBase)
    assert env.num_workers == 2
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_batched_env_with_one_transform(self, tmpdir):
        """Test batched environment with one transform."""
        yaml_config = """
defaults:
  - env@training_env: batched_env
  - env@training_env.create_env_fn: transformed_env
  - env@training_env.create_env_fn.base_env: gym
  - transform@training_env.create_env_fn.transform: noop_reset
  - _self_

training_env:
  num_workers: 2
  create_env_fn:
    base_env:
      env_name: CartPole-v1
    transform:
      noops: 10
      random: true
"""

        test_code = """
    env = hydra.utils.instantiate(cfg.training_env)
    assert isinstance(env, torchrl.envs.EnvBase)
    assert env.num_workers == 2
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_batched_env_with_two_transforms(self, tmpdir):
        """Test batched environment with two transforms using Compose."""
        yaml_config = """
defaults:
  - env@training_env: batched_env
  - env@training_env.create_env_fn: transformed_env
  - env@training_env.create_env_fn.base_env: gym
  - transform@training_env.create_env_fn.transform: compose
  - transform@transform0: noop_reset
  - transform@transform1: step_counter
  - _self_

transform0:
  noops: 10
  random: true

transform1:
  max_steps: 200
  step_count_key: "step_count"

training_env:
  num_workers: 2
  create_env_fn:
    base_env:
      env_name: CartPole-v1
    transform:
      transforms:
        - ${transform0}
        - ${transform1}
    _partial_: true
"""

        test_code = """
    env = hydra.utils.instantiate(cfg.training_env)
    assert isinstance(env, torchrl.envs.EnvBase)
    assert env.num_workers == 2
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_simple_config_instantiation(self, tmpdir):
        """Test that simple configs can be instantiated using registered names."""
        yaml_config = """
defaults:
  - env: gym
  - network: mlp
  - _self_

env:
  env_name: CartPole-v1

network:
  in_features: 10
  out_features: 5
"""

        test_code = """
    # Test environment config
    env = hydra.utils.instantiate(cfg.env)
    assert isinstance(env, torchrl.envs.EnvBase)
    assert env.env_name == "CartPole-v1"

    # Test network config
    network = hydra.utils.instantiate(cfg.network)
    assert isinstance(network, torchrl.modules.MLP)
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_env_parsing(self, tmpdir):
        """Test environment parsing with overrides."""
        yaml_config = """
defaults:
  - env: gym
  - _self_

env:
  env_name: CartPole-v1
"""

        test_code = """
    env = hydra.utils.instantiate(cfg.env)
    assert isinstance(env, torchrl.envs.EnvBase)
    assert env.env_name == "CartPole-v1"
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_env_parsing_with_file(self, tmpdir):
        """Test environment parsing with file config."""
        yaml_config = """
defaults:
  - env: gym
  - _self_

env:
  env_name: CartPole-v1
"""

        test_code = """
    env = hydra.utils.instantiate(cfg.env)
    assert isinstance(env, torchrl.envs.EnvBase)
    assert env.env_name == "CartPole-v1"
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_collector_parsing_with_file(self, tmpdir):
        """Test collector parsing with file config."""
        yaml_config = """
defaults:
  - env: gym
  - model: tanh_normal
  - network: mlp
  - collector: sync
  - _self_

network:
  out_features: 2
  in_features: 4

model:
  return_log_prob: true
  in_keys: ["observation"]
  param_keys: ["loc", "scale"]
  out_keys: ["action"]
  network:
    out_features: 2
    in_features: 4

env:
  env_name: CartPole-v1

collector:
  create_env_fn: ${env}
  policy: ${model}
  total_frames: 1000
  frames_per_batch: 100
"""

        test_code = """
    collector = hydra.utils.instantiate(cfg.collector)
    assert isinstance(collector, torchrl.collectors.Collector)
    # Just verify we can create the collector without running it
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_trainer_parsing_with_file(self, tmpdir):
        """Test trainer parsing with file config."""
        import os

        os.makedirs(tmpdir / "save", exist_ok=True)

        yaml_config = f"""
defaults:
  - env@training_env: gym
  - model@models.policy_model: tanh_normal
  - model@models.value_model: value
  - network@networks.policy_network: mlp
  - network@networks.value_network: mlp
  - collector@data_collector: sync
  - replay_buffer@replay_buffer: base
  - storage@storage: tensor
  - sampler@sampler: without_replacement
  - writer@writer: round_robin
  - trainer@trainer: ppo
  - optimizer@optimizer: adam
  - loss@loss: ppo
  - logger@logger: csv
  - _self_

networks:
  policy_network:
    out_features: 2
    in_features: 4

  value_network:
    out_features: 1
    in_features: 4

models:
  policy_model:
    return_log_prob: true
    in_keys: ["observation"]
    param_keys: ["loc", "scale"]
    out_keys: ["action"]
    network: ${{networks.policy_network}}

  value_model:
    in_keys: ["observation"]
    out_keys: ["state_value"]
    network: ${{networks.value_network}}

training_env:
  env_name: CartPole-v1

storage:
  max_size: 1000
  device: cpu
  ndim: 1

replay_buffer:
  storage: ${{storage}}
  sampler: ${{sampler}}
  writer: ${{writer}}

loss:
  actor_network: ${{models.policy_model}}
  critic_network: ${{models.value_model}}

data_collector:
  create_env_fn: ${{training_env}}
  policy: ${{models.policy_model}}
  total_frames: 1000
  frames_per_batch: 100

optimizer:
  lr: 0.001

logger:
  exp_name: test_exp

trainer:
  collector: ${{data_collector}}
  optimizer: ${{optimizer}}
  replay_buffer: ${{replay_buffer}}
  loss_module: ${{loss}}
  logger: ${{logger}}
  total_frames: 1000
  frame_skip: 1
  clip_grad_norm: true
  clip_norm: 100.0
  progress_bar: false
  seed: 42
  save_trainer_interval: 100
  log_interval: 100
  save_trainer_file: {tmpdir}/save/ckpt.pt
  optim_steps_per_batch: 1
"""

        test_code = """
    # Just verify we can instantiate the main components without running
    loss = hydra.utils.instantiate(cfg.loss)
    assert isinstance(loss, torchrl.objectives.PPOLoss)

    collector = hydra.utils.instantiate(cfg.data_collector)
    assert isinstance(collector, torchrl.collectors.Collector)

    trainer = hydra.utils.instantiate(cfg.trainer)
    assert isinstance(trainer, torchrl.trainers.algorithms.ppo.PPOTrainer)
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")

    @pytest.mark.skipif(not _has_gymnasium, reason="Gymnasium is not installed")
    def test_transformed_env_parsing_with_file(self, tmpdir):
        """Test transformed environment configuration using the same pattern as the working PPO trainer."""
        yaml_config = """
defaults:
  - env@training_env: batched_env
  - env@training_env.create_env_fn: transformed_env
  - env@training_env.create_env_fn.base_env: gym
  - transform@training_env.create_env_fn.transform: compose
  - transform@transform0: noop_reset
  - transform@transform1: step_counter
  - _self_

transform0:
  noops: 30
  random: true

transform1:
  max_steps: 200
  step_count_key: "step_count"

training_env:
  num_workers: 2
  create_env_fn:
    base_env:
      env_name: Pendulum-v1
    transform:
      transforms:
        - ${transform0}
        - ${transform1}
    _partial_: true
"""

        test_code = """
    env = hydra.utils.instantiate(cfg.training_env)
    assert isinstance(env, torchrl.envs.EnvBase)
"""

        self._run_hydra_test(tmpdir, yaml_config, test_code, "SUCCESS")


@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(
    not _configs_available, reason="Config system requires hydra-core and omegaconf"
)
class TestWeightUpdaterConfigs:
    """Test cases for weight_update.py configuration classes."""

    def test_weight_updater_config(self):
        """Test basic WeightUpdaterConfig."""
        from torchrl.trainers.algorithms.configs.weight_update import (
            WeightUpdaterConfig,
        )

        cfg = WeightUpdaterConfig()
        assert cfg._target_ == "torchrl.collectors.WeightUpdaterBase"
        assert cfg._partial_ is True

    def test_vanilla_weight_updater_config(self):
        """Test VanillaWeightUpdaterConfig."""
        from torchrl.trainers.algorithms.configs.weight_update import (
            VanillaWeightUpdaterConfig,
        )

        cfg = VanillaWeightUpdaterConfig()
        assert cfg._target_ == "torchrl.collectors.VanillaWeightUpdater"
        assert cfg._partial_ is True
        assert cfg.weight_getter is None
        assert cfg.policy_weights is None

    def test_multiprocessed_weight_updater_config(self):
        """Test MultiProcessedWeightUpdaterConfig."""
        from torchrl.trainers.algorithms.configs.weight_update import (
            MultiProcessedWeightUpdaterConfig,
        )

        cfg = MultiProcessedWeightUpdaterConfig()
        assert cfg._target_ == "torchrl.collectors.MultiProcessedWeightUpdater"
        assert cfg._partial_ is True
        assert cfg.get_server_weights is None
        assert cfg.policy_weights is None

    def test_ray_weight_updater_config(self):
        """Test RayWeightUpdaterConfig."""
        from torchrl.trainers.algorithms.configs.weight_update import (
            RayWeightUpdaterConfig,
        )

        cfg = RayWeightUpdaterConfig(max_interval=5)
        assert cfg._target_ == "torchrl.collectors.RayWeightUpdater"
        assert cfg._partial_ is True
        assert cfg.policy_weights is None
        assert cfg.remote_collectors is None
        assert cfg.max_interval == 5

    def test_rpc_weight_updater_config(self):
        """Test RPCWeightUpdaterConfig."""
        from torchrl.trainers.algorithms.configs.weight_update import (
            RPCWeightUpdaterConfig,
        )

        cfg = RPCWeightUpdaterConfig(num_workers=4)
        assert cfg._target_ == "torchrl.collectors.distributed.RPCWeightUpdater"
        assert cfg._partial_ is True
        assert cfg.collector_infos is None
        assert cfg.collector_class is None
        assert cfg.collector_rrefs is None
        assert cfg.policy_weights is None
        assert cfg.num_workers == 4

    def test_distributed_weight_updater_config(self):
        """Test DistributedWeightUpdaterConfig."""
        from torchrl.trainers.algorithms.configs.weight_update import (
            DistributedWeightUpdaterConfig,
        )

        cfg = DistributedWeightUpdaterConfig(num_workers=8, sync=False)
        assert cfg._target_ == "torchrl.collectors.distributed.DistributedWeightUpdater"
        assert cfg._partial_ is True
        assert cfg.store is None
        assert cfg.policy_weights is None
        assert cfg.num_workers == 8
        assert cfg.sync is False

    def test_remote_module_weight_updater_config(self):
        """Test RemoteModuleWeightUpdaterConfig."""
        from torchrl.trainers.algorithms.configs.weight_update import (
            RemoteModuleWeightUpdaterConfig,
        )

        cfg = RemoteModuleWeightUpdaterConfig()
        assert cfg._target_ == "torchrl.collectors.RemoteModuleWeightUpdater"
        assert cfg._partial_ is True

    def test_vllm_updater_config(self):
        """Test vLLMUpdaterConfig."""
        from torchrl.trainers.algorithms.configs.weight_update import vLLMUpdaterConfig

        cfg = vLLMUpdaterConfig(
            master_address="localhost", master_port=12345, vllm_tp_size=2
        )
        assert cfg._target_ == "torchrl.collectors.llm.vLLMUpdater"
        assert cfg._partial_ is True
        assert cfg.master_address == "localhost"
        assert cfg.master_port == 12345
        assert cfg.model_metadata is None
        assert cfg.vllm_tp_size == 2


@pytest.mark.skipif(
    not _python_version_compatible, reason="Python 3.10+ required for config system"
)
@pytest.mark.skipif(
    not _configs_available, reason="Config system requires hydra-core and omegaconf"
)
class TestTransformConfigs:
    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_init_tracker_config(self):
        from hydra.utils import instantiate
        from torchrl.trainers.algorithms.configs.transforms import InitTrackerConfig

        cfg = InitTrackerConfig(init_key="is_test_init")
        assert cfg.init_key == "is_test_init"
        instantiate(cfg)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
