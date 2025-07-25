# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Demonstrating the use of compressing a rollout of Atari transitions on the GPU and batch decompressing them on the GPU.
This example may be helpful in the multi-environment case, or when multiple agents and environments are vmapped on the GPU.
Additionally, we can batch our decompression on the GPU in one go using the collate function.

Below are the results of running this example with different compression levels on an Atari rollout of Pong.
+---------------------+--------+--------+--------+--------+--------+
| Compressor Level    | 1      | 3      | 8      | 12     | 23     |
+=====================+========+========+========+========+========+
| Compression Ratio   | 95x    | 99x    | 106x   | 111x   | 122x   |
+---------------------+--------+--------+--------+--------+--------+

"""

from __future__ import annotations

import importlib

import sys

import time
from typing import Any, NamedTuple

import gymnasium as gym

import numpy as np
import torch
from tensordict import TensorDict
from torchrl import torchrl_logger as logger
from torchrl.data import CompressedListStorage, ListStorage, ReplayBuffer

# check if nvidia.nvcomp is available
has_nvcomp = importlib.util.find_spec("nvidia.nvcomp") is not None
if not has_nvcomp:
    raise ImportError(
        "Please pip install nvidia-nvcomp to use this example with GPU compression."
    )
else:
    import nvidia.nvcomp as nvcomp


class AtariTransition(NamedTuple):
    observations: np.uint8
    actions: np.uint8
    next_observations: np.uint8
    rewards: np.float32
    terminated: np.bool
    truncated: np.bool
    info: dict[str, Any]


def setup_atari_environment(seed: int = 42) -> gym.Env:
    import ale_py

    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=5)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.TransformReward(env, np.sign)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def run_rollout_benchmark(
    rb: ReplayBuffer,
    env: gym.Env,
    calculate_compression_ratio_fn: callable,
    create_and_add_transition_fn: callable,
    compress_obs: callable,
    num_transitions: int = 2000,
):
    """Run a rollout benchmark collecting transitions and measuring steps per second and compression ratios."""
    compression_ratios = []
    terminated = truncated = True
    next_obs = compressed_next_obs = None

    start_time = time.time()
    for _ in range(num_transitions):
        if terminated or truncated:
            obs, _ = env.reset()
            compressed_obs = compress_obs(obs)
        else:
            obs = next_obs
            compressed_obs = compressed_next_obs

        # perform some fake inference with the obs
        obs = torch.from_numpy(obs).cuda(non_blocking=True)
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Compress next observation
        compressed_next_obs = compress_obs(next_obs)
        compression_ratios.append(
            calculate_compression_ratio_fn(next_obs, compressed_next_obs)
        )

        # Create and add transition
        create_and_add_transition_fn(
            rb,
            compressed_obs,
            action,
            compressed_next_obs,
            reward,
            terminated,
            truncated,
            info,
        )

    rollout_time = time.time() - start_time
    return rollout_time, compression_ratios


def run_sampling_benchmark(rb: ReplayBuffer, num_samples=100, batch_size=32) -> float:
    """Run a sampling replaybuffer benchmark measuring decompression speed."""
    start_time = time.time()
    for _ in range(num_samples):
        rb.sample(batch_size)
    sampling_time = time.time() - start_time
    return sampling_time


def get_cpu_codec(level=1):
    """Returns compression and decompression functions for CPU."""
    if sys.version_info >= (3, 14):
        from compression import zstd

        def compress_fn(data):
            return zstd.compress(data, level)

        return compress_fn, zstd.decompress
    else:
        try:
            import zstd

            def compress_fn(data):
                return zstd.compress(data, level)

            return compress_fn, zstd.decompress
        except ImportError:
            raise ImportError(
                "Please `pip install zstd` to use this example with CPU compression."
            )


def get_gpu_codec(level=1):
    """Returns compression and decompression functions for GPU using NVIDIA NVCOMP.

    See the python API docs here: https://docs.nvidia.com/cuda/nvcomp/py_api.html
    """
    # RAW = Does not add header with nvCOMP metadata, so that the codec can read compressed data from the CPU library
    bitstream_kind = nvcomp.BitstreamKind.RAW
    # Note: NVCOMP may not support all compression levels the same way as CPU zstd
    codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=bitstream_kind)

    def compressor_fn(data: nvcomp.Array) -> nvcomp.Array:
        return codec.encode(data)

    def decompressor_fn(compressed_data: nvcomp.Array) -> nvcomp.Array:
        return codec.decode(compressed_data, data_type="|u1")

    return compressor_fn, decompressor_fn


def make_batch_decompressing_replay_buffer(decompressor_fn) -> ReplayBuffer:
    """
    Creates a ReplayBuffer with batched decompression on the GPU.
    """
    storage = ListStorage(
        max_size=1000,
        device="cuda",
    )

    def collate_compressed_data_and_batch_decompress(
        data: list[AtariTransition],
    ) -> list[AtariTransition]:
        """We collate the compressed data together so that we can decompress it in a single batch operation."""
        transitions = data

        # gather compressed data
        compressed_obs = [transition.observations for transition in transitions]
        compressed_next_obs = [
            transition.next_observations for transition in transitions
        ]

        # optional checks
        assert all(isinstance(arr, nvcomp.Array) for arr in compressed_obs)
        assert all(isinstance(arr, nvcomp.Array) for arr in compressed_next_obs)

        # batched decompress is faster
        decompressed_data = decompressor_fn(compressed_obs + compressed_next_obs)

        # gather decompressed data
        decompressed_obses = decompressed_data[: len(compressed_obs)]
        decompressed_next_obses = decompressed_data[len(compressed_obs) :]

        # repack data
        for i, (transition, obs, next_obs) in enumerate(
            zip(transitions, decompressed_obses, decompressed_next_obses)
        ):
            transitions[i] = transition._replace(
                observations=torch.from_dlpack(obs).view(4, 84, 84),
                next_observations=torch.from_dlpack(next_obs).view(4, 84, 84),
            )

        return transitions

    return ReplayBuffer(
        storage=storage,
        batch_size=32,
        collate_fn=collate_compressed_data_and_batch_decompress,
    )


def cpu_compress_to_gpu_decompress(level=1):
    env = setup_atari_environment(seed=0)

    compressor_fn, _ = get_cpu_codec(level)
    _, decompressor_fn = get_gpu_codec(level)

    obs, _ = env.reset(seed=0)
    compressed_obs = compressor_fn(obs.tobytes())
    decompressed_obs = decompressor_fn(compressed_obs)
    pt_obs = torch.from_dlpack(decompressed_obs).clone().view(4, 84, 84)
    assert np.allclose(obs, pt_obs.cpu().numpy())

    rb = make_batch_decompressing_replay_buffer(decompressor_fn)

    def calculate_compression_ratio(obs, compressed_next_obs):
        return len(obs.tobytes()) / len(compressor_fn(obs.tobytes()))

    def compress_obs(obs):
        return compressor_fn(obs.tobytes())

    def create_and_add_transition(
        rb,
        compressed_obs,
        action,
        compressed_next_obs,
        reward,
        terminated,
        truncated,
        info,
    ):
        # Convert compressed bytes to nvcomp arrays for GPU storage
        compressed_obs_data = nvcomp.as_array(compressed_obs).cuda(synchronize=False)
        compressed_nv_next_obs = nvcomp.as_array(compressed_next_obs).cuda(
            synchronize=False
        )

        transition = AtariTransition(
            compressed_obs_data,
            action,
            compressed_nv_next_obs,
            reward,
            terminated,
            truncated,
            info,
        )
        rb.add(transition)

    torch.cuda.synchronize()
    rollout_time, compression_ratios = run_rollout_benchmark(
        rb,
        env,
        calculate_compression_ratio,
        create_and_add_transition,
        compress_obs,
        2000,
    )

    torch.cuda.synchronize()
    sample_time = run_sampling_benchmark(rb, 100, 32)

    output = [
        "\nListStorage + ReplayBuffer (CPU compress, GPU decompress, storage on GPU) Example:",
        f"avg_compression_ratio={np.array(compression_ratios).mean():0.0f}",
        f"rollout with zstd, @ transitions/s={2000 / rollout_time:0.0f}",
        "batch sampling and decompression with zstd @ transitions/s={:0.0f}".format(
            (100 * 32) / sample_time
        ),
    ]

    logger.info("\n\t".join(output))


def cpu_only(level=1):
    env = setup_atari_environment(seed=0)

    compressor_fn, decompressor_fn = get_cpu_codec(level)

    # Test compression/decompression works correctly
    obs, _ = env.reset(seed=0)
    compressed_obs = compressor_fn(obs.tobytes())
    decompressed_obs = decompressor_fn(compressed_obs)
    recovered_obs = np.frombuffer(decompressed_obs, dtype=np.uint8).reshape(obs.shape)
    assert np.allclose(obs, recovered_obs)

    def compress_from_torch(data: torch.Tensor) -> bytes:
        """
        Convert a tensor to a byte stream for compression.
        """
        return compressor_fn(data.cpu().numpy().tobytes())

    def decompress_from_bytes(data: bytes, metadata: dict) -> torch.Tensor:
        """
        Convert a byte stream back to a tensor.
        """
        decompressed_data = bytearray(decompressor_fn(data))
        dtype = metadata.get("dtype", torch.float32)
        device = metadata.get("device", "cpu")
        shape = metadata.get("shape", ())

        return (
            torch.frombuffer(
                decompressed_data,
                dtype=dtype,
            )
            .view(shape)
            .to(device)
        )

    storage = CompressedListStorage(
        max_size=1000,
        compression_level=level,  # Use the passed compression level
        device="cpu",
        compression_fn=compress_from_torch,
        decompression_fn=decompress_from_bytes,
    )

    rb = ReplayBuffer(storage=storage, batch_size=32)

    def calculate_compression_ratio(obs, compressed_obs):
        # For cpu_only, the CompressedListStorage handles compression internally
        # so we calculate the ratio based on the original observation size vs compressed bytes
        original_size = obs.nbytes
        compressed_size = len(compressor_fn(obs.tobytes()))
        return original_size / compressed_size

    def compress_obs(obs):
        return torch.from_numpy(obs).clone()

    def create_and_add_transition(
        rb,
        compressed_obs,
        action,
        compressed_next_obs,
        reward,
        terminated,
        truncated,
        info,
    ):
        transition_tuple = AtariTransition(
            observations=compressed_obs,
            actions=torch.tensor(action),
            next_observations=compressed_next_obs,
            rewards=torch.tensor(reward, dtype=torch.float32),
            terminated=torch.tensor(terminated),
            truncated=torch.tensor(truncated),
            info=info,
        )
        transition = TensorDict.from_namedtuple(transition_tuple, batch_size=[])
        rb.add(transition)

    # Run rollout benchmark
    rollout_time, compression_ratios = run_rollout_benchmark(
        rb,
        env,
        calculate_compression_ratio,
        create_and_add_transition,
        compress_obs,
        2000,
    )

    sample_time = run_sampling_benchmark(rb, 100, 32)

    output = [
        "\nCompressedListStorage + ReplayBuffer (CPU compress, CPU decompress, storage on CPU) Example:",
        f"avg_compression_ratio={np.array(compression_ratios).mean():0.0f}",
        f"rollout with zstd, @ transitions/s={2000 / rollout_time:0.0f}",
        "batch sampling and decompression with zstd @ transitions/s={:0.0f}".format(
            (100 * 32) / sample_time
        ),
    ]

    logger.info("\n\t".join(output))


def gpu_only(level=1):
    env = setup_atari_environment(seed=0)

    compressor_fn, decompressor_fn = get_gpu_codec(level)

    obs, _ = env.reset(seed=0)
    nv_obs = nvcomp.as_array(obs).cuda(synchronize=False)
    compressed_obs = compressor_fn(nv_obs)
    decompressed_obs = decompressor_fn(compressed_obs)
    pt_obs = torch.from_dlpack(decompressed_obs).clone().view(4, 84, 84)
    assert np.allclose(obs, pt_obs.cpu().numpy())

    rb = make_batch_decompressing_replay_buffer(decompressor_fn)

    # State for tracking GPU observations between transitions
    def calculate_compression_ratio(obs, compressed_obs):
        nv_obs_temp = nvcomp.as_array(obs).cuda(synchronize=False)
        return nv_obs_temp.buffer_size / compressed_obs.buffer_size

    def compress_obs(obs):
        nv_obs = nvcomp.as_array(obs).cuda(synchronize=False)
        return compressor_fn(nv_obs)

    def create_and_add_transition(
        rb,
        compressed_obs,
        action,
        compressed_next_obs,
        reward,
        terminated,
        truncated,
        info,
    ):
        transition = AtariTransition(
            compressed_obs,
            action,
            compressed_next_obs,
            reward,
            terminated,
            truncated,
            info,
        )
        rb.add(transition)

    torch.cuda.synchronize()
    rollout_time, compression_ratios = run_rollout_benchmark(
        rb,
        env,
        calculate_compression_ratio,
        create_and_add_transition,
        compress_obs,
        2000,
    )

    torch.cuda.synchronize()
    sample_time = run_sampling_benchmark(rb, 100, 32)

    output = [
        "\nListStorage + ReplayBuffer (GPU compress, GPU decompress, storage on GPU) Example:",
        f"avg_compression_ratio={np.array(compression_ratios).mean():0.0f}",
        f"rollout with zstd, @ transitions/s={2000 / rollout_time:0.0f}",
        "batch sampling and decompression with zstd @ transitions/s={:0.0f}".format(
            (100 * 32) / sample_time
        ),
    ]

    logger.info("\n\t".join(output))


if __name__ == "__main__":
    for level in [1, 3, 8, 12, 22]:
        print(f"Running with compression level {level}...")
        cpu_only(level)
        gpu_only(level)
        cpu_compress_to_gpu_decompress(level)
