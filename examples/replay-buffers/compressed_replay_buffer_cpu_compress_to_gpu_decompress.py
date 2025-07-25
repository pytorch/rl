# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Demonstrating the use of compressing a rollout of Atari transitions using zstd from the CPU and batch decompressing them on the GPU.
This example is helpful for the single-environment Atari case, where it is faster to compress a single transition on the CPU then send it to the GPU for storage.
Additionally, we can batch our decompression on the GPU in one go using the collate function.

Results showing transitions per second (T/s).
Mode (using zstd)   Rollout T/s  Sample T/s   Compression Ratio
------------------------------------------------------------
gpu_only            1103         27616        96x
cpu_to_gpu          2095         27870        97x
"""

import numpy as np

import torch
from compressed_replay_buffer_cpu_only import get_cpu_codec
from compressed_replay_buffer_gpu_only import (
    AtariTransition,
    get_gpu_codec,
    make_compressing_replay_buffer,
    setup_atari_environment,
)
from torchrl import torchrl_logger as logger


def main():
    import time

    import nvidia.nvcomp as nvcomp

    # Create Pong environment and get a frame
    env = setup_atari_environment(seed=0)

    compressor_fn, _ = get_cpu_codec()
    _, decompressor_fn = get_gpu_codec()

    obs, _ = env.reset(seed=0)
    compressed_obs = compressor_fn(obs.tobytes())
    decompressed_obs = decompressor_fn(compressed_obs)
    pt_obs = torch.from_dlpack(decompressed_obs).clone().view(4, 84, 84)
    assert np.allclose(obs, pt_obs.cpu().numpy())

    rb = make_compressing_replay_buffer(decompressor_fn)

    compression_ratios = []
    num_transitions_in_rollout = 2000

    torch.cuda.synchronize()

    obs, _ = env.reset(seed=0)
    compressed_obs: bytes = compressor_fn(obs.tobytes())
    compressed_nv_obs = nvcomp.as_array(compressed_obs).cuda(synchronize=False)

    start_time = time.time()
    for _ in range(num_transitions_in_rollout):
        # get the torch observation onto the gpu as we would normally do inference here...
        pt_obs = torch.from_numpy(obs).cuda(non_blocking=True)
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        # replay buffer
        compressed_next_obs: bytes = compressor_fn(next_obs.tobytes())
        compressed_nv_next_obs = nvcomp.as_array(compressed_next_obs).cuda(
            synchronize=False
        )

        transition = AtariTransition(
            compressed_nv_obs,
            action,
            compressed_nv_next_obs,
            reward,
            terminated,
            truncated,
            info,
        )
        rb.add(transition)

        # logging
        compression_ratios.append(len(next_obs.tobytes()) / len(compressed_next_obs))

        # reset
        if terminated or truncated:
            obs, _ = env.reset()
            nvcomp.as_array(obs).cuda()
        else:
            obs: np.ndarray = next_obs

    rollout_time = time.time() - start_time

    batch_size = 32
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        rb.sample(batch_size)
    sample_time = time.time() - start_time

    output = [
        "\nListStorage + ReplayBuffer (CPU compress, GPU decompress) Example:",
        f"avg_compression_ratio={np.array(compression_ratios).mean():0.0f}",
        "rollout with zstd, @ transitions/s={:0.0f}".format(
            num_transitions_in_rollout / rollout_time
        ),
        "batch sampling and decompression with zstd @ transitions/s={:0.0f}".format(
            (100 * batch_size) / sample_time
        ),
    ]

    logger.info("\n\t".join(output))


if __name__ == "__main__":
    main()
