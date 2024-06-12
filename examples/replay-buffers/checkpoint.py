# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""An example of a replay buffer being checkpointed at each iteration.

To explore this feature, try replacing the H5StorageCheckpointer with a NestedStorageCheckpointer or a
FlatStorageCheckpointer instance!

"""
import tempfile

import tensordict.utils
import torch

from torchrl.collectors import SyncDataCollector
from torchrl.data import H5StorageCheckpointer, LazyMemmapStorage, ReplayBuffer
from torchrl.envs import GymEnv, SerialEnv

with tempfile.TemporaryDirectory() as path_to_save_dir:
    env = SerialEnv(3, lambda: GymEnv("CartPole-v1", device=None))
    env.set_seed(0)
    torch.manual_seed(0)
    collector = SyncDataCollector(
        env, policy=env.rand_step, total_frames=200, frames_per_batch=22
    )
    rb = ReplayBuffer(storage=LazyMemmapStorage(100, ndim=2))
    rb_test = ReplayBuffer(storage=LazyMemmapStorage(100, ndim=2))
    rb.storage.checkpointer = H5StorageCheckpointer()
    rb_test.storage.checkpointer = H5StorageCheckpointer()
    for data in collector:
        rb.extend(data)
        assert rb._storage.max_size == 102
        rb.dumps(path_to_save_dir)
        rb_test.loads(path_to_save_dir)
        tensordict.assert_allclose_td(rb_test[:], rb[:])
    # Print the directory structure:
    tensordict.utils.print_directory_tree(path_to_save_dir)
