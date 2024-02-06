# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import time
import warnings

import pytest
import torch
from tensordict import LazyStackedTensorDict, TensorDict
from torch import multiprocessing as mp
from torchrl._utils import logger as torchrl_logger


class TestShared:
    @staticmethod
    def remote_process(command_pipe_child, command_pipe_parent, tensordict):
        command_pipe_parent.close()
        assert tensordict.is_shared()
        t0 = time.time()
        tensordict.zero_()
        torchrl_logger.info(f"zeroing time: {time.time() - t0}")
        command_pipe_child.send("done")
        command_pipe_child.close()
        del command_pipe_child, command_pipe_parent, tensordict

    @staticmethod
    def driver_func(subtd, td):
        assert subtd.is_shared()
        command_pipe_parent, command_pipe_child = mp.Pipe()
        proc = mp.Process(
            target=TestShared.remote_process,
            args=(command_pipe_child, command_pipe_parent, subtd),
        )
        proc.start()
        command_pipe_child.close()
        command_pipe_parent.recv()
        for item in subtd.values():
            assert (item == 0).all()

        for item in td[0].values():
            assert (item == 0).all()
        command_pipe_parent.close()
        proc.join()
        del command_pipe_child, command_pipe_parent, proc

    @pytest.mark.parametrize("indexing_method", range(3))
    def test_shared(self, indexing_method):
        torch.manual_seed(0)
        tensordict = TensorDict(
            source={
                "a": torch.randn(1000, 200),
                "b": torch.randn(1000, 100),
                "done": torch.zeros(1000, 100, dtype=torch.bool).bernoulli_(),
            },
            batch_size=[1000],
        )

        td = tensordict.clone().share_memory_()
        if indexing_method == 0:
            subtd = TensorDict(
                source={key: item[0] for key, item in td.items()},
                batch_size=[],
            ).share_memory_()
        elif indexing_method == 1:
            subtd = td.get_sub_tensordict(0)
        elif indexing_method == 2:
            subtd = td[0]
        else:
            raise NotImplementedError

        assert subtd.is_shared()

        self.driver_func(subtd, td)


class TestStack:
    @staticmethod
    def remote_process(command_pipe_child, command_pipe_parent, tensordict):
        command_pipe_parent.close()
        assert isinstance(tensordict, TensorDict), f"td is of type {type(tensordict)}"
        assert tensordict.is_shared() or tensordict.is_memmap()
        new_tensordict = LazyStackedTensorDict.lazy_stack(
            [
                tensordict[i].contiguous().clone().zero_()
                for i in range(tensordict.shape[0])
            ],
            0,
        )
        cmd = command_pipe_child.recv()
        t0 = time.time()
        if cmd == "stack":
            tensordict.copy_(new_tensordict)
        elif cmd == "serial":
            for i, td in enumerate(new_tensordict.tensordicts):
                tensordict.update_at_(td, i)
        time_spent = time.time() - t0
        command_pipe_child.send(time_spent)
        command_pipe_child.close()
        del command_pipe_child, command_pipe_parent

    @staticmethod
    def driver_func(td, stack):

        command_pipe_parent, command_pipe_child = mp.Pipe()
        proc = mp.Process(
            target=TestStack.remote_process,
            args=(command_pipe_child, command_pipe_parent, td),
        )
        proc.start()
        command_pipe_child.close()
        command_pipe_parent.send("stack" if stack else "serial")
        time_spent = command_pipe_parent.recv()
        torchrl_logger.info(f"stack {stack}: time={time_spent}")
        for item in td.values():
            assert (item == 0).all()
        proc.join()
        command_pipe_parent.close()
        return time_spent

    @pytest.mark.parametrize("shared", ["shared", "memmap"])
    def test_shared(self, shared):
        torchrl_logger.info(f"test_shared: shared={shared}")
        torch.manual_seed(0)
        tensordict = TensorDict(
            source={
                "a": torch.randn(100, 2),
                "b": torch.randn(100, 1),
                "done": torch.zeros(100, 1, dtype=torch.bool).bernoulli_(),
            },
            batch_size=[100],
        )
        if shared == "shared":
            tensordict.share_memory_()
        else:
            tensordict.memmap_()
        t_true = self.driver_func(tensordict, True)
        t_false = self.driver_func(tensordict, False)
        if t_true > t_false:
            warnings.warn(
                "Updating each element of the tensordict did "
                "not take longer than updating the stack."
            )


@pytest.mark.parametrize("idx", [0, slice(200)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bool])
def test_memmap(idx, dtype, large_scale=False):
    N = 5000 if large_scale else 10
    H = 128 if large_scale else 8
    td = TensorDict(
        source={
            "a": torch.zeros(N, 3, H, H, dtype=dtype),
            "b": torch.zeros(N, 3, H, H, dtype=dtype),
            "c": torch.zeros(N, 3, H, H, dtype=dtype),
        },
        batch_size=[
            N,
        ],
    )

    td_sm = td.clone().share_memory_()
    td_memmap = td.clone().memmap_()

    torchrl_logger.info("\nTesting reading from TD")
    for i in range(2):
        t0 = time.time()
        td_sm[idx].clone()
        if i == 1:
            torchrl_logger.info(f"sm: {time.time() - t0:4.4f} sec")

        t0 = time.time()
        td_memmap[idx].clone()
        if i == 1:
            torchrl_logger.info(f"memmap: {time.time() - t0:4.4f} sec")

    td_to_copy = td[idx].contiguous()
    for k in td_to_copy.keys():
        td_to_copy.set_(k, torch.ones_like(td_to_copy.get(k)))

    torchrl_logger.info("\nTesting writing to TD")
    for i in range(2):
        t0 = time.time()
        sub_td_sm = td_sm.get_sub_tensordict(idx)
        sub_td_sm.update_(td_to_copy)
        if i == 1:
            torchrl_logger.info(f"sm td: {time.time() - t0:4.4f} sec")
        torch.testing.assert_close(sub_td_sm.get("a"), td_to_copy.get("a"))

        t0 = time.time()
        sub_td_sm = td_memmap.get_sub_tensordict(idx)
        sub_td_sm.update_(td_to_copy)
        if i == 1:
            torchrl_logger.info(f"memmap td: {time.time() - t0:4.4f} sec")
        torch.testing.assert_close(sub_td_sm.get("a")._tensor, td_to_copy.get("a"))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
