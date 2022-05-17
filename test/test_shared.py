# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import warnings

import pytest
import torch
from torch import multiprocessing as mp
from torchrl.data import SavedTensorDict
from torchrl.data import TensorDict


class TestShared:
    @staticmethod
    def remote_process(command_pipe_child, command_pipe_parent, tensordict):
        command_pipe_parent.close()
        assert tensordict.is_shared()
        t0 = time.time()
        tensordict.zero_()
        print(f"zeroing time: {time.time() - t0}")
        command_pipe_child.send("done")

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
        for key, item in subtd.items():
            assert (item == 0).all()

        for key, item in td[0].items():
            assert (item == 0).all()
        proc.join()
        command_pipe_parent.close()

    def test_shared(self):
        torch.manual_seed(0)
        tensordict = TensorDict(
            source={
                "a": torch.randn(1000, 200),
                "b": torch.randn(1000, 100),
                "done": torch.zeros(1000, 100, dtype=torch.bool).bernoulli_(),
            },
            batch_size=[1000],
        )

        td1 = tensordict.clone().share_memory_()
        td2 = tensordict.clone().share_memory_()
        td3 = tensordict.clone().share_memory_()
        subtd2 = TensorDict(
            source={key: item[0] for key, item in td2.items()}, batch_size=[]
        )
        assert subtd2.is_shared()
        print("sub td2 is shared: ", subtd2.is_shared())

        subtd1 = td1.get_sub_tensordict(0)
        t0 = time.time()
        self.driver_func(subtd1, td1)
        t_elapsed = time.time() - t0
        print(f"execution on subtd: {t_elapsed}")

        t0 = time.time()
        self.driver_func(subtd2, td2)
        t_elapsed = time.time() - t0
        print(f"execution on plain td: {t_elapsed}")

        subtd3 = td3[0]
        t0 = time.time()
        self.driver_func(subtd3, td3)
        t_elapsed = time.time() - t0
        print(f"execution on regular indexed td: {t_elapsed}")


class TestStack:
    @staticmethod
    def remote_process(command_pipe_child, command_pipe_parent, tensordict):
        command_pipe_parent.close()
        assert isinstance(tensordict, TensorDict), f"td is of type {type(tensordict)}"
        assert tensordict.is_shared() or tensordict.is_memmap()
        new_tensordict = torch.stack(
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
        print(f"stack {stack}: time={time_spent}")
        for key, item in td.items():
            assert (item == 0).all()
        proc.join()
        command_pipe_parent.close()
        return time_spent

    @pytest.mark.parametrize("shared", ["shared", "memmap"])
    def test_shared(self, shared):
        print(f"test_shared: shared={shared}")
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


@pytest.mark.parametrize(
    "idx",
    [
        torch.tensor(
            [
                3,
                5,
                7,
                8,
            ]
        ),
        slice(200),
    ],
)
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
    td_saved = td.to(SavedTensorDict)

    print("\nTesting reading from TD")
    for i in range(2):
        t0 = time.time()
        td_sm[idx].clone()
        if i == 1:
            print(f"sm: {time.time() - t0:4.4f} sec")

        t0 = time.time()
        td_memmap[idx].clone()
        if i == 1:
            print(f"memmap: {time.time() - t0:4.4f} sec")

        t0 = time.time()
        td_saved[idx].clone()
        if i == 1:
            print(f"saved td: {time.time() - t0:4.4f} sec")

    td_to_copy = td[idx].contiguous()
    for k in td_to_copy.keys():
        td_to_copy.set_(k, torch.ones_like(td_to_copy.get(k)))

    print("\nTesting writing to TD")
    for i in range(2):
        t0 = time.time()
        td_sm[idx].update_(td_to_copy)
        if i == 1:
            print(f"sm td: {time.time() - t0:4.4f} sec")
        torch.testing.assert_allclose(td_sm[idx].get("a"), td_to_copy.get("a"))

        t0 = time.time()
        td_memmap[idx].update_(td_to_copy)
        if i == 1:
            print(f"memmap td: {time.time() - t0:4.4f} sec")
        torch.testing.assert_allclose(td_memmap[idx].get("a"), td_to_copy.get("a"))

        t0 = time.time()
        td_saved[idx].update_(td_to_copy)
        if i == 1:
            print(f"saved td: {time.time() - t0:4.4f} sec")
        torch.testing.assert_allclose(td_saved[idx].get("a"), td_to_copy.get("a"))


if __name__ == "__main__":
    pytest.main([__file__, "--capture", "no"])
