import torch
from torch import nn
from torch import multiprocessing as mp
from tensordict import TensorDict


def exec_test(qin, qout):
    mod, super_mod = qin.get()

    # Check relationship between mod and super_mod
    assert mod.weight is super_mod[0].weight
    assert mod.bias is super_mod[0].bias

    qout.put("succeeded")
    print('succeeded')


def exec_test_td(qin, qout):
    td1, td2 = qin.get()

    # Check relationship between td1 and td2
    assert td1["a"] is td2["a"]

    qout.put("succeeded")
    print('succeeded')


if __name__ == "__main__":

    # Create torch module
    mod = nn.Linear(3, 4)

    # Create super module
    super_mod = nn.Sequential(mod)

    # Create multiprocessing queues
    qin = mp.Queue(1)
    qout = mp.Queue(1)

    p = mp.Process(target=exec_test, args=(qin, qout))
    p.start()
    qin.put((mod, super_mod))
    assert qout.get() == "succeeded"
    p.join()

    # with td
    td1 = TensorDict({"a": torch.randn(())}, [])
    td2 = td1.clone(False)
    p = mp.Process(target=exec_test_td, args=(qin, qout))
    p.start()
    qin.put((td1, td2))
    assert qout.get() == "succeeded"
    p.join()
