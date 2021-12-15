from torchrl.data.tensor_specs import *
import pytest
import torch


def test_bounded():
    raise NotImplementedError


def test_onehot():
    ts = OneHotDiscreteTensorSpec(10)
    for _ in range(100):
        r = ts.rand()
        ts.to_numpy(r)
        ts.encode(torch.tensor([5]))
        ts.encode(torch.tensor([5]).numpy())
        ts.encode(9)  # out of bounds
        with pytest.raises(RuntimeError):
            ts.encode(torch.tensor([11]))
        assert ts.is_in(r)


def test_unbounded():
    raise NotImplementedError


def test_ndbounded():
    ts = NdBoundedTensorSpec(torch.zeros(10) - 1, torch.ones(10) * 2)
    for _ in range(100):
        r = ts.rand()
        assert (r > -1).all() and (r < 2).all()
        ts.to_numpy(r)
        ts.encode(torch.rand(10) * 3 - 1)
        ts.encode(torch.rand(10).numpy())
        with pytest.raises(AssertionError):
            ts.encode(torch.rand(10) + 3)  # out of bounds
        with pytest.raises(AssertionError):
            ts.to_numpy(torch.rand(10) + 3)  # out of bounds
        assert ts.is_in(r)

def test_ndunbounded():
    raise NotImplementedError


def test_binary():
    raise NotImplementedError


def test_mult_onehot():
    raise NotImplementedError


if __name__ == "__main__":
    pytest.main([__file__])
