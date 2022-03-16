import os.path
import pickle
import tempfile

import numpy as np
import pytest
import torch
from torchrl.data.tensordict.memmap import MemmapTensor


def test_memmap_type():
    array = np.random.rand(1)
    with pytest.raises(
        TypeError, match="convert input to torch.Tensor before calling MemmapTensor"
    ):
        MemmapTensor(array)


def test_grad():
    t = torch.tensor([1.0])
    MemmapTensor(t)
    t = t.requires_grad_()
    with pytest.raises(
        RuntimeError, match="MemmapTensor is incompatible with tensor.requires_grad"
    ):
        MemmapTensor(t)
    with pytest.raises(
        RuntimeError, match="MemmapTensor is incompatible with tensor.requires_grad"
    ):
        MemmapTensor(t + 1)


@pytest.mark.parametrize("dtype", [torch.float, torch.int, torch.double, torch.bool])
@pytest.mark.parametrize(
    "shape",
    [
        [
            2,
        ],
        [1, 2],
    ],
)
def test_memmap_metadata(dtype, shape):
    t = torch.tensor([1, 0]).reshape(shape)
    m = MemmapTensor(t)
    assert m.dtype == t.dtype
    assert (m == t).all()
    assert m.shape == t.shape

    assert m.contiguous().dtype == t.dtype
    assert (m.contiguous() == t).all()
    assert m.contiguous().shape == t.shape

    assert m.clone().dtype == t.dtype
    assert (m.clone() == t).all()
    assert m.clone().shape == t.shape


def test_memmap_del():
    t = torch.tensor([1])
    m = MemmapTensor(t)
    filename = m.filename
    assert os.path.isfile(filename)
    del m
    with pytest.raises(AssertionError):
        assert os.path.isfile(filename)


@pytest.mark.parametrize("value", [True, False])
def test_memmap_ownership(value):
    t = torch.tensor([1])
    m = MemmapTensor(t, transfer_ownership=value)
    assert m.file.delete
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        pickle.dump(m, tmp)
        m2 = pickle.load(open(tmp.name, "rb"))
        assert m2._memmap_array is None  # assert data is not actually loaded
        assert isinstance(m2, MemmapTensor)
        assert m2.filename == m.filename
        assert m2.file.name == m2.filename
        assert m2.file._closer.name == m2.filename
        assert (
            m.file.delete is not m2.file.delete
        )  # delete attributes must have changed
        assert (
            m.file._closer.delete is not m2.file._closer.delete
        )  # delete attributes must have changed
        del m
        if value:
            assert os.path.isfile(m2.filename)
        else:
            # m2 should point to a non-existing file
            assert not os.path.isfile(m2.filename)
            with pytest.raises(FileNotFoundError):
                m2.contiguous()


@pytest.mark.parametrize("value", [True, False])
def test_memmap_ownership_2pass(value):
    t = torch.tensor([1])
    m1 = MemmapTensor(t, transfer_ownership=value)
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp2:
        pickle.dump(m1, tmp2)
        m2 = pickle.load(open(tmp2.name, "rb"))
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp3:
            pickle.dump(m2, tmp3)
            m3 = pickle.load(open(tmp3.name, "rb"))
            assert m1._has_ownership + m2._has_ownership + m3._has_ownership == 1

    del m1, m2, m3
    m1 = MemmapTensor(t, transfer_ownership=value)
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp2:
        pickle.dump(m1, tmp2)
        m2 = pickle.load(open(tmp2.name, "rb"))
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp3:
            pickle.dump(m1, tmp3)
            m3 = pickle.load(open(tmp3.name, "rb"))
            assert m1._has_ownership + m2._has_ownership + m3._has_ownership == 1


def test_memmap_clone():
    t = torch.tensor([1])
    m1 = MemmapTensor(t)
    m2 = m1.clone()
    assert isinstance(m2, MemmapTensor)
    assert m2.filename != m1.filename
    assert m2.filename == m2.file.name
    assert m2.filename == m2.file._closer.name
    m2c = m2.contiguous()
    assert isinstance(m2c, torch.Tensor)
    assert m2c == m1


if __name__ == "__main__":
    pytest.main([__file__, "--capture", "no"])
