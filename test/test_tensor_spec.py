# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import numpy as np
import pytest
import torch
from _utils_internal import get_available_devices
from torchrl.data.tensor_specs import (
    NdUnboundedContinuousTensorSpec,
    NdBoundedTensorSpec,
    CompositeSpec,
    MultOneHotDiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,
)
from torchrl.data.tensordict.tensordict import TensorDict, TensorDictBase


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
def test_bounded(dtype):
    torch.manual_seed(0)
    np.random.seed(0)
    for _ in range(100):
        bounds = torch.randn(2).sort()[0]
        ts = BoundedTensorSpec(bounds[0].item(), bounds[1].item(), dtype=dtype)
        _dtype = dtype
        if dtype is None:
            _dtype = torch.get_default_dtype()

        r = ts.rand()
        assert ts.is_in(r)
        assert r.dtype is _dtype
        ts.is_in(ts.encode(bounds.mean()))
        ts.is_in(ts.encode(bounds.mean().item()))
        assert (ts.encode(ts.to_numpy(r)) == r).all()


def test_onehot():
    torch.manual_seed(0)
    np.random.seed(0)

    ts = OneHotDiscreteTensorSpec(10)
    for _ in range(100):
        r = ts.rand()
        ts.to_numpy(r)
        ts.encode(torch.tensor([5]))
        ts.encode(torch.tensor([5]).numpy())
        ts.encode(9)
        with pytest.raises(RuntimeError):
            ts.encode(torch.tensor([11]))  # out of bounds
        assert ts.is_in(r)
        assert (ts.encode(ts.to_numpy(r)) == r).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
def test_unbounded(dtype):
    torch.manual_seed(0)
    np.random.seed(0)
    ts = UnboundedContinuousTensorSpec(dtype=dtype)

    if dtype is None:
        dtype = torch.get_default_dtype()
    for _ in range(100):
        r = ts.rand()
        ts.to_numpy(r)
        assert ts.is_in(r)
        assert r.dtype is dtype
        assert (ts.encode(ts.to_numpy(r)) == r).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        torch.Size(
            [
                3,
            ]
        ),
    ],
)
def test_ndbounded(dtype, shape):
    torch.manual_seed(0)
    np.random.seed(0)

    for _ in range(100):
        lb = torch.rand(10) - 1
        ub = torch.rand(10) + 1
        ts = NdBoundedTensorSpec(lb, ub, dtype=dtype)
        _dtype = dtype
        if dtype is None:
            _dtype = torch.get_default_dtype()

        r = ts.rand(shape)
        assert r.dtype is _dtype
        assert r.shape == torch.Size([*shape, 10])
        assert (r >= lb.to(dtype)).all() and (
            r <= ub.to(dtype)
        ).all(), f"{r[r <= lb] - lb.expand_as(r)[r <= lb]} -- {r[r >= ub] - ub.expand_as(r)[r >= ub]} "
        ts.to_numpy(r)
        assert ts.is_in(r)
        ts.encode(lb + torch.rand(10) * (ub - lb))
        ts.encode((lb + torch.rand(10) * (ub - lb)).numpy())
        assert (ts.encode(ts.to_numpy(r)) == r).all()
        with pytest.raises(AssertionError):
            ts.encode(torch.rand(10) + 3)  # out of bounds
        with pytest.raises(AssertionError):
            ts.to_numpy(torch.rand(10) + 3)  # out of bounds


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
@pytest.mark.parametrize("n", range(3, 10))
@pytest.mark.parametrize(
    "shape",
    [
        [],
        torch.Size(
            [
                3,
            ]
        ),
    ],
)
def test_ndunbounded(dtype, n, shape):
    torch.manual_seed(0)
    np.random.seed(0)

    ts = NdUnboundedContinuousTensorSpec(
        shape=[
            n,
        ],
        dtype=dtype,
    )

    if dtype is None:
        dtype = torch.get_default_dtype()

    for _ in range(100):
        r = ts.rand(shape)
        assert r.shape == torch.Size(
            [
                *shape,
                n,
            ]
        )
        ts.to_numpy(r)
        assert ts.is_in(r)
        assert r.dtype is dtype
        assert (ts.encode(ts.to_numpy(r)) == r).all()


@pytest.mark.parametrize("n", range(3, 10))
@pytest.mark.parametrize(
    "shape",
    [
        [],
        torch.Size(
            [
                3,
            ]
        ),
    ],
)
def test_binary(n, shape):
    torch.manual_seed(0)
    np.random.seed(0)

    ts = BinaryDiscreteTensorSpec(n)
    for _ in range(100):
        r = ts.rand(shape)
        assert r.shape == torch.Size(
            [
                *shape,
                n,
            ]
        )
        assert ts.is_in(r)
        assert ((r == 0) | (r == 1)).all()
        assert (ts.encode(r.numpy()) == r).all()
        assert (ts.encode(ts.to_numpy(r)) == r).all()


@pytest.mark.parametrize(
    "ns",
    [
        [
            5,
        ],
        [5, 2, 3],
        [4, 4, 1],
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [],
        torch.Size(
            [
                3,
            ]
        ),
    ],
)
def test_mult_onehot(shape, ns):
    torch.manual_seed(0)
    np.random.seed(0)
    ts = MultOneHotDiscreteTensorSpec(nvec=ns)
    for _ in range(100):
        r = ts.rand(shape)
        assert r.shape == torch.Size(
            [
                *shape,
                sum(ns),
            ]
        )
        assert ts.is_in(r)
        assert ((r == 0) | (r == 1)).all()
        rsplit = r.split(ns, dim=-1)
        for _r, _n in zip(rsplit, ns):
            assert (_r.sum(-1) == 1).all()
            assert _r.shape[-1] == _n
        np_r = ts.to_numpy(r)
        assert (ts.encode(np_r) == r).all()


@pytest.mark.parametrize("is_complete", [True, False])
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
class TestComposite:
    @staticmethod
    def _composite_spec(is_complete=True, device=None, dtype=None):
        torch.manual_seed(0)
        np.random.seed(0)

        return CompositeSpec(
            obs=NdBoundedTensorSpec(
                torch.zeros(3, 32, 32),
                torch.ones(3, 32, 32),
                dtype=dtype,
                device=device,
            ),
            act=NdUnboundedContinuousTensorSpec((7,), dtype=dtype, device=device)
            if is_complete
            else None,
        )

    def test_getitem(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        assert isinstance(ts["obs"], NdBoundedTensorSpec)
        if is_complete:
            assert isinstance(ts["act"], NdUnboundedContinuousTensorSpec)
        else:
            assert ts["act"] is None
        with pytest.raises(KeyError):
            _ = ts["UNK"]

    def test_setitem_forbidden_keys(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        for key in {"shape", "device", "dtype", "space"}:
            with pytest.raises(AttributeError, match="cannot be set"):
                ts[key] = 42

    @pytest.mark.parametrize("dest", get_available_devices())
    def test_setitem_matches_device(self, is_complete, device, dtype, dest):
        ts = self._composite_spec(is_complete, device, dtype)

        if dest == device:
            ts["good"] = UnboundedContinuousTensorSpec(device=dest, dtype=dtype)
            assert ts["good"].device == dest
        else:
            with pytest.raises(
                RuntimeError, match="All devices of CompositeSpec must match"
            ):
                ts["bad"] = UnboundedContinuousTensorSpec(device=dest, dtype=dtype)

    def test_del(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        assert "obs" in ts.keys()
        assert "act" in ts.keys()
        del ts["obs"]
        assert "obs" not in ts.keys()
        assert "act" in ts.keys()

    def test_encode(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        if dtype is None:
            dtype = torch.get_default_dtype()

        for _ in range(100):
            r = ts.rand()
            raw_vals = {"obs": r["obs"].cpu().numpy()}
            if is_complete:
                raw_vals["act"] = r["act"].cpu().numpy()
            encoded_vals = ts.encode(raw_vals)

            assert encoded_vals["obs"].dtype == dtype
            assert (encoded_vals["obs"] == r["obs"]).all()
            if is_complete:
                assert encoded_vals["act"].dtype == dtype
                assert (encoded_vals["act"] == r["act"]).all()

    def test_is_in(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        for _ in range(100):
            r = ts.rand()
            assert ts.is_in(r)

    @pytest.mark.parametrize("shape", [[], [3]])
    def test_project(self, is_complete, device, dtype, shape):
        ts = self._composite_spec(is_complete, device, dtype)
        # Using normal distribution to get out of bounds
        tensors = {"obs": torch.randn(*shape, 3, 32, 32, dtype=dtype, device=device)}
        if is_complete:
            tensors["act"] = torch.randn(*shape, 7, dtype=dtype, device=device)
        out_of_bounds_td = TensorDict(tensors, batch_size=shape)

        assert not ts.is_in(out_of_bounds_td)
        ts.project(out_of_bounds_td)
        assert ts.is_in(out_of_bounds_td)
        assert out_of_bounds_td.shape == torch.Size(shape)

    @pytest.mark.parametrize("shape", [[], [3]])
    def test_rand(self, is_complete, device, dtype, shape):
        ts = self._composite_spec(is_complete, device, dtype)
        if dtype is None:
            dtype = torch.get_default_dtype()

        rand_td = ts.rand(shape)
        assert rand_td.shape == torch.Size(shape)
        assert rand_td.get("obs").shape == torch.Size([*shape, 3, 32, 32])
        assert rand_td.get("obs").dtype == dtype
        if is_complete:
            assert rand_td.get("act").shape == torch.Size([*shape, 7])
            assert rand_td.get("act").dtype == dtype

    def test_repr(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        output = repr(ts)
        assert output.startswith("CompositeSpec")
        assert "obs: " in output
        assert "act: " in output

    def test_device_cast_with_dtype_fails(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        with pytest.raises(ValueError, match="Only device casting is allowed"):
            ts.to(torch.float16)

    @pytest.mark.parametrize("dest", get_available_devices())
    def test_device_cast(self, is_complete, device, dtype, dest):
        # Note: trivial test in case there is only one device available.
        ts = self._composite_spec(is_complete, device, dtype)
        original_r = ts.rand()
        ts.to(dest)
        cast_r = ts.rand()

        assert ts.device == dest
        assert cast_r["obs"].device == dest
        if is_complete:
            assert cast_r["act"].device == dest

    def test_type_check(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        rand_td = ts.rand()
        ts.type_check(rand_td)
        ts.type_check(rand_td["obs"], "obs")
        if is_complete:
            ts.type_check(rand_td["act"], "act")

    def test_nested_composite_spec(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        td = ts.rand()
        assert isinstance(td["nested_cp"], TensorDictBase)
        keys = list(td.keys())
        for key in keys:
            if key != "nested_cp":
                assert key in td["nested_cp"].keys()


class TestEquality:
    @staticmethod
    def _ts_make_all_fields_equal(ts_to, ts_from):
        ts_to.shape = ts_from.shape
        ts_to.space = ts_from.space
        ts_to.device = ts_from.device
        ts_to.dtype = ts_from.dtype
        ts_to.domain = ts_from.domain
        return ts_to

    def test_equality_bounded(self):
        MIN = 10
        MAX = 100
        DEVICE = "cpu"
        DTYPE = torch.float16

        ts = BoundedTensorSpec(MIN, MAX, DEVICE, DTYPE)

        ts_same = BoundedTensorSpec(MIN, MAX, DEVICE, DTYPE)
        assert ts == ts_same

        ts_other = BoundedTensorSpec(MIN + 1, MAX, DEVICE, DTYPE)
        assert ts != ts_other

        ts_other = BoundedTensorSpec(MIN, MAX + 1, DEVICE, DTYPE)
        assert ts != ts_other

        ts_other = BoundedTensorSpec(MIN, MAX, "cpu:2", DTYPE)
        assert ts != ts_other

        ts_other = BoundedTensorSpec(MIN, MAX, DEVICE, torch.float64)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            UnboundedContinuousTensorSpec(DEVICE, DTYPE), ts
        )
        assert ts != ts_other

    def test_equality_onehot(self):
        N = 5
        DEVICE = "cpu"
        DTYPE = torch.float16
        USER_REGISTER = False

        ts = OneHotDiscreteTensorSpec(N, DEVICE, DTYPE, USER_REGISTER)

        ts_same = OneHotDiscreteTensorSpec(N, DEVICE, DTYPE, USER_REGISTER)
        assert ts == ts_same

        ts_other = OneHotDiscreteTensorSpec(N + 1, DEVICE, DTYPE, USER_REGISTER)
        assert ts != ts_other

        ts_other = OneHotDiscreteTensorSpec(N, "cpu:2", DTYPE, USER_REGISTER)
        assert ts != ts_other

        ts_other = OneHotDiscreteTensorSpec(N, DEVICE, torch.float64, USER_REGISTER)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            UnboundedContinuousTensorSpec(DEVICE, DTYPE), ts
        )
        assert ts != ts_other

    def test_equality_unbounded(self):
        DEVICE = "cpu"
        DTYPE = torch.float16

        ts = UnboundedContinuousTensorSpec(DEVICE, DTYPE)

        ts_same = UnboundedContinuousTensorSpec(DEVICE, DTYPE)
        assert ts == ts_same

        ts_other = UnboundedContinuousTensorSpec("cpu:2", DTYPE)
        assert ts != ts_other

        ts_other = UnboundedContinuousTensorSpec(DEVICE, torch.float64)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, DEVICE, DTYPE), ts
        )
        assert ts != ts_other

    def test_equality_ndbounded(self):
        MIN = np.arange(12).reshape((3, 4))
        MAX = MIN + 100
        DEVICE = "cpu"
        DTYPE = torch.float16

        ts = NdBoundedTensorSpec(minimum=MIN, maximum=MAX, device=DEVICE, dtype=DTYPE)

        ts_same = NdBoundedTensorSpec(
            minimum=MIN, maximum=MAX, device=DEVICE, dtype=DTYPE
        )
        assert ts == ts_same

        ts_other = NdBoundedTensorSpec(
            minimum=MIN + 1, maximum=MAX, device=DEVICE, dtype=DTYPE
        )
        assert ts != ts_other

        ts_other = NdBoundedTensorSpec(
            minimum=MIN, maximum=MAX + 1, device=DEVICE, dtype=DTYPE
        )
        assert ts != ts_other

        ts_other = NdBoundedTensorSpec(
            minimum=MIN, maximum=MAX, device="cpu:2", dtype=DTYPE
        )
        assert ts != ts_other

        ts_other = NdBoundedTensorSpec(
            minimum=MIN, maximum=MAX, device=DEVICE, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, DEVICE, DTYPE), ts
        )
        assert ts != ts_other

    @pytest.mark.parametrize(
        "shape",
        [
            3,
            torch.Size([4]),
            torch.Size([5, 6]),
        ],
    )
    def test_equality_ndunbounded(self, shape):
        DEVICE = "cpu"
        DTYPE = torch.float16

        ts = NdUnboundedContinuousTensorSpec(shape=shape, device=DEVICE, dtype=DTYPE)

        ts_same = NdUnboundedContinuousTensorSpec(
            shape=shape, device=DEVICE, dtype=DTYPE
        )
        assert ts == ts_same

        other_shape = 13 if type(shape) == int else torch.Size(np.array(shape) + 10)
        ts_other = NdUnboundedContinuousTensorSpec(
            shape=other_shape, device=DEVICE, dtype=DTYPE
        )
        assert ts != ts_other

        ts_other = NdUnboundedContinuousTensorSpec(
            shape=shape, device="cpu:2", dtype=DTYPE
        )
        assert ts != ts_other

        ts_other = NdUnboundedContinuousTensorSpec(
            shape=shape, device=DEVICE, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, DEVICE, DTYPE), ts
        )
        assert ts != ts_other

    def test_equality_binary(self):
        N = 5
        DEVICE = "cpu"
        DTYPE = torch.float16

        ts = BinaryDiscreteTensorSpec(n=N, device=DEVICE, dtype=DTYPE)

        ts_same = BinaryDiscreteTensorSpec(n=N, device=DEVICE, dtype=DTYPE)
        assert ts == ts_same

        ts_other = BinaryDiscreteTensorSpec(n=N + 5, device=DEVICE, dtype=DTYPE)
        assert ts != ts_other

        ts_other = BinaryDiscreteTensorSpec(n=N, device="cpu:2", dtype=DTYPE)
        assert ts != ts_other

        ts_other = BinaryDiscreteTensorSpec(n=N, device=DEVICE, dtype=torch.float64)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, DEVICE, DTYPE), ts
        )
        assert ts != ts_other

    @pytest.mark.parametrize("nvec", [[3], [3, 4], [3, 4, 5]])
    def test_equality_multi_onehot(self, nvec):
        N = 5
        DEVICE = "cpu"
        DTYPE = torch.float16

        ts = MultOneHotDiscreteTensorSpec(nvec=nvec, device=DEVICE, dtype=DTYPE)

        ts_same = MultOneHotDiscreteTensorSpec(nvec=nvec, device=DEVICE, dtype=DTYPE)
        assert ts == ts_same

        other_nvec = np.array(nvec) + 3
        ts_other = MultOneHotDiscreteTensorSpec(
            nvec=other_nvec, device=DEVICE, dtype=DTYPE
        )
        assert ts != ts_other

        other_nvec = [12]
        ts_other = MultOneHotDiscreteTensorSpec(
            nvec=other_nvec, device=DEVICE, dtype=DTYPE
        )
        assert ts != ts_other

        other_nvec = [12, 13]
        ts_other = MultOneHotDiscreteTensorSpec(
            nvec=other_nvec, device=DEVICE, dtype=DTYPE
        )
        assert ts != ts_other

        ts_other = MultOneHotDiscreteTensorSpec(nvec=nvec, device="cpu:2", dtype=DTYPE)
        assert ts != ts_other

        ts_other = MultOneHotDiscreteTensorSpec(
            nvec=nvec, device=DEVICE, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, DEVICE, DTYPE), ts
        )
        assert ts != ts_other


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
