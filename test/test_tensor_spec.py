# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import numpy as np
import pytest
import torch
from _utils_internal import get_available_devices
from scipy.stats import chisquare
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    _keys_to_empty_composite_spec,
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultOneHotDiscreteTensorSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)


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


@pytest.mark.parametrize("cls", [OneHotDiscreteTensorSpec, DiscreteTensorSpec])
def test_discrete(cls):
    torch.manual_seed(0)
    np.random.seed(0)

    ts = cls(10)
    for _ in range(100):
        r = ts.rand()
        ts.to_numpy(r)
        ts.encode(torch.tensor([5]))
        ts.encode(torch.tensor(5).numpy())
        ts.encode(9)
        with pytest.raises(AssertionError):
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
        ts.rand()
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
        assert set(ts.keys()) == {
            "obs",
            "act",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
        }
        assert len(ts.keys()) == len(ts.keys(yield_nesting_keys=True)) - 1
        assert set(ts.keys(yield_nesting_keys=True)) == {
            "obs",
            "act",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
            "nested_cp",
        }
        td = ts.rand()
        assert isinstance(td["nested_cp"], TensorDictBase)
        keys = list(td.keys())
        for key in keys:
            if key != "nested_cp":
                assert key in td["nested_cp"].keys()

    def test_nested_composite_spec_index(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"]["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        assert ts["nested_cp"]["nested_cp"] is ts["nested_cp", "nested_cp"]
        assert (
            ts["nested_cp"]["nested_cp"]["obs"] is ts["nested_cp", "nested_cp", "obs"]
        )

    def test_nested_composite_spec_rand(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"]["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        r = ts.rand()
        assert (r["nested_cp", "nested_cp", "obs"] >= 0).all()

    def test_nested_composite_spec_zero(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"]["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        r = ts.zero()
        assert (r["nested_cp", "nested_cp", "obs"] == 0).all()

    def test_nested_composite_spec_setitem(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"]["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp", "nested_cp", "obs"] = None
        assert (
            ts["nested_cp"]["nested_cp"]["obs"] is ts["nested_cp", "nested_cp", "obs"]
        )
        assert ts["nested_cp"]["nested_cp"]["obs"] is None

    def test_nested_composite_spec_update(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        td2 = CompositeSpec(new=None)
        ts.update(td2)
        assert set(ts.keys()) == {
            "obs",
            "act",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
            "new",
        }

        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        td2 = CompositeSpec(nested_cp=CompositeSpec(new=None).to(device))
        ts.update(td2)
        assert set(ts.keys()) == {
            "obs",
            "act",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
            ("nested_cp", "new"),
        }

        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        td2 = CompositeSpec(nested_cp=CompositeSpec(act=None).to(device))
        ts.update(td2)
        assert set(ts.keys()) == {
            "obs",
            "act",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
        }
        assert ts["nested_cp"]["act"] is None

        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        td2 = CompositeSpec(nested_cp=CompositeSpec(act=None).to(device))
        ts.update(td2)
        td2 = CompositeSpec(
            nested_cp=CompositeSpec(act=UnboundedContinuousTensorSpec(device))
        )
        ts.update(td2)
        assert set(ts.keys()) == {
            "obs",
            "act",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
        }
        assert ts["nested_cp"]["act"] is not None


def test_keys_to_empty_composite_spec():
    keys = [("key1", "out"), ("key1", "in"), "key2", ("key1", "subkey1", "subkey2")]
    composite = _keys_to_empty_composite_spec(keys)
    assert set(composite.keys()) == set(keys)


class TestEquality:
    """Tests spec comparison."""

    @staticmethod
    def _ts_make_all_fields_equal(ts_to, ts_from):
        ts_to.shape = ts_from.shape
        ts_to.space = ts_from.space
        ts_to.device = ts_from.device
        ts_to.dtype = ts_from.dtype
        ts_to.domain = ts_from.domain
        return ts_to

    def test_equality_bounded(self):
        minimum = 10
        maximum = 100
        device = "cpu"
        dtype = torch.float16

        ts = BoundedTensorSpec(minimum, maximum, device, dtype)

        ts_same = BoundedTensorSpec(minimum, maximum, device, dtype)
        assert ts == ts_same

        ts_other = BoundedTensorSpec(minimum + 1, maximum, device, dtype)
        assert ts != ts_other

        ts_other = BoundedTensorSpec(minimum, maximum + 1, device, dtype)
        assert ts != ts_other

        ts_other = BoundedTensorSpec(minimum, maximum, "cpu:0", dtype)
        assert ts != ts_other

        ts_other = BoundedTensorSpec(minimum, maximum, device, torch.float64)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            UnboundedContinuousTensorSpec(device, dtype), ts
        )
        assert ts != ts_other

    def test_equality_onehot(self):
        n = 5
        device = "cpu"
        dtype = torch.float16
        use_register = False

        ts = OneHotDiscreteTensorSpec(n, device, dtype, use_register)

        ts_same = OneHotDiscreteTensorSpec(n, device, dtype, use_register)
        assert ts == ts_same

        ts_other = OneHotDiscreteTensorSpec(n + 1, device, dtype, use_register)
        assert ts != ts_other

        ts_other = OneHotDiscreteTensorSpec(n, "cpu:0", dtype, use_register)
        assert ts != ts_other

        ts_other = OneHotDiscreteTensorSpec(n, device, torch.float64, use_register)
        assert ts != ts_other

        ts_other = OneHotDiscreteTensorSpec(n, device, dtype, not use_register)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            UnboundedContinuousTensorSpec(device, dtype), ts
        )
        assert ts != ts_other

    def test_equality_unbounded(self):
        device = "cpu"
        dtype = torch.float16

        ts = UnboundedContinuousTensorSpec(device, dtype)

        ts_same = UnboundedContinuousTensorSpec(device, dtype)
        assert ts == ts_same

        ts_other = UnboundedContinuousTensorSpec("cpu:0", dtype)
        assert ts != ts_other

        ts_other = UnboundedContinuousTensorSpec(device, torch.float64)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, device, dtype), ts
        )
        assert ts != ts_other

    def test_equality_ndbounded(self):
        minimum = np.arange(12).reshape((3, 4))
        maximum = minimum + 100
        device = "cpu"
        dtype = torch.float16

        ts = NdBoundedTensorSpec(
            minimum=minimum, maximum=maximum, device=device, dtype=dtype
        )

        ts_same = NdBoundedTensorSpec(
            minimum=minimum, maximum=maximum, device=device, dtype=dtype
        )
        assert ts == ts_same

        ts_other = NdBoundedTensorSpec(
            minimum=minimum + 1, maximum=maximum, device=device, dtype=dtype
        )
        assert ts != ts_other

        ts_other = NdBoundedTensorSpec(
            minimum=minimum, maximum=maximum + 1, device=device, dtype=dtype
        )
        assert ts != ts_other

        ts_other = NdBoundedTensorSpec(
            minimum=minimum, maximum=maximum, device="cpu:0", dtype=dtype
        )
        assert ts != ts_other

        ts_other = NdBoundedTensorSpec(
            minimum=minimum, maximum=maximum, device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, device, dtype), ts
        )
        assert ts != ts_other

    def test_equality_discrete(self):
        n = 5
        shape = torch.Size([1])
        device = "cpu"
        dtype = torch.float16

        ts = DiscreteTensorSpec(n, shape, device, dtype)

        ts_same = DiscreteTensorSpec(n, shape, device, dtype)
        assert ts == ts_same

        ts_other = DiscreteTensorSpec(n + 1, shape, device, dtype)
        assert ts != ts_other

        ts_other = DiscreteTensorSpec(n, shape, "cpu:0", dtype)
        assert ts != ts_other

        ts_other = DiscreteTensorSpec(n, shape, device, torch.float64)
        assert ts != ts_other

        ts_other = DiscreteTensorSpec(n, torch.Size([2]), device, torch.float64)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            UnboundedContinuousTensorSpec(device, dtype), ts
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
        device = "cpu"
        dtype = torch.float16

        ts = NdUnboundedContinuousTensorSpec(shape=shape, device=device, dtype=dtype)

        ts_same = NdUnboundedContinuousTensorSpec(
            shape=shape, device=device, dtype=dtype
        )
        assert ts == ts_same

        other_shape = 13 if type(shape) == int else torch.Size(np.array(shape) + 10)
        ts_other = NdUnboundedContinuousTensorSpec(
            shape=other_shape, device=device, dtype=dtype
        )
        assert ts != ts_other

        ts_other = NdUnboundedContinuousTensorSpec(
            shape=shape, device="cpu:0", dtype=dtype
        )
        assert ts != ts_other

        ts_other = NdUnboundedContinuousTensorSpec(
            shape=shape, device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, device, dtype), ts
        )
        assert ts != ts_other

    def test_equality_binary(self):
        n = 5
        device = "cpu"
        dtype = torch.float16

        ts = BinaryDiscreteTensorSpec(n=n, device=device, dtype=dtype)

        ts_same = BinaryDiscreteTensorSpec(n=n, device=device, dtype=dtype)
        assert ts == ts_same

        ts_other = BinaryDiscreteTensorSpec(n=n + 5, device=device, dtype=dtype)
        assert ts != ts_other

        ts_other = BinaryDiscreteTensorSpec(n=n, device="cpu:0", dtype=dtype)
        assert ts != ts_other

        ts_other = BinaryDiscreteTensorSpec(n=n, device=device, dtype=torch.float64)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, device, dtype), ts
        )
        assert ts != ts_other

    @pytest.mark.parametrize("nvec", [[3], [3, 4], [3, 4, 5]])
    def test_equality_multi_onehot(self, nvec):
        device = "cpu"
        dtype = torch.float16

        ts = MultOneHotDiscreteTensorSpec(nvec=nvec, device=device, dtype=dtype)

        ts_same = MultOneHotDiscreteTensorSpec(nvec=nvec, device=device, dtype=dtype)
        assert ts == ts_same

        other_nvec = np.array(nvec) + 3
        ts_other = MultOneHotDiscreteTensorSpec(
            nvec=other_nvec, device=device, dtype=dtype
        )
        assert ts != ts_other

        other_nvec = [12]
        ts_other = MultOneHotDiscreteTensorSpec(
            nvec=other_nvec, device=device, dtype=dtype
        )
        assert ts != ts_other

        other_nvec = [12, 13]
        ts_other = MultOneHotDiscreteTensorSpec(
            nvec=other_nvec, device=device, dtype=dtype
        )
        assert ts != ts_other

        ts_other = MultOneHotDiscreteTensorSpec(nvec=nvec, device="cpu:0", dtype=dtype)
        assert ts != ts_other

        ts_other = MultOneHotDiscreteTensorSpec(
            nvec=nvec, device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, device, dtype), ts
        )
        assert ts != ts_other

    def test_equality_composite(self):
        minimum = np.arange(12).reshape((3, 4))
        maximum = minimum + 100
        device = "cpu"
        dtype = torch.float16

        bounded = BoundedTensorSpec(0, 1, device, dtype)
        bounded_same = BoundedTensorSpec(0, 1, device, dtype)
        bounded_other = BoundedTensorSpec(0, 2, device, dtype)

        nd = NdBoundedTensorSpec(
            minimum=minimum, maximum=maximum + 1, device=device, dtype=dtype
        )
        nd_same = NdBoundedTensorSpec(
            minimum=minimum, maximum=maximum + 1, device=device, dtype=dtype
        )
        _ = NdBoundedTensorSpec(
            minimum=minimum, maximum=maximum + 3, device=device, dtype=dtype
        )

        # Equality tests
        ts = CompositeSpec(ts1=bounded)
        ts_same = CompositeSpec(ts1=bounded)
        assert ts == ts_same

        ts = CompositeSpec(ts1=bounded)
        ts_same = CompositeSpec(ts1=bounded_same)
        assert ts == ts_same

        ts = CompositeSpec(ts1=bounded, ts2=nd)
        ts_same = CompositeSpec(ts1=bounded, ts2=nd)
        assert ts == ts_same

        ts = CompositeSpec(ts1=bounded, ts2=nd)
        ts_same = CompositeSpec(ts1=bounded_same, ts2=nd_same)
        assert ts == ts_same

        ts = CompositeSpec(ts1=bounded, ts2=nd)
        ts_same = CompositeSpec(ts2=nd_same, ts1=bounded_same)
        assert ts == ts_same

        # Inequality tests
        ts = CompositeSpec(ts1=bounded)
        ts_other = CompositeSpec(ts5=bounded)
        assert ts != ts_other

        ts = CompositeSpec(ts1=bounded)
        ts_other = CompositeSpec(ts1=bounded_other)
        assert ts != ts_other

        ts = CompositeSpec(ts1=bounded)
        ts_other = CompositeSpec(ts1=nd)
        assert ts != ts_other

        ts = CompositeSpec(ts1=bounded)
        ts_other = CompositeSpec(ts1=bounded, ts2=nd)
        assert ts != ts_other

        ts = CompositeSpec(ts1=bounded, ts2=nd)
        ts_other = CompositeSpec(ts2=nd)
        assert ts != ts_other

        ts = CompositeSpec(ts1=bounded, ts2=nd)
        ts_other = CompositeSpec(ts1=bounded, ts2=nd, ts3=bounded_other)
        assert ts != ts_other


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)


class TestSpec:
    @pytest.mark.parametrize(
        "action_spec_cls", [OneHotDiscreteTensorSpec, DiscreteTensorSpec]
    )
    def test_discrete_action_spec_reconstruct(self, action_spec_cls):
        torch.manual_seed(0)
        action_spec = action_spec_cls(10)

        actions_tensors = [action_spec.rand() for _ in range(10)]
        actions_numpy = [action_spec.to_numpy(a) for a in actions_tensors]
        actions_tensors_2 = [action_spec.encode(a) for a in actions_numpy]
        assert all(
            [(a1 == a2).all() for a1, a2 in zip(actions_tensors, actions_tensors_2)]
        )

        actions_numpy = [int(np.random.randint(0, 10, (1,))) for a in actions_tensors]
        actions_tensors = [action_spec.encode(a) for a in actions_numpy]
        actions_numpy_2 = [action_spec.to_numpy(a) for a in actions_tensors]
        assert all([(a1 == a2) for a1, a2 in zip(actions_numpy, actions_numpy_2)])

    def test_mult_discrete_action_spec_reconstruct(self):
        torch.manual_seed(0)
        action_spec = MultOneHotDiscreteTensorSpec((10, 5))

        actions_tensors = [action_spec.rand() for _ in range(10)]
        actions_numpy = [action_spec.to_numpy(a) for a in actions_tensors]
        actions_tensors_2 = [action_spec.encode(a) for a in actions_numpy]
        assert all(
            [(a1 == a2).all() for a1, a2 in zip(actions_tensors, actions_tensors_2)]
        )

        actions_numpy = [
            np.concatenate(
                [np.random.randint(0, 10, (1,)), np.random.randint(0, 5, (1,))], 0
            )
            for a in actions_tensors
        ]
        actions_tensors = [action_spec.encode(a) for a in actions_numpy]
        actions_numpy_2 = [action_spec.to_numpy(a) for a in actions_tensors]
        assert all([(a1 == a2).all() for a1, a2 in zip(actions_numpy, actions_numpy_2)])

    def test_one_hot_discrete_action_spec_rand(self):
        torch.manual_seed(0)
        action_spec = OneHotDiscreteTensorSpec(10)

        sample = torch.stack([action_spec.rand() for _ in range(10000)], 0)

        sample_list = sample.argmax(-1)
        sample_list = [sum(sample_list == i).item() for i in range(10)]
        assert chisquare(sample_list).pvalue > 0.1

        sample = action_spec.to_numpy(sample)
        sample = [sum(sample == i) for i in range(10)]
        assert chisquare(sample).pvalue > 0.1

    def test_categorical_action_spec_rand(self):
        torch.manual_seed(1)
        action_spec = DiscreteTensorSpec(10)

        sample = action_spec.rand((10000,))

        sample_list = sample
        sample_list = [sum(sample_list == i).item() for i in range(10)]
        assert chisquare(sample_list).pvalue > 0.1

        sample = action_spec.to_numpy(sample)
        sample = [sum(sample == i) for i in range(10)]
        assert chisquare(sample).pvalue > 0.1

    def test_mult_discrete_action_spec_rand(self):
        torch.manual_seed(0)
        ns = (10, 5)
        N = 100000
        action_spec = MultOneHotDiscreteTensorSpec((10, 5))

        actions_tensors = [action_spec.rand() for _ in range(10)]
        actions_numpy = [action_spec.to_numpy(a) for a in actions_tensors]
        actions_tensors_2 = [action_spec.encode(a) for a in actions_numpy]
        assert all(
            [(a1 == a2).all() for a1, a2 in zip(actions_tensors, actions_tensors_2)]
        )

        sample = np.stack(
            [action_spec.to_numpy(action_spec.rand()) for _ in range(N)], 0
        )
        assert sample.shape[0] == N
        assert sample.shape[1] == 2
        assert sample.ndim == 2, f"found shape: {sample.shape}"

        sample0 = sample[:, 0]
        sample_list = [sum(sample0 == i) for i in range(ns[0])]
        assert chisquare(sample_list).pvalue > 0.1

        sample1 = sample[:, 1]
        sample_list = [sum(sample1 == i) for i in range(ns[1])]
        assert chisquare(sample_list).pvalue > 0.1

    def test_categorical_action_spec_encode(self):
        action_spec = DiscreteTensorSpec(10)

        projected = action_spec.project(
            torch.tensor([-100, -1, 0, 1, 9, 10, 100], dtype=torch.long)
        )
        assert (
            projected == torch.tensor([0, 0, 0, 1, 9, 9, 9], dtype=torch.long)
        ).all()

        projected = action_spec.project(
            torch.tensor([-100.0, -1.0, 0.0, 1.0, 9.0, 10.0, 100.0], dtype=torch.float)
        )
        assert (
            projected == torch.tensor([0, 0, 0, 1, 9, 9, 9], dtype=torch.long)
        ).all()

    def test_bounded_rand(self):
        spec = BoundedTensorSpec(-3, 3)
        sample = torch.stack([spec.rand() for _ in range(100)])
        assert (-3 <= sample).all() and (3 >= sample).all()

    def test_ndbounded_shape(self):
        spec = NdBoundedTensorSpec(-3, 3 * torch.ones(10, 5), shape=[10, 5])
        sample = torch.stack([spec.rand() for _ in range(100)], 0)
        assert (-3 <= sample).all() and (3 >= sample).all()
        assert sample.shape == torch.Size([100, 10, 5])
