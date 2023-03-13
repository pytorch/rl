# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import numpy as np
import pytest
import torch
import torchrl.data.tensor_specs
from _utils_internal import get_available_devices, set_global_var
from scipy.stats import chisquare
from tensordict.tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    _keys_to_empty_composite_spec,
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    LazyStackedCompositeSpec,
    MultiDiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
def test_bounded(dtype):
    torch.manual_seed(0)
    np.random.seed(0)
    for _ in range(100):
        bounds = torch.randn(2).sort()[0]
        ts = BoundedTensorSpec(
            bounds[0].item(), bounds[1].item(), torch.Size((1,)), dtype=dtype
        )
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
        with pytest.raises(AssertionError), set_global_var(
            torchrl.data.tensor_specs, "_CHECK_SPEC_ENCODE", True
        ):
            ts.encode(torch.tensor([11]))  # out of bounds
        assert not torchrl.data.tensor_specs._CHECK_SPEC_ENCODE
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
        ts = BoundedTensorSpec(lb, ub, dtype=dtype)
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
        with pytest.raises(AssertionError), set_global_var(
            torchrl.data.tensor_specs, "_CHECK_SPEC_ENCODE", True
        ):
            ts.encode(torch.rand(10) + 3)  # out of bounds
        with pytest.raises(AssertionError), set_global_var(
            torchrl.data.tensor_specs, "_CHECK_SPEC_ENCODE", True
        ):
            ts.to_numpy(torch.rand(10) + 3)  # out of bounds
        assert not torchrl.data.tensor_specs._CHECK_SPEC_ENCODE


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

    ts = UnboundedContinuousTensorSpec(
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
    ts = MultiOneHotDiscreteTensorSpec(nvec=ns)
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
        categorical = ts.to_categorical(r)
        assert not ts.is_in(categorical)
        assert (ts.encode(categorical) == r).all()


@pytest.mark.parametrize(
    "ns",
    [
        5,
        [5, 2, 3],
        [4, 5, 1, 3],
        [[1, 2], [3, 4]],
        [[[2, 4], [3, 5]], [[4, 5], [2, 3]], [[2, 3], [3, 2]]],
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        None,
        [],
        torch.Size([3]),
        torch.Size([4, 5]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.int, torch.long])
def test_multi_discrete(shape, ns, dtype):
    torch.manual_seed(0)
    np.random.seed(0)
    ts = MultiDiscreteTensorSpec(ns, dtype=dtype)
    _real_shape = shape if shape is not None else []
    nvec_shape = torch.tensor(ns).size()
    for _ in range(100):
        r = ts.rand(shape)

        assert r.shape == torch.Size(
            [
                *_real_shape,
                *nvec_shape,
            ]
        ), (r.shape, ns, shape, _real_shape, nvec_shape)
        assert ts.is_in(r), (r, r.shape, ns)
    rand = torch.rand(
        torch.Size(
            [
                *_real_shape,
                *nvec_shape,
            ]
        )
    )
    projection = ts._project(rand)

    assert rand.shape == projection.shape
    assert ts.is_in(projection)
    if projection.ndim < 1:
        projection.fill_(-1)
    else:
        projection[..., 0] = -1
    assert not ts.is_in(projection)


@pytest.mark.parametrize(
    "n",
    [
        1,
        4,
        7,
        99,
    ],
)
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "shape",
    [
        None,
        [],
        [
            1,
        ],
        [1, 2],
    ],
)
def test_discrete_conversion(n, device, shape):
    categorical = DiscreteTensorSpec(n, device=device, shape=shape)
    shape_one_hot = [n] if not shape else [*shape, n]
    one_hot = OneHotDiscreteTensorSpec(n, device=device, shape=shape_one_hot)

    assert categorical != one_hot
    assert categorical.to_one_hot_spec() == one_hot
    assert one_hot.to_categorical_spec() == categorical

    assert categorical.is_in(one_hot.to_categorical(one_hot.rand(shape)))
    assert one_hot.is_in(categorical.to_one_hot(categorical.rand(shape)))


@pytest.mark.parametrize(
    "ns",
    [
        [
            5,
        ],
        [5, 2, 3],
        [4, 5, 1, 3],
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        torch.Size([3]),
        torch.Size([4, 5]),
    ],
)
@pytest.mark.parametrize("device", get_available_devices())
def test_multi_discrete_conversion(ns, shape, device):
    categorical = MultiDiscreteTensorSpec(ns, device=device)
    one_hot = MultiOneHotDiscreteTensorSpec(ns, device=device)

    assert categorical != one_hot
    assert categorical.to_one_hot_spec() == one_hot
    assert one_hot.to_categorical_spec() == categorical

    assert categorical.is_in(one_hot.to_categorical(one_hot.rand(shape)))
    assert one_hot.is_in(categorical.to_one_hot(categorical.rand(shape)))


@pytest.mark.parametrize("is_complete", [True, False])
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
class TestComposite:
    @staticmethod
    def _composite_spec(is_complete=True, device=None, dtype=None):
        torch.manual_seed(0)
        np.random.seed(0)

        return CompositeSpec(
            obs=BoundedTensorSpec(
                torch.zeros(3, 32, 32),
                torch.ones(3, 32, 32),
                dtype=dtype,
                device=device,
            ),
            act=UnboundedContinuousTensorSpec((7,), dtype=dtype, device=device)
            if is_complete
            else None,
        )

    def test_getitem(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        assert isinstance(ts["obs"], BoundedTensorSpec)
        if is_complete:
            assert isinstance(ts["act"], UnboundedContinuousTensorSpec)
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

    def test_to_numpy(self, is_complete, device, dtype):
        ts = self._composite_spec(is_complete, device, dtype)
        for _ in range(100):
            r = ts.rand()
            for key, value in ts.to_numpy(r).items():
                spec = ts[key]
                assert (spec.to_numpy(r[key]) == value).all()

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
        td_to = ts.to(dest)
        cast_r = td_to.rand()

        assert td_to.device == dest
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
            "nested_cp",
        }
        assert set(ts.keys(include_nested=True)) == {
            "obs",
            "act",
            "nested_cp",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
        }
        assert set(ts.keys(include_nested=True, leaves_only=True)) == {
            "obs",
            "act",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
        }
        assert set(ts.keys(leaves_only=True)) == {
            "obs",
            "act",
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
        assert set(ts.keys(include_nested=True)) == {
            "obs",
            "act",
            "nested_cp",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
            "new",
        }

        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        td2 = CompositeSpec(nested_cp=CompositeSpec(new=None).to(device))
        ts.update(td2)
        assert set(ts.keys(include_nested=True)) == {
            "obs",
            "act",
            "nested_cp",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
            ("nested_cp", "new"),
        }

        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        td2 = CompositeSpec(nested_cp=CompositeSpec(act=None).to(device))
        ts.update(td2)
        assert set(ts.keys(include_nested=True)) == {
            "obs",
            "act",
            "nested_cp",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
        }
        assert ts["nested_cp"]["act"] is None

        ts = self._composite_spec(is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(is_complete, device, dtype)
        td2 = CompositeSpec(nested_cp=CompositeSpec(act=None).to(device))
        ts.update(td2)
        td2 = CompositeSpec(
            nested_cp=CompositeSpec(act=UnboundedContinuousTensorSpec(device=device))
        )
        ts.update(td2)
        assert set(ts.keys(include_nested=True)) == {
            "obs",
            "act",
            "nested_cp",
            ("nested_cp", "obs"),
            ("nested_cp", "act"),
        }
        assert ts["nested_cp"]["act"] is not None


def test_keys_to_empty_composite_spec():
    keys = [("key1", "out"), ("key1", "in"), "key2", ("key1", "subkey1", "subkey2")]
    composite = _keys_to_empty_composite_spec(keys)
    assert set(composite.keys(True, True)) == set(keys)


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

        ts = BoundedTensorSpec(minimum, maximum, torch.Size((1,)), device, dtype)

        ts_same = BoundedTensorSpec(minimum, maximum, torch.Size((1,)), device, dtype)
        assert ts == ts_same

        ts_other = BoundedTensorSpec(
            minimum + 1, maximum, torch.Size((1,)), device, dtype
        )
        assert ts != ts_other

        ts_other = BoundedTensorSpec(
            minimum, maximum + 1, torch.Size((1,)), device, dtype
        )
        assert ts != ts_other

        ts_other = BoundedTensorSpec(minimum, maximum, torch.Size((1,)), "cpu:0", dtype)
        assert ts != ts_other

        ts_other = BoundedTensorSpec(
            minimum, maximum, torch.Size((1,)), device, torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            UnboundedContinuousTensorSpec(device=device, dtype=dtype), ts
        )
        assert ts != ts_other

    def test_equality_onehot(self):
        n = 5
        device = "cpu"
        dtype = torch.float16
        use_register = False

        ts = OneHotDiscreteTensorSpec(
            n=n, device=device, dtype=dtype, use_register=use_register
        )

        ts_same = OneHotDiscreteTensorSpec(
            n=n, device=device, dtype=dtype, use_register=use_register
        )
        assert ts == ts_same

        ts_other = OneHotDiscreteTensorSpec(
            n=n + 1, device=device, dtype=dtype, use_register=use_register
        )
        assert ts != ts_other

        ts_other = OneHotDiscreteTensorSpec(
            n=n, device="cpu:0", dtype=dtype, use_register=use_register
        )
        assert ts != ts_other

        ts_other = OneHotDiscreteTensorSpec(
            n=n, device=device, dtype=torch.float64, use_register=use_register
        )
        assert ts != ts_other

        ts_other = OneHotDiscreteTensorSpec(
            n=n, device=device, dtype=dtype, use_register=not use_register
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            UnboundedContinuousTensorSpec(device=device, dtype=dtype), ts
        )
        assert ts != ts_other

    def test_equality_unbounded(self):
        device = "cpu"
        dtype = torch.float16

        ts = UnboundedContinuousTensorSpec(device=device, dtype=dtype)

        ts_same = UnboundedContinuousTensorSpec(device=device, dtype=dtype)
        assert ts == ts_same

        ts_other = UnboundedContinuousTensorSpec(device="cpu:0", dtype=dtype)
        assert ts != ts_other

        ts_other = UnboundedContinuousTensorSpec(device=device, dtype=torch.float64)
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, torch.Size((1,)), device, dtype), ts
        )
        assert ts != ts_other

    def test_equality_ndbounded(self):
        minimum = np.arange(12).reshape((3, 4))
        maximum = minimum + 100
        device = "cpu"
        dtype = torch.float16

        ts = BoundedTensorSpec(
            minimum=minimum, maximum=maximum, device=device, dtype=dtype
        )

        ts_same = BoundedTensorSpec(
            minimum=minimum, maximum=maximum, device=device, dtype=dtype
        )
        assert ts == ts_same

        ts_other = BoundedTensorSpec(
            minimum=minimum + 1, maximum=maximum, device=device, dtype=dtype
        )
        assert ts != ts_other

        ts_other = BoundedTensorSpec(
            minimum=minimum, maximum=maximum + 1, device=device, dtype=dtype
        )
        assert ts != ts_other

        ts_other = BoundedTensorSpec(
            minimum=minimum, maximum=maximum, device="cpu:0", dtype=dtype
        )
        assert ts != ts_other

        ts_other = BoundedTensorSpec(
            minimum=minimum, maximum=maximum, device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            UnboundedContinuousTensorSpec(device=device, dtype=dtype), ts
        )
        assert ts != ts_other

    def test_equality_discrete(self):
        n = 5
        shape = torch.Size([1])
        device = "cpu"
        dtype = torch.float16

        ts = DiscreteTensorSpec(n=n, shape=shape, device=device, dtype=dtype)

        ts_same = DiscreteTensorSpec(n=n, shape=shape, device=device, dtype=dtype)
        assert ts == ts_same

        ts_other = DiscreteTensorSpec(n=n + 1, shape=shape, device=device, dtype=dtype)
        assert ts != ts_other

        ts_other = DiscreteTensorSpec(n=n, shape=shape, device="cpu:0", dtype=dtype)
        assert ts != ts_other

        ts_other = DiscreteTensorSpec(
            n=n, shape=shape, device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = DiscreteTensorSpec(
            n=n, shape=torch.Size([2]), device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            UnboundedContinuousTensorSpec(device=device, dtype=dtype), ts
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

        ts = UnboundedContinuousTensorSpec(shape=shape, device=device, dtype=dtype)

        ts_same = UnboundedContinuousTensorSpec(shape=shape, device=device, dtype=dtype)
        assert ts == ts_same

        other_shape = 13 if type(shape) == int else torch.Size(np.array(shape) + 10)
        ts_other = UnboundedContinuousTensorSpec(
            shape=other_shape, device=device, dtype=dtype
        )
        assert ts != ts_other

        ts_other = UnboundedContinuousTensorSpec(
            shape=shape, device="cpu:0", dtype=dtype
        )
        assert ts != ts_other

        ts_other = UnboundedContinuousTensorSpec(
            shape=shape, device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, torch.Size((1,)), device, dtype), ts
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
            BoundedTensorSpec(0, 1, torch.Size((1,)), device, dtype), ts
        )
        assert ts != ts_other

    @pytest.mark.parametrize("nvec", [[3], [3, 4], [3, 4, 5]])
    def test_equality_multi_onehot(self, nvec):
        device = "cpu"
        dtype = torch.float16

        ts = MultiOneHotDiscreteTensorSpec(nvec=nvec, device=device, dtype=dtype)

        ts_same = MultiOneHotDiscreteTensorSpec(nvec=nvec, device=device, dtype=dtype)
        assert ts == ts_same

        other_nvec = np.array(nvec) + 3
        ts_other = MultiOneHotDiscreteTensorSpec(
            nvec=other_nvec, device=device, dtype=dtype
        )
        assert ts != ts_other

        other_nvec = [12]
        ts_other = MultiOneHotDiscreteTensorSpec(
            nvec=other_nvec, device=device, dtype=dtype
        )
        assert ts != ts_other

        other_nvec = [12, 13]
        ts_other = MultiOneHotDiscreteTensorSpec(
            nvec=other_nvec, device=device, dtype=dtype
        )
        assert ts != ts_other

        ts_other = MultiOneHotDiscreteTensorSpec(nvec=nvec, device="cpu:0", dtype=dtype)
        assert ts != ts_other

        ts_other = MultiOneHotDiscreteTensorSpec(
            nvec=nvec, device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, torch.Size((1,)), device, dtype), ts
        )
        assert ts != ts_other

    @pytest.mark.parametrize("nvec", [[3], [3, 4], [3, 4, 5], [[1, 2], [3, 4]]])
    def test_equality_multi_discrete(self, nvec):
        device = "cpu"
        dtype = torch.float16

        ts = MultiDiscreteTensorSpec(nvec=nvec, device=device, dtype=dtype)

        ts_same = MultiDiscreteTensorSpec(nvec=nvec, device=device, dtype=dtype)
        assert ts == ts_same

        other_nvec = np.array(nvec) + 3
        ts_other = MultiDiscreteTensorSpec(nvec=other_nvec, device=device, dtype=dtype)
        assert ts != ts_other

        other_nvec = [12]
        ts_other = MultiDiscreteTensorSpec(nvec=other_nvec, device=device, dtype=dtype)
        assert ts != ts_other

        other_nvec = [12, 13]
        ts_other = MultiDiscreteTensorSpec(nvec=other_nvec, device=device, dtype=dtype)
        assert ts != ts_other

        ts_other = MultiDiscreteTensorSpec(nvec=nvec, device="cpu:0", dtype=dtype)
        assert ts != ts_other

        ts_other = MultiDiscreteTensorSpec(
            nvec=nvec, device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, torch.Size((1,)), device, dtype), ts
        )
        assert ts != ts_other

    def test_equality_composite(self):
        minimum = np.arange(12).reshape((3, 4))
        maximum = minimum + 100
        device = "cpu"
        dtype = torch.float16

        bounded = BoundedTensorSpec(0, 1, torch.Size((1,)), device, dtype)
        bounded_same = BoundedTensorSpec(0, 1, torch.Size((1,)), device, dtype)
        bounded_other = BoundedTensorSpec(0, 2, torch.Size((1,)), device, dtype)

        nd = BoundedTensorSpec(
            minimum=minimum, maximum=maximum + 1, device=device, dtype=dtype
        )
        nd_same = BoundedTensorSpec(
            minimum=minimum, maximum=maximum + 1, device=device, dtype=dtype
        )
        _ = BoundedTensorSpec(
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
            (a1 == a2).all() for a1, a2 in zip(actions_tensors, actions_tensors_2)
        )

        actions_numpy = [int(np.random.randint(0, 10, (1,))) for a in actions_tensors]
        actions_tensors = [action_spec.encode(a) for a in actions_numpy]
        actions_numpy_2 = [action_spec.to_numpy(a) for a in actions_tensors]
        assert all((a1 == a2) for a1, a2 in zip(actions_numpy, actions_numpy_2))

    def test_mult_discrete_action_spec_reconstruct(self):
        torch.manual_seed(0)
        action_spec = MultiOneHotDiscreteTensorSpec((10, 5))

        actions_tensors = [action_spec.rand() for _ in range(10)]
        actions_categorical = [action_spec.to_categorical(a) for a in actions_tensors]
        actions_tensors_2 = [action_spec.encode(a) for a in actions_categorical]
        assert all(
            [(a1 == a2).all() for a1, a2 in zip(actions_tensors, actions_tensors_2)]
        )

        actions_categorical = [
            torch.cat((torch.randint(0, 10, (1,)), torch.randint(0, 5, (1,))), 0)
            for a in actions_tensors
        ]
        actions_tensors = [action_spec.encode(a) for a in actions_categorical]
        actions_categorical_2 = [action_spec.to_categorical(a) for a in actions_tensors]
        assert all(
            (a1 == a2).all()
            for a1, a2 in zip(actions_categorical, actions_categorical_2)
        )

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
        action_spec = MultiOneHotDiscreteTensorSpec((10, 5))

        actions_tensors = [action_spec.rand() for _ in range(10)]
        actions_categorical = [action_spec.to_categorical(a) for a in actions_tensors]
        actions_tensors_2 = [action_spec.encode(a) for a in actions_categorical]
        assert all(
            [(a1 == a2).all() for a1, a2 in zip(actions_tensors, actions_tensors_2)]
        )

        sample = torch.stack(
            [action_spec.to_categorical(action_spec.rand()) for _ in range(N)], 0
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
        spec = BoundedTensorSpec(-3, 3, torch.Size((1,)))
        sample = torch.stack([spec.rand() for _ in range(100)])
        assert (-3 <= sample).all() and (3 >= sample).all()

    def test_ndbounded_shape(self):
        spec = BoundedTensorSpec(-3, 3 * torch.ones(10, 5), shape=[10, 5])
        sample = torch.stack([spec.rand() for _ in range(100)], 0)
        assert (-3 <= sample).all() and (3 >= sample).all()
        assert sample.shape == torch.Size([100, 10, 5])


class TestExpand:
    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (4,),
            (5, 4),
        ],
    )
    @pytest.mark.parametrize("shape2", [(), (10,)])
    def test_binary(self, shape1, shape2):
        spec = BinaryDiscreteTensorSpec(
            n=4, shape=shape1, device="cpu", dtype=torch.bool
        )
        if shape1 is not None:
            shape2_real = (*shape2, *shape1)
        else:
            shape2_real = (*shape2, 4)

        spec2 = spec.expand(shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape
        spec2 = spec.expand(*shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape

    @pytest.mark.parametrize("shape2", [(), (5,)])
    @pytest.mark.parametrize(
        "shape1,mini,maxi",
        [
            [(10,), -torch.ones([]), torch.ones([])],
            [None, -torch.ones([10]), torch.ones([])],
            [None, -torch.ones([]), torch.ones([10])],
            [(10,), -torch.ones([]), torch.ones([10])],
            [(10,), -torch.ones([10]), torch.ones([])],
            [(10,), -torch.ones([10]), torch.ones([10])],
        ],
    )
    def test_bounded(self, shape1, shape2, mini, maxi):
        spec = BoundedTensorSpec(
            mini, maxi, shape=shape1, device="cpu", dtype=torch.bool
        )
        shape1 = spec.shape
        assert shape1 == torch.Size([10])
        shape2_real = (*shape2, *shape1)

        spec2 = spec.expand(shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape
        spec2 = spec.expand(*shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape

    def test_composite(self):
        batch_size = (5,)
        spec1 = BoundedTensorSpec(
            -torch.ones([*batch_size, 10]),
            torch.ones([*batch_size, 10]),
            shape=(
                *batch_size,
                10,
            ),
            device="cpu",
            dtype=torch.bool,
        )
        spec2 = BinaryDiscreteTensorSpec(
            n=4, shape=(*batch_size, 4), device="cpu", dtype=torch.bool
        )
        spec3 = DiscreteTensorSpec(
            n=4, shape=batch_size, device="cpu", dtype=torch.long
        )
        spec4 = MultiDiscreteTensorSpec(
            nvec=(4, 5, 6), shape=(*batch_size, 3), device="cpu", dtype=torch.long
        )
        spec5 = MultiOneHotDiscreteTensorSpec(
            nvec=(4, 5, 6), shape=(*batch_size, 15), device="cpu", dtype=torch.long
        )
        spec6 = OneHotDiscreteTensorSpec(
            n=15, shape=(*batch_size, 15), device="cpu", dtype=torch.long
        )
        spec7 = UnboundedContinuousTensorSpec(
            shape=(*batch_size, 9),
            device="cpu",
            dtype=torch.float64,
        )
        spec8 = UnboundedDiscreteTensorSpec(
            shape=(*batch_size, 9),
            device="cpu",
            dtype=torch.long,
        )
        spec = CompositeSpec(
            spec1=spec1,
            spec2=spec2,
            spec3=spec3,
            spec4=spec4,
            spec5=spec5,
            spec6=spec6,
            spec7=spec7,
            spec8=spec8,
            shape=batch_size,
        )
        for new_spec in (spec.expand((4, *batch_size)), spec.expand(4, *batch_size)):
            assert new_spec is not spec
            assert new_spec.shape == torch.Size([4, *batch_size])
            assert new_spec["spec1"].shape == torch.Size([4, *batch_size, 10])
            assert new_spec["spec2"].shape == torch.Size([4, *batch_size, 4])
            assert new_spec["spec3"].shape == torch.Size(
                [
                    4,
                    *batch_size,
                ]
            )
            assert new_spec["spec4"].shape == torch.Size([4, *batch_size, 3])
            assert new_spec["spec5"].shape == torch.Size([4, *batch_size, 15])
            assert new_spec["spec6"].shape == torch.Size([4, *batch_size, 15])
            assert new_spec["spec7"].shape == torch.Size([4, *batch_size, 9])
            assert new_spec["spec8"].shape == torch.Size([4, *batch_size, 9])

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    @pytest.mark.parametrize("shape2", [(), (10,)])
    def test_discrete(self, shape1, shape2):
        spec = DiscreteTensorSpec(n=4, shape=shape1, device="cpu", dtype=torch.long)
        if shape1 is not None:
            shape2_real = (*shape2, *shape1)
        else:
            shape2_real = shape2

        spec2 = spec.expand(shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape
        spec2 = spec.expand(*shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    @pytest.mark.parametrize("shape2", [(), (10,)])
    def test_multidiscrete(self, shape1, shape2):
        if shape1 is None:
            shape1 = (3,)
        else:
            shape1 = (*shape1, 3)
        spec = MultiDiscreteTensorSpec(
            nvec=(4, 5, 6), shape=shape1, device="cpu", dtype=torch.long
        )
        if shape1 is not None:
            shape2_real = (*shape2, *shape1)
        else:
            shape2_real = shape2

        spec2 = spec.expand(shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape
        spec2 = spec.expand(*shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    @pytest.mark.parametrize("shape2", [(), (10,)])
    def test_multionehot(self, shape1, shape2):
        if shape1 is None:
            shape1 = (15,)
        else:
            shape1 = (*shape1, 15)
        spec = MultiOneHotDiscreteTensorSpec(
            nvec=(4, 5, 6), shape=shape1, device="cpu", dtype=torch.long
        )
        if shape1 is not None:
            shape2_real = (*shape2, *shape1)
        else:
            shape2_real = shape2

        spec2 = spec.expand(shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape
        spec2 = spec.expand(*shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    @pytest.mark.parametrize("shape2", [(), (10,)])
    def test_onehot(self, shape1, shape2):
        if shape1 is None:
            shape1 = (15,)
        else:
            shape1 = (*shape1, 15)
        spec = OneHotDiscreteTensorSpec(
            n=15, shape=shape1, device="cpu", dtype=torch.long
        )
        if shape1 is not None:
            shape2_real = (*shape2, *shape1)
        else:
            shape2_real = shape2

        spec2 = spec.expand(shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape
        spec2 = spec.expand(*shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    @pytest.mark.parametrize("shape2", [(), (10,)])
    def test_unbounded(self, shape1, shape2):
        if shape1 is None:
            shape1 = (15,)
        else:
            shape1 = (*shape1, 15)
        spec = UnboundedContinuousTensorSpec(
            shape=shape1, device="cpu", dtype=torch.float64
        )
        if shape1 is not None:
            shape2_real = (*shape2, *shape1)
        else:
            shape2_real = shape2

        spec2 = spec.expand(shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape
        spec2 = spec.expand(*shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    @pytest.mark.parametrize("shape2", [(), (10,)])
    def test_unboundeddiscrete(self, shape1, shape2):
        if shape1 is None:
            shape1 = (15,)
        else:
            shape1 = (*shape1, 15)
        spec = UnboundedDiscreteTensorSpec(shape=shape1, device="cpu", dtype=torch.long)
        if shape1 is not None:
            shape2_real = (*shape2, *shape1)
        else:
            shape2_real = shape2

        spec2 = spec.expand(shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        spec2 = spec.expand(*shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert (spec2.zero() == spec.zero()).all()
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape
        spec2 = spec.expand(*shape2_real)
        assert spec2 is not spec
        assert spec2.dtype == spec.dtype
        assert spec2.rand().shape == spec2.shape
        assert spec2.zero().shape == spec2.shape


class TestClone:
    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (4,),
            (5, 4),
        ],
    )
    def test_binary(self, shape1):
        spec = BinaryDiscreteTensorSpec(
            n=4, shape=shape1, device="cpu", dtype=torch.bool
        )
        assert spec == spec.clone()
        assert spec is not spec.clone()

    @pytest.mark.parametrize(
        "shape1,mini,maxi",
        [
            [(10,), -torch.ones([]), torch.ones([])],
            [None, -torch.ones([10]), torch.ones([])],
            [None, -torch.ones([]), torch.ones([10])],
            [(10,), -torch.ones([]), torch.ones([10])],
            [(10,), -torch.ones([10]), torch.ones([])],
            [(10,), -torch.ones([10]), torch.ones([10])],
        ],
    )
    def test_bounded(self, shape1, mini, maxi):
        spec = BoundedTensorSpec(
            mini, maxi, shape=shape1, device="cpu", dtype=torch.bool
        )
        assert spec == spec.clone()
        assert spec is not spec.clone()

    def test_composite(self):
        batch_size = (5,)
        spec1 = BoundedTensorSpec(
            -torch.ones([*batch_size, 10]),
            torch.ones([*batch_size, 10]),
            shape=(
                *batch_size,
                10,
            ),
            device="cpu",
            dtype=torch.bool,
        )
        spec2 = BinaryDiscreteTensorSpec(
            n=4, shape=(*batch_size, 4), device="cpu", dtype=torch.bool
        )
        spec3 = DiscreteTensorSpec(
            n=4, shape=batch_size, device="cpu", dtype=torch.long
        )
        spec4 = MultiDiscreteTensorSpec(
            nvec=(4, 5, 6), shape=(*batch_size, 3), device="cpu", dtype=torch.long
        )
        spec5 = MultiOneHotDiscreteTensorSpec(
            nvec=(4, 5, 6), shape=(*batch_size, 15), device="cpu", dtype=torch.long
        )
        spec6 = OneHotDiscreteTensorSpec(
            n=15, shape=(*batch_size, 15), device="cpu", dtype=torch.long
        )
        spec7 = UnboundedContinuousTensorSpec(
            shape=(*batch_size, 9),
            device="cpu",
            dtype=torch.float64,
        )
        spec8 = UnboundedDiscreteTensorSpec(
            shape=(*batch_size, 9),
            device="cpu",
            dtype=torch.long,
        )
        spec = CompositeSpec(
            spec1=spec1,
            spec2=spec2,
            spec3=spec3,
            spec4=spec4,
            spec5=spec5,
            spec6=spec6,
            spec7=spec7,
            spec8=spec8,
            shape=batch_size,
        )
        assert spec is not spec.clone()
        spec_clone = spec.clone()
        for key, item in spec.items():
            assert item == spec_clone[key], key
        assert spec == spec.clone()

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    def test_discrete(
        self,
        shape1,
    ):
        spec = DiscreteTensorSpec(n=4, shape=shape1, device="cpu", dtype=torch.long)
        assert spec == spec.clone()
        assert spec is not spec.clone()

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    def test_multidiscrete(
        self,
        shape1,
    ):
        if shape1 is None:
            shape1 = (3,)
        else:
            shape1 = (*shape1, 3)
        spec = MultiDiscreteTensorSpec(
            nvec=(4, 5, 6), shape=shape1, device="cpu", dtype=torch.long
        )
        assert spec == spec.clone()
        assert spec is not spec.clone()

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    def test_multionehot(
        self,
        shape1,
    ):
        if shape1 is None:
            shape1 = (15,)
        else:
            shape1 = (*shape1, 15)
        spec = MultiOneHotDiscreteTensorSpec(
            nvec=(4, 5, 6), shape=shape1, device="cpu", dtype=torch.long
        )
        assert spec == spec.clone()
        assert spec is not spec.clone()

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    def test_onehot(
        self,
        shape1,
    ):
        if shape1 is None:
            shape1 = (15,)
        else:
            shape1 = (*shape1, 15)
        spec = OneHotDiscreteTensorSpec(
            n=15, shape=shape1, device="cpu", dtype=torch.long
        )
        assert spec == spec.clone()
        assert spec is not spec.clone()

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    def test_unbounded(
        self,
        shape1,
    ):
        if shape1 is None:
            shape1 = (15,)
        else:
            shape1 = (*shape1, 15)
        spec = UnboundedContinuousTensorSpec(
            shape=shape1, device="cpu", dtype=torch.float64
        )
        assert spec == spec.clone()
        assert spec is not spec.clone()

    @pytest.mark.parametrize(
        "shape1",
        [
            None,
            (),
            (5,),
        ],
    )
    def test_unboundeddiscrete(
        self,
        shape1,
    ):
        if shape1 is None:
            shape1 = (15,)
        else:
            shape1 = (*shape1, 15)
        spec = UnboundedDiscreteTensorSpec(shape=shape1, device="cpu", dtype=torch.long)
        assert spec == spec.clone()
        assert spec is not spec.clone()


@pytest.mark.parametrize(
    "shape,stack_dim",
    [[(), 0], [(2,), 0], [(2,), 1], [(2, 3), 0], [(2, 3), 1], [(2, 3), 2]],
)
class TestStack:
    def test_stack_binarydiscrete(self, shape, stack_dim):
        n = 5
        shape = (*shape, n)
        c1 = BinaryDiscreteTensorSpec(n=n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        assert isinstance(c, BinaryDiscreteTensorSpec)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        assert c.shape == torch.Size(shape)

    def test_stack_binarydiscrete_expand(self, shape, stack_dim):
        n = 5
        shape = (*shape, n)
        c1 = BinaryDiscreteTensorSpec(n=n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        cexpand = c.expand(3, 2, *shape)
        assert cexpand.shape == torch.Size([3, 2, *shape])

    def test_stack_binarydiscrete_rand(self, shape, stack_dim):
        n = 5
        shape = (*shape, n)
        c1 = BinaryDiscreteTensorSpec(n=n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.rand()
        assert r.shape == c.shape

    def test_stack_binarydiscrete_zero(self, shape, stack_dim):
        n = 5
        shape = (*shape, n)
        c1 = BinaryDiscreteTensorSpec(n=n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.zero()
        assert r.shape == c.shape

    def test_stack_bounded(self, shape, stack_dim):
        mini = -1
        maxi = 1
        shape = (*shape,)
        c1 = BoundedTensorSpec(mini, maxi, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        assert isinstance(c, BoundedTensorSpec)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        assert c.shape == torch.Size(shape)

    def test_stack_bounded_expand(self, shape, stack_dim):
        mini = -1
        maxi = 1
        shape = (*shape,)
        c1 = BoundedTensorSpec(mini, maxi, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        cexpand = c.expand(3, 2, *shape)
        assert cexpand.shape == torch.Size([3, 2, *shape])

    def test_stack_bounded_rand(self, shape, stack_dim):
        mini = -1
        maxi = 1
        shape = (*shape,)
        c1 = BoundedTensorSpec(mini, maxi, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.rand()
        assert r.shape == c.shape

    def test_stack_bounded_zero(self, shape, stack_dim):
        mini = -1
        maxi = 1
        shape = (*shape,)
        c1 = BoundedTensorSpec(mini, maxi, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.zero()
        assert r.shape == c.shape

    def test_stack_discrete(self, shape, stack_dim):
        n = 4
        shape = (*shape,)
        c1 = DiscreteTensorSpec(n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        assert isinstance(c, DiscreteTensorSpec)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        assert c.shape == torch.Size(shape)

    def test_stack_discrete_expand(self, shape, stack_dim):
        n = 4
        shape = (*shape,)
        c1 = DiscreteTensorSpec(n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        cexpand = c.expand(3, 2, *shape)
        assert cexpand.shape == torch.Size([3, 2, *shape])

    def test_stack_discrete_rand(self, shape, stack_dim):
        n = 4
        shape = (*shape,)
        c1 = DiscreteTensorSpec(n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.rand()
        assert r.shape == c.shape

    def test_stack_discrete_zero(self, shape, stack_dim):
        n = 4
        shape = (*shape,)
        c1 = DiscreteTensorSpec(n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.zero()
        assert r.shape == c.shape

    def test_stack_multidiscrete(self, shape, stack_dim):
        nvec = [4, 5]
        shape = (*shape, 2)
        c1 = MultiDiscreteTensorSpec(nvec, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        assert isinstance(c, MultiDiscreteTensorSpec)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        assert c.shape == torch.Size(shape)

    def test_stack_multidiscrete_expand(self, shape, stack_dim):
        nvec = [4, 5]
        shape = (*shape, 2)
        c1 = MultiDiscreteTensorSpec(nvec, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        cexpand = c.expand(3, 2, *shape)
        assert cexpand.shape == torch.Size([3, 2, *shape])

    def test_stack_multidiscrete_rand(self, shape, stack_dim):
        nvec = [4, 5]
        shape = (*shape, 2)
        c1 = MultiDiscreteTensorSpec(nvec, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.rand()
        assert r.shape == c.shape

    def test_stack_multidiscrete_zero(self, shape, stack_dim):
        nvec = [4, 5]
        shape = (*shape, 2)
        c1 = MultiDiscreteTensorSpec(nvec, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.zero()
        assert r.shape == c.shape

    def test_stack_multionehot(self, shape, stack_dim):
        nvec = [4, 5]
        shape = (*shape, 9)
        c1 = MultiOneHotDiscreteTensorSpec(nvec, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        assert isinstance(c, MultiOneHotDiscreteTensorSpec)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        assert c.shape == torch.Size(shape)

    def test_stack_multionehot_expand(self, shape, stack_dim):
        nvec = [4, 5]
        shape = (*shape, 9)
        c1 = MultiOneHotDiscreteTensorSpec(nvec, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        cexpand = c.expand(3, 2, *shape)
        assert cexpand.shape == torch.Size([3, 2, *shape])

    def test_stack_multionehot_rand(self, shape, stack_dim):
        nvec = [4, 5]
        shape = (*shape, 9)
        c1 = MultiOneHotDiscreteTensorSpec(nvec, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.rand()
        assert r.shape == c.shape

    def test_stack_multionehot_zero(self, shape, stack_dim):
        nvec = [4, 5]
        shape = (*shape, 9)
        c1 = MultiOneHotDiscreteTensorSpec(nvec, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.zero()
        assert r.shape == c.shape

    def test_stack_onehot(self, shape, stack_dim):
        n = 5
        shape = (*shape, 5)
        c1 = OneHotDiscreteTensorSpec(n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        assert isinstance(c, OneHotDiscreteTensorSpec)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        assert c.shape == torch.Size(shape)

    def test_stack_onehot_expand(self, shape, stack_dim):
        n = 5
        shape = (*shape, 5)
        c1 = OneHotDiscreteTensorSpec(n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        cexpand = c.expand(3, 2, *shape)
        assert cexpand.shape == torch.Size([3, 2, *shape])

    def test_stack_onehot_rand(self, shape, stack_dim):
        n = 5
        shape = (*shape, 5)
        c1 = OneHotDiscreteTensorSpec(n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.rand()
        assert r.shape == c.shape

    def test_stack_onehot_zero(self, shape, stack_dim):
        n = 5
        shape = (*shape, 5)
        c1 = OneHotDiscreteTensorSpec(n, shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.zero()
        assert r.shape == c.shape

    def test_stack_unboundedcont(self, shape, stack_dim):
        shape = (*shape,)
        c1 = UnboundedContinuousTensorSpec(shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        assert isinstance(c, UnboundedContinuousTensorSpec)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        assert c.shape == torch.Size(shape)

    def test_stack_unboundedcont_expand(self, shape, stack_dim):
        shape = (*shape,)
        c1 = UnboundedContinuousTensorSpec(shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        cexpand = c.expand(3, 2, *shape)
        assert cexpand.shape == torch.Size([3, 2, *shape])

    def test_stack_unboundedcont_rand(self, shape, stack_dim):
        shape = (*shape,)
        c1 = UnboundedContinuousTensorSpec(shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.rand()
        assert r.shape == c.shape

    def test_stack_unboundedcont_zero(self, shape, stack_dim):
        shape = (*shape,)
        c1 = UnboundedDiscreteTensorSpec(shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.zero()
        assert r.shape == c.shape

    def test_stack_unboundeddiscrete(self, shape, stack_dim):
        shape = (*shape,)
        c1 = UnboundedDiscreteTensorSpec(shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        assert isinstance(c, UnboundedDiscreteTensorSpec)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        assert c.shape == torch.Size(shape)

    def test_stack_unboundeddiscrete_expand(self, shape, stack_dim):
        shape = (*shape,)
        c1 = UnboundedDiscreteTensorSpec(shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        shape = list(shape)
        if stack_dim < 0:
            stack_dim = len(shape) + stack_dim + 1
        shape.insert(stack_dim, 2)
        cexpand = c.expand(3, 2, *shape)
        assert cexpand.shape == torch.Size([3, 2, *shape])

    def test_stack_unboundeddiscrete_rand(self, shape, stack_dim):
        shape = (*shape,)
        c1 = UnboundedDiscreteTensorSpec(shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.rand()
        assert r.shape == c.shape

    def test_stack_unboundeddiscrete_zero(self, shape, stack_dim):
        shape = (*shape,)
        c1 = UnboundedDiscreteTensorSpec(shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        r = c.zero()
        assert r.shape == c.shape


class TestStackComposite:
    def test_stack(self):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec())
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        assert isinstance(c, CompositeSpec)

    def test_stack_index(self):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec())
        c2 = CompositeSpec(
            a=UnboundedContinuousTensorSpec(), b=UnboundedDiscreteTensorSpec()
        )
        c = torch.stack([c1, c2], 0)
        assert c.shape == torch.Size([2])
        assert c[0] is c1
        assert c[1] is c2
        assert c[..., 0] is c1
        assert c[..., 1] is c2
        assert c[0, ...] is c1
        assert c[1, ...] is c2
        assert isinstance(c[:], LazyStackedCompositeSpec)

    @pytest.mark.parametrize("stack_dim", [0, 1, 2, -3, -2, -1])
    def test_stack_index_multdim(self, stack_dim):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec(shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], stack_dim)
        if stack_dim in (0, -3):
            assert isinstance(c[:], LazyStackedCompositeSpec)
            assert c.shape == torch.Size([2, 1, 3])
            assert c[0] is c1
            assert c[1] is c2
            with pytest.raises(
                IndexError,
                match="only permitted if the stack dimension is the last dimension",
            ):
                assert c[..., 0] is c1
            with pytest.raises(
                IndexError,
                match="only permitted if the stack dimension is the last dimension",
            ):
                assert c[..., 1] is c2
            assert c[0, ...] is c1
            assert c[1, ...] is c2
        elif stack_dim == (1, -2):
            assert isinstance(c[:, :], LazyStackedCompositeSpec)
            assert c.shape == torch.Size([1, 2, 3])
            assert c[:, 0] is c1
            assert c[:, 1] is c2
            with pytest.raises(
                IndexError, match="along dimension 0 when the stack dimension is 1."
            ):
                assert c[0] is c1
            with pytest.raises(
                IndexError, match="along dimension 0 when the stack dimension is 1."
            ):
                assert c[1] is c1
            with pytest.raises(
                IndexError,
                match="only permitted if the stack dimension is the last dimension",
            ):
                assert c[..., 0] is c1
            with pytest.raises(
                IndexError,
                match="only permitted if the stack dimension is the last dimension",
            ):
                assert c[..., 1] is c2
            assert c[..., 0, :] is c1
            assert c[..., 1, :] is c2
            assert c[:, 0, ...] is c1
            assert c[:, 1, ...] is c2
        elif stack_dim == (2, -1):
            assert isinstance(c[:, :, :], LazyStackedCompositeSpec)
            with pytest.raises(
                IndexError, match="along dimension 0 when the stack dimension is 2."
            ):
                assert c[0] is c1
            with pytest.raises(
                IndexError, match="along dimension 0 when the stack dimension is 2."
            ):
                assert c[1] is c1
            assert c.shape == torch.Size([1, 3, 2])
            assert c[:, :, 0] is c1
            assert c[:, :, 1] is c2
            assert c[..., 0] is c1
            assert c[..., 1] is c2
            assert c[:, :, 0, ...] is c1
            assert c[:, :, 1, ...] is c2

    @pytest.mark.parametrize("stack_dim", [0, 1, 2, -3, -2, -1])
    def test_stack_expand_multi(self, stack_dim):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec(shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], stack_dim)
        if stack_dim in (0, -3):
            c_expand = c.expand([4, 2, 1, 3])
            assert c_expand.shape == torch.Size([4, 2, 1, 3])
            assert c_expand.dim == 1
        elif stack_dim in (1, -2):
            c_expand = c.expand([4, 1, 2, 3])
            assert c_expand.shape == torch.Size([4, 1, 2, 3])
            assert c_expand.dim == 2
        elif stack_dim in (2, -1):
            c_expand = c.expand(
                [
                    4,
                    1,
                    3,
                    2,
                ]
            )
            assert c_expand.shape == torch.Size([4, 1, 3, 2])
            assert c_expand.dim == 3
        else:
            raise NotImplementedError

    @pytest.mark.parametrize("stack_dim", [0, 1, 2, -3, -2, -1])
    def test_stack_rand(self, stack_dim):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec(shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], stack_dim)
        r = c.rand()
        assert isinstance(r, LazyStackedTensorDict)
        if stack_dim in (0, -3):
            assert r.shape == torch.Size([2, 1, 3])
            assert r["a"].shape == torch.Size([2, 1, 3])  # access tensor
        elif stack_dim in (1, -2):
            assert r.shape == torch.Size([1, 2, 3])
            assert r["a"].shape == torch.Size([1, 2, 3])  # access tensor
        elif stack_dim in (2, -1):
            assert r.shape == torch.Size([1, 3, 2])
            assert r["a"].shape == torch.Size([1, 3, 2])  # access tensor
        assert (r["a"] != 0).all()

    @pytest.mark.parametrize("stack_dim", [0, 1, 2, -3, -2, -1])
    def test_stack_rand_shape(self, stack_dim):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec(shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], stack_dim)
        shape = [5, 6]
        r = c.rand(shape)
        assert isinstance(r, LazyStackedTensorDict)
        if stack_dim in (0, -3):
            assert r.shape == torch.Size([*shape, 2, 1, 3])
            assert r["a"].shape == torch.Size([*shape, 2, 1, 3])  # access tensor
        elif stack_dim in (1, -2):
            assert r.shape == torch.Size([*shape, 1, 2, 3])
            assert r["a"].shape == torch.Size([*shape, 1, 2, 3])  # access tensor
        elif stack_dim in (2, -1):
            assert r.shape == torch.Size([*shape, 1, 3, 2])
            assert r["a"].shape == torch.Size([*shape, 1, 3, 2])  # access tensor
        assert (r["a"] != 0).all()

    @pytest.mark.parametrize("stack_dim", [0, 1, 2, -3, -2, -1])
    def test_stack_zero(self, stack_dim):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec(shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], stack_dim)
        r = c.zero()
        assert isinstance(r, LazyStackedTensorDict)
        if stack_dim in (0, -3):
            assert r.shape == torch.Size([2, 1, 3])
            assert r["a"].shape == torch.Size([2, 1, 3])  # access tensor
        elif stack_dim in (1, -2):
            assert r.shape == torch.Size([1, 2, 3])
            assert r["a"].shape == torch.Size([1, 2, 3])  # access tensor
        elif stack_dim in (2, -1):
            assert r.shape == torch.Size([1, 3, 2])
            assert r["a"].shape == torch.Size([1, 3, 2])  # access tensor
        assert (r["a"] == 0).all()

    @pytest.mark.parametrize("stack_dim", [0, 1, 2, -3, -2, -1])
    def test_stack_zero_shape(self, stack_dim):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec(shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], stack_dim)
        shape = [5, 6]
        r = c.zero(shape)
        assert isinstance(r, LazyStackedTensorDict)
        if stack_dim in (0, -3):
            assert r.shape == torch.Size([*shape, 2, 1, 3])
            assert r["a"].shape == torch.Size([*shape, 2, 1, 3])  # access tensor
        elif stack_dim in (1, -2):
            assert r.shape == torch.Size([*shape, 1, 2, 3])
            assert r["a"].shape == torch.Size([*shape, 1, 2, 3])  # access tensor
        elif stack_dim in (2, -1):
            assert r.shape == torch.Size([*shape, 1, 3, 2])
            assert r["a"].shape == torch.Size([*shape, 1, 3, 2])  # access tensor
        assert (r["a"] == 0).all()

    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda")
    @pytest.mark.parametrize("stack_dim", [0, 1, 2, -3, -2, -1])
    def test_to(self, stack_dim):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec(shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], stack_dim)
        assert isinstance(c, LazyStackedCompositeSpec)
        cdevice = c.to("cuda:0")
        assert cdevice.device != c.device
        assert cdevice.device == torch.device("cuda:0")
        if stack_dim < 0:
            stack_dim += 3
        index = (slice(None),) * stack_dim + (0,)
        assert cdevice[index].device == torch.device("cuda:0")

    def test_clone(self):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec(shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], 0)
        cclone = c.clone()
        assert cclone[0] is not c[0]
        assert cclone[0] == c[0]


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
