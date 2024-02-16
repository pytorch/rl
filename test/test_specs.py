# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import contextlib

import numpy as np
import pytest
import torch
import torchrl.data.tensor_specs
from _utils_internal import get_available_devices, get_default_devices, set_global_var
from scipy.stats import chisquare
from tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase
from tensordict.utils import _unravel_key_to_tuple

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
    TensorSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.data.utils import check_no_exclusive_keys, consolidate_spec


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
@pytest.mark.parametrize("shape", [[], torch.Size([3])])
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

        if not shape:
            assert (ts.encode(ts.to_numpy(r)) == r).all()
        else:
            with pytest.raises(RuntimeError, match="Shape mismatch"):
                ts.encode(ts.to_numpy(r))
            assert (ts.expand(*shape, *ts.shape).encode(ts.to_numpy(r)) == r).all()

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
        if not shape:
            assert (ts.encode(ts.to_numpy(r)) == r).all()
        else:
            with pytest.raises(RuntimeError, match="Shape mismatch"):
                ts.encode(ts.to_numpy(r))
            assert (ts.expand(*shape, *ts.shape).encode(ts.to_numpy(r)) == r).all()


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
        if not shape:
            assert (ts.encode(ts.to_numpy(r)) == r).all()
        else:
            with pytest.raises(RuntimeError, match="Shape mismatch"):
                ts.encode(ts.to_numpy(r))
            assert (ts.expand(*shape, *ts.shape).encode(ts.to_numpy(r)) == r).all()


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
        # assert (ts.encode(categorical) == r).all()
        if not shape:
            assert (ts.encode(categorical) == r).all()
        else:
            with pytest.raises(RuntimeError, match="is invalid for input of size"):
                ts.encode(categorical)
            assert (ts.expand(*shape, *ts.shape).encode(categorical) == r).all()


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
@pytest.mark.parametrize("shape", [None, [], torch.Size([3]), torch.Size([4, 5])])
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


@pytest.mark.parametrize("n", [1, 4, 7, 99])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("shape", [None, [], [1], [1, 2]])
def test_discrete_conversion(n, device, shape):
    categorical = DiscreteTensorSpec(n, device=device, shape=shape)
    shape_one_hot = [n] if not shape else [*shape, n]
    one_hot = OneHotDiscreteTensorSpec(n, device=device, shape=shape_one_hot)

    assert categorical != one_hot
    assert categorical.to_one_hot_spec() == one_hot
    assert one_hot.to_categorical_spec() == categorical

    assert categorical.is_in(one_hot.to_categorical(one_hot.rand(shape)))
    assert one_hot.is_in(categorical.to_one_hot(categorical.rand(shape)))


@pytest.mark.parametrize("ns", [[5], [5, 2, 3], [4, 5, 1, 3]])
@pytest.mark.parametrize("shape", [torch.Size([3]), torch.Size([4, 5])])
@pytest.mark.parametrize("device", get_default_devices())
def test_multi_discrete_conversion(ns, shape, device):
    categorical = MultiDiscreteTensorSpec(ns, device=device)
    one_hot = MultiOneHotDiscreteTensorSpec(ns, device=device)

    assert categorical != one_hot
    assert categorical.to_one_hot_spec() == one_hot
    assert one_hot.to_categorical_spec() == categorical

    assert categorical.is_in(one_hot.to_categorical(one_hot.rand(shape)))
    assert one_hot.is_in(categorical.to_one_hot(categorical.rand(shape)))


@pytest.mark.parametrize("is_complete", [True, False])
@pytest.mark.parametrize("device", [None, *get_default_devices()])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
@pytest.mark.parametrize("shape", [(), (2, 3)])
class TestComposite:
    @staticmethod
    def _composite_spec(shape, is_complete=True, device=None, dtype=None):
        torch.manual_seed(0)
        np.random.seed(0)

        return CompositeSpec(
            obs=BoundedTensorSpec(
                torch.zeros(*shape, 3, 32, 32),
                torch.ones(*shape, 3, 32, 32),
                dtype=dtype,
                device=device,
            ),
            act=UnboundedContinuousTensorSpec(
                (
                    *shape,
                    7,
                ),
                dtype=dtype,
                device=device,
            )
            if is_complete
            else None,
            shape=shape,
            device=device,
        )

    def test_getitem(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        assert isinstance(ts["obs"], BoundedTensorSpec)
        if is_complete:
            assert isinstance(ts["act"], UnboundedContinuousTensorSpec)
        else:
            assert ts["act"] is None
        with pytest.raises(KeyError):
            _ = ts["UNK"]

    def test_setitem_forbidden_keys(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        for key in {"shape", "device", "dtype", "space"}:
            with pytest.raises(AttributeError, match="cannot be set"):
                ts[key] = 42

    @pytest.mark.parametrize("dest", get_available_devices())
    def test_setitem_matches_device(self, shape, is_complete, device, dtype, dest):
        ts = self._composite_spec(shape, is_complete, device, dtype)

        ts["good"] = UnboundedContinuousTensorSpec(
            shape=shape, device=device, dtype=dtype
        )
        cm = (
            contextlib.nullcontext()
            if (device == dest) or (device is None)
            else pytest.raises(
                RuntimeError, match="All devices of CompositeSpec must match"
            )
        )
        with cm:
            # auto-casting is introduced since v0.3
            ts["bad"] = UnboundedContinuousTensorSpec(
                shape=shape, device=dest, dtype=dtype
            )
            assert ts.device == device
            assert ts["good"].device == (
                device if device is not None else torch.zeros(()).device
            )
            assert ts["bad"].device == (device if device is not None else dest)

    def test_del(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        assert "obs" in ts.keys()
        assert "act" in ts.keys()
        del ts["obs"]
        assert "obs" not in ts.keys()
        assert "act" in ts.keys()

    def test_encode(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
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

    def test_is_in(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        for _ in range(100):
            r = ts.rand()
            assert ts.is_in(r)

    def test_to_numpy(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        for _ in range(100):
            r = ts.rand()
            for key, value in ts.to_numpy(r).items():
                spec = ts[key]
                assert (spec.to_numpy(r[key]) == value).all()

    @pytest.mark.parametrize("shape_other", [[], [5]])
    def test_project(self, shape, is_complete, device, dtype, shape_other):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        # Using normal distribution to get out of bounds
        shape = (*shape_other, *shape)
        tensors = {"obs": torch.randn(*shape, 3, 32, 32, dtype=dtype, device=device)}
        if is_complete:
            tensors["act"] = torch.randn(*shape, 7, dtype=dtype, device=device)
        out_of_bounds_td = TensorDict(tensors, batch_size=shape)

        assert not ts.is_in(out_of_bounds_td)
        ts.project(out_of_bounds_td)
        assert ts.is_in(out_of_bounds_td)
        assert out_of_bounds_td.shape == torch.Size(shape)

    @pytest.mark.parametrize("shape_other", [[], [3]])
    def test_rand(self, shape, is_complete, device, dtype, shape_other):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        if dtype is None:
            dtype = torch.get_default_dtype()
        shape = (*shape_other, *shape)
        rand_td = ts.rand(shape_other)
        assert rand_td.shape == torch.Size(shape)
        assert rand_td.get("obs").shape == torch.Size([*shape, 3, 32, 32])
        assert rand_td.get("obs").dtype == dtype
        if is_complete:
            assert rand_td.get("act").shape == torch.Size([*shape, 7])
            assert rand_td.get("act").dtype == dtype

    def test_repr(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        output = repr(ts)
        assert output.startswith("CompositeSpec")
        assert "obs: " in output
        assert "act: " in output

    def test_device_cast_with_dtype_fails(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        with pytest.raises(ValueError, match="Only device casting is allowed"):
            ts.to(torch.float16)

    @pytest.mark.parametrize("dest", get_available_devices())
    def test_device_cast(self, shape, is_complete, device, dtype, dest):
        # Note: trivial test in case there is only one device available.
        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts.rand()
        td_to = ts.to(dest)
        cast_r = td_to.rand()

        assert td_to.device == dest
        assert cast_r["obs"].device == dest
        if is_complete:
            assert cast_r["act"].device == dest

    def test_type_check(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        rand_td = ts.rand()
        ts.type_check(rand_td)
        ts.type_check(rand_td["obs"], "obs")
        if is_complete:
            ts.type_check(rand_td["act"], "act")

    def test_nested_composite_spec(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
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

    def test_nested_composite_spec_index(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"]["nested_cp"] = self._composite_spec(
            shape, is_complete, device, dtype
        )
        assert ts["nested_cp"]["nested_cp"] is ts["nested_cp", "nested_cp"]
        assert (
            ts["nested_cp"]["nested_cp"]["obs"] is ts["nested_cp", "nested_cp", "obs"]
        )

    def test_nested_composite_spec_rand(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"]["nested_cp"] = self._composite_spec(
            shape, is_complete, device, dtype
        )
        r = ts.rand()
        assert (r["nested_cp", "nested_cp", "obs"] >= 0).all()

    def test_nested_composite_spec_zero(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"]["nested_cp"] = self._composite_spec(
            shape, is_complete, device, dtype
        )
        r = ts.zero()
        assert (r["nested_cp", "nested_cp", "obs"] == 0).all()

    def test_nested_composite_spec_setitem(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"]["nested_cp"] = self._composite_spec(
            shape, is_complete, device, dtype
        )
        ts["nested_cp", "nested_cp", "obs"] = None
        assert (
            ts["nested_cp"]["nested_cp"]["obs"] is ts["nested_cp", "nested_cp", "obs"]
        )
        assert ts["nested_cp"]["nested_cp"]["obs"] is None
        ts["nested_cp", "another", "obs"] = None

    def test_nested_composite_spec_delitem(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"]["nested_cp"] = self._composite_spec(
            shape, is_complete, device, dtype
        )
        del ts["nested_cp", "nested_cp", "obs"]
        assert ("nested_cp", "nested_cp", "obs") not in ts.keys(True, True)

    def test_nested_composite_spec_update(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
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

        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
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

        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
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

        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested_cp"] = self._composite_spec(shape, is_complete, device, dtype)
        td2 = CompositeSpec(
            nested_cp=CompositeSpec(act=None, shape=shape).to(device), shape=shape
        )
        ts.update(td2)
        td2 = CompositeSpec(
            nested_cp=CompositeSpec(
                act=UnboundedContinuousTensorSpec(shape=shape, device=device),
                shape=shape,
            ),
            shape=shape,
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

    def test_change_batch_size(self, shape, is_complete, device, dtype):
        ts = self._composite_spec(shape, is_complete, device, dtype)
        ts["nested"] = CompositeSpec(
            leaf=UnboundedContinuousTensorSpec(shape, device=device),
            shape=shape,
            device=device,
        )
        ts = ts.expand(3, *shape)
        assert ts["nested"].shape == (3, *shape)
        assert ts["nested", "leaf"].shape == (3, *shape)
        ts.shape = ()
        # this does not change
        assert ts["nested"].shape == (3, *shape)
        assert ts.shape == ()
        ts["nested"].shape = ()
        ts.shape = (3,)
        assert ts.shape == (3,)
        assert ts["nested"].shape == (3,)


@pytest.mark.parametrize("shape", [(), (2, 3)])
@pytest.mark.parametrize("device", get_default_devices())
def test_create_composite_nested(shape, device):
    d = [
        {("a", "b"): UnboundedContinuousTensorSpec(shape=shape, device=device)},
        {"a": {"b": UnboundedContinuousTensorSpec(shape=shape, device=device)}},
    ]
    for _d in d:
        c = CompositeSpec(_d, shape=shape)
        assert isinstance(c["a", "b"], UnboundedContinuousTensorSpec)
        assert c["a"].shape == torch.Size(shape)
        assert c.device is None  # device not explicitly passed
        assert c["a"].device is None  # device not explicitly passed
        assert c["a", "b"].device == device
        c = c.to(device)
        assert c.device == device
        assert c["a"].device == device


@pytest.mark.parametrize("recurse", [True, False])
def test_lock(recurse):
    shape = [3, 4, 5]
    spec = CompositeSpec(
        a=CompositeSpec(
            b=CompositeSpec(shape=shape[:3], device="cpu"), shape=shape[:2]
        ),
        shape=shape[:1],
    )
    spec["a"] = spec["a"].clone()
    spec["a", "b"] = spec["a", "b"].clone()
    assert not spec.locked
    spec.lock_(recurse=recurse)
    assert spec.locked
    with pytest.raises(RuntimeError, match="Cannot modify a locked CompositeSpec."):
        spec["a"] = spec["a"].clone()
    with pytest.raises(RuntimeError, match="Cannot modify a locked CompositeSpec."):
        spec.set("a", spec["a"].clone())
    if recurse:
        assert spec["a"].locked
        with pytest.raises(RuntimeError, match="Cannot modify a locked CompositeSpec."):
            spec["a"].set("b", spec["a", "b"].clone())
        with pytest.raises(RuntimeError, match="Cannot modify a locked CompositeSpec."):
            spec["a", "b"] = spec["a", "b"].clone()
    else:
        assert not spec["a"].locked
        spec["a", "b"] = spec["a", "b"].clone()
        spec["a"].set("b", spec["a", "b"].clone())
    spec.unlock_(recurse=recurse)
    spec["a"] = spec["a"].clone()
    spec["a", "b"] = spec["a", "b"].clone()
    spec["a"].set("b", spec["a", "b"].clone())


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
        if torch.cuda.device_count():
            ts_other = BoundedTensorSpec(
                minimum, maximum, torch.Size((1,)), "cuda:0", dtype
            )
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

        if torch.cuda.device_count():
            ts_other = OneHotDiscreteTensorSpec(
                n=n, device="cuda:0", dtype=dtype, use_register=use_register
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

        if torch.cuda.device_count():
            ts_other = UnboundedContinuousTensorSpec(device="cuda:0", dtype=dtype)
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

        ts = BoundedTensorSpec(low=minimum, high=maximum, device=device, dtype=dtype)

        ts_same = BoundedTensorSpec(
            low=minimum, high=maximum, device=device, dtype=dtype
        )
        assert ts == ts_same

        ts_other = BoundedTensorSpec(
            low=minimum + 1, high=maximum, device=device, dtype=dtype
        )
        assert ts != ts_other

        ts_other = BoundedTensorSpec(
            low=minimum, high=maximum + 1, device=device, dtype=dtype
        )
        assert ts != ts_other

        if torch.cuda.device_count():
            ts_other = BoundedTensorSpec(
                low=minimum, high=maximum, device="cuda:0", dtype=dtype
            )
            assert ts != ts_other

        ts_other = BoundedTensorSpec(
            low=minimum, high=maximum, device=device, dtype=torch.float64
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

        if torch.cuda.device_count():
            ts_other = DiscreteTensorSpec(
                n=n, shape=shape, device="cuda:0", dtype=dtype
            )
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

        if torch.cuda.device_count():
            ts_other = UnboundedContinuousTensorSpec(
                shape=shape, device="cuda:0", dtype=dtype
            )
            assert ts != ts_other

        ts_other = UnboundedContinuousTensorSpec(
            shape=shape, device=device, dtype=torch.float64
        )
        assert ts != ts_other

        ts_other = TestEquality._ts_make_all_fields_equal(
            BoundedTensorSpec(0, 1, torch.Size((1,)), device, dtype), ts
        )
        # Unbounded and bounded without space are technically the same
        assert ts == ts_other

    def test_equality_binary(self):
        n = 5
        device = "cpu"
        dtype = torch.float16

        ts = BinaryDiscreteTensorSpec(n=n, device=device, dtype=dtype)

        ts_same = BinaryDiscreteTensorSpec(n=n, device=device, dtype=dtype)
        assert ts == ts_same

        ts_other = BinaryDiscreteTensorSpec(n=n + 5, device=device, dtype=dtype)
        assert ts != ts_other

        if torch.cuda.device_count():
            ts_other = BinaryDiscreteTensorSpec(n=n, device="cuda:0", dtype=dtype)
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

        if torch.cuda.device_count():
            ts_other = MultiOneHotDiscreteTensorSpec(
                nvec=nvec, device="cuda:0", dtype=dtype
            )
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

        if torch.cuda.device_count():
            ts_other = MultiDiscreteTensorSpec(nvec=nvec, device="cuda:0", dtype=dtype)
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
            low=minimum, high=maximum + 1, device=device, dtype=dtype
        )
        nd_same = BoundedTensorSpec(
            low=minimum, high=maximum + 1, device=device, dtype=dtype
        )
        _ = BoundedTensorSpec(low=minimum, high=maximum + 3, device=device, dtype=dtype)

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

        sample = action_spec.rand((100000,))

        sample_list = sample.long().argmax(-1)
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
    @pytest.mark.parametrize("shape1", [None, (4,), (5, 4)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
    def test_discrete(
        self,
        shape1,
    ):
        spec = DiscreteTensorSpec(n=4, shape=shape1, device="cpu", dtype=torch.long)
        assert spec == spec.clone()
        assert spec is not spec.clone()

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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

    @pytest.mark.parametrize("shape1", [None, (), (5,)])
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


class TestUnbind:
    @pytest.mark.parametrize("shape1", [(5, 4)])
    def test_binary(self, shape1):
        spec = BinaryDiscreteTensorSpec(
            n=4, shape=shape1, device="cpu", dtype=torch.bool
        )
        assert spec == torch.stack(spec.unbind(0), 0)
        with pytest.raises(ValueError):
            spec.unbind(-1)

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
        assert spec == torch.stack(spec.unbind(0), 0)
        with pytest.raises(ValueError):
            spec.unbind(-1)

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
        assert spec == torch.stack(spec.unbind(0), 0)
        assert spec == torch.stack(spec.unbind(-1), -1)

    @pytest.mark.parametrize("shape1", [(5,), (5, 6)])
    def test_discrete(
        self,
        shape1,
    ):
        spec = DiscreteTensorSpec(n=4, shape=shape1, device="cpu", dtype=torch.long)
        assert spec == torch.stack(spec.unbind(0), 0)
        assert spec == torch.stack(spec.unbind(-1), -1)

    @pytest.mark.parametrize("shape1", [(5,), (5, 6)])
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
        assert spec == torch.stack(spec.unbind(0), 0)
        with pytest.raises(ValueError):
            spec.unbind(-1)

    @pytest.mark.parametrize("shape1", [(5,), (5, 6)])
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
        assert spec == torch.stack(spec.unbind(0), 0)
        with pytest.raises(ValueError):
            spec.unbind(-1)

    @pytest.mark.parametrize("shape1", [(5,), (5, 6)])
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
        assert spec == torch.stack(spec.unbind(0), 0)
        with pytest.raises(ValueError):
            spec.unbind(-1)

    @pytest.mark.parametrize("shape1", [(5,), (5, 6)])
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
        assert spec == torch.stack(spec.unbind(0), 0)
        assert spec == torch.stack(spec.unbind(-1), -1)

    @pytest.mark.parametrize("shape1", [(5,), (5, 6)])
    def test_unboundeddiscrete(
        self,
        shape1,
    ):
        if shape1 is None:
            shape1 = (15,)
        else:
            shape1 = (*shape1, 15)
        spec = UnboundedDiscreteTensorSpec(shape=shape1, device="cpu", dtype=torch.long)
        assert spec == torch.stack(spec.unbind(0), 0)
        assert spec == torch.stack(spec.unbind(-1), -1)


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
        c = torch.stack([c1, c2], stack_dim)
        r = c.rand()
        assert r.shape == c.shape

    def test_stack_unboundeddiscrete_zero(self, shape, stack_dim):
        shape = (*shape,)
        c1 = UnboundedDiscreteTensorSpec(shape=shape)
        c2 = c1.clone()
        c = torch.stack([c1, c2], stack_dim)
        r = c.zero()
        assert r.shape == c.shape

    def test_to_numpy(self, shape, stack_dim):
        c1 = BoundedTensorSpec(-1, 1, shape=shape, dtype=torch.float64)
        c2 = BoundedTensorSpec(-1, 1, shape=shape, dtype=torch.float64)

        c = torch.stack([c1, c2], stack_dim)

        torch.manual_seed(0)

        shape = list(shape)
        shape.insert(stack_dim, 2)
        shape = tuple(shape)

        val = 2 * torch.rand(torch.Size(shape)) - 1

        val_np = c.to_numpy(val)
        assert isinstance(val_np, np.ndarray)
        assert (val.numpy() == val_np).all()

        with pytest.raises(AssertionError):
            c.to_numpy(val + 1, safe=True)

    def test_malformed_stack(self, shape, stack_dim):
        c1 = BoundedTensorSpec(-1, 1, shape=shape, dtype=torch.float64)
        c2 = BoundedTensorSpec(-1, 1, shape=shape, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Dtypes differ"):
            torch.stack([c1, c2], stack_dim)

        c1 = BoundedTensorSpec(-1, 1, shape=shape, dtype=torch.float32)
        c2 = UnboundedContinuousTensorSpec(shape=shape, dtype=torch.float32)
        c3 = UnboundedDiscreteTensorSpec(shape=shape, dtype=torch.float32)
        with pytest.raises(
            RuntimeError,
            match="Stacking specs cannot occur: Found more than one type of specs in the list.",
        ):
            torch.stack([c1, c2], stack_dim)
            torch.stack([c3, c2], stack_dim)

        c1 = BoundedTensorSpec(-1, 1, shape=shape, dtype=torch.float32)
        c2 = BoundedTensorSpec(-1, 1, shape=shape + (3,), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Ndims differ"):
            torch.stack([c1, c2], stack_dim)


class TestDenseStackedCompositeSpecs:
    def test_stack(self):
        c1 = CompositeSpec(a=UnboundedContinuousTensorSpec())
        c2 = c1.clone()
        c = torch.stack([c1, c2], 0)
        assert isinstance(c, CompositeSpec)


class TestLazyStackedCompositeSpecs:
    def _get_heterogeneous_specs(
        self,
        batch_size=(),
        stack_dim: int = 0,
    ):
        shared = BoundedTensorSpec(low=0, high=1, shape=(*batch_size, 32, 32, 3))
        hetero_3d = UnboundedContinuousTensorSpec(
            shape=(
                *batch_size,
                3,
            )
        )
        hetero_2d = UnboundedContinuousTensorSpec(
            shape=(
                *batch_size,
                2,
            )
        )
        lidar = BoundedTensorSpec(
            low=0,
            high=5,
            shape=(
                *batch_size,
                20,
            ),
        )

        individual_0_obs = CompositeSpec(
            {
                "individual_0_obs_0": UnboundedContinuousTensorSpec(
                    shape=(
                        *batch_size,
                        3,
                        1,
                    )
                )
            },
            shape=(*batch_size, 3),
        )
        individual_1_obs = CompositeSpec(
            {
                "individual_1_obs_0": BoundedTensorSpec(
                    low=0, high=3, shape=(*batch_size, 3, 1, 2)
                )
            },
            shape=(*batch_size, 3),
        )
        individual_2_obs = CompositeSpec(
            {
                "individual_1_obs_0": UnboundedContinuousTensorSpec(
                    shape=(*batch_size, 3, 1, 2, 3)
                )
            },
            shape=(*batch_size, 3),
        )

        spec_list = [
            CompositeSpec(
                {
                    "shared": shared,
                    "lidar": lidar,
                    "hetero": hetero_3d,
                    "individual_0_obs": individual_0_obs,
                },
                shape=batch_size,
            ),
            CompositeSpec(
                {
                    "shared": shared,
                    "lidar": lidar,
                    "hetero": hetero_2d,
                    "individual_1_obs": individual_1_obs,
                },
                shape=batch_size,
            ),
            CompositeSpec(
                {
                    "shared": shared,
                    "hetero": hetero_2d,
                    "individual_2_obs": individual_2_obs,
                },
                shape=batch_size,
            ),
        ]

        return torch.stack(spec_list, dim=stack_dim).cpu()

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

    def test_to_numpy(self):
        c1 = CompositeSpec(a=BoundedTensorSpec(-1, 1, shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=BoundedTensorSpec(-1, 1, shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], 0)
        for _ in range(100):
            r = c.rand()
            for key, value in c.to_numpy(r).items():
                spec = c[key]
                assert (spec.to_numpy(r[key]) == value).all()

        td_fail = TensorDict({"a": torch.rand((2, 1, 3)) + 1}, [2, 1, 3])
        with pytest.raises(AssertionError):
            c.to_numpy(td_fail, safe=True)

    def test_unsqueeze(self):
        c1 = CompositeSpec(a=BoundedTensorSpec(-1, 1, shape=(1, 3)), shape=(1, 3))
        c2 = CompositeSpec(
            a=BoundedTensorSpec(-1, 1, shape=(1, 3)),
            b=UnboundedDiscreteTensorSpec(shape=(1, 3)),
            shape=(1, 3),
        )
        c = torch.stack([c1, c2], 1)
        for unsq in range(-2, 3):
            cu = c.unsqueeze(unsq)
            shape = list(c.shape)
            new_unsq = unsq if unsq >= 0 else c.ndim + unsq + 1
            shape.insert(new_unsq, 1)
            assert cu.shape == torch.Size(shape)
            cus = cu.squeeze(unsq)
            assert c.shape == cus.shape, unsq
            assert cus == c

        assert c.squeeze().shape == torch.Size([2, 3])

        c = self._get_heterogeneous_specs()
        cu = c.unsqueeze(0)
        assert cu.shape == torch.Size([1, 3])
        cus = cu.squeeze(0)
        assert cus == c

    @pytest.mark.parametrize("batch_size", [(), (4,), (4, 2)])
    def test_len(self, batch_size):
        c = self._get_heterogeneous_specs(batch_size=batch_size)
        assert len(c) == c.shape[0]
        assert len(c) == len(c.rand())

    @pytest.mark.parametrize("batch_size", [(), (4,), (4, 2)])
    def test_eq(self, batch_size):
        c = self._get_heterogeneous_specs(batch_size=batch_size)
        c2 = self._get_heterogeneous_specs(batch_size=batch_size)

        assert c == c2 and not c != c2
        assert c == c.clone() and not c != c.clone()

        del c2["shared"]
        assert not c == c2 and c != c2

        c2 = self._get_heterogeneous_specs(batch_size=batch_size)
        del c2[0]["lidar"]

        assert not c == c2 and c != c2

        c2 = self._get_heterogeneous_specs(batch_size=batch_size)
        c2[0]["lidar"].space.low += 1
        assert not c == c2 and c != c2

    @pytest.mark.parametrize("batch_size", [(), (4,), (4, 2)])
    @pytest.mark.parametrize("include_nested", [True, False])
    @pytest.mark.parametrize("leaves_only", [True, False])
    def test_del(self, batch_size, include_nested, leaves_only):
        c = self._get_heterogeneous_specs(batch_size=batch_size)
        td_c = c.rand()

        keys = list(c.keys(include_nested=include_nested, leaves_only=leaves_only))
        for k in keys:
            del c[k]
            del td_c[k]
        assert len(c.keys(include_nested=include_nested, leaves_only=leaves_only)) == 0
        assert (
            len(td_c.keys(include_nested=include_nested, leaves_only=leaves_only)) == 0
        )

        keys = list(c[0].keys(include_nested=include_nested, leaves_only=leaves_only))
        for k in keys:
            del c[k]
            del td_c[k]
        assert (
            len(c[0].keys(include_nested=include_nested, leaves_only=leaves_only)) == 0
        )
        assert (
            len(td_c[0].keys(include_nested=include_nested, leaves_only=leaves_only))
            == 0
        )
        with pytest.raises(KeyError):
            del c["individual_1_obs_0"]
        with pytest.raises(KeyError):
            del td_c["individual_1_obs_0"]

        del c[("individual_1_obs", "individual_1_obs_0")]
        del td_c[("individual_1_obs", "individual_1_obs_0")]

    @pytest.mark.parametrize("batch_size", [(), (4,), (4, 2)])
    def test_is_in(self, batch_size):
        c = self._get_heterogeneous_specs(batch_size=batch_size)
        td_c = c.rand()
        assert c.is_in(td_c)

        del td_c["shared"]
        with pytest.raises(KeyError):
            assert not c.is_in(td_c)

        td_c = c.rand()
        del td_c[("individual_1_obs", "individual_1_obs_0")]
        with pytest.raises(KeyError):
            assert not c.is_in(td_c)

        td_c = c.rand()
        td_c["shared"] += 1
        assert not c.is_in(td_c)

        td_c = c.rand()
        td_c[1]["individual_1_obs", "individual_1_obs_0"] += 4
        assert not c.is_in(td_c)

        td_c = c.rand()
        td_c[0]["individual_0_obs", "individual_0_obs_0"] += 1
        assert c.is_in(td_c)

    def test_type_check(self):
        c = self._get_heterogeneous_specs()
        td_c = c.rand()

        c.type_check(td_c)
        c.type_check(td_c["shared"], "shared")

    @pytest.mark.parametrize("batch_size", [(), (4,), (4, 2)])
    def test_project(self, batch_size):
        c = self._get_heterogeneous_specs(batch_size=batch_size)
        td_c = c.rand()
        assert c.is_in(td_c)
        val = c.project(td_c)
        assert c.is_in(val)

        del td_c["shared"]
        with pytest.raises(KeyError):
            c.is_in(td_c)

        td_c = c.rand()
        del td_c[("individual_1_obs", "individual_1_obs_0")]
        with pytest.raises(KeyError):
            c.is_in(td_c)

        td_c = c.rand()
        td_c["shared"] += 1
        assert not c.is_in(td_c)
        val = c.project(td_c)
        assert c.is_in(val)

        td_c = c.rand()
        td_c[1]["individual_1_obs", "individual_1_obs_0"] += 4
        assert not c.is_in(td_c)
        val = c.project(td_c)
        assert c.is_in(val)

        td_c = c.rand()
        td_c[0]["individual_0_obs", "individual_0_obs_0"] += 1
        assert c.is_in(td_c)

    def test_repr(self):
        c = self._get_heterogeneous_specs()

        expected = f"""LazyStackedCompositeSpec(
    fields={{
        hetero: LazyStackedUnboundedContinuousTensorSpec(
            shape=torch.Size([3, -1]), device=cpu, dtype=torch.float32, domain=continuous),
        shared: BoundedTensorSpec(
            shape=torch.Size([3, 32, 32, 3]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([3, 32, 32, 3]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([3, 32, 32, 3]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous)}},
    exclusive_fields={{
        0 ->
            lidar: BoundedTensorSpec(
                shape=torch.Size([20]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([20]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([20]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous),
            individual_0_obs: CompositeSpec(
                individual_0_obs_0: UnboundedContinuousTensorSpec(
                    shape=torch.Size([3, 1]),
                    space=None,
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([3])),
        1 ->
            lidar: BoundedTensorSpec(
                shape=torch.Size([20]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([20]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([20]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous),
            individual_1_obs: CompositeSpec(
                individual_1_obs_0: BoundedTensorSpec(
                    shape=torch.Size([3, 1, 2]),
                    space=ContinuousBox(
                        low=Tensor(shape=torch.Size([3, 1, 2]), device=cpu, dtype=torch.float32, contiguous=True),
                        high=Tensor(shape=torch.Size([3, 1, 2]), device=cpu, dtype=torch.float32, contiguous=True)),
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([3])),
        2 ->
            individual_2_obs: CompositeSpec(
                individual_1_obs_0: UnboundedContinuousTensorSpec(
                    shape=torch.Size([3, 1, 2, 3]),
                    space=None,
                    device=cpu,
                    dtype=torch.float32,
                    domain=continuous), device=cpu, shape=torch.Size([3]))}},
    device=cpu,
    shape={torch.Size((3,))},
    stack_dim={c.stack_dim})"""
        assert expected == repr(c)

        c = c[0:2]
        del c["individual_0_obs"]
        del c["individual_1_obs"]
        expected = f"""LazyStackedCompositeSpec(
    fields={{
        hetero: LazyStackedUnboundedContinuousTensorSpec(
            shape=torch.Size([2, -1]), device=cpu, dtype=torch.float32, domain=continuous),
        lidar: BoundedTensorSpec(
            shape=torch.Size([2, 20]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([2, 20]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([2, 20]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous),
        shared: BoundedTensorSpec(
            shape=torch.Size([2, 32, 32, 3]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([2, 32, 32, 3]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([2, 32, 32, 3]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous)}},
    exclusive_fields={{
    }},
    device=cpu,
    shape={torch.Size((2,))},
    stack_dim={c.stack_dim})"""
        assert expected == repr(c)

    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 1)])
    def test_consolidate_spec(self, batch_size):
        spec = self._get_heterogeneous_specs(batch_size)
        spec_lazy = spec.clone()

        assert not check_no_exclusive_keys(spec_lazy)

        spec_lazy = consolidate_spec(spec_lazy, recurse_through_entries=False)
        assert check_no_exclusive_keys(spec_lazy, recurse=False)

        spec_lazy = consolidate_spec(spec_lazy, recurse_through_entries=True)
        assert check_no_exclusive_keys(spec_lazy, recurse=True)

        assert get_all_keys(spec, include_exclusive=True) == get_all_keys(
            spec_lazy, include_exclusive=False
        )

    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 1)])
    def test_consolidate_spec_exclusive_lazy_stacked(self, batch_size):
        shared = UnboundedContinuousTensorSpec(
            shape=(
                *batch_size,
                5,
                5,
                5,
            )
        )
        lazy_spec = torch.stack(
            [
                UnboundedContinuousTensorSpec(shape=(*batch_size, 5, 6, 7)),
                UnboundedContinuousTensorSpec(shape=(*batch_size, 5, 7, 7)),
                UnboundedContinuousTensorSpec(shape=(*batch_size, 5, 8, 7)),
                UnboundedContinuousTensorSpec(shape=(*batch_size, 5, 8, 7)),
            ],
            dim=len(batch_size),
        )

        spec_list = [
            CompositeSpec(
                {
                    "shared": shared,
                    "lazy_spec": lazy_spec,
                },
                shape=batch_size,
            ),
            CompositeSpec(
                {
                    "shared": shared,
                },
                shape=batch_size,
            ),
            CompositeSpec(
                {},
                shape=batch_size,
                device="cpu",
            ),
        ]

        spec = torch.stack(spec_list, dim=0)
        spec_consolidated = consolidate_spec(spec)

        assert spec_consolidated["shared"].shape == (3, *batch_size, -1, -1, -1)
        assert spec_consolidated["lazy_spec"].shape == (3, *batch_size, 4, 5, -1, 7)

        assert check_no_exclusive_keys(spec_consolidated, recurse=True)
        assert get_all_keys(spec, include_exclusive=True) == get_all_keys(
            spec_consolidated, include_exclusive=False
        )

    @pytest.mark.parametrize("batch_size", [(2,), (2, 1)])
    def test_update(self, batch_size, stack_dim=0):
        spec = self._get_heterogeneous_specs(batch_size, stack_dim)
        spec2 = self._get_heterogeneous_specs(batch_size, stack_dim)

        del spec2["shared"]
        spec2["hetero"] = spec2["hetero"].unsqueeze(-1)
        assert spec["hetero"].shape == (3, *batch_size, -1)
        spec.update(spec2)
        assert spec["hetero"].shape == (3, *batch_size, -1, 1)

        spec2[1]["individual_1_obs"]["individual_1_obs_0"].space.low += 1
        assert spec[1]["individual_1_obs"]["individual_1_obs_0"].space.low.sum() == 0
        spec.update(spec2)
        assert (
            spec[1]["individual_1_obs"]["individual_1_obs_0"].space.low.sum() == 0
        )  # Only non exclusive keys will be updated

        new = torch.stack(
            [UnboundedContinuousTensorSpec(shape=(*batch_size, i)) for i in range(3)], 0
        )
        spec2["new"] = new
        spec.update(spec2)
        assert spec["new"] == new

    @pytest.mark.parametrize("batch_size", [(2,), (2, 1)])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_set_item(self, batch_size, stack_dim):
        spec = self._get_heterogeneous_specs(batch_size, stack_dim)

        new = torch.stack(
            [UnboundedContinuousTensorSpec(shape=(*batch_size, i)) for i in range(3)],
            stack_dim,
        )
        spec["new"] = new
        assert spec["new"] == new

        new = new.unsqueeze(-1)
        spec["new"] = new
        assert spec["new"] == new

        new = new.squeeze(-1)
        assert spec["new"] == new.unsqueeze(-1)

        spec[("other", "key")] = new
        assert spec[("other", "key")] == new
        assert isinstance(spec["other"], LazyStackedCompositeSpec)

        with pytest.raises(RuntimeError, match="key should be a Sequence<NestedKey>"):
            spec[0] = new

        comp = torch.stack(
            [
                CompositeSpec(
                    {"a": UnboundedContinuousTensorSpec(shape=(*batch_size, i))},
                    shape=batch_size,
                )
                for i in range(3)
            ],
            stack_dim,
        )
        spec["comp"] = comp
        assert spec["comp"] == comp.to(spec.device)
        assert spec["comp", "a"] == new.to(spec.device)


# MultiDiscreteTensorSpec: Pending resolution of https://github.com/pytorch/pytorch/issues/100080.
@pytest.mark.parametrize(
    "spec_class",
    [
        BinaryDiscreteTensorSpec,
        OneHotDiscreteTensorSpec,
        MultiOneHotDiscreteTensorSpec,
        CompositeSpec,
    ],
)
@pytest.mark.parametrize(
    "idx",
    [
        5,
        (0, 1),
        range(10),
        np.array([[2, 10]]),
        (slice(None), slice(1, 2), 1),
        (1, ..., 2, ..., 3),
        (1, 1, 1, 1),
        torch.tensor([10, 2]),
    ],  # [:,1:2,1]
)
def test_invalid_indexing(spec_class, idx):
    if spec_class in [BinaryDiscreteTensorSpec, OneHotDiscreteTensorSpec]:
        spec = spec_class(n=4, shape=[3, 4])
    elif spec_class == MultiDiscreteTensorSpec:
        spec = spec_class([2, 2, 2], shape=[3])
    elif spec_class == MultiOneHotDiscreteTensorSpec:
        spec = spec_class([4], shape=[3, 4])
    elif spec_class == CompositeSpec:
        spec = spec_class(k=UnboundedDiscreteTensorSpec(shape=(3, 4)), shape=(3,))
    with pytest.raises(IndexError):
        spec[idx]


# BoundedTensorSpec, MultiDiscreteTensorSpec: Pending resolution of https://github.com/pytorch/pytorch/issues/100080.
@pytest.mark.parametrize(
    "spec_class",
    [
        BinaryDiscreteTensorSpec,
        DiscreteTensorSpec,
        MultiOneHotDiscreteTensorSpec,
        OneHotDiscreteTensorSpec,
        UnboundedContinuousTensorSpec,
        UnboundedDiscreteTensorSpec,
        CompositeSpec,
    ],
)
def test_valid_indexing(spec_class):
    # Default args. UnboundedContinuousTensorSpec, UnboundedDiscreteTensorSpec, MultiDiscreteTensorSpec, MultiOneHotDiscreteTensorSpec
    args = {"0d": [], "2d": [], "3d": [], "4d": [], "5d": []}
    kwargs = {}
    if spec_class in [
        BinaryDiscreteTensorSpec,
        DiscreteTensorSpec,
        OneHotDiscreteTensorSpec,
    ]:
        args = {"0d": [0], "2d": [3], "3d": [4], "4d": [6], "5d": [7]}
    elif spec_class == MultiOneHotDiscreteTensorSpec:
        args = {"0d": [[0]], "2d": [[3]], "3d": [[4]], "4d": [[6]], "5d": [[7]]}
    elif spec_class == MultiDiscreteTensorSpec:
        args = {
            "0d": [[0]],
            "2d": [[2] * 3],
            "3d": [[2] * 4],
            "4d": [[1] * 6],
            "5d": [[2] * 7],
        }
    elif spec_class == BoundedTensorSpec:
        min_max = (-1, -1)
        args = {
            "0d": min_max,
            "2d": min_max,
            "3d": min_max,
            "4d": min_max,
            "5d": min_max,
        }
    elif spec_class == CompositeSpec:
        kwargs = {
            "k1": UnboundedDiscreteTensorSpec(shape=(5, 3, 4, 6, 7, 8)),
            "k2": OneHotDiscreteTensorSpec(n=7, shape=(5, 3, 4, 6, 7)),
        }

    spec_0d = spec_class(*args["0d"], **kwargs)
    if spec_class in [
        UnboundedContinuousTensorSpec,
        UnboundedDiscreteTensorSpec,
        CompositeSpec,
    ]:
        spec_0d = spec_class(*args["0d"], shape=[], **kwargs)
    spec_2d = spec_class(*args["2d"], shape=[5, 3], **kwargs)
    spec_3d = spec_class(*args["3d"], shape=[5, 3, 4], **kwargs)
    spec_4d = spec_class(*args["4d"], shape=[5, 3, 4, 6], **kwargs)
    spec_5d = spec_class(*args["5d"], shape=[5, 3, 4, 6, 7], **kwargs)

    # Integers
    assert spec_2d[1].shape == torch.Size([3])
    # Lists
    assert spec_3d[[1, 2]].shape == torch.Size([2, 3, 4])
    assert spec_2d[[0]].shape == torch.Size([1, 3])
    assert spec_2d[[[[0]]]].shape == torch.Size([1, 1, 1, 3])
    assert spec_2d[[0, 1]].shape == torch.Size([2, 3])
    assert spec_2d[[[0, 1]]].shape == torch.Size([1, 2, 3])
    assert spec_3d[[0, 1], [0, 1]].shape == torch.Size([2, 4])
    assert spec_2d[[[0, 1], [0, 1]]].shape == torch.Size([2, 2, 3])
    # Tuples
    assert spec_3d[1, 2].shape == torch.Size([4])
    assert spec_3d[(1, 2)].shape == torch.Size([4])
    assert spec_3d[((1, 2))].shape == torch.Size([4])
    # Ranges
    assert spec_2d[range(2)].shape == torch.Size([2, 3])
    # Slices
    assert spec_2d[:].shape == torch.Size([5, 3])
    assert spec_2d[10:].shape == torch.Size([0, 3])
    assert spec_2d[:1].shape == torch.Size([1, 3])
    assert spec_2d[1:2].shape == torch.Size([1, 3])
    assert spec_2d[10:1:-1].shape == torch.Size([3, 3])
    assert spec_2d[-5:-1].shape == torch.Size([4, 3])
    assert spec_3d[[1, 2], 3:].shape == torch.Size([2, 0, 4])
    # None (adds a singleton dimension where needed)
    assert spec_2d[None].shape == torch.Size([1, 5, 3])
    assert spec_2d[None, :2].shape == torch.Size([1, 2, 3])
    # Ellipsis
    assert spec_2d[1, ...].shape == torch.Size([3])
    # Numpy arrays
    assert spec_2d[np.array([[1, 2]])].shape == torch.Size([1, 2, 3])
    # Tensors
    assert spec_2d[torch.randint(3, (3, 2))].shape == torch.Size([3, 2, 3])
    # Tuples
    # Note: nested tuples are supported by specs but transformed into lists, similarity to numpy
    assert spec_3d[(0, 1), (0, 1)].shape == torch.Size([2, 4])
    assert spec_3d[:2, (0, 1)].shape == torch.Size([2, 2, 4])
    assert spec_3d[:2, [0, 1]].shape == torch.Size([2, 2, 4])
    assert spec_3d[:2, torch.tensor([0, 1])].shape == torch.Size([2, 2, 4])
    assert spec_3d[:2, range(3)].shape == torch.Size([2, 3, 4])
    assert spec_3d[:2, np.array([[1, 2]])].shape == torch.Size([2, 1, 2, 4])
    assert spec_3d[:2, [0]].shape == torch.Size([2, 1, 4])
    assert spec_3d[:2, 0].shape == torch.Size([2, 4])
    assert spec_3d[[0, 1], [0]].shape == torch.Size([2, 4])
    assert spec_4d[:, 1:2, 1].shape == torch.Size([5, 1, 6])
    assert spec_3d[1:, range(3)].shape == torch.Size([4, 3, 4])
    assert spec_3d[[[[[0, 1]]]], [[0]]].shape == torch.Size([1, 1, 1, 2, 4])
    assert spec_3d[0, [[[[0, 1]]]]].shape == torch.Size([1, 1, 1, 2, 4])
    assert spec_3d[0, ((((0, 1))))].shape == torch.Size([2, 4])
    assert spec_3d[((((0, 1)))), [0, 2]].shape == torch.Size([2, 4])
    assert spec_4d[2:, [[[0, 1]]], :3].shape == torch.Size([3, 1, 1, 2, 3, 6])
    assert spec_5d[2:, [[[0, 1]]], [[0, 1]], :3].shape == torch.Size([3, 1, 1, 2, 3, 7])
    assert spec_5d[2:, [[[0, 1]]], 0, :3].shape == torch.Size([3, 1, 1, 2, 3, 7])
    assert spec_5d[2:, [[[0, 1]]], :3, 0].shape == torch.Size(
        [3, 1, 1, 2, 3, 7]
    )  # Matches tensordict & tensor's behavior. Numpy would return (1, 1, 2, 3, 3, 7).
    # TODO: Fix these tests.
    # assert spec_5d[2:, [[[0, 1]]], :3, [0]].shape == torch.Size([1, 1, 2, 3, 3, 7])
    # assert spec_5d[2:, [[[0, 1]]], :3, [[[0, 1]]]].shape == torch.Size([1, 1, 2, 3, 3, 7])

    # Specific tests when specs have non-indexable dimensions
    if spec_class in [
        BinaryDiscreteTensorSpec,
        OneHotDiscreteTensorSpec,
        MultiDiscreteTensorSpec,
        MultiOneHotDiscreteTensorSpec,
    ]:
        # Ellipsis
        assert spec_0d[None].shape == torch.Size([1, 0])
        assert spec_0d[...].shape == torch.Size([0])
        assert spec_2d[..., :2].shape == torch.Size([2, 3])
        assert spec_2d[..., :2, None, None].shape == torch.Size([2, 1, 1, 3])
        assert spec_4d[1, ..., 2].shape == torch.Size([3, 6])
        assert spec_2d[1, ..., None].shape == torch.Size([1, 3])
        assert spec_3d[..., [0, 1], [0]].shape == torch.Size([2, 4])
        assert spec_3d[None, 1, ..., None].shape == torch.Size([1, 3, 1, 4])
        assert spec_4d[:, None, ..., None, :].shape == torch.Size([5, 1, 3, 1, 4, 6])

    # BoundedTensorSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec, UnboundedDiscreteTensorSpec, CompositeSpec
    else:
        # Integers
        assert spec_2d[0, 1].shape == torch.Size([])

        # Ellipsis
        assert spec_0d[None].shape == torch.Size([1])
        assert spec_0d[...].shape == torch.Size([])
        assert spec_2d[..., :2].shape == torch.Size([5, 2])
        assert spec_2d[..., :2, None, None].shape == torch.Size([5, 2, 1, 1])
        assert spec_4d[1, ..., 2].shape == torch.Size([3, 4])
        assert spec_2d[1, ..., None].shape == torch.Size([3, 1])
        assert spec_3d[..., [0, 1], [0]].shape == torch.Size([5, 2])
        assert spec_3d[None, 1, ..., None].shape == torch.Size([1, 3, 4, 1])
        assert spec_4d[:, None, ..., None, :].shape == torch.Size([5, 1, 3, 4, 1, 6])

    # Additional tests for composite spec
    if spec_class == CompositeSpec:
        assert spec_2d[1]["k1"].shape == torch.Size([3, 4, 6, 7, 8])
        assert spec_3d[[1, 2]]["k1"].shape == torch.Size([2, 3, 4, 6, 7, 8])
        assert spec_2d[torch.randint(3, (3, 2))]["k1"].shape == torch.Size(
            [3, 2, 3, 4, 6, 7, 8]
        )
        assert spec_0d["k1"].shape == torch.Size([5, 3, 4, 6, 7, 8])
        assert spec_0d[None]["k1"].shape == torch.Size([1, 5, 3, 4, 6, 7, 8])

        assert spec_2d[..., 0]["k1"].shape == torch.Size([5, 4, 6, 7, 8])
        assert spec_4d[1, ..., 2]["k2"].shape == torch.Size([3, 4, 7])
        assert spec_2d[1, ..., None]["k2"].shape == torch.Size([3, 1, 4, 6, 7])


def test_composite_contains():
    spec = CompositeSpec(
        a=CompositeSpec(b=CompositeSpec(c=UnboundedContinuousTensorSpec()))
    )
    assert "a" in spec.keys()
    assert "a" in spec.keys(True)
    assert ("a",) in spec.keys()
    assert ("a",) in spec.keys(True)
    assert ("a", "b", "c") in spec.keys(True)
    assert ("a", "b", "c") in spec.keys(True, True)
    assert ("a", ("b", ("c",))) in spec.keys(True)
    assert ("a", ("b", ("c",))) in spec.keys(True, True)


def get_all_keys(spec: TensorSpec, include_exclusive: bool):
    """Given a TensorSpec, returns all exclusive and non-exclusive keys as a set of tuples.

    Args:
        spec (TensorSpec): the spec to get keys from.
        include_exclusive (bool: if True, include also exclusive keys in the result.

    """
    keys = set()
    if isinstance(spec, LazyStackedCompositeSpec) and include_exclusive:
        for t in spec._specs:
            keys = keys.union(get_all_keys(t, include_exclusive))
    if isinstance(spec, CompositeSpec):
        for key in spec.keys():
            keys.add((key,))
            inner_keys = get_all_keys(spec[key], include_exclusive)
            for inner_key in inner_keys:
                keys.add((key,) + _unravel_key_to_tuple(inner_key))

    return keys


@pytest.mark.parametrize("shape", ((), (1,), (2, 3), (2, 3, 4)))
@pytest.mark.parametrize(
    "spectype", ["one_hot", "categorical", "mult_one_hot", "mult_discrete"]
)
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("rand_shape", ((), (2,), (2, 3)))
class TestSpecMasking:
    def _make_mask(self, shape):
        torch.manual_seed(0)
        mask = torch.zeros(shape, dtype=torch.bool).bernoulli_()
        if len(shape) == 1:
            while not mask.any() or mask.all():
                mask = torch.zeros(shape, dtype=torch.bool).bernoulli_()
            return mask
        mask_view = mask.view(-1, shape[-1])
        for i in range(mask_view.shape[0]):
            t = mask_view[i]
            while not t.any() or t.all():
                t.copy_(torch.zeros_like(t).bernoulli_())
        return mask

    def _one_hot_spec(self, shape, device, n):
        shape = torch.Size([*shape, n])
        mask = self._make_mask(shape).to(device)
        return OneHotDiscreteTensorSpec(n, shape, device, mask=mask)

    def _mult_one_hot_spec(self, shape, device, n):
        shape = torch.Size([*shape, n + n + 2])
        mask = torch.cat(
            [
                self._make_mask(shape[:-1] + (n,)).to(device),
                self._make_mask(shape[:-1] + (n + 2,)).to(device),
            ],
            -1,
        )
        return MultiOneHotDiscreteTensorSpec([n, n + 2], shape, device, mask=mask)

    def _discrete_spec(self, shape, device, n):
        mask = self._make_mask(torch.Size([*shape, n])).to(device)
        return DiscreteTensorSpec(n, shape, device, mask=mask)

    def _mult_discrete_spec(self, shape, device, n):
        shape = torch.Size([*shape, 2])
        mask = torch.cat(
            [
                self._make_mask(shape[:-1] + (n,)).to(device),
                self._make_mask(shape[:-1] + (n + 2,)).to(device),
            ],
            -1,
        )
        return MultiDiscreteTensorSpec([n, n + 2], shape, device, mask=mask)

    def test_equal(self, shape, device, spectype, rand_shape, n=5):
        shape = torch.Size(shape)
        spec = (
            self._one_hot_spec(shape, device, n=n)
            if spectype == "one_hot"
            else self._discrete_spec(shape, device, n=n)
            if spectype == "categorical"
            else self._mult_one_hot_spec(shape, device, n=n)
            if spectype == "mult_one_hot"
            else self._mult_discrete_spec(shape, device, n=n)
            if spectype == "mult_discrete"
            else None
        )
        spec_clone = spec.clone()
        assert spec == spec_clone
        assert spec.unsqueeze(0).squeeze(0) == spec
        spec.update_mask(~spec.mask)
        assert (spec.mask != spec_clone.mask).any()
        assert spec != spec_clone

    def test_is_in(self, shape, device, spectype, rand_shape, n=5):
        shape = torch.Size(shape)
        rand_shape = torch.Size(rand_shape)
        spec = (
            self._one_hot_spec(shape, device, n=n)
            if spectype == "one_hot"
            else self._discrete_spec(shape, device, n=n)
            if spectype == "categorical"
            else self._mult_one_hot_spec(shape, device, n=n)
            if spectype == "mult_one_hot"
            else self._mult_discrete_spec(shape, device, n=n)
            if spectype == "mult_discrete"
            else None
        )
        s = spec.rand(rand_shape)
        assert spec.is_in(s)
        spec.update_mask(~spec.mask)
        assert not spec.is_in(s)

    def test_project(self, shape, device, spectype, rand_shape, n=5):
        shape = torch.Size(shape)
        rand_shape = torch.Size(rand_shape)
        spec = (
            self._one_hot_spec(shape, device, n=n)
            if spectype == "one_hot"
            else self._discrete_spec(shape, device, n=n)
            if spectype == "categorical"
            else self._mult_one_hot_spec(shape, device, n=n)
            if spectype == "mult_one_hot"
            else self._mult_discrete_spec(shape, device, n=n)
            if spectype == "mult_discrete"
            else None
        )
        s = spec.rand(rand_shape)
        assert (spec.project(s) == s).all()
        spec.update_mask(~spec.mask)
        sp = spec.project(s)
        assert sp.shape == s.shape
        if spectype == "one_hot":
            assert (sp != s).any(-1).all()
            assert (sp.any(-1)).all()
        elif spectype == "mult_one_hot":
            assert (sp != s).any(-1).all()
            assert (sp.sum(-1) == 2).all()
        else:
            assert (sp != s).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
