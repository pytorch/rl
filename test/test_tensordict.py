# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path
import re

import numpy as np
import pytest
import torch
from _utils_internal import get_available_devices
from torch import multiprocessing as mp
from torchrl.data import SavedTensorDict, TensorDict
from torchrl.data.tensordict.tensordict import assert_allclose_td, LazyStackedTensorDict
from torchrl.data.tensordict.utils import _getitem_batch_size


@pytest.mark.parametrize("device", get_available_devices())
def test_tensordict_set(device):
    torch.manual_seed(1)
    td = TensorDict({}, batch_size=(4, 5))
    td.set("key1", torch.randn(4, 5, device=device))
    assert td.device == torch.device(device)
    # by default inplace:
    with pytest.raises(RuntimeError):
        td.set("key1", torch.randn(5, 5, device=device))

    # robust to dtype casting
    td.set_("key1", torch.ones(4, 5, device=device, dtype=torch.double))
    assert (td.get("key1") == 1).all()

    # robust to device casting
    td.set("key_device", torch.ones(4, 5, device="cpu", dtype=torch.double))
    assert td.get("key_device").device == torch.device(device)

    with pytest.raises(
        AttributeError, match="for populating tensordict with new key-value pair"
    ):
        td.set_("smartypants", torch.ones(4, 5, device="cpu", dtype=torch.double))
    # test set_at_
    td.set("key2", torch.randn(4, 5, 6, device=device))
    x = torch.randn(6, device=device)
    td.set_at_("key2", x, (2, 2))
    assert (td.get("key2")[2, 2] == x).all()

    # test set_at_ with dtype casting
    x = torch.randn(6, dtype=torch.double, device=device)
    td.set_at_("key2", x, (2, 2))  # robust to dtype casting
    torch.testing.assert_allclose(td.get("key2")[2, 2], x.to(torch.float))

    td.set("key1", torch.zeros(4, 5, dtype=torch.double, device=device), inplace=True)
    assert (td.get("key1") == 0).all()
    td.set(
        "key1",
        torch.randn(4, 5, 1, 2, dtype=torch.double, device=device),
        inplace=False,
    )
    assert td._tensordict_meta["key1"].shape == td._tensordict["key1"].shape


@pytest.mark.parametrize("device", get_available_devices())
def test_stack(device):
    torch.manual_seed(1)
    tds_list = [TensorDict(source={}, batch_size=(4, 5)) for _ in range(3)]
    tds = torch.stack(tds_list, 0)
    assert tds[0] is tds_list[0]

    td = TensorDict(
        source={"a": torch.randn(4, 5, 3, device=device)}, batch_size=(4, 5)
    )
    td_list = list(td)
    td_reconstruct = torch.stack(td_list, 0)
    assert td_reconstruct.batch_size == td.batch_size
    assert (td_reconstruct == td).all()


@pytest.mark.parametrize("device", get_available_devices())
def test_tensordict_indexing(device):
    torch.manual_seed(1)
    td = TensorDict({}, batch_size=(4, 5))
    td.set("key1", torch.randn(4, 5, 1, device=device))
    td.set("key2", torch.randn(4, 5, 6, device=device, dtype=torch.double))

    td_select = td[2, 2]
    td_select._check_batch_size()

    td_select = td[2, :2]
    td_select._check_batch_size()

    td_select = td[None, :2]
    td_select._check_batch_size()

    td_reconstruct = torch.stack([_td for _td in td], 0)
    assert (
        td_reconstruct == td
    ).all(), f"td and td_reconstruct differ, got {td} and {td_reconstruct}"

    superlist = [torch.stack([__td for __td in _td], 0) for _td in td]
    td_reconstruct = torch.stack(superlist, 0)
    assert (
        td_reconstruct == td
    ).all(), f"td and td_reconstruct differ, got {td == td_reconstruct}"

    x = torch.randn(4, 5, device=device)
    td = TensorDict(
        source={"key1": torch.zeros(3, 4, 5, device=device)},
        batch_size=[3, 4],
    )
    td[0].set_("key1", x)
    torch.testing.assert_allclose(td.get("key1")[0], x)
    torch.testing.assert_allclose(td.get("key1")[0], td[0].get("key1"))

    y = torch.randn(3, 5, device=device)
    td[:, 0].set_("key1", y)
    torch.testing.assert_allclose(td.get("key1")[:, 0], y)
    torch.testing.assert_allclose(td.get("key1")[:, 0], td[:, 0].get("key1"))


@pytest.mark.parametrize("device", get_available_devices())
def test_subtensordict_construction(device):
    torch.manual_seed(1)
    td = TensorDict({}, batch_size=(4, 5))
    td.set("key1", torch.randn(4, 5, 1, device=device))
    td.set("key2", torch.randn(4, 5, 6, dtype=torch.double, device=device))
    std1 = td.get_sub_tensordict(2)
    std2 = std1.get_sub_tensordict(2)
    std_control = td.get_sub_tensordict((2, 2))
    assert (std_control.get("key1") == std2.get("key1")).all()
    assert (std_control.get("key2") == std2.get("key2")).all()

    # write values
    std_control.set("key1", torch.randn(1, device=device))
    std_control.set("key2", torch.randn(6, device=device, dtype=torch.double))

    assert (std_control.get("key1") == std2.get("key1")).all()
    assert (std_control.get("key2") == std2.get("key2")).all()

    assert std_control.get_parent_tensordict() is td
    assert (
        std_control.get_parent_tensordict()
        is std2.get_parent_tensordict().get_parent_tensordict()
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_mask_td(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    mask = torch.zeros(4, 5, dtype=torch.bool, device=device).bernoulli_()
    td = TensorDict(batch_size=(4, 5), source=d)
    td_masked = torch.masked_select(td, mask)
    assert len(td_masked.get("key1")) == td_masked.shape[0]


@pytest.mark.parametrize("device", get_available_devices())
def test_unbind_td(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    td = TensorDict(batch_size=(4, 5), source=d)
    td_unbind = torch.unbind(td, dim=1)
    assert (
        td_unbind[0].batch_size == td[:, 0].batch_size
    ), f"got {td_unbind[0].batch_size} and {td[:, 0].batch_size}"


@pytest.mark.parametrize("device", get_available_devices())
def test_cat_td(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5), source=d)
    d = {
        "key1": torch.randn(4, 10, 6, device=device),
        "key2": torch.randn(4, 10, 10, device=device),
    }
    td2 = TensorDict(batch_size=(4, 10), source=d)

    td_cat = torch.cat([td1, td2], 1)
    assert td_cat.batch_size == torch.Size([4, 15])
    d = {"key1": torch.randn(4, 15, 6), "key2": torch.randn(4, 15, 10)}
    td_out = TensorDict(batch_size=(4, 15), source=d)
    torch.cat([td1, td2], 1, out=td_out)
    assert td_out.batch_size == torch.Size([4, 15])


@pytest.mark.parametrize("device", get_available_devices())
def test_expand(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5), source=d)
    td2 = td1.expand(3, 7)
    assert td2.batch_size == torch.Size([3, 7, 4, 5])
    assert td2.get("key1").shape == torch.Size([3, 7, 4, 5, 6])
    assert td2.get("key2").shape == torch.Size([3, 7, 4, 5, 10])


@pytest.mark.parametrize("device", get_available_devices())
def test_squeeze(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5), source=d)
    td2 = torch.unsqueeze(td1, dim=1)
    assert td2.batch_size == torch.Size([4, 1, 5])

    td1b = torch.squeeze(td2, dim=1)
    assert td1b.batch_size == td1.batch_size


@pytest.mark.parametrize("device", get_available_devices())
def test_permute(device):
    torch.manual_seed(1)
    d = {
        "a": torch.randn(4, 5, 6, 9, device=device),
        "b": torch.randn(4, 5, 6, 7, device=device),
        "c": torch.randn(4, 5, 6, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5, 6), source=d)
    td2 = torch.permute(td1, dims=(2, 1, 0))
    assert td2.shape == torch.Size((6, 5, 4))
    assert td2["a"].shape == torch.Size((6, 5, 4, 9))

    td2 = torch.permute(td1, dims=(-1, -3, -2))
    assert td2.shape == torch.Size((6, 4, 5))
    assert td2["c"].shape == torch.Size((6, 4, 5, 1))

    td2 = torch.permute(td1, dims=(0, 1, 2))
    assert td2["a"].shape == torch.Size((4, 5, 6, 9))

    t = TensorDict({"a": torch.randn(3, 4, 1)}, [3, 4])
    torch.permute(t, dims=(1, 0)).set("b", torch.randn(4, 3))
    assert t["b"].shape == torch.Size((3, 4, 1))

    torch.permute(t, dims=(1, 0)).fill_("a", 0.0)
    assert torch.sum(t["a"]) == torch.Tensor([0])


@pytest.mark.parametrize("device", get_available_devices())
def test_permute_exceptions(device):
    torch.manual_seed(1)
    d = {
        "a": torch.randn(4, 5, 6, 7, device=device),
        "b": torch.randn(4, 5, 6, 8, 9, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5, 6), source=d)

    with pytest.raises(RuntimeError) as e_info:
        td2 = td1.permute(1, 1, 0)
        _ = td2.shape

    with pytest.raises(RuntimeError) as e_info:
        td2 = td1.permute(3, 2, 1, 0)
        _ = td2.shape

    with pytest.raises(RuntimeError) as e_info:
        td2 = td1.permute(2, -1, 0)
        _ = td2.shape

    with pytest.raises(IndexError) as e_info:
        td2 = td1.permute(2, 3, 0)
        _ = td2.shape

    with pytest.raises(IndexError) as e_info:
        td2 = td1.permute(2, -4, 0)
        _ = td2.shape

    with pytest.raises(RuntimeError) as e_info:
        td2 = td1.permute(2, 1)
        _ = td2.shape


@pytest.mark.parametrize("device", get_available_devices())
def test_permute_with_tensordict_operations(device):
    torch.manual_seed(1)
    d = {
        "a": torch.randn(20, 6, 9, device=device),
        "b": torch.randn(20, 6, 7, device=device),
        "c": torch.randn(20, 6, device=device),
    }
    td1 = TensorDict(batch_size=(20, 6), source=d).view(4, 5, 6).permute(2, 1, 0)
    assert td1.shape == torch.Size((6, 5, 4))

    d = {
        "a": torch.randn(4, 5, 6, 9, device=device),
        "b": torch.randn(4, 5, 6, 7, device=device),
        "c": torch.randn(4, 5, 6, device=device),
    }
    td1 = (
        TensorDict(batch_size=(4, 5, 6), source=d).to(SavedTensorDict).permute(2, 1, 0)
    )
    assert td1.shape == torch.Size((6, 5, 4))

    d = {
        "a": torch.randn(4, 5, 6, 7, 9, device=device),
        "b": torch.randn(4, 5, 6, 7, 7, device=device),
        "c": torch.randn(4, 5, 6, 7, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5, 6, 7), source=d)[
        :, :, :, torch.tensor([1, 2])
    ].permute(3, 2, 1, 0)
    assert td1.shape == torch.Size((2, 6, 5, 4))

    d = {
        "a": torch.randn(4, 5, 9, device=device),
        "b": torch.randn(4, 5, 7, device=device),
        "c": torch.randn(4, 5, device=device),
    }
    td1 = torch.stack(
        [TensorDict(batch_size=(4, 5), source=d).clone() for _ in range(6)], 2
    ).permute(2, 1, 0)
    assert td1.shape == torch.Size((6, 5, 4))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("stack_dim", [0, 1])
def test_stacked_td(stack_dim, device):
    tensordicts = [
        TensorDict(
            batch_size=[11, 12],
            source={
                "key1": torch.randn(11, 12, 5, device=device),
                "key2": torch.zeros(
                    11, 12, 50, device=device, dtype=torch.bool
                ).bernoulli_(),
            },
        )
        for _ in range(10)
    ]

    tensordicts0 = tensordicts[0]
    tensordicts1 = tensordicts[1]
    tensordicts2 = tensordicts[2]
    tensordicts3 = tensordicts[3]
    sub_td = LazyStackedTensorDict(*tensordicts, stack_dim=stack_dim)

    std_bis = torch.stack(tensordicts, dim=stack_dim)
    assert (sub_td == std_bis).all()

    item = tuple([*[slice(None) for _ in range(stack_dim)], 0])
    tensordicts0.zero_()
    assert (sub_td[item].get("key1") == sub_td.get("key1")[item]).all()
    assert (
        sub_td.contiguous()[item].get("key1") == sub_td.contiguous().get("key1")[item]
    ).all()
    assert (sub_td.contiguous().get("key1")[item] == 0).all()

    item = tuple([*[slice(None) for _ in range(stack_dim)], 1])
    std2 = sub_td[:5]
    tensordicts1.zero_()
    assert (std2[item].get("key1") == std2.get("key1")[item]).all()
    assert (
        std2.contiguous()[item].get("key1") == std2.contiguous().get("key1")[item]
    ).all()
    assert (std2.contiguous().get("key1")[item] == 0).all()

    std3 = sub_td[:5, :, :5]
    tensordicts2.zero_()
    item = tuple([*[slice(None) for _ in range(stack_dim)], 2])
    assert (std3[item].get("key1") == std3.get("key1")[item]).all()
    assert (
        std3.contiguous()[item].get("key1") == std3.contiguous().get("key1")[item]
    ).all()
    assert (std3.contiguous().get("key1")[item] == 0).all()

    std4 = sub_td.select("key1")
    tensordicts3.zero_()
    item = tuple([*[slice(None) for _ in range(stack_dim)], 3])
    assert (std4[item].get("key1") == std4.get("key1")[item]).all()
    assert (
        std4.contiguous()[item].get("key1") == std4.contiguous().get("key1")[item]
    ).all()
    assert (std4.contiguous().get("key1")[item] == 0).all()

    std5 = sub_td.unbind(1)[0]
    assert (std5.contiguous() == sub_td.contiguous().unbind(1)[0]).all()


@pytest.mark.parametrize("device", get_available_devices())
def test_savedtensordict(device):
    vals = [torch.randn(3, 1, device=device) for _ in range(4)]
    ss_list = [
        SavedTensorDict(
            source=TensorDict(
                source={"a": vals[i]},
                batch_size=[
                    3,
                ],
            )
        )
        for i in range(4)
    ]
    ss = torch.stack(ss_list, 0)
    assert ss_list[1] is ss[1]
    torch.testing.assert_allclose(ss_list[1].get("a"), vals[1])
    torch.testing.assert_allclose(ss_list[1].get("a"), ss[1].get("a"))
    torch.testing.assert_allclose(ss[1].get("a"), ss.get("a")[1])
    assert ss.get("a").device == device


class TestTensorDicts:
    @property
    def td(self):
        return TensorDict(
            source={
                "a": torch.randn(3, 1, 5),
                "b": torch.randn(3, 1, 10),
                "c": torch.randint(10, (3, 1, 3)),
            },
            batch_size=[3, 1],
        )

    @property
    def stacked_td(self):
        return torch.stack([self.td for _ in range(2)], 0)

    @property
    def idx_td(self):
        return self.td[0]

    @property
    def sub_td(self):
        return self.td.get_sub_tensordict(0)

    @property
    def saved_td(self):
        return SavedTensorDict(source=self.td)

    @property
    def unsqueezed_td(self):
        return self.td.unsqueeze(0)

    @property
    def td_reset_bs(self):
        td = self.td
        td = td.unsqueeze(-1).to_tensordict()
        td.batch_size = torch.Size([3, 1])
        return td

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_select(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td2 = td.select("a")
        assert td2 is not td
        assert len(list(td2.keys())) == 1 and "a" in td2.keys()
        assert len(list(td2.clone().keys())) == 1 and "a" in td2.clone().keys()

        td2 = td.select("a", inplace=True)
        assert td2 is td

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_exclude(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td2 = td.exclude("a")
        assert td2 is not td
        assert len(list(td2.keys())) == 2 and "a" not in td2.keys()
        assert len(list(td2.clone().keys())) == 2 and "a" not in td2.clone().keys()

        td2 = td.exclude("a", inplace=True)
        assert td2 is td

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_assert(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        with pytest.raises(
            ValueError,
            match="Converting a tensordict to boolean value is not permitted",
        ):
            assert td

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_expand(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        batch_size = td.batch_size
        new_td = td.expand(3)
        assert new_td.batch_size == torch.Size([3, *batch_size])
        assert all((_new_td == td).all() for _new_td in new_td)

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_cast(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td_td = td.to(TensorDict)
        assert (td == td_td).all()

        td = getattr(self, td_name)
        td_saved = td.to(SavedTensorDict)
        assert (td == td_saved).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_remove(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td = td.del_("a")
        assert td is not None
        assert "a" not in td.keys()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_set_unexisting(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td.set("z", torch.ones_like(td.get("a")))
        assert (td.get("z") == 1).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_fill_(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        new_td = td.fill_("a", 0.1)
        assert (td.get("a") == 0.1).all()
        assert new_td is td

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_masked_fill_(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        mask = torch.zeros(td.shape, dtype=torch.bool).bernoulli_()
        new_td = td.masked_fill_(mask, -10.0)
        assert new_td is td
        for k, item in td.items():
            assert (item[mask] == -10).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_masked_fill(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        mask = torch.zeros(td.shape, dtype=torch.bool).bernoulli_()
        new_td = td.masked_fill(mask, -10.0)
        assert new_td is not td
        for k, item in new_td.items():
            assert (item[mask] == -10).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_zero_(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        new_td = td.zero_()
        assert new_td is td
        for k in td.keys():
            assert (td.get(k) == 0).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_from_empty(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        new_td = TensorDict({}, batch_size=td.batch_size)
        for key, item in td.items():
            new_td.set(key, item)
        assert_allclose_td(td, new_td)
        assert td.device == new_td.device
        assert td.shape == new_td.shape

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_masking(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        mask = torch.zeros(td.batch_size, dtype=torch.bool).bernoulli_(0.8)
        td_masked = td[mask]
        td_masked2 = torch.masked_select(td, mask)
        assert_allclose_td(td_masked, td_masked2)
        assert td_masked.batch_size[0] == mask.sum()
        assert td_masked.batch_dims == 1

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0, reason="No cuda device detected"
    )
    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    @pytest.mark.parametrize("device", [0, "cuda:0", "cuda", torch.device("cuda:0")])
    def test_pin_memory(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        if td_name != "saved_td":
            td.pin_memory()
            td_device = td.to(device)
            _device = torch.device("cuda:0")
            assert td_device.device == _device
            assert td_device.clone().device == _device
            assert td_device is not td
            for k, item in td_device.items():
                assert item.device == _device
            for k, item in td_device.clone().items():
                assert item.device == _device
            # assert type(td_device) is type(td)
            assert_allclose_td(td, td_device.to("cpu"))
        else:
            with pytest.raises(
                RuntimeError,
                match="pin_memory requires tensordicts that live in memory",
            ):
                td.pin_memory()

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0, reason="No cuda device detected"
    )
    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_cast_device(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td_device = td.to(device)

        for k, item in td_device.items_meta():
            assert item.device == device
        for k, item in td_device.items():
            assert item.device == device
        for k, item in td_device.clone().items():
            assert item.device == device

        assert td_device.device == device, (
            f"td_device first tensor device is " f"{next(td_device.items())[1].device}"
        )
        assert td_device.clone().device == device
        if device != td.device:
            assert td_device is not td
        assert td_device.to(device) is td_device
        assert td.to("cpu") is td
        # assert type(td_device) is type(td)
        assert_allclose_td(td, td_device.to("cpu"))

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0, reason="No cuda device detected"
    )
    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_cpu_cuda(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td_device = td.cuda()
        td_back = td_device.cpu()
        assert td_device.device == torch.device("cuda:0")
        assert td_back.device == torch.device("cpu")

    @pytest.mark.parametrize(
        "td_name", ["td", "stacked_td", "saved_td", "unsqueezed_td"]
    )
    def test_unbind(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td_unbind = torch.unbind(td, dim=0)
        assert (td == torch.stack(td_unbind, 0)).all()
        assert (td[0] == td_unbind[0]).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    @pytest.mark.parametrize("squeeze_dim", [0, 1])
    def test_unsqueeze(self, td_name, squeeze_dim):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td_unsqueeze = torch.unsqueeze(td, dim=squeeze_dim)
        tensor = torch.ones_like(td.get("a").unsqueeze(squeeze_dim))
        td_unsqueeze.set("a", tensor)
        assert (td_unsqueeze.get("a") == tensor).all()
        assert (td.get("a") == tensor.squeeze(squeeze_dim)).all()
        assert td_unsqueeze.squeeze(squeeze_dim) is td
        assert (td_unsqueeze.get("a") == 1).all()
        assert (td.get("a") == 1).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_squeeze(self, td_name, squeeze_dim=-1):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td_squeeze = torch.squeeze(td, dim=-1)
        tensor_squeeze_dim = td.batch_dims + squeeze_dim
        tensor = torch.ones_like(td.get("a").squeeze(tensor_squeeze_dim))
        td_squeeze.set("a", tensor)
        assert (td_squeeze.get("a") == tensor).all()
        assert (td.get("a") == tensor.unsqueeze(tensor_squeeze_dim)).all()
        assert td_squeeze.unsqueeze(squeeze_dim) is td
        assert (td_squeeze.get("a") == 1).all()
        assert (td.get("a") == 1).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_view(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td_view = td.view(-1)
        tensor = td.get("a")
        tensor = tensor.view(-1, tensor.numel() // np.prod(td.batch_size))
        tensor = torch.ones_like(tensor)
        td_view.set("a", tensor)
        assert (td_view.get("a") == tensor).all()
        assert (td.get("a") == tensor.view(td.get("a").shape)).all()
        assert td_view.view(td.shape) is td
        assert td_view.view(*td.shape) is td
        assert (td_view.get("a") == 1).all()
        assert (td.get("a") == 1).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_clone_td(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        assert (torch.clone(td) == td).all()
        assert td.batch_size == torch.clone(td).batch_size
        if td_name in ("stacked_td", "saved_td", "unsqueezed_td", "sub_td"):
            with pytest.raises(AssertionError):
                assert td.clone(recursive=False).get("a") is td.get("a")
        else:
            assert td.clone(recursive=False).get("a") is td.get("a")

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_rename_key(self, td_name) -> None:
        torch.manual_seed(1)
        td = getattr(self, td_name)
        with pytest.raises(KeyError, match="already present in TensorDict"):
            td.rename_key("a", "b", safe=True)
        a = td.get("a")
        td.rename_key("a", "z")
        with pytest.raises(KeyError):
            td.get("a")
        assert "a" not in td.keys()

        z = td.get("z")
        torch.testing.assert_allclose(a, z)

        new_z = torch.randn_like(z)
        td.set("z", new_z)
        torch.testing.assert_allclose(new_z, td.get("z"))

        new_z = torch.randn_like(z)
        td.set_("z", new_z)
        torch.testing.assert_allclose(new_z, td.get("z"))

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_set_nontensor(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        r = torch.randn_like(td.get("a"))
        td.set("numpy", r.numpy())
        torch.testing.assert_allclose(td.get("numpy"), r)

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    @pytest.mark.parametrize("idx", [slice(1), torch.tensor([0]), torch.tensor([0, 1])])
    def test_setitem(self, td_name, idx):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        if isinstance(idx, torch.Tensor) and idx.numel() > 1 and td.shape[0] == 1:
            pytest.mark.skip("cannot index tensor with desired index")
            return

        td_clone = td[idx].clone().zero_()
        td[idx] = td_clone
        assert (td[idx].get("a") == 0).all()

        td_clone = torch.cat([td_clone, td_clone], 0)
        with pytest.raises(RuntimeError, match="differs from the source batch size"):
            td[idx] = td_clone

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("chunks", [1, 2])
    def test_chunk(self, td_name, dim, chunks):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        if len(td.shape) - 1 < dim:
            pytest.mark.skip(f"no dim {dim} in td")
            return

        chunks = min(td.shape[dim], chunks)
        td_chunks = td.chunk(chunks, dim)
        assert len(td_chunks) == chunks
        assert sum([_td.shape[dim] for _td in td_chunks]) == td.shape[dim]
        assert (torch.cat(td_chunks, dim) == td).all()

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_items_values_keys(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        keys = list(td.keys())
        values = list(td.values())
        items = list(td.items())

        # Test td.items()
        constructed_td1 = TensorDict({}, batch_size=td.shape)
        for key, value in items:
            constructed_td1.set(key, value)

        assert (td == constructed_td1).all()

        # Test td.keys() and td.values()
        # items = [key, value] should be verified
        assert len(values) == len(items)
        assert len(keys) == len(items)
        constructed_td2 = TensorDict({}, batch_size=td.shape)
        for key, value in list(zip(td.keys(), td.values())):
            constructed_td2.set(key, value)

        assert (td == constructed_td2).all()

        # Test that keys is sorted
        assert all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))

        # Add new element to tensor
        a = td.get("a")
        td.set("x", torch.randn_like(a))
        keys = list(td.keys())
        values = list(td.values())
        items = list(td.items())

        # Test that keys is still sorted after adding the element
        assert all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))

        # Test td.items()
        # after adding the new element
        constructed_td1 = TensorDict({}, batch_size=td.shape)
        for key, value in items:
            constructed_td1.set(key, value)

        assert (td == constructed_td1).all()

        # Test td.keys() and td.values()
        # items = [key, value] should be verified
        # even after adding the new element
        assert len(values) == len(items)
        assert len(keys) == len(items)

        constructed_td2 = TensorDict({}, batch_size=td.shape)
        for key, value in list(zip(td.keys(), td.values())):
            constructed_td2.set(key, value)

        assert (td == constructed_td2).all()

        # Test the methods values_meta() and items_meta()
        values_meta = list(td.values_meta())
        items_meta = list(td.items_meta())
        assert len(values_meta) == len(items_meta)

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "saved_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_set_requires_grad(self, td_name):
        td = getattr(self, td_name)
        assert not td._get_meta("a").requires_grad
        td.set("a", torch.randn_like(td.get("a")).requires_grad_())
        assert td._get_meta("a").requires_grad


class TestTensorDictsRequiresGrad:
    @property
    def td(self):
        return TensorDict(
            source={
                "a": torch.randn(3, 1, 5),
                "b": torch.randn(3, 1, 10, requires_grad=True),
                "c": torch.randint(10, (3, 1, 3)),
            },
            batch_size=[3, 1],
        )

    @property
    def stacked_td(self):
        return torch.stack([self.td for _ in range(2)], 0)

    @property
    def idx_td(self):
        return self.td[0]

    @property
    def sub_td(self):
        return self.td.get_sub_tensordict(0)

    @property
    def unsqueezed_td(self):
        return self.td.unsqueeze(0)

    @property
    def td_reset_bs(self):
        td = self.td
        td = td.unsqueeze(-1).to_tensordict()
        td.batch_size = torch.Size([3, 1])
        return td

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_init_requires_grad(self, td_name):
        td = getattr(self, td_name)
        assert td._get_meta("b").requires_grad

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_view(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td_view = td.view(-1)
        assert td_view._get_meta("b").requires_grad

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_expand(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        batch_size = td.batch_size
        new_td = td.expand(3)
        assert new_td._get_meta("b").requires_grad

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_cast(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        td_td = td.to(TensorDict)
        assert td_td._get_meta("b").requires_grad

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_clone_td(self, td_name):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        assert torch.clone(td)._get_meta("b").requires_grad

    @pytest.mark.parametrize(
        "td_name",
        [
            "td",
            "stacked_td",
            "sub_td",
            "idx_td",
            "unsqueezed_td",
            "td_reset_bs",
        ],
    )
    def test_squeeze(self, td_name, squeeze_dim=-1):
        torch.manual_seed(1)
        td = getattr(self, td_name)
        assert torch.squeeze(td, dim=-1)._get_meta("b").requires_grad


def test_batchsize_reset():
    td = TensorDict(
        {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)}, batch_size=[3, 4]
    )
    # smoke-test
    td.batch_size = torch.Size([3])

    # test with list
    td.batch_size = [3]

    # test with tuple
    td.batch_size = (3,)

    # incompatible size
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "the tensor a has shape torch.Size([3, 4, 5, "
            "6]) which is incompatible with the new shape torch.Size([3, 5])"
        ),
    ):
        td.batch_size = [3, 5]

    # test set
    td.set("c", torch.randn(3))

    # test index
    subtd = td[torch.tensor([1, 2])]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The shape torch.Size([3]) is incompatible with the index (slice(None, None, None), 0)."
        ),
    ):
        td[:, 0]

    # test a greater batch_size
    td = TensorDict(
        {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)}, batch_size=[3, 4]
    )
    td.batch_size = torch.Size([3, 4, 5])
    td.set("c", torch.randn(3, 4, 5, 6))
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "batch dimension mismatch, "
            "got self.batch_size=torch.Size([3, 4, 5]) and tensor.shape[:self.batch_dims]=torch.Size([3, 4, 2])"
        ),
    ):
        td.set("d", torch.randn(3, 4, 2))

    # test with saved tensordict
    td = SavedTensorDict(TensorDict({"a": torch.randn(3, 4)}, [3, 4]))
    td.batch_size = [3]
    assert td.to_tensordict().batch_size == torch.Size([3])

    # test that lazy tds return an exception
    td_stack = torch.stack([TensorDict({"a": torch.randn(3)}, [3]) for _ in range(2)])
    td_stack.to_tensordict().batch_size = [2]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "modifying the batch size of a lazy repesentation "
            "of a tensordict is not permitted. Consider instantiating the tensordict fist by calling `td = td.to_tensordict()` before resetting the batch size."
        ),
    ):
        td_stack.batch_size = [2]

    td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
    subtd = td[:, torch.tensor([1, 2])]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "modifying the batch size of a lazy repesentation of a tensordict is not permitted. Consider instantiating the tensordict fist by calling `td = td.to_tensordict()` before resetting the batch size."
        ),
    ):
        subtd.batch_size = [3, 2]
    subtd.to_tensordict().batch_size = [3, 2]

    td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
    td_u = td.unsqueeze(0)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "modifying the batch size of a lazy repesentation of a tensordict is not permitted. Consider instantiating the tensordict fist by calling `td = td.to_tensordict()` before resetting the batch size."
        ),
    ):
        td_u.batch_size = [1]
    td_u.to_tensordict().batch_size = [1]


@pytest.mark.parametrize("index0", [None, slice(None)])
def test_set_sub_key(index0):
    # tests that parent tensordict is affected when subtensordict is set with a new key
    batch_size = [10, 10]
    source = {"a": torch.randn(10, 10, 10), "b": torch.ones(10, 10, 2)}
    td = TensorDict(source, batch_size=batch_size)
    idx0 = (index0, 0) if index0 is not None else 0
    td0 = td.get_sub_tensordict(idx0)
    idx = (index0, slice(2, 4)) if index0 is not None else slice(2, 4)
    sub_td = td.get_sub_tensordict(idx)
    if index0 is None:
        c = torch.randn(2, 10, 10)
    else:
        c = torch.randn(10, 2, 10)
    sub_td.set("c", c)
    assert (td.get("c")[idx] == sub_td.get("c")).all()
    assert (sub_td.get("c") == c).all()
    assert (td.get("c")[idx0] == 0).all()
    assert (td.get_sub_tensordict(idx0).get("c") == 0).all()
    assert (td0.get("c") == 0).all()


def _remote_process(worker_id, command_pipe_child, command_pipe_parent, tensordict):
    command_pipe_parent.close()
    while True:
        cmd, val = command_pipe_child.recv()
        if cmd == "recv":
            b = tensordict.get("b")
            assert (b == val).all()
            command_pipe_child.send("done")
        elif cmd == "send":
            a = torch.ones(2) * val
            tensordict.set_("a", a)
            assert (
                tensordict.get("a") == a
            ).all(), f'found {a} and {tensordict.get("a")}'
            command_pipe_child.send("done")
        elif cmd == "set_done":
            tensordict.set_("done", torch.ones(1, dtype=torch.bool))
            command_pipe_child.send("done")
        elif cmd == "set_undone_":
            tensordict.set_("done", torch.zeros(1, dtype=torch.bool))
            command_pipe_child.send("done")
        elif cmd == "update":
            tensordict.update_(
                TensorDict(
                    source={"a": tensordict.get("a").clone() + 1},
                    batch_size=tensordict.batch_size,
                )
            )
            command_pipe_child.send("done")
        elif cmd == "update_":
            tensordict.update_(
                TensorDict(
                    source={"a": tensordict.get("a").clone() - 1},
                    batch_size=tensordict.batch_size,
                )
            )
            command_pipe_child.send("done")

        elif cmd == "close":
            command_pipe_child.close()
            break


def _driver_func(tensordict, tensordict_unbind):
    procs = []
    children = []
    parents = []

    for i in range(2):
        command_pipe_parent, command_pipe_child = mp.Pipe()
        proc = mp.Process(
            target=_remote_process,
            args=(i, command_pipe_child, command_pipe_parent, tensordict_unbind[i]),
        )
        proc.start()
        command_pipe_child.close()
        parents.append(command_pipe_parent)
        children.append(command_pipe_child)
        procs.append(proc)

    b = torch.ones(2, 1) * 10
    tensordict.set_("b", b)
    for i in range(2):
        parents[i].send(("recv", 10))
        is_done = parents[i].recv()
        assert is_done == "done"

    for i in range(2):
        parents[i].send(("send", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    a = tensordict.get("a").clone()
    assert (a[0] == 0).all()
    assert (a[1] == 1).all()

    assert not tensordict.get("done").any()
    for i in range(2):
        parents[i].send(("set_done", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    assert tensordict.get("done").all()

    for i in range(2):
        parents[i].send(("set_undone_", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    assert not tensordict.get("done").any()

    a_prev = tensordict.get("a").clone().contiguous()
    for i in range(2):
        parents[i].send(("update_", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    new_a = tensordict.get("a").clone().contiguous()
    torch.testing.assert_allclose(a_prev - 1, new_a)

    a_prev = tensordict.get("a").clone().contiguous()
    for i in range(2):
        parents[i].send(("update", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    new_a = tensordict.get("a").clone().contiguous()
    torch.testing.assert_allclose(a_prev + 1, new_a)

    for i in range(2):
        parents[i].send(("close", None))
        procs[i].join()


@pytest.mark.parametrize(
    "td_type", ["contiguous", "stack", "saved", "memmap", "memmap_stack"]
)
def test_mp(td_type):
    tensordict = TensorDict(
        source={
            "a": torch.randn(2, 2),
            "b": torch.randn(2, 1),
            "done": torch.zeros(2, 1, dtype=torch.bool),
        },
        batch_size=[2],
    )
    if td_type == "contiguous":
        tensordict = tensordict.share_memory_()
    elif td_type == "stack":
        tensordict = torch.stack(
            [
                tensordict[0].clone().share_memory_(),
                tensordict[1].clone().share_memory_(),
            ],
            0,
        )
    elif td_type == "saved":
        tensordict = tensordict.clone().to(SavedTensorDict)
    elif td_type == "memmap":
        tensordict = tensordict.memmap_()
    elif td_type == "memmap_stack":
        tensordict = torch.stack(
            [tensordict[0].clone().memmap_(), tensordict[1].clone().memmap_()], 0
        )
    else:
        raise NotImplementedError
    _driver_func(tensordict, tensordict.unbind(0))


def test_saved_delete():
    td = TensorDict(source={"a": torch.randn(3)}, batch_size=[])
    td = td.to(SavedTensorDict)
    file = td.file.name
    assert os.path.isfile(file)
    del td
    assert not os.path.isfile(file)


def test_stack_keys():
    td1 = TensorDict(source={"a": torch.randn(3)}, batch_size=[])
    td2 = TensorDict(
        source={
            "a": torch.randn(3),
            "b": torch.randn(3),
            "c": torch.randn(4),
            "d": torch.randn(5),
        },
        batch_size=[],
    )
    td = torch.stack([td1, td2], 0)
    assert "a" in td.keys()
    assert "b" not in td.keys()
    assert "b" in td[1].keys()
    td.set("b", torch.randn(2, 10), inplace=False)  # overwrites
    with pytest.raises(KeyError):
        td.set_("c", torch.randn(2, 10))  # overwrites
    td.set_("b", torch.randn(2, 10))  # b has been set before

    td1.set("c", torch.randn(4))
    assert "c" in td.keys()  # now all tds have the key c
    td.get("c")

    td1.set("d", torch.randn(6))
    with pytest.raises(RuntimeError):
        td.get("d")


def test_getitem_batch_size():
    shape = [
        10,
        7,
        11,
        5,
    ]
    mocking_tensor = torch.zeros(*shape)
    for idx in [
        (slice(None),),
        slice(None),
        (3, 4),
        (3, slice(None), slice(2, 2, 2)),
        (torch.tensor([1, 2, 3]),),
        ([1, 2, 3]),
        (
            torch.tensor([1, 2, 3]),
            torch.tensor([2, 3, 4]),
            torch.tensor([0, 10, 2]),
            torch.tensor([2, 4, 1]),
        ),
    ]:
        expected_shape = mocking_tensor[idx].shape
        resulting_shape = _getitem_batch_size(shape, idx)
        assert expected_shape == resulting_shape, idx


@pytest.mark.parametrize("device", get_available_devices())
def test_requires_grad(device):
    torch.manual_seed(1)
    # Just one of the tensors have requires_grad
    tensordicts = [
        TensorDict(
            batch_size=[11, 12],
            source={
                "key1": torch.randn(
                    11, 12, 5, device=device, requires_grad=True if i == 5 else False
                ),
                "key2": torch.zeros(
                    11, 12, 50, device=device, dtype=torch.bool
                ).bernoulli_(),
            },
        )
        for i in range(10)
    ]
    stacked_td = LazyStackedTensorDict(*tensordicts, stack_dim=0)
    # First stacked tensor has requires_grad == True
    assert list(stacked_td.values_meta())[0].requires_grad is True
    td0 = SavedTensorDict(tensordicts[0])
    with pytest.raises(
        Exception,
        match=re.escape(
            "SavedTensorDicts is not compatible with gradients, one of Tensors has requires_grad equals True"
        ),
    ):
        td5 = SavedTensorDict(tensordicts[5])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
