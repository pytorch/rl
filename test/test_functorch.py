import argparse

import pytest
import torch
from functorch import vmap
from torch import nn
from torchrl.data import TensorDict
from torchrl.modules import TensorDictModule, TensorDictSequential
from torchrl.modules.functional_modules import (
    FunctionalModule,
    FunctionalModuleWithBuffers,
)


@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_patch(moduletype, batch_params):
    if moduletype == "linear":
        module = nn.Linear(3, 4)
    elif moduletype == "bn1":
        module = nn.BatchNorm1d(3)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        fmodule, params = FunctionalModule._create_from(module)
        x = torch.randn(10, 1, 3)
        if batch_params:
            params = params.expand(10, *params.batch_size)
            y = vmap(fmodule, (0, 0))(params, x)
        else:
            y = vmap(fmodule, (None, 0))(params, x)
        assert y.shape == torch.Size([10, 1, 4])
    elif moduletype == "bn1":
        fmodule, params, buffers = FunctionalModuleWithBuffers._create_from(module)
        x = torch.randn(10, 2, 3)
        if batch_params:
            params = params.expand(10, *params.batch_size).contiguous()
            buffers = buffers.expand(10, *buffers.batch_size).contiguous()
            y = vmap(fmodule, (0, 0, 0))(params, buffers, x)
        else:
            raise NotImplementedError
        assert y.shape == torch.Size([10, 2, 3])


@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_tdmodule(moduletype, batch_params):
    if moduletype == "linear":
        module = nn.Linear(3, 4)
    elif moduletype == "bn1":
        module = nn.BatchNorm1d(3)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        fmodule, params = FunctionalModule._create_from(module)
        tdmodule = TensorDictModule(fmodule, in_keys=["x"], out_keys=["y"])
        x = torch.randn(10, 1, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size)
            tdmodule(td, params=params, vmap=(0, 0))
        else:
            tdmodule(td, params=params, vmap=(None, 0))
        y = td["y"]
        assert y.shape == torch.Size([10, 1, 4])
    elif moduletype == "bn1":
        fmodule, params, buffers = FunctionalModuleWithBuffers._create_from(module)
        tdmodule = TensorDictModule(fmodule, in_keys=["x"], out_keys=["y"])
        x = torch.randn(10, 2, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size).contiguous()
            buffers = buffers.expand(10, *buffers.batch_size).contiguous()
            tdmodule(td, params=params, buffers=buffers, vmap=(0, 0, 0))
        else:
            raise NotImplementedError
        y = td["y"]
        assert y.shape == torch.Size([10, 2, 3])


@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_tdmodule_nativebuilt(moduletype, batch_params):
    if moduletype == "linear":
        module = nn.Linear(3, 4)
    elif moduletype == "bn1":
        module = nn.BatchNorm1d(3)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        tdmodule = TensorDictModule(module, in_keys=["x"], out_keys=["y"])
        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers(native=True)
        x = torch.randn(10, 1, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size)
            buffers = buffers.expand(10, *buffers.batch_size)
            tdmodule(td, params=params, buffers=buffers, vmap=(0, 0, 0))
        else:
            tdmodule(td, params=params, buffers=buffers, vmap=(None, None, 0))
        y = td["y"]
        assert y.shape == torch.Size([10, 1, 4])
    elif moduletype == "bn1":
        tdmodule = TensorDictModule(module, in_keys=["x"], out_keys=["y"])
        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers(native=True)
        x = torch.randn(10, 2, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size).contiguous()
            buffers = buffers.expand(10, *buffers.batch_size).contiguous()
            tdmodule(td, params=params, buffers=buffers, vmap=(0, 0, 0))
        else:
            raise NotImplementedError
        y = td["y"]
        assert y.shape == torch.Size([10, 2, 3])


@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_tdsequence(moduletype, batch_params):
    if moduletype == "linear":
        module1 = nn.Linear(3, 4)
        fmodule1, params1 = FunctionalModule._create_from(module1)
        module2 = nn.Linear(4, 5)
        fmodule2, params2 = FunctionalModule._create_from(module2)
    elif moduletype == "bn1":
        module1 = nn.BatchNorm1d(3)
        fmodule1, params1, buffers1 = FunctionalModuleWithBuffers._create_from(module1)
        module2 = nn.BatchNorm1d(3)
        fmodule2, params2, buffers2 = FunctionalModuleWithBuffers._create_from(module2)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        tdmodule1 = TensorDictModule(fmodule1, in_keys=["x"], out_keys=["y"])
        tdmodule2 = TensorDictModule(fmodule2, in_keys=["y"], out_keys=["z"])
        params = TensorDict({"0": params1, "1": params2}, [])
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)
        assert {"0", "1"} == set(params.keys())
        x = torch.randn(10, 1, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size)
            tdmodule(td, params=params, vmap=(0, 0))
        else:
            tdmodule(td, params=params, vmap=(None, 0))
        z = td["z"]
        assert z.shape == torch.Size([10, 1, 5])
    elif moduletype == "bn1":
        tdmodule1 = TensorDictModule(fmodule1, in_keys=["x"], out_keys=["y"])
        tdmodule2 = TensorDictModule(fmodule2, in_keys=["y"], out_keys=["z"])
        params = TensorDict({"0": params1, "1": params2}, [])
        buffers = TensorDict({"0": buffers1, "1": buffers2}, [])
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)
        assert {"0", "1"} == set(params.keys())
        assert {"0", "1"} == set(buffers.keys())
        x = torch.randn(10, 2, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size).contiguous()
            buffers = buffers.expand(10, *buffers.batch_size).contiguous()
            tdmodule(td, params=params, buffers=buffers, vmap=(0, 0, 0))
        else:
            raise NotImplementedError
        z = td["z"]
        assert z.shape == torch.Size([10, 2, 3])


@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_tdsequence_nativebuilt(moduletype, batch_params):
    if moduletype == "linear":
        module1 = nn.Linear(3, 4)
        module2 = nn.Linear(4, 5)
    elif moduletype == "bn1":
        module1 = nn.BatchNorm1d(3)
        module2 = nn.BatchNorm1d(3)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        tdmodule1 = TensorDictModule(module1, in_keys=["x"], out_keys=["y"])
        tdmodule2 = TensorDictModule(module2, in_keys=["y"], out_keys=["z"])
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)
        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers(native=True)
        assert {"0", "1"} == set(params.keys())
        x = torch.randn(10, 1, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size)
            buffers = buffers.expand(10, *buffers.batch_size)
            tdmodule(td, params=params, buffers=buffers, vmap=(0, 0, 0))
        else:
            tdmodule(td, params=params, buffers=buffers, vmap=(None, None, 0))
        z = td["z"]
        assert z.shape == torch.Size([10, 1, 5])
    elif moduletype == "bn1":
        tdmodule1 = TensorDictModule(module1, in_keys=["x"], out_keys=["y"])
        tdmodule2 = TensorDictModule(module2, in_keys=["y"], out_keys=["z"])
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)
        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers(native=True)
        assert {"0", "1"} == set(params.keys())
        assert {"0", "1"} == set(buffers.keys())
        x = torch.randn(10, 2, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size).contiguous()
            buffers = buffers.expand(10, *buffers.batch_size).contiguous()
            tdmodule(td, params=params, buffers=buffers, vmap=(0, 0, 0))
        else:
            raise NotImplementedError
        z = td["z"]
        assert z.shape == torch.Size([10, 2, 3])


class TestNativeFunctorch:
    def test_vamp_basic(self):
        class MyModule(torch.nn.Module):
            def forward(self, tensordict):
                a = tensordict["a"]
                return TensorDict(
                    {"a": a}, tensordict.batch_size, device=tensordict.device
                )

        tensordict = TensorDict({"a": torch.randn(3)}, []).expand(4)
        out = vmap(MyModule(), (0,))(tensordict)
        assert out.shape == torch.Size([4])
        assert out["a"].shape == torch.Size([4, 3])

    def test_vamp_composed(self):
        class MyModule(torch.nn.Module):
            def forward(self, tensordict, tensor):
                a = tensordict["a"]
                return (
                    TensorDict(
                        {"a": a}, tensordict.batch_size, device=tensordict.device
                    ),
                    tensor,
                )

        tensor = torch.randn(3)
        tensordict = TensorDict({"a": torch.randn(3, 1)}, [3]).expand(4, 3)
        out = vmap(MyModule(), (0, None))(tensordict, tensor)

        assert out[0].shape == torch.Size([4, 3])
        assert out[1].shape == torch.Size([4, 3])
        assert out[0]["a"].shape == torch.Size([4, 3, 1])

    def test_vamp_composed_flipped(self):
        class MyModule(torch.nn.Module):
            def forward(self, tensordict, tensor):
                a = tensordict["a"]
                return (
                    TensorDict(
                        {"a": a}, tensordict.batch_size, device=tensordict.device
                    ),
                    tensor,
                )

        tensor = torch.randn(3).expand(4, 3)
        tensordict = TensorDict({"a": torch.randn(3, 1)}, [3])
        out = vmap(MyModule(), (None, 0))(tensordict, tensor)

        assert out[0].shape == torch.Size([4, 3])
        assert out[1].shape == torch.Size([4, 3])
        assert out[0]["a"].shape == torch.Size([4, 3, 1])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
