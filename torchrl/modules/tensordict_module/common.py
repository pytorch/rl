# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import inspect
import re
import warnings
from typing import Iterable, List, Optional, Type, Union

import torch

from tensordict import TensorDictBase, unravel_key_list

from tensordict.nn import dispatch, TensorDictModule, TensorDictModuleBase
from tensordict.utils import NestedKey

from torch import nn
from torch.nn import functional as F

from torchrl.data.tensor_specs import CompositeSpec, TensorSpec

from torchrl.data.utils import DEVICE_TYPING

_has_functorch = importlib.util.find_spec("functorch") is not None
if _has_functorch:
    from functorch import FunctionalModule, FunctionalModuleWithBuffers
else:
    warnings.warn(
        "failed to import functorch. TorchRL's features that do not require "
        "functional programming should work, but functionality and performance "
        "may be affected. Consider installing functorch and/or upgrating pytorch."
    )

    class FunctionalModule:  # noqa: D101
        pass

    class FunctionalModuleWithBuffers:  # noqa: D101
        pass


def _check_all_str(list_of_str, first_level=True):
    if isinstance(list_of_str, str) and first_level:
        raise RuntimeError(
            f"Expected a list of strings but got a string: {list_of_str}"
        )
    elif not isinstance(list_of_str, str):
        try:
            return [_check_all_str(item, False) for item in list_of_str]
        except Exception as err:
            raise TypeError(
                f"Expected a list of strings but got: {list_of_str}."
            ) from err


def _forward_hook_safe_action(module, tensordict_in, tensordict_out):
    try:
        spec = module.spec
        if len(module.out_keys) > 1 and not isinstance(spec, CompositeSpec):
            raise RuntimeError(
                "safe TensorDictModules with multiple out_keys require a CompositeSpec with matching keys. Got "
                f"keys {module.out_keys}."
            )
        elif not isinstance(spec, CompositeSpec):
            out_key = module.out_keys[0]
            keys = [out_key]
            values = [spec]
        else:
            keys = list(spec.keys(True, True))
            values = [spec[key] for key in keys]
        for _spec, _key in zip(values, keys):
            if _spec is None:
                continue
            item = tensordict_out.get(_key, None)
            if item is None:
                # this will happen when an exploration (e.g. OU) writes a key only
                # during exploration, but is missing otherwise.
                # it's fine since what we want here it to make sure that a key
                # is within bounds if it is present
                continue
            if not _spec.is_in(item):
                try:
                    tensordict_out.set_(
                        _key,
                        _spec.project(tensordict_out.get(_key)),
                    )
                except RuntimeError:
                    tensordict_out.set(
                        _key,
                        _spec.project(tensordict_out.get(_key)),
                    )
    except RuntimeError as err:
        if re.search(
            "attempting to use a Tensor in some data-dependent control flow", str(err)
        ):
            # "_is_stateless" in module.__dict__ and module._is_stateless:
            raise RuntimeError(
                "vmap cannot be used with safe=True, consider turning the safe mode off."
            ) from err
        else:
            raise err


class SafeModule(TensorDictModule):
    """:class:`tensordict.nn.TensorDictModule` subclass that accepts a :class:`~torchrl.data.TensorSpec` as argument to control the output domain.

    Args:
        module (nn.Module): a nn.Module used to map the input to the output
            parameter space. Can be a functional
            module (FunctionalModule or FunctionalModuleWithBuffers), in which
            case the :obj:`forward` method will expect
            the params (and possibly) buffers keyword arguments.
        in_keys (iterable of str): keys to be read from input tensordict and
            passed to the module. If it
            contains more than one element, the values will be passed in the
            order given by the in_keys iterable.
        out_keys (iterable of str): keys to be written to the input tensordict.
            The length of out_keys must match the
            number of tensors returned by the embedded module. Using "_" as a
            key avoid writing tensor to output.
        spec (TensorSpec, optional): specs of the output tensor. If the module
            outputs multiple output tensors,
            spec characterize the space of the first output tensor.
        safe (bool): if ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.

    Embedding a neural network in a TensorDictModule only requires to specify the input and output keys. The domain spec can
        be passed along if needed. TensorDictModule support functional and regular :obj:`nn.Module` objects. In the functional
        case, the 'params' (and 'buffers') keyword argument must be specified:

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import UnboundedContinuousTensorSpec
        >>> from torchrl.modules import TensorDictModule
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> spec = UnboundedContinuousTensorSpec(8)
        >>> module = torch.nn.GRUCell(4, 8)
        >>> td_fmodule = TensorDictModule(
        ...    module=module,
        ...    spec=spec,
        ...    in_keys=["input", "hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> params = TensorDict.from_module(td_fmodule)
        >>> with params.to_module(td_module):
        ...     td_functional = td_fmodule(td.clone())
        >>> print(td_functional)
        TensorDict(
            fields={
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    In the stateful case:
        >>> td_module = TensorDictModule(
        ...    module=torch.nn.GRUCell(4, 8),
        ...    spec=spec,
        ...    in_keys=["input", "hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> td_stateful = td_module(td.clone())
        >>> print(td_stateful)
        TensorDict(
            fields={
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    One can use a vmap operator to call the functional module. In this case the tensordict is expanded to match the
    batch size (i.e. the tensordict isn't modified in-place anymore):
        >>> # Model ensemble using vmap
        >>> from torch import vmap
        >>> params_repeat = params.expand(4, *params.shape)
        >>> td_vmap = vmap(td_fmodule, (None, 0))(td.clone(), params_repeat)
        >>> print(td_vmap)
        TensorDict(
            fields={
                hidden: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32),
                input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                output: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32)},
            batch_size=torch.Size([4, 3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        module: Union[
            FunctionalModule, FunctionalModuleWithBuffers, TensorDictModule, nn.Module
        ],
        in_keys: Iterable[str],
        out_keys: Iterable[str],
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
    ):
        super().__init__(module, in_keys, out_keys)
        self.register_spec(safe=safe, spec=spec)

    def register_spec(self, safe, spec):
        if spec is not None:
            spec = spec.clone()
        if spec is not None and not isinstance(spec, TensorSpec):
            raise TypeError("spec must be a TensorSpec subclass")
        elif spec is not None and not isinstance(spec, CompositeSpec):
            if len(self.out_keys) > 1:
                raise RuntimeError(
                    f"got more than one out_key for the TensorDictModule: {self.out_keys},\nbut only one spec. "
                    "Consider using a CompositeSpec object or no spec at all."
                )
            spec = CompositeSpec({self.out_keys[0]: spec})
        elif spec is not None and isinstance(spec, CompositeSpec):
            if "_" in spec.keys() and spec["_"] is not None:
                warnings.warn('got a spec with key "_": it will be ignored')
        elif spec is None:
            spec = CompositeSpec()

        # unravel_key_list(self.out_keys) can be removed once 473 is merged in tensordict
        spec_keys = set(unravel_key_list(list(spec.keys(True, True))))
        out_keys = set(unravel_key_list(self.out_keys))
        if spec_keys != out_keys:
            # then assume that all the non indicated specs are None
            for key in out_keys:
                if key not in spec_keys:
                    spec[key] = None
            spec_keys = set(unravel_key_list(list(spec.keys(True, True))))
        if spec_keys != out_keys:
            raise RuntimeError(
                f"spec keys and out_keys do not match, got: {spec_keys} and {out_keys} respectively"
            )

        self._spec = spec
        self.safe = safe
        if safe:
            if spec is None or (
                isinstance(spec, CompositeSpec)
                and all(_spec is None for _spec in spec.values())
            ):
                raise RuntimeError(
                    "`TensorDictModule(spec=None, safe=True)` is not a valid configuration as the tensor "
                    "specs are not specified"
                )
            self.register_forward_hook(_forward_hook_safe_action)

    @property
    def spec(self) -> CompositeSpec:
        return self._spec

    @spec.setter
    def spec(self, spec: CompositeSpec) -> None:
        if not isinstance(spec, CompositeSpec):
            raise RuntimeError(
                f"Trying to set an object of type {type(spec)} as a tensorspec but expected a CompositeSpec instance."
            )
        self._spec = spec

    def random(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Samples a random element in the target space, irrespective of any input.

        If multiple output keys are present, only the first will be written in the input :obj:`tensordict`.

        Args:
            tensordict (TensorDictBase): tensordict where the output value should be written.

        Returns:
            the original tensordict with a new/updated value for the output key.

        """
        key0 = self.out_keys[0]
        tensordict.set(key0, self.spec.rand(tensordict.batch_size))
        return tensordict

    def random_sample(self, tensordict: TensorDictBase) -> TensorDictBase:
        """See :obj:`TensorDictModule.random(...)`."""
        return self.random(tensordict)

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> TensorDictModule:
        if hasattr(self, "spec") and self.spec is not None:
            self.spec = self.spec.to(dest)
        out = super().to(dest)
        return out


def is_tensordict_compatible(module: Union[TensorDictModule, nn.Module]):
    """Returns `True` if a module can be used as a TensorDictModule, and False if it can't.

    If the signature is misleading an error is raised.

    Examples:
        >>> module = nn.Linear(3, 4)
        >>> is_tensordict_compatible(module)
        False
        >>> class CustomModule(nn.Module):
        ...    def __init__(self, module):
        ...        super().__init__()
        ...        self.linear = module
        ...        self.in_keys = ["x"]
        ...        self.out_keys = ["y"]
        ...    def forward(self, tensordict):
        ...        tensordict["y"] = self.linear(tensordict["x"])
        ...        return tensordict
        >>> tensordict_module = CustomModule(module)
        >>> is_tensordict_compatible(tensordict_module)
        True
        >>> class CustomModule(nn.Module):
        ...    def __init__(self, module):
        ...        super().__init__()
        ...        self.linear = module
        ...        self.in_keys = ["x"]
        ...        self.out_keys = ["y"]
        ...    def forward(self, tensordict, other_key):
        ...        tensordict["y"] = self.linear(tensordict["x"])
        ...        return tensordict
        >>> tensordict_module = CustomModule(module)
        >>> try:
        ...     is_tensordict_compatible(tensordict_module)
        ... except TypeError:
        ...     print("passing")
        passing
    """
    sig = inspect.signature(module.forward)

    if isinstance(module, TensorDictModule) or (
        len(sig.parameters) == 1
        and hasattr(module, "in_keys")
        and hasattr(module, "out_keys")
    ):
        # if the module is a TensorDictModule or takes a single argument and defines
        # in_keys and out_keys then we assume it can already deal with TensorDict input
        # to forward and we return True
        return True
    elif not hasattr(module, "in_keys") and not hasattr(module, "out_keys"):
        # if it's not a TensorDictModule, and in_keys and out_keys are not defined then
        # we assume no TensorDict compatibility and will try to wrap it.
        return False

    # if in_keys or out_keys were defined but module is not a TensorDictModule or
    # accepts multiple arguments then it's likely the user is trying to do something
    # that will have undetermined behaviour, we raise an error
    raise TypeError(
        "Received a module that defines in_keys or out_keys and also expects multiple "
        "arguments to module.forward. If the module is compatible with TensorDict, it "
        "should take a single argument of type TensorDict to module.forward and define "
        "both in_keys and out_keys. Alternatively, module.forward can accept "
        "arbitrarily many tensor inputs and leave in_keys and out_keys undefined and "
        "TorchRL will attempt to automatically wrap the module with a TensorDictModule."
    )


def ensure_tensordict_compatible(
    module: Union[
        FunctionalModule, FunctionalModuleWithBuffers, TensorDictModule, nn.Module
    ],
    in_keys: Optional[List[NestedKey]] = None,
    out_keys: Optional[List[NestedKey]] = None,
    safe: bool = False,
    wrapper_type: Optional[Type] = TensorDictModule,
    **kwargs,
):
    """Ensures module is compatible with TensorDictModule and, if not, it wraps it."""
    in_keys = unravel_key_list(in_keys) if in_keys else in_keys
    out_keys = unravel_key_list(out_keys) if out_keys else out_keys

    """Checks and ensures an object with forward method is TensorDict compatible."""
    if is_tensordict_compatible(module):
        if in_keys is not None and set(in_keys) != set(module.in_keys):
            raise TypeError(
                f"Arguments to module.forward ({set(module.in_keys)}) doesn't match "
                f"with the expected TensorDict in_keys ({set(in_keys)})."
            )
        if out_keys is not None and set(module.out_keys) != set(out_keys):
            raise TypeError(
                f"Outputs of module.forward ({set(module.out_keys)}) doesn't match "
                f"with the expected TensorDict out_keys ({set(out_keys)})."
            )
        # return module itself if it's already tensordict compatible
        return module

    if not isinstance(module, nn.Module):
        raise TypeError(
            "Argument to ensure_tensordict_compatible should be either "
            "a TensorDictModule or an nn.Module"
        )

    sig = inspect.signature(module.forward)
    if in_keys is not None and set(sig.parameters) != set(in_keys):
        raise TypeError(
            "Arguments to module.forward are incompatible with entries in "
            "env.observation_spec. If you want TorchRL to automatically "
            "wrap your module with a TensorDictModule then the arguments "
            "to module must correspond one-to-one with entries in "
            "in_keys. For more complex behaviour and more control you can "
            "consider writing your own TensorDictModule."
        )

    # TODO: Check whether out_keys match (at least in number) if they are provided.
    if in_keys is not None:
        kwargs["in_keys"] = in_keys
    if out_keys is not None:
        kwargs["out_keys"] = out_keys
    return wrapper_type(module, **kwargs)


class VmapModule(TensorDictModuleBase):
    """A TensorDictModule wrapper to vmap over the input.

    It is intended to be used with modules that accept data with one less batch
    dimension than the one provided. By using this wrapper, one can hide a
    batch dimension and satisfy the wrapped module.

    Args:
        module (TensorDictModuleBase): the module to vmap over.
        vmap_dim (int, optional): the vmap input and output dim.
            If none is provided, the last dimension of the tensordict is
            assumed.

    .. note::

      Since vmap requires to have control over the batch size of the input
      this module does not support dispatched arguments

    Example:
        >>> lam = TensorDictModule(lambda x: x[0], in_keys=["x"], out_keys=["y"])
        >>> sample_in = torch.ones((10,3,2))
        >>> sample_in_td = TensorDict({"x":sample_in}, batch_size=[10])
        >>> lam(sample_in)
        >>> vm = VmapModule(lam, 0)
        >>> vm(sample_in_td)
        >>> assert (sample_in_td["x"][:, 0] == sample_in_td["y"]).all()
    """

    def __init__(self, module: TensorDictModuleBase, vmap_dim=None):
        if not _has_functorch:
            raise ImportError("VmapModule requires torch>=1.13.")
        super().__init__()
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys
        self.module = module
        self.vmap_dim = vmap_dim
        if torch.__version__ >= "2.0":
            self._vmap = torch.vmap
        else:
            import functorch

            self._vmap = functorch.vmap

    def forward(self, tensordict):
        # TODO: there is a risk of segfault if input is not a tensordict.
        # We should investigate (possibly prevent it c++ side?)
        vmap_dim = self.vmap_dim
        if vmap_dim is None:
            ndim = tensordict.ndim
            vmap_dim = ndim - 1
        td = self._vmap(self.module, (vmap_dim,), (vmap_dim,))(tensordict)
        return tensordict.update(td)


class DistributionalDQNnet(TensorDictModuleBase):
    """Distributional Deep Q-Network softmax layer.

    This layer should be used in between a regular model that predicts the
    action values and a distribution which acts on logits values.

    Args:
        in_keys (list of str or tuples of str): input keys to the log-softmax
            operation. Defaults to ``["action_value"]``.
        out_keys (list of str or tuples of str): output keys to the log-softmax
            operation. Defaults to ``["action_value"]``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> net = DistributionalDQNnet()
        >>> td = TensorDict({"action_value": torch.randn(10, 5)}, batch_size=[10])
        >>> net(td)
        TensorDict(
            fields={
                action_value: Tensor(shape=torch.Size([10, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False)

    """

    _wrong_out_feature_dims_error = (
        "DistributionalDQNnet requires dqn output to be at least "
        "2-dimensional, with dimensions *Batch x #Atoms x #Actions. Got {0} "
        "instead."
    )

    def __init__(self, *, in_keys=None, out_keys=None):
        super().__init__()
        if in_keys is None:
            in_keys = ["action_value"]
        if out_keys is None:
            out_keys = ["action_value"]
        self.in_keys = in_keys
        self.out_keys = out_keys

    @dispatch(auto_batch_size=False)
    def forward(self, tensordict):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            q_values = tensordict.get(in_key)
            if q_values.ndimension() < 2:
                raise RuntimeError(
                    self._wrong_out_feature_dims_error.format(q_values.shape)
                )
            tensordict.set(out_key, F.log_softmax(q_values, dim=-2))
        return tensordict
