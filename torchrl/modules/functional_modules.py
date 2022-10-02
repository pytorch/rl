# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
from torch import nn

from torchrl.data import TensorDict
from torchrl.data.tensordict.tensordict import TensorDictBase

_RESET_OLD_TENSORDICT = True
try:
    import functorch._src.vmap

    _has_functorch = True
except ImportError:
    _has_functorch = False

# Monky-patch functorch, mainly for cases where a "isinstance(obj, Tensor) is invoked
if _has_functorch:
    from functorch._src.vmap import (
        _get_name,
        tree_flatten,
        _broadcast_to_and_flatten,
        Tensor,
        _validate_and_get_batch_size,
        _add_batch_dim,
        tree_unflatten,
        _remove_batch_dim,
    )

    # Monkey-patches

    def _process_batched_inputs(in_dims, args, func):
        if not isinstance(in_dims, int) and not isinstance(in_dims, tuple):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"expected `in_dims` to be int or a (potentially nested) tuple "
                f"matching the structure of inputs, got: {type(in_dims)}."
            )
        if len(args) == 0:
            raise ValueError(
                f"vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add "
                f"inputs, or you are trying to vmap over a function with no inputs. "
                f"The latter is unsupported."
            )

        flat_args, args_spec = tree_flatten(args)
        flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
        if flat_in_dims is None:
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                f"in_dims is not compatible with the structure of `inputs`. "
                f"in_dims has structure {tree_flatten(in_dims)[1]} but inputs "
                f"has structure {args_spec}."
            )

        for i, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
            if not isinstance(in_dim, int) and in_dim is not None:
                raise ValueError(
                    f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                    f"Got in_dim={in_dim} for an input but in_dim must be either "
                    f"an integer dimension or None."
                )
            if isinstance(in_dim, int) and not isinstance(
                arg, (Tensor, TensorDictBase)
            ):
                raise ValueError(
                    f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                    f"Got in_dim={in_dim} for an input but the input is of type "
                    f"{type(arg)}. We cannot vmap over non-Tensor arguments, "
                    f"please use None as the respective in_dim"
                )
            if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
                raise ValueError(
                    f"vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): "
                    f"Got in_dim={in_dim} for some input, but that input is a Tensor "
                    f"of dimensionality {arg.dim()} so expected in_dim to satisfy "
                    f"-{arg.dim()} <= in_dim < {arg.dim()}."
                )
            if in_dim is not None and in_dim < 0:
                flat_in_dims[i] = in_dim % arg.dim()

        return (
            _validate_and_get_batch_size(flat_in_dims, flat_args),
            flat_in_dims,
            flat_args,
            args_spec,
        )

    functorch._src.vmap._process_batched_inputs = _process_batched_inputs

    def _create_batched_inputs(flat_in_dims, flat_args, vmap_level: int, args_spec):
        # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
        # If tensordict, we remove the dim at batch_size[in_dim] such that the TensorDict can accept
        # the batched tensors. This will be added in _unwrap_batched
        batched_inputs = [
            arg
            if in_dim is None
            else arg.apply(
                lambda _arg: _add_batch_dim(_arg, in_dim, vmap_level),
                batch_size=[b for i, b in enumerate(arg.batch_size) if i != in_dim],
            )
            if isinstance(arg, TensorDictBase)
            else _add_batch_dim(arg, in_dim, vmap_level)
            for in_dim, arg in zip(flat_in_dims, flat_args)
        ]
        return tree_unflatten(batched_inputs, args_spec)

    functorch._src.vmap._create_batched_inputs = _create_batched_inputs

    def _unwrap_batched(
        batched_outputs, out_dims, vmap_level: int, batch_size: int, func
    ):
        flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

        for out in flat_batched_outputs:
            # Change here:
            if isinstance(out, (TensorDictBase, torch.Tensor)):
                continue
            raise ValueError(
                f"vmap({_get_name(func)}, ...): `{_get_name(func)}` must only return "
                f"Tensors, got type {type(out)} as a return."
            )

        def incompatible_error():
            raise ValueError(
                f"vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
                f"out_dims is not compatible with the structure of `outputs`. "
                f"out_dims has structure {tree_flatten(out_dims)[1]} but outputs "
                f"has structure {output_spec}."
            )

        # Here:
        if isinstance(batched_outputs, (TensorDictBase, torch.Tensor)):
            # Some weird edge case requires us to spell out the following
            # see test_out_dims_edge_case
            if isinstance(out_dims, int):
                flat_out_dims = [out_dims]
            elif isinstance(out_dims, tuple) and len(out_dims) == 1:
                flat_out_dims = out_dims
                out_dims = out_dims[0]
            else:
                incompatible_error()
        else:
            flat_out_dims = _broadcast_to_and_flatten(out_dims, output_spec)
            if flat_out_dims is None:
                incompatible_error()

        flat_outputs = []
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims):
            if not isinstance(batched_output, TensorDictBase):
                out = _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
            else:
                out = batched_output.apply(
                    lambda x: _remove_batch_dim(x, vmap_level, batch_size, out_dim),
                    batch_size=[batch_size, *batched_output.batch_size],
                )
            flat_outputs.append(out)
        return tree_unflatten(flat_outputs, output_spec)

    functorch._src.vmap._unwrap_batched = _unwrap_batched

# Tensordict-compatible Functional modules


class FunctionalModule(nn.Module):
    """
    This is the callable object returned by :func:`make_functional`.
    """

    def __init__(self, stateless_model):
        super(FunctionalModule, self).__init__()
        self.stateless_model = stateless_model

    @staticmethod
    def _create_from(model, disable_autograd_tracking=False):
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = deepcopy(model)
        param_tensordict = extract_weights(model_copy)
        if disable_autograd_tracking:
            param_tensordict.apply(lambda x: x.requires_grad_(False), inplace=True)
        return FunctionalModule(model_copy), param_tensordict

    def forward(self, params, *args, **kwargs):
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(
            self.stateless_model, params, return_old_tensordict=_RESET_OLD_TENSORDICT
        )
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            if _RESET_OLD_TENSORDICT:
                _swap_state(self.stateless_model, old_state)


class FunctionalModuleWithBuffers(nn.Module):
    """
    This is the callable object returned by :func:`make_functional`.
    """

    def __init__(self, stateless_model):
        super(FunctionalModuleWithBuffers, self).__init__()
        self.stateless_model = stateless_model

    @staticmethod
    def _create_from(model, disable_autograd_tracking=False):
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = deepcopy(model)
        param_tensordict = extract_weights(model_copy)
        buffers = extract_buffers(model_copy)
        if buffers is None:
            buffers = TensorDict(
                {}, param_tensordict.batch_size, device=param_tensordict.device
            )
        if disable_autograd_tracking:
            param_tensordict.apply(lambda x: x.requires_grad_(False), inplace=True)
        return FunctionalModuleWithBuffers(model_copy), param_tensordict, buffers

    def forward(self, params, buffers, *args, **kwargs):
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(
            self.stateless_model, params, return_old_tensordict=_RESET_OLD_TENSORDICT
        )
        old_state_buffers = _swap_state(
            self.stateless_model, buffers, return_old_tensordict=_RESET_OLD_TENSORDICT
        )

        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            if _RESET_OLD_TENSORDICT:
                _swap_state(self.stateless_model, old_state)
                _swap_state(self.stateless_model, old_state_buffers)


# Some utils for these


def extract_weights(model):
    tensordict = TensorDict({}, [])
    for name, param in list(model.named_parameters(recurse=False)):
        setattr(model, name, None)
        tensordict[name] = param
    for name, module in model.named_children():
        module_tensordict = extract_weights(module)
        if module_tensordict is not None:
            tensordict[name] = module_tensordict
    if len(tensordict.keys()):
        return tensordict
    else:
        return None


def extract_buffers(model):
    tensordict = TensorDict({}, [])
    for name, param in list(model.named_buffers(recurse=False)):
        setattr(model, name, None)
        tensordict[name] = param
    for name, module in model.named_children():
        module_tensordict = extract_buffers(module)
        if module_tensordict is not None:
            tensordict[name] = module_tensordict
    if len(tensordict.keys()):
        return tensordict
    else:
        return None


def _swap_state(model, tensordict, return_old_tensordict=False):
    #     if return_old_tensordict:
    #         old_tensordict = tensordict.clone(recurse=False)
    #         old_tensordict.batch_size = []

    if return_old_tensordict:
        old_tensordict = TensorDict({}, [], device=tensordict.device)

    for key, value in list(tensordict.items()):
        if isinstance(value, TensorDictBase):
            _swap_state(getattr(model, key), value)
        else:
            if return_old_tensordict:
                old_attr = getattr(model, key)
                if old_attr is None:
                    old_attr = torch.tensor([]).view(*value.shape, 0)
            delattr(model, key)
            setattr(model, key, value)
            if return_old_tensordict:
                old_tensordict.set(key, old_attr)
    if return_old_tensordict:
        return old_tensordict
