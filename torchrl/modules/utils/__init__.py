# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.12.0"):
    from torch.nn.parameter import _disabled_torch_function_impl, _ParameterMeta
else:
    from torch.nn.parameter import _disabled_torch_function_impl

    # Metaclass to combine _TensorMeta and the instance check override for Parameter.
    class _ParameterMeta(torch._C._TensorMeta):
        # Make `isinstance(t, Parameter)` return True for custom tensor instances that have the _is_param flag.
        def __instancecheck__(self, instance):
            return super().__instancecheck__(instance) or (
                isinstance(instance, torch.Tensor)
                and getattr(instance, "_is_param", False)
            )


from .mappings import biased_softplus, inv_softplus, mappings


class Buffer(torch.Tensor, metaclass=_ParameterMeta):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`locally-disable-grad-doc` for more details. Default: `True`
    """

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Buffer:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return torch.Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a Parameter from an instance of type {type(data).__name__} "
                "requires that detach() returns an instance of the same type, but return "
                f"type {type(t).__name__} was found instead. To use the type as a "
                "Parameter, please correct the detach() semantics defined by "
                "its __torch_dispatch__() implementation."
            )
        t._is_param = True
        return t

    # Note: the 3 methods below only apply to standard Tensor. Parameters of custom tensor types
    # are still considered that custom tensor type and these methods will not be called for them.
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format), self.requires_grad
            )
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "Buffer containing:\n" + super(Buffer, self).__repr__()

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict()),
        )

    __torch_function__ = _disabled_torch_function_impl
