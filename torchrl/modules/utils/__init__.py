# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from packaging import version


if version.parse(torch.__version__) >= version.parse("1.12.0"):
    from torch.nn.parameter import _ParameterMeta
else:
    pass

    # Metaclass to combine _TensorMeta and the instance check override for Parameter.
    class _ParameterMeta(torch._C._TensorMeta):
        # Make `isinstance(t, Parameter)` return True for custom tensor instances that have the _is_param flag.
        def __instancecheck__(self, instance):
            return super().__instancecheck__(instance) or (
                isinstance(instance, torch.Tensor)
                and getattr(instance, "_is_param", False)
            )


from .mappings import biased_softplus, inv_softplus, mappings
from .utils import get_primers_from_module

__all__ = [
    "OrderedDict",
    "torch",
    "version",
    "biased_softplus",
    "inv_softplus",
    "mappings",
    "get_primers_from_module",
]
