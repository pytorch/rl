# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torchrl.modules import TensorDictModule

__all__ = [
    "partial_tensordictmodule",
]


def partial_tensordictmodule(
    module, in_keys, out_keys, spec=None, safe=False, **kwargs
):
    return TensorDictModule(
        module=module(**kwargs),
        in_keys=in_keys,
        out_keys=out_keys,
        spec=spec,
        safe=safe,
    )
