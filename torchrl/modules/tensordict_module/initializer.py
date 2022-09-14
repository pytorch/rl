# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from torchrl.modules.tensordict_module.common import TensorDictModule

from torch import nn, Tensor

from torchrl.data.tensordict.tensordict import TensorDict, TensorDictBase

class _DefaultInitializer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, initializer_dict: dict, batch_size, device) -> dict:
        outputs = {}
        for key, value in initializer_dict.items():
            if "args" in value and "kwargs" in value:
                outputs[key] = value["initializer"]( *batch_size, *value["shape"],
                        *value["args"], **value["kwargs"], device=device
                )
            elif "args" in value:
                outputs[key] = value["initializer"]( *batch_size, *value["shape"],
                        *value["args"], device=device
                )
            elif "kwargs" in value:
                outputs[key] = value["initializer"]( *batch_size, *value["shape"],
                        **value["kwargs"], device=device
                )
            else:
                outputs[key] = value["initializer"]( *batch_size, *value["shape"], device=device)
        return outputs

class TensorDictDefaultInitializer(TensorDictModule):
    """
    TensorDictDefaultInitializer
    This module is meant to initialize missing values in a TensorDict.
    It takes a dictionary of default parameters and initializes the TensorDict with them.

    Args:
        defaults_init (Dict[Dict]): Dictionary of default parameters to initialize the TensorDict with.
            Takes the form of {key: {"initializer": initializer, "shape": shape, "args": args, "kwargs": kwargs}}
            with key key associated with the tensor to be initialized, initializer the initializer function,
            shape the shape of the tensor, args the positional arguments to pass to the initializer function,
            and kwargs the keyword arguments to pass to the initializer function.
            the initializer function must be of the following form initializer (*batch_size, *args, **kwargs, device=device)
        reinit (bool): Whether to reinitialize already initialized tensors.
    
    Returns:
        TensorDictBase: The TensorDict to whom we added the initialized tensors.


    """
    def __init__(
        self,
        defaults_init: Dict[Dict],
        reinit = False,
    ):

        module = _DefaultInitializer()

        in_keys = []
        out_keys = defaults_init.keys()
        self.reinit = reinit

        self.defaults_init = defaults_init

        super().__init__(module, in_keys, out_keys)

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
        params: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        buffers: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        **kwargs,
    ) -> TensorDictBase:
        batch_size = tensordict.batch_size
        device = tensordict.device_safe()
        tensors = self.module(self.defaults_init, batch_size, device)
        if not self.reinit:
            tensors = {k: v for k, v in tensors.items() if k not in tensordict}
        new_tensors_td = TensorDict(tensors, batch_size=batch_size)
        if tensordict_out is None:
            tensordict_out = tensordict
        tensordict_out.update(new_tensors_td)
        return tensordict_out