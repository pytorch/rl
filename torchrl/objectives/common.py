# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Iterator, List, Optional, Tuple, Union

import torch

from tensordict.nn import make_functional, repopulate_module

from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn, Tensor
from torch.nn import Parameter

from torchrl.modules import SafeModule

_has_functorch = False
try:
    import functorch

    _has_functorch = True
except ImportError:
    print(
        "failed to import functorch. TorchRL's features that do not require "
        "functional programming should work, but functionality and performance "
        "may be affected. Consider installing functorch and/or upgrating pytorch."
    )
    FUNCTORCH_ERROR = "functorch not installed. Consider installing functorch to use this functionality."


class LossModule(nn.Module):
    """A parent class for RL losses.

    LossModule inherits from nn.Module. It is designed to read an input TensorDict and return another tensordict
    with loss keys named "loss_*".
    Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
    training. Other scalars present in the output tensordict will be logged too.

    """

    def __init__(self):
        super().__init__()
        self._param_maps = {}

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

        Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
        training. Other scalars present in the output tensordict will be logged too.

        Args:
            tensordict: an input tensordict with the values required to compute the loss.

        Returns:
            A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
            is essential that the losses are returned with this name as they will be read by the trainer before
            backpropagation.

        """
        raise NotImplementedError

    def convert_to_functional(
        self,
        module: SafeModule,
        module_name: str,
        expand_dim: Optional[int] = None,
        create_target_params: bool = False,
        compare_against: Optional[List[Parameter]] = None,
    ) -> None:
        return self._convert_to_functional(
            module,
            module_name,
            expand_dim,
            create_target_params,
            compare_against,
        )

    def _convert_to_functional(
        self,
        module: SafeModule,
        module_name: str,
        expand_dim: Optional[int] = None,
        create_target_params: bool = False,
        compare_against: Optional[List[Parameter]] = None,
    ) -> None:
        # To make it robust to device casting, we must register list of
        # tensors as lazy calls to `getattr(self, name_of_tensor)`.
        # Otherwise, casting the module to a device will keep old references
        # to uncast tensors
        try:
            buffer_names = next(itertools.islice(zip(*module.named_buffers()), 1))
        except StopIteration:
            buffer_names = ()
        params = make_functional(module)
        functional_module = deepcopy(module)
        module = repopulate_module(module, params)

        # separate params and buffers
        params_and_buffers = params
        buffers = params.flatten_keys(".").select(*buffer_names)
        params = params.flatten_keys(".").exclude(*buffer_names)
        # we transform the buffers in params to make sure they follow the device
        # as tensor = nn.Parameter(tensor) keeps its identity when moved to another device
        def buffer_to_param(tensor):
            if isinstance(tensor, torch.Tensor) and not isinstance(
                tensor, nn.Parameter
            ):
                return nn.Parameter(tensor, requires_grad=tensor.requires_grad)
            return tensor

        params_and_buffers = params_and_buffers.apply(buffer_to_param)

        if expand_dim and not _has_functorch:
            raise ImportError(
                "expanding params is only possible when functorch is installed,"
                "as this feature requires calls to the vmap operator."
            )

        param_name = module_name + "_params"

        params_list = list(params.values())
        prev_set_params = set(self.parameters())
        setattr(
            self,
            "_" + param_name,
            nn.ParameterList(
                [
                    p
                    for p in params_list
                    if isinstance(p, nn.Parameter) and p not in prev_set_params
                ]
            ),
        )
        setattr(self, param_name, params_and_buffers)
        # we set the functional module
        setattr(self, module_name, functional_module)

        name_params_target = "_target_" + param_name
        print("create_target_params", create_target_params)
        if create_target_params:
            target_params = getattr(self, param_name).detach().clone()
            target_params_items = sorted(target_params.flatten_keys(".").items())
            target_params_list = []
            for i, (key, val) in enumerate(target_params_items):
                name = "_".join([name_params_target, str(i)])
                self.register_buffer(name, val)
                target_params_list.append((name, key))
            setattr(
                self.__class__,
                name_params_target,
                property(
                    lambda _self: TensorDict(
                        {
                            key: getattr(_self, _name)
                            for (_name, key) in target_params_list
                        },
                        [],
                        device=self.device,
                    ).unflatten_keys(".")
                ),
            )

        else:
            setattr(self.__class__, name_params_target, None)

        setattr(
            self.__class__,
            name_params_target[1:],
            property(lambda _self: self._target_param_getter(module_name)),
        )

    def _target_param_getter(self, network_name):
        target_name = "_target_" + network_name + "_params"
        param_name = network_name + "_params"
        if hasattr(self, target_name):
            target_params = getattr(self, target_name)
            if target_params is not None:
                return target_params
            else:
                params = getattr(self, param_name)
                return params.detach()

        else:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have the target param {target_name}"
            )

    def _networks(self) -> Iterator[nn.Module]:
        for item in self.__dir__():
            if isinstance(item, nn.Module):
                yield item

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        tensor = tensor.to(self.device)
        return super().register_buffer(name, tensor, persistent)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
            if not name.startswith("_target"):
                yield name, param

    def reset(self) -> None:
        # mainly used for PPO with KL target
        pass

    def to(self, *args, **kwargs):
        # get the names of the parameters to map
        out = super().to(*args, **kwargs)
        lists_of_params = {
            name: value
            for name, value in self.__dict__.items()
            if name.endswith("_params") and (type(value) is list)
        }
        for _, list_of_params in lists_of_params.items():
            for i, param in enumerate(list_of_params):
                # we replace the param by the expanded form if needs be
                if param in self._param_maps:
                    list_of_params[i] = self._param_maps[param].data.expand_as(param)
        return out

    def cuda(self, device: Optional[Union[int, device]] = None) -> LossModule:
        if device is None:
            return self.to("cuda")
        else:
            return self.to(device)

    def double(self) -> LossModule:
        return self.to(torch.double)

    def float(self) -> LossModule:
        return self.to(torch.float)

    def half(self) -> LossModule:
        return self.to(torch.half)

    def cpu(self) -> LossModule:
        return self.to(torch.device("cpu"))
