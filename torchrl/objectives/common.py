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
from torchrl.modules.utils import Buffer

_has_functorch = False
try:
    import functorch as ft  # noqa

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
        funs_to_decorate=None,
    ) -> None:
        if funs_to_decorate is None:
            funs_to_decorate = ["forward"]
        # To make it robust to device casting, we must register list of
        # tensors as lazy calls to `getattr(self, name_of_tensor)`.
        # Otherwise, casting the module to a device will keep old references
        # to uncast tensors
        try:
            buffer_names = next(itertools.islice(zip(*module.named_buffers()), 1))
        except StopIteration:
            buffer_names = ()
        params = make_functional(module, funs_to_decorate=funs_to_decorate)
        functional_module = deepcopy(module)
        module = repopulate_module(module, params)

        # separate params and buffers
        params_and_buffers = params
        params_and_buffers_flat = params.flatten_keys(".")
        buffers = params_and_buffers_flat.select(*buffer_names)
        params = params_and_buffers_flat.exclude(*buffer_names)

        # we transform the buffers in params to make sure they follow the device
        # as tensor = nn.Parameter(tensor) keeps its identity when moved to another device
        def buffer_to_param(tensor):

            if isinstance(tensor, torch.Tensor) and not isinstance(
                tensor, nn.Parameter
            ):
                return Buffer(tensor, requires_grad=tensor.requires_grad)
            return tensor

        params_and_buffers = params_and_buffers.apply(buffer_to_param)

        if expand_dim and not _has_functorch:
            raise ImportError(
                "expanding params is only possible when functorch is installed,"
                "as this feature requires calls to the vmap operator."
            )
        if expand_dim:
            if compare_against is not None:
                compare_against = set(compare_against)
            else:
                compare_against = set()

            def _compare_and_expand(param):

                if param in compare_against:
                    expanded_param = param.data.expand(expand_dim, *param.shape)
                    # the expanded parameter must be sent to device when to()
                    # is called:
                    self._param_maps[expanded_param] = param
                    return expanded_param
                else:
                    p_out = param.repeat(expand_dim, *[1 for _ in param.shape])
                    p_out = nn.Parameter(
                        p_out.uniform_(
                            p_out.min().item(), p_out.max().item()
                        ).requires_grad_()
                    )
                    return p_out

            params_udpated = params.apply(
                _compare_and_expand, batch_size=[expand_dim, *params.shape]
            )

            def select_params(param):
                if isinstance(param, nn.Parameter):
                    return param

            params_all = params.update(params_udpated.apply(select_params))
            params = params_udpated
            params_list = list(params_all.values())
            buffers = buffers.apply(
                lambda buffer: buffer.expand(expand_dim, *buffer.shape).clone(),
                batch_size=[expand_dim, *buffers.shape],
            )

            params_and_buffers.update(params.unflatten_keys("."))
            params_and_buffers.update(buffers.unflatten_keys("."))
            params_and_buffers.batch_size = params.batch_size
        else:
            params_list = list(params.values())

        param_name = module_name + "_params"

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
        if create_target_params:
            target_params = getattr(self, param_name).detach().clone()
            target_params_items = sorted(target_params.flatten_keys(".").items())
            target_params_list = []
            for i, (key, val) in enumerate(target_params_items):
                name = "_".join([name_params_target, str(i)])
                self.register_buffer(name, val)
                target_params_list.append((name, key))
            # TODO: simplify the tensordict construction
            # We create a property as because setting the tensordict as an attribute
            # won't cast it to the right device when .to() is being called.
            # The property fetches the tensors and rebuilds the tensordict
            setattr(
                self.__class__,
                name_params_target,
                property(
                    lambda _self: TensorDict(
                        {
                            key: getattr(_self, _name)
                            for (_name, key) in target_params_list
                        },
                        target_params.batch_size,
                        device=self.device,
                        _run_checks=False,
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
            if name.endswith("_params") and isinstance(value, TensorDictBase)
        }
        for list_of_params in lists_of_params.values():
            for key, param in list(list_of_params.items(True)):
                if isinstance(param, TensorDictBase):
                    continue
                # we replace the param by the expanded form if needs be
                if param in self._param_maps:
                    list_of_params[key] = self._param_maps[param].data.expand_as(param)
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
