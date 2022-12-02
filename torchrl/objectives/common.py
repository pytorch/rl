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

from tensordict.tensordict import TensorDictBase
from torch import nn, Tensor
from torch.nn import Parameter

from torchrl.modules import SafeModule
from torchrl.modules.utils import Buffer

_has_functorch = False
try:
    import functorch as ft  # noqa

    _has_functorch = True
    FUNCTORCH_ERR = ""
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
        # self.register_forward_pre_hook(_parameters_to_tensordict)

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
        repopulate_module(module, params)

        params_and_buffers = params
        # we transform the buffers in params to make sure they follow the device
        # as tensor = nn.Parameter(tensor) keeps its identity when moved to another device

        def create_buffers(tensor):

            if isinstance(tensor, torch.Tensor) and not isinstance(
                tensor, (Buffer, nn.Parameter)
            ):
                return Buffer(tensor, requires_grad=tensor.requires_grad)
            return tensor

        # separate params and buffers
        params_and_buffers = params_and_buffers.apply(create_buffers)
        for key in params_and_buffers.keys(True):
            if "_sep_" in key:
                raise KeyError(
                    f"The key {key} contains the '_sep_' pattern which is prohibited. Consider renaming the parameter / buffer."
                )
        params_and_buffers_flat = params_and_buffers.flatten_keys("_sep_")
        buffers = params_and_buffers_flat.select(*buffer_names)
        params = params_and_buffers_flat.exclude(*buffer_names)

        if expand_dim and not _has_functorch:
            raise ImportError(
                "expanding params is only possible when functorch is installed,"
                "as this feature requires calls to the vmap operator."
            )
        if expand_dim:
            # Expands the dims of params and buffers.
            # If the param already exist in the module, we return a simple expansion of the
            # original one. Otherwise, we expand and resample it.
            # For buffers, a cloned expansion (or equivalently a repeat) is returned.
            if compare_against is not None:
                compare_against = set(compare_against)
            else:
                compare_against = set()

            def _compare_and_expand(param):

                if param in compare_against:
                    expanded_param = param.data.expand(expand_dim, *param.shape)
                    # the expanded parameter must be sent to device when to()
                    # is called:
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

            params = params_udpated
            buffers = buffers.apply(
                lambda buffer: Buffer(buffer.expand(expand_dim, *buffer.shape).clone()),
                batch_size=[expand_dim, *buffers.shape],
            )

            params_and_buffers.update(params.unflatten_keys("_sep_"))
            params_and_buffers.update(buffers.unflatten_keys("_sep_"))
            params_and_buffers.batch_size = params.batch_size

            # self.params_to_map = params_to_map

        param_name = module_name + "_params"

        prev_set_params = set(self.parameters())

        # register parameters and buffers
        for key, parameter in params.items():
            if parameter not in prev_set_params:
                setattr(self, "_sep_".join([module_name, key]), parameter)
            else:
                for _param_name, p in self.named_parameters():
                    if parameter is p:
                        break
                else:
                    raise RuntimeError("parameter not found")
                setattr(self, "_sep_".join([module_name, key]), _param_name)
        prev_set_buffers = set(self.buffers())
        for key, buffer in buffers.items():
            if buffer not in prev_set_buffers:
                self.register_buffer("_sep_".join([module_name, key]), buffer)
            else:
                for _buffer_name, b in self.named_buffers():
                    if buffer is b:
                        break
                else:
                    raise RuntimeError("buffer not found")
                setattr(self, "_sep_".join([module_name, key]), _buffer_name)

        setattr(self, "_" + param_name, params_and_buffers)
        setattr(
            self.__class__,
            param_name,
            property(lambda _self=self: _self._param_getter(module_name)),
        )

        # set the functional module
        setattr(self, module_name, functional_module)

        # creates a map nn.Parameter name -> expanded parameter name
        for key, value in params.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            if not isinstance(value, nn.Parameter):
                # find the param name
                for name, param in self.named_parameters():
                    if param.data.data_ptr() == value.data_ptr() and param is not value:
                        self._param_maps[name] = "_sep_".join([module_name, *key])
                        break
                else:
                    raise RuntimeError("did not find matching param.")

        name_params_target = "_target_" + module_name
        if create_target_params:
            target_params = params_and_buffers.detach().clone()
            target_params_items = target_params.items(True, True)
            target_params_list = []
            for (key, val) in target_params_items:
                if not isinstance(key, tuple):
                    key = (key,)
                name = "_sep_".join([name_params_target, *key])
                self.register_buffer(name, Buffer(val))
                target_params_list.append((name, key))
            setattr(self, name_params_target + "_params", target_params)
        else:
            setattr(self, name_params_target + "_params", None)
        setattr(
            self.__class__,
            name_params_target[1:] + "_params",
            property(lambda _self=self: _self._target_param_getter(module_name)),
        )

    def _param_getter(self, network_name):
        name = "_" + network_name + "_params"
        param_name = network_name + "_params"
        if name in self.__dict__:
            params = getattr(self, name)
            if params is not None:
                # get targets and update
                for key in params.keys(True, True):
                    if not isinstance(key, tuple):
                        key = (key,)
                    value_to_set = getattr(self, "_sep_".join([network_name, *key]))
                    if isinstance(value_to_set, str):
                        value_to_set = getattr(self, value_to_set).detach()
                    params.set(key, value_to_set)
                return params
            else:
                params = getattr(self, param_name)
                return params.detach()

        else:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have the target param {name}"
            )

    def _target_param_getter(self, network_name):
        target_name = "_target_" + network_name + "_params"
        param_name = network_name + "_params"
        if target_name in self.__dict__:
            target_params = getattr(self, target_name)
            if target_params is not None:
                # get targets and update
                for key in target_params.keys(True, True):
                    if not isinstance(key, tuple):
                        key = (key,)
                    value_to_set = getattr(
                        self, "_sep_".join(["_target_" + network_name, *key])
                    )
                    target_params.set(key, value_to_set)
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
        # tensor = tensor.to(self.device)
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
        for origin, target in self._param_maps.items():
            origin_value = getattr(self, origin)
            target_value = getattr(self, target)
            setattr(self, target, origin_value.expand_as(target_value))

        # lists_of_params = {
        #     name: value
        #     for name, value in self.__dict__.items()
        #     if name.endswith("_params") and isinstance(value, TensorDictBase)
        # }
        # for list_of_params in lists_of_params.values():
        #     for key, param in list(list_of_params.items(True)):
        #         if isinstance(param, TensorDictBase):
        #             continue
        #         # we replace the param by the expanded form if needs be
        #         if param in self._param_maps:
        #             list_of_params[key] = self._param_maps[param].data.expand_as(param)
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
