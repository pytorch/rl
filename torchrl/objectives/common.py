# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterator, Optional, Tuple, List, Union

import torch

from torchrl.modules.functional_modules import FunctionalModuleWithBuffers

_has_functorch = False
try:
    import functorch
    from functorch._src.make_functional import _swap_state

    _has_functorch = True
except ImportError:
    print(
        "failed to import functorch. TorchRL's features that do not require "
        "functional programming should work, but functionality and performance "
        "may be affected. Consider installing functorch and/or upgrating pytorch."
    )
    FUNCTORCH_ERROR = "functorch not installed. Consider installing functorch to use this functionality."

from tensordict.tensordict import TensorDictBase, TensorDict
from torch import nn, Tensor
from torch.nn import Parameter

from torchrl.modules import TensorDictModule


class LossModule(nn.Module):
    """A parent class for RL losses.

    LossModule inherits from nn.Module. It is designed to read an input TensorDict and return another tensordict
    with loss keys named "loss_*".
    Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
    training. Other scalars present in the output tensordict will be logged too.

    """

    def __init__(self):
        super().__init__()
        self._param_maps = dict()

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
        module: TensorDictModule,
        module_name: str,
        expand_dim: Optional[int] = None,
        create_target_params: bool = False,
        compare_against: Optional[List[Parameter]] = None,
    ) -> None:
        if _has_functorch:
            return self._convert_to_functional_functorch(
                module,
                module_name,
                expand_dim,
                create_target_params,
                compare_against,
            )
        else:
            return self._convert_to_functional_native(
                module,
                module_name,
                expand_dim,
                create_target_params,
                compare_against,
            )

    def _convert_to_functional_functorch(
        self,
        module: TensorDictModule,
        module_name: str,
        expand_dim: Optional[int] = None,
        create_target_params: bool = False,
        compare_against: Optional[List[Parameter]] = None,
    ) -> None:
        # To make it robust to device casting, we must register list of
        # tensors as lazy calls to `getattr(self, name_of_tensor)`.
        # Otherwise, casting the module to a device will keep old references
        # to uncast tensors

        network_orig = module
        if hasattr(module, "make_functional_with_buffers"):
            functional_module, (
                _,
                module_buffers,
            ) = module.make_functional_with_buffers(clone=True)
        else:
            (
                functional_module,
                module_params,
                module_buffers,
            ) = functorch.make_functional_with_buffers(module)

            for _ in functional_module.parameters():
                # Erase meta params
                none_state = [None for _ in module_params + module_buffers]
                if hasattr(functional_module, "all_names_map"):
                    # functorch >= 0.2.0
                    _swap_state(
                        functional_module.stateless_model,
                        functional_module.all_names_map,
                        none_state,
                    )
                else:
                    # functorch < 0.2.0
                    _swap_state(
                        functional_module.stateless_model,
                        functional_module.split_names,
                        none_state,
                    )
                break
            del module_params

        param_name = module_name + "_params"

        # we keep the original parameters and not the copy returned by functorch
        params = network_orig.parameters()

        # unless we need to expand them, in that case we'll delete the weights to make sure that the user does not
        # run anything with them expecting them to be updated
        params = list(params)
        module_buffers = list(module_buffers)

        if expand_dim:
            if compare_against is not None:
                compare_against = set(compare_against)
            else:
                compare_against = set()
            for i, p in enumerate(params):
                if p in compare_against:
                    # expanded parameters are 'detached': the parameter will not
                    # be trained to minimize loss involving this network.
                    p_out = p.data.expand(expand_dim, *p.shape)
                    # the expanded parameter must be sent to device when to()
                    # is called:
                    self._param_maps[p_out] = p
                else:
                    p_out = p.repeat(expand_dim, *[1 for _ in p.shape])
                    p_out = nn.Parameter(
                        p_out.uniform_(
                            p_out.min().item(), p_out.max().item()
                        ).requires_grad_()
                    )
                params[i] = p_out

            for i, b in enumerate(module_buffers):
                b = b.expand(expand_dim, *b.shape).clone()
                module_buffers[i] = b

            # # delete weights of original model as they do not correspond to the optimized weights
            # network_orig.to("meta")

        params_list = params
        set_params = set(self.parameters())
        setattr(
            self,
            "_" + param_name,
            nn.ParameterList(
                [
                    p
                    for p in params_list
                    if isinstance(p, nn.Parameter) and p not in set_params
                ]
            ),
        )
        setattr(self, param_name, params)

        buffer_name = module_name + "_buffers"
        # we register each buffer independently
        for i, p in enumerate(module_buffers):
            _name = module_name + f"_buffer_{i}"
            self.register_buffer(_name, p)
            # replace buffer by its name
            module_buffers[i] = _name
        setattr(
            self.__class__,
            buffer_name,
            property(lambda _self: [getattr(_self, _name) for _name in module_buffers]),
        )

        # we set the functional module
        setattr(self, module_name, functional_module)

        name_params_target = "_target_" + param_name
        name_buffers_target = "_target_" + buffer_name
        if create_target_params:
            target_params = [p.detach().clone() for p in getattr(self, param_name)]
            for i, p in enumerate(target_params):
                name = "_".join([name_params_target, str(i)])
                self.register_buffer(name, p)
                target_params[i] = name
            setattr(
                self.__class__,
                name_params_target,
                property(
                    lambda _self: [getattr(_self, _name) for _name in target_params]
                ),
            )

            target_buffers = [p.detach().clone() for p in getattr(self, buffer_name)]
            for i, p in enumerate(target_buffers):
                name = "_".join([name_buffers_target, str(i)])
                self.register_buffer(name, p)
                target_buffers[i] = name
            setattr(
                self.__class__,
                name_buffers_target,
                property(
                    lambda _self: [getattr(_self, _name) for _name in target_buffers]
                ),
            )

        else:
            setattr(self.__class__, name_params_target, None)
            setattr(self.__class__, name_buffers_target, None)

        setattr(
            self.__class__,
            name_params_target[1:],
            property(lambda _self: self._target_param_getter(module_name)),
        )
        setattr(
            self.__class__,
            name_buffers_target[1:],
            property(lambda _self: self._target_buffer_getter(module_name)),
        )

    def _convert_to_functional_native(
        self,
        module: TensorDictModule,
        module_name: str,
        expand_dim: Optional[int] = None,
        create_target_params: bool = False,
        compare_against: Optional[List[Parameter]] = None,
    ) -> None:
        # To make it robust to device casting, we must register list of
        # tensors as lazy calls to `getattr(self, name_of_tensor)`.
        # Otherwise, casting the module to a device will keep old references
        # to uncast tensors

        network_orig = module
        if hasattr(module, "make_functional_with_buffers"):
            functional_module, (
                params,
                module_buffers,
            ) = module.make_functional_with_buffers(clone=True)
        else:
            (
                functional_module,
                params,
                module_buffers,
            ) = FunctionalModuleWithBuffers._create_from(module)

        param_name = module_name + "_params"

        # params must be retrieved directly because make_functional will copy the content
        params_vals = TensorDict(
            {name: value for name, value in network_orig.named_parameters()}, []
        )
        # rename params_vals keys to match params: otherwise we'll have to deal with
        # module.module.param or such names. We assume that there is a constant prefix
        # and that, when sorted, all keys will match. We could check that the values
        # do match too.
        keys1 = sorted(list(params.flatten_keys(".").keys()))
        keys2 = sorted(list(params_vals.keys()))
        for key1, key2 in zip(keys1, keys2):
            params_vals.rename_key(key2, key1)
        params = params_vals.unflatten_keys(".")

        if expand_dim:
            raise ImportError(
                "expanding params is only possible when functorch is installed,"
                "as this feature requires calls to the vmap operator."
            )

        params_list = list(params.flatten_keys(".").values())
        set_params = set(self.parameters())
        setattr(
            self,
            "_" + param_name,
            nn.ParameterList(
                [
                    p
                    for p in params_list
                    if isinstance(p, nn.Parameter) and p not in set_params
                ]
            ),
        )
        setattr(self, param_name, params)

        buffer_name = module_name + "_buffers"
        buffers_iter = list(module_buffers.flatten_keys(".").items())
        module_buffers_list = []
        for i, (key, value) in enumerate(sorted(buffers_iter)):
            _name = module_name + f"_buffer_{i}"
            self.register_buffer(_name, value)
            # replace buffer by its name
            module_buffers_list.append((_name, key))
        setattr(
            self.__class__,
            buffer_name,
            property(
                lambda _self: TensorDict(
                    {
                        key: getattr(_self, _name)
                        for (_name, key) in module_buffers_list
                    },
                    [],
                    device=self.device,
                ).unflatten_keys(".")
            ),
        )

        # we set the functional module
        setattr(self, module_name, functional_module)

        name_params_target = "_target_" + param_name
        name_buffers_target = "_target_" + buffer_name
        if create_target_params:
            target_params = getattr(self, param_name).detach().clone()
            target_params_items = sorted(list(target_params.flatten_keys(".").items()))
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

            target_buffers = getattr(self, buffer_name).detach().clone()
            target_buffers_items = sorted(
                list(target_buffers.flatten_keys(".").items())
            )
            target_buffers_list = []
            for i, (key, val) in enumerate(target_buffers_items):
                name = "_".join([name_buffers_target, str(i)])
                self.register_buffer(name, val)
                target_buffers_list.append((name, key))
            setattr(
                self.__class__,
                name_buffers_target,
                property(
                    lambda _self: TensorDict(
                        {
                            key: getattr(_self, _name)
                            for (_name, key) in target_buffers_list
                        },
                        [],
                        device=self.device,
                    ).unflatten_keys(".")
                ),
            )

        else:
            setattr(self.__class__, name_params_target, None)
            setattr(self.__class__, name_buffers_target, None)

        setattr(
            self.__class__,
            name_params_target[1:],
            property(lambda _self: self._target_param_getter(module_name)),
        )
        setattr(
            self.__class__,
            name_buffers_target[1:],
            property(lambda _self: self._target_buffer_getter(module_name)),
        )

    def _target_param_getter(self, network_name):
        target_name = "_target_" + network_name + "_params"
        param_name = network_name + "_params"
        if hasattr(self, target_name):
            target_params = getattr(self, target_name)
            if target_params is not None:
                if isinstance(target_params, TensorDictBase):
                    return target_params
                return tuple(target_params)
            else:
                params = getattr(self, param_name)
                if isinstance(params, TensorDictBase):
                    return params.detach()
                else:
                    # detach params as a surrogate for targets
                    return tuple(p.detach() for p in params)

        else:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have the target param {target_name}"
            )

    def _target_buffer_getter(self, network_name):
        target_name = "_target_" + network_name + "_buffers"
        buffer_name = network_name + "_buffers"
        if hasattr(self, target_name):
            target_buffers = getattr(self, target_name)
            if target_buffers is not None:
                if isinstance(target_buffers, TensorDictBase):
                    return target_buffers
                return tuple(target_buffers)
            else:
                buffers = getattr(self, buffer_name)
                if isinstance(buffers, TensorDictBase):
                    return buffers.detach()
                else:
                    return tuple(p.detach() for p in buffers)

        else:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have the target buffer {target_name}"
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
