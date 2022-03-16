__all__ = ["_LossModule"]

from typing import Iterator, Tuple, Optional, Iterable

import torch
from torch import nn
from torch.nn import Parameter

from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.modules import TDModule


class _LossModule(nn.Module):
    """
    A parent class for RL losses.
    _LossModule inherits from nn.Module. It is designed to read an input TensorDict and return another tensordict
    with loss keys named "loss_*".
    Splitting the loss in its component can then be used by the agent to log the various loss values throughout
    training. Other scalars present in the output tensordict will be logged too.
    """

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        """It is designed to read an input TensorDict and return another tensordict
        with loss keys named "loss*".
        Splitting the loss in its component can then be used by the agent to log the various loss values throughout
        training. Other scalars present in the output tensordict will be logged too.

        Args:
            tensordict: an input tensordict with the values required to compute the loss.

        Returns:
            A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
            is essential that the losses are returned with this name as they will be read by the agent before
            backpropagation.
        """
        raise NotImplementedError

    def convert_to_functional(
        self,
        module: TDModule,
        module_name: str,
        expand_dim: Optional[int] = None,
        create_target_params: bool = False,
    ) -> None:

        network_orig = module
        functional_module, (
            _,
            module_buffers,
        ) = module.make_functional_with_buffers(clone=True)
        param_name = module_name + "_params"
        # we keep the original parameters and not the copy returned by functorch
        params = network_orig.parameters()
        # unless we need to expand them, in that case we'll delete the weights to make sure that the user does not
        # run anything with them expecting them to be updated
        if expand_dim:
            params = list(params)
            for i, p in enumerate(params):
                p = p.repeat(expand_dim, *[1 for _ in p.shape])
                p = nn.Parameter(
                    p.uniform_(p.min().item(), p.max().item()).requires_grad_()
                )
                params[i] = p

            module_buffers = list(module_buffers)
            for i, p in enumerate(module_buffers):
                p = p.expand(expand_dim, *p.shape).clone()
                module_buffers[i] = p

            # delete weights of original model as they do not correspond to the optimized weights
            network_orig.to("meta")

        setattr(self, param_name, nn.ParameterList(params))

        # we register each buffer independently
        for i, p in enumerate(module_buffers):
            _name = module_name + f"_buffer_{i}"
            self.register_buffer(_name, p)
            module_buffers[i] = getattr(self, _name)
        buffer_name = module_name + "_buffers"
        setattr(self, buffer_name, module_buffers)

        # we set the functional module
        setattr(self, module_name, functional_module)

        name_params_target = "_target_" + param_name
        name_buffers_target = "_target_" + buffer_name
        if create_target_params:
            target_params = [p.detach().clone() for p in getattr(self, param_name)]
            for i, p in enumerate(target_params):
                name = "_".join([name_params_target, str(i)])
                self.register_buffer(name, p)
                target_params[i] = getattr(self, name)
            setattr(self, name_params_target, target_params)

            target_buffers = [p.detach().clone() for p in getattr(self, buffer_name)]
            for i, p in enumerate(target_buffers):
                name = "_".join([name_buffers_target, str(i)])
                self.register_buffer(name, p)
                target_buffers[i] = getattr(self, name)
            setattr(self, name_buffers_target, target_buffers)
        else:
            setattr(self, name_params_target, None)
            setattr(self, name_buffers_target, None)

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
                return tuple(target_params)
            else:
                return tuple(p.detach() for p in getattr(self, param_name))

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
                return tuple(target_buffers)
            else:
                return tuple(p.detach() for p in getattr(self, buffer_name))

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

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        for name, param in super().named_parameters(
            prefix=prefix, recurse=recurse
        ):
            if not name.startswith("_target"):
                yield name, param

    def reset(self) -> None:
        # mainly used for PPO with KL target
        pass
