from copy import deepcopy

import torch
from torch import nn

from torchrl.data import TensorDict
from torchrl.data.tensordict.tensordict import TensorDictBase

_RESET_OLD_TENSORDICT = True


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
            buffers = TensorDict({}, param_tensordict.batch_size, device=param_tensordict.device)
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
    #         old_tensordict = tensordict.clone(recursive=False)
    #         old_tensordict.batch_size = []

    if return_old_tensordict:
        old_tensordict = TensorDict({}, [], device=tensordict._device_safe())

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
