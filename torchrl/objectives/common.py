# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from tensordict import TensorDictBase

from tensordict.nn import (
    make_functional,
    repopulate_module,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictParams,
)
from torch import nn
from torch.nn import Parameter

from torchrl._utils import RL_WARNINGS
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.utils import ValueEstimators
from torchrl.objectives.value import ValueEstimatorBase


def _updater_check_forward_prehook(module, *args, **kwargs):
    if not all(v for v in module._has_update_associated.values()) and RL_WARNINGS:
        warnings.warn(
            module.TARGET_NET_WARNING,
            category=UserWarning,
        )


class LossModule(TensorDictModuleBase):
    """A parent class for RL losses.

    LossModule inherits from nn.Module. It is designed to read an input
    TensorDict and return another tensordict
    with loss keys named ``"loss_*"``.

    Splitting the loss in its component can then be used by the trainer to log
    the various loss values throughout
    training. Other scalars present in the output tensordict will be logged too.

    :cvar default_value_estimator: The default value type of the class.
        Losses that require a value estimation are equipped with a default value
        pointer. This class attribute indicates which value estimator will be
        used if none other is specified.
        The value estimator can be changed using the :meth:`~.make_value_estimator` method.

    By default, the forward method is always decorated with a
    gh :class:`torchrl.envs.ExplorationType.MODE`

    To utilize the ability configuring the tensordict keys via
    :meth:`~.set_keys()` a subclass must define an _AcceptedKeys dataclass.
    This dataclass should include all keys that are intended to be configurable.
    In addition, the subclass must implement the
    :meth:._forward_value_estimator_keys() method. This function is crucial for
    forwarding any altered tensordict keys to the underlying value_estimator.

    Examples:
        >>> class MyLoss(LossModule):
        >>>     @dataclass
        >>>     class _AcceptedKeys:
        >>>         action = "action"
        >>>
        >>>     def _forward_value_estimator_keys(self, **kwargs) -> None:
        >>>         pass
        >>>
        >>> loss = MyLoss()
        >>> loss.set_keys(action="action2")
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.
        """

        pass

    default_value_estimator: ValueEstimators = None
    SEP = "_sep_"
    TARGET_NET_WARNING = (
        "No target network updater has been associated "
        "with this loss module, but target parameters have been found. "
        "While this is supported, it is expected that the target network "
        "updates will be manually performed. You can deactivate this warning "
        "by turning the RL_WARNINGS env variable to False."
    )

    @property
    def tensor_keys(self) -> _AcceptedKeys:
        return self._tensor_keys

    def __new__(cls, *args, **kwargs):
        cls.forward = set_exploration_type(ExplorationType.MODE)(cls.forward)
        self = super().__new__(cls)
        return self

    def __init__(self):
        super().__init__()
        self._cache = {}
        self._param_maps = {}
        self._value_estimator = None
        self._has_update_associated = {}
        self.value_type = self.default_value_estimator
        self._tensor_keys = self._AcceptedKeys()
        self.register_forward_pre_hook(_updater_check_forward_prehook)
        # self.register_forward_pre_hook(_parameters_to_tensordict)

    def _set_deprecated_ctor_keys(self, **kwargs) -> None:
        """Helper function to set a tensordict key from a constructor and raise a warning simultaneously."""
        for key, value in kwargs.items():
            if value is not None:
                warnings.warn(
                    f"Setting '{key}' via the constructor is deprecated, use .set_keys(<key>='some_key') instead.",
                    category=DeprecationWarning,
                )
                self.set_keys(**{key: value})

    def set_keys(self, **kwargs) -> None:
        """Set tensordict key names.

        Examples:
            >>> from torchrl.objectives import DQNLoss
            >>> # initialize the DQN loss
            >>> actor = torch.nn.Linear(3, 4)
            >>> dqn_loss = DQNLoss(actor, action_space="one-hot")
            >>> dqn_loss.set_keys(priority_key="td_error", action_value_key="action_value")
        """
        for key, value in kwargs.items():
            if key not in self._AcceptedKeys.__dict__:
                raise ValueError(f"{key} it not an accepted tensordict key")
            if value is not None:
                setattr(self.tensor_keys, key, value)
            else:
                setattr(self.tensor_keys, key, self.default_keys.key)

        try:
            self._forward_value_estimator_keys(**kwargs)
        except AttributeError as err:
            raise AttributeError(
                "To utilize `.set_keys(...)` for tensordict key configuration, the subclassed loss module "
                "must define an _AcceptedKeys dataclass containing all keys intended for configuration. "
                "Moreover, the subclass needs to implement `._forward_value_estimator_keys()` method to "
                "facilitate forwarding of any modified tensordict keys to the underlying value_estimator."
            ) from err

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
        funs_to_decorate=None,
    ) -> None:
        """Converts a module to functional to be used in the loss.

        Args:
            module (TensorDictModule or compatible): a stateful tensordict module.
                This module will be made functional, yet still stateful, meaning
                that it will be callable with the following alternative signatures:

                >>> module(tensordict)
                >>> module(tensordict, params=params)

                ``params`` is a :class:`tensordict.TensorDict` instance with parameters
                stuctured as the output of :func:`tensordict.nn.make_functional`
                is.
            module_name (str): name where the module will be found.
                The parameters of the module will be found under ``loss_module.<module_name>_params``
                whereas the module will be found under ``loss_module.<module_name>``.
            expand_dim (int, optional): if provided, the parameters of the module
                will be expanded ``N`` times, where ``N = expand_dim`` along the
                first dimension. This option is to be used whenever a target
                network with more than one configuration is to be used.

                .. note::
                  If a ``compare_against`` list of values is provided, the
                  resulting parameters will simply be a detached expansion
                  of the original parameters. If ``compare_against`` is not
                  provided, the value of the parameters will be resampled uniformly
                  between the minimum and maximum value of the parameter content.

             create_target_params (bool, optional): if ``True``, a detached
                copy of the parameter will be available to feed a target network
                under the name ``loss_module.<module_name>_target_params``.
                If ``False`` (default), this attribute will still be available
                but it will be a detached instance of the parameters, not a copy.
                In other words, any modification of the parameter value
                will directly be reflected in the target parameters.
            compare_against (iterable of parameters, optional): if provided,
                this list of parameters will be used as a comparison set for
                the parameters of the module. If the parameters are expanded
                (``expand_dim > 0``), the resulting parameters for the module
                will be a simple expansion of the original parameter. Otherwise,
                the resulting parameters will be a detached version of the
                original parameters. If ``None``, the resulting parameters
                will carry gradients as expected.
            funs_to_decorate (list of str, optional): if provided, the list of
                methods of ``module`` to make functional, ie the list of
                methods that will accept the ``params`` keyword argument.

        """
        if funs_to_decorate is None:
            funs_to_decorate = ["forward"]
        # To make it robust to device casting, we must register list of
        # tensors as lazy calls to `getattr(self, name_of_tensor)`.
        # Otherwise, casting the module to a device will keep old references
        # to uncast tensors
        sep = self.SEP
        params = make_functional(module, funs_to_decorate=funs_to_decorate)
        # buffer_names = next(itertools.islice(zip(*module.named_buffers()), 1))
        buffer_names = []
        for key, value in params.items(True, True):
            # we just consider all that is not param as a buffer, but if the module has been made
            # functional and the params have been replaced this may break
            if not isinstance(value, nn.Parameter):
                key = sep.join(key) if not isinstance(key, str) else key
                buffer_names.append(key)
        functional_module = deepcopy(module)
        repopulate_module(module, params)

        params_and_buffers = params
        # we transform the buffers in params to make sure they follow the device
        # as tensor = nn.Parameter(tensor) keeps its identity when moved to another device

        # separate params and buffers
        params_and_buffers = TensorDictParams(params_and_buffers, no_convert=True)
        # sanity check
        for key in params_and_buffers.keys(True):
            if sep in key:
                raise KeyError(
                    f"The key {key} contains the '_sep_' pattern which is prohibited. Consider renaming the parameter / buffer."
                )
        params_and_buffers_flat = params_and_buffers.flatten_keys(sep)
        buffers = params_and_buffers_flat.select(*buffer_names)
        params = params_and_buffers_flat.exclude(*buffer_names)
        if compare_against is not None:
            compare_against = set(compare_against)
        else:
            compare_against = set()
        if expand_dim:
            # Expands the dims of params and buffers.
            # If the param already exist in the module, we return a simple expansion of the
            # original one. Otherwise, we expand and resample it.
            # For buffers, a cloned expansion (or equivalently a repeat) is returned.

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

            params = params.apply(
                _compare_and_expand, batch_size=[expand_dim, *params.shape]
            )

            buffers = buffers.apply(
                lambda buffer: buffer.expand(expand_dim, *buffer.shape).clone(),
                batch_size=[expand_dim, *buffers.shape],
            )

            params_and_buffers.update(params.unflatten_keys(sep))
            params_and_buffers.update(buffers.unflatten_keys(sep))
            params_and_buffers.batch_size = params.batch_size

            # self.params_to_map = params_to_map

        param_name = module_name + "_params"

        prev_set_params = set(self.parameters())

        # register parameters and buffers
        for key, parameter in list(params_and_buffers.items(True, True)):
            if parameter not in prev_set_params:
                pass
            elif compare_against is not None and parameter in compare_against:
                params_and_buffers.set(key, parameter.data)

        setattr(self, param_name, params_and_buffers)

        # set the functional module
        setattr(self, module_name, functional_module)

        name_params_target = "target_" + module_name
        if create_target_params:
            # if create_target_params:
            # we create a TensorDictParams to keep the target params as Buffer instances
            target_params = TensorDictParams(
                params_and_buffers.apply(
                    _make_target_param(clone=create_target_params)
                ),
                no_convert=True,
            )
            setattr(self, name_params_target + "_params", target_params)
        self._has_update_associated[module_name] = not create_target_params

    def __getattr__(self, item):
        if item.startswith("target_") and item.endswith("_params"):
            params = self._modules.get(item, None)
            if params is None:
                # no target param, take detached data
                params = getattr(self, item[7:])
                params = params.data
            elif not self._has_update_associated[item[7:-7]] and RL_WARNINGS:
                # no updater associated
                warnings.warn(
                    self.TARGET_NET_WARNING,
                    category=UserWarning,
                )
            return params
        return super().__getattr__(item)

    def _apply(self, fn):
        # any call to apply erases the cache: the reason is that detached
        # params will fail to be cast so we need to get the cache back
        self._erase_cache()
        return super()._apply(fn)

    def _erase_cache(self):
        for key in list(self.__dict__):
            if key.startswith("_cache"):
                del self.__dict__[key]

    def _networks(self) -> Iterator[nn.Module]:
        for item in self.__dir__():
            if isinstance(item, nn.Module):
                yield item

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

    @property
    def value_estimator(self) -> ValueEstimatorBase:
        """The value function blends in the reward and value estimate(s) from upcoming state(s)/state-action pair(s) into a target value estimate for the value network."""
        out = self._value_estimator
        if out is None:
            self._default_value_estimator()
            return self._value_estimator
        return out

    @value_estimator.setter
    def value_estimator(self, value):
        self._value_estimator = value

    def _default_value_estimator(self):
        """A value-function constructor when none is provided.

        No kwarg should be present as default parameters should be retrieved
        from :obj:`torchrl.objectives.utils.DEFAULT_VALUE_FUN_PARAMS`.

        """
        self.make_value_estimator(self.default_value_estimator)

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        """Value-function constructor.

        If the non-default value function is wanted, it must be built using
        this method.

        Args:
            value_type (ValueEstimators): A :class:`~torchrl.objectives.utils.ValueEstimators`
                enum type indicating the value function to use. If none is provided,
                the default stored in the ``default_value_estimator``
                attribute will be used. The resulting value estimator class
                will be registered in ``self.value_type``, allowing
                future refinements.
            **hyperparams: hyperparameters to use for the value function.
                If not provided, the value indicated by
                :func:`~torchrl.objectives.utils.default_value_kwargs` will be
                used.

        Examples:
            >>> from torchrl.objectives import DQNLoss
            >>> # initialize the DQN loss
            >>> actor = torch.nn.Linear(3, 4)
            >>> dqn_loss = DQNLoss(actor, action_space="one-hot")
            >>> # updating the parameters of the default value estimator
            >>> dqn_loss.make_value_estimator(gamma=0.9)
            >>> dqn_loss.make_value_estimator(
            ...     ValueEstimators.TD1,
            ...     gamma=0.9)
            >>> # if we want to change the gamma value
            >>> dqn_loss.make_value_estimator(dqn_loss.value_type, gamma=0.9)

        """
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        if value_type == ValueEstimators.TD1:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == ValueEstimators.TD0:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == ValueEstimators.TDLambda:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        # def _apply(self, fn, recurse=True):
        #     """Modifies torch.nn.Module._apply to work with Buffer class."""
        #     if recurse:
        #         for module in self.children():
        #             module._apply(fn)
        #
        #     def compute_should_use_set_data(tensor, tensor_applied):
        #         if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
        #             # If the new tensor has compatible tensor type as the existing tensor,
        #             # the current behavior is to change the tensor in-place using `.data =`,
        #             # and the future behavior is to overwrite the existing tensor. However,
        #             # changing the current behavior is a BC-breaking change, and we want it
        #             # to happen in future releases. So for now we introduce the
        #             # `torch.__future__.get_overwrite_module_params_on_conversion()`
        #             # global flag to let the user control whether they want the future
        #             # behavior of overwriting the existing tensor or not.
        #             return not torch.__future__.get_overwrite_module_params_on_conversion()
        #         else:
        #             return False
        #
        #     for key, param in self._parameters.items():
        #         if param is None:
        #             continue
        #         # Tensors stored in modules are graph leaves, and we don't want to
        #         # track autograd history of `param_applied`, so we have to use
        #         # `with torch.no_grad():`
        #         with torch.no_grad():
        #             param_applied = fn(param)
        #         should_use_set_data = compute_should_use_set_data(param, param_applied)
        #         if should_use_set_data:
        #             param.data = param_applied
        #             out_param = param
        #         else:
        #             assert isinstance(param, Parameter)
        #             assert param.is_leaf
        #             out_param = Parameter(param_applied, param.requires_grad)
        #             self._parameters[key] = out_param
        #
        #         if param.grad is not None:
        #             with torch.no_grad():
        #                 grad_applied = fn(param.grad)
        #             should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
        #             if should_use_set_data:
        #                 assert out_param.grad is not None
        #                 out_param.grad.data = grad_applied
        #             else:
        #                 assert param.grad.is_leaf
        #                 out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)
        #
        #     for key, buffer in self._buffers.items():
        #         if buffer is None:
        #             continue
        #         # Tensors stored in modules are graph leaves, and we don't want to
        #         # track autograd history of `buffer_applied`, so we have to use
        #         # `with torch.no_grad():`
        #         with torch.no_grad():
        #             buffer_applied = fn(buffer)
        #         should_use_set_data = compute_should_use_set_data(buffer, buffer_applied)
        #         if should_use_set_data:
        #             buffer.data = buffer_applied
        #             out_buffer = buffer
        #         else:
        #             assert isinstance(buffer, Buffer)
        #             assert buffer.is_leaf
        #             out_buffer = Buffer(buffer_applied, buffer.requires_grad)
        #             self._buffers[key] = out_buffer
        #
        #         if buffer.grad is not None:
        #             with torch.no_grad():
        #                 grad_applied = fn(buffer.grad)
        #             should_use_set_data = compute_should_use_set_data(buffer.grad, grad_applied)
        #             if should_use_set_data:
        #                 assert out_buffer.grad is not None
        #                 out_buffer.grad.data = grad_applied
        #             else:
        #                 assert buffer.grad.is_leaf
        #                 out_buffer.grad = grad_applied.requires_grad_(buffer.grad.requires_grad)

        return self


class _make_target_param:
    def __init__(self, clone):
        self.clone = clone

    def __call__(self, x):
        if isinstance(x, nn.Parameter):
            return nn.Parameter(
                x.data.clone() if self.clone else x.data, requires_grad=False
            )
        return x.data.clone() if self.clone else x.data
