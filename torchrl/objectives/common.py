# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import functools
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from tensordict import is_tensor_collection, TensorDict, TensorDictBase

from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictParams
from tensordict.utils import Buffer
from torch import nn
from torch.nn import Parameter
from torchrl._utils import RL_WARNINGS
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.objectives.utils import RANDOM_MODULE_LIST, ValueEstimators
from torchrl.objectives.value import ValueEstimatorBase

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling


def _updater_check_forward_prehook(module, *args, **kwargs):
    if (
        not all(module._has_update_associated.values())
        and RL_WARNINGS
        and not is_dynamo_compiling()
    ):
        warnings.warn(
            module.TARGET_NET_WARNING,
            category=UserWarning,
        )


def _forward_wrapper(func):
    @functools.wraps(func)
    def new_forward(self, *args, **kwargs):
        with set_exploration_type(self.deterministic_sampling_mode):
            return func(self, *args, **kwargs)

    return new_forward


class _LossMeta(abc.ABCMeta):
    def __init__(cls, name, bases, attr_dict):
        super().__init__(name, bases, attr_dict)
        cls.forward = _forward_wrapper(cls.forward)


class LossModule(TensorDictModuleBase, metaclass=_LossMeta):
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
    gh :class:`torchrl.envs.ExplorationType.MEAN`

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

    .. note:: When a policy that is wrapped or augmented with an exploration module is passed
        to the loss, we want to deactivate the exploration through ``set_exploration_type(<exploration>)`` where
        ``<exploration>`` is either ``ExplorationType.MEAN``, ``ExplorationType.MODE`` or
        ``ExplorationType.DETERMINISTIC``. The default value is ``DETERMINISTIC`` and it is set
        through the ``deterministic_sampling_mode`` loss attribute. If another
        exploration mode is required (or if ``DETERMINISTIC`` is not available), one can
        change the value of this attribute which will change the mode.

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.
        """

        pass

    _vmap_randomness = None
    default_value_estimator: ValueEstimators = None

    deterministic_sampling_mode: ExplorationType = ExplorationType.DETERMINISTIC

    SEP = "."
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

    @property
    def functional(self):
        """Whether the module is functional.

        Unless it has been specifically designed not to be functional, all losses are functional.
        """
        return True

    def get_stateful_net(self, network_name: str, copy: bool | None = None):
        """Returns a stateful version of the network.

        This can be used to initialize parameters.

        Such networks will often not be callable out-of-the-box and will require a `vmap` call
        to be executable.

        Args:
            network_name (str): the network name to gather.
            copy (bool, optional): if ``True``, a deepcopy of the network is made.
                Defaults to ``True``.

                .. note:: if the module is not functional, no copy is made.
        """
        net = getattr(self, network_name)
        if not self.functional:
            if copy is not None and copy:
                raise RuntimeError("Cannot copy module in non-functional mode.")
            return net
        copy = True if copy is None else copy
        if copy:
            net = deepcopy(net)
        params = getattr(self, network_name + "_params")
        params.to_module(net)
        return net

    def from_stateful_net(self, network_name: str, stateful_net: nn.Module):
        """Populates the parameters of a model given a stateful version of the network.

        See :meth:`~.get_stateful_net` for details on how to gather a stateful version of the network.

        Args:
            network_name (str): the network name to reset.
            stateful_net (nn.Module): the stateful network from which the params should be
                gathered.

        """
        if not self.functional:
            getattr(self, network_name).load_state_dict(stateful_net.state_dict())
            return
        params = TensorDict.from_module(stateful_net, as_module=True)
        keyset0 = set(params.keys(True, True))
        self_params = getattr(self, network_name + "_params")
        keyset1 = set(self_params.keys(True, True))
        if keyset0 != keyset1:
            raise RuntimeError(
                f"The keys of params and provided module differ: "
                f"{keyset1-keyset0} are in self.params and not in the module, "
                f"{keyset0-keyset1} are in the module but not in self.params."
            )
        self_params.data.update_(params.data)

    def _set_deprecated_ctor_keys(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if value is not None:
                raise RuntimeError(
                    f"Setting '{key}' via the constructor is deprecated, use .set_keys(<key>='some_key') instead.",
                )

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
            if key not in self._AcceptedKeys.__dataclass_fields__:
                raise ValueError(
                    f"{key} is not an accepted tensordict key. Accepted keys are: {self._AcceptedKeys.__dataclass_fields__}."
                )
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
        **kwargs,
    ) -> None:
        """Converts a module to functional to be used in the loss.

        Args:
            module (TensorDictModule or compatible): a stateful tensordict module.
                Parameters from this module will be isolated in the `<module_name>_params`
                attribute and a stateless version of the module will be registed
                under the `module_name` attribute.
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

        """
        for name in (
            module_name,
            module_name + "_params",
            "target_" + module_name + "_params",
        ):
            if name not in self.__class__.__annotations__.keys():
                warnings.warn(
                    f"The name {name} wasn't part of the annotations ({self.__class__.__annotations__.keys()}). Make sure it is present in the definition class."
                )

        if kwargs:
            raise TypeError(f"Unrecognised keyword arguments {list(kwargs.keys())}")
        # To make it robust to device casting, we must register list of
        # tensors as lazy calls to `getattr(self, name_of_tensor)`.
        # Otherwise, casting the module to a device will keep old references
        # to uncast tensors
        sep = self.SEP
        if isinstance(module, (list, tuple)):
            if len(module) != expand_dim:
                raise RuntimeError(
                    "The ``expand_dim`` value must match the length of the module list/tuple "
                    "if a single module isn't provided."
                )
            params = TensorDict.from_modules(
                *module, as_module=True, expand_identical=True
            )
        else:
            params = TensorDict.from_module(module, as_module=True)

            for key in params.keys(True):
                if sep in key:
                    raise KeyError(
                        f"The key {key} contains the '_sep_' pattern which is prohibited. Consider renaming the parameter / buffer."
                    )
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
                    if is_tensor_collection(param):
                        return param._apply_nest(
                            _compare_and_expand,
                            batch_size=[expand_dim, *param.shape],
                            filter_empty=False,
                            call_on_nested=True,
                        )
                    if not isinstance(param, nn.Parameter):
                        buffer = param.expand(expand_dim, *param.shape).clone()
                        return buffer
                    if param in compare_against:
                        expanded_param = param.data.expand(expand_dim, *param.shape)
                        # the expanded parameter must be sent to device when to()
                        # is called:
                        return expanded_param
                    else:
                        p_out = param.expand(expand_dim, *param.shape).clone()
                        p_out = nn.Parameter(
                            p_out.uniform_(
                                p_out.min().item(), p_out.max().item()
                            ).requires_grad_()
                        )
                        return p_out

                params = TensorDictParams(
                    params.apply(
                        _compare_and_expand,
                        batch_size=[expand_dim, *params.shape],
                        filter_empty=False,
                        call_on_nested=True,
                    ),
                    no_convert=True,
                )

        param_name = module_name + "_params"

        prev_set_params = set(self.parameters())

        # register parameters and buffers
        for key, parameter in list(params.items(True, True)):
            if parameter not in prev_set_params:
                pass
            elif compare_against is not None and parameter in compare_against:
                params.set(key, parameter.data)

        setattr(self, param_name, params)

        # Set the module in the __dict__ directly to avoid listing its params
        # A deepcopy with meta device could be used but that assumes that the model is copyable!
        self.__dict__[module_name] = module

        name_params_target = "target_" + module_name
        if create_target_params:
            # if create_target_params:
            # we create a TensorDictParams to keep the target params as Buffer instances
            target_params = TensorDictParams(
                params.apply(
                    _make_target_param(clone=create_target_params), filter_empty=False
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
            elif (
                not self._has_update_associated[item[7:-7]]
                and RL_WARNINGS
                and not is_dynamo_compiling()
            ):
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
                delattr(self, key)

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
        elif value_type == ValueEstimators.VTrace:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == ValueEstimators.TDLambda:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        return self

    @property
    def vmap_randomness(self):
        """Vmap random mode.

        The vmap randomness mode controls what :func:`~torch.vmap` should do when dealing with
        functions with a random outcome such as :func:`~torch.randn` and :func:`~torch.rand`.
        If `"error"`, any random function will raise an exception indicating that `vmap` does not
        know how to handle the random call.

        If `"different"`, every element of the batch along which vmap is being called will
        behave differently. If `"same"`, vmaps will copy the same result across all elements.

        ``vmap_randomness`` defaults to `"error"` if no random module is detected, and to `"different"` in
        other cases. By default, only a limited number of modules are listed as random, but the list can be extended
        using the :func:`~torchrl.objectives.common.add_random_module` function.

        This property supports setting its value.

        """
        if self._vmap_randomness is None:
            main_modules = list(self.__dict__.values()) + list(self.children())
            modules = (
                module
                for main_module in main_modules
                if isinstance(main_module, nn.Module)
                for module in main_module.modules()
            )
            for val in modules:
                if isinstance(val, RANDOM_MODULE_LIST):
                    self._vmap_randomness = "different"
                    break
            else:
                self._vmap_randomness = "error"

        return self._vmap_randomness

    def set_vmap_randomness(self, value):
        if value not in ("error", "same", "different"):
            raise ValueError(
                "Wrong vmap randomness, should be one of 'error', 'same' or 'different'."
            )
        self._vmap_randomness = value
        self._make_vmap()

    @staticmethod
    def _make_meta_params(param):
        is_param = isinstance(param, nn.Parameter)

        pd = param.detach().to("meta")

        if is_param:
            pd = nn.Parameter(pd, requires_grad=False)
        return pd

    def _make_vmap(self):
        """Caches the the vmap callers to reduce the overhead at runtime."""
        raise NotImplementedError(
            f"_make_vmap has been called but is not implemented for loss of type {type(self).__name__}."
        )


class _make_target_param:
    def __init__(self, clone):
        self.clone = clone

    def __call__(self, x):
        x = x.data.clone() if self.clone else x.data
        if isinstance(x, nn.Parameter):
            return Buffer(x)
        return x


def add_ramdom_module(module):
    """Adds a random module to the list of modules that will be detected by :meth:`~torchrl.objectives.LossModule.vmap_randomness` as random."""
    global RANDOM_MODULE_LIST
    RANDOM_MODULE_LIST = RANDOM_MODULE_LIST + (module,)
