# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from omegaconf import MISSING

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class ActivationConfig(ConfigBase):
    """A class to configure an activation function.

    Defaults to :class:`torch.nn.Tanh`.

    .. seealso:: :class:`torch.nn.Tanh`
    """

    _target_: str = "torch.nn.Tanh"
    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for activation configurations."""


@dataclass
class LayerConfig(ConfigBase):
    """A class to configure a layer.

    Defaults to :class:`torch.nn.Linear`.

    .. seealso:: :class:`torch.nn.Linear`
    """

    _target_: str = "torch.nn.Linear"
    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for layer configurations."""


@dataclass
class NetworkConfig(ConfigBase):
    """Parent class to configure a network."""

    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for network configurations."""


@dataclass
class MLPConfig(NetworkConfig):
    """A class to configure a multi-layer perceptron.

    Example:
        >>> cfg = MLPConfig(in_features=10, out_features=5, depth=2, num_cells=32)
        >>> net = instantiate(cfg)
        >>> y = net(torch.randn(1, 10))
        >>> assert y.shape == (1, 5)

    .. seealso:: :class:`torchrl.modules.MLP`
    """

    in_features: int | None = None
    out_features: Any = None
    depth: int | None = None
    num_cells: Any = None
    activation_class: ActivationConfig = field(
        default_factory=partial(
            ActivationConfig, _target_="torch.nn.Tanh", _partial_=True
        )
    )
    activation_kwargs: Any = None
    norm_class: Any = None
    norm_kwargs: Any = None
    dropout: float | None = None
    bias_last_layer: bool = True
    single_bias_last_layer: bool = False
    layer_class: LayerConfig = field(
        default_factory=partial(LayerConfig, _target_="torch.nn.Linear", _partial_=True)
    )
    layer_kwargs: dict | None = None
    activate_last_layer: bool = False
    device: Any = None
    _target_: str = "torchrl.modules.MLP"

    def __post_init__(self):
        if isinstance(self.activation_class, str):
            self.activation_class = ActivationConfig(
                _target_=self.activation_class, _partial_=True
            )
        if isinstance(self.layer_class, str):
            self.layer_class = LayerConfig(_target_=self.layer_class, _partial_=True)


@dataclass
class NormConfig(ConfigBase):
    """A class to configure a normalization layer.

    Defaults to :class:`torch.nn.BatchNorm1d`.

    .. seealso:: :class:`torch.nn.BatchNorm1d`
    """

    _target_: str = "torch.nn.BatchNorm1d"
    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for normalization configurations."""


@dataclass
class AggregatorConfig(ConfigBase):
    """A class to configure an aggregator layer.

    Defaults to :class:`torchrl.modules.models.utils.SquashDims`.

    .. seealso:: :class:`torchrl.modules.models.utils.SquashDims`
    """

    _target_: str = "torchrl.modules.models.utils.SquashDims"
    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for aggregator configurations."""


@dataclass
class ConvNetConfig(NetworkConfig):
    """A class to configure a convolutional network.

    Defaults to :class:`torchrl.modules.ConvNet`.

    Example:
        >>> cfg = ConvNetConfig(in_features=3, depth=2, num_cells=[32, 64], kernel_sizes=[3, 5], strides=[1, 2], paddings=[1, 2])
        >>> net = instantiate(cfg)
        >>> y = net(torch.randn(1, 3, 32, 32))
        >>> assert y.shape == (1, 64)

    .. seealso:: :class:`torchrl.modules.ConvNet`
    """

    in_features: int | None = None
    depth: int | None = None
    num_cells: Any = None
    kernel_sizes: Any = 3
    strides: Any = 1
    paddings: Any = 0
    activation_class: ActivationConfig = field(
        default_factory=partial(
            ActivationConfig, _target_="torch.nn.ELU", _partial_=True
        )
    )
    activation_kwargs: Any = None
    norm_class: NormConfig | None = None
    norm_kwargs: Any = None
    bias_last_layer: bool = True
    aggregator_class: AggregatorConfig = field(
        default_factory=partial(
            AggregatorConfig,
            _target_="torchrl.modules.models.utils.SquashDims",
            _partial_=True,
        )
    )
    aggregator_kwargs: dict | None = None
    squeeze_output: bool = False
    device: Any = None
    _target_: str = "torchrl.modules.ConvNet"

    def __post_init__(self):
        if self.activation_class is None and isinstance(self.activation_class, str):
            self.activation_class = ActivationConfig(
                _target_=self.activation_class, _partial_=True
            )
        if self.norm_class is None and isinstance(self.norm_class, str):
            self.norm_class = NormConfig(_target_=self.norm_class, _partial_=True)
        if self.aggregator_class is None and isinstance(self.aggregator_class, str):
            self.aggregator_class = AggregatorConfig(
                _target_=self.aggregator_class, _partial_=True
            )


@dataclass
class ModelConfig(ConfigBase):
    """Parent class to configure a model.

    A model can be made of several networks. It is always a :class:`~tensordict.nn.TensorDictModuleBase` instance.

    .. seealso:: :class:`TanhNormalModelConfig`, :class:`ValueModelConfig`
    """

    _partial_: bool = False
    in_keys: Any = None
    out_keys: Any = None
    shared: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for model configurations."""


@dataclass
class TensorDictModuleConfig(ModelConfig):
    """A class to configure a TensorDictModule.

    Example:
        >>> cfg = TensorDictModuleConfig(module=MLPConfig(in_features=10, out_features=10, depth=2, num_cells=32), in_keys=["observation"], out_keys=["action"])
        >>> module = instantiate(cfg)
        >>> assert isinstance(module, TensorDictModule)
        >>> assert module(observation=torch.randn(10, 10)).shape == (10, 10)

    .. seealso:: :class:`tensordict.nn.TensorDictModule`
    """

    module: MLPConfig = MISSING
    _target_: str = (
        "torchrl.trainers.algorithms.configs.modules._make_tensordict_module"
    )
    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for TensorDict module configurations."""
        return super().__post_init__()


@dataclass
class TensorDictSequentialConfig(ModelConfig):
    """A class to configure a TensorDictSequential.

    Example:
        >>> cfg = TensorDictSequentialConfig(
        ...     modules=[
        ...         TensorDictModuleConfig(module=MLPConfig(in_features=10, out_features=10, depth=2, num_cells=32), in_keys=["observation"], out_keys=["hidden"]),
        ...         TensorDictModuleConfig(module=MLPConfig(in_features=10, out_features=5, depth=2, num_cells=32), in_keys=["hidden"], out_keys=["action"])
        ...     ]
        ... )
        >>> seq = instantiate(cfg)
        >>> assert isinstance(seq, TensorDictSequential)

    .. seealso:: :class:`tensordict.nn.TensorDictSequential`
    """

    modules: Any | None = None
    partial_tolerant: bool = False
    selected_out_keys: Any | None = None
    inplace: bool | str | None = None
    _target_: str = (
        "torchrl.trainers.algorithms.configs.modules._make_tensordict_sequential"
    )
    _partial_: bool = False

    def __post_init__(self) -> None:
        return super().__post_init__()


@dataclass
class TanhNormalModelConfig(ModelConfig):
    """A class to configure a TanhNormal model.

    Example:
        >>> cfg = TanhNormalModelConfig(network=MLPConfig(in_features=10, out_features=5, depth=2, num_cells=32))
        >>> net = instantiate(cfg)
        >>> y = net(torch.randn(1, 10))
        >>> assert y.shape == (1, 5)

    .. seealso:: :class:`torchrl.modules.TanhNormal`
    """

    network: MLPConfig = MISSING
    eval_mode: bool = False

    extract_normal_params: bool = True
    scale_mapping: str = "biased_softplus_1.0"
    scale_lb: float = 1e-4

    param_keys: Any = None

    exploration_type: Any = "RANDOM"

    return_log_prob: bool = False

    _target_: str = (
        "torchrl.trainers.algorithms.configs.modules._make_tanh_normal_model"
    )

    def __post_init__(self):
        """Post-initialization hook for TanhNormal model configurations."""
        super().__post_init__()
        if self.in_keys is None:
            self.in_keys = ["observation"]
        if self.param_keys is None:
            self.param_keys = ["loc", "scale"]
        if self.out_keys is None:
            self.out_keys = ["action"]


@dataclass
class ValueModelConfig(ModelConfig):
    """A class to configure a Value model.

    Example:
        >>> cfg = ValueModelConfig(network=MLPConfig(in_features=10, out_features=5, depth=2, num_cells=32))
        >>> net = instantiate(cfg)
        >>> y = net(torch.randn(1, 10))
        >>> assert y.shape == (1, 5)

    .. seealso:: :class:`torchrl.modules.ValueOperator`
    """

    _target_: str = "torchrl.trainers.algorithms.configs.modules._make_value_model"
    network: NetworkConfig = MISSING

    def __post_init__(self) -> None:
        """Post-initialization hook for value model configurations."""
        super().__post_init__()


@dataclass
class TanhModuleConfig(ModelConfig):
    """A class to configure a TanhModule.

    Example:
        >>> cfg = TanhModuleConfig(in_keys=["action"], out_keys=["action"], low=-1.0, high=1.0)
        >>> module = instantiate(cfg)
        >>> assert isinstance(module, TanhModule)

    .. seealso:: :class:`torchrl.modules.TanhModule`
    """

    spec: Any = None
    low: Any = None
    high: Any = None
    clamp: bool = False
    _target_: str = "torchrl.trainers.algorithms.configs.modules._make_tanh_module"

    def __post_init__(self) -> None:
        """Post-initialization hook for TanhModule configurations."""
        super().__post_init__()


def _make_tensordict_module(*args, **kwargs):
    """Helper function to create a TensorDictModule."""
    from hydra.utils import instantiate
    from tensordict.nn import TensorDictModule

    module = kwargs.pop("module")
    shared = kwargs.pop("shared", False)

    # Instantiate the module if it's a config
    if hasattr(module, "_target_"):
        module = instantiate(module)
    elif callable(module) and hasattr(module, "func"):  # partial function
        module = module()

    # Create the TensorDictModule
    tensordict_module = TensorDictModule(module, **kwargs)

    # Apply share_memory if needed
    if shared:
        tensordict_module = tensordict_module.share_memory()

    return tensordict_module


def _make_tensordict_sequential(*args, **kwargs):
    """Helper function to create a TensorDictSequential."""
    from hydra.utils import instantiate
    from omegaconf import DictConfig, ListConfig
    from tensordict.nn import TensorDictSequential

    modules = kwargs.pop("modules")
    shared = kwargs.pop("shared", False)
    partial_tolerant = kwargs.pop("partial_tolerant", False)
    selected_out_keys = kwargs.pop("selected_out_keys", None)
    inplace = kwargs.pop("inplace", None)

    def _instantiate_module(module):
        if hasattr(module, "_target_"):
            return instantiate(module)
        elif callable(module) and hasattr(module, "func"):
            return module()
        else:
            return module

    if isinstance(modules, (dict, DictConfig)):
        instantiated_modules = {
            key: _instantiate_module(module) for key, module in modules.items()
        }
    elif isinstance(modules, (list, ListConfig)):
        instantiated_modules = [_instantiate_module(module) for module in modules]
    else:
        raise ValueError(
            f"modules must be a dict or list, got {type(modules).__name__}"
        )

    tensordict_sequential = TensorDictSequential(
        instantiated_modules,
        partial_tolerant=partial_tolerant,
        selected_out_keys=selected_out_keys,
        inplace=inplace,
    )

    if shared:
        tensordict_sequential = tensordict_sequential.share_memory()

    return tensordict_sequential


def _make_tanh_normal_model(*args, **kwargs):
    """Helper function to create a TanhNormal model with ProbabilisticTensorDictSequential."""
    from hydra.utils import instantiate
    from tensordict.nn import (
        ProbabilisticTensorDictModule,
        ProbabilisticTensorDictSequential,
        TensorDictModule,
    )
    from torchrl.modules import NormalParamExtractor, TanhNormal

    # Extract parameters
    network = kwargs.pop("network")
    in_keys = list(kwargs.pop("in_keys", ["observation"]))
    param_keys = list(kwargs.pop("param_keys", ["loc", "scale"]))
    out_keys = list(kwargs.pop("out_keys", ["action"]))
    extract_normal_params = kwargs.pop("extract_normal_params", True)
    scale_mapping = kwargs.pop("scale_mapping", "biased_softplus_1.0")
    scale_lb = kwargs.pop("scale_lb", 1e-4)
    return_log_prob = kwargs.pop("return_log_prob", False)
    eval_mode = kwargs.pop("eval_mode", False)
    exploration_type = kwargs.pop("exploration_type", "RANDOM")
    shared = kwargs.pop("shared", False)

    # Now instantiate the network
    if hasattr(network, "_target_"):
        network = instantiate(network)
    elif callable(network) and hasattr(network, "func"):  # partial function
        network = network()

    # Create the sequential
    if extract_normal_params:
        # Add NormalParamExtractor to split the output
        network = torch.nn.Sequential(
            network,
            NormalParamExtractor(scale_mapping=scale_mapping, scale_lb=scale_lb),
        )

    module = TensorDictModule(network, in_keys=in_keys, out_keys=param_keys)
    if shared:
        module = module.share_memory()

    # Create ProbabilisticTensorDictModule
    prob_module = ProbabilisticTensorDictModule(
        in_keys=param_keys,
        out_keys=out_keys,
        distribution_class=TanhNormal,
        return_log_prob=return_log_prob,
        default_interaction_type=exploration_type,
        **kwargs,
    )

    result = ProbabilisticTensorDictSequential(module, prob_module)
    if eval_mode:
        result.eval()
    return result


def _make_value_model(*args, **kwargs):
    """Helper function to create a ValueOperator with the given network."""
    from hydra.utils import instantiate

    from torchrl.modules import ValueOperator

    network = kwargs.pop("network")
    shared = kwargs.pop("shared", False)

    # Instantiate the network if it's a config
    if hasattr(network, "_target_"):
        network = instantiate(network)
    elif callable(network) and hasattr(network, "func"):  # partial function
        network = network()

    # Create the ValueOperator
    value_operator = ValueOperator(network, **kwargs)

    # Apply share_memory if needed
    if shared:
        value_operator = value_operator.share_memory()

    return value_operator


def _make_tanh_module(*args, **kwargs):
    """Helper function to create a TanhModule."""
    from omegaconf import ListConfig

    from torchrl.modules import TanhModule

    kwargs.pop("shared", False)

    if "in_keys" in kwargs and isinstance(kwargs["in_keys"], ListConfig):
        kwargs["in_keys"] = list(kwargs["in_keys"])
    if "out_keys" in kwargs and isinstance(kwargs["out_keys"], ListConfig):
        kwargs["out_keys"] = list(kwargs["out_keys"])

    return TanhModule(**kwargs)
