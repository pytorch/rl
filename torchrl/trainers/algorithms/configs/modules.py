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
    _target_: str = "tensordict.nn.TensorDictModule"
    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for TensorDict module configurations."""
        super().__post_init__()


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
    return_log_prob = kwargs.pop("return_log_prob", False)
    eval_mode = kwargs.pop("eval_mode", False)
    exploration_type = kwargs.pop("exploration_type", "RANDOM")

    # Now instantiate the network
    if hasattr(network, "_target_"):
        network = instantiate(network)
    elif callable(network) and hasattr(network, "func"):  # partial function
        network = network()

    # Create the sequential
    if extract_normal_params:
        # Add NormalParamExtractor to split the output
        network = torch.nn.Sequential(network, NormalParamExtractor())

    module = TensorDictModule(network, in_keys=in_keys, out_keys=param_keys)

    # Create ProbabilisticTensorDictModule
    prob_module = ProbabilisticTensorDictModule(
        in_keys=param_keys,
        out_keys=out_keys,
        distribution_class=TanhNormal,
        return_log_prob=return_log_prob,
        default_interaction_type=exploration_type,
        **kwargs
    )

    result = ProbabilisticTensorDictSequential(module, prob_module)
    if eval_mode:
        result.eval()
    return result


def _make_value_model(*args, **kwargs):
    """Helper function to create a ValueOperator with the given network."""
    from torchrl.modules import ValueOperator

    network = kwargs.pop("network")
    return ValueOperator(network, **kwargs)
