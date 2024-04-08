# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np

import torch

from tensordict import TensorDict
from torch import nn
from torchrl.data.utils import DEVICE_TYPING

from torchrl.modules.models import ConvNet, MLP
from torchrl.modules.models.utils import _reset_parameters_recursive


class MultiAgentNetBase(nn.Module):
    """A base class for multi-agent networks."""

    _empty_net: nn.Module

    def __init__(
        self,
        *,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        agent_dim: int,
        vmap_randomness: str = "different",
        **kwargs,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.share_params = share_params
        self.centralised = centralised
        self.agent_dim = agent_dim
        self._vmap_randomness = vmap_randomness

        agent_networks = [
            self._build_single_net(**kwargs)
            for _ in range(self.n_agents if not self.share_params else 1)
        ]
        initialized = True
        for p in agent_networks[0].parameters():
            if isinstance(p, torch.nn.UninitializedParameter):
                initialized = False
                break
        self.initialized = initialized
        self._make_params(agent_networks)
        kwargs["device"] = "meta"
        self.__dict__["_empty_net"] = self._build_single_net(**kwargs)

    @property
    def vmap_randomness(self):
        if self.initialized:
            return self._vmap_randomness
        # The class _BatchedUninitializedParameter and buffer are not batched
        # by vmap so using "different" will raise an exception because vmap can't find
        # the batch dimension. This is ok though since we won't have the same config
        # for every element (as one might expect from "same").
        return "same"

    def _make_params(self, agent_networks):
        if self.share_params:
            self.params = TensorDict.from_module(agent_networks[0], as_module=True)
        else:
            self.params = TensorDict.from_modules(*agent_networks, as_module=True)

    @abc.abstractmethod
    def _build_single_net(self, *, device, **kwargs):
        ...

    @abc.abstractmethod
    def _pre_forward_check(self, inputs):
        ...

    @staticmethod
    def vmap_func_module(module, *args, **kwargs):
        def exec_module(params, *input):
            with params.to_module(module):
                return module(*input)

        return torch.vmap(exec_module, *args, **kwargs)

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = torch.cat([*inputs], -1)
        else:
            inputs = inputs[0]

        inputs = self._pre_forward_check(inputs)
        # If parameters are not shared, each agent has its own network
        if not self.share_params:
            if self.centralised:
                output = self.vmap_func_module(
                    self._empty_net, (0, None), (-2,), randomness=self.vmap_randomness
                )(self.params, inputs)
            else:
                output = self.vmap_func_module(
                    self._empty_net,
                    (0, self.agent_dim),
                    (-2,),
                    randomness=self.vmap_randomness,
                )(self.params, inputs)

        # If parameters are shared, agents use the same network
        else:
            with self.params.to_module(self._empty_net):
                output = self._empty_net(inputs)

            if self.centralised:
                # If the parameters are shared, and it is centralised, all agents will have the same output
                # We expand it to maintain the agent dimension, but values will be the same for all agents
                n_agent_outputs = output.shape[-1]
                output = output.view(*output.shape[:-1], n_agent_outputs)
                output = output.unsqueeze(-2)
                output = output.expand(
                    *output.shape[:-2], self.n_agents, n_agent_outputs
                )

        if output.shape[-2] != (self.n_agents):
            raise ValueError(
                f"Multi-agent network expected output with shape[-2]={self.n_agents}"
                f" but got {output.shape}"
            )

        return output

    def reset_parameters(self):
        """Resets the parameters of the model."""

        def vmap_reset_module(module, *args, **kwargs):
            def reset_module(params):
                with params.to_module(module):
                    _reset_parameters_recursive(module)
                    return params

            return torch.vmap(reset_module, *args, **kwargs)

        if not self.share_params:
            vmap_reset_module(self._empty_net, randomness="different")(self.params)
        else:
            with self.params.to_module(self._empty_net):
                _reset_parameters_recursive(self._empty_net)


class MultiAgentMLP(MultiAgentNetBase):
    """Mult-agent MLP.

    This is an MLP that can be used in multi-agent contexts.
    For example, as a policy or as a value function.
    See `examples/multiagent` for examples.

    It expects inputs with shape (*B, n_agents, n_agent_inputs)
    It returns outputs with shape (*B, n_agents, n_agent_outputs)

    If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies).
    Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).

    If `centralised` is True, each agent will use the inputs of all agents to compute its output
    (n_agent_inputs * n_agents will be the number of inputs for one agent).
    Otherwise, each agent will only use its data as input.

    Args:
        n_agent_inputs (int or None): number of inputs for each agent. If ``None``,
            the number of inputs is lazily instantiated during the first call.
        n_agent_outputs (int): number of outputs for each agent.
        n_agents (int): number of agents.
        centralised (bool): If `centralised` is True, each agent will use the inputs of all agents to compute its output
            (n_agent_inputs * n_agents will be the number of inputs for one agent).
            Otherwise, each agent will only use its data as input.
        share_params (bool): If `share_params` is True, the same MLP will be used to make the forward pass
            for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process
            its input (heterogeneous policies).
        device (str or toech.device, optional): device to create the module on.
        depth (int, optional): depth of the network. A depth of 0 will produce a single linear layer network with the
            desired input and output size. A length of 1 will create 2 linear layers etc. If no depth is indicated,
            the depth information should be contained in the num_cells argument (see below). If num_cells is an
            iterable and depth is indicated, both should match: len(num_cells) must be equal to depth.
            default: 3.
        num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers out_features will match the content of num_cells.
            default: 32.
        activation_class (Type[nn.Module]): activation class to be used.
            default: nn.Tanh.
        **kwargs: for :class:`torchrl.modules.models.MLP` can be passed to customize the MLPs.

    Examples:
        >>> from torchrl.modules import MultiAgentMLP
        >>> import torch
        >>> n_agents = 6
        >>> n_agent_inputs=3
        >>> n_agent_outputs=2
        >>> batch = 64
        >>> obs = torch.zeros(batch, n_agents, n_agent_inputs
        First let's instantiate a local network shared by all agents (e.g. a parameter-shared policy)
        >>> mlp = MultiAgentMLP(
        ...     n_agent_inputs=n_agent_inputs,
        ...     n_agent_outputs=n_agent_outputs,
        ...     n_agents=n_agents,
        ...     centralised=False,
        ...     share_params=True,
        ...     depth=2,
        ... )
        >>> print(mlp)
        MultiAgentMLP(
          (agent_networks): ModuleList(
            (0): MLP(
              (0): Linear(in_features=3, out_features=32, bias=True)
              (1): Tanh()
              (2): Linear(in_features=32, out_features=32, bias=True)
              (3): Tanh()
              (4): Linear(in_features=32, out_features=2, bias=True)
            )
          )
        )
        >>> assert mlp(obs).shape == (batch, n_agents, n_agent_outputs)
        Now let's instantiate a centralised network shared by all agents (e.g. a centalised value function)
        >>> mlp = MultiAgentMLP(
        ...     n_agent_inputs=n_agent_inputs,
        ...     n_agent_outputs=n_agent_outputs,
        ...     n_agents=n_agents,
        ...     centralised=True,
        ...     share_params=True,
        ...     depth=2,
        ... )
        >>> print(mlp)
        MultiAgentMLP(
          (agent_networks): ModuleList(
            (0): MLP(
              (0): Linear(in_features=18, out_features=32, bias=True)
              (1): Tanh()
              (2): Linear(in_features=32, out_features=32, bias=True)
              (3): Tanh()
              (4): Linear(in_features=32, out_features=2, bias=True)
            )
          )
        )
        We can see that the input to the first layer is n_agents * n_agent_inputs,
        this is because in the case the net acts as a centralised mlp (like a single huge agent)
        >>> assert mlp(obs).shape == (batch, n_agents, n_agent_outputs)
        Outputs will be identical for all agents.
        Now we can do both examples just shown but with an independent set of parameters for each agent
        Let's show the centralised=False case.
        >>> mlp = MultiAgentMLP(
        ...     n_agent_inputs=n_agent_inputs,
        ...     n_agent_outputs=n_agent_outputs,
        ...     n_agents=n_agents,
        ...     centralised=False,
        ...     share_params=False,
        ...     depth=2,
        ... )
        >>> print(mlp)
        MultiAgentMLP(
          (agent_networks): ModuleList(
            (0-5): 6 x MLP(
              (0): Linear(in_features=3, out_features=32, bias=True)
              (1): Tanh()
              (2): Linear(in_features=32, out_features=32, bias=True)
              (3): Tanh()
              (4): Linear(in_features=32, out_features=2, bias=True)
            )
          )
        )
        We can see that this is the same as in the first example, but now we have 6 MLPs, one per agent!
        >>> assert mlp(obs).shape == (batch, n_agents, n_agent_outputs)
    """

    def __init__(
        self,
        n_agent_inputs: int | None,
        n_agent_outputs: int,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: Optional[DEVICE_TYPING] = None,
        depth: Optional[int] = None,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        **kwargs,
    ):
        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.share_params = share_params
        self.centralised = centralised
        self.num_cells = num_cells
        self.activation_class = activation_class
        self.depth = depth

        super().__init__(
            n_agents=n_agents,
            centralised=centralised,
            share_params=share_params,
            device=device,
            agent_dim=-2,
            **kwargs,
        )

    def _pre_forward_check(self, inputs):
        if inputs.shape[-2] != self.n_agents:
            raise ValueError(
                f"Multi-agent network expected input with shape[-2]={self.n_agents},"
                f" but got {inputs.shape}"
            )
        # If the model is centralized, agents have full observability
        if self.centralised:
            inputs = inputs.flatten(-2, -1)
        return inputs

    def _build_single_net(self, *, device, **kwargs):
        n_agent_inputs = self.n_agent_inputs
        if self.centralised and n_agent_inputs is not None:
            n_agent_inputs = self.n_agent_inputs * self.n_agents
        return MLP(
            in_features=n_agent_inputs,
            out_features=self.n_agent_outputs,
            depth=self.depth,
            num_cells=self.num_cells,
            activation_class=self.activation_class,
            device=device,
            **kwargs,
        )


class MultiAgentConvNet(MultiAgentNetBase):
    """Multi-agent CNN.

    In MARL settings, agents may or may not share the same policy for their actions: we say that the parameters can be shared or not. Similarly, a network may take the entire observation space (across agents) or on a per-agent basis to compute its output, which we refer to as "centralized" and "non-centralized", respectively.

    It expects inputs with shape ``(*B, n_agents, channels, x, y)``.

    Args:
        n_agents (int): number of agents.
        centralised (bool): If ``True``, each agent will use the inputs of all agents to compute its output, resulting in input of shape ``(*B, n_agents * channels, x, y)``. Otherwise, each agent will only use its data as input.
        share_params (bool): If ``True``, the same :class:`~torchrl.modules.ConvNet` will be used to make the forward pass
            for all agents (homogeneous policies). Otherwise, each agent will use a different :class:`~torchrl.modules.ConvNet` to process
            its input (heterogeneous policies).

    Keyword Args:
        in_features (int, optional): the input feature dimension. If left to ``None``,
            a lazy module is used.
        device (str or torch.device, optional): device to create the module on.
        num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers ``out_features`` will match the content of ``num_cells``.
        kernel_sizes (int, Sequence[Union[int, Sequence[int]]]): Kernel size(s) of the convolutional network.
            Defaults to ``5``.
        strides (int or Sequence[int]): Stride(s) of the convolutional network. If iterable, the length must match the
            depth, defined by the num_cells or depth arguments.
            Defaults to ``2``.
        activation_class (Type[nn.Module]): activation class to be used.
            Default to :class:`torch.nn.ELU`.
        **kwargs: for :class:`~torchrl.modules.models.ConvNet` can be passed to customize the ConvNet.


    Examples:
        >>> import torch
        >>> from torchrl.modules import MultiAgentConvNet
        >>> batch = (3,2)
        >>> n_agents = 7
        >>> channels, x, y = 3, 100, 100
        >>> obs = torch.randn(*batch, n_agents, channels, x, y)
        >>> # First lets consider a centralised network with shared parameters.
        >>> cnn = MultiAgentConvNet(
        ...     n_agents,
        ...     centralised = True,
        ...     share_params = True
        ... )
        >>> print(cnn)
        MultiAgentConvNet(
            (agent_networks): ModuleList(
                (0): ConvNet(
                (0): LazyConv2d(0, 32, kernel_size=(5, 5), stride=(2, 2))
                (1): ELU(alpha=1.0)
                (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
                (3): ELU(alpha=1.0)
                (4): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
                (5): ELU(alpha=1.0)
                (6): SquashDims()
                )
            )
        )
        >>> result = cnn(obs)
        >>> # The final dimension of the resulting tensor would be determined based on the layer definition arguments and the shape of input 'obs'.
        >>> print(result.shape)
        torch.Size([3, 2, 7, 2592])
        >>> # Since both observations and parameters are shared, we expect all agents to have identical outputs (eg. for a value function)
        >>> print(all(result[0,0,0] == result[0,0,1]))
        True

        >>> # Alternatively, a local network with parameter sharing (eg. decentralised weight sharing policy)
        >>> cnn = MultiAgentConvNet(
        ...     n_agents,
        ...     centralised = False,
        ...     share_params = True
        ... )
        >>> print(cnn)
        MultiAgentConvNet(
            (agent_networks): ModuleList(
                (0): ConvNet(
                (0): Conv2d(4, 32, kernel_size=(5, 5), stride=(2, 2))
                (1): ELU(alpha=1.0)
                (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
                (3): ELU(alpha=1.0)
                (4): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
                (5): ELU(alpha=1.0)
                (6): SquashDims()
                )
            )
        )
        >>> print(result.shape)
        torch.Size([3, 2, 7, 2592])
        >>> # Parameters are shared but not observations, hence each agent has a different output.
        >>> print(all(result[0,0,0] == result[0,0,1]))
        False

        >>> # Or multiple local networks identical in structure but with differing weights.
        >>> cnn = MultiAgentConvNet(
        ...     n_agents,
        ...     centralised = False,
        ...     share_params = False
        ... )
        >>> print(cnn)
        MultiAgentConvNet(
            (agent_networks): ModuleList(
                (0-6): 7 x ConvNet(
                (0): Conv2d(4, 32, kernel_size=(5, 5), stride=(2, 2))
                (1): ELU(alpha=1.0)
                (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
                (3): ELU(alpha=1.0)
                (4): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
                (5): ELU(alpha=1.0)
                (6): SquashDims()
                )
            )
        )
        >>> print(result.shape)
        torch.Size([3, 2, 7, 2592])
        >>> print(all(result[0,0,0] == result[0,0,1]))
        False

        >>> # Or where inputs are shared but not parameters.
        >>> cnn = MultiAgentConvNet(
        ...     n_agents,
        ...     centralised = True,
        ...     share_params = False
        ... )
        >>> print(cnn)
        MultiAgentConvNet(
            (agent_networks): ModuleList(
                (0-6): 7 x ConvNet(
                (0): Conv2d(28, 32, kernel_size=(5, 5), stride=(2, 2))
                (1): ELU(alpha=1.0)
                (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
                (3): ELU(alpha=1.0)
                (4): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
                (5): ELU(alpha=1.0)
                (6): SquashDims()
                )
            )
        )
        >>> print(result.shape)
        torch.Size([3, 2, 7, 2592])
        >>> print(all(result[0,0,0] == result[0,0,1]))
        False
    """

    def __init__(
        self,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        *,
        in_features: int | None = None,
        device: DEVICE_TYPING | None = None,
        num_cells: Sequence[int] | None = None,
        kernel_sizes: Union[Sequence[Union[int, Sequence[int]]], int] = 5,
        strides: Union[Sequence, int] = 2,
        paddings: Union[Sequence, int] = 0,
        activation_class: Type[nn.Module] = nn.ELU,
        **kwargs,
    ):
        self.in_features = in_features
        self.num_cells = num_cells
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.paddings = paddings
        self.activation_class = activation_class
        super().__init__(
            n_agents=n_agents,
            centralised=centralised,
            share_params=share_params,
            device=device,
            agent_dim=-4,
            **kwargs,
        )

    def _build_single_net(self, *, device, **kwargs):
        in_features = self.in_features
        if self.centralised and in_features is not None:
            in_features = in_features * self.n_agents
        return ConvNet(
            in_features=in_features,
            num_cells=self.num_cells,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            activation_class=self.activation_class,
            device=device,
            **kwargs,
        )

    def _pre_forward_check(self, inputs):
        if len(inputs.shape) < 4:
            raise ValueError(
                """Multi-agent network expects (*batch_size, agent_index, x, y, channels)"""
            )
        if inputs.shape[-4] != self.n_agents:
            raise ValueError(
                f"""Multi-agent network expects {self.n_agents} but got {inputs.shape[-4]}"""
            )
        if self.centralised:
            # If the model is centralized, agents have full observability
            inputs = torch.flatten(inputs, -4, -3)
        return inputs


class Mixer(nn.Module):
    """A multi-agent value mixer.

    It transforms the local value of each agent's chosen action of shape (*B, self.n_agents, 1),
    into a global value with shape (*B, 1).
    Used with the :class:`torchrl.objectives.QMixerLoss`.
    See `examples/multiagent/qmix_vdn.py` for examples.

    Args:
        n_agents (int): number of agents.
        needs_state (bool): whether the mixer takes a global state as input.
        state_shape (tuple or torch.Size): the shape of the state (excluding eventual leading batch dimensions).
        device (str or torch.Device): torch device for the network.

    Examples:
        Creating a VDN mixer
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.multiagent import VDNMixer
        >>> n_agents = 4
        >>> vdn = TensorDictModule(
        ...     module=VDNMixer(
        ...         n_agents=n_agents,
        ...         device="cpu",
        ...     ),
        ...     in_keys=[("agents","chosen_action_value")],
        ...     out_keys=["chosen_action_value"],
        ... )
        >>> td = TensorDict({"agents": TensorDict({"chosen_action_value": torch.zeros(32, n_agents, 1)}, [32, n_agents])}, [32])
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
        >>> vdn(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
        Creating a QMix mixer
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.multiagent import QMixer
        >>> n_agents = 4
        >>> qmix = TensorDictModule(
        ...     module=QMixer(
        ...         state_shape=(64, 64, 3),
        ...         mixing_embed_dim=32,
        ...         n_agents=n_agents,
        ...         device="cpu",
        ...     ),
        ...     in_keys=[("agents", "chosen_action_value"), "state"],
        ...     out_keys=["chosen_action_value"],
        ... )
        >>> td = TensorDict({"agents": TensorDict({"chosen_action_value": torch.zeros(32, n_agents, 1)}, [32, n_agents]), "state": torch.zeros(32, 64, 64, 3)}, [32])
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                state: Tensor(shape=torch.Size([32, 64, 64, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
        >>> vdn(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                state: Tensor(shape=torch.Size([32, 64, 64, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
    """

    def __init__(
        self,
        n_agents: int,
        needs_state: bool,
        state_shape: Union[Tuple[int, ...], torch.Size],
        device: DEVICE_TYPING,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.device = device
        self.needs_state = needs_state
        self.state_shape = state_shape

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        """Forward pass of the mixer.

        Args:
            *inputs: The first input should be the value of the chosen action of shape (*B, self.n_agents, 1),
                representing the local q value of each agent.
                The second input (optional, used only in some mixers)
                is the shared state of all agents of shape (*B, *self.state_shape).

        Returns:
            The global value of the chosen actions obtained after mixing, with shape (*B, 1)

        """
        if not self.needs_state:
            if len(inputs) > 1:
                raise ValueError(
                    "Mixer that doesn't need state was passed more than 1 input"
                )
            chosen_action_value = inputs[0]
        else:
            if len(inputs) != 2:
                raise ValueError("Mixer that needs state was passed more than 2 inputs")

            chosen_action_value, state = inputs

            if state.shape[-len(self.state_shape) :] != self.state_shape:
                raise ValueError(
                    f"Mixer network expected state with ending shape {self.state_shape},"
                    f" but got state shape {state.shape}"
                )

        if chosen_action_value.shape[-2:] != (self.n_agents, 1):
            raise ValueError(
                f"Mixer network expected chosen_action_value with last 2 dimensions {(self.n_agents,1)},"
                f" but got {chosen_action_value.shape}"
            )
        batch_dims = chosen_action_value.shape[:-2]

        if not self.needs_state:
            output = self.mix(chosen_action_value, None)
        else:
            output = self.mix(chosen_action_value, state)

        if output.shape != (*batch_dims, 1):
            raise ValueError(
                f"Mixer network expected output with same shape as input minus the multi-agent dimension,"
                f" but got {output.shape}"
            )

        return output

    def mix(self, chosen_action_value: torch.Tensor, state: torch.Tensor):
        """Forward pass for the mixer.

        Args:
            chosen_action_value: Tensor of shape [*B, n_agents]

        Returns:
            chosen_action_value: Tensor of shape [*B]
        """
        raise NotImplementedError


class VDNMixer(Mixer):
    """Value-Decomposition Network mixer.

    Mixes the local Q values of the agents into a global Q value by summing them together.
    From the paper https://arxiv.org/abs/1706.05296 .

    It transforms the local value of each agent's chosen action of shape (*B, self.n_agents, 1),
    into a global value with shape (*B, 1).
    Used with the :class:`torchrl.objectives.QMixerLoss`.
    See `examples/multiagent/qmix_vdn.py` for examples.

    Args:
        n_agents (int): number of agents.
        device (str or torch.Device): torch device for the network.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.multiagent import VDNMixer
        >>> n_agents = 4
        >>> vdn = TensorDictModule(
        ...     module=VDNMixer(
        ...         n_agents=n_agents,
        ...         device="cpu",
        ...     ),
        ...     in_keys=[("agents","chosen_action_value")],
        ...     out_keys=["chosen_action_value"],
        ... )
        >>> td = TensorDict({"agents": TensorDict({"chosen_action_value": torch.zeros(32, n_agents, 1)}, [32, n_agents])}, [32])
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
        >>> vdn(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
    """

    def __init__(
        self,
        n_agents: int,
        device: DEVICE_TYPING,
    ):
        super().__init__(
            needs_state=False,
            state_shape=torch.Size([]),
            n_agents=n_agents,
            device=device,
        )

    def mix(self, chosen_action_value: torch.Tensor, state: torch.Tensor):
        return chosen_action_value.sum(dim=-2)


class QMixer(Mixer):
    """QMix mixer.

    Mixes the local Q values of the agents into a global Q value through a monotonic
    hyper-network whose parameters are obtained from a global state.
    From the paper https://arxiv.org/abs/1803.11485 .

    It transforms the local value of each agent's chosen action of shape (*B, self.n_agents, 1),
    into a global value with shape (*B, 1).
    Used with the :class:`torchrl.objectives.QMixerLoss`.
    See `examples/multiagent/qmix_vdn.py` for examples.

    Args:
        state_shape (tuple or torch.Size): the shape of the state (excluding eventual leading batch dimensions).
        mixing_embed_dim (int): the size of the mixing embedded dimension.
        n_agents (int): number of agents.
        device (str or torch.Device): torch device for the network.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.multiagent import QMixer
        >>> n_agents = 4
        >>> qmix = TensorDictModule(
        ...     module=QMixer(
        ...         state_shape=(64, 64, 3),
        ...         mixing_embed_dim=32,
        ...         n_agents=n_agents,
        ...         device="cpu",
        ...     ),
        ...     in_keys=[("agents", "chosen_action_value"), "state"],
        ...     out_keys=["chosen_action_value"],
        ... )
        >>> td = TensorDict({"agents": TensorDict({"chosen_action_value": torch.zeros(32, n_agents, 1)}, [32, n_agents]), "state": torch.zeros(32, 64, 64, 3)}, [32])
        >>> td
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                state: Tensor(shape=torch.Size([32, 64, 64, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
        >>> vdn(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        chosen_action_value: Tensor(shape=torch.Size([32, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 4]),
                    device=None,
                    is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                state: Tensor(shape=torch.Size([32, 64, 64, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32]),
            device=None,
            is_shared=False)
    """

    def __init__(
        self,
        state_shape: Union[Tuple[int, ...], torch.Size],
        mixing_embed_dim: int,
        n_agents: int,
        device: DEVICE_TYPING,
    ):
        super().__init__(
            needs_state=True, state_shape=state_shape, n_agents=n_agents, device=device
        )

        self.embed_dim = mixing_embed_dim
        self.state_dim = int(np.prod(state_shape))

        self.hyper_w_1 = nn.Linear(
            self.state_dim, self.embed_dim * self.n_agents, device=self.device
        )
        self.hyper_w_final = nn.Linear(
            self.state_dim, self.embed_dim, device=self.device
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim, device=self.device)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1, device=self.device),
        )

    def mix(self, chosen_action_value: torch.Tensor, state: torch.Tensor):
        bs = chosen_action_value.shape[:-2]
        state = state.view(-1, self.state_dim)
        chosen_action_value = chosen_action_value.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(state))
        b1 = self.hyper_b_1(state)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = nn.functional.elu(
            torch.bmm(chosen_action_value, w1) + b1
        )  # [-1, 1, self.embed_dim]
        # Second layer
        w_final = torch.abs(self.hyper_w_final(state))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(state).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v  # [-1, 1, 1]
        # Reshape and return
        q_tot = y.view(*bs, 1)
        return q_tot
