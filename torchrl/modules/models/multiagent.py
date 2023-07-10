from typing import Tuple, Union

import numpy as np
import torch
from torch import nn


class Mixer(nn.Module):
    """A multi-agent value mixer.

    It transforms the local value of each agent's chosen action of shape (*B, self.n_agents, 1),
    into a global value with shape (*B, 1).

    Args:
        n_agents (int): number of agents,
        device (str or torch.Device): torch device for the network
        needs_state (bool): whether the mixer takes a global state as input
        state_shape (tuple or torch.Size): the shape of the state (excluding eventual leading batch dimensions)

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
        device,
        needs_state: bool,
        state_shape: Union[Tuple[int, ...], torch.Size],
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
            if len(inputs) > 2:
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
    """Mixer from https://arxiv.org/abs/1706.05296 .

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
    """

    def __init__(
        self,
        n_agents: int,
        device,
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
    """Mixer from https://arxiv.org/abs/1803.11485 .

    Examples:
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
        state_shape,
        mixing_embed_dim,
        n_agents: int,
        device,
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
