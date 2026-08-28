# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Cross-group centralised critic for heterogeneous or ad-hoc multi-agent teams.

References:
    - Yu, C. et al. *The Surprising Effectiveness of PPO in Cooperative
      Multi-Agent Games.* NeurIPS 2022. https://arxiv.org/abs/2103.01955
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from torch import nn

from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules.models.models import MLP


@dataclasses.dataclass
class CrossCriticGroupSpec:
    """Specification for one agent group used by :class:`CrossGroupCritic`.

    Args:
        obs_dim (int): dimensionality of each agent's observation vector.
        n_agents (int): number of agents in the group.
        obs_key (NestedKey): tensordict key holding this group's observations,
            e.g. ``("soldiers", "observation")``.
        value_key (NestedKey): tensordict key where this group's state values
            will be written, e.g. ``("soldiers", "state_value")``.
    """

    obs_dim: int
    n_agents: int
    obs_key: NestedKey
    value_key: NestedKey


class _CrossGroupNet(nn.Module):
    """Inner nn.Module for :class:`CrossGroupCritic`.

    Takes one observation tensor per group and returns one value tensor per
    group. Separating the plain nn.Module from the TensorDictModule wrapper
    keeps the forward signature simple and testable without tensordict.
    """

    def __init__(
        self,
        group_specs: dict[str, CrossCriticGroupSpec],
        d_model: int,
        trunk_depth: int,
        trunk_cells: int,
        activation_class: type[nn.Module],
        share_params: bool,
        detach_groups: Iterable[str] | None,
        device: DEVICE_TYPING | None,
    ) -> None:
        super().__init__()
        self._group_names: list[str] = list(group_specs.keys())
        self._group_n_agents: list[int] = [s.n_agents for s in group_specs.values()]
        self._group_obs_dims: list[int] = [s.obs_dim for s in group_specs.values()]
        self._n_agents_total = sum(self._group_n_agents)
        self._joint_dim = self._n_agents_total * d_model
        self._detach_groups: frozenset[str] = frozenset(detach_groups or [])
        self.shared_head: nn.Linear | None
        self.group_heads: nn.ModuleDict | None

        # One encoder per group so heterogeneous obs_dims are handled uniformly.
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(spec.obs_dim, d_model, device=device),
                    activation_class(),
                )
                for name, spec in group_specs.items()
            }
        )

        # Shared MLP trunk processes the flattened team state, so each output
        # embedding can depend on every group's observations.
        self.trunk = MLP(
            in_features=self._joint_dim,
            out_features=self._joint_dim,
            depth=trunk_depth,
            num_cells=trunk_cells,
            activation_class=activation_class,
            device=device,
        )

        # Value heads — optionally shared across groups.
        if share_params:
            self.shared_head = nn.Linear(d_model, 1, device=device)
            self.group_heads = None
        else:
            self.shared_head = None
            self.group_heads = nn.ModuleDict(
                {name: nn.Linear(d_model, 1, device=device) for name in group_specs}
            )

    def forward(self, *group_obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # group_obs[i]: [*B, n_agents_i, obs_dim_i]
        encoded = []
        for obs, name, n_agents, obs_dim in zip(
            group_obs, self._group_names, self._group_n_agents, self._group_obs_dims
        ):
            if obs.shape[-2:] != (n_agents, obs_dim):
                raise ValueError(
                    f"Group '{name}' expected observation shape ending in "
                    f"{(n_agents, obs_dim)}, but got {obs.shape}."
                )
            enc = self.encoders[name](obs)  # [*B, n_agents_i, d_model]
            if name in self._detach_groups:
                enc = enc.detach()
            encoded.append(enc)

        # Joint representation across all groups: [*B, n_total * d_model]
        joint = torch.cat(encoded, dim=-2).flatten(-2, -1)
        joint = self.trunk(joint)
        joint = joint.view(*joint.shape[:-1], self._n_agents_total, -1)

        splits = torch.split(joint, self._group_n_agents, dim=-2)
        if self.shared_head is not None:
            return tuple(self.shared_head(g) for g in splits)
        return tuple(
            self.group_heads[name](g) for name, g in zip(self._group_names, splits)
        )


class CrossGroupCritic(TensorDictModule):
    """Centralised critic that conditions on observations from multiple agent groups.

    Standard :class:`~torchrl.modules.MultiAgentMLP` centralises only within a
    single group.  ``CrossGroupCritic`` removes that restriction: it reads
    observations from an arbitrary number of groups (each potentially with a
    different observation dimensionality), encodes them to a shared embedding
    space, processes the joint representation through a shared MLP trunk, and
    writes a per-group value estimate back to the tensordict.

    This enables two use-cases that single-group critics cannot handle:

    - **Heterogeneous teams** — agents in different groups have different
      observation / action specs. Each group gets its own encoder
      (``Linear(obs_dim_g → d_model)``), so no padding or obs-dim alignment
      is required.
    - **Ad-hoc teamwork** — one group follows a fixed (non-training) policy
      but its observations still inform the value baseline of the training
      group. Pass the fixed group's name via ``detach_groups`` so its encoder
      output is detached before building the team state: the critic sees the
      full team state but gradients do not flow into the fixed group's
      observations.

    Because ``CrossGroupCritic`` is a plain :class:`~tensordict.nn.TensorDictModule`,
    it plugs into :class:`~torchrl.objectives.multiagent.MAPPOLoss` and
    :class:`~torchrl.objectives.multiagent.IPPOLoss` without any changes to
    those classes.

    Args:
        group_map (dict[str, CrossCriticGroupSpec]): ordered mapping from a group name
            to a :class:`CrossCriticGroupSpec` that describes the group's observation
            dimensionality, agent count, and tensordict keys.

    Keyword Args:
        d_model (int): common embedding dimension. All per-group encoders
            project to this size. Defaults to ``64``.
        trunk_depth (int): number of hidden layers in the shared MLP trunk.
            Defaults to ``2``.
        trunk_cells (int): width of each trunk hidden layer. Defaults to ``256``.
        activation_class (type[nn.Module]): activation used in encoders and
            trunk. Defaults to :class:`~torch.nn.Tanh`.
        share_params (bool): if ``True`` a single value head is shared across
            all groups (useful when groups are homogeneous or have the same
            role). If ``False`` each group gets its own head. Encoders are
            always group-specific and the central trunk is always shared.
            Defaults to ``False``.
        detach_groups (iterable of str, optional): names of groups whose encoder
            outputs should be detached before the trunk. Use this to include
            fixed-policy agents in the centralised state without propagating
            gradients to their observations. Defaults to ``None``.
        device (DEVICE_TYPING, optional): device on which to allocate
            parameters. Defaults to ``None`` (CPU).

    .. note::
        The order of keys in ``group_map`` determines the order of positional
        inputs to the inner network. Python ``dict`` preserves insertion order
        (Python 3.7+), so the mapping is stable.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.modules.models.cross_group_critic import CrossGroupCritic, CrossCriticGroupSpec
        >>> group_map = {
        ...     "soldiers": CrossCriticGroupSpec(obs_dim=12, n_agents=3,
        ...         obs_key=("soldiers", "observation"),
        ...         value_key=("soldiers", "state_value")),
        ...     "medics": CrossCriticGroupSpec(obs_dim=8, n_agents=2,
        ...         obs_key=("medics", "observation"),
        ...         value_key=("medics", "state_value")),
        ... }
        >>> critic = CrossGroupCritic(group_map, d_model=32, trunk_depth=1, trunk_cells=64)
        >>> td = TensorDict(
        ...     {
        ...         "soldiers": {"observation": torch.zeros(4, 3, 12)},
        ...         "medics":   {"observation": torch.zeros(4, 2, 8)},
        ...     },
        ...     batch_size=[4],
        ... )
        >>> td = critic(td)
        >>> print(td["soldiers", "state_value"].shape)
        torch.Size([4, 3, 1])
        >>> print(td["medics", "state_value"].shape)
        torch.Size([4, 2, 1])
    """

    def __init__(
        self,
        group_map: dict[str, CrossCriticGroupSpec],
        *,
        d_model: int = 64,
        trunk_depth: int = 2,
        trunk_cells: int = 256,
        activation_class: type[nn.Module] = nn.Tanh,
        share_params: bool = False,
        detach_groups: Iterable[str] | None = None,
        device: DEVICE_TYPING | None = None,
    ) -> None:
        net = _CrossGroupNet(
            group_specs=group_map,
            d_model=d_model,
            trunk_depth=trunk_depth,
            trunk_cells=trunk_cells,
            activation_class=activation_class,
            share_params=share_params,
            detach_groups=detach_groups,
            device=device,
        )
        super().__init__(
            module=net,
            in_keys=[spec.obs_key for spec in group_map.values()],
            out_keys=[spec.value_key for spec in group_map.values()],
        )
